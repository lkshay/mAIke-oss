"""Search tools backed by ripgrep when available."""

from __future__ import annotations

import shlex
import shutil

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.runtime.protocol import ExecutionRuntime
from maike.tools.context import peek_current_code_index
from maike.tools.registry import ToolRegistry


def register_search_tools(registry: ToolRegistry, runtime: ExecutionRuntime) -> None:
    async def search_files(path: str = ".") -> ToolResult:
        if shutil.which("rg"):
            result = await runtime.execute_bash(f"rg --files {shlex.quote(path)}")
        else:
            result = await runtime.execute_bash(f"find {shlex.quote(path)} -type f")
        return result.model_copy(update={"tool_name": "search_files"})

    async def grep_codebase(
        pattern: str,
        path: str = ".",
        *,
        context_lines: int = 0,
        before_context: int = 0,
        after_context: int = 0,
        file_type: str | None = None,
        glob_pattern: str | None = None,
        case_insensitive: bool = False,
        output_mode: str = "content",
        max_results: int = 200,
    ) -> ToolResult:
        if not pattern or not pattern.strip():
            return ToolResult(
                tool_name="Grep",
                success=False,
                output=(
                    "Empty pattern. To list files, use pattern='.' with "
                    "path='<directory>' (e.g. path='maike/agents'). "
                    "The 'path' parameter scopes the search directory; "
                    "'glob_pattern' only filters within that scope."
                ),
                error="empty_pattern",
            )
        if shutil.which("rg"):
            parts: list[str] = ["rg", "-n"]
            if case_insensitive:
                parts.append("-i")
            if context_lines > 0:
                parts.extend(["-C", str(min(context_lines, 10))])
            else:
                if before_context > 0:
                    parts.extend(["-B", str(min(before_context, 10))])
                if after_context > 0:
                    parts.extend(["-A", str(min(after_context, 10))])
            if file_type:
                parts.extend(["--type", shlex.quote(file_type)])
            if glob_pattern:
                parts.extend(["--glob", shlex.quote(glob_pattern)])
            if output_mode == "files_only":
                parts.append("-l")
            elif output_mode == "count":
                parts.append("-c")
            # Exclude VCS and dependency directories.
            parts.extend([
                "--glob", "!.git/",
                "--glob", "!node_modules/",
                "--glob", "!.venv/",
                "--glob", "!__pycache__/",
            ])
            parts.extend(["--max-count", str(min(max_results, 500))])
            parts.append(shlex.quote(pattern))
            parts.append(shlex.quote(path))
            cmd = " ".join(parts)
        else:
            # Fallback to grep with basic flags.
            parts = ["grep", "-R", "-n"]
            if case_insensitive:
                parts.append("-i")
            if context_lines > 0:
                parts.extend(["-C", str(min(context_lines, 10))])
            else:
                if before_context > 0:
                    parts.extend(["-B", str(min(before_context, 10))])
                if after_context > 0:
                    parts.extend(["-A", str(min(after_context, 10))])
            if output_mode == "files_only":
                parts.append("-l")
            elif output_mode == "count":
                parts.append("-c")
            if glob_pattern:
                parts.extend(["--include", shlex.quote(glob_pattern)])
            parts.append(shlex.quote(pattern))
            parts.append(shlex.quote(path))
            cmd = " ".join(parts)

        result = await runtime.execute_bash(cmd)
        # ripgrep returns exit code 1 for no matches — treat as success.
        if result.metadata.get("returncode") == 1 and not result.raw_output.strip():
            result = result.model_copy(
                update={
                    "success": True,
                    "error": None,
                    "output": "",
                    "raw_output": "",
                }
            )
        return result.model_copy(update={"tool_name": "grep_codebase"})

    registry.register(
        ToolSchema(
            name="Grep",
            description=(
                "Search the codebase for a regex pattern. Supports context lines, "
                "file type filtering, glob patterns, and multiple output modes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Defaults to workspace root.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context before and after each match (like grep -C). Max 10.",
                    },
                    "before_context": {
                        "type": "integer",
                        "description": "Lines of context before each match (like grep -B). Max 10.",
                    },
                    "after_context": {
                        "type": "integer",
                        "description": "Lines of context after each match (like grep -A). Max 10.",
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Restrict to file type: py, js, ts, go, rs, java, etc.",
                    },
                    "glob_pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py', 'src/**/*.ts').",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive matching.",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_only", "count"],
                        "description": "Output mode: content (matching lines, default), files_only (file paths only), count (match counts per file).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches per file. Default 200, max 500.",
                    },
                },
                "required": ["pattern"],
            },
        ),
        fn=grep_codebase,
        risk_level=RiskLevel.READ,
    )


def register_semantic_search_tool(
    registry: ToolRegistry, runtime: ExecutionRuntime
) -> None:
    async def semantic_code_search(query: str, limit: int = 10) -> ToolResult:
        """Natural-language search over the codebase using embeddings."""
        code_index = peek_current_code_index()
        if code_index is None or not code_index.is_built:
            return ToolResult(
                tool_name="SemanticSearch",
                success=False,
                output="Code index not available. Use Grep for regex-based search instead.",
                error="code_index_unavailable",
            )
        chunks = await code_index.semantic_search(query, limit=min(limit, 20))
        if not chunks:
            return ToolResult(
                tool_name="SemanticSearch",
                success=True,
                output="No results found. Try rephrasing the query or use Grep for exact patterns.",
                raw_output="",
            )
        lines: list[str] = []
        for chunk in chunks:
            header = f"--- {chunk.file}:{chunk.start_line}-{chunk.end_line}"
            if chunk.symbol_name:
                header += f" ({chunk.kind or 'code'}: {chunk.symbol_name})"
            lines.append(header)
            lines.append(chunk.content)
            lines.append("")
        output = "\n".join(lines)
        return ToolResult(
            tool_name="SemanticSearch",
            success=True,
            output=output,
            raw_output=output,
            metadata={"result_count": len(chunks)},
        )

    registry.register(
        ToolSchema(
            name="SemanticSearch",
            description=(
                "Natural-language search over the codebase. Use this when you need to find "
                "code by intent rather than exact patterns — e.g. 'where do we validate user input' "
                "or 'caching logic'. For exact string/regex matching, use Grep instead."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the code you're looking for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default 10, max 20.",
                    },
                },
                "required": ["query"],
            },
        ),
        fn=semantic_code_search,
        risk_level=RiskLevel.READ,
    )
