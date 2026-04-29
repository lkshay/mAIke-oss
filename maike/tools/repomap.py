"""Repository map tool — AST-based structural overview."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.runtime.protocol import ExecutionRuntime
from maike.tools.registry import ToolRegistry

_SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    ".maike",
}

_OUTPUT_CAP = 8000

# ---------------------------------------------------------------------------
# Python: AST-based extraction
# ---------------------------------------------------------------------------


def _parse_python_signatures(source: str) -> list[str]:
    """Extract class/function signatures from Python source using AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _regex_python_signatures(source)

    sigs: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            sigs.append(f"def {node.name}(...)")
        elif isinstance(node, ast.ClassDef):
            sigs.append(f"class {node.name}:")
            for item in ast.iter_child_nodes(node):
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    sigs.append(f"  def {item.name}(...)")
    return sigs


def _regex_python_signatures(source: str) -> list[str]:
    """Fallback regex extraction for unparseable Python files."""
    sigs: list[str] = []
    for m in re.finditer(r"^(class\s+\w+|(?:async\s+)?def\s+\w+)", source, re.MULTILINE):
        sigs.append(m.group(0).rstrip(":") + (":" if m.group(0).startswith("class") else "(...)"))
    return sigs


# ---------------------------------------------------------------------------
# JS/TS: regex-based extraction
# ---------------------------------------------------------------------------

_JS_PATTERNS = [
    re.compile(r"^export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:async\s+)?function\s+(\w+)", re.MULTILINE),
    re.compile(r"^export\s+(?:default\s+)?class\s+(\w+)", re.MULTILINE),
    re.compile(r"^class\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:export\s+)?interface\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE),
]


def _parse_js_signatures(source: str) -> list[str]:
    sigs: list[str] = []
    seen: set[str] = set()
    for pat in _JS_PATTERNS:
        for m in pat.finditer(source):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                full = m.group(0).strip()
                if full.startswith("export default "):
                    full = full[len("export default ") :]
                elif full.startswith("export "):
                    full = full[len("export ") :]
                sigs.append(full)
    return sigs


# ---------------------------------------------------------------------------
# Go: regex-based extraction
# ---------------------------------------------------------------------------

_GO_PATTERNS = [
    re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE),
    re.compile(r"^type\s+(\w+)\s+struct\s*\{", re.MULTILINE),
    re.compile(r"^type\s+(\w+)\s+interface\s*\{", re.MULTILINE),
]


def _parse_go_signatures(source: str) -> list[str]:
    sigs: list[str] = []
    seen: set[str] = set()
    for pat in _GO_PATTERNS:
        for m in pat.finditer(source):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                sigs.append(m.group(0).rstrip("{").strip())
    return sigs


# ---------------------------------------------------------------------------
# Rust: regex-based extraction
# ---------------------------------------------------------------------------

_RUST_PATTERNS = [
    re.compile(r"^pub\s+(?:async\s+)?fn\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", re.MULTILINE),
    re.compile(r"^pub\s+struct\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:pub\s+)?struct\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:pub\s+)?trait\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:pub\s+)?enum\s+(\w+)", re.MULTILINE),
    re.compile(r"^impl(?:<[^>]+>)?\s+(\w+)", re.MULTILINE),
]


def _parse_rust_signatures(source: str) -> list[str]:
    sigs: list[str] = []
    seen: set[str] = set()
    for pat in _RUST_PATTERNS:
        for m in pat.finditer(source):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                sigs.append(m.group(0).strip())
    return sigs


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, callable] = {
    ".py": _parse_python_signatures,
    ".js": _parse_js_signatures,
    ".jsx": _parse_js_signatures,
    ".ts": _parse_js_signatures,
    ".tsx": _parse_js_signatures,
    ".go": _parse_go_signatures,
    ".rs": _parse_rust_signatures,
}


def _should_skip(name: str) -> bool:
    if name.startswith("."):
        return True
    return name in _SKIP_DIRS


def _build_repo_map(
    root: str,
    max_depth: int = 3,
    include_signatures: bool = True,
) -> str:
    """Build a compact structural map of the repository."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        return f"Not a directory: {root}"

    lines: list[str] = []
    total_chars = 0

    def walk(directory: Path, depth: int, prefix: str) -> None:
        nonlocal total_chars
        if total_chars >= _OUTPUT_CAP:
            return
        if depth > max_depth:
            return

        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return

        dirs = [e for e in entries if e.is_dir() and not _should_skip(e.name)]
        files = [e for e in entries if e.is_file() and not e.name.startswith(".")]

        for f in files:
            if total_chars >= _OUTPUT_CAP:
                return
            line = f"{prefix}{f.name}"
            lines.append(line)
            total_chars += len(line) + 1

            if include_signatures and f.suffix in _EXTRACTORS and total_chars < _OUTPUT_CAP:
                try:
                    source = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                sigs = _EXTRACTORS[f.suffix](source)
                for sig in sigs:
                    if total_chars >= _OUTPUT_CAP:
                        break
                    sig_line = f"{prefix}  {sig}"
                    lines.append(sig_line)
                    total_chars += len(sig_line) + 1

        for d in dirs:
            if total_chars >= _OUTPUT_CAP:
                return
            dir_line = f"{prefix}{d.name}/"
            lines.append(dir_line)
            total_chars += len(dir_line) + 1
            walk(d, depth + 1, prefix + "  ")

    walk(root_path, 0, "")

    if total_chars >= _OUTPUT_CAP:
        lines.append(f"\n... (output capped at ~{_OUTPUT_CAP} chars)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_repomap_tools(
    registry: ToolRegistry,
    runtime: ExecutionRuntime,
) -> None:

    async def repo_map(
        path: str = ".",
        max_depth: int = 3,
        include_signatures: bool = True,
    ) -> ToolResult:
        # Resolve path relative to workspace
        resolved = await runtime.resolve_path(path) if hasattr(runtime, "resolve_path") else path
        output = _build_repo_map(str(resolved), max_depth=max_depth, include_signatures=include_signatures)
        return ToolResult(
            tool_name="repo_map",
            success=True,
            output=output,
            raw_output=output,
            metadata={"path": str(resolved), "max_depth": max_depth},
        )

    registry.register(
        schema=ToolSchema(
            name="repo_map",
            description=(
                "Generate a structural map of the repository showing files "
                "and their top-level symbols (classes, functions, interfaces). "
                "Use at the start of a task to understand the codebase layout."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root directory to map. Defaults to workspace root.",
                        "default": ".",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to traverse (default 3).",
                        "default": 3,
                    },
                    "include_signatures": {
                        "type": "boolean",
                        "description": "Include function/class signatures (default true).",
                        "default": True,
                    },
                },
            },
        ),
        fn=repo_map,
        risk_level=RiskLevel.READ,
        profiles=["all"],
    )
