"""Tool-result compression — reduce large tool outputs before they enter
the conversation history.

Includes AST-aware compression for Python files: keeps imports, class/function
signatures, and docstrings while dropping function bodies.  This preserves the
structural information an agent needs while cutting tokens 60-80%.
"""

from __future__ import annotations

import ast
import re

# Thresholds (characters).
COMPRESSION_THRESHOLD = 4_000
READ_FILE_MAX = 12_000   # ~3K tokens — enough for a 200-line file with docstrings
GREP_MAX = 6_000
BASH_MAX = 4_000
DEFAULT_MAX = 4_000

# AST compression triggers at this line count (below this, head+tail is fine).
_AST_MIN_LINES = 80

# Keywords that indicate a docstring contains algorithmic information worth
# preserving during compression (e.g. recursion strategy, complexity notes).
_ALGORITHMIC_KEYWORDS = frozenset({
    "recursion", "recursive", "base case", "terminat", "invariant",
    "complexity", "o(n", "o(log", "depth", "stack", "overflow",
    "converge", "fixpoint", "memoiz", "backtrack",
})


def _is_algorithmic_docstring(doc: str) -> bool:
    """Check if a docstring contains algorithmic information worth preserving."""
    lower = doc.lower()
    return any(kw in lower for kw in _ALGORITHMIC_KEYWORDS)


class ToolResultCompressor:
    """Compress large tool outputs using tool-specific strategies."""

    def compress(
        self,
        tool_name: str,
        raw_output: str,
        *,
        max_chars: int = 0,
        file_path: str = "",
        is_targeted_read: bool = False,
        seen_files: set[str] | None = None,
    ) -> str:
        """Return a compressed version of *raw_output*.

        If the output is below ``COMPRESSION_THRESHOLD`` it is returned as-is.

        Parameters:
            file_path: Path of the file being read (for first-read tracking).
            is_targeted_read: True when the agent used start_line/end_line —
                skip AST compression since the agent deliberately requested
                a specific section.
            seen_files: Set of file paths already read this session.  First
                reads get full content; re-reads get AST-compressed.
        """
        if len(raw_output) <= COMPRESSION_THRESHOLD:
            return raw_output

        if tool_name in ("Read", "read_file", "open_file"):
            return self._compress_file_read(
                raw_output, max_chars or READ_FILE_MAX,
                file_path=file_path,
                is_targeted=is_targeted_read,
                seen_files=seen_files,
            )
        if tool_name in ("Grep", "grep_codebase", "search_files"):
            return self._compress_grep(raw_output, max_chars or GREP_MAX)
        if tool_name in ("Bash", "execute_bash", "run_tests"):
            return self._compress_bash(raw_output, max_chars or BASH_MAX)
        if tool_name == "repo_map":
            return _head_tail(raw_output, max_chars or DEFAULT_MAX)

        return _head_tail(raw_output, max_chars or DEFAULT_MAX)

    # ------------------------------------------------------------------ #
    # Tool-specific strategies
    # ------------------------------------------------------------------ #

    def _compress_file_read(
        self,
        output: str,
        max_chars: int,
        *,
        file_path: str = "",
        is_targeted: bool = False,
        seen_files: set[str] | None = None,
    ) -> str:
        """Compress a file read result.

        For Python files, uses AST-aware compression that preserves imports,
        class/function signatures, and docstrings while dropping bodies.
        For other files, uses head+tail strategy.

        First-read exemption: if *file_path* has not been seen before (not in
        *seen_files*), returns head+tail instead of AST compression — the agent
        needs full content on first encounter.

        Targeted read passthrough: if *is_targeted* is True (agent used
        start_line/end_line), skips AST compression entirely.
        """
        lines = output.split("\n")

        # Targeted reads (agent explicitly requested a section) — never
        # AST-compress, just cap the length.
        if is_targeted:
            return output[:max_chars] if len(output) > max_chars else output

        # First-read exemption: on the first encounter with a file, give
        # the agent full content (head+tail) instead of AST skeleton.
        # Re-reads get AST-compressed since the agent already knows the file.
        is_first_read = seen_files is not None and file_path and file_path not in seen_files
        if is_first_read:
            # Mark as seen for future reads.
            seen_files.add(file_path)
            # Return head+tail with a generous budget — agent needs to
            # understand the implementation, not just signatures.
            return _head_tail(output, max_chars)

        # Mark as seen if tracking (even for AST path).
        if seen_files is not None and file_path:
            seen_files.add(file_path)

        # Try AST-aware compression for Python content.
        if len(lines) > _AST_MIN_LINES and _looks_like_python(output):
            ast_result = _ast_compress_python(output)
            if ast_result is not None:
                return ast_result[:max_chars]

        # Try signature extraction for JS/TS/Go/Rust.
        if len(lines) > _AST_MIN_LINES:
            sig_result = _signature_compress(output)
            if sig_result is not None:
                return sig_result[:max_chars]

        if len(lines) <= 120:
            return _head_tail(output, max_chars)

        # For very large files (>500 lines) the head matters less — the
        # agent is almost certainly looking for a specific section.
        # Shift the budget towards the tail where errors tend to live.
        if len(lines) > 500:
            head_count = 50
            tail_count = 30
        else:
            head_count = 80
            tail_count = 20
        head = lines[:head_count]
        tail = lines[-tail_count:]
        omitted = len(lines) - head_count - tail_count
        return "\n".join([
            *head,
            f"\n[...{omitted} lines omitted...]\n",
            *tail,
        ])[:max_chars]

    def _compress_grep(self, output: str, max_chars: int) -> str:
        """Group by file, keep first N matches per file.

        Handles context-line output (groups separated by ``--``) and
        passes through ``files_only`` / ``count`` mode output directly.
        """
        lines = output.split("\n")

        # Detect files_only or count mode (lines without file:lineno: pattern).
        if lines and all(
            ":" not in line or (
                ":" in line
                and not (
                    len(line.split(":")) >= 2
                    and line.split(":")[1].strip()[:6].replace("-", "").isdigit()
                )
            )
            for line in lines[:10]
            if line.strip()
        ):
            return _head_tail(output, max_chars)

        # Content mode: group matches by file, handle context separators.
        files_seen: dict[str, list[str]] = {}
        other: list[str] = []
        max_per_file = 8

        for line in lines:
            if line == "--":
                # Context group separator — include in current file group.
                continue
            if ":" in line:
                file_part = line.split(":")[0]
                files_seen.setdefault(file_part, [])
                if len(files_seen[file_part]) < max_per_file:
                    files_seen[file_part].append(line)
            else:
                if line.strip():
                    other.append(line)

        parts: list[str] = []
        total_matches = sum(len(v) for v in files_seen.values())
        parts.append(f"[{len(files_seen)} files, {total_matches} matches shown]")
        for file_path, matches in files_seen.items():
            parts.append(f"\n--- {file_path} ---")
            parts.extend(matches)
        if other:
            parts.extend(other[:5])

        result = "\n".join(parts)
        return result[:max_chars] if len(result) > max_chars else result

    def _compress_bash(self, output: str, max_chars: int) -> str:
        """Keep head (setup) + tail (results/errors).

        Error output is always preserved in full up to max_chars.
        """
        lines = output.split("\n")

        # If it looks like mostly errors, keep it all.
        error_lines = [line for line in lines if any(kw in line.lower() for kw in ("error", "traceback", "failed", "exception"))]
        if len(error_lines) > len(lines) * 0.3:
            return output[:max_chars]

        head_count = 30
        tail_count = 50

        if len(lines) <= head_count + tail_count:
            return _head_tail(output, max_chars)

        head = lines[:head_count]
        tail = lines[-tail_count:]
        omitted = len(lines) - head_count - tail_count
        return "\n".join([
            *head,
            f"\n[...{omitted} lines omitted...]\n",
            *tail,
        ])[:max_chars]


# Singleton for convenience.
_DEFAULT_COMPRESSOR = ToolResultCompressor()


def compress_tool_result(
    tool_name: str,
    raw_output: str,
    *,
    max_chars: int = 0,
    file_path: str = "",
    is_targeted_read: bool = False,
    seen_files: set[str] | None = None,
) -> str:
    """Module-level shortcut."""
    return _DEFAULT_COMPRESSOR.compress(
        tool_name, raw_output,
        max_chars=max_chars,
        file_path=file_path,
        is_targeted_read=is_targeted_read,
        seen_files=seen_files,
    )


# ---------------------------------------------------------------------- #
# AST-aware compression for Python
# ---------------------------------------------------------------------- #


def _looks_like_python(text: str) -> bool:
    """Heuristic: does this text look like Python source code?"""
    indicators = 0
    first_500 = text[:500]
    if "import " in first_500 or "from " in first_500:
        indicators += 1
    if "def " in text:
        indicators += 1
    if "class " in text:
        indicators += 1
    if "    " in text or "\t" in text:
        indicators += 1
    return indicators >= 2


def _get_docstring(node: ast.AST) -> str | None:
    """Extract docstring from a function/class/module node."""
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def _format_function_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str = "") -> str:
    """Format a function signature from an AST node."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args) if hasattr(ast, "unparse") else "..."
    returns = f" -> {ast.unparse(node.returns)}" if node.returns and hasattr(ast, "unparse") else ""
    decorators = ""
    for dec in node.decorator_list:
        dec_text = ast.unparse(dec) if hasattr(ast, "unparse") else "..."
        decorators += f"{indent}@{dec_text}\n"
    return f"{decorators}{indent}{prefix} {node.name}({args}){returns}:"


def _ast_compress_python(source: str) -> str | None:
    """Compress Python source using AST — keep structure, drop bodies.

    Preserves:
    - All imports
    - Module-level assignments (constants)
    - Class definitions with method signatures
    - Function signatures with decorators
    - Docstrings (truncated to first 2 lines)
    - Type annotations

    Drops:
    - Function bodies (replaced with ``...``)
    - Inline comments within bodies
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines: list[str] = []
    source_lines = source.split("\n")
    total_lines = len(source_lines)

    # Module docstring
    mod_doc = _get_docstring(tree)
    if mod_doc:
        doc_lines = mod_doc.strip().split("\n")
        if len(doc_lines) > 3:
            lines.append(f'"""{doc_lines[0]}')
            lines.append(f"[...{len(doc_lines) - 2} lines...]")
            lines.append(f'{doc_lines[-1]}"""')
        else:
            lines.append(f'"""{mod_doc}"""')
        lines.append("")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node) if hasattr(ast, "unparse") else source_lines[node.lineno - 1])

        elif isinstance(node, ast.Assign):
            # Module-level constants/assignments.
            lines.append(ast.unparse(node) if hasattr(ast, "unparse") else source_lines[node.lineno - 1])

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append("")
            sig = _format_function_sig(node)
            lines.append(sig)
            doc = _get_docstring(node)
            if doc:
                doc_lines = doc.strip().split("\n")
                if _is_algorithmic_docstring(doc) and len(doc_lines) > 1:
                    kept = doc_lines[:3]
                    lines.append(f'    """{kept[0]}')
                    for dl in kept[1:]:
                        lines.append(f'    {dl.strip()}')
                    if len(doc_lines) > 3:
                        lines.append(f'    [...{len(doc_lines) - 3} more lines]')
                    lines.append('    """')
                else:
                    doc_first = doc_lines[0]
                    lines.append(f'    """{doc_first}"""')
            lines.append("    ...")

        elif isinstance(node, ast.ClassDef):
            lines.append("")
            # Class decorators
            for dec in node.decorator_list:
                dec_text = ast.unparse(dec) if hasattr(ast, "unparse") else "..."
                lines.append(f"@{dec_text}")
            # Base classes
            bases = ", ".join(ast.unparse(b) if hasattr(ast, "unparse") else "..." for b in node.bases)
            class_line = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
            lines.append(class_line)

            # Class docstring
            cls_doc = _get_docstring(node)
            if cls_doc:
                doc_lines = cls_doc.strip().split("\n")
                if _is_algorithmic_docstring(cls_doc) and len(doc_lines) > 1:
                    kept = doc_lines[:3]
                    lines.append(f'    """{kept[0]}')
                    for dl in kept[1:]:
                        lines.append(f'    {dl.strip()}')
                    if len(doc_lines) > 3:
                        lines.append(f'    [...{len(doc_lines) - 3} more lines]')
                    lines.append('    """')
                else:
                    doc_first = doc_lines[0]
                    lines.append(f'    """{doc_first}"""')

            # Class body: method signatures + class-level assignments
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines.append("")
                    sig = _format_function_sig(item, indent="    ")
                    lines.append(sig)
                    method_doc = _get_docstring(item)
                    if method_doc:
                        doc_lines = method_doc.strip().split("\n")
                        if _is_algorithmic_docstring(method_doc) and len(doc_lines) > 1:
                            kept = doc_lines[:3]
                            lines.append(f'        """{kept[0]}')
                            for dl in kept[1:]:
                                lines.append(f'        {dl.strip()}')
                            if len(doc_lines) > 3:
                                lines.append(f'        [...{len(doc_lines) - 3} more lines]')
                            lines.append('        """')
                        else:
                            doc_first = doc_lines[0]
                            lines.append(f'        """{doc_first}"""')
                    lines.append("        ...")
                elif isinstance(item, ast.Assign):
                    lines.append(f"    {ast.unparse(item)}" if hasattr(ast, "unparse") else f"    {source_lines[item.lineno - 1].strip()}")

    result = "\n".join(lines)
    compressed_lines = len(lines)
    header = f"[AST-compressed: {total_lines} lines -> {compressed_lines} lines, bodies omitted]\n\n"
    return header + result


# ---------------------------------------------------------------------- #
# Regex-based signature compression for JS/TS/Go/Rust
# ---------------------------------------------------------------------- #

_JS_SIG_PATTERNS = [
    re.compile(r"^(export\s+(?:default\s+)?(?:async\s+)?function\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^((?:async\s+)?function\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^(export\s+(?:default\s+)?class\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^(class\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^((?:export\s+)?interface\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^((?:export\s+)?type\s+\w+\s*=[^;]*)", re.MULTILINE),
]

_GO_SIG_PATTERNS = [
    re.compile(r"^(func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\s*\([^)]*\)[^{]*)", re.MULTILINE),
    re.compile(r"^(type\s+\w+\s+(?:struct|interface)\s*)\{", re.MULTILINE),
]

_RUST_SIG_PATTERNS = [
    re.compile(r"^((?:pub\s+)?(?:async\s+)?fn\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^((?:pub\s+)?struct\s+\w+[^{;]*)", re.MULTILINE),
    re.compile(r"^((?:pub\s+)?trait\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^((?:pub\s+)?enum\s+\w+[^{]*)", re.MULTILINE),
    re.compile(r"^(impl(?:<[^>]+>)?\s+\w+[^{]*)", re.MULTILINE),
]

# Import patterns per language.
_IMPORT_PATTERNS = {
    "js": re.compile(r"^(import\s+.+)", re.MULTILINE),
    "go": re.compile(r"^(import\s+(?:\([\s\S]*?\)|\".+?\"))", re.MULTILINE),
    "rust": re.compile(r"^(use\s+.+;)", re.MULTILINE),
}


def _detect_language(text: str) -> str | None:
    """Detect language from content heuristics."""
    first_300 = text[:300]
    if "package main" in first_300 or "func " in text[:1000]:
        return "go"
    if "fn " in text[:1000] and ("use " in first_300 or "mod " in first_300):
        return "rust"
    if ("import " in first_300 or "require(" in first_300) and ("function " in text or "=>" in text or "class " in text):
        return "js"
    return None


def _signature_compress(source: str) -> str | None:
    """Extract imports + signatures for JS/TS/Go/Rust files."""
    lang = _detect_language(source)
    if lang is None:
        return None

    total_lines = len(source.split("\n"))
    parts: list[str] = []

    # Extract imports.
    import_pat = _IMPORT_PATTERNS.get(lang)
    if import_pat:
        for m in import_pat.finditer(source):
            parts.append(m.group(1).strip())

    if parts:
        parts.append("")

    # Extract signatures.
    sig_patterns = {
        "js": _JS_SIG_PATTERNS,
        "go": _GO_SIG_PATTERNS,
        "rust": _RUST_SIG_PATTERNS,
    }.get(lang, [])

    seen: set[str] = set()
    for pat in sig_patterns:
        for m in pat.finditer(source):
            sig = m.group(1).strip()
            if sig not in seen:
                seen.add(sig)
                parts.append(sig)

    if not seen:
        return None

    compressed_lines = len(parts)
    header = f"[Signature-compressed: {total_lines} lines -> {compressed_lines} lines, bodies omitted]\n\n"
    return header + "\n".join(parts)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_budget = int(max_chars * 0.7)
    tail_budget = max_chars - head_budget - 40  # room for marker
    head = text[:head_budget]
    tail = text[-tail_budget:] if tail_budget > 0 else ""
    omitted = len(text) - head_budget - tail_budget
    return f"{head}\n[...{omitted} chars omitted...]\n{tail}"
