"""Code indexer — Python AST + regex-based extraction for other languages."""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

from maike.intelligence.models import FileEntry, ImportRef, Symbol, SymbolKind

logger = logging.getLogger(__name__)

# Reuse skip set from repomap.
_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".eggs", ".maike", ".egg-info",
}

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

_MAX_FILE_SIZE = 500_000  # 500KB — skip huge files


class CodeIndexer:
    """Indexes source files into Symbol and ImportRef collections."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    async def build_full_index(self) -> dict[str, FileEntry]:
        """Walk workspace and index every supported source file."""
        entries: dict[str, FileEntry] = {}
        for dirpath, dirnames, filenames in os.walk(self.workspace):
            # Prune skipped directories in-place.
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            ]
            for filename in filenames:
                full_path = Path(dirpath) / filename
                suffix = full_path.suffix.lower()
                if suffix not in _LANGUAGE_MAP:
                    continue
                try:
                    size = full_path.stat().st_size
                except OSError:
                    continue
                if size > _MAX_FILE_SIZE or size == 0:
                    continue
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rel_path = str(full_path.relative_to(self.workspace))
                entry = self.index_file(rel_path, content, suffix)
                if entry is not None:
                    entries[rel_path] = entry
        return entries

    def index_file(self, rel_path: str, content: str, suffix: str | None = None) -> FileEntry | None:
        """Index a single file. Returns None if unsupported."""
        if suffix is None:
            suffix = Path(rel_path).suffix.lower()
        language = _LANGUAGE_MAP.get(suffix)
        if language is None:
            return None
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        line_count = content.count("\n") + 1
        if language == "python":
            return self._index_python(rel_path, content, content_hash, line_count)
        return self._index_regex(rel_path, content, language, content_hash, line_count)

    # ------------------------------------------------------------------
    # Python AST indexer
    # ------------------------------------------------------------------

    def _index_python(
        self, rel_path: str, content: str, content_hash: str, line_count: int,
    ) -> FileEntry:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._index_regex(rel_path, content, "python", content_hash, line_count)

        module_qname = self._module_qualified_name(rel_path)
        symbols: list[Symbol] = []
        imports: list[ImportRef] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._python_function_symbol(
                    node, rel_path, module_qname, scope=None,
                ))
            elif isinstance(node, ast.ClassDef):
                symbols.append(self._python_class_symbol(node, rel_path, module_qname))
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(self._python_function_symbol(
                            item, rel_path, module_qname, scope=node.name,
                        ))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.append(Symbol(
                            name=target.id,
                            qualified_name=f"{module_qname}.{target.id}",
                            kind=SymbolKind.VARIABLE,
                            file=rel_path,
                            line=node.lineno,
                            end_line=node.end_lineno,
                        ))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportRef(
                        source_file=rel_path,
                        imported_name=alias.asname or alias.name.split(".")[-1],
                        module_path=alias.name,
                        is_from_import=False,
                        alias=alias.asname,
                        line=node.lineno,
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Handle relative imports.
                if node.level and node.level > 0:
                    module = self._resolve_relative_import(rel_path, module, node.level)
                for alias in node.names or []:
                    imports.append(ImportRef(
                        source_file=rel_path,
                        imported_name=alias.asname or alias.name,
                        module_path=module,
                        is_from_import=True,
                        alias=alias.asname,
                        line=node.lineno,
                    ))

        return FileEntry(
            path=rel_path,
            language="python",
            content_hash=content_hash,
            symbols=symbols,
            imports=imports,
            line_count=line_count,
        )

    def _python_function_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        rel_path: str,
        module_qname: str,
        scope: str | None,
    ) -> Symbol:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        if scope:
            kind = SymbolKind.ASYNC_METHOD if is_async else SymbolKind.METHOD
            qname = f"{module_qname}.{scope}.{node.name}"
        else:
            kind = SymbolKind.ASYNC_FUNCTION if is_async else SymbolKind.FUNCTION
            qname = f"{module_qname}.{node.name}"

        # Build signature.
        prefix = "async def" if is_async else "def"
        try:
            args_str = ast.unparse(node.args)
        except Exception:
            args_str = "..."
        returns = ""
        if node.returns:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass
        signature = f"{prefix} {node.name}({args_str}){returns}"

        # Docstring (first line, capped).
        docstring = ast.get_docstring(node)
        if docstring:
            docstring = docstring.split("\n")[0][:200]

        # Decorators.
        decorators: list[str] = []
        for dec in node.decorator_list:
            try:
                decorators.append(f"@{ast.unparse(dec)}")
            except Exception:
                decorators.append("@<unknown>")

        return Symbol(
            name=node.name,
            qualified_name=qname,
            kind=kind,
            file=rel_path,
            line=node.lineno,
            end_line=node.end_lineno,
            signature=signature,
            scope=scope,
            docstring=docstring,
            decorators=decorators,
        )

    def _python_class_symbol(
        self,
        node: ast.ClassDef,
        rel_path: str,
        module_qname: str,
    ) -> Symbol:
        qname = f"{module_qname}.{node.name}"
        bases = ""
        if node.bases:
            try:
                bases = ", ".join(ast.unparse(b) for b in node.bases)
            except Exception:
                bases = "..."
        signature = f"class {node.name}({bases})" if bases else f"class {node.name}"
        docstring = ast.get_docstring(node)
        if docstring:
            docstring = docstring.split("\n")[0][:200]
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(f"@{ast.unparse(dec)}")
            except Exception:
                decorators.append("@<unknown>")
        return Symbol(
            name=node.name,
            qualified_name=qname,
            kind=SymbolKind.CLASS,
            file=rel_path,
            line=node.lineno,
            end_line=node.end_lineno,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
        )

    def _module_qualified_name(self, rel_path: str) -> str:
        """Convert 'maike/agents/core.py' to 'maike.agents.core'."""
        path = Path(rel_path)
        parts = list(path.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _resolve_relative_import(self, source_file: str, module: str, level: int) -> str:
        """Resolve relative import like 'from . import X' or 'from ..foo import Y'."""
        parts = Path(source_file).parts[:-1]  # directory of source file
        # Go up 'level' directories.
        if level > len(parts):
            return module
        base_parts = parts[: len(parts) - level + 1]
        base = ".".join(base_parts)
        if module:
            return f"{base}.{module}" if base else module
        return base

    # ------------------------------------------------------------------
    # Regex-based indexer for JS/TS/Go/Rust
    # ------------------------------------------------------------------

    # Python regex patterns (fallback for syntax errors).
    _PY_SIG_PATTERNS = [
        re.compile(r"^(?:async\s+)?def\s+(\w+)", re.MULTILINE),
        re.compile(r"^class\s+(\w+)", re.MULTILINE),
    ]
    _PY_IMPORT_PATTERN = re.compile(
        r"^(?:from\s+([\w.]+)\s+import\s+|import\s+([\w.]+))", re.MULTILINE,
    )

    # JS/TS patterns (reused from repomap.py, enhanced with import extraction).
    _JS_SIG_PATTERNS = [
        re.compile(r"^export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)", re.MULTILINE),
        re.compile(r"^(?:async\s+)?function\s+(\w+)", re.MULTILINE),
        re.compile(r"^export\s+(?:default\s+)?class\s+(\w+)", re.MULTILINE),
        re.compile(r"^class\s+(\w+)", re.MULTILINE),
        re.compile(r"^(?:export\s+)?interface\s+(\w+)", re.MULTILINE),
        re.compile(r"^(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE),
    ]
    _JS_IMPORT_PATTERN = re.compile(
        r"""^import\s+(?:\{[^}]*\}\s+from\s+|.*\s+from\s+)?['"]([^'"]+)['"]""",
        re.MULTILINE,
    )

    # Go patterns.
    _GO_SIG_PATTERNS = [
        re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE),
        re.compile(r"^type\s+(\w+)\s+struct\s*\{", re.MULTILINE),
        re.compile(r"^type\s+(\w+)\s+interface\s*\{", re.MULTILINE),
    ]
    _GO_IMPORT_PATTERN = re.compile(r"""^\s*"([^"]+)"$""", re.MULTILINE)

    # Rust patterns.
    _RUST_SIG_PATTERNS = [
        re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", re.MULTILINE),
        re.compile(r"^(?:pub\s+)?struct\s+(\w+)", re.MULTILINE),
        re.compile(r"^(?:pub\s+)?trait\s+(\w+)", re.MULTILINE),
        re.compile(r"^(?:pub\s+)?enum\s+(\w+)", re.MULTILINE),
        re.compile(r"^impl(?:<[^>]+>)?\s+(\w+)", re.MULTILINE),
    ]
    _RUST_IMPORT_PATTERN = re.compile(r"^use\s+([\w:]+)", re.MULTILINE)

    _SYMBOL_KIND_MAP: dict[str, dict[str, SymbolKind]] = {
        "python": {
            "def": SymbolKind.FUNCTION,
            "async def": SymbolKind.ASYNC_FUNCTION,
            "class": SymbolKind.CLASS,
        },
        "javascript": {
            "function": SymbolKind.FUNCTION,
            "class": SymbolKind.CLASS,
            "interface": SymbolKind.INTERFACE,
            "type": SymbolKind.TYPE_ALIAS,
        },
        "typescript": {
            "function": SymbolKind.FUNCTION,
            "class": SymbolKind.CLASS,
            "interface": SymbolKind.INTERFACE,
            "type": SymbolKind.TYPE_ALIAS,
        },
        "go": {
            "func": SymbolKind.FUNCTION,
            "struct": SymbolKind.STRUCT,
            "interface": SymbolKind.INTERFACE,
        },
        "rust": {
            "fn": SymbolKind.FUNCTION,
            "struct": SymbolKind.STRUCT,
            "trait": SymbolKind.TRAIT,
            "enum": SymbolKind.ENUM,
            "impl": SymbolKind.IMPL,
        },
    }

    def _index_regex(
        self,
        rel_path: str,
        content: str,
        language: str,
        content_hash: str,
        line_count: int,
    ) -> FileEntry:
        symbols: list[Symbol] = []
        imports: list[ImportRef] = []
        lines = content.split("\n")

        # Select patterns by language.
        if language == "python":
            sig_patterns = self._PY_SIG_PATTERNS
            import_pattern = self._PY_IMPORT_PATTERN
        elif language in ("javascript", "typescript"):
            sig_patterns = self._JS_SIG_PATTERNS
            import_pattern = self._JS_IMPORT_PATTERN
        elif language == "go":
            sig_patterns = self._GO_SIG_PATTERNS
            import_pattern = self._GO_IMPORT_PATTERN
        elif language == "rust":
            sig_patterns = self._RUST_SIG_PATTERNS
            import_pattern = self._RUST_IMPORT_PATTERN
        else:
            sig_patterns = []
            import_pattern = None

        # Extract symbols.
        seen_names: set[str] = set()
        for pat in sig_patterns:
            for m in pat.finditer(content):
                name = m.group(1)
                if name in seen_names:
                    continue
                seen_names.add(name)
                line_no = content[:m.start()].count("\n") + 1
                sig = m.group(0).strip().rstrip("{").strip()
                # Infer kind from the signature text.
                kind = self._infer_kind(sig, language)
                symbols.append(Symbol(
                    name=name,
                    qualified_name=f"{rel_path}::{name}",
                    kind=kind,
                    file=rel_path,
                    line=line_no,
                    signature=sig,
                ))

        # Extract imports.
        if import_pattern is not None:
            for m in import_pattern.finditer(content):
                line_no = content[:m.start()].count("\n") + 1
                if language == "python":
                    # Python pattern has 2 groups: group(1) for 'from X import', group(2) for 'import X'
                    from_module = m.group(1)
                    import_module = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                    module_path = from_module or import_module or ""
                    is_from = from_module is not None
                else:
                    module_path = m.group(1)
                    is_from = True
                if module_path:
                    imports.append(ImportRef(
                        source_file=rel_path,
                        imported_name=module_path.split("/")[-1].split("::")[-1].split(".")[-1],
                        module_path=module_path,
                        is_from_import=is_from,
                        line=line_no,
                    ))

        return FileEntry(
            path=rel_path,
            language=language,
            content_hash=content_hash,
            symbols=symbols,
            imports=imports,
            line_count=line_count,
        )

    def _infer_kind(self, sig: str, language: str) -> SymbolKind:
        """Infer SymbolKind from a regex-matched signature string."""
        sig_lower = sig.lower()
        kind_map = self._SYMBOL_KIND_MAP.get(language, {})
        for keyword, kind in kind_map.items():
            if keyword in sig_lower:
                return kind
        return SymbolKind.FUNCTION
