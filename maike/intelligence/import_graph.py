"""Import graph — directed file-to-file dependency edges."""

from __future__ import annotations

from pathlib import Path

from maike.intelligence.models import FileEntry


class ImportGraph:
    """Directed graph of file-to-file import relationships.

    Forward edges: file A imports file B.
    Reverse edges: file B is imported by file A.
    """

    def __init__(self) -> None:
        self._forward: dict[str, set[str]] = {}
        self._reverse: dict[str, set[str]] = {}

    def build_from_entries(self, entries: dict[str, FileEntry], workspace: Path) -> None:
        """Build the graph from indexed FileEntry objects."""
        self._forward.clear()
        self._reverse.clear()
        workspace = workspace.resolve()

        for path, entry in entries.items():
            self._forward.setdefault(path, set())
            for imp in entry.imports:
                target = self._resolve_import(
                    imp.module_path,
                    source_file=path,
                    language=entry.language,
                    workspace=workspace,
                    known_files=set(entries.keys()),
                )
                if target is not None and target != path:
                    self._forward[path].add(target)
                    self._reverse.setdefault(target, set()).add(path)

    def imports_of(self, file_path: str) -> set[str]:
        """Files that the given file imports (forward edges)."""
        return set(self._forward.get(file_path, set()))

    def importers_of(self, file_path: str) -> set[str]:
        """Files that import the given file (reverse edges)."""
        return set(self._reverse.get(file_path, set()))

    def related_files(self, file_path: str, depth: int = 1) -> set[str]:
        """Files related by imports (both directions), up to *depth* hops."""
        result: set[str] = set()
        frontier = {file_path}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for f in frontier:
                next_frontier |= self.imports_of(f)
                next_frontier |= self.importers_of(f)
            next_frontier -= result
            next_frontier.discard(file_path)
            result |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return result

    def _resolve_import(
        self,
        module_path: str,
        source_file: str,
        language: str,
        workspace: Path,
        known_files: set[str],
    ) -> str | None:
        """Resolve a module path to a workspace file path."""
        if language == "python":
            return self._resolve_python_import(module_path, workspace, known_files)
        if language in ("javascript", "typescript"):
            return self._resolve_js_import(module_path, source_file, known_files)
        return None

    def _resolve_python_import(
        self,
        module_path: str,
        workspace: Path,
        known_files: set[str],
    ) -> str | None:
        """Resolve 'maike.agents.core' to 'maike/agents/core.py'."""
        if not module_path:
            return None
        parts = module_path.split(".")
        # Try direct file: maike/agents/core.py
        candidate = str(Path(*parts).with_suffix(".py"))
        if candidate in known_files:
            return candidate
        # Try package __init__: maike/agents/core/__init__.py
        init_candidate = str(Path(*parts) / "__init__.py")
        if init_candidate in known_files:
            return init_candidate
        # Try parent module: from maike.agents import core
        # → maike/agents.py (where 'core' is a name in that file)
        if len(parts) > 1:
            parent = str(Path(*parts[:-1]).with_suffix(".py"))
            if parent in known_files:
                return parent
            parent_init = str(Path(*parts[:-1]) / "__init__.py")
            if parent_init in known_files:
                return parent_init
        return None

    def _resolve_js_import(
        self,
        module_path: str,
        source_file: str,
        known_files: set[str],
    ) -> str | None:
        """Resolve './foo' or '../bar' to a workspace file."""
        if not module_path.startswith("."):
            return None  # Skip node_modules / absolute imports.
        source_dir = str(Path(source_file).parent)
        resolved = str(Path(source_dir) / module_path)
        # Normalize.
        resolved = str(Path(resolved))
        # Try with extensions.
        for ext in (".js", ".jsx", ".ts", ".tsx"):
            candidate = resolved + ext
            if candidate in known_files:
                return candidate
        # Try index file.
        for ext in (".js", ".ts"):
            candidate = str(Path(resolved) / f"index{ext}")
            if candidate in known_files:
                return candidate
        return None
