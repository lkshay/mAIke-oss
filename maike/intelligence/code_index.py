"""Code index facade — combines symbol index, import graph, and embeddings."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from maike.intelligence.import_graph import ImportGraph
from maike.intelligence.indexer import CodeIndexer
from maike.intelligence.models import (
    CodeChunk,
    FileEntry,
    ImportRef,
    IndexStats,
    Symbol,
    SymbolKind,
)

logger = logging.getLogger(__name__)

# Output caps.
_SMART_MAP_CAP = 12_000
_FIND_SYMBOL_MAX = 30
_FIND_REFERENCES_MAX = 50


class CodeIndex:
    """Facade combining symbol index, import graph, and (optionally) code embeddings."""

    def __init__(self, workspace: Path, session_id: str) -> None:
        self.workspace = workspace.resolve()
        self.session_id = session_id
        self._indexer = CodeIndexer(self.workspace)
        self._entries: dict[str, FileEntry] = {}
        self._symbol_name_index: dict[str, list[Symbol]] = {}
        self._qualified_name_index: dict[str, Symbol] = {}
        self._import_graph = ImportGraph()
        self._built = False
        self._stats = IndexStats()

        # Initialize embedding store (graceful degradation if ChromaDB unavailable).
        try:
            from maike.intelligence.embeddings import CodeEmbeddingStore

            self._embedding_store: Any = CodeEmbeddingStore(session_id, self.workspace)
            if not self._embedding_store.available:
                self._embedding_store = None
        except Exception:
            self._embedding_store = None

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def stats(self) -> IndexStats:
        return self._stats

    async def build(self) -> IndexStats:
        """Build the full index: symbols, import graph, embeddings."""
        start = time.monotonic()
        self._entries = await self._indexer.build_full_index()
        self._build_symbol_indices()
        self._import_graph.build_from_entries(self._entries, self.workspace)

        # Build embeddings if available.
        total_chunks = 0
        if self._embedding_store is not None:
            try:
                total_chunks = self._embedding_store.build_from_entries(self._entries)
            except Exception:
                logger.debug("Embedding build failed", exc_info=True)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        languages: dict[str, int] = {}
        for entry in self._entries.values():
            languages[entry.language] = languages.get(entry.language, 0) + 1

        self._stats = IndexStats(
            total_files=len(self._entries),
            total_symbols=sum(len(e.symbols) for e in self._entries.values()),
            total_imports=sum(len(e.imports) for e in self._entries.values()),
            total_chunks=total_chunks,
            build_time_ms=elapsed_ms,
            languages=languages,
        )
        self._built = True
        return self._stats

    async def update_file(self, path: str, content: str) -> None:
        """Incrementally update the index for a single file."""
        rel_path = self._to_relative(path)
        suffix = Path(rel_path).suffix.lower()
        entry = self._indexer.index_file(rel_path, content, suffix)
        if entry is None:
            return

        # Check staleness.
        existing = self._entries.get(rel_path)
        if existing is not None and existing.content_hash == entry.content_hash:
            return

        # Remove old symbols from indices.
        if existing is not None:
            self._remove_from_symbol_indices(existing)

        # Insert new.
        self._entries[rel_path] = entry
        self._add_to_symbol_indices(entry)
        self._import_graph.build_from_entries(self._entries, self.workspace)

        # Update embeddings.
        if self._embedding_store is not None:
            try:
                self._embedding_store.update_file(rel_path, entry, content)
            except Exception:
                logger.debug("Embedding update failed for %s", rel_path, exc_info=True)

    async def remove_file(self, path: str) -> None:
        """Remove a file from the index."""
        rel_path = self._to_relative(path)
        existing = self._entries.pop(rel_path, None)
        if existing is not None:
            self._remove_from_symbol_indices(existing)
            self._import_graph.build_from_entries(self._entries, self.workspace)
            if self._embedding_store is not None:
                try:
                    self._embedding_store.remove_file(rel_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Symbol queries
    # ------------------------------------------------------------------

    def find_symbol(
        self,
        name: str,
        *,
        kind: SymbolKind | None = None,
        file_filter: str | None = None,
    ) -> list[Symbol]:
        """Find symbols by name (exact or substring). O(1) for exact match."""
        # Exact match first.
        results = list(self._symbol_name_index.get(name, []))
        # If no exact match, try case-insensitive substring.
        if not results:
            name_lower = name.lower()
            for sym_name, syms in self._symbol_name_index.items():
                if name_lower in sym_name.lower():
                    results.extend(syms)
        if kind is not None:
            results = [s for s in results if s.kind == kind]
        if file_filter is not None:
            results = [s for s in results if file_filter in s.file]
        return results[:_FIND_SYMBOL_MAX]

    def find_references(self, symbol_name: str) -> list[ImportRef]:
        """Find all import references to a symbol name."""
        results: list[ImportRef] = []
        for entry in self._entries.values():
            for imp in entry.imports:
                if (
                    imp.imported_name == symbol_name
                    or imp.module_path.endswith(f".{symbol_name}")
                    or imp.module_path == symbol_name
                ):
                    results.append(imp)
        return results[:_FIND_REFERENCES_MAX]

    # ------------------------------------------------------------------
    # Import graph queries
    # ------------------------------------------------------------------

    def find_related_files(self, file_path: str, depth: int = 1) -> dict[str, str]:
        """Find files related by imports. Returns {path: relationship}."""
        rel_path = self._to_relative(file_path)
        result: dict[str, str] = {}
        for f in self._import_graph.imports_of(rel_path):
            result[f] = "imported_by_target"
        for f in self._import_graph.importers_of(rel_path):
            result[f] = "imports_target"
        if depth > 1:
            for f in self._import_graph.related_files(rel_path, depth=depth):
                if f not in result:
                    result[f] = "transitive"
        return result

    # ------------------------------------------------------------------
    # Semantic search (wired in Phase 6)
    # ------------------------------------------------------------------

    async def semantic_search(self, query: str, limit: int = 10) -> list[CodeChunk]:
        """Natural language code search via embeddings."""
        if self._embedding_store is None:
            return []
        try:
            return self._embedding_store.search(query, limit=limit)
        except Exception:
            logger.debug("Semantic search failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Enhanced repo map
    # ------------------------------------------------------------------

    def smart_repo_map(
        self,
        path: str = ".",
        *,
        focus_file: str | None = None,
        max_depth: int = 3,
    ) -> str:
        """Repo map that uses the symbol index for richer output.

        When *focus_file* is provided, shows that file in full detail
        with its related files, then other files in abbreviated form.
        """
        focus_rel = self._to_relative(focus_file) if focus_file else None
        related: set[str] = set()
        if focus_rel and focus_rel in self._entries:
            related = self._import_graph.related_files(focus_rel, depth=1)

        lines: list[str] = []
        total_chars = 0

        # Focus file first (full detail).
        if focus_rel and focus_rel in self._entries:
            lines.append(f"### {focus_rel} (focus)")
            for sym in self._entries[focus_rel].symbols:
                line = f"  {sym.signature or f'{sym.kind.value} {sym.name}'}"
                lines.append(line)
                total_chars += len(line) + 1

        # Related files (signatures).
        if related:
            lines.append("\n### Related files")
            for rel_path in sorted(related):
                if total_chars >= _SMART_MAP_CAP:
                    break
                entry = self._entries.get(rel_path)
                if entry is None:
                    continue
                lines.append(f"\n  {rel_path}")
                total_chars += len(rel_path) + 3
                for sym in entry.symbols[:5]:
                    if total_chars >= _SMART_MAP_CAP:
                        break
                    line = f"    {sym.signature or sym.name}"
                    lines.append(line)
                    total_chars += len(line) + 1

        # Remaining files (names only).
        remaining = sorted(
            p for p in self._entries
            if p != focus_rel and p not in related
        )
        if remaining:
            lines.append("\n### Other files")
            for p in remaining:
                if total_chars >= _SMART_MAP_CAP:
                    lines.append(f"  ... ({len(remaining) - len(lines)} more files)")
                    break
                lines.append(f"  {p}")
                total_chars += len(p) + 3

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_symbol_indices(self) -> None:
        self._symbol_name_index.clear()
        self._qualified_name_index.clear()
        for entry in self._entries.values():
            self._add_to_symbol_indices(entry)

    def _add_to_symbol_indices(self, entry: FileEntry) -> None:
        for sym in entry.symbols:
            self._symbol_name_index.setdefault(sym.name, []).append(sym)
            self._qualified_name_index[sym.qualified_name] = sym

    def _remove_from_symbol_indices(self, entry: FileEntry) -> None:
        for sym in entry.symbols:
            syms = self._symbol_name_index.get(sym.name, [])
            self._symbol_name_index[sym.name] = [s for s in syms if s.file != entry.path]
            if not self._symbol_name_index[sym.name]:
                del self._symbol_name_index[sym.name]
            self._qualified_name_index.pop(sym.qualified_name, None)

    def _to_relative(self, path: str) -> str:
        """Convert absolute or mixed path to workspace-relative."""
        try:
            return str(Path(path).resolve().relative_to(self.workspace))
        except (ValueError, RuntimeError):
            return path
