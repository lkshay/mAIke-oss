"""Code embedding store — ChromaDB-backed semantic search over code chunks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from maike.intelligence.models import CodeChunk, FileEntry

logger = logging.getLogger(__name__)

_MIN_CHUNK_LINES = 5
_MAX_CHUNK_LINES = 100
_HEAD_KEEP = 40
_TAIL_KEEP = 40


def _chunk_metadata(c: CodeChunk) -> dict[str, str | int | bool]:
    """Build ChromaDB metadata dict for a CodeChunk."""
    return {
        "file": c.file,
        "kind": c.kind,
        "language": c.language,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "symbol_name": c.symbol_name or "",
        "parent_class": c.parent_class or "",
        "has_overlap": c.has_overlap,
        "overlap_lines": c.overlap_lines,
    }


class CodeEmbeddingStore:
    """ChromaDB-backed store for code chunk embeddings with semantic search.

    Gracefully degrades to empty results if ChromaDB is unavailable.
    """

    def __init__(self, session_id: str, workspace: Path) -> None:
        self.session_id = session_id
        self.workspace = workspace
        self._collection_name = f"code_{session_id[:8]}"
        self._client: Any = None
        self._collection: Any = None
        self._available = False
        try:
            import chromadb

            persist_dir = str(workspace / ".maike" / "chromadb")
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
            )
            self._available = True
        except Exception:
            logger.debug("ChromaDB not available for code embeddings", exc_info=True)

    @property
    def available(self) -> bool:
        return self._available

    def build_from_entries(self, entries: dict[str, FileEntry]) -> int:
        """Chunk all indexed files and embed into ChromaDB. Returns chunk count."""
        if not self._available or self._collection is None:
            return 0

        all_chunks: list[CodeChunk] = []
        for path, entry in entries.items():
            try:
                content = (self.workspace / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            chunks = self._chunk_file(path, content, entry)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Batch upsert into ChromaDB.
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.content for c in batch],
                metadatas=[_chunk_metadata(c) for c in batch],
            )
        return len(all_chunks)

    def search(self, query: str, limit: int = 10) -> list[CodeChunk]:
        """Semantic search across code chunks."""
        if not self._available or self._collection is None:
            return []
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(limit, self._collection.count()),
        )
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        chunks: list[CodeChunk] = []
        for chunk_id, content, meta in zip(ids, docs, metadatas):
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file=meta.get("file", ""),
                start_line=int(meta.get("start_line", 0)),
                end_line=int(meta.get("end_line", 0)),
                content=content,
                symbol_name=meta.get("symbol_name") or None,
                kind=meta.get("kind", "function"),
                language=meta.get("language", "python"),
                parent_class=meta.get("parent_class") or None,
                has_overlap=bool(meta.get("has_overlap", False)),
                overlap_lines=int(meta.get("overlap_lines", 0)),
            ))
        return chunks

    def update_file(self, path: str, entry: FileEntry, content: str) -> None:
        """Incrementally update chunks for a single file."""
        if not self._available or self._collection is None:
            return
        self.remove_file(path)
        chunks = self._chunk_file(path, content, entry)
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[_chunk_metadata(c) for c in chunks],
        )

    def remove_file(self, path: str) -> None:
        """Remove all chunks for a file."""
        if not self._available or self._collection is None:
            return
        try:
            self._collection.delete(where={"file": path})
        except Exception:
            pass

    def _chunk_file(
        self,
        path: str,
        content: str,
        entry: FileEntry,
    ) -> list[CodeChunk]:
        """Split a file into embeddable chunks at symbol boundaries."""
        lines = content.split("\n")
        if not lines:
            return []

        chunks: list[CodeChunk] = []
        used_lines: set[int] = set()

        # Sort symbols by line number.
        symbols = sorted(entry.symbols, key=lambda s: s.line)

        # Build a map of class scopes for parent class context.
        # Each entry: (class_name, class_start_0idx, class_end_0idx, signature_line)
        class_scopes: list[tuple[str, int, int, str]] = []
        for sym in symbols:
            if sym.kind.value == "class":
                cls_start = sym.line - 1
                cls_end = (sym.end_line or sym.line) - 1
                cls_end = min(cls_end, len(lines) - 1)
                sig_line = lines[cls_start] if cls_start < len(lines) else ""
                class_scopes.append((sym.name, cls_start, cls_end, sig_line))

        prev_chunk_lines: list[str] = []

        for sym in symbols:
            start = sym.line - 1  # 0-indexed
            end = (sym.end_line or sym.line + _MIN_CHUNK_LINES) - 1
            end = min(end, len(lines) - 1)

            if end - start + 1 < _MIN_CHUNK_LINES:
                # Extend to minimum chunk size.
                end = min(start + _MIN_CHUNK_LINES - 1, len(lines) - 1)

            chunk_lines = lines[start : end + 1]

            # Determine parent class context for methods.
            parent_class: str | None = None
            class_prefix: str = ""
            if sym.kind.value in ("method", "async_method"):
                for cls_name, cls_start, cls_end, sig_line in class_scopes:
                    if cls_start <= start <= cls_end:
                        parent_class = cls_name
                        class_prefix = sig_line.rstrip()
                        break

            # Head+tail truncation for large chunks.
            if len(chunk_lines) > _MAX_CHUNK_LINES:
                omitted = len(chunk_lines) - _HEAD_KEEP - _TAIL_KEEP
                head = chunk_lines[:_HEAD_KEEP]
                tail = chunk_lines[-_TAIL_KEEP:]
                marker = f"# ... ({omitted} lines omitted for embedding)"
                chunk_lines = head + [marker] + tail
                # end_line still reflects the original span
            else:
                end = start + len(chunk_lines) - 1

            # Calculate overlap from previous chunk.
            overlap_count = 0
            overlap_prefix: list[str] = []
            if prev_chunk_lines:
                overlap_count = max(1, int(len(prev_chunk_lines) * 0.2))
                overlap_prefix = prev_chunk_lines[-overlap_count:]

            # Build final content: class prefix + overlap + chunk body.
            prefix_parts: list[str] = []
            if class_prefix:
                prefix_parts.append(class_prefix)
            if overlap_prefix:
                prefix_parts.extend(overlap_prefix)

            if prefix_parts:
                chunk_content = "\n".join(prefix_parts) + "\n" + "\n".join(chunk_lines)
            else:
                chunk_content = "\n".join(chunk_lines)

            chunks.append(CodeChunk(
                chunk_id=f"{path}::{sym.qualified_name}",
                file=path,
                start_line=start + 1,
                end_line=end + 1,
                content=chunk_content,
                symbol_name=sym.qualified_name,
                kind=sym.kind.value,
                language=entry.language,
                parent_class=parent_class,
                has_overlap=overlap_count > 0,
                overlap_lines=overlap_count,
            ))
            used_lines.update(range(start, end + 1))
            prev_chunk_lines = chunk_lines

        # Module header: imports + module-level code not covered by symbols.
        header_lines: list[str] = []
        for i, line in enumerate(lines):
            if i in used_lines:
                break
            header_lines.append(line)
        if header_lines and len(header_lines) >= 2:
            chunks.insert(0, CodeChunk(
                chunk_id=f"{path}::module_header",
                file=path,
                start_line=1,
                end_line=len(header_lines),
                content="\n".join(header_lines),
                symbol_name=None,
                kind="module_header",
                language=entry.language,
            ))

        return chunks
