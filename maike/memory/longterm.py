"""Long-term memory with keyword and vector backends."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from maike.constants import DEFAULT_VECTOR_STORE_RELATIVE_PATH

if TYPE_CHECKING:
    from maike.memory.taxonomy import MemoryEntry

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Keyword-matching long-term memory (original implementation)."""

    def __init__(self, base_path: Path) -> None:
        self.path = base_path / DEFAULT_VECTOR_STORE_RELATIVE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def add(self, collection: str, key: str, content: str) -> None:
        data = self._load()
        bucket = data.setdefault(collection, {})
        bucket[key] = content
        self._store(data)

    def query(self, collection: str, query: str, limit: int = 5) -> list[dict[str, str]]:
        data = self._load().get(collection, {})
        scored: list[tuple[int, str, str]] = []
        query_terms = set(query.lower().split())
        for key, content in data.items():
            score = sum(1 for term in query_terms if term in content.lower() or term in key.lower())
            if score:
                scored.append((score, key, content))
        scored.sort(reverse=True)
        return [
            {"key": key, "content": content}
            for _, key, content in scored[:limit]
        ]

    def _load(self) -> dict[str, dict[str, str]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _store(self, data: dict[str, dict[str, str]]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


# Keep old name for backward compatibility
KeywordLongTermMemory = LongTermMemory


class VectorLongTermMemory(LongTermMemory):
    """ChromaDB-backed long-term memory with semantic search.

    Falls back to keyword matching if chromadb is unavailable.
    """

    def __init__(self, base_path: Path) -> None:
        super().__init__(base_path)
        self._chroma_client = None
        self._chroma_available = False
        try:
            import chromadb

            persist_dir = str(base_path / ".maike" / "chromadb")
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            self._chroma_available = True
        except Exception:
            logger.debug("ChromaDB not available, falling back to keyword matching")

    def add(self, collection: str, key: str, content: str) -> None:
        super().add(collection, key, content)
        if self._chroma_available and self._chroma_client is not None:
            try:
                coll = self._chroma_client.get_or_create_collection(name=collection)
                coll.upsert(ids=[key], documents=[content])
            except Exception:
                logger.debug("ChromaDB upsert failed, keyword store is primary", exc_info=True)

    def query(self, collection: str, query: str, limit: int = 5) -> list[dict[str, str]]:
        if self._chroma_available and self._chroma_client is not None:
            try:
                coll = self._chroma_client.get_or_create_collection(name=collection)
                if coll.count() == 0:
                    return []
                results = coll.query(query_texts=[query], n_results=min(limit, coll.count()))
                ids = results.get("ids", [[]])[0]
                docs = results.get("documents", [[]])[0]
                return [{"key": k, "content": d} for k, d in zip(ids, docs)]
            except Exception:
                logger.debug("ChromaDB query failed, falling back to keyword", exc_info=True)
        return super().query(collection, query, limit)


class TypedLongTermMemory:
    """File-based long-term memory with structured type taxonomy.

    Stores memories as individual Markdown files with YAML frontmatter in a
    ``{workspace}/.maike/memories/`` directory.  Maintains a ``MEMORY.md``
    index file for quick scanning.

    This is the successor to the keyword/vector backends.  The old backends
    are preserved for backward compatibility.
    """

    def __init__(self, base_path: Path, *, memory_dir: Path | None = None) -> None:
        if memory_dir is not None:
            self.memory_dir = memory_dir
        else:
            self.memory_dir = base_path / ".maike" / "memories"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if memory_dir is None:
            self._migrate_from_json(base_path)

    def add(self, entry: "MemoryEntry") -> Path:
        """Save a memory entry as a Markdown file.

        Returns the path to the created file.
        """
        from maike.memory.taxonomy import write_memory_index

        slug = _slugify(entry.name)
        path = self.memory_dir / f"{slug}.md"

        # Avoid overwriting: append a counter if needed.
        counter = 1
        while path.exists():
            path = self.memory_dir / f"{slug}-{counter}.md"
            counter += 1

        path.write_text(entry.to_frontmatter_md(), encoding="utf-8")
        entry.file_path = str(path)

        # Rebuild index.
        entries = self.list_all()
        write_memory_index(self.memory_dir, entries)

        return path

    def list_all(self) -> "list[MemoryEntry]":
        """List all memory entries in the directory."""
        from maike.memory.taxonomy import MemoryEntry

        entries: list[MemoryEntry] = []
        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            entry = MemoryEntry.from_file(path)
            if entry:
                entries.append(entry)
        return entries

    def query(self, query: str, limit: int = 5) -> "list[MemoryEntry]":
        """Simple keyword-based query across memory entries.

        Scores entries by keyword overlap with their name, description,
        and content.  For semantic search, use the surfacer module instead.
        """
        entries = self.list_all()
        query_terms = set(query.lower().split())

        scored: list[tuple[int, "MemoryEntry"]] = []
        for entry in entries:
            searchable = f"{entry.name} {entry.description} {entry.content}".lower()
            score = sum(1 for term in query_terms if term in searchable)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def read_index(self) -> str:
        """Read the MEMORY.md index file, or generate it if missing."""
        from maike.memory.taxonomy import write_memory_index

        index_path = self.memory_dir / "MEMORY.md"
        if not index_path.exists():
            entries = self.list_all()
            if entries:
                write_memory_index(self.memory_dir, entries)
            else:
                return ""
        try:
            return index_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def load_topics_text(self, cap: int = 8000, per_file_cap: int = 2000) -> str:
        """Load all memories as formatted text sections.

        Returns a combined string of all topic files, each under a heading,
        truncated to *cap* total characters.  Individual files are capped at
        *per_file_cap* characters.

        Used by ``build_react_context()`` and ``build_delegate_context()`` to
        inject persistent memory into agent prompts.
        """
        entries = self.list_all()
        if not entries:
            return ""

        parts: list[str] = []
        total = 0
        for entry in entries:
            content = entry.content.strip()
            if not content:
                continue
            if len(content) > per_file_cap:
                content = content[:per_file_cap] + "\n... (truncated)"
            section = f"### {entry.name} ({entry.type.value})\n{content}"
            if total + len(section) > cap:
                break
            parts.append(section)
            total += len(section)

        return "\n\n".join(parts)

    def _migrate_from_json(self, base_path: Path) -> None:
        """Migrate old JSON-based memories to typed Markdown files.

        Only runs once — skips if the memory directory already has .md files
        (beyond MEMORY.md).
        """
        from maike.memory.taxonomy import MemoryEntry, MemoryType

        existing = [p for p in self.memory_dir.glob("*.md") if p.name != "MEMORY.md"]
        if existing:
            return  # Already migrated

        old_path = base_path / ".maike" / "long_term_memory.json"
        if not old_path.exists():
            return

        try:
            data = json.loads(old_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to read old memory file for migration: %s", old_path)
            return

        migrated = 0
        for collection, entries in data.items():
            if not isinstance(entries, dict):
                continue
            for key, content in entries.items():
                entry = MemoryEntry(
                    name=key,
                    description=f"Migrated from {collection}",
                    type=MemoryType.REFERENCE,  # safe default
                    content=str(content),
                )
                self.add(entry)
                migrated += 1

        if migrated:
            logger.info("Migrated %d memories from JSON to typed format", migrated)


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    import re
    slug = re.sub(r"[^\w\s-]", "", text.lower().strip())
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:64] or "memory"


def create_long_term_memory(base_path: Path, backend: str = "vector") -> LongTermMemory:
    """Factory function for long-term memory instances."""
    if backend == "vector":
        return VectorLongTermMemory(base_path)
    return LongTermMemory(base_path)


def create_typed_memory(base_path: Path) -> TypedLongTermMemory:
    """Factory function for the typed memory backend."""
    return TypedLongTermMemory(base_path)
