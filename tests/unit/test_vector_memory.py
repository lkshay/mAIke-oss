"""Tests for maike.memory.longterm — keyword and vector memory backends."""

from maike.memory.longterm import (
    KeywordLongTermMemory,
    LongTermMemory,
    VectorLongTermMemory,
    create_long_term_memory,
)


def test_keyword_memory_add_and_query(tmp_path):
    memory = LongTermMemory(tmp_path)
    memory.add("notes", "note-1", "Python debugging with breakpoints")
    memory.add("notes", "note-2", "JavaScript async patterns")

    results = memory.query("notes", "Python debugging")
    assert len(results) >= 1
    assert any("Python" in r["content"] for r in results)


def test_keyword_memory_empty_query(tmp_path):
    memory = LongTermMemory(tmp_path)
    results = memory.query("notes", "anything")
    assert results == []


def test_keyword_memory_alias():
    assert KeywordLongTermMemory is LongTermMemory


def test_vector_memory_falls_back_to_keyword(tmp_path):
    """VectorLongTermMemory should always work, falling back to keyword if chromadb fails."""
    memory = VectorLongTermMemory(tmp_path)
    memory.add("learnings", "sess-1", "Used edit_file for Python refactoring")
    memory.add("learnings", "sess-2", "Flask REST API with SQLAlchemy")

    results = memory.query("learnings", "Python refactoring")
    assert len(results) >= 1
    contents = " ".join(r["content"] for r in results)
    assert "refactoring" in contents.lower()


def test_vector_memory_empty_collection(tmp_path):
    memory = VectorLongTermMemory(tmp_path)
    results = memory.query("empty_collection", "anything")
    assert results == []


def test_create_long_term_memory_keyword(tmp_path):
    memory = create_long_term_memory(tmp_path, backend="keyword")
    assert isinstance(memory, LongTermMemory)
    assert not isinstance(memory, VectorLongTermMemory)


def test_create_long_term_memory_vector(tmp_path):
    memory = create_long_term_memory(tmp_path, backend="vector")
    assert isinstance(memory, VectorLongTermMemory)


def test_create_long_term_memory_default_is_vector(tmp_path):
    memory = create_long_term_memory(tmp_path)
    assert isinstance(memory, VectorLongTermMemory)


def test_vector_memory_upsert_overwrites(tmp_path):
    memory = VectorLongTermMemory(tmp_path)
    memory.add("test", "key1", "original content")
    memory.add("test", "key1", "updated content")

    results = memory.query("test", "updated content")
    assert len(results) >= 1
    assert any("updated" in r["content"] for r in results)
