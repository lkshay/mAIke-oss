"""Tests for maike.intelligence.code_index — facade over indexer + graph."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

from maike.intelligence.code_index import CodeIndex
from maike.intelligence.embeddings import CodeEmbeddingStore
from maike.intelligence.models import FileEntry, Symbol, SymbolKind


def _write_files(root: Path, files: dict[str, str]) -> None:
    for path, content in files.items():
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)


SAMPLE_FILES = {
    "app.py": textwrap.dedent("""\
        from lib import Calculator
        from utils.helper import greet

        def main():
            calc = Calculator()
            print(greet("World"))
            print(calc.add(1, 2))
    """),
    "lib.py": textwrap.dedent("""\
        class Calculator:
            def add(self, a: int, b: int) -> int:
                return a + b

            def multiply(self, a: int, b: int) -> int:
                return a * b
    """),
    "utils/__init__.py": "",
    "utils/helper.py": textwrap.dedent("""\
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        def farewell(name: str) -> str:
            return f"Goodbye, {name}!"
    """),
}


class TestCodeIndexBuild:
    def test_build_indexes_all_files(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        stats = asyncio.run(index.build())
        assert stats.total_files >= 3  # app.py, lib.py, utils/helper.py (empty __init__.py may be skipped)
        assert stats.total_symbols > 0
        assert stats.total_imports > 0
        assert index.is_built

    def test_build_records_languages(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        stats = asyncio.run(index.build())
        assert stats.languages.get("python", 0) >= 3


class TestFindSymbol:
    def test_exact_match(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        results = index.find_symbol("Calculator")
        assert len(results) >= 1
        assert results[0].name == "Calculator"
        assert results[0].kind == SymbolKind.CLASS

    def test_method_search(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        results = index.find_symbol("add")
        assert len(results) >= 1
        assert any(s.kind == SymbolKind.METHOD and s.scope == "Calculator" for s in results)

    def test_kind_filter(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        results = index.find_symbol("greet", kind=SymbolKind.FUNCTION)
        assert len(results) >= 1
        assert all(s.kind == SymbolKind.FUNCTION for s in results)

    def test_no_match(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        results = index.find_symbol("nonexistent_xyz_abc")
        assert results == []

    def test_substring_match(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        results = index.find_symbol("Calc")
        assert len(results) >= 1  # should match Calculator


class TestFindReferences:
    def test_import_references(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        refs = index.find_references("Calculator")
        assert len(refs) >= 1
        assert any(r.source_file == "app.py" for r in refs)

    def test_no_references(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        refs = index.find_references("nonexistent_symbol")
        assert refs == []


class TestFindRelatedFiles:
    def test_related_files(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        related = index.find_related_files("app.py")
        # app.py imports lib and utils.helper
        assert len(related) >= 1


class TestSmartRepoMap:
    def test_basic_map(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        output = index.smart_repo_map()
        assert "app.py" in output or "lib.py" in output

    def test_focused_map(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())
        output = index.smart_repo_map(focus_file="lib.py")
        assert "lib.py" in output
        assert "(focus)" in output


class TestSemanticSearch:
    def test_semantic_search_returns_results_or_degrades(self, tmp_path):
        """If embeddings are available, semantic_search returns results;
        otherwise it degrades gracefully with an empty list."""
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())

        results = asyncio.run(index.semantic_search("calculator add numbers"))
        # Whether ChromaDB is installed determines the outcome.
        if index._embedding_store is not None:
            # Embeddings were built — expect at least one hit.
            assert len(results) >= 1
            assert results[0].file  # non-empty file path
            assert results[0].content  # non-empty content
        else:
            # No embeddings — graceful degradation to empty list.
            assert results == []

    def test_semantic_search_without_build(self, tmp_path):
        """Calling semantic_search before build returns empty list."""
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        results = asyncio.run(index.semantic_search("anything"))
        assert results == []


class TestUpdateFile:
    def test_update_adds_new_symbols(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())

        # Initially no 'new_function'
        assert index.find_symbol("new_function") == []

        # Write new file
        new_content = "def new_function():\n    pass\n"
        asyncio.run(index.update_file("new_module.py", new_content))

        # Now found
        results = index.find_symbol("new_function")
        assert len(results) == 1

    def test_update_replaces_symbols(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())

        # Update lib.py with new method
        new_content = textwrap.dedent("""\
            class Calculator:
                def subtract(self, a: int, b: int) -> int:
                    return a - b
        """)
        asyncio.run(index.update_file("lib.py", new_content))

        # subtract should exist, add should not
        assert len(index.find_symbol("subtract")) >= 1
        add_results = [s for s in index.find_symbol("add") if s.file == "lib.py"]
        assert add_results == []

    def test_remove_file_clears_symbols(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())

        # "greet" should exist in utils/helper.py before removal.
        results = index.find_symbol("greet")
        assert len(results) >= 1
        assert any(s.file == "utils/helper.py" for s in results)

        # Remove the file from the index.
        asyncio.run(index.remove_file("utils/helper.py"))

        # Symbol should no longer be found.
        assert index.find_symbol("greet") == []
        assert index.find_symbol("farewell") == []

    def test_update_skips_if_hash_unchanged(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test-session")
        asyncio.run(index.build())

        # Re-update with same content should be a no-op.
        content = (tmp_path / "lib.py").read_text()
        asyncio.run(index.update_file("lib.py", content))
        assert len(index.find_symbol("Calculator")) >= 1


# ---------------------------------------------------------------------------
# Tests for _chunk_file embedding quality improvements
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> CodeEmbeddingStore:
    """Create a CodeEmbeddingStore (ChromaDB availability not required for _chunk_file)."""
    return CodeEmbeddingStore("test-session-1234", tmp_path)


def _make_entry(
    path: str,
    language: str,
    symbols: list[Symbol],
) -> FileEntry:
    return FileEntry(
        path=path,
        language=language,
        content_hash="fakehash",
        symbols=symbols,
        line_count=0,
    )


class TestChunkFileParentClassContext:
    """Method chunks should include the parent class signature line."""

    def test_method_chunk_includes_class_signature(self, tmp_path):
        content = textwrap.dedent("""\
            class CacheStore:
                def __init__(self, max_size: int = 100):
                    self._cache = {}
                    self._max_size = max_size

                def get(self, key: str) -> str | None:
                    return self._cache.get(key)

                def set(self, key: str, value: str) -> None:
                    self._cache[key] = value
        """)
        symbols = [
            Symbol(
                name="CacheStore",
                qualified_name="CacheStore",
                kind=SymbolKind.CLASS,
                file="cache.py",
                line=1,
                end_line=10,
                signature="class CacheStore:",
            ),
            Symbol(
                name="__init__",
                qualified_name="CacheStore.__init__",
                kind=SymbolKind.METHOD,
                file="cache.py",
                line=2,
                end_line=4,
                scope="CacheStore",
            ),
            Symbol(
                name="get",
                qualified_name="CacheStore.get",
                kind=SymbolKind.METHOD,
                file="cache.py",
                line=6,
                end_line=7,
                scope="CacheStore",
            ),
            Symbol(
                name="set",
                qualified_name="CacheStore.set",
                kind=SymbolKind.METHOD,
                file="cache.py",
                line=9,
                end_line=10,
                scope="CacheStore",
            ),
        ]
        entry = _make_entry("cache.py", "python", symbols)
        store = _make_store(tmp_path)

        chunks = store._chunk_file("cache.py", content, entry)

        # Find method chunks (skip the class chunk itself).
        method_chunks = [c for c in chunks if c.kind == "method"]
        assert len(method_chunks) >= 1

        for mc in method_chunks:
            assert mc.parent_class == "CacheStore"
            # Content should start with the class signature line.
            assert mc.content.startswith("class CacheStore:")

    def test_standalone_function_has_no_parent_class(self, tmp_path):
        content = textwrap.dedent("""\
            def standalone():
                return 42
        """)
        symbols = [
            Symbol(
                name="standalone",
                qualified_name="standalone",
                kind=SymbolKind.FUNCTION,
                file="funcs.py",
                line=1,
                end_line=2,
            ),
        ]
        entry = _make_entry("funcs.py", "python", symbols)
        store = _make_store(tmp_path)

        chunks = store._chunk_file("funcs.py", content, entry)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].parent_class is None


class TestChunkFileOverlap:
    """Consecutive chunks should share overlap lines."""

    def test_chunks_have_overlap(self, tmp_path):
        # Build a file with two functions, each 10 lines.
        func_a_lines = ["def func_a():"] + [f"    x_{i} = {i}" for i in range(9)]
        func_b_lines = ["def func_b():"] + [f"    y_{i} = {i}" for i in range(9)]
        content = "\n".join(func_a_lines + func_b_lines)

        symbols = [
            Symbol(
                name="func_a",
                qualified_name="func_a",
                kind=SymbolKind.FUNCTION,
                file="two_funcs.py",
                line=1,
                end_line=10,
            ),
            Symbol(
                name="func_b",
                qualified_name="func_b",
                kind=SymbolKind.FUNCTION,
                file="two_funcs.py",
                line=11,
                end_line=20,
            ),
        ]
        entry = _make_entry("two_funcs.py", "python", symbols)
        store = _make_store(tmp_path)

        chunks = store._chunk_file("two_funcs.py", content, entry)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 2

        first_chunk = func_chunks[0]
        second_chunk = func_chunks[1]

        # First chunk has no overlap (no predecessor).
        assert first_chunk.has_overlap is False
        assert first_chunk.overlap_lines == 0

        # Second chunk has overlap from the first.
        assert second_chunk.has_overlap is True
        assert second_chunk.overlap_lines >= 1

        # The overlap lines from the end of first chunk should appear
        # at the start of the second chunk's content.
        first_content_lines = first_chunk.content.split("\n")
        overlap_n = second_chunk.overlap_lines
        expected_overlap = first_content_lines[-overlap_n:]
        second_content_lines = second_chunk.content.split("\n")
        assert second_content_lines[:overlap_n] == expected_overlap

    def test_metadata_overlap_fields(self, tmp_path):
        func_a_lines = ["def a():"] + ["    pass"] * 5
        func_b_lines = ["def b():"] + ["    pass"] * 5
        content = "\n".join(func_a_lines + func_b_lines)

        symbols = [
            Symbol(name="a", qualified_name="a", kind=SymbolKind.FUNCTION,
                   file="f.py", line=1, end_line=6),
            Symbol(name="b", qualified_name="b", kind=SymbolKind.FUNCTION,
                   file="f.py", line=7, end_line=12),
        ]
        entry = _make_entry("f.py", "python", symbols)
        store = _make_store(tmp_path)

        chunks = store._chunk_file("f.py", content, entry)
        func_chunks = [c for c in chunks if c.kind == "function"]

        # Verify metadata fields exist and are correctly typed.
        for fc in func_chunks:
            assert isinstance(fc.has_overlap, bool)
            assert isinstance(fc.overlap_lines, int)


class TestChunkFileTruncation:
    """Large functions should be truncated with head+tail strategy."""

    def test_large_function_head_tail_truncation(self, tmp_path):
        # Create a 200-line function.
        func_lines = ["def big_function():"]
        for i in range(1, 200):
            func_lines.append(f"    line_{i} = {i}")
        content = "\n".join(func_lines)

        symbols = [
            Symbol(
                name="big_function",
                qualified_name="big_function",
                kind=SymbolKind.FUNCTION,
                file="big.py",
                line=1,
                end_line=200,
            ),
        ]
        entry = _make_entry("big.py", "python", symbols)
        store = _make_store(tmp_path)

        chunks = store._chunk_file("big.py", content, entry)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1

        chunk = func_chunks[0]
        chunk_lines = chunk.content.split("\n")

        # Should contain head (40) + marker (1) + tail (40) = 81 lines.
        assert len(chunk_lines) == 81

        # First line is the function def.
        assert chunk_lines[0] == "def big_function():"

        # Marker should be present.
        marker_lines = [l for l in chunk_lines if "lines omitted for embedding" in l]
        assert len(marker_lines) == 1
        assert "120 lines omitted" in marker_lines[0]

        # Last line should be from the tail (the last line of the original).
        assert chunk_lines[-1] == "    line_199 = 199"

    def test_small_function_not_truncated(self, tmp_path):
        func_lines = ["def small():"] + [f"    x = {i}" for i in range(20)]
        content = "\n".join(func_lines)

        symbols = [
            Symbol(
                name="small",
                qualified_name="small",
                kind=SymbolKind.FUNCTION,
                file="small.py",
                line=1,
                end_line=21,
            ),
        ]
        entry = _make_entry("small.py", "python", symbols)
        store = _make_store(tmp_path)

        chunks = store._chunk_file("small.py", content, entry)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1

        # No truncation marker.
        assert "lines omitted" not in func_chunks[0].content


class TestChunkFileMetadata:
    """Verify all enhanced metadata fields are populated correctly."""

    def test_full_metadata(self, tmp_path):
        content = textwrap.dedent("""\
            class MyService:
                def handle(self, request):
                    return "ok"

            def helper():
                pass
        """)
        symbols = [
            Symbol(name="MyService", qualified_name="MyService",
                   kind=SymbolKind.CLASS, file="svc.py", line=1, end_line=3),
            Symbol(name="handle", qualified_name="MyService.handle",
                   kind=SymbolKind.METHOD, file="svc.py", line=2, end_line=3,
                   scope="MyService"),
            Symbol(name="helper", qualified_name="helper",
                   kind=SymbolKind.FUNCTION, file="svc.py", line=5, end_line=6),
        ]
        entry = _make_entry("svc.py", "python", symbols)
        store = _make_store(tmp_path)

        chunks = store._chunk_file("svc.py", content, entry)

        # Find chunks by symbol name.
        by_name = {c.symbol_name: c for c in chunks if c.symbol_name}

        # Class chunk.
        cls_chunk = by_name["MyService"]
        assert cls_chunk.parent_class is None
        assert cls_chunk.kind == "class"

        # Method chunk.
        method_chunk = by_name["MyService.handle"]
        assert method_chunk.parent_class == "MyService"
        assert method_chunk.kind == "method"
        assert method_chunk.content.startswith("class MyService:")

        # After the class chunk, the method chunk should have overlap.
        assert method_chunk.has_overlap is True
        assert method_chunk.overlap_lines >= 1

        # Function chunk.
        func_chunk = by_name["helper"]
        assert func_chunk.parent_class is None
        assert func_chunk.kind == "function"
        # helper comes after handle, so it should have overlap.
        assert func_chunk.has_overlap is True
