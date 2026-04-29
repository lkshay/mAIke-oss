"""Tests for maike.intelligence.import_graph."""

from pathlib import Path

from maike.intelligence.import_graph import ImportGraph
from maike.intelligence.models import FileEntry, ImportRef


def _entry(path: str, imports: list[ImportRef] | None = None, lang: str = "python") -> FileEntry:
    return FileEntry(
        path=path,
        language=lang,
        content_hash="abc123",
        imports=imports or [],
    )


class TestImportGraph:
    def test_forward_edges(self):
        entries = {
            "app.py": _entry("app.py", [
                ImportRef(source_file="app.py", imported_name="Foo", module_path="lib", is_from_import=True),
            ]),
            "lib.py": _entry("lib.py"),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        assert "lib.py" in graph.imports_of("app.py")

    def test_reverse_edges(self):
        entries = {
            "app.py": _entry("app.py", [
                ImportRef(source_file="app.py", imported_name="Foo", module_path="lib", is_from_import=True),
            ]),
            "lib.py": _entry("lib.py"),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        assert "app.py" in graph.importers_of("lib.py")

    def test_no_self_loops(self):
        entries = {
            "app.py": _entry("app.py", [
                ImportRef(source_file="app.py", imported_name="app", module_path="app", is_from_import=False),
            ]),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        assert "app.py" not in graph.imports_of("app.py")

    def test_stdlib_produces_no_edge(self):
        entries = {
            "app.py": _entry("app.py", [
                ImportRef(source_file="app.py", imported_name="os", module_path="os", is_from_import=False),
            ]),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        assert graph.imports_of("app.py") == set()

    def test_nested_module_resolution(self):
        entries = {
            "main.py": _entry("main.py", [
                ImportRef(source_file="main.py", imported_name="helper", module_path="utils.helper", is_from_import=True),
            ]),
            "utils/helper.py": _entry("utils/helper.py"),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        assert "utils/helper.py" in graph.imports_of("main.py")

    def test_package_init_resolution(self):
        entries = {
            "main.py": _entry("main.py", [
                ImportRef(source_file="main.py", imported_name="utils", module_path="utils", is_from_import=False),
            ]),
            "utils/__init__.py": _entry("utils/__init__.py"),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        assert "utils/__init__.py" in graph.imports_of("main.py")

    def test_related_files_depth_1(self):
        entries = {
            "a.py": _entry("a.py", [
                ImportRef(source_file="a.py", imported_name="B", module_path="b", is_from_import=True),
            ]),
            "b.py": _entry("b.py", [
                ImportRef(source_file="b.py", imported_name="C", module_path="c", is_from_import=True),
            ]),
            "c.py": _entry("c.py"),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        related = graph.related_files("b.py", depth=1)
        assert "a.py" in related  # imports b
        assert "c.py" in related  # imported by b
        assert "b.py" not in related  # excludes self

    def test_related_files_depth_2(self):
        entries = {
            "a.py": _entry("a.py", [
                ImportRef(source_file="a.py", imported_name="B", module_path="b", is_from_import=True),
            ]),
            "b.py": _entry("b.py", [
                ImportRef(source_file="b.py", imported_name="C", module_path="c", is_from_import=True),
            ]),
            "c.py": _entry("c.py"),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        related = graph.related_files("a.py", depth=2)
        assert "b.py" in related
        assert "c.py" in related  # 2 hops: a→b→c

    def test_cycle_handling(self):
        entries = {
            "a.py": _entry("a.py", [
                ImportRef(source_file="a.py", imported_name="B", module_path="b", is_from_import=True),
            ]),
            "b.py": _entry("b.py", [
                ImportRef(source_file="b.py", imported_name="A", module_path="a", is_from_import=True),
            ]),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        # Should not crash.
        related = graph.related_files("a.py", depth=3)
        assert "b.py" in related

    def test_js_relative_import_resolution(self):
        entries = {
            "src/app.js": _entry("src/app.js", [
                ImportRef(source_file="src/app.js", imported_name="utils", module_path="./utils", is_from_import=True),
            ], lang="javascript"),
            "src/utils.js": _entry("src/utils.js", lang="javascript"),
        }
        graph = ImportGraph()
        graph.build_from_entries(entries, Path("/fake"))
        assert "src/utils.js" in graph.imports_of("src/app.js")

    def test_empty_graph(self):
        graph = ImportGraph()
        graph.build_from_entries({}, Path("/fake"))
        assert graph.imports_of("nonexistent.py") == set()
        assert graph.importers_of("nonexistent.py") == set()
        assert graph.related_files("nonexistent.py") == set()
