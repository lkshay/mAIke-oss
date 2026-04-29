"""Tests for maike.intelligence.indexer — Python AST and regex indexers."""

import asyncio
import textwrap
from pathlib import Path

from maike.intelligence.indexer import CodeIndexer
from maike.intelligence.models import SymbolKind


PYTHON_SAMPLE = textwrap.dedent("""\
    \"\"\"Sample module.\"\"\"

    import os
    from pathlib import Path
    from maike.atoms.tool import ToolResult

    MAX_SIZE = 100

    def greet(name: str) -> str:
        \"\"\"Say hello.\"\"\"
        return f"Hello, {name}!"

    async def fetch_data(url: str) -> dict:
        pass

    class Calculator:
        \"\"\"A simple calculator.\"\"\"

        def add(self, a: int, b: int) -> int:
            return a + b

        async def slow_add(self, a: int, b: int) -> int:
            return a + b

        @staticmethod
        def multiply(a: int, b: int) -> int:
            return a * b
""")


class TestPythonIndexer:
    def setup_method(self):
        self.indexer = CodeIndexer(Path("/tmp/fake"))

    def test_extracts_module_level_function(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        assert entry is not None
        names = [s.name for s in entry.symbols]
        assert "greet" in names
        greet = next(s for s in entry.symbols if s.name == "greet")
        assert greet.kind == SymbolKind.FUNCTION
        assert "name: str" in greet.signature
        assert "-> str" in greet.signature
        assert greet.docstring == "Say hello."
        assert greet.line > 0

    def test_extracts_async_function(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        fetch = next(s for s in entry.symbols if s.name == "fetch_data")
        assert fetch.kind == SymbolKind.ASYNC_FUNCTION
        assert "async def" in fetch.signature

    def test_extracts_class(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        calc = next(s for s in entry.symbols if s.name == "Calculator")
        assert calc.kind == SymbolKind.CLASS
        assert "class Calculator" in calc.signature
        assert calc.docstring == "A simple calculator."

    def test_extracts_methods(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        add = next(s for s in entry.symbols if s.name == "add")
        assert add.kind == SymbolKind.METHOD
        assert add.scope == "Calculator"

    def test_extracts_async_method(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        slow = next(s for s in entry.symbols if s.name == "slow_add")
        assert slow.kind == SymbolKind.ASYNC_METHOD

    def test_extracts_decorated_method(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        mul = next(s for s in entry.symbols if s.name == "multiply")
        assert "@staticmethod" in mul.decorators

    def test_extracts_variable(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        var = next(s for s in entry.symbols if s.name == "MAX_SIZE")
        assert var.kind == SymbolKind.VARIABLE

    def test_extracts_imports(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        assert len(entry.imports) >= 3
        modules = [imp.module_path for imp in entry.imports]
        assert "os" in modules
        assert "pathlib" in modules
        assert "maike.atoms.tool" in modules

    def test_from_import_flag(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        path_imp = next(i for i in entry.imports if i.imported_name == "Path")
        assert path_imp.is_from_import is True
        os_imp = next(i for i in entry.imports if i.imported_name == "os")
        assert os_imp.is_from_import is False

    def test_qualified_names(self):
        entry = self.indexer.index_file("maike/agents/core.py", PYTHON_SAMPLE)
        greet = next(s for s in entry.symbols if s.name == "greet")
        assert greet.qualified_name == "maike.agents.core.greet"
        add = next(s for s in entry.symbols if s.name == "add")
        assert add.qualified_name == "maike.agents.core.Calculator.add"

    def test_content_hash(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        assert len(entry.content_hash) == 16

    def test_line_count(self):
        entry = self.indexer.index_file("app.py", PYTHON_SAMPLE)
        assert entry.line_count > 0

    def test_syntax_error_falls_back_to_regex(self):
        bad_python = "def foo(:\n  pass\n\ndef bar():\n  pass\n"
        entry = self.indexer.index_file("bad.py", bad_python)
        assert entry is not None
        # Should still find at least bar via regex fallback.
        names = [s.name for s in entry.symbols]
        assert "bar" in names

    def test_unsupported_extension_returns_none(self):
        entry = self.indexer.index_file("readme.md", "# Hello", suffix=".md")
        assert entry is None


class TestRegexIndexer:
    def setup_method(self):
        self.indexer = CodeIndexer(Path("/tmp/fake"))

    def test_javascript_function(self):
        source = "export function greet(name) {\n  return `Hello ${name}`;\n}\n"
        entry = self.indexer.index_file("app.js", source, suffix=".js")
        assert entry is not None
        assert entry.language == "javascript"
        names = [s.name for s in entry.symbols]
        assert "greet" in names

    def test_typescript_class(self):
        source = "export class Calculator {\n  add(a: number, b: number): number { return a + b; }\n}\n"
        entry = self.indexer.index_file("calc.ts", source, suffix=".ts")
        names = [s.name for s in entry.symbols]
        assert "Calculator" in names

    def test_go_function(self):
        source = 'package main\n\nfunc main() {\n}\n\ntype Config struct {\n}\n'
        entry = self.indexer.index_file("main.go", source, suffix=".go")
        names = [s.name for s in entry.symbols]
        assert "main" in names
        assert "Config" in names

    def test_rust_function(self):
        source = "pub fn greet(name: &str) -> String {\n}\n\npub struct Config {\n}\n"
        entry = self.indexer.index_file("lib.rs", source, suffix=".rs")
        names = [s.name for s in entry.symbols]
        assert "greet" in names
        assert "Config" in names

    def test_js_import_extraction(self):
        source = "import { foo } from './bar';\nimport React from 'react';\n\nfunction App() {}\n"
        entry = self.indexer.index_file("app.js", source, suffix=".js")
        assert len(entry.imports) >= 1
        paths = [i.module_path for i in entry.imports]
        assert "./bar" in paths


class TestBuildFullIndex:
    def test_indexes_python_files(self, tmp_path):
        (tmp_path / "app.py").write_text("def main():\n  pass\n")
        (tmp_path / "lib.py").write_text("class Foo:\n  pass\n")
        (tmp_path / "readme.md").write_text("# readme\n")

        indexer = CodeIndexer(tmp_path)
        entries = asyncio.run(indexer.build_full_index())
        assert "app.py" in entries
        assert "lib.py" in entries
        assert "readme.md" not in entries

    def test_skips_hidden_dirs(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("def hidden(): pass\n")
        (tmp_path / "app.py").write_text("def visible(): pass\n")

        indexer = CodeIndexer(tmp_path)
        entries = asyncio.run(indexer.build_full_index())
        assert "app.py" in entries
        assert any(".venv" in k for k in entries) is False
