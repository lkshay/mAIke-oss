"""Tests for maike.intelligence.hot_context — task keyword extraction and context assembly."""

import asyncio
import textwrap
from pathlib import Path

from maike.intelligence.code_index import CodeIndex
from maike.intelligence.hot_context import HotContextAssembler, HotContext


def _write_files(root: Path, files: dict[str, str]) -> None:
    for path, content in files.items():
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)


SAMPLE_FILES = {
    "auth/service.py": textwrap.dedent("""\
        from auth.models import User

        class AuthService:
            def authenticate(self, username: str, password: str) -> bool:
                return True

            def logout(self, user: User) -> None:
                pass
    """),
    "auth/models.py": textwrap.dedent("""\
        class User:
            def __init__(self, username: str):
                self.username = username
    """),
    "api/routes.py": textwrap.dedent("""\
        from auth.service import AuthService

        def login_route():
            svc = AuthService()
            svc.authenticate("user", "pass")

        def dashboard_route():
            pass
    """),
    "utils/helpers.py": textwrap.dedent("""\
        def format_date(dt):
            return str(dt)

        def validate_email(email):
            return "@" in email
    """),
}


class TestKeywordExtraction:
    def test_extracts_identifiers(self):
        assembler = HotContextAssembler.__new__(HotContextAssembler)
        keywords = assembler._extract_keywords("Fix the authenticate method in AuthService")
        assert "authenticate" in keywords
        assert "AuthService" in keywords

    def test_filters_stopwords(self):
        assembler = HotContextAssembler.__new__(HotContextAssembler)
        keywords = assembler._extract_keywords("fix the bug in the login flow")
        assert "the" not in keywords
        assert "bug" in keywords
        assert "login" in keywords
        assert "flow" in keywords

    def test_filters_short_tokens(self):
        assembler = HotContextAssembler.__new__(HotContextAssembler)
        keywords = assembler._extract_keywords("a b cd efg")
        assert "a" not in keywords
        assert "b" not in keywords
        assert "cd" not in keywords
        assert "efg" in keywords

    def test_deduplicates(self):
        assembler = HotContextAssembler.__new__(HotContextAssembler)
        keywords = assembler._extract_keywords("User user USER")
        assert keywords.count("User") == 1


class TestAssemble:
    def test_finds_relevant_symbols(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test")
        asyncio.run(index.build())
        assembler = HotContextAssembler(index)
        ctx = assembler.assemble("Fix the authenticate method", "coder")
        names = [s.name for s in ctx.relevant_symbols]
        assert "authenticate" in names

    def test_expands_related_files(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test")
        asyncio.run(index.build())
        assembler = HotContextAssembler(index)
        ctx = assembler.assemble("Fix AuthService authentication", "coder")
        # AuthService is in auth/service.py which imports auth/models.py
        # and is imported by api/routes.py
        assert len(ctx.related_files) >= 1

    def test_empty_task_returns_empty(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test")
        asyncio.run(index.build())
        assembler = HotContextAssembler(index)
        ctx = assembler.assemble("", "coder")
        assert ctx.relevant_symbols == []

    def test_unbuilt_index_returns_empty(self, tmp_path):
        index = CodeIndex(tmp_path, "test")
        assembler = HotContextAssembler(index)
        ctx = assembler.assemble("Fix authenticate", "coder")
        assert ctx.relevant_symbols == []

    def test_no_matching_symbols_returns_empty(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test")
        asyncio.run(index.build())
        assembler = HotContextAssembler(index)
        ctx = assembler.assemble("quantum_teleportation_flux_capacitor", "coder")
        assert ctx.relevant_symbols == []


class TestFormatHotContext:
    def test_formats_symbols_and_files(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test")
        asyncio.run(index.build())
        assembler = HotContextAssembler(index)
        ctx = assembler.assemble("Fix authenticate method", "coder")
        formatted = assembler.format_hot_context(ctx)
        assert "## Code Intelligence" in formatted
        assert "Relevant Symbols" in formatted
        assert "authenticate" in formatted

    def test_empty_context_returns_empty(self):
        assembler = HotContextAssembler.__new__(HotContextAssembler)
        formatted = assembler.format_hot_context(HotContext())
        assert formatted == ""

    def test_includes_cold_retrieval_guidance(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test")
        asyncio.run(index.build())
        assembler = HotContextAssembler(index)
        ctx = assembler.assemble("Fix authenticate", "coder")
        formatted = assembler.format_hot_context(ctx)
        assert "find_symbol" in formatted
        assert "grep_codebase" in formatted


class TestScoring:
    def test_exact_match_scores_highest(self, tmp_path):
        _write_files(tmp_path, SAMPLE_FILES)
        index = CodeIndex(tmp_path, "test")
        asyncio.run(index.build())
        assembler = HotContextAssembler(index)
        scored = assembler._find_and_score_symbols(["authenticate"])
        # authenticate should be an exact match with high score
        assert len(scored) >= 1
        top_sym, top_score = scored[0]
        assert top_sym.name == "authenticate"
        assert top_score >= 10.0
