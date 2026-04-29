"""Tests for task-constraint extraction (markdown output)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from maike.agents import constraints as constraints_mod
from maike.agents.constraints import (
    _EMPTY_MARKDOWN,
    _is_conversational,
    _normalize_markdown,
    build_task_constraints,
    check_path_against_constraints,
    extract_task_constraints,
    merge_constraint_sources,
)


# ---------------------------------------------------------------------------
# check_path_against_constraints  (retained for MAIKE.md post-hoc check)
# ---------------------------------------------------------------------------


class TestCheckPathAgainstConstraints:
    def test_exact_match(self):
        assert check_path_against_constraints("src/foo.ts", ["src/foo.ts"]) == "src/foo.ts"

    def test_no_match(self):
        assert check_path_against_constraints("tests/foo.test.ts", ["src/foo.ts"]) is None

    def test_glob_star(self):
        assert check_path_against_constraints("src/analysis/scoring.ts", ["src/**"]) == "src/**"

    def test_glob_extension(self):
        assert check_path_against_constraints("src/foo.ts", ["*.ts"]) == "*.ts"

    def test_empty_patterns(self):
        assert check_path_against_constraints("anything.py", []) is None

    def test_relative_path_suffix_match(self):
        assert check_path_against_constraints("src/analysis/scoring.ts", ["scoring.ts"]) == "scoring.ts"

    def test_multiple_patterns_first_match_wins(self):
        result = check_path_against_constraints("src/foo.ts", ["lib/**", "src/**"])
        assert result == "src/**"

    def test_test_files_not_matched_by_src_pattern(self):
        assert check_path_against_constraints("tests/foo.test.ts", ["src/**"]) is None

    def test_fnmatch_question_mark(self):
        assert check_path_against_constraints("src/a.ts", ["src/?.ts"]) == "src/?.ts"


class TestMergeConstraintSources:
    def test_cli_takes_precedence(self):
        result = merge_constraint_sources(
            extracted=["a.py"], maike_md_patterns=["b.py"], cli_patterns=["c.py"],
        )
        assert result[0] == "c.py"

    def test_deduplication(self):
        result = merge_constraint_sources(
            extracted=["src/**"], maike_md_patterns=["src/**"], cli_patterns=["src/**"],
        )
        assert result == ["src/**"]

    def test_all_empty(self):
        assert merge_constraint_sources(extracted=[], maike_md_patterns=[], cli_patterns=[]) == []

    def test_merge_order(self):
        result = merge_constraint_sources(
            extracted=["ext.py"], maike_md_patterns=["maike.py"], cli_patterns=["cli.py"],
        )
        assert result == ["cli.py", "maike.py", "ext.py"]


# ---------------------------------------------------------------------------
# _normalize_markdown
# ---------------------------------------------------------------------------


class TestNormalizeMarkdown:
    def test_empty_returns_empty_marker(self):
        assert _normalize_markdown("") == _EMPTY_MARKDOWN
        assert _normalize_markdown("   \n ") == _EMPTY_MARKDOWN

    def test_strips_code_fence(self):
        fenced = "```markdown\n## Task rules\n- Do X\n```"
        assert "## Task rules" in _normalize_markdown(fenced)
        assert "```" not in _normalize_markdown(fenced)

    def test_strips_preamble_before_heading(self):
        preamble = (
            "Here are the rules you requested:\n\n"
            "## Task rules\n\n- Do not run tests."
        )
        out = _normalize_markdown(preamble)
        assert out.startswith("## Task rules")
        assert "Here are the rules" not in out

    def test_passes_clean_output_through(self):
        clean = "## Task rules\n\n- Do not run tests.\n- Verify via ast.parse."
        assert _normalize_markdown(clean) == clean

    def test_truncates_long_output(self):
        long = "## Task rules\n\n" + ("- A very long rule. " * 200)
        out = _normalize_markdown(long)
        assert len(out) <= 1600
        assert "(truncated)" in out


# ---------------------------------------------------------------------------
# _is_conversational
# ---------------------------------------------------------------------------


class TestIsConversational:
    def test_continuation_words(self):
        for t in ["continue", "go ahead", "yes", "proceed", "ok"]:
            assert _is_conversational(t), f"'{t}' should be conversational"

    def test_short_questions(self):
        assert _is_conversational("What is this project about?")
        assert _is_conversational("How does the advisor work?")

    def test_action_prompts_not_conversational(self):
        assert not _is_conversational("Fix the bug in widgets.py")
        assert not _is_conversational("Add tests for the User model")

    def test_question_with_action_word_not_conversational(self):
        # Contains "fix" — even though question-shaped, it's an action request.
        assert not _is_conversational("Can you fix the bug in widgets.py?")


# ---------------------------------------------------------------------------
# Extractor (uses FakeGateway)
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    content: str


class _FakeGateway:
    def __init__(self, response: str = ""):
        self.response = response or _EMPTY_MARKDOWN
        self.calls: list[dict] = []

    async def call(self, *, system, messages, model, temperature, max_tokens, tools):
        self.calls.append({"system": system, "messages": messages, "model": model})
        return _FakeResult(content=self.response)


async def _async_false(*args, **kwargs):
    return False


@pytest.fixture(autouse=True)
def _clear_ollama_cache():
    constraints_mod._OLLAMA_PROBE_CACHE.clear()
    yield
    constraints_mod._OLLAMA_PROBE_CACHE.clear()


def _run(coro):
    return asyncio.run(coro)


class TestExtractTaskConstraints:
    def test_conversational_short_circuits(self):
        gw = _FakeGateway(response="## Task rules\n- something")
        result = _run(extract_task_constraints(
            task="what is this codebase?",
            session_bg_gateway=gw, session_provider="gemini",
        ))
        assert result == _EMPTY_MARKDOWN
        assert gw.calls == []

    def test_returns_markdown_verbatim(self):
        canned = (
            "## Task rules\n\n"
            "- Do not run tests.\n"
            "- Do not install dependencies."
        )
        gw = _FakeGateway(response=canned)
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            result = _run(extract_task_constraints(
                task="Fix the bug.  Do not run tests.",
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert "## Task rules" in result
        assert "Do not run tests" in result
        assert "Do not install dependencies" in result

    def test_respects_node_vocabulary(self):
        # Simulate the model emitting Node-ecosystem markdown when given a
        # Node task.  We're NOT testing the model itself — we're testing
        # that our extractor passes through Node vocabulary without
        # Python-izing it.
        node_rules = (
            "## Task rules\n\n"
            "- Do not run `npm test` or `jest`.\n"
            "- Do not modify `package.json`.\n"
        )
        gw = _FakeGateway(response=node_rules)
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            result = _run(extract_task_constraints(
                task="Fix the TypeScript bug. Do not run npm test.",
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert "npm test" in result
        assert "package.json" in result
        # No Python vocabulary leaked in
        assert "pytest" not in result
        assert "pyproject" not in result

    def test_respects_rust_vocabulary(self):
        rust_rules = (
            "## Task rules\n\n"
            "- Do not run `cargo test`.\n"
            "- Do not modify `Cargo.toml`.\n"
        )
        gw = _FakeGateway(response=rust_rules)
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            result = _run(extract_task_constraints(
                task="Refactor the Rust parser. Do not run cargo test.",
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert "cargo test" in result
        assert "Cargo.toml" in result

    def test_strips_preamble(self):
        gw = _FakeGateway(
            response=(
                "Sure, here are the rules:\n\n## Task rules\n\n- No tests."
            ),
        )
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            result = _run(extract_task_constraints(
                task="Fix the bug.  Do not run tests.",
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert result.startswith("## Task rules")
        assert "Sure, here are the rules" not in result

    def test_empty_model_response_returns_empty_marker(self):
        gw = _FakeGateway(response="")
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            result = _run(extract_task_constraints(
                task="Fix the bug.",
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert result == _EMPTY_MARKDOWN

    def test_gateway_error_returns_empty_marker(self):
        class _BadGateway(_FakeGateway):
            async def call(self, **kwargs):
                raise RuntimeError("boom")
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            result = _run(extract_task_constraints(
                task="Fix the bug.",
                session_bg_gateway=_BadGateway(),
                session_provider="gemini",
            ))
        assert result == _EMPTY_MARKDOWN

    def test_timeout_returns_empty_marker(self):
        class _SlowGateway(_FakeGateway):
            async def call(self, **kwargs):
                await asyncio.sleep(30)
                return _FakeResult("## Task rules\n- X")

        async def _slow():
            result = await asyncio.wait_for(
                extract_task_constraints(
                    task="Fix the bug.",
                    session_bg_gateway=_SlowGateway(),
                    session_provider="gemini",
                ),
                timeout=16.0,
            )
            return result

        # Patch the extractor's internal timeout to 0.1s so the test is fast.
        original = extract_task_constraints.__globals__["asyncio"].wait_for
        async def _short_wait_for(coro, timeout):
            return await original(coro, timeout=0.1)
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false), \
             patch.object(constraints_mod.asyncio, "wait_for", new=_short_wait_for):
            result = _run(extract_task_constraints(
                task="Fix the bug.",
                session_bg_gateway=_SlowGateway(),
                session_provider="gemini",
            ))
        assert result == _EMPTY_MARKDOWN


# ---------------------------------------------------------------------------
# build_task_constraints — full path including MAIKE.md parsing
# ---------------------------------------------------------------------------


class TestBuildTaskConstraints:
    def test_maike_md_plus_extractor_both_present(self, tmp_path):
        (tmp_path / "MAIKE.md").write_text(
            "# Project\n\n## Protected Files\n- src/sacred.py\n- **/vendor/**\n"
        )
        gw = _FakeGateway(response="## Task rules\n\n- Do not run tests.")
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            md, patterns = _run(build_task_constraints(
                task="Fix the bug.  Do not run tests.",
                workspace=tmp_path,
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert patterns == ["src/sacred.py", "**/vendor/**"]
        assert "Protected files (from MAIKE.md)" in md
        assert "src/sacred.py" in md
        assert "## Task rules" in md
        assert "Do not run tests" in md

    def test_only_maike_md_present(self, tmp_path):
        (tmp_path / "MAIKE.md").write_text(
            "## Protected Files\n- src/sacred.py\n"
        )
        gw = _FakeGateway(response=_EMPTY_MARKDOWN)
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            md, patterns = _run(build_task_constraints(
                task="Add a new feature.",
                workspace=tmp_path,
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert patterns == ["src/sacred.py"]
        assert "Protected files" in md

    def test_no_maike_md(self, tmp_path):
        gw = _FakeGateway(response="## Task rules\n\n- Do X.")
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            md, patterns = _run(build_task_constraints(
                task="Do X.",
                workspace=tmp_path,
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert patterns == []
        assert md.startswith("## Task rules")

    def test_extraction_failure_returns_empty_marker(self, tmp_path):
        class _BadGw:
            async def call(self, **kwargs):
                raise RuntimeError("boom")
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            md, patterns = _run(build_task_constraints(
                task="Do X.",
                workspace=tmp_path,
                session_bg_gateway=_BadGw(),
                session_provider="gemini",
            ))
        assert patterns == []
        assert md == _EMPTY_MARKDOWN

    def test_no_workspace(self):
        gw = _FakeGateway(response="## Task rules\n- X")
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            md, patterns = _run(build_task_constraints(
                task="Do X.", workspace=None,
                session_bg_gateway=gw, session_provider="gemini",
            ))
        assert patterns == []
        assert "## Task rules" in md


# ---------------------------------------------------------------------------
# Ollama fallback (one behavioral test, the full probe lives elsewhere)
# ---------------------------------------------------------------------------


class TestOllamaFallback:
    def test_falls_back_to_bg_gateway_when_ollama_unreachable(self):
        bg = _FakeGateway(response="## Task rules\n- X")
        with patch.object(constraints_mod, "_ollama_reachable_with_model", new=_async_false):
            gw, provider, model = _run(
                constraints_mod._select_extractor_gateway(bg, "gemini")
            )
        assert gw is bg
        assert provider == "gemini"
        assert model  # non-empty cheap-model name
