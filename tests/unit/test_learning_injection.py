"""Tests for Phase 7: learning injection into agent context builders."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from maike.agents.react import _inject_learning_context, _LEARNING_CHAR_CAP, build_react_context
from maike.memory.learning import (
    SessionLearning,
    SessionLearner,
    build_learning_context_blocks,
)
from maike.memory.longterm import LongTermMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeManifest:
    language: str = "python"


@dataclass
class _FakeSession:
    """Minimal stand-in for OrchestratorSession."""
    id: str = "test-session"
    task: str = "Build API"
    workspace: Path | None = None
    learner: SessionLearner | None = None
    environment_manifest: Any = None
    thread: dict | None = None

    def environment_context_block(self) -> str | None:
        return None

    def environment_metadata(self) -> dict[str, Any]:
        return {}


def _make_learner_with_content(tmp_path: Path, content: str) -> SessionLearner:
    memory = LongTermMemory(tmp_path)
    memory.add(collection="session_learnings", key="s1", content=content)
    return SessionLearner(memory)


# ---------------------------------------------------------------------------
# Tests for SessionLearning role-specific content
# ---------------------------------------------------------------------------


class TestSessionLearningRoleSpecific:
    def test_to_content_includes_role_sections(self):
        learning = SessionLearning(
            task_summary="Build API",
            outcome="success",
            pipeline_used="development",
            language="python",
            cost_usd=1.50,
            role_specific_learnings={
                "coder": ["Use type hints", "Prefer dataclasses"],
                "tester": ["Mock external APIs"],
            },
        )
        content = learning.to_content()
        assert "Learnings (coder): Use type hints; Prefer dataclasses" in content
        assert "Learnings (tester): Mock external APIs" in content

    def test_to_content_without_role_learnings(self):
        learning = SessionLearning(
            task_summary="Build API",
            outcome="success",
            pipeline_used="development",
            language="python",
            cost_usd=1.50,
        )
        content = learning.to_content()
        assert "Learnings" not in content

    def test_to_content_sorted_by_role(self):
        learning = SessionLearning(
            task_summary="t",
            outcome="success",
            pipeline_used="dev",
            language="py",
            cost_usd=0,
            role_specific_learnings={
                "tester": ["mock"],
                "coder": ["types"],
            },
        )
        content = learning.to_content()
        coder_pos = content.index("Learnings (coder)")
        tester_pos = content.index("Learnings (tester)")
        assert coder_pos < tester_pos  # sorted alphabetically


class TestRecordSessionRoleLearnings:
    def test_record_with_role_specific_learnings(self, tmp_path):
        memory = LongTermMemory(tmp_path)
        learner = SessionLearner(memory)
        learning = learner.record_session(
            session_id="s1",
            task="Build API",
            outcome="success",
            pipeline="development",
            language="python",
            role_specific_learnings={"coder": ["Use type hints"]},
        )
        assert learning.role_specific_learnings == {"coder": ["Use type hints"]}

    def test_record_without_role_specific_learnings(self, tmp_path):
        memory = LongTermMemory(tmp_path)
        learner = SessionLearner(memory)
        learning = learner.record_session(
            session_id="s1",
            task="Fix bug",
            outcome="success",
            pipeline="debugging",
        )
        assert learning.role_specific_learnings == {}


# ---------------------------------------------------------------------------
# Tests for build_learning_context_blocks (no role filtering)
# ---------------------------------------------------------------------------


class TestBuildLearningContextBlocksNoRoleFiltering:
    def test_all_learnings_included_regardless_of_role(self, tmp_path):
        content = (
            "Task: Build API\n"
            "Outcome: success\n"
            "Pipeline: development\n"
            "Language: python\n"
            "Cost: $1.50\n"
            "Learnings (coder): Use type hints\n"
            "Learnings (tester): Mock external APIs"
        )
        learner = _make_learner_with_content(tmp_path, content)
        blocks = build_learning_context_blocks(learner, "Build API", "python", role="coder")
        assert len(blocks) == 1
        assert "Use type hints" in blocks[0]
        assert "Mock external APIs" in blocks[0]

    def test_no_role_returns_all_content(self, tmp_path):
        content = (
            "Task: Build API\n"
            "Learnings (coder): Use type hints\n"
            "Learnings (tester): Mock external APIs"
        )
        learner = _make_learner_with_content(tmp_path, content)
        blocks = build_learning_context_blocks(learner, "Build API", role="")
        assert len(blocks) == 1
        assert "Use type hints" in blocks[0]
        assert "Mock external APIs" in blocks[0]

    def test_any_role_returns_all_content(self, tmp_path):
        content = (
            "Task: Build API\n"
            "Outcome: success\n"
            "Failures: timeout\n"
            "Learnings (coder): Use type hints"
        )
        learner = _make_learner_with_content(tmp_path, content)
        blocks = build_learning_context_blocks(learner, "Build API", role="react")
        assert len(blocks) == 1
        assert "Build API" in blocks[0]
        assert "Use type hints" in blocks[0]

    def test_none_learner_returns_empty(self):
        blocks = build_learning_context_blocks(None, "task")
        assert blocks == []

    def test_all_lines_included(self, tmp_path):
        content = (
            "Task: Build API\n"
            "Outcome: success\n"
            "Pipeline: dev\n"
            "Cost: $1.00\n"
            "Failures: timeout\n"
            "Strategies: retry\n"
            "Learnings (reviewer): check imports"
        )
        learner = _make_learner_with_content(tmp_path, content)
        blocks = build_learning_context_blocks(learner, "Build API", role="react")
        text = blocks[0]
        assert "Task: Build API" in text
        assert "Outcome: success" in text
        assert "Failures: timeout" in text
        assert "Strategies: retry" in text
        assert "check imports" in text


# ---------------------------------------------------------------------------
# Tests for _inject_learning_context
# ---------------------------------------------------------------------------


class TestInjectLearningContext:
    """Tests for the _inject_learning_context helper in react.py."""

    def test_insights_appear_in_context(self, tmp_path):
        content = "Task: Build API\nOutcome: success\nStrategies: Use retry logic"
        learner = _make_learner_with_content(tmp_path, content)
        session = _FakeSession(learner=learner)

        blocks: list[str] = []
        _inject_learning_context(session, "Build API with retry", blocks)

        assert len(blocks) == 1
        assert "Lessons from Previous Sessions" in blocks[0]
        assert "Use retry logic" in blocks[0]

    def test_token_cap_truncates_large_output(self, tmp_path):
        huge_content = "Task: Build API\n" + "x" * (_LEARNING_CHAR_CAP + 5000)
        learner = _make_learner_with_content(tmp_path, huge_content)
        session = _FakeSession(learner=learner)

        blocks: list[str] = []
        _inject_learning_context(session, "Build API", blocks)

        assert len(blocks) == 1
        assert len(blocks[0]) <= _LEARNING_CHAR_CAP + 50  # small margin for "(truncated)"
        assert "(truncated)" in blocks[0]

    def test_graceful_skip_when_learner_is_none(self):
        session = _FakeSession(learner=None)
        blocks: list[str] = []
        _inject_learning_context(session, "some task", blocks)
        assert blocks == []

    def test_graceful_skip_when_learner_returns_empty(self, tmp_path):
        memory = LongTermMemory(tmp_path)
        learner = SessionLearner(memory)
        session = _FakeSession(learner=learner)

        blocks: list[str] = []
        _inject_learning_context(session, "completely unrelated query xyz987", blocks)
        assert blocks == []

    def test_no_learner_attr_on_session(self):
        session = MagicMock(spec=[])
        blocks: list[str] = []
        _inject_learning_context(session, "task", blocks)
        assert blocks == []

    def test_learning_context_does_not_duplicate_hot_context(self, tmp_path):
        content = "Task: Build API\nOutcome: success\nStrategies: Prefer dataclasses"
        learner = _make_learner_with_content(tmp_path, content)
        session = _FakeSession(learner=learner)

        blocks: list[str] = ["## Hot Context\n\nSome code symbols..."]
        _inject_learning_context(session, "Build API", blocks)

        assert len(blocks) == 2
        assert blocks[0].startswith("## Hot Context")
        assert "Lessons from Previous Sessions" in blocks[1]

    def test_language_extracted_from_environment_manifest(self, tmp_path):
        content = "Task: Build python API\nOutcome: success\nLanguage: python"
        learner = _make_learner_with_content(tmp_path, content)
        manifest = _FakeManifest(language="python")
        session = _FakeSession(learner=learner, environment_manifest=manifest)

        blocks: list[str] = []
        _inject_learning_context(session, "Build API", blocks)

        assert len(blocks) == 1
        assert "python" in blocks[0].lower()


# ---------------------------------------------------------------------------
# Tests for build_react_context learning integration
# ---------------------------------------------------------------------------


class TestBuildReactContextLearningIntegration:
    """Verify build_react_context includes learning blocks end-to-end."""

    def test_learning_blocks_in_built_messages(self, tmp_path):
        content = "Task: Build REST API\nOutcome: success\nStrategies: Use FastAPI"
        learner = _make_learner_with_content(tmp_path, content)
        session = _FakeSession(
            learner=learner,
            workspace=tmp_path,
        )

        ctx, messages = asyncio.run(
            build_react_context("Build REST API endpoint", session)
        )

        combined = "\n".join(m.get("content", "") for m in messages)
        assert "Lessons from Previous Sessions" in combined
        assert "Use FastAPI" in combined

    def test_no_learning_when_learner_none(self, tmp_path):
        session = _FakeSession(learner=None, workspace=tmp_path)

        ctx, messages = asyncio.run(
            build_react_context("Build REST API", session)
        )

        combined = "\n".join(m.get("content", "") for m in messages)
        assert "Lessons from Previous Sessions" not in combined
