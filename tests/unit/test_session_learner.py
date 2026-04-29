"""Tests for maike.memory.learning — session learner."""

from maike.memory.learning import SessionLearner, SessionLearning, build_learning_context_blocks
from maike.memory.longterm import LongTermMemory


def test_record_session_stores_learning(tmp_path):
    memory = LongTermMemory(tmp_path)
    learner = SessionLearner(memory)

    learning = learner.record_session(
        session_id="sess-001",
        task="Create a hello world app",
        outcome="success",
        pipeline="MINIMAL_PIPELINE",
        language="python",
        cost_usd=0.15,
        successful_strategies=["Used syntax_check after writes"],
    )

    assert learning.outcome == "success"
    assert learning.pipeline_used == "MINIMAL_PIPELINE"
    assert learning.timestamp != ""


def test_retrieve_relevant_learnings(tmp_path):
    memory = LongTermMemory(tmp_path)
    learner = SessionLearner(memory)

    learner.record_session(
        session_id="sess-001",
        task="Create a hello world Python app",
        outcome="success",
        pipeline="minimal",
        language="python",
        cost_usd=0.10,
    )
    learner.record_session(
        session_id="sess-002",
        task="Build a REST API with Flask",
        outcome="partial",
        pipeline="full",
        language="python",
        cost_usd=1.20,
        failure_reasons=["Tests timed out"],
    )

    results = learner.retrieve_relevant_learnings("Create a Python script", language="python")
    assert len(results) > 0
    # Should find at least one relevant session
    contents = " ".join(r["content"] for r in results)
    assert "python" in contents.lower()


def test_retrieve_empty_store_returns_empty(tmp_path):
    memory = LongTermMemory(tmp_path)
    learner = SessionLearner(memory)
    results = learner.retrieve_relevant_learnings("anything")
    assert results == []


def test_build_learning_context_blocks_with_learnings(tmp_path):
    memory = LongTermMemory(tmp_path)
    learner = SessionLearner(memory)

    learner.record_session(
        session_id="sess-001",
        task="Create a hello world app",
        outcome="success",
        pipeline="minimal",
        language="python",
    )

    blocks = build_learning_context_blocks(learner, "Create a Python app", "python")
    assert len(blocks) == 1
    assert "Lessons from Previous Sessions" in blocks[0]
    assert "hello world" in blocks[0]


def test_build_learning_context_blocks_none_learner():
    blocks = build_learning_context_blocks(None, "anything")
    assert blocks == []


def test_build_learning_context_blocks_no_matches(tmp_path):
    memory = LongTermMemory(tmp_path)
    learner = SessionLearner(memory)
    blocks = build_learning_context_blocks(learner, "completely unrelated query xyz123")
    assert blocks == []


def test_session_learning_to_content():
    learning = SessionLearning(
        task_summary="Fix the auth bug",
        outcome="success",
        pipeline_used="debugging",
        language="python",
        cost_usd=0.50,
        failure_reasons=["Initial fix missed edge case"],
        successful_strategies=["Used run_tests after each fix"],
        timestamp="2026-03-20T10:00:00Z",
    )
    content = learning.to_content()
    assert "Fix the auth bug" in content
    assert "success" in content
    assert "$0.50" in content
    assert "Initial fix missed edge case" in content
    assert "Used run_tests after each fix" in content
