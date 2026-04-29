"""Tests for task complexity classifier (conversation-based)."""

from maike.agents.classifier import classify_task_complexity


def _make_conversation(messages=None, n=2):
    """Build a minimal conversation list."""
    if messages is not None:
        return messages
    return [{"role": "user", "content": f"msg {i}"} for i in range(n)]


def test_default_tier_for_simple_conversation():
    conv = _make_conversation(n=3)
    result = classify_task_complexity(conv, iteration=1, failure_count=0, has_errors=False)
    assert result in ("default", "cheap")


def test_strong_on_high_failure_count():
    conv = _make_conversation(n=5)
    result = classify_task_complexity(conv, iteration=3, failure_count=3, has_errors=True)
    assert result == "strong"


def test_strong_on_architecture_keywords():
    conv = [{"role": "user", "content": "refactor the architecture of the auth system"}]
    result = classify_task_complexity(conv, iteration=1, failure_count=0, has_errors=False)
    assert result == "strong"


def test_strong_on_high_iteration_ratio():
    conv = _make_conversation(n=10)
    result = classify_task_complexity(
        conv, iteration=18, failure_count=0, has_errors=False, max_iterations=20
    )
    assert result == "strong"


def test_cheap_on_short_readonly_conversation():
    conv = [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "Read"}],
        },
    ]
    result = classify_task_complexity(conv, iteration=0, failure_count=0, has_errors=False)
    assert result == "cheap"
