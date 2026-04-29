from maike.constants import PRUNE_THRESHOLD
from maike.memory.working import WorkingMemory


def _make_text_messages(n: int, *, prefix: str = "recent message") -> list[dict]:
    """Create n short text messages."""
    return [{"role": "assistant", "content": f"{prefix} {i}"} for i in range(n)]


def _make_heavy_messages(n: int) -> list[dict]:
    """Create n messages heavy enough that the content-aware window won't
    swallow them all.  ~2500 chars each ≈ 625 tokens.  8 messages ≈ 5K tokens.
    At 40 messages the window hits the 40K token cap.
    """
    return [
        {"role": "assistant", "content": f"Working on task {i}. " + "x" * 2500}
        for i in range(n)
    ]


def test_working_memory_prunes_histories_but_preserves_task_and_recent_tail():
    memory = WorkingMemory()
    memory._estimate_tokens = lambda messages: PRUNE_THRESHOLD + 1
    # Recent messages: 10 heavy messages (~6K tokens).
    recent_messages = _make_heavy_messages(10)
    # Middle: 40 heavy messages (~25K tokens) — enough that the window
    # can't keep everything (40K cap).
    middle_messages = _make_heavy_messages(40)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Build the app"},
        *middle_messages,
        *recent_messages,
    ]

    pruned = memory.prune(messages)

    assert pruned[0] == messages[0]
    assert pruned[1] == messages[1]
    # Middle was pruned — a summary/context-note was inserted.
    assert any("[PRUNED CONTEXT" in str(m.get("content", "")) for m in pruned)
    # Recent messages preserved (they're within the window).
    for msg in recent_messages:
        assert msg in pruned


def test_working_memory_prunes_against_agent_budget_before_global_threshold():
    memory = WorkingMemory()
    memory._estimate_tokens = lambda messages: 95_000
    recent_messages = _make_heavy_messages(10)
    middle_messages = _make_heavy_messages(40)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Build the app"},
        *middle_messages,
        *recent_messages,
    ]

    pruned = memory.prune_to_budget(messages, token_budget=100_000)

    assert pruned[0] == messages[0]
    assert pruned[1] == messages[1]
    assert any("[PRUNED CONTEXT" in str(m.get("content", "")) for m in pruned)
    for msg in recent_messages:
        assert msg in pruned


def test_working_memory_summarizes_mixed_text_and_tool_messages():
    memory = WorkingMemory()

    summary, _note = memory._summarize(
        [
            {"role": "assistant", "content": "Investigated the traceback and found the failing import."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "read_file", "input": {"path": "app.py"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_name": "read_file", "content": "print('hi')", "is_error": False}],
            },
        ]
    )

    assert "Investigated the traceback" in summary["content"]
    assert "read_file" in summary["content"]
    assert "print('hi')" in summary["content"]


def test_working_memory_summarizes_all_structured_histories_and_errors():
    memory = WorkingMemory()

    summary, _note = memory._summarize(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "execute_bash", "input": {"cmd": "pytest -q"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_name": "execute_bash",
                        "content": "tests failed with exit code 1",
                        "is_error": True,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "read_file", "input": {"path": "test_app.py"}}],
            },
        ]
    )

    assert "execute_bash" in summary["content"]
    assert "ERROR" in summary["content"]
    assert "tests failed with exit code 1" in summary["content"]
    assert "read_file" in summary["content"]


def test_working_memory_leaves_short_histories_unchanged():
    memory = WorkingMemory()
    messages = [
        {"role": "user", "content": "Build the app"},
        {"role": "assistant", "content": "I will inspect the workspace."},
    ]

    assert memory.prune(messages) == messages


def test_working_memory_estimates_tokens_from_structured_content_values():
    memory = WorkingMemory()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A" * 120},
                {"type": "tool_result", "tool_name": "read_file", "content": "B" * 80, "is_error": False},
            ],
        }
    ]

    estimate = memory.estimate_tokens(messages)
    # With tiktoken, the exact count depends on BPE encoding of the content.
    # The estimate should be positive and reasonable (not zero, not wildly off).
    assert estimate > 0
    # Repeated single chars like "AAA..." encode efficiently in tiktoken,
    # but the estimate should still be in a plausible range.
    assert 10 < estimate < 300
