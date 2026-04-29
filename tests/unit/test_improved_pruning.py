"""Tests for semantic pruning: PrunedEvent, priority-based capping, categorized summaries.

Includes Phase 5 tests for design decisions, error-fix sequences, and variable window.
"""

from maike.constants import MAX_PRUNED_EVENTS
from maike.memory.working import PrunedEvent, WorkingMemory, _CATEGORY_PRIORITY


def test_extract_events_from_text_message():
    memory = WorkingMemory()
    events = memory._extract_events([
        {"role": "assistant", "content": "I will fix the bug."},
    ])
    assert len(events) == 1
    assert events[0].category == "decision"
    assert "fix the bug" in events[0].content


def test_extract_events_from_tool_use():
    memory = WorkingMemory()
    events = memory._extract_events([
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "write_file", "input": {"path": "app.py"}}],
        },
    ])
    assert len(events) == 1
    assert events[0].category == "file_write"


def test_extract_events_from_tool_result_error():
    memory = WorkingMemory()
    events = memory._extract_events([
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_name": "execute_bash",
                    "content": "SyntaxError: invalid syntax",
                    "is_error": True,
                }
            ],
        },
    ])
    assert len(events) == 1
    assert events[0].category == "error"
    assert events[0].priority == 0  # Highest priority
    assert "ERROR" in events[0].content


def test_extract_events_classifies_tool_categories():
    memory = WorkingMemory()
    assert memory._tool_category("write_file") == "file_write"
    assert memory._tool_category("edit_file") == "file_write"
    assert memory._tool_category("read_file") == "file_read"
    assert memory._tool_category("repo_map") == "file_read"
    assert memory._tool_category("execute_bash") == "command"
    assert memory._tool_category("run_tests") == "command"
    assert memory._tool_category("syntax_check") == "command"
    assert memory._tool_category("request_specialist") == "decision"


def test_prioritize_events_keeps_all_when_under_cap():
    memory = WorkingMemory()
    events = [
        PrunedEvent(category="command", content="cmd", priority=4),
        PrunedEvent(category="error", content="err", priority=0),
    ]
    result = memory._prioritize_events(events)
    assert len(result) == 2


def test_prioritize_events_caps_at_max_keeping_errors():
    memory = WorkingMemory()
    errors = [PrunedEvent(category="error", content=f"err{i}", priority=0) for i in range(5)]
    commands = [PrunedEvent(category="command", content=f"cmd{i}", priority=4) for i in range(100)]
    events = errors + commands

    result = memory._prioritize_events(events)
    assert len(result) == MAX_PRUNED_EVENTS
    # All errors must be retained
    error_contents = {e.content for e in result if e.category == "error"}
    assert len(error_contents) == 5


def test_prioritize_events_preserves_original_order():
    memory = WorkingMemory()
    events = [
        PrunedEvent(category="command", content="cmd_first", priority=4),
        PrunedEvent(category="error", content="err", priority=0),
        PrunedEvent(category="command", content="cmd_last", priority=4),
    ]
    result = memory._prioritize_events(events)
    # Should maintain original order
    contents = [e.content for e in result]
    assert contents.index("cmd_first") < contents.index("err")
    assert contents.index("err") < contents.index("cmd_last")


def test_format_event_summary_groups_by_category():
    memory = WorkingMemory()
    events = [
        PrunedEvent(category="error", content="tool bash ERROR: fail", priority=0),
        PrunedEvent(category="file_write", content="assistant tool: write_file", priority=1),
        PrunedEvent(category="file_read", content="assistant tool: read_file", priority=3),
        PrunedEvent(category="command", content="assistant tool: execute_bash", priority=4),
    ]
    output = memory._format_event_summary(events)
    # Category headers should appear
    assert "[Error]" in output
    assert "[File Write]" in output
    assert "[File Read]" in output
    assert "[Command]" in output
    # Errors should come first
    assert output.index("[Error]") < output.index("[File Write]")
    assert output.index("[File Write]") < output.index("[File Read]")


def test_format_event_summary_empty_returns_placeholder():
    memory = WorkingMemory()
    output = memory._format_event_summary([])
    assert "No significant events" in output


def test_tool_result_truncated_at_200_chars():
    memory = WorkingMemory()
    long_content = "x" * 300
    events = memory._extract_events([
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_name": "read_file",
                    "content": long_content,
                    "is_error": False,
                }
            ],
        },
    ])
    assert len(events) == 1
    # Content should be truncated (200 chars + "...")
    assert len(events[0].content) < 300


def test_full_summarize_produces_categorized_output():
    memory = WorkingMemory()
    messages = [
        {"role": "assistant", "content": "I will implement the feature."},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "write_file", "input": {"path": "app.py"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_name": "write_file", "content": "ok", "is_error": False}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_name": "execute_bash",
                    "content": "Error: compilation failed",
                    "is_error": True,
                }
            ],
        },
    ]
    summary, _note = memory._summarize(messages)
    assert "[PRUNED CONTEXT" in summary["content"]
    assert "[Error]" in summary["content"]
    assert "[File Write]" in summary["content"]


# ---------------------------------------------------------------------------
# Phase 5: Design decision detection
# ---------------------------------------------------------------------------

class TestDesignDecisionDetection:
    def setup_method(self):
        self.wm = WorkingMemory()

    def test_assistant_with_decision_keyword_gets_design_decision(self):
        events = self.wm._classify_text_event(
            "assistant",
            "I decided to use the strategy pattern because it separates concerns.",
        )
        assert len(events) == 1
        assert events[0].category == "design_decision"
        assert events[0].priority == 0

    def test_assistant_with_approach_keyword(self):
        events = self.wm._classify_text_event(
            "assistant",
            "The approach is to split the module into two files.",
        )
        assert events[0].category == "design_decision"

    def test_assistant_with_trade_off_keyword(self):
        events = self.wm._classify_text_event(
            "assistant",
            "There is a trade-off between simplicity and extensibility.",
        )
        assert events[0].category == "design_decision"

    def test_assistant_with_because_keyword(self):
        events = self.wm._classify_text_event(
            "assistant",
            "I used asyncio because the codebase is fully async.",
        )
        assert events[0].category == "design_decision"

    def test_assistant_without_keywords_stays_decision(self):
        events = self.wm._classify_text_event(
            "assistant",
            "I will fix the bug now.",
        )
        assert events[0].category == "decision"
        assert events[0].priority == 2

    def test_user_text_stays_decision_even_with_keywords(self):
        events = self.wm._classify_text_event(
            "user",
            "I decided to use the strategy pattern.",
        )
        assert events[0].category == "decision"

    def test_case_insensitive_matching(self):
        events = self.wm._classify_text_event("assistant", "BECAUSE of performance.")
        assert events[0].category == "design_decision"

    def test_design_decisions_survive_prioritization(self):
        events = [
            PrunedEvent(category="design_decision", content="Important", priority=0),
            *[PrunedEvent(category="command", content=f"cmd{i}", priority=4) for i in range(100)],
        ]
        result = self.wm._prioritize_events(events)
        assert any(e.category == "design_decision" for e in result)


# ---------------------------------------------------------------------------
# Phase 5: Error-fix sequence detection
# ---------------------------------------------------------------------------

class TestErrorFixSequenceDetection:
    def setup_method(self):
        self.wm = WorkingMemory()

    def _error_msg(self, text="SyntaxError"):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_name": "run_tests", "content": text, "is_error": True}],
        }

    def _reasoning_msg(self, text="I see the issue, let me fix it."):
        return {"role": "assistant", "content": text}

    def _fix_msg(self, tool="edit_file"):
        return {"role": "assistant", "content": [{"type": "tool_use", "name": tool, "input": {}}]}

    def test_detects_triplet(self):
        messages = [self._error_msg(), self._reasoning_msg(), self._fix_msg()]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert any(e.category == "error_fix_sequence" for e in result)

    def test_replaces_individual_events(self):
        messages = [self._error_msg(), self._reasoning_msg(), self._fix_msg()]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert len(result) < len(events)

    def test_no_match_when_no_error(self):
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "tool_name": "run_tests", "content": "ok", "is_error": False}]},
            self._reasoning_msg(),
            self._fix_msg(),
        ]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert not any(e.category == "error_fix_sequence" for e in result)

    def test_no_match_when_no_write_tool(self):
        messages = [
            self._error_msg(),
            self._reasoning_msg(),
            {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file", "input": {}}]},
        ]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert not any(e.category == "error_fix_sequence" for e in result)

    def test_fewer_than_3_messages(self):
        messages = [self._error_msg(), self._reasoning_msg()]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert result == events

    def test_multiple_sequences(self):
        messages = [
            self._error_msg("Error 1"),
            self._reasoning_msg("Fix 1"),
            self._fix_msg("write_file"),
            self._error_msg("Error 2"),
            self._reasoning_msg("Fix 2"),
            self._fix_msg("edit_file"),
        ]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert sum(1 for e in result if e.category == "error_fix_sequence") == 2

    def test_write_file_matches(self):
        messages = [self._error_msg(), self._reasoning_msg(), self._fix_msg("write_file")]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert any(e.category == "error_fix_sequence" for e in result)

    def test_delete_file_matches(self):
        messages = [self._error_msg(), self._reasoning_msg(), self._fix_msg("delete_file")]
        events = self.wm._extract_events(messages)
        result = self.wm._detect_error_fix_sequences(messages, events)
        assert any(e.category == "error_fix_sequence" for e in result)


# ---------------------------------------------------------------------------
# Phase 5: Variable recent window
# ---------------------------------------------------------------------------

class TestContentAwareRecentWindow:
    """Test the content-aware recent window that expands backward until
    both a token minimum (10K) and text-block minimum (5) are met."""

    def setup_method(self):
        self.wm = WorkingMemory()

    def test_empty_messages(self):
        assert self.wm._effective_recent_window([]) == 0

    def test_small_conversation_keeps_all(self):
        """With few short messages, all are kept (both minimums unsatisfied)."""
        msgs = [{"role": "assistant", "content": f"msg {i}"} for i in range(3)]
        assert self.wm._effective_recent_window(msgs) == 3

    def test_expands_until_token_minimum(self):
        """Short text messages: needs many to reach 10K token minimum."""
        # Each ~10 chars ≈ 3 tokens. Need ~3333 messages to hit 10K.
        # But with 5+ text blocks, the text-block minimum is met early,
        # so it keeps expanding until 10K tokens.
        msgs = [{"role": "assistant", "content": f"short msg {i}"} for i in range(20)]
        window = self.wm._effective_recent_window(msgs)
        # All 20 kept — total tokens well under 10K
        assert window == 20

    def test_large_messages_hit_cap(self):
        """Large messages hit the 40K token cap quickly."""
        big = "x" * 20_000  # ~5000 tokens per message
        msgs = [{"role": "assistant", "content": big} for _ in range(20)]
        window = self.wm._effective_recent_window(msgs)
        # ~5K tokens/msg, cap at 40K → ~8 messages
        assert window < 20

    def test_tool_results_not_counted_as_text_blocks(self):
        """Tool results (no text content) don't satisfy the text-block minimum."""
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": f"result {i}"}]}
            for i in range(10)
        ]
        window = self.wm._effective_recent_window(msgs)
        # No text blocks, so it expands until token cap or all messages
        assert window == 10

    def test_mixed_content_satisfies_both(self):
        """Mix of text + tool results: stops when both minimums met."""
        msgs = []
        # 5 large text messages (~12K tokens total)
        for i in range(5):
            msgs.append({"role": "assistant", "content": "x" * 10_000})
        # 10 tool results
        for i in range(10):
            msgs.append({"role": "user", "content": [{"type": "tool_result", "content": "ok"}]})
        window = self.wm._effective_recent_window(msgs)
        # Expands from end: 10 tool results (0 text blocks) + 5 text msgs
        # Text blocks = 5 (met), tokens ≈ 12K (met) → stops
        assert window == 15


# ---------------------------------------------------------------------------
# Phase 5: New category formatting
# ---------------------------------------------------------------------------

class TestNewCategoryFormatting:
    def setup_method(self):
        self.wm = WorkingMemory()

    def test_design_decision_section_appears(self):
        events = [PrunedEvent(category="design_decision", content="important", priority=0)]
        result = self.wm._format_event_summary(events)
        assert "[Design Decision]" in result

    def test_error_fix_sequence_section_appears(self):
        events = [PrunedEvent(category="error_fix_sequence", content="fix chain", priority=0)]
        result = self.wm._format_event_summary(events)
        assert "[Error Fix Sequence]" in result

    def test_category_ordering(self):
        events = [
            PrunedEvent(category="command", content="cmd", priority=4),
            PrunedEvent(category="error", content="err", priority=0),
            PrunedEvent(category="design_decision", content="design", priority=0),
            PrunedEvent(category="error_fix_sequence", content="fix", priority=0),
        ]
        result = self.wm._format_event_summary(events)
        assert result.index("[Error]") < result.index("[Error Fix Sequence]")
        assert result.index("[Error Fix Sequence]") < result.index("[Design Decision]")
        assert result.index("[Design Decision]") < result.index("[Command]")


# ---------------------------------------------------------------------------
# Tool result clearing (pre-pruning)
# ---------------------------------------------------------------------------

def _make_tool_result_message(tool_name: str, content: str, is_error: bool = False) -> dict:
    """Build a user message containing a single tool_result block."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_name": tool_name,
                "content": content,
                "is_error": is_error,
            }
        ],
    }


def _make_tool_use_message(tool_name: str, input_data: dict | None = None) -> dict:
    """Build an assistant message containing a single tool_use block."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "name": tool_name,
                "input": input_data or {},
            }
        ],
    }


def _make_bash_tool_use(cmd: str) -> dict:
    """Build an assistant message with a Bash tool_use block."""
    return _make_tool_use_message("Bash", {"cmd": cmd})


def test_clear_stale_tool_results_replaces_read_output():
    mem = WorkingMemory()
    messages = [_make_tool_result_message("Read", "file contents here\nline 2\nline 3")]
    result = mem._clear_stale_tool_results(messages)
    assert len(result) == 1
    block = result[0]["content"][0]
    assert block["content"] == "[cleared \u2014 re-fetch if needed]"
    assert block["type"] == "tool_result"
    assert block["tool_name"] == "Read"


def test_clear_stale_tool_results_preserves_errors():
    mem = WorkingMemory()
    original_content = "ImportError: no module named foo"
    messages = [_make_tool_result_message("Bash", original_content, is_error=True)]
    result = mem._clear_stale_tool_results(messages)
    assert len(result) == 1
    block = result[0]["content"][0]
    assert block["content"] == original_content


def test_clear_stale_tool_results_preserves_write_results():
    mem = WorkingMemory()
    original_content = "File written successfully"
    for tool in ("Write", "Edit"):
        messages = [_make_tool_result_message(tool, original_content)]
        result = mem._clear_stale_tool_results(messages)
        block = result[0]["content"][0]
        assert block["content"] == original_content, f"{tool} result should not be cleared"


def test_clear_stale_tool_results_preserves_tool_use_blocks():
    mem = WorkingMemory()
    messages = [_make_tool_use_message("Read", {"path": "foo.py"})]
    result = mem._clear_stale_tool_results(messages)
    assert result == messages
    block = result[0]["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "Read"
    assert block["input"] == {"path": "foo.py"}


# ---------------------------------------------------------------------------
# Milestone note detection and formatting
# ---------------------------------------------------------------------------


def test_milestone_note_detection():
    mem = WorkingMemory()
    for prefix in [
        "## Milestone: Done with phase 1",
        "[MILESTONE] Phase 1 complete",
        "## Note: Switching approach",
        "[NOTE] Important observation",
    ]:
        events = mem._classify_text_event("assistant", prefix)
        assert len(events) == 1, f"Expected 1 event for '{prefix}'"
        assert events[0].category == "milestone_note", f"'{prefix}' should be milestone_note"


def test_milestone_note_priority_zero():
    assert _CATEGORY_PRIORITY["milestone_note"] == 0


def test_milestone_note_in_summary():
    mem = WorkingMemory()
    messages = [
        {"role": "assistant", "content": "## Milestone: Implemented BFS. Moving to DFS."},
    ]
    summary, _note = mem._summarize(messages)
    assert "Milestone Note" in summary["content"]
    assert "Implemented BFS" in summary["content"]


# ---------------------------------------------------------------------------
# Environment state extraction
# ---------------------------------------------------------------------------


def test_environment_state_extraction_venv():
    mem = WorkingMemory()
    messages = [
        _make_bash_tool_use("source .venv/bin/activate && python main.py"),
    ]
    state = mem._extract_environment_state(messages)
    assert "Venv:" in state
    assert ".venv" in state


def test_environment_state_extraction_packages():
    mem = WorkingMemory()
    messages = [
        _make_bash_tool_use("pip install pytest requests flask"),
    ]
    state = mem._extract_environment_state(messages)
    assert "Installed:" in state
    assert "pytest" in state
    assert "requests" in state
    assert "flask" in state


def test_environment_state_empty_when_no_setup():
    mem = WorkingMemory()
    messages = [
        {"role": "assistant", "content": "Looking at the code structure."},
        _make_tool_use_message("Read", {"path": "README.md"}),
    ]
    state = mem._extract_environment_state(messages)
    assert state == ""
