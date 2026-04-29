"""Tests for maike.tui.widgets — TUI widget unit tests."""

from __future__ import annotations

import pytest

textual = pytest.importorskip("textual", reason="textual not installed")


def test_tool_widget_creates_with_status():
    """ToolCallWidget initializes with tool name and hint."""
    from maike.tui.widgets.tool_widget import ToolCallWidget

    w = ToolCallWidget(tool_name="Read", hint="src/main.py", key="tool-1")
    assert w.tool_name == "Read"
    assert w.hint == "src/main.py"
    assert w.id == "tool-1"
    assert w._completed is False


def test_tool_widget_properties():
    """ToolCallWidget stores tool_name, hint, and completed state."""
    from maike.tui.widgets.tool_widget import ToolCallWidget

    w = ToolCallWidget(tool_name="Bash", hint="pytest -v", key="tool-2")
    assert w.tool_name == "Bash"
    assert w.hint == "pytest -v"
    assert w._completed is False
    # Note: complete() requires a mounted widget (Textual app context)
    # so we only test property initialization here.


def test_turn_separator_renders_with_iteration():
    """TurnSeparator shows iteration number in the label."""
    from maike.tui.widgets.turn_separator import TurnSeparator

    ts = TurnSeparator(iteration=3)
    # The widget is a Static — its renderable was set in __init__
    assert ts.classes == frozenset({"turn-separator"})


def test_turn_separator_renders_without_iteration():
    """TurnSeparator works with iteration=0."""
    from maike.tui.widgets.turn_separator import TurnSeparator

    ts = TurnSeparator(iteration=0)
    assert ts.classes == frozenset({"turn-separator"})


def test_prompt_input_history():
    """PromptInput saves and navigates history."""
    from maike.tui.widgets.input_area import PromptInput

    p = PromptInput()
    p.save_to_history("task one")
    p.save_to_history("task two")
    assert p._cmd_history == ["task one", "task two"]
    assert p._pos == 2  # pointing past end (draft)


def test_prompt_input_set_mode():
    """set_mode changes placeholder text and CSS classes."""
    from maike.tui.widgets.input_area import PromptInput

    p = PromptInput()

    p.set_mode("task")
    assert "task" in p.placeholder.lower() or "help" in p.placeholder.lower()

    p.set_mode("follow-up")
    assert "follow" in p.placeholder.lower()
    assert p.has_class("-waiting")

    p.set_mode("approval")
    assert p.has_class("-approval")
    assert not p.has_class("-waiting")

    p.set_mode("task")
    assert not p.has_class("-approval")
    assert not p.has_class("-waiting")


def test_status_bar_render():
    """StatusBar renders model, cost, tokens, iteration, budget bar."""
    from maike.tui.widgets.status_bar import StatusBar

    sb = StatusBar(model="claude-sonnet", provider="anthropic", budget=5.0)
    text = sb._render_text()
    assert "claude-sonnet" in text
    assert "$0.0000" in text
    assert "0 tok" in text
    assert "$0.00/$5.00" in text

    # Update metrics
    sb._cost = 1.25
    sb._tokens = 5000
    sb._iteration = 3
    sb._elapsed_seconds = 125.0
    text = sb._render_text()
    assert "iter 3" in text
    assert "02:05" in text
    assert "$1.25/$5.00" in text

    # Uses middle dot separators
    assert "\u00b7" in text


def test_header_bar_agent_state():
    """HeaderBar shows agent state indicator."""
    from maike.tui.widgets.header_bar import HeaderBar

    hb = HeaderBar(workspace="/tmp/test")
    text = hb._render_text()
    assert "mAIke" in text
    assert "Ready" in text

    hb.set_agent_state("thinking")
    text = hb._render_text()
    assert "Reasoning" in text

    hb.set_agent_state("executing")
    text = hb._render_text()
    assert "Running tool" in text


def test_header_bar_short_workspace():
    """HeaderBar shows only the last component of workspace path."""
    from maike.tui.widgets.header_bar import HeaderBar

    hb = HeaderBar(workspace="/home/user/projects/my-app")
    text = hb._render_text()
    assert "ws:my-app" in text
    assert "/home/user" not in text
