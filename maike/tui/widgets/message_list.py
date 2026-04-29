"""Scrollable message list — the main content area of the TUI.

Visual patterns:
- User messages in a light-background box
- Assistant responses rendered as Rich Markdown
- Tool calls with collapsible output (ToolCallWidget)
- Thinking blocks collapsible and dimmed
- Delegate lifecycle grouped in collapsible blocks
- Turn separators between agent iterations
- Spinner with animated diamond pulse
"""

from __future__ import annotations

from rich.markup import escape as _rich_escape
from rich.text import Text as RichText
from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widgets import Collapsible, Markdown, Static

import time

from maike.tui.theme import (
    DOT,
    DOT_ERROR,
    DOT_SUCCESS,
    MAIKE_ACCENT,
    MAIKE_ERROR,
    MAIKE_WARNING,
    SPINNER_FRAMES,
    SPINNER_INTERVAL_S,
    SPINNER_STALL_ALERT_S,
    SPINNER_STALL_WARN_S,
    TOOL_ROLLUP_THRESHOLD,
)
from maike.tui.widgets.delegate_widget import DelegateWidget
from maike.tui.widgets.tool_widget import ToolCallWidget
from maike.tui.widgets.turn_separator import TurnSeparator


# ---------------------------------------------------------------------------
# Agent output cleaning — strips internal bookkeeping from TUI display
# ---------------------------------------------------------------------------

import re

_MILESTONE_RE = re.compile(r"^##\s*Milestone:.*$", re.MULTILINE)
_SCRATCHPAD_RE = re.compile(
    r"<(?:scratchpad|internal[_-]?note|progress[_-]?note)>.*?</(?:scratchpad|internal[_-]?note|progress[_-]?note)>",
    re.DOTALL | re.IGNORECASE,
)


def _clean_agent_output(text: str) -> str:
    """Strip internal bookkeeping (milestones, scratchpad) from agent output."""
    text = _MILESTONE_RE.sub("", text)
    text = _SCRATCHPAD_RE.sub("", text)
    # Collapse runs of 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class MessageList(VerticalScroll):
    """Scrollable container for the chat transcript."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tool_widgets: dict[str, ToolCallWidget] = {}
        self._delegate_widgets: dict[str, DelegateWidget] = {}
        self._tool_counter: int = 0
        self._spinner_widget: Static | None = None
        self._spinner_timer: Timer | None = None
        self._spinner_frame: int = 0
        self._spinner_verb: str = ""
        self._spinner_start_time: float = 0.0
        self._streaming_buffer: str = ""
        # Direct reference to the active streaming widget.  Tracking it by
        # attribute (instead of querying an id) avoids DuplicateIds errors
        # when we retire the widget at finalize and a new one mounts for
        # the next turn before Textual's id bookkeeping catches up.
        self._streaming_widget: Static | None = None
        # Latch set by finalize_streaming; keeps late-arriving
        # STREAM_DELTA events from re-creating a streaming widget and
        # producing a duplicate reply in the transcript.  Cleared at the
        # top of each turn via begin_streaming_turn().
        self._streaming_finalized: bool = False
        # Tool rollup tracking
        self._consecutive_tools: int = 0
        self._rollup_tools: list[ToolCallWidget] = []
        self._rollup_names: list[str] = []
        # Auto-scroll throttle — multiple rapid updates (stream deltas,
        # tool starts, spinner) all want to keep the view pinned to the
        # bottom.  Debounce them into one scroll per refresh tick so we
        # don't queue dozens of redundant scroll_end calls.
        self._scroll_scheduled: bool = False

    # ------------------------------------------------------------------
    # User messages — light background box
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        self._reset_tool_rollup()
        w = Static(RichText(text), classes="user-msg")
        self.mount(w)
        self._auto_scroll()

    # ------------------------------------------------------------------
    # Assistant messages — markdown rendered
    # ------------------------------------------------------------------

    def add_assistant_message(self, text: str) -> None:
        self._reset_tool_rollup()
        cleaned = _clean_agent_output(text)
        if not cleaned:
            return
        # Textual's Markdown widget — slower to mount than Static with a
        # Rich renderable, but it's the only form that integrates with
        # Textual's text-selection offset mapping.  Users need to be able
        # to select + copy the assistant's reply; a single jank-spike at
        # the end of a turn is an acceptable cost to preserve that.
        self.mount(Markdown(cleaned, classes="gutter-msg"))
        self._auto_scroll()

    # ------------------------------------------------------------------
    # Tool calls — ToolCallWidget with rollup after threshold
    # ------------------------------------------------------------------

    def _reset_tool_rollup(self) -> None:
        """Reset consecutive tool tracking (called on non-tool events)."""
        self._consecutive_tools = 0
        self._rollup_tools.clear()
        self._rollup_names.clear()

    def add_tool_start(self, tool_name: str, hint: str = "") -> str:
        """Add a tool-in-progress widget. Returns a key for complete_tool()."""
        self._consecutive_tools += 1
        self._tool_counter += 1
        key = f"tool-{self._tool_counter}"
        widget = ToolCallWidget(tool_name=tool_name, hint=hint, key=key)
        self._tool_widgets[key] = widget
        self.mount(widget)
        self._auto_scroll()
        return key

    def complete_tool(
        self,
        key: str,
        success: bool,
        output: str = "",
        error: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Complete a tool call — updates status, rolls up if threshold hit."""
        widget = self._tool_widgets.pop(key, None)
        if widget is None:
            return
        widget.complete(
            success=success,
            output=output,
            error=error,
            duration_ms=duration_ms,
        )

        # Track for rollup — only successful tools with output get rolled up
        if success and not error:
            self._rollup_tools.append(widget)
            self._rollup_names.append(widget.tool_name)

        # Apply rollup: collapse older tools, keep latest visible
        if len(self._rollup_tools) == TOOL_ROLLUP_THRESHOLD:
            self._apply_tool_rollup()

        self._auto_scroll()

    def _apply_tool_rollup(self) -> None:
        """Collapse accumulated tool widgets into a summary Collapsible."""
        to_collapse = self._rollup_tools[:-1]  # keep the latest one visible
        if not to_collapse:
            return

        # Deduplicate tool names for summary
        seen = []
        for n in self._rollup_names[:-1]:
            if n not in seen:
                seen.append(n)
        names_str = ", ".join(seen[:4])
        if len(seen) > 4:
            names_str += f" +{len(seen) - 4}"

        count = len(to_collapse)
        title = f"  {DOT_SUCCESS} +{count} tool call{'s' if count != 1 else ''} ({names_str})"

        # Remove individual widgets from DOM and re-mount inside a collapsible
        for w in to_collapse:
            w.remove()

        # Create summary widgets inside the collapsible
        summary_lines = []
        for w in to_collapse:
            summary_lines.append(w.tool_name + (f": {w.hint}" if w.hint else ""))
        summary_text = "\n".join(f"  {DOT} {line}" for line in summary_lines)

        collapsible = Collapsible(
            Static(summary_text, classes="tool-output"),
            title=title,
            collapsed=True,
        )
        # Mount before the latest tool widget
        latest = self._rollup_tools[-1]
        self.mount(collapsible, before=latest)

        # Reset — keep only the latest tool in tracking
        self._rollup_tools = [self._rollup_tools[-1]]
        self._rollup_names = [self._rollup_names[-1]]

    # ------------------------------------------------------------------
    # Thinking blocks — collapsible, dimmed
    # ------------------------------------------------------------------

    def add_thinking(self, text: str) -> None:
        if not text.strip():
            return
        lines = text.strip().splitlines()
        n_lines = len(lines)
        preview = _rich_escape(lines[0][:70]) if lines else ""
        count_info = f" ({n_lines} line{'s' if n_lines != 1 else ''})" if n_lines > 1 else ""
        title = f"  [#8b5cf6]{DOT} thinking{count_info}:[/#8b5cf6] [dim]{preview}[/dim]"
        collapsible = Collapsible(
            Static(text, classes="thinking-content"),
            title=title,
            collapsed=True,
        )
        self.mount(collapsible)
        self._auto_scroll()

    # ------------------------------------------------------------------
    # Turn separators
    # ------------------------------------------------------------------

    def add_turn_separator(self, iteration: int = 0) -> None:
        """Add a visual separator between agent turns."""
        self._reset_tool_rollup()
        self.mount(TurnSeparator(iteration))
        self._auto_scroll()

    def add_user_turn_separator(self, label: str = " New Prompt ") -> None:
        """Strong divider inserted when the user submits a follow-up prompt."""
        self._reset_tool_rollup()
        self.mount(TurnSeparator(label=label))
        self._auto_scroll()

    # ------------------------------------------------------------------
    # Delegates — grouped spawn → complete lifecycle
    # ------------------------------------------------------------------

    def add_delegate(self, task: str, key: str) -> str:
        """Add a DelegateWidget for a spawned delegate. Returns the key."""
        self._reset_tool_rollup()
        widget = DelegateWidget(task=task, key=key)
        self._delegate_widgets[key] = widget
        self.mount(widget)
        self._auto_scroll()
        return key

    def complete_delegate(
        self,
        key: str,
        success: bool,
        cost: float = 0.0,
        output: str = "",
    ) -> None:
        """Update a DelegateWidget with completion status and output."""
        widget = self._delegate_widgets.pop(key, None)
        if widget is None:
            return
        widget.complete(success=success, cost=cost, output=output)
        self._auto_scroll()

    # ------------------------------------------------------------------
    # Info / status / error lines
    # ------------------------------------------------------------------

    def add_info(self, text: str) -> None:
        w = Static(RichText.from_markup(f"  {text}"), classes="info-msg")
        self.mount(w)
        self._auto_scroll()

    def add_error(self, text: str) -> None:
        w = Static(
            RichText.from_markup(f"  {DOT_ERROR} {text}"),
            classes="error-msg",
        )
        self.mount(w)
        self._auto_scroll()

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear_content(self) -> None:
        """Remove all child widgets (clear the message list)."""
        for child in list(self.children):
            child.remove()
        self._tool_widgets.clear()
        self._tool_counter = 0
        self._streaming_widget = None
        self._streaming_buffer = ""

    # ------------------------------------------------------------------
    # Full-output lookup — used by the Ctrl+O action
    # ------------------------------------------------------------------

    def latest_output_widget(self) -> "ToolCallWidget | DelegateWidget | None":
        """Return the most recently completed tool or delegate widget.

        Walks the mounted children in reverse order and returns the first
        ToolCallWidget / DelegateWidget that has populated
        ``full_output``.  Used by the Ctrl+O action to open the full
        text in a modal.
        """
        for child in reversed(list(self.children)):
            if isinstance(child, (ToolCallWidget, DelegateWidget)):
                if getattr(child, "full_output", ""):
                    return child
        return None

    # ------------------------------------------------------------------
    # Spinner — animated during LLM calls, with elapsed time + stall color
    # ------------------------------------------------------------------

    def start_spinner(self, verb: str = "Thinking") -> None:
        """Show an animated spinner (diamond pulse) with elapsed time."""
        self.stop_spinner()
        self._spinner_verb = verb
        self._spinner_frame = 0
        self._spinner_start_time = time.monotonic()
        content = self._render_spinner_frame()
        self._spinner_widget = Static(content, classes="spinner-msg")
        self.mount(self._spinner_widget)
        self._spinner_timer = self.set_interval(
            SPINNER_INTERVAL_S, self._advance_spinner,
        )
        self._auto_scroll()

    def update_spinner_verb(self, verb: str) -> None:
        self._spinner_verb = verb

    def stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        if self._spinner_widget is not None:
            self._spinner_widget.remove()
            self._spinner_widget = None

    def _advance_spinner(self) -> None:
        if self._spinner_widget is None:
            return
        self._spinner_frame += 1
        self._spinner_widget.update(self._render_spinner_frame())

    def _spinner_color(self) -> str:
        """Color shifts from teal → amber → red based on elapsed time."""
        elapsed = time.monotonic() - self._spinner_start_time
        if elapsed >= SPINNER_STALL_ALERT_S:
            return MAIKE_ERROR
        if elapsed >= SPINNER_STALL_WARN_S:
            return MAIKE_WARNING
        return MAIKE_ACCENT

    def _render_spinner_frame(self) -> RichText:
        # Simple forward cycle — every frame has the same cell width so
        # the line does not jitter.
        char = SPINNER_FRAMES[self._spinner_frame % len(SPINNER_FRAMES)]
        color = self._spinner_color()
        elapsed = int(time.monotonic() - self._spinner_start_time)
        elapsed_str = f" {elapsed}s" if elapsed > 0 else ""
        return RichText.from_markup(
            f"  [{color}]{char}[/{color}] {self._spinner_verb}\u2026{elapsed_str}"
        )

    # ------------------------------------------------------------------
    # Streaming text — throttled markdown rendering
    # ------------------------------------------------------------------

    def update_streaming(self, text: str, render_markdown: bool = True) -> None:
        """Update the streaming text block.

        During streaming we always display plain RichText (no markdown
        re-parse) — it renders monospace, preserves whitespace, and avoids
        the visual flashing that a partial-markdown parse creates when
        a fence or heading opens and later closes mid-stream.  The
        finalize path swaps the contents to a full markdown render once
        the stream is complete.

        Guarded by ``_streaming_finalized``: if the stream has already
        been finalized for this turn but a late STREAM_DELTA drains
        afterward (the ~100 ms queue-drain lag can outlast the
        synchronous ``_turn_callback`` call), this becomes a no-op
        instead of spawning a second streaming widget that would then
        finalize into a duplicate Markdown block in the transcript.
        """
        _ = render_markdown  # kept for API compatibility
        if self._streaming_finalized:
            return
        self._streaming_buffer = text
        content = RichText(text)
        if self._streaming_widget is not None:
            self._streaming_widget.update(content)
        else:
            self._streaming_widget = Static(content, classes="gutter-msg streaming")
            self.mount(self._streaming_widget)
        self._auto_scroll()

    def finalize_streaming(self) -> None:
        """Replace the streaming Static with a selectable Markdown widget.

        Streaming renders into a plain-text ``Static`` for cheap
        per-delta updates, then at finalize we remove that widget and
        mount a Textual ``Markdown`` widget with the cleaned text.  The
        Markdown widget is heavier to mount (~100-150 ms) but it's the
        only form that integrates with Textual's text-selection offset
        mapping — without it, the user's mouse drag can't select the
        assistant's reply, which is the primary copy-to-clipboard path.

        Sets ``_streaming_finalized`` so any stale STREAM_DELTA events
        that drain after this call (queue-drain lag) become no-ops.
        ``begin_streaming_turn`` clears the flag before the next turn.
        """
        self._streaming_finalized = True
        widget = self._streaming_widget
        if widget is None:
            self._streaming_buffer = ""
            return
        final_text = _clean_agent_output(self._streaming_buffer)
        self._streaming_buffer = ""
        self._streaming_widget = None
        widget.remove()
        if final_text:
            self.mount(Markdown(final_text, classes="gutter-msg"))
            self._auto_scroll()

    def begin_streaming_turn(self) -> None:
        """Re-arm the streaming state for a new turn.

        Clears the finalize guard so the first STREAM_DELTA of the next
        turn is accepted.  Called from the bridge on LLM_START.
        """
        self._streaming_finalized = False

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def _auto_scroll(self) -> None:
        """Scroll to the end, debounced to one call per refresh tick."""
        if self._scroll_scheduled:
            return
        self._scroll_scheduled = True
        self.call_after_refresh(self._do_auto_scroll)

    def _do_auto_scroll(self) -> None:
        self._scroll_scheduled = False
        self.scroll_end(animate=False)
