"""ToolCallWidget — Collapsible tool output display.

Shows tool calls as a single status line during execution, then
expands to show collapsible output on completion.  Errors auto-expand;
successes are collapsed by default.  The full (untruncated) output is
retained on the widget so the user can open it in a modal via the
``view_full_output`` action (Ctrl+O on the app).
"""

from __future__ import annotations

from rich.text import Text as RichText
from textual.widgets import Collapsible, Static

from maike.tui.theme import DOT_ACTIVE, DOT_ERROR, DOT_SUCCESS, MAIKE_DIM

# Inline preview caps — generous so the collapsible is useful on its own.
# The full output is preserved separately for the view-full-output action.
_INLINE_PREVIEW_MAX_LINES = 120
_INLINE_PREVIEW_MAX_CHARS = 12_000


class ToolCallWidget(Static):
    """A single tool call with collapsible output.

    Lifecycle:
      1. Created on TOOL_START — shows running indicator (teal dot).
      2. ``complete()`` called on TOOL_RESULT — shows success/error dot,
         duration, and mounts collapsible output.
      3. Full output accessible via ``full_output`` attribute.
    """

    DEFAULT_CSS = """
    ToolCallWidget {
        margin: 0 0 0 2;
        height: auto;
    }
    ToolCallWidget .tool-output {
        margin: 0 0 0 4;
        padding: 0 1;
        background: $surface;
        max-height: 25;
    }
    ToolCallWidget .tool-output-hint {
        margin: 0 0 0 4;
        padding: 0 1;
        color: $text-muted;
    }
    ToolCallWidget Collapsible {
        padding: 0;
        margin: 0;
    }
    ToolCallWidget CollapsibleTitle {
        padding: 0;
    }
    """

    def __init__(self, tool_name: str, hint: str, key: str) -> None:
        super().__init__(id=key, classes="tool-msg")
        self.tool_name = tool_name
        self.hint = hint
        self._completed = False
        # Full, untruncated output.  Persisted so the user can open the
        # full-output modal via Ctrl+O to see everything beyond the inline
        # preview cap.
        self.full_output: str = ""
        self.is_error: bool = False

    def on_mount(self) -> None:
        display = f"{self.tool_name}: {self.hint}" if self.hint else self.tool_name
        self.update(RichText.from_markup(f"  {DOT_ACTIVE} {display}"))

    def complete(
        self,
        success: bool,
        output: str,
        error: str | None,
        duration_ms: int,
    ) -> None:
        """Update the widget with completion status and output."""
        self._completed = True
        self.is_error = not success
        dot = DOT_SUCCESS if success else DOT_ERROR
        display = f"{self.tool_name}: {self.hint}" if self.hint else self.tool_name
        dur = f"  [dim]({duration_ms / 1000:.1f}s)[/dim]" if duration_ms else ""
        header = f"  {dot} {display}{dur}"

        # Pick the best text to show
        full_text = ""
        if not success and error:
            full_text = error
        elif output:
            full_text = output

        self.full_output = full_text

        if not full_text or not full_text.strip():
            # No output — just show the status line
            self.update(RichText.from_markup(header))
            return

        # Preview is inline-rendered; full text available via Ctrl+O modal
        preview_text, was_truncated = _truncate_for_preview(full_text)

        # Mount a Collapsible with the preview output inside
        self.update("")  # clear the running-state text
        collapsed = success  # auto-expand errors, collapse successes
        children: list = [Static(preview_text, classes="tool-output", markup=False)]
        if was_truncated:
            total_lines = full_text.count("\n") + 1
            children.append(
                Static(
                    RichText.from_markup(
                        f"[{MAIKE_DIM}]\u2026 {total_lines:,} lines total "
                        f"\u00b7 Ctrl+O to view full output[/{MAIKE_DIM}]"
                    ),
                    classes="tool-output-hint",
                )
            )
        collapsible = Collapsible(
            *children,
            title=header,
            collapsed=collapsed,
        )
        self.mount(collapsible)


def _truncate_for_preview(text: str) -> tuple[str, bool]:
    """Return a preview of *text* and whether truncation happened.

    Caps at ``_INLINE_PREVIEW_MAX_LINES`` lines and
    ``_INLINE_PREVIEW_MAX_CHARS`` characters.  The full output is always
    retained on the widget for the view-full-output modal.
    """
    truncated = False
    lines = text.split("\n")
    if len(lines) > _INLINE_PREVIEW_MAX_LINES:
        lines = lines[:_INLINE_PREVIEW_MAX_LINES]
        truncated = True
    preview = "\n".join(lines)
    if len(preview) > _INLINE_PREVIEW_MAX_CHARS:
        preview = preview[:_INLINE_PREVIEW_MAX_CHARS]
        truncated = True
    return preview, truncated
