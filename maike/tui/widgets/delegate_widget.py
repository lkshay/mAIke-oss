"""DelegateWidget — groups delegate spawn + result lifecycle.

Shows delegate activity as a single collapsible block:
  \u25b6 \u23fa Delegate: Explore the orchestrator layer...     (spawned)
  \u25bc \u23fa Delegate: Explore the orchestrator layer...     \u2713 $0.08
    [delegate output / result preview]

The full (untruncated) delegate output is retained on the widget so the
user can open it in a full-view modal via the ``view_full_output``
action (Ctrl+O on the app).
"""

from __future__ import annotations

from rich.markup import escape as _rich_escape
from rich.text import Text as RichText
from textual.widgets import Collapsible, Static

from maike.tui.theme import DOT_DELEGATE, DOT_ERROR, DOT_SUCCESS, MAIKE_DIM
from maike.tui.widgets.tool_widget import _truncate_for_preview


class DelegateWidget(Static):
    """Groups a delegate's spawn \u2192 complete lifecycle in one widget."""

    DEFAULT_CSS = """
    DelegateWidget {
        margin: 0 0 0 2;
        height: auto;
    }
    DelegateWidget .delegate-output {
        margin: 0 0 0 4;
        padding: 0 1;
        background: $surface;
        max-height: 25;
    }
    DelegateWidget .delegate-output-hint {
        margin: 0 0 0 4;
        padding: 0 1;
        color: $text-muted;
    }
    DelegateWidget Collapsible {
        padding: 0;
        margin: 0;
    }
    DelegateWidget CollapsibleTitle {
        padding: 0;
    }
    """

    def __init__(self, task: str, key: str) -> None:
        super().__init__(id=key, classes="tool-msg")
        self._task_description = task
        self._task_preview = task[:60] + "..." if len(task) > 60 else task
        self._completed = False
        self.full_output: str = ""
        self.is_error: bool = False
        self.tool_name: str = "Delegate"

    def on_mount(self) -> None:
        self.update(RichText.from_markup(
            f"  {DOT_DELEGATE} Delegate: {_rich_escape(self._task_preview)}  [dim](running)[/dim]"
        ))

    def complete(
        self,
        success: bool,
        cost: float = 0.0,
        output: str = "",
    ) -> None:
        """Update with delegate completion status and output."""
        self._completed = True
        self.is_error = not success
        self.full_output = output or ""
        dot = DOT_SUCCESS if success else DOT_ERROR
        cost_str = f"  [dim](${cost:.3f})[/dim]" if cost else ""
        header = f"  {dot} Delegate: {_rich_escape(self._task_preview)}{cost_str}"

        if not output or not output.strip():
            self.update(RichText.from_markup(header))
            return

        preview_text, was_truncated = _truncate_for_preview(output)

        self.update("")
        collapsed = success  # auto-expand failures
        children: list = [Static(preview_text, classes="delegate-output", markup=False)]
        if was_truncated:
            total_lines = output.count("\n") + 1
            children.append(
                Static(
                    RichText.from_markup(
                        f"[{MAIKE_DIM}]\u2026 {total_lines:,} lines total "
                        f"\u00b7 Ctrl+O to view full output[/{MAIKE_DIM}]"
                    ),
                    classes="delegate-output-hint",
                )
            )
        collapsible = Collapsible(
            *children,
            title=header,
            collapsed=collapsed,
        )
        self.mount(collapsible)
