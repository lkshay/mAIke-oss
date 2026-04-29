"""TurnSeparator — visual break between agent iterations."""

from __future__ import annotations

from rich.text import Text as RichText
from textual.widgets import Static

from maike.tui.theme import TURN_SEPARATOR_CHAR


class TurnSeparator(Static):
    """A full-width separator between agent turns in the message list.

    When *label* is provided it is used verbatim (including surrounding
    spaces) in place of the default ``" Turn {iteration} "`` formatting —
    used for user-turn separators on follow-up prompts.
    """

    def __init__(self, iteration: int = 0, label: str | None = None) -> None:
        self._iteration = iteration
        self._label_override = label
        super().__init__(classes="turn-separator")

    def on_mount(self) -> None:
        self._render_line()

    def on_resize(self, event) -> None:
        self._render_line()

    def _render_line(self) -> None:
        width = max(self.size.width - 4, 20)  # account for padding
        if self._label_override is not None:
            label = self._label_override
        else:
            label = f" Turn {self._iteration} " if self._iteration else ""
        remaining = max(width - len(label), 10)
        left = remaining // 2
        right = remaining - left
        line = TURN_SEPARATOR_CHAR * left + label + TURN_SEPARATOR_CHAR * right
        self.update(RichText.from_markup(f"[bold dim]{line}[/bold dim]"))
