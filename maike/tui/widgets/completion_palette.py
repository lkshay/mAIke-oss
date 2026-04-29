"""Base completion palette — floating OptionList above the prompt input.

Subclasses provide the source of entries and the callback that fires when
the user selects one.  The palette itself is a thin wrapper around
``OptionList`` with two essential constraints:

* ``can_focus = False`` — the input widget keeps focus, so the user's
  keystrokes still go to the TextArea.  Arrow keys, Tab, and Enter are
  forwarded from the input via :meth:`move_highlight` and :meth:`current`.
* Mount/unmount drives visibility — there is no hidden state.  The owner
  (``app.py``) mounts the palette when the trigger fires and removes it
  on dismiss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from textual.widgets import OptionList
from textual.widgets.option_list import Option


@dataclass(frozen=True)
class PaletteEntry:
    """One row in a completion palette."""

    id: str
    label: str
    description: str = ""
    payload: dict = field(default_factory=dict)


class CompletionPalette(OptionList):
    """Floating list of completions anchored above the prompt input."""

    can_focus = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: list[PaletteEntry] = []
        self.add_class("completion-palette")

    # ------------------------------------------------------------------
    # Public API (called from the input widget / app)
    # ------------------------------------------------------------------

    def update_entries(self, entries: list[PaletteEntry]) -> None:
        """Replace the visible entries.  Preserves highlight at index 0."""
        self._entries = entries
        self.clear_options()
        for e in entries:
            self.add_option(Option(self._render_row(e), id=e.id))
        if entries:
            self.highlighted = 0

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def current(self) -> PaletteEntry | None:
        """Return the entry currently highlighted, or None."""
        if not self._entries:
            return None
        idx = self.highlighted if self.highlighted is not None else 0
        if 0 <= idx < len(self._entries):
            return self._entries[idx]
        return None

    def move_highlight(self, delta: int) -> None:
        """Cycle the highlight by *delta* (±1) and keep it in view.

        ``scroll_to_highlight`` must run *after* the highlight change has
        been rendered — invoking it synchronously in the same tick
        sometimes scrolls against the stale viewport, leaving the new
        highlight row outside the visible slice (the "highlight
        disappears when navigating past the first N items" bug).
        """
        if not self._entries:
            return
        n = len(self._entries)
        cur = self.highlighted if self.highlighted is not None else 0
        self.highlighted = (cur + delta) % n
        self.call_after_refresh(self.scroll_to_highlight)

    # ------------------------------------------------------------------
    # Row rendering — subclasses can override for custom layout
    # ------------------------------------------------------------------

    def _render_row(self, entry: PaletteEntry) -> str:
        """Render one palette row as Rich-markup text."""
        if entry.description:
            return f"{entry.label}  [dim]— {entry.description}[/dim]"
        return entry.label
