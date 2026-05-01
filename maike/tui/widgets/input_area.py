"""Prompt input widget — multi-line TextArea with history and mode indicator.

Enter submits.  Shift+Enter inserts a newline.  Up/Down cycle history
when the cursor is on the first/last line.  Soft-wraps at the terminal
edge.  Height auto-grows from 3 to MAX_INPUT_HEIGHT lines.

When the user types ``/`` at the start of a line, a :class:`SlashPalette`
appears above the input with Tab-completion and arrow-key navigation.
Typing ``@`` at a word boundary opens a :class:`MentionPalette` with
typed-tag references (``@file:...``, ``@skill:...``, ``@agent:...``,
``@team:...``).
"""

from __future__ import annotations

import re
from pathlib import Path

from textual.message import Message
from textual.widgets import TextArea


# ANSI escape sequences that occasionally bleed through to the input when
# Textual's mouse-event parser misses one (heavy redraw during agent
# streaming, terminal/version mismatch, etc.).  Without filtering, mouse
# motion bytes pile up in the user's prompt as visible junk like
# ``^[[<35;58;39M^[[<35;66;40M…``.  We strip these on every Change event.
#
# Matches both real ESC byte (``\x1b[<…``) and the caret-bracket
# fallback rendering (``^[[<…``) some renderers produce when the byte
# isn't consumed as an escape.
_MOUSE_ESC_RE = re.compile(
    r"(?:\x1b|\^\[)\[<\d+;\d+;\d+[Mm]"
)

from maike.tui.theme import MAIKE_ACCENT, MAIKE_SECONDARY, MAIKE_WARNING
from maike.tui.widgets.mention_palette import (
    MentionPalette,
    MentionSources,
    apply_mention_completion,
    detect_mention_trigger,
)
from maike.tui.widgets.slash_palette import (
    SlashPalette,
    apply_slash_completion,
    detect_slash_trigger,
)

# Mode label text and colors
_MODE_CONFIG = {
    "task":        ("\u276f", MAIKE_ACCENT),        # ❯ teal
    "follow-up":   ("\u276f\u276f", MAIKE_SECONDARY),   # ❯❯ purple
    "approval":    ("!", MAIKE_WARNING),             # ! amber
    "text-prompt": ("?", MAIKE_SECONDARY),           # ? purple
}

# Height bounds (rows) — keep in sync with the ``PromptInput`` rule in
# app.tcss (``height: 5; max-height: 12;``).  The widget auto-grows with
# the number of lines typed, bounded by these values.
MIN_INPUT_HEIGHT = 5
MAX_INPUT_HEIGHT = 12


class PromptInput(TextArea):
    """Multi-line input with history, mode-aware styling, and auto-resize.

    - **Enter** submits the content (or applies a palette selection if one
      is active).
    - **Shift+Enter** inserts a newline.
    - **Up/Down** — when a palette is active, navigates completions;
      otherwise cycles history at the first/last line.
    - **Tab** — completes the active palette selection when open; moves
      focus otherwise (unchanged default).
    - **Escape** — dismisses an open palette.
    - Pasted multi-line text is preserved.
    - Soft-wraps at the terminal edge.
    """

    class Submitted(Message):
        """Posted when the user presses Enter to submit."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, **kwargs) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            theme="css",
            compact=True,
            **kwargs,
        )
        # Named _cmd_history to avoid colliding with TextArea.history (EditHistory).
        self._cmd_history: list[str] = []
        self._pos: int = 0
        self._draft: str = ""
        self._mode: str = "task"
        # Palette state — populated on demand by _ensure_*_palette.
        self._slash_palette: SlashPalette | None = None
        self._mention_palette: MentionPalette | None = None
        self._mention_sources: MentionSources | None = None
        # Guard to skip re-entry when we programmatically replace text.
        self._applying_completion: bool = False
        self._update_placeholder()

    # ------------------------------------------------------------------
    # Public API expected by app.py
    # ------------------------------------------------------------------

    @property
    def value(self) -> str:
        """Compatibility shim — app.py reads .value from the Submitted event."""
        return self.text

    def save_to_history(self, text: str) -> None:
        if text and (not self._cmd_history or self._cmd_history[-1] != text):
            self._cmd_history.append(text)
        self._pos = len(self._cmd_history)
        self._draft = ""

    def set_mode(self, mode: str) -> None:
        """Update placeholder and border style for current mode."""
        self._mode = mode
        self.remove_class("-approval", "-waiting")

        if mode == "follow-up":
            self.add_class("-waiting")
        elif mode == "approval":
            self.add_class("-approval")
        elif mode == "text-prompt":
            self.add_class("-waiting")

        self._update_placeholder()

    def _update_placeholder(self) -> None:
        symbol, _ = _MODE_CONFIG.get(self._mode, ("\u276f", MAIKE_ACCENT))
        placeholders = {
            "task": f"{symbol} Type a task or /help ...",
            "follow-up": f"{symbol} Follow-up (Enter to send, /quit to exit) ...",
            "approval": f"{symbol} y/yes = approve, n/no = deny ...",
            "text-prompt": f"{symbol} Type your response ...",
        }
        self.placeholder = placeholders.get(self._mode, f"{symbol} ...")

    # ------------------------------------------------------------------
    # Key handling — palette takes precedence when active
    # ------------------------------------------------------------------

    async def _on_key(self, event) -> None:
        palette = self._active_palette()
        if palette is not None:
            key = event.key
            if key == "up":
                event.stop()
                event.prevent_default()
                palette.move_highlight(-1)
                return
            if key == "down":
                event.stop()
                event.prevent_default()
                palette.move_highlight(1)
                return
            if key == "tab":
                event.stop()
                event.prevent_default()
                self._apply_palette_completion(submit=False)
                return
            if key == "enter":
                event.stop()
                event.prevent_default()
                self._apply_palette_completion(submit=True)
                return
            if key == "escape":
                event.stop()
                event.prevent_default()
                self._dismiss_palettes()
                return
            # Any other key falls through to TextArea so the user can keep
            # editing the query; on_text_area_changed re-filters.

        if event.key == "enter":
            # Submit on Enter (no modifier)
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if text:
                self._dismiss_palettes()
                self.post_message(self.Submitted(text))
            return

        if event.key == "shift+enter":
            # Insert newline on Shift+Enter — let the base class handle
            # "enter" insertion, which is its default behavior.
            event.stop()
            event.prevent_default()
            self.insert("\n")
            self._auto_resize()
            return

        if event.key == "up":
            # History navigation when cursor is on the first line
            row, _ = self.cursor_location
            if row == 0 and self.document.line_count <= 1:
                event.stop()
                event.prevent_default()
                self._go_older()
                return

        if event.key == "down":
            # History navigation when cursor is on the last line
            row, _ = self.cursor_location
            last_row = self.document.line_count - 1
            if row >= last_row and self.document.line_count <= 1:
                event.stop()
                event.prevent_default()
                self._go_newer()
                return

        # CRITICAL: call super() for all other keys — this is where TextArea
        # handles character insertion, deletion, paste, and all editing.
        await super()._on_key(event)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Auto-resize height when content changes (e.g. paste)."""
        # Defensive: strip mouse-tracking escape sequences that bled
        # through Textual's parser.  ``self.text = cleaned`` retriggers
        # this handler with the cleaned buffer; the regex won't match
        # anymore so it terminates after one pass.
        current = self.text
        cleaned = _MOUSE_ESC_RE.sub("", current)
        if cleaned != current:
            self.text = cleaned
            return
        self._auto_resize()
        if not self._applying_completion:
            self._refresh_palette()

    # ------------------------------------------------------------------
    # Auto-resize — grow/shrink between MIN and MAX height
    # ------------------------------------------------------------------

    def _auto_resize(self) -> None:
        line_count = self.document.line_count
        # +2 for border top/bottom
        target = max(MIN_INPUT_HEIGHT, min(line_count + 2, MAX_INPUT_HEIGHT))
        if self.styles.height is None or self.styles.height.value != target:
            self.styles.height = target

    # ------------------------------------------------------------------
    # History navigation
    # ------------------------------------------------------------------

    def _go_older(self) -> None:
        if not self._cmd_history:
            return
        if self._pos == len(self._cmd_history):
            self._draft = self.text
        if self._pos > 0:
            self._pos -= 1
            self._set_text(self._cmd_history[self._pos])

    def _go_newer(self) -> None:
        if not self._cmd_history:
            return
        if self._pos < len(self._cmd_history):
            self._pos += 1
            if self._pos == len(self._cmd_history):
                self._set_text(self._draft)
            else:
                self._set_text(self._cmd_history[self._pos])

    def _set_text(self, text: str) -> None:
        """Replace text and move cursor to end."""
        self.clear()
        self.insert(text)
        self._auto_resize()

    # ------------------------------------------------------------------
    # Palette lifecycle
    # ------------------------------------------------------------------

    def _active_palette(self) -> SlashPalette | MentionPalette | None:
        """Return whichever palette is currently mounted with at least one entry."""
        if self._slash_palette is not None and not self._slash_palette.is_empty:
            return self._slash_palette
        if self._mention_palette is not None and not self._mention_palette.is_empty:
            return self._mention_palette
        return None

    def _refresh_palette(self) -> None:
        """Decide which palette (if any) should be active based on current text/cursor."""
        text = self.text
        row, col = self.cursor_location

        slash_query = detect_slash_trigger(text, row, col)
        if slash_query is not None:
            self._dismiss_mention_palette()
            self._ensure_slash_palette(slash_query)
            return

        mention_query = detect_mention_trigger(text, row, col)
        if mention_query is not None:
            self._dismiss_slash_palette()
            self._ensure_mention_palette(mention_query)
            return

        self._dismiss_palettes()

    def _palette_host(self):
        """Return the Screen so palettes mount in the root layer.

        The input lives inside the ``#bottom-stack`` Vertical container,
        so ``self.parent`` is that Vertical — mounting the palette there
        traps it inside the docked stack's local layout and the
        ``layer: popup`` rule can't float it above the stack.  Mounting
        into the Screen itself puts the palette on the root layer where
        ``dock: bottom`` + ``offset`` work as intended.
        """
        try:
            return self.screen
        except Exception:
            return self.parent

    def _ensure_slash_palette(self, query: str) -> None:
        if self._slash_palette is None:
            self._slash_palette = SlashPalette()
            host = self._palette_host()
            if host is not None:
                try:
                    host.mount(self._slash_palette)
                except Exception:
                    # Mounting can fail if widget tree isn't ready; retry next change.
                    self._slash_palette = None
                    return
        self._slash_palette.update_query(query)
        # Nothing matched — drop the popup instead of showing an empty box.
        if self._slash_palette.is_empty:
            self._dismiss_slash_palette()
            return
        self._position_palette(self._slash_palette)

    def _ensure_mention_palette(self, query: str) -> None:
        if self._mention_palette is None:
            if self._mention_sources is None:
                workspace = getattr(self.app, "workspace", None) or Path.cwd()
                self._mention_sources = MentionSources(Path(workspace))
            self._mention_palette = MentionPalette(self._mention_sources)
            host = self._palette_host()
            if host is not None:
                try:
                    host.mount(self._mention_palette)
                except Exception:
                    self._mention_palette = None
                    return
        self._mention_palette.update_query(query)
        if self._mention_palette.is_empty:
            self._dismiss_mention_palette()
            return
        self._position_palette(self._mention_palette)

    def _position_palette(self, palette) -> None:
        """Lift *palette* above the input + status-bar stack.

        Per app.tcss: the bottom-stack container docks to bottom with a
        1-row margin below; it contains the input (with a 1-row bottom
        margin) and a 1-row status bar.  Below the palette, counting
        from the screen bottom upward, we have:
        stack_margin (1) + status_bar (1) + input_margin (1) +
        full input height including borders = 3 + input_full_height.

        ``self.region.height`` is the full rendered height (borders
        included); ``self.size.height`` is the inner content area
        excluding borders, which would put the palette 2 rows too low
        and visibly cover the input's top border and first line.
        """
        try:
            input_full_rows = int(self.region.height) or MIN_INPUT_HEIGHT
        except Exception:
            input_full_rows = MIN_INPUT_HEIGHT
        offset_rows = input_full_rows + 3
        try:
            palette.styles.offset = (0, -offset_rows)
        except Exception:
            pass

    def _dismiss_slash_palette(self) -> None:
        if self._slash_palette is not None:
            try:
                self._slash_palette.remove()
            except Exception:
                pass
            self._slash_palette = None

    def _dismiss_mention_palette(self) -> None:
        if self._mention_palette is not None:
            try:
                self._mention_palette.remove()
            except Exception:
                pass
            self._mention_palette = None

    def _dismiss_palettes(self) -> None:
        self._dismiss_slash_palette()
        self._dismiss_mention_palette()

    # ------------------------------------------------------------------
    # Apply completion — Tab or Enter while a palette is active
    # ------------------------------------------------------------------

    def _apply_palette_completion(self, *, submit: bool) -> None:
        text = self.text
        row, col = self.cursor_location

        if self._slash_palette is not None and not self._slash_palette.is_empty:
            cmd = self._slash_palette.selected_command()
            if cmd is None:
                return
            new_text, new_row, new_col = apply_slash_completion(text, row, col, cmd)
            self._replace_text(new_text, new_row, new_col)
            if submit and not cmd.takes_args:
                final = self.text.strip()
                self._dismiss_palettes()
                if final:
                    self.post_message(self.Submitted(final))
                return
            # Tab, or Enter on an args-taking command: dismiss, let user type args
            self._dismiss_palettes()
            return

        if self._mention_palette is not None and not self._mention_palette.is_empty:
            entry = self._mention_palette.current()
            if entry is None:
                return
            new_text, new_row, new_col = apply_mention_completion(text, row, col, entry)
            self._replace_text(new_text, new_row, new_col)
            self._dismiss_palettes()
            return

    def _replace_text(self, new_text: str, new_row: int, new_col: int) -> None:
        """Replace whole buffer and move cursor, suppressing re-entrant palette refresh."""
        self._applying_completion = True
        try:
            self.load_text(new_text)
            self.move_cursor((new_row, new_col))
        finally:
            self._applying_completion = False
        self._auto_resize()
        self._refresh_palette()
