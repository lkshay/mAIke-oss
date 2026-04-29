"""Slash-command palette — filters :mod:`maike.tui.commands.SLASH_COMMANDS`."""

from __future__ import annotations

from maike.tui.commands import SlashCommand, find_command, rank_slash_commands
from maike.tui.widgets.completion_palette import CompletionPalette, PaletteEntry


class SlashPalette(CompletionPalette):
    """Palette backed by the slash-command registry."""

    def update_query(self, query: str) -> None:
        """Update filtered entries.  *query* is the text after the leading ``/``."""
        commands = rank_slash_commands(query)
        entries = [
            PaletteEntry(
                id=cmd.name,
                label=f"/{cmd.name}",
                description=cmd.description,
            )
            for cmd in commands
        ]
        self.update_entries(entries)

    def selected_command(self) -> SlashCommand | None:
        """Return the currently highlighted command, or None."""
        entry = self.current()
        if entry is None:
            return None
        return find_command(entry.id)


def detect_slash_trigger(text: str, cursor_row: int, cursor_col: int) -> str | None:
    """Return the slash query (chars between ``/`` and the cursor) or None.

    Triggers at a word boundary — either the start of the line or after
    whitespace — so ``/`` works both at the start of a prompt and in the
    middle of a sentence (``hello /he|``).  Mirrors the ``@`` mention
    trigger to keep behavior consistent across both palettes.

    Rules:
    * A ``/`` must appear at a word boundary before the cursor.
    * No whitespace between that ``/`` and the cursor (still inside the
      command token, not past it into args).

    Returns None if the palette should be dismissed.
    """
    lines = text.split("\n") if text else [""]
    if cursor_row < 0 or cursor_row >= len(lines):
        return None
    line = lines[cursor_row]
    prefix = line[:cursor_col]
    slash_idx = prefix.rfind("/")
    if slash_idx < 0:
        return None
    # Word-boundary check: either the slash is at column 0, or the char
    # immediately before it is whitespace.
    if slash_idx > 0 and not prefix[slash_idx - 1].isspace():
        return None
    token = prefix[slash_idx + 1 :]
    if any(c.isspace() for c in token):
        return None
    return token


def apply_slash_completion(
    text: str,
    cursor_row: int,
    cursor_col: int,
    command: SlashCommand,
) -> tuple[str, int, int]:
    """Replace the slash token under the cursor with ``/{command.name} ``.

    Only the slash token itself is replaced — any prose to the left
    (e.g. ``hello ``) and any args to the right (e.g. ``arg1 arg2``) are
    preserved verbatim.  Returns (new_text, new_cursor_row,
    new_cursor_col) with the cursor positioned immediately after the
    trailing space of the inserted command.
    """
    lines = text.split("\n") if text else [""]
    if cursor_row < 0 or cursor_row >= len(lines):
        return text, cursor_row, cursor_col
    line = lines[cursor_row]
    prefix = line[:cursor_col]
    slash_idx = prefix.rfind("/")
    if slash_idx < 0:
        return text, cursor_row, cursor_col
    # Word-boundary check must match detect_slash_trigger.
    if slash_idx > 0 and not prefix[slash_idx - 1].isspace():
        return text, cursor_row, cursor_col

    # Slash token spans slash_idx through the first whitespace after it
    # (which may be past the cursor if the user typed the token fully
    # and then moved).  Extend forward from the cursor to the end of
    # the contiguous non-whitespace run so we replace the whole token.
    end = cursor_col
    while end < len(line) and not line[end].isspace():
        end += 1

    replacement = f"/{command.name} "
    new_line = line[:slash_idx] + replacement + line[end:].lstrip(" ")
    lines[cursor_row] = new_line
    new_text = "\n".join(lines)
    new_col = slash_idx + len(replacement)
    return new_text, cursor_row, new_col
