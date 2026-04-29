"""Tests for slash + @-mention palette trigger detection and completion.

These tests exercise the pure-Python helpers from
:mod:`maike.tui.widgets.slash_palette` and
:mod:`maike.tui.widgets.mention_palette`.  Widget-level tests that require
Textual's :class:`~textual.pilot.Pilot` are intentionally left to manual
QA — they are flaky in headless environments and duplicate what the pure
helpers already cover.
"""

from __future__ import annotations

import pytest

from maike.tui.commands import find_command
from maike.tui.widgets.completion_palette import PaletteEntry
from maike.tui.widgets.mention_palette import (
    apply_mention_completion,
    detect_mention_trigger,
)
from maike.tui.widgets.slash_palette import (
    apply_slash_completion,
    detect_slash_trigger,
)


# ===========================================================================
# Slash trigger detection
# ===========================================================================


def test_slash_trigger_at_start_of_line():
    """Typing `/he` at col 3 of line 0 yields query `"he"`."""
    assert detect_slash_trigger("/he", 0, 3) == "he"


def test_slash_trigger_empty_after_slash():
    """Lone `/` yields empty query — palette shows all commands."""
    assert detect_slash_trigger("/", 0, 1) == ""


def test_slash_trigger_dismissed_after_space():
    """Once the user types a space, the palette should dismiss."""
    assert detect_slash_trigger("/help ", 0, 6) is None


def test_slash_trigger_dismissed_if_no_leading_slash():
    """Prose without `/` → no palette."""
    assert detect_slash_trigger("hello", 0, 5) is None


def test_slash_trigger_multiline_only_first_line_counts():
    """Command only active if `/` is at start of the CURRENT line — but the
    detector currently checks only the cursor's line, so follow-up lines work
    too.  Document that behavior here."""
    # Cursor on line 1 (second line), which starts with /foo
    text = "first line\n/foo"
    assert detect_slash_trigger(text, 1, 4) == "foo"


def test_slash_trigger_out_of_bounds_row_safe():
    """Bogus cursor row → None, not a crash."""
    assert detect_slash_trigger("/help", 5, 0) is None


def test_slash_trigger_mid_sentence_after_space():
    """`hello /he|` — slash after whitespace should trigger the palette."""
    assert detect_slash_trigger("hello /he", 0, 9) == "he"


def test_slash_trigger_mid_sentence_lone_slash():
    """`tell me about /|` — lone slash after whitespace triggers empty query."""
    assert detect_slash_trigger("tell me about /", 0, 15) == ""


def test_slash_trigger_embedded_slash_not_at_word_boundary():
    """`foo/bar|` — slash glued to a preceding word is NOT a command trigger
    (it's likely a path or URL, not a slash command).
    """
    assert detect_slash_trigger("foo/bar", 0, 7) is None


def test_slash_trigger_mid_sentence_dismissed_after_space():
    """`hello /help |` — once there's a space after the token, dismiss."""
    assert detect_slash_trigger("hello /help ", 0, 12) is None


def test_slash_palette_update_query_no_match_yields_empty_entries():
    """No-match query leaves the palette empty so the input widget can
    dismiss it instead of rendering a zero-row box.  Regression: earlier
    versions left an empty popup on screen while the user kept typing.
    """
    from maike.tui.widgets.slash_palette import SlashPalette

    palette = SlashPalette()
    palette.update_query("xyzqwerty")
    assert palette.is_empty, (
        "update_query with a no-match string must leave the palette empty"
    )
    assert palette._entries == []


# ===========================================================================
# Slash completion application
# ===========================================================================


def test_apply_slash_completion_replaces_partial_token():
    """`/he` + select(`help`) → `/help ` with cursor after trailing space."""
    help_cmd = find_command("help")
    assert help_cmd is not None
    new_text, new_row, new_col = apply_slash_completion("/he", 0, 3, help_cmd)
    assert new_text == "/help "
    assert new_row == 0
    assert new_col == len("/help ")


def test_apply_slash_completion_replaces_full_token_not_just_prefix():
    """`/xyz` + select(`help`) → `/help ` (the bogus token is fully replaced)."""
    help_cmd = find_command("help")
    assert help_cmd is not None
    new_text, _, _ = apply_slash_completion("/xyz", 0, 4, help_cmd)
    assert new_text == "/help "


def test_apply_slash_completion_preserves_existing_trailing_args():
    """If the user typed `/h world` and accepts `help`, the ` world` arg is kept."""
    help_cmd = find_command("help")
    assert help_cmd is not None
    # Cursor at end of "/h" (col 2), not in the args
    new_text, _, _ = apply_slash_completion("/h world", 0, 2, help_cmd)
    assert new_text == "/help world"


def test_apply_slash_completion_mid_sentence_preserves_prefix():
    """`hello /he|` + accept(`help`) → `hello /help ` with prefix kept intact."""
    help_cmd = find_command("help")
    assert help_cmd is not None
    new_text, _, new_col = apply_slash_completion("hello /he", 0, 9, help_cmd)
    assert new_text == "hello /help "
    # Cursor lands right after the trailing space of the inserted command.
    assert new_col == len("hello /help ")


def test_apply_slash_completion_mid_sentence_with_trailing_args():
    """`tell me /h about X` + accept(`help`) → `tell me /help about X`."""
    help_cmd = find_command("help")
    assert help_cmd is not None
    new_text, _, _ = apply_slash_completion("tell me /h about X", 0, 10, help_cmd)
    assert new_text == "tell me /help about X"


# ===========================================================================
# Mention trigger detection — now returns the query string, not a tuple.
# The palette is unified (one list across files/skills/agents/teams) so the
# detector no longer tracks a "category" axis.
# ===========================================================================


def test_mention_trigger_at_start_of_line_empty_query():
    """`@` at start → empty query (palette shows top matches from all sources)."""
    assert detect_mention_trigger("@", 0, 1) == ""


def test_mention_trigger_partial_query():
    """`@fi` → query `"fi"` (ranks across all sources)."""
    assert detect_mention_trigger("@fi", 0, 3) == "fi"


def test_mention_trigger_colon_dismisses():
    """Once the token contains `:`, we assume it's an already-inserted tag —
    the palette should NOT re-open on it."""
    assert detect_mention_trigger("@file:foo", 0, 9) is None


def test_mention_trigger_requires_word_boundary():
    """`foo@bar` is NOT an @-mention — @ must follow whitespace or start of line."""
    assert detect_mention_trigger("foo@bar", 0, 7) is None


def test_mention_trigger_after_whitespace():
    """Space before @ is OK."""
    assert detect_mention_trigger("hello @fi", 0, 9) == "fi"


def test_mention_trigger_dismissed_after_space():
    """Whitespace in the token terminates the mention."""
    assert detect_mention_trigger("@file ", 0, 6) is None


def test_mention_trigger_multiline():
    """Mentions work on any line of a multi-line buffer."""
    text = "first\n@app"
    assert detect_mention_trigger(text, 1, 4) == "app"


# ===========================================================================
# Mention completion application
#
# Labels shown in the palette are plain (no `@file:` chip) — but the inserted
# text is the typed tag `@{category}:{value}` so the downstream agent can
# disambiguate when parsing the prompt.
# ===========================================================================


def test_apply_mention_completion_inserts_typed_tag_from_payload():
    """Selecting a file entry inserts `@file:path ` even though the label was plain."""
    entry = PaletteEntry(
        id="file:maike/tui/app.py",
        label="app.py",  # plain label shown in UI
        description="maike/tui",
        payload={"category": "file", "value": "maike/tui/app.py"},
    )
    new_text, new_row, new_col = apply_mention_completion("@ap", 0, 3, entry)
    assert new_text == "@file:maike/tui/app.py "
    assert new_row == 0
    assert new_col == len("@file:maike/tui/app.py ")


def test_apply_mention_completion_skill_entry():
    """Non-file entries still use the typed-tag form — `@skill:pdf `."""
    entry = PaletteEntry(
        id="skill:pdf",
        label="pdf",
        description="skill · Extract text from PDFs",
        payload={"category": "skill", "value": "pdf"},
    )
    new_text, _, _ = apply_mention_completion("@pd", 0, 3, entry)
    assert new_text == "@skill:pdf "


def test_apply_mention_completion_preserves_surrounding_text():
    """Text on both sides of the @-token is preserved."""
    entry = PaletteEntry(
        id="file:README.md",
        label="README.md",
        payload={"category": "file", "value": "README.md"},
    )
    new_text, _, new_col = apply_mention_completion(
        "see @RE for details", 0, 7, entry
    )
    assert new_text == "see @file:README.md  for details"
    assert new_col == len("see @file:README.md ")


def test_apply_mention_completion_no_payload_is_noop():
    """Defensive: an entry missing payload keys doesn't corrupt the buffer."""
    entry = PaletteEntry(id="malformed", label="x", payload={})
    new_text, _, _ = apply_mention_completion("@fo", 0, 3, entry)
    assert new_text == "@fo"


# ===========================================================================
# Turn separator label override
# ===========================================================================


def test_turn_separator_accepts_label_override():
    """The new `label` kwarg is stored and preferred over iteration rendering."""
    pytest.importorskip("textual", reason="textual not installed")
    from maike.tui.widgets.turn_separator import TurnSeparator

    ts = TurnSeparator(label=" New Prompt ")
    assert ts._label_override == " New Prompt "
    assert ts._iteration == 0


def test_turn_separator_label_default_is_none():
    """Default behavior unchanged — no label override means iteration-based rendering."""
    pytest.importorskip("textual", reason="textual not installed")
    from maike.tui.widgets.turn_separator import TurnSeparator

    ts = TurnSeparator(iteration=3)
    assert ts._label_override is None
    assert ts._iteration == 3
