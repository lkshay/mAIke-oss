"""Tests for maike.tui.commands — slash-command registry + match scoring."""

from __future__ import annotations

from maike.tui.commands import (
    SLASH_COMMANDS,
    SlashCommand,
    find_command,
    rank_slash_commands,
    score_match,
)


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_has_thirteen_commands():
    """The registry currently enumerates all 13 supported slash commands."""
    assert len(SLASH_COMMANDS) == 13


def test_registry_names_unique():
    """No duplicate command names (ranker relies on uniqueness)."""
    names = [c.name for c in SLASH_COMMANDS]
    assert len(names) == len(set(names))


def test_takes_args_reflects_usage():
    """takes_args is True iff usage is non-empty."""
    help_cmd = find_command("help")
    skill_cmd = find_command("skill")
    assert help_cmd is not None and skill_cmd is not None
    assert help_cmd.takes_args is False  # no usage
    assert skill_cmd.takes_args is True   # has usage


# ---------------------------------------------------------------------------
# score_match — scoring tiers
# ---------------------------------------------------------------------------


def test_score_empty_query_matches_everything():
    """Empty query returns 0 — used to show the full list on first `/`."""
    assert score_match("", "help") == 0
    assert score_match("", "create-agent") == 0


def test_score_exact_match_beats_prefix():
    """Exact match (1000-tier) beats prefix match (500-tier)."""
    assert score_match("help", "help") > score_match("hel", "help")


def test_score_prefix_beats_substring():
    assert score_match("cre", "create-agent") > score_match("age", "create-agent")


def test_score_substring_beats_subsequence():
    # "age" substring match vs. "cta" subsequence match
    assert score_match("age", "create-agent") > score_match("cta", "create-agent")


def test_score_no_match_returns_negative():
    assert score_match("xyz", "help") == -1


def test_score_case_insensitive():
    assert score_match("HELP", "help") == score_match("help", "help")
    assert score_match("hElP", "help") == score_match("help", "help")


def test_score_alias_match():
    """Alias exact match (tier 900) beats prefix (tier 500)."""
    # "q" is an alias for "quit"
    alias_score = score_match("q", "quit", aliases=("exit", "q"))
    prefix_score = score_match("q", "quit")  # no aliases — pure prefix
    assert alias_score > prefix_score


def test_score_shorter_name_wins_tie():
    """Tie-breaker: shorter names win (higher score)."""
    # Both are prefix matches for "c"
    assert score_match("c", "cost") > score_match("c", "create-agent")


# ---------------------------------------------------------------------------
# rank_slash_commands — end-to-end ordering
# ---------------------------------------------------------------------------


def test_rank_he_puts_help_first():
    """`/he` should surface `help` at the top."""
    ranked = rank_slash_commands("he")
    assert ranked[0].name == "help"


def test_rank_create_puts_create_agent_and_create_team_first():
    """`/create` should surface both create-* commands at the top."""
    ranked = rank_slash_commands("create")
    top_two = {c.name for c in ranked[:2]}
    assert top_two == {"create-agent", "create-team"}


def test_rank_subsequence_fuzzy_cst_matches_cost():
    """`/cst` (subsequence) should match `cost`."""
    ranked = rank_slash_commands("cst")
    assert ranked[0].name == "cost"


def test_rank_empty_query_returns_all_commands_alphabetically():
    """Empty query returns every command in alphabetical order."""
    ranked = rank_slash_commands("")
    assert len(ranked) == len(SLASH_COMMANDS)
    names = [c.name for c in ranked]
    assert names == sorted(names)


def test_rank_unknown_query_returns_empty():
    """No matches → empty list (palette will dismiss itself)."""
    ranked = rank_slash_commands("zzzzzz")
    assert ranked == []


def test_rank_q_finds_quit_via_alias():
    """`/q` should match `quit` via its alias."""
    ranked = rank_slash_commands("q")
    names = [c.name for c in ranked]
    assert "quit" in names
    # And it should be first (alias-tier beats any prefix matches at "q")
    assert names[0] == "quit"


# ---------------------------------------------------------------------------
# find_command — lookup by name or alias
# ---------------------------------------------------------------------------


def test_find_command_by_name():
    cmd = find_command("help")
    assert cmd is not None
    assert cmd.name == "help"


def test_find_command_by_alias():
    """`exit` and `q` are aliases for `quit`."""
    assert find_command("exit") is not None
    assert find_command("exit").name == "quit"
    assert find_command("q") is not None
    assert find_command("q").name == "quit"


def test_find_command_case_insensitive():
    cmd = find_command("HELP")
    assert cmd is not None
    assert cmd.name == "help"


def test_find_command_unknown_returns_none():
    assert find_command("zzz") is None


# ---------------------------------------------------------------------------
# Ranker with custom command list (regression — accepts any Sequence)
# ---------------------------------------------------------------------------


def test_ranker_accepts_custom_list():
    """rank_slash_commands should work with an ad-hoc command list (used by tests)."""
    custom = (
        SlashCommand("foo", "desc"),
        SlashCommand("bar", "desc"),
        SlashCommand("foobar", "desc"),
    )
    ranked = rank_slash_commands("foo", custom)
    # "foo" (exact) > "foobar" (prefix) > "bar" (no match — excluded)
    names = [c.name for c in ranked]
    assert names == ["foo", "foobar"]
