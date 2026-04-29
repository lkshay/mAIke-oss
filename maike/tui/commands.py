"""Slash-command registry and match scoring.

Single source of truth for the commands handled in ``app._handle_slash_command``.
The ``/help`` command and the command palette both iterate this registry —
avoiding the old duplicated hard-coded list.

``score_match`` / ``rank_slash_commands`` are pure functions, unit-testable,
and reused by the @-mention palette for fuzzy filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SlashCommand:
    """Metadata for one slash command."""

    name: str
    description: str
    usage: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def takes_args(self) -> bool:
        """Whether this command accepts arguments (influences post-select behavior)."""
        return bool(self.usage)


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("help", "Show help and keybindings"),
    SlashCommand("cost", "Show session cost and tokens"),
    SlashCommand("status", "Show provider, model, budget, workspace"),
    SlashCommand(
        "skill",
        "List, load, or install skills",
        usage="/skill [list | load <name> | install <source>]",
    ),
    SlashCommand("agent", "List custom agents", usage="/agent [list]"),
    SlashCommand("team", "List agent teams", usage="/team [list]"),
    SlashCommand(
        "plugin",
        "Manage plugins",
        usage="/plugin [list | install <source> | uninstall <name> | enable <name> | disable <name>]",
    ),
    SlashCommand(
        "create-agent",
        "Create a new custom agent",
        usage="/create-agent <name> [--scope user|project]",
    ),
    SlashCommand("create-team", "Create a new agent team", usage="/create-team <name>"),
    SlashCommand(
        "worktree",
        "Manage git worktrees",
        usage="/worktree [list | add <name> | remove <name>]",
    ),
    SlashCommand("new", "Start a new conversation thread"),
    SlashCommand("clear", "Clear the screen"),
    SlashCommand("quit", "Exit mAIke", aliases=("exit", "q")),
)


def score_match(query: str, name: str, aliases: Iterable[str] = ()) -> int:
    """Score how well *name* matches *query* (case-insensitive).

    Returns -1 for no match.  Higher is better.  Tie-breaker is shorter
    names (users usually want the most specific short name first).

    Scoring tiers — empty query returns 0 (match everything):
      - exact match on name:    1000 - len(name)
      - exact match on alias:    900 - len(name)
      - prefix match on name:    500 - len(name)
      - substring anywhere:      200 - len(name)
      - subsequence fuzzy:        50 - len(name)
      - no match:                 -1
    """
    if not query:
        return 0
    q = query.lower()
    n = name.lower()
    if n == q:
        return 1000 - len(n)
    for a in aliases:
        if a.lower() == q:
            return 900 - len(n)
    if n.startswith(q):
        return 500 - len(n)
    if q in n:
        return 200 - len(n)
    # Subsequence: every char in q appears in n in order.
    i = 0
    for ch in n:
        if i < len(q) and ch == q[i]:
            i += 1
    if i == len(q):
        return 50 - len(n)
    return -1


def rank_slash_commands(
    query: str,
    commands: Sequence[SlashCommand] = SLASH_COMMANDS,
) -> list[SlashCommand]:
    """Return commands sorted by match score for *query* (best first).

    Commands with no match are dropped.  Ties broken alphabetically.
    Empty query returns every command in alphabetical order.
    """
    scored: list[tuple[int, SlashCommand]] = []
    for cmd in commands:
        s = score_match(query, cmd.name, cmd.aliases)
        if s >= 0:
            scored.append((s, cmd))
    scored.sort(key=lambda sc: (-sc[0], sc[1].name))
    return [c for _, c in scored]


def find_command(name: str) -> SlashCommand | None:
    """Look up a command by name or alias.  Used by the dispatcher."""
    n = name.lower()
    for cmd in SLASH_COMMANDS:
        if cmd.name == n or n in cmd.aliases:
            return cmd
    return None
