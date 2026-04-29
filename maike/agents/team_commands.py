"""Shared team command logic for REPL and TUI surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from maike.agents.team_resolver import TeamResolver
from maike.constants import TEAMS_PROJECT_SUBDIR, TEAMS_USER_DIR


# ---------------------------------------------------------------------------
# Team listing
# ---------------------------------------------------------------------------

def format_team_list(resolver: TeamResolver) -> list[str]:
    """Format the full team list as display lines.

    Returns plain-text lines suitable for both REPL and TUI.
    """
    teams = resolver.list_available()
    if not teams:
        return [
            "No teams found.",
            "  Create one with /create-team <name>",
            f"  Or place a definition in {TEAMS_USER_DIR}",
        ]

    lines: list[str] = [f"Teams ({len(teams)}):"]
    for t in teams:
        member_names = ", ".join(m.display_name for m in t.members)
        lines.append(
            f"  {t.name}  ({t.source})  process={t.process_type}  "
            f"on_failure={t.on_failure}"
        )
        lines.append(f"    {t.description}")
        lines.append(f"    members: {member_names}")
    return lines


# ---------------------------------------------------------------------------
# Wizard data model
# ---------------------------------------------------------------------------

@dataclass
class TeamWizardMemberData:
    """A single member entry for the creation wizard."""

    agent: str | None = None
    role: str | None = None
    agent_type: str = "explore"
    budget_weight: float = 1.0


@dataclass
class TeamWizardData:
    """All fields needed to create a team definition."""

    name: str
    description: str = ""
    process_type: str = "parallel"
    on_failure: str = "continue"
    members: list[TeamWizardMemberData] = field(default_factory=list)
    synthesis_prompt: str = ""
    scope: str = "project"


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------

def sanitize_team_name(name: str) -> str:
    """Normalise a user-provided name to a valid team slug."""
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "my-team"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def preview_team_markdown(data: TeamWizardData) -> str:
    """Render the team definition markdown without writing to disk."""
    lines: list[str] = ["---"]
    lines.append(f"name: {data.name}")
    lines.append(f"description: {data.description}")
    lines.append(f"process: {data.process_type}")
    lines.append(f"on_failure: {data.on_failure}")
    lines.append("---")
    lines.append("")
    lines.append("## Members")
    lines.append("")

    for m in data.members:
        if m.agent:
            lines.append(f"- agent: {m.agent}")
            if m.role:
                lines.append(f"  role: {m.role}")
        elif m.role:
            lines.append(f"- role: {m.role}")
        else:
            continue

        if m.agent_type != "explore":
            lines.append(f"  agent_type: {m.agent_type}")
        if m.budget_weight != 1.0:
            lines.append(f"  budget_weight: {m.budget_weight}")
        lines.append("")

    if data.synthesis_prompt.strip():
        lines.append("## Synthesis")
        lines.append("")
        lines.append(data.synthesis_prompt.strip())
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def create_team_file(
    data: TeamWizardData,
    *,
    workspace: Path | None = None,
) -> Path:
    """Write a team definition markdown file. Returns the created path."""
    if data.scope == "project" and workspace is not None:
        directory = workspace / TEAMS_PROJECT_SUBDIR
    else:
        directory = TEAMS_USER_DIR

    directory.mkdir(parents=True, exist_ok=True)

    slug = sanitize_team_name(data.name)
    path = directory / f"{slug}.md"
    path.write_text(preview_team_markdown(data), encoding="utf-8")
    return path
