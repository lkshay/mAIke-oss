"""Team resolver — discovery and parsing of team definitions.

Teams are loaded from two sources with resolution priority (last wins):

    user teams  →  project teams

User teams:    ``~/.config/maike/teams/*.md``
Project teams: ``{workspace}/.maike/teams/*.md``

Team markdown files use YAML frontmatter + a ``## Members`` section::

    ---
    name: code-review-team
    description: Comprehensive code review
    process: parallel
    on_failure: continue
    ---

    ## Members

    - agent: security-reviewer
      role: Review for security vulnerabilities
      budget_weight: 2

    - role: Style reviewer
      agent_type: review
      tools: Read, Grep

    ## Synthesis

    Combine findings into a unified report.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamMember:
    """A single member of a team — either a reference to an existing agent
    or an inline role description."""

    agent: str | None = None          # Reference to existing agent name
    role: str | None = None           # Inline role (if no agent ref)
    model_tier: str = "default"
    agent_type: str = "explore"
    budget_weight: float = 1.0
    tools: list[str] | None = None    # Only for inline roles

    @property
    def display_name(self) -> str:
        return self.agent or self.role or "unnamed"


@dataclass(frozen=True)
class TeamDefinition:
    """A team of agents that coordinate on a task."""

    name: str
    description: str
    members: list[TeamMember]
    source: str                        # "user" or "project"
    process_type: str = "parallel"     # "parallel", "sequential", "mixed"
    on_failure: str = "continue"       # "continue", "abort", "retry"
    synthesis_prompt: str | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML frontmatter from markdown body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    fm: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()

    return fm, match.group(2)


def _parse_members(body: str) -> list[TeamMember]:
    """Parse the ``## Members`` section of a team definition.

    Each member is a YAML-like block starting with ``- agent:`` or ``- role:``.
    """
    # Extract the Members section.
    members_match = re.search(
        r"^##\s+Members\s*\n(.*?)(?=^##|\Z)",
        body, re.MULTILINE | re.DOTALL,
    )
    if not members_match:
        return []

    members_text = members_match.group(1)
    members: list[TeamMember] = []

    # Split on top-level list items (lines starting with "- ").
    blocks = re.split(r"^- ", members_text, flags=re.MULTILINE)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Parse key: value pairs from the block.
        fields: dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()

        if not fields:
            continue

        agent = fields.get("agent")
        role = fields.get("role")
        if not agent and not role:
            continue

        model_tier = fields.get("model", fields.get("model_tier", "default"))
        agent_type = fields.get("agent_type", "explore")
        try:
            budget_weight = float(fields.get("budget_weight", "1.0"))
        except (ValueError, TypeError):
            budget_weight = 1.0

        tools: list[str] | None = None
        if "tools" in fields:
            tools = [t.strip() for t in fields["tools"].split(",") if t.strip()]

        members.append(TeamMember(
            agent=agent,
            role=role,
            model_tier=model_tier,
            agent_type=agent_type,
            budget_weight=budget_weight,
            tools=tools,
        ))

    return members


def _parse_synthesis(body: str) -> str | None:
    """Extract the ``## Synthesis`` section content."""
    match = re.search(
        r"^##\s+Synthesis\s*\n(.*?)(?=^##|\Z)",
        body, re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    text = match.group(1).strip()
    return text if text else None


def parse_team_file(path: Path, source: str = "project") -> TeamDefinition | None:
    """Parse a team markdown file. Returns None if invalid."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read team file: %s", path)
        return None

    fm, body = _parse_frontmatter(text)

    name = fm.get("name")
    description = fm.get("description")
    if not name or not description:
        logger.warning("Team file %s missing name or description", path)
        return None

    members = _parse_members(body)
    if not members:
        logger.warning("Team file %s has no members", path)
        return None

    process_type = fm.get("process", "parallel")
    if process_type not in ("parallel", "sequential", "mixed"):
        process_type = "parallel"

    on_failure = fm.get("on_failure", "continue")
    if on_failure not in ("continue", "abort", "retry"):
        on_failure = "continue"

    synthesis_prompt = _parse_synthesis(body)

    return TeamDefinition(
        name=name,
        description=description,
        members=members,
        source=source,
        process_type=process_type,
        on_failure=on_failure,
        synthesis_prompt=synthesis_prompt,
    )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def _load_teams_from_dir(directory: Path, source: str) -> list[TeamDefinition]:
    """Load team definitions from a directory of markdown files."""
    if not directory.is_dir():
        return []

    teams: list[TeamDefinition] = []
    for md_file in sorted(directory.glob("*.md")):
        defn = parse_team_file(md_file, source=source)
        if defn is not None:
            teams.append(defn)
    return teams


class TeamResolver:
    """Resolves team names to definitions across user and project sources.

    Resolution priority (last wins for same name):
        user → project
    """

    def __init__(
        self,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
    ):
        self._user_dir = user_dir
        self._project_dir = project_dir
        self._teams: dict[str, TeamDefinition] = {}
        self._load_all()

    def _load_all(self) -> None:
        """(Re)populate the team registry from all sources."""
        self._teams.clear()

        if self._user_dir:
            for defn in _load_teams_from_dir(self._user_dir, "user"):
                self._teams[defn.name] = defn

        if self._project_dir:
            for defn in _load_teams_from_dir(self._project_dir, "project"):
                self._teams[defn.name] = defn

    def reload(self) -> None:
        """Rescan all sources and refresh the registry."""
        self._load_all()

    def resolve(self, name: str) -> TeamDefinition | None:
        """Resolve a team by name. Returns None if not found."""
        return self._teams.get(name)

    def list_available(self) -> list[TeamDefinition]:
        """Return all available team definitions."""
        seen: dict[str, TeamDefinition] = {}
        for defn in self._teams.values():
            seen[defn.name] = defn
        return sorted(seen.values(), key=lambda d: d.name)

    def list_names(self) -> list[str]:
        """Return sorted list of available team names."""
        return sorted(set(self._teams.keys()))

    def build_catalog(self, cap: int = 2000) -> str:
        """One-liner-per-team catalog for context injection."""
        lines: list[str] = []
        for defn in self.list_available():
            member_count = len(defn.members)
            lines.append(
                f"- **{defn.name}**: {defn.description} "
                f"({member_count} members, {defn.process_type})"
            )
        text = "\n".join(lines)
        return text[:cap] if len(text) > cap else text
