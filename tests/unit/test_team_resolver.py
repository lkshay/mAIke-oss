"""Tests for team definition parsing and resolution."""

from __future__ import annotations

from pathlib import Path

from maike.agents.team_commands import format_team_list
from maike.agents.team_resolver import (
    TeamDefinition,
    TeamResolver,
    parse_team_file,
)


# ---------------------------------------------------------------------------
# parse_team_file
# ---------------------------------------------------------------------------


def _write_team(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


_MINIMAL_TEAM = """\
---
name: review-team
description: Code review pipeline
process: parallel
---

## Members

- agent: doc-auditor
  role: Check documentation accuracy

- agent: test-finder
  role: Find untested functions
"""

_FULL_TEAM = """\
---
name: full-review
description: Full review with synthesis
process: parallel
on_failure: retry
---

## Members

- agent: security-reviewer
  role: Review for security vulnerabilities
  model: strong
  budget_weight: 2

- agent: perf-reviewer
  role: Review for performance issues
  agent_type: review

- role: Style reviewer (inline)
  agent_type: review
  tools: Read, Grep

## Synthesis

Combine all findings. Prioritize security over style.
"""


def test_parse_minimal_team(tmp_path: Path):
    path = _write_team(tmp_path / "review-team.md", _MINIMAL_TEAM)
    defn = parse_team_file(path)
    assert defn is not None
    assert defn.name == "review-team"
    assert defn.description == "Code review pipeline"
    assert defn.process_type == "parallel"
    assert defn.on_failure == "continue"  # default
    assert len(defn.members) == 2
    assert defn.members[0].agent == "doc-auditor"
    assert defn.members[0].role == "Check documentation accuracy"
    assert defn.members[1].agent == "test-finder"
    assert defn.synthesis_prompt is None


def test_parse_full_team(tmp_path: Path):
    path = _write_team(tmp_path / "full-review.md", _FULL_TEAM)
    defn = parse_team_file(path)
    assert defn is not None
    assert defn.name == "full-review"
    assert defn.on_failure == "retry"
    assert len(defn.members) == 3

    # Agent-referenced member with custom model and weight.
    m0 = defn.members[0]
    assert m0.agent == "security-reviewer"
    assert m0.model_tier == "strong"
    assert m0.budget_weight == 2.0

    # Agent-referenced member with agent_type.
    m1 = defn.members[1]
    assert m1.agent == "perf-reviewer"
    assert m1.agent_type == "review"

    # Inline role with tools.
    m2 = defn.members[2]
    assert m2.agent is None
    assert m2.role is not None
    assert "Style reviewer" in m2.role
    assert m2.tools == ["Read", "Grep"]

    # Synthesis prompt.
    assert defn.synthesis_prompt is not None
    assert "Prioritize security" in defn.synthesis_prompt


def test_parse_missing_name(tmp_path: Path):
    path = _write_team(tmp_path / "bad.md", "---\ndescription: No name\n---\n\n## Members\n\n- role: A\n")
    assert parse_team_file(path) is None


def test_parse_no_members(tmp_path: Path):
    path = _write_team(tmp_path / "empty.md", "---\nname: empty\ndescription: No members\n---\n\nNothing here.\n")
    assert parse_team_file(path) is None


# ---------------------------------------------------------------------------
# TeamResolver
# ---------------------------------------------------------------------------


def test_resolver_empty(tmp_path: Path):
    resolver = TeamResolver(project_dir=tmp_path / "no-such-dir")
    assert len(resolver.list_available()) == 0
    assert resolver.resolve("anything") is None


def test_resolver_discovers_teams(tmp_path: Path):
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    _write_team(teams_dir / "review-team.md", _MINIMAL_TEAM)
    _write_team(teams_dir / "full-review.md", _FULL_TEAM)

    resolver = TeamResolver(project_dir=teams_dir)
    assert len(resolver.list_available()) == 2
    assert resolver.resolve("review-team") is not None
    assert resolver.resolve("full-review") is not None


def test_resolver_project_overrides_user(tmp_path: Path):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    _write_team(user_dir / "team.md", "---\nname: team\ndescription: User version\n---\n\n## Members\n\n- role: A\n")
    _write_team(project_dir / "team.md", "---\nname: team\ndescription: Project version\n---\n\n## Members\n\n- role: B\n")

    resolver = TeamResolver(user_dir=user_dir, project_dir=project_dir)
    defn = resolver.resolve("team")
    assert defn is not None
    assert defn.description == "Project version"
    assert defn.source == "project"


def test_resolver_reload(tmp_path: Path):
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    resolver = TeamResolver(project_dir=teams_dir)
    assert len(resolver.list_available()) == 0

    _write_team(teams_dir / "new-team.md", _MINIMAL_TEAM)
    resolver.reload()
    assert len(resolver.list_available()) == 1


# ---------------------------------------------------------------------------
# build_catalog
# ---------------------------------------------------------------------------


def test_build_catalog_empty(tmp_path: Path):
    resolver = TeamResolver(project_dir=tmp_path / "empty")
    assert resolver.build_catalog() == ""


def test_build_catalog_with_teams(tmp_path: Path):
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    _write_team(teams_dir / "review-team.md", _MINIMAL_TEAM)

    resolver = TeamResolver(project_dir=teams_dir)
    catalog = resolver.build_catalog()
    assert "review-team" in catalog
    assert "2 members" in catalog
    assert "parallel" in catalog


# ---------------------------------------------------------------------------
# format_team_list
# ---------------------------------------------------------------------------


def test_format_team_list_empty(tmp_path: Path):
    resolver = TeamResolver(project_dir=tmp_path / "empty")
    lines = format_team_list(resolver)
    assert "No teams found" in lines[0]


def test_format_team_list_with_teams(tmp_path: Path):
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    _write_team(teams_dir / "review-team.md", _MINIMAL_TEAM)

    resolver = TeamResolver(project_dir=teams_dir)
    lines = format_team_list(resolver)
    assert any("review-team" in line for line in lines)
    assert any("doc-auditor" in line for line in lines)
