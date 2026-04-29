"""@-mention palette — typed tags for files, skills, agents, teams.

Inserted tokens are of the form ``@file:path/to/foo.py`` / ``@skill:pdf`` /
``@agent:explorer`` / ``@team:backend``.  Selection only manipulates the
visible text — no system-prompt injection, no context-building changes.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from maike.tui.commands import score_match
from maike.tui.widgets.completion_palette import CompletionPalette, PaletteEntry

MENTION_CATEGORIES: tuple[str, ...] = ("file", "skill", "agent", "team")

_CACHE_TTL_SECONDS = 5.0
_FILE_CAP = 5000
_MAX_ENTRIES = 15


@dataclass
class _MentionCache:
    timestamp: float = 0.0
    files: list[str] = field(default_factory=list)
    skills: list[tuple[str, str]] = field(default_factory=list)
    agents: list[tuple[str, str]] = field(default_factory=list)
    teams: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        return time.monotonic() - self.timestamp > _CACHE_TTL_SECONDS


class MentionSources:
    """Aggregates the pool of @-mention candidates with a 5s TTL cache."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = Path(workspace)
        self._cache = _MentionCache()

    def refresh_if_stale(self) -> None:
        if self._cache.is_stale:
            self._reload()

    def files(self) -> list[str]:
        self.refresh_if_stale()
        return self._cache.files

    def skills(self) -> list[tuple[str, str]]:
        self.refresh_if_stale()
        return self._cache.skills

    def agents(self) -> list[tuple[str, str]]:
        self.refresh_if_stale()
        return self._cache.agents

    def teams(self) -> list[tuple[str, str]]:
        self.refresh_if_stale()
        return self._cache.teams

    # ------------------------------------------------------------------
    # Source loaders — each is resilient to missing deps / config
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        self._cache = _MentionCache(
            timestamp=time.monotonic(),
            files=self._list_files(),
            skills=self._load_skills(),
            agents=self._load_agents(),
            teams=self._load_teams(),
        )

    def _list_files(self) -> list[str]:
        """Enumerate workspace files via git-ls (tracked + untracked-respecting-gitignore).

        Falls back to ``rglob("*")`` when git isn't available.  Capped at
        ``_FILE_CAP`` entries.
        """
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=self._workspace,
                capture_output=True,
                timeout=2,
            )
            untracked = subprocess.run(
                ["git", "ls-files", "-z", "--others", "--exclude-standard"],
                cwd=self._workspace,
                capture_output=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return self._list_files_fallback()

        files: set[str] = set()
        for result in (tracked, untracked):
            if result.returncode == 0 and result.stdout:
                for p in result.stdout.decode("utf-8", errors="replace").split("\x00"):
                    if p:
                        files.add(p)
        if not files:
            return self._list_files_fallback()
        return sorted(files)[:_FILE_CAP]

    def _list_files_fallback(self) -> list[str]:
        ws = self._workspace
        _SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
        out: list[str] = []
        try:
            for path in ws.rglob("*"):
                if not path.is_file():
                    continue
                if any(part in _SKIP_DIRS for part in path.relative_to(ws).parts):
                    continue
                out.append(str(path.relative_to(ws)))
                if len(out) >= _FILE_CAP:
                    break
        except Exception:
            pass
        return sorted(out)

    def _load_skills(self) -> list[tuple[str, str]]:
        try:
            from maike.agents.knowledge import _KNOWLEDGE_DIR
            from maike.agents.skill import SkillLoader
            from maike.constants import (
                PLUGIN_PROJECT_SUBDIR,
                PLUGIN_USER_DIR,
                SKILL_PROJECT_SUBDIR,
                SKILL_USER_DIR,
            )
            from maike.plugins.discovery import PluginDiscovery
            from maike.plugins.loader import PluginLoader
        except Exception:
            return []

        try:
            search_dirs: list[Path] = []
            if PLUGIN_USER_DIR.is_dir():
                search_dirs.append(PLUGIN_USER_DIR)
            proj_pdir = self._workspace / PLUGIN_PROJECT_SUBDIR
            if proj_pdir.is_dir():
                search_dirs.append(proj_pdir)
            manifests = PluginDiscovery.discover_enabled(search_dirs)
            plugin_skills = PluginLoader.load_all_plugin_skills(manifests) if manifests else []
            user_dir = SKILL_USER_DIR if SKILL_USER_DIR.is_dir() else None
            proj_dir = self._workspace / SKILL_PROJECT_SUBDIR
            proj_dir = proj_dir if proj_dir.is_dir() else None
            loader = SkillLoader(
                builtin_dir=_KNOWLEDGE_DIR,
                user_dir=user_dir,
                project_dir=proj_dir,
                extra_skills=plugin_skills,
            )
            return [(s.name, (s.description or "")[:80]) for s in loader.load_all()]
        except Exception:
            return []

    def _load_agents(self) -> list[tuple[str, str]]:
        try:
            from maike.agents.agent_resolver import AgentResolver
            from maike.constants import AGENTS_PROJECT_SUBDIR, AGENTS_USER_DIR
        except Exception:
            return []

        try:
            resolver = AgentResolver(
                user_dir=AGENTS_USER_DIR,
                project_dir=self._workspace / AGENTS_PROJECT_SUBDIR,
            )
            out: list[tuple[str, str]] = []
            for agent in resolver.list_agents():
                name = getattr(agent, "name", None) or getattr(agent, "slug", "")
                desc = (getattr(agent, "description", "") or "")[:80]
                if name:
                    out.append((name, desc))
            return out
        except Exception:
            return []

    def _load_teams(self) -> list[tuple[str, str]]:
        try:
            from maike.agents.team_resolver import TeamResolver
            from maike.constants import TEAMS_PROJECT_SUBDIR, TEAMS_USER_DIR
        except Exception:
            return []

        try:
            resolver = TeamResolver(
                user_dir=TEAMS_USER_DIR,
                project_dir=self._workspace / TEAMS_PROJECT_SUBDIR,
            )
            out: list[tuple[str, str]] = []
            for team in resolver.list_teams():
                name = getattr(team, "name", None) or getattr(team, "slug", "")
                desc = (getattr(team, "description", "") or "")[:80]
                if name:
                    out.append((name, desc))
            return out
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Palette widget
# ---------------------------------------------------------------------------


_FILE_LIMIT = 10       # files dominate the palette when the user just types @
_OTHER_LIMIT = 3       # skills/agents/teams each contribute up to this many

# Per-category bias so file matches win ties against same-scored non-file
# matches — users overwhelmingly @-mention files.
_CATEGORY_BIAS = {"file": 1, "skill": 0, "agent": 0, "team": 0}


class MentionPalette(CompletionPalette):
    """Unified @-mention palette — files, skills, agents, teams in one list.

    Rows display plain names (no ``@file:`` / ``@team:`` chips in the UI).
    Selection inserts a typed tag (``@file:src/foo.py``) so the downstream
    agent can disambiguate category when parsing the prompt.
    """

    def __init__(self, sources: MentionSources, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sources = sources

    def update_query(self, query: str) -> None:
        """Refresh entries for the current ``@query`` typing state."""
        scored: list[tuple[int, PaletteEntry]] = []
        scored.extend(self._score_files(query))
        scored.extend(self._score_tagged("skill", self._sources.skills(), query))
        scored.extend(self._score_tagged("agent", self._sources.agents(), query))
        scored.extend(self._score_tagged("team", self._sources.teams(), query))
        # Sort by score desc, then label asc for stable ordering.
        scored.sort(key=lambda se: (-se[0], se[1].label))
        self.update_entries([e for _, e in scored[:_MAX_ENTRIES]])

    # ------------------------------------------------------------------
    # Score builders — one per source.  File entries get a category bias
    # so that when queries match a file and a skill equally, the file wins.
    # ------------------------------------------------------------------

    def _score_files(self, query: str) -> list[tuple[int, PaletteEntry]]:
        files = self._sources.files()
        scored: list[tuple[int, PaletteEntry]] = []
        bias = _CATEGORY_BIAS["file"]
        for f in files:
            basename = Path(f).name
            s = max(score_match(query, basename), score_match(query, f))
            if s >= 0:
                parent = Path(f).parent.as_posix()
                desc = parent if parent != "." else ""
                scored.append(
                    (
                        s + bias,
                        PaletteEntry(
                            id=f"file:{f}",
                            label=basename,
                            description=desc,
                            payload={"category": "file", "value": f},
                        ),
                    )
                )
        # Trim per-category to bound palette size.
        scored.sort(key=lambda se: (-se[0], se[1].label))
        return scored[:_FILE_LIMIT]

    def _score_tagged(
        self,
        category: str,
        items: list[tuple[str, str]],
        query: str,
    ) -> list[tuple[int, PaletteEntry]]:
        """Score a non-file source (skills/agents/teams)."""
        bias = _CATEGORY_BIAS.get(category, 0)
        scored: list[tuple[int, PaletteEntry]] = []
        for name, desc in items:
            s_name = score_match(query, name)
            s_desc = score_match(query, desc) if desc else -1
            s = max(s_name, s_desc)
            if s >= 0:
                # Prefix the description with the category name so the user
                # sees the bucket without an explicit chip.
                row_desc = f"{category} · {desc}" if desc else category
                scored.append(
                    (
                        s + bias,
                        PaletteEntry(
                            id=f"{category}:{name}",
                            label=name,
                            description=row_desc,
                            payload={"category": category, "value": name},
                        ),
                    )
                )
        scored.sort(key=lambda se: (-se[0], se[1].label))
        return scored[:_OTHER_LIMIT]


# ---------------------------------------------------------------------------
# Trigger detection + completion application
# ---------------------------------------------------------------------------


def detect_mention_trigger(
    text: str,
    cursor_row: int,
    cursor_col: int,
) -> str | None:
    """Return the @-mention query if the palette should be active, else None.

    Rules:
    * An ``@`` must appear at a word boundary (start of line or after
      whitespace) before the cursor.
    * No whitespace between that ``@`` and the cursor.
    * If the token contains a colon (user is re-editing a previously
      inserted ``@file:...`` tag), the palette stays dismissed — those
      tokens are considered complete and not re-editable through the
      palette.
    """
    lines = text.split("\n") if text else [""]
    if cursor_row < 0 or cursor_row >= len(lines):
        return None
    line = lines[cursor_row]
    prefix = line[:cursor_col]
    at_idx = prefix.rfind("@")
    if at_idx < 0:
        return None
    if at_idx > 0 and not prefix[at_idx - 1].isspace():
        return None
    token = prefix[at_idx + 1 :]
    if any(c.isspace() for c in token):
        return None
    if ":" in token:
        return None
    return token


def apply_mention_completion(
    text: str,
    cursor_row: int,
    cursor_col: int,
    entry: PaletteEntry,
) -> tuple[str, int, int]:
    """Replace the ``@...`` token ending at the cursor with the typed tag.

    Each entry carries a ``payload`` of ``{"category": ..., "value": ...}``;
    the inserted text is ``@{category}:{value} `` so the downstream agent
    sees an unambiguous typed tag even though the palette row itself was
    plain (no ``@file:`` prefix).  Returns (new_text, new_row, new_col).
    """
    lines = text.split("\n") if text else [""]
    if cursor_row < 0 or cursor_row >= len(lines):
        return text, cursor_row, cursor_col
    line = lines[cursor_row]
    prefix = line[:cursor_col]
    suffix = line[cursor_col:]
    at_idx = prefix.rfind("@")
    if at_idx < 0:
        return text, cursor_row, cursor_col

    # Trim any same-token content on the suffix side (until whitespace).
    tail_end = 0
    while tail_end < len(suffix) and not suffix[tail_end].isspace():
        tail_end += 1
    tail_rest = suffix[tail_end:]

    category = entry.payload.get("category", "")
    value = entry.payload.get("value", "")
    if not category or not value:
        return text, cursor_row, cursor_col
    replacement = f"@{category}:{value}"
    trailing = " "

    new_line = line[:at_idx] + replacement + trailing + tail_rest
    lines[cursor_row] = new_line
    new_text = "\n".join(lines)
    new_col = at_idx + len(replacement) + len(trailing)
    return new_text, cursor_row, new_col
