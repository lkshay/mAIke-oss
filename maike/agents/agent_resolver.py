"""Agent resolver — unified discovery and resolution of agent definitions.

Agents are loaded from four sources with resolution priority (last wins):

    builtin types  →  plugin agents  →  user agents  →  project agents

User agents:    ``~/.config/maike/agents/*.md``
Project agents: ``{workspace}/.maike/agents/*.md``

Agent markdown files use the same frontmatter format as plugin agents::

    ---
    name: my-agent
    description: "When to use this agent"
    model: strong
    maxTurns: 40
    tools: Read, Grep, Bash, Edit
    disallowedTools: Write
    skills: test-methodology
    initialPrompt: "Read the alembic.ini first."
    ---

    You are a database migration specialist...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from maike.plugins.agent_loader import PluginAgent, parse_agent_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDefinition:
    """Unified agent definition — works across all sources."""

    name: str
    description: str
    system_prompt: str
    source: str  # "builtin", "plugin:{namespace}", "user", "project"
    model_tier: str = "default"  # "cheap", "default", "strong"
    max_turns: int | None = None  # None = use global default
    allowed_tools: list[str] | None = None  # None = all tools
    disallowed_tools: list[str] | None = None
    skills: list[str] = field(default_factory=list)
    initial_prompt: str | None = None
    background: bool | None = None  # None = caller decides
    critical_reminder: str | None = None

    @staticmethod
    def from_plugin_agent(pa: PluginAgent) -> AgentDefinition:
        """Convert a PluginAgent to an AgentDefinition."""
        source = f"plugin:{pa.namespace}" if pa.namespace else "plugin"
        return AgentDefinition(
            name=pa.qualified_name,
            description=pa.description,
            system_prompt=pa.system_prompt,
            source=source,
            model_tier=pa.model_tier,
            max_turns=pa.max_turns if pa.max_turns != 20 else None,
            allowed_tools=pa.allowed_tools,
            disallowed_tools=pa.disallowed_tools,
            skills=list(pa.skills),
        )


def _load_agents_from_dir(
    directory: Path, source: str,
) -> list[AgentDefinition]:
    """Load agent definitions from a directory of markdown files."""
    if not directory.is_dir():
        return []

    agents: list[AgentDefinition] = []
    for md_file in sorted(directory.glob("*.md")):
        pa = parse_agent_file(md_file)
        if pa is None:
            continue
        # Parse extended fields not in PluginAgent.
        # Re-read frontmatter to get fields that parse_agent_file doesn't
        # capture (initialPrompt, critical_reminder, background).
        extra = _parse_extended_fields(md_file)
        agents.append(AgentDefinition(
            name=pa.name,
            description=pa.description,
            system_prompt=pa.system_prompt,
            source=source,
            model_tier=pa.model_tier,
            max_turns=pa.max_turns if pa.max_turns != 20 else None,
            allowed_tools=pa.allowed_tools,
            disallowed_tools=pa.disallowed_tools,
            skills=list(pa.skills),
            initial_prompt=extra.get("initialPrompt"),
            background=extra.get("background"),
            critical_reminder=extra.get("critical_reminder"),
        ))
    return agents


def _parse_extended_fields(path: Path) -> dict:
    """Parse frontmatter fields not covered by PluginAgent."""
    import re

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}

    result: dict = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if key == "initialPrompt":
            result["initialPrompt"] = value.strip('"').strip("'")
        elif key == "background":
            result["background"] = value.lower() in ("true", "yes", "1")
        elif key == "critical_reminder":
            result["critical_reminder"] = value.strip('"').strip("'")

    return result


class AgentResolver:
    """Resolves agent names to definitions across all sources.

    Resolution priority (last wins for same name):
        builtin → plugin → user → project
    """

    def __init__(
        self,
        plugin_agents: list[PluginAgent] | None = None,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
    ):
        # Store constructor args for reload().
        self._plugin_agents = list(plugin_agents or [])
        self._user_dir = user_dir
        self._project_dir = project_dir
        self._agents: dict[str, AgentDefinition] = {}
        self._load_all()

    def _load_all(self) -> None:
        """(Re)populate the agent registry from all sources."""
        self._agents.clear()

        # Layer 1: plugin agents (lowest priority among custom agents)
        for pa in self._plugin_agents:
            defn = AgentDefinition.from_plugin_agent(pa)
            self._agents[pa.qualified_name] = defn
            # Also register bare name for convenience
            if pa.namespace and pa.name not in self._agents:
                self._agents[pa.name] = defn

        # Layer 2: user agents (~/.config/maike/agents/)
        if self._user_dir:
            for defn in _load_agents_from_dir(self._user_dir, "user"):
                self._agents[defn.name] = defn

        # Layer 3: project agents ({workspace}/.maike/agents/) — highest
        if self._project_dir:
            for defn in _load_agents_from_dir(self._project_dir, "project"):
                self._agents[defn.name] = defn

    def reload(self) -> None:
        """Rescan all sources and refresh the registry.

        Call after creating or deleting agent files so the resolver
        picks up the changes without needing a new instance.
        """
        self._load_all()

    def resolve(self, name: str) -> AgentDefinition | None:
        """Resolve an agent by name. Returns None if not found."""
        return self._agents.get(name)

    def list_available(self) -> list[AgentDefinition]:
        """Return all available agent definitions (deduplicated by name)."""
        # Use dict to deduplicate — later entries override earlier
        seen: dict[str, AgentDefinition] = {}
        for defn in self._agents.values():
            seen[defn.name] = defn
        return sorted(seen.values(), key=lambda d: d.name)

    def list_names(self) -> list[str]:
        """Return sorted list of available agent names."""
        return sorted(set(self._agents.keys()))

    def build_catalog(self, cap: int = 2000) -> str:
        """One-line-per-agent catalog for context injection.

        Returns a compact markdown list of available agents, truncated
        to *cap* characters.  Used by ``build_react_context()`` to tell
        the main agent which specialists it can delegate to.
        """
        lines: list[str] = []
        for defn in self.list_available():
            tools = ", ".join(defn.allowed_tools) if defn.allowed_tools else "all"
            lines.append(
                f"- **{defn.name}**: {defn.description} (tools: {tools})"
            )
        text = "\n".join(lines)
        return text[:cap] if len(text) > cap else text
