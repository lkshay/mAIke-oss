"""Plugin agent loader — parse subagent definitions from plugin directories.

Plugin agents are markdown files with YAML frontmatter in the ``agents/``
directory.  They define specialized subagents that can be invoked via the
Delegate tool.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from maike.plugins.manifest import PluginManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginAgent:
    """A subagent definition from a plugin."""

    name: str
    description: str
    system_prompt: str
    model_tier: str = "default"       # "cheap", "default", "strong"
    max_turns: int = 20
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    skills: list[str] = field(default_factory=list)
    namespace: str | None = None      # plugin name

    @property
    def qualified_name(self) -> str:
        """Return ``namespace:name`` if namespaced, else bare name."""
        if self.namespace:
            return f"{self.namespace}:{self.name}"
        return self.name


def _parse_agent_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML frontmatter from markdown body.

    Returns (frontmatter_dict, body_text).
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    fm_text = match.group(1)
    body = match.group(2)

    # Simple YAML-ish parsing (same approach as skill.py)
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()

    return fm, body


def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated list of strings."""
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_agent_file(path: Path, namespace: str | None = None) -> PluginAgent | None:
    """Parse a single agent markdown file.

    Returns None if required fields (name, description) are missing.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read agent file: %s", path)
        return None

    fm, body = _parse_agent_frontmatter(text)

    name = fm.get("name")
    description = fm.get("description")
    if not name or not description:
        logger.warning("Agent file %s missing name or description", path)
        return None

    # Trust boundary: plugin agents cannot declare hooks, MCP servers,
    # or permission overrides in frontmatter — these must be declared
    # at the manifest level to stay within install-time trust.
    if namespace:
        _restricted = {"hooks", "mcpServers", "permissionMode"}
        for key in _restricted:
            if key in fm:
                logger.warning(
                    "Plugin agent %s:%s declares '%s' in frontmatter "
                    "(ignored — use plugin manifest instead)",
                    namespace, name, key,
                )

    # Parse tool lists
    allowed = None
    if "tools" in fm:
        allowed = _parse_csv(fm["tools"])
    disallowed = None
    if "disallowedTools" in fm:
        disallowed = _parse_csv(fm["disallowedTools"])

    # Parse skills
    skills_list: list[str] = []
    if "skills" in fm:
        skills_list = _parse_csv(fm["skills"])

    return PluginAgent(
        name=name,
        description=description,
        system_prompt=body.strip(),
        model_tier=fm.get("model", "default"),
        max_turns=int(fm.get("maxTurns", "20")),
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        skills=skills_list,
        namespace=namespace,
    )


def load_plugin_agents(manifest: PluginManifest) -> list[PluginAgent]:
    """Load all agent definitions from a plugin."""
    agents_dir = manifest.agents_dir
    if not agents_dir.is_dir():
        return []

    agents: list[PluginAgent] = []
    for md_file in sorted(agents_dir.glob("*.md")):
        agent = parse_agent_file(md_file, namespace=manifest.name)
        if agent is not None:
            agents.append(agent)
    return agents


def load_all_plugin_agents(
    manifests: list[PluginManifest] | None = None,
) -> list[PluginAgent]:
    """Load agents from all plugins."""
    all_agents: list[PluginAgent] = []
    for manifest in (manifests or []):
        try:
            agents = load_plugin_agents(manifest)
            all_agents.extend(agents)
            if agents:
                logger.info(
                    "Loaded %d agents from plugin '%s'",
                    len(agents),
                    manifest.name,
                )
        except Exception:
            logger.warning(
                "Failed to load agents from plugin '%s'",
                manifest.name,
                exc_info=True,
            )
    return all_agents
