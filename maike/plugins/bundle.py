"""PluginBundle — aggregates all discovered plugin components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maike.agents.skill import Skill
    from maike.plugins.agent_loader import PluginAgent
    from maike.plugins.hooks import HookConfig
    from maike.plugins.lsp_config import LSPServerConfig
    from maike.plugins.manifest import PluginManifest
    from maike.plugins.mcp_config import MCPServerConfig


@dataclass
class PluginBundle:
    """All plugin components discovered for a session."""

    manifests: list[PluginManifest] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    agents: list[PluginAgent] = field(default_factory=list)
    hooks: HookConfig | None = None
    mcp_configs: list[MCPServerConfig] = field(default_factory=list)
    lsp_configs: list[LSPServerConfig] = field(default_factory=list)
