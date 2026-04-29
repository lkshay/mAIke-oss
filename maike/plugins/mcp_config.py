"""MCP server configuration parsing.

Loads ``.mcp.json`` files from three sources:
1. Plugin-bundled: ``<plugin>/.mcp.json``
2. Project-level: ``<workspace>/.mcp.json``
3. User-level: ``~/.config/maike/.mcp.json``

Project overrides user; plugin configs are namespaced by plugin name.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maike.plugins.manifest import PluginManifest, _substitute_env_vars

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    source: str = "project"  # "plugin", "project", "user"


def _parse_mcp_json(path: Path, source: str, env: dict[str, str] | None = None) -> list[MCPServerConfig]:
    """Parse a ``.mcp.json`` file into server configs."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse MCP config: %s", path)
        return []

    servers_raw = data.get("mcpServers", {})
    if not isinstance(servers_raw, dict):
        return []

    env = env or {}
    configs: list[MCPServerConfig] = []
    for name, spec in servers_raw.items():
        if not isinstance(spec, dict):
            continue
        command = spec.get("command")
        if not command:
            logger.warning("MCP server '%s' in %s missing 'command', skipping", name, path)
            continue

        command = _substitute_env_vars(str(command), env)
        args = tuple(
            _substitute_env_vars(str(a), env)
            for a in spec.get("args", [])
        )
        server_env = {
            str(k): _substitute_env_vars(str(v), env)
            for k, v in spec.get("env", {}).items()
            if isinstance(k, str)
        }
        cwd_raw = spec.get("cwd")
        cwd = _substitute_env_vars(str(cwd_raw), env) if cwd_raw else None

        configs.append(MCPServerConfig(
            name=str(name),
            command=command,
            args=args,
            env=server_env,
            cwd=cwd,
            source=source,
        ))

    return configs


def _parse_inline_mcp(
    data: dict[str, Any],
    source: str,
    env: dict[str, str] | None = None,
) -> list[MCPServerConfig]:
    """Parse inline ``mcpServers`` from a plugin.json manifest."""
    servers_raw = data if isinstance(data, dict) else {}
    env = env or {}
    configs: list[MCPServerConfig] = []
    for name, spec in servers_raw.items():
        if not isinstance(spec, dict):
            continue
        command = spec.get("command")
        if not command:
            continue
        command = _substitute_env_vars(str(command), env)
        args = tuple(
            _substitute_env_vars(str(a), env)
            for a in spec.get("args", [])
        )
        server_env = {
            str(k): _substitute_env_vars(str(v), env)
            for k, v in spec.get("env", {}).items()
            if isinstance(k, str)
        }
        cwd_raw = spec.get("cwd")
        cwd = _substitute_env_vars(str(cwd_raw), env) if cwd_raw else None
        configs.append(MCPServerConfig(
            name=str(name),
            command=command,
            args=args,
            env=server_env,
            cwd=cwd,
            source=source,
        ))
    return configs


def load_mcp_configs(
    workspace: Path,
    plugin_manifests: list[PluginManifest] | None = None,
) -> list[MCPServerConfig]:
    """Load MCP configs from all sources, merged by name.

    Precedence (later wins): user < project < plugin.
    """
    from maike.constants import MCP_PROJECT_CONFIG_NAME, MCP_USER_CONFIG

    configs_by_name: dict[str, MCPServerConfig] = {}

    # 1. User-level
    for cfg in _parse_mcp_json(MCP_USER_CONFIG, source="user"):
        configs_by_name[cfg.name] = cfg

    # 2. Project-level
    project_config = workspace / MCP_PROJECT_CONFIG_NAME
    for cfg in _parse_mcp_json(project_config, source="project"):
        configs_by_name[cfg.name] = cfg

    # 3. Plugin-bundled
    for manifest in (plugin_manifests or []):
        env = manifest.env_vars()

        # Check for inline mcpServers in the manifest first
        manifest_path = manifest.path / ".maike-plugin" / "plugin.json"
        if manifest_path.is_file():
            try:
                mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
                mcp_raw = mdata.get("mcpServers")
                if isinstance(mcp_raw, dict) and not isinstance(mcp_raw, str):
                    # Check if any value is itself a dict (inline server defs)
                    if any(isinstance(v, dict) for v in mcp_raw.values()):
                        inline_cfgs = _parse_inline_mcp(mcp_raw, source="plugin", env=env)
                        for cfg in inline_cfgs:
                            namespaced = f"{manifest.name}:{cfg.name}"
                            configs_by_name[namespaced] = MCPServerConfig(
                                name=namespaced,
                                command=cfg.command,
                                args=cfg.args,
                                env=cfg.env,
                                cwd=cfg.cwd,
                                source="plugin",
                            )
            except (json.JSONDecodeError, OSError):
                pass

        # Also check .mcp.json file
        for cfg in _parse_mcp_json(manifest.mcp_config_file, source="plugin", env=env):
            namespaced = f"{manifest.name}:{cfg.name}"
            configs_by_name[namespaced] = MCPServerConfig(
                name=namespaced,
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
                cwd=cfg.cwd,
                source="plugin",
            )

    return sorted(configs_by_name.values(), key=lambda c: c.name)
