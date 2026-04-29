"""LSP server configuration parsing.

Parses ``.lsp.json`` files from plugins, project, or user directories.
Each LSP server provides code intelligence (diagnostics, go-to-definition,
find references) for specific file types.
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
class LSPServerConfig:
    """Configuration for a single LSP server."""

    name: str
    command: str
    args: tuple[str, ...]
    extension_to_language: dict[str, str]   # e.g. {".py": "python"}
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"
    initialization_options: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    startup_timeout_ms: int = 10_000
    shutdown_timeout_ms: int = 5_000
    restart_on_crash: bool = True
    max_restarts: int = 3
    source: str = "project"  # "plugin", "project", "user"


def _parse_lsp_json(
    path: Path,
    source: str,
    env: dict[str, str] | None = None,
) -> list[LSPServerConfig]:
    """Parse a ``.lsp.json`` file into server configs."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse LSP config: %s", path)
        return []

    if not isinstance(data, dict):
        return []

    env = env or {}
    configs: list[LSPServerConfig] = []

    for name, spec in data.items():
        if not isinstance(spec, dict):
            continue
        command = spec.get("command")
        if not command:
            logger.warning("LSP server '%s' in %s missing 'command', skipping", name, path)
            continue

        ext_to_lang = spec.get("extensionToLanguage", {})
        if not isinstance(ext_to_lang, dict):
            ext_to_lang = {}

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

        configs.append(LSPServerConfig(
            name=str(name),
            command=command,
            args=args,
            extension_to_language={str(k): str(v) for k, v in ext_to_lang.items()},
            env=server_env,
            transport=str(spec.get("transport", "stdio")),
            initialization_options=spec.get("initializationOptions", {}),
            settings=spec.get("settings", {}),
            startup_timeout_ms=int(spec.get("startupTimeout", 10_000)),
            shutdown_timeout_ms=int(spec.get("shutdownTimeout", 5_000)),
            restart_on_crash=bool(spec.get("restartOnCrash", True)),
            max_restarts=int(spec.get("maxRestarts", 3)),
            source=source,
        ))

    return configs


def load_lsp_configs(
    plugin_manifests: list[PluginManifest] | None = None,
    project_lsp_path: Path | None = None,
) -> list[LSPServerConfig]:
    """Load LSP configs from plugins and project, merged by name."""
    configs_by_name: dict[str, LSPServerConfig] = {}

    # Project-level
    if project_lsp_path is not None:
        for cfg in _parse_lsp_json(project_lsp_path, source="project"):
            configs_by_name[cfg.name] = cfg

    # Plugin-bundled
    for manifest in (plugin_manifests or []):
        env = manifest.env_vars()
        for cfg in _parse_lsp_json(manifest.lsp_config_file, source="plugin", env=env):
            namespaced = f"{manifest.name}:{cfg.name}"
            configs_by_name[namespaced] = LSPServerConfig(
                name=namespaced,
                command=cfg.command,
                args=cfg.args,
                extension_to_language=cfg.extension_to_language,
                env=cfg.env,
                transport=cfg.transport,
                initialization_options=cfg.initialization_options,
                settings=cfg.settings,
                startup_timeout_ms=cfg.startup_timeout_ms,
                shutdown_timeout_ms=cfg.shutdown_timeout_ms,
                restart_on_crash=cfg.restart_on_crash,
                max_restarts=cfg.max_restarts,
                source="plugin",
            )

    return sorted(configs_by_name.values(), key=lambda c: c.name)
