"""Plugin hook system — event-driven actions.

Hooks respond to agent lifecycle events (PostToolUse, SessionStart, etc.)
and execute shell commands.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from maike.plugins.manifest import PluginManifest, _substitute_env_vars

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """Events that can trigger hooks."""

    SESSION_START = "SessionStart"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    STOP = "Stop"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"


class HookType(str, Enum):
    """Types of hook actions."""

    COMMAND = "command"


@dataclass(frozen=True)
class HookDefinition:
    """A single hook action bound to an event."""

    event: HookEvent
    matcher: str | None = None  # regex pattern for tool names
    hook_type: HookType = HookType.COMMAND
    command: str = ""
    timeout_s: int = 10
    source_plugin: str | None = None  # plugin name, or None for project/user hooks

    def matches_tool(self, tool_name: str) -> bool:
        """Return True if this hook's matcher matches the tool name."""
        if self.matcher is None:
            return True
        try:
            return bool(re.search(self.matcher, tool_name))
        except re.error:
            return False


@dataclass
class HookConfig:
    """Aggregated hook configuration from all sources."""

    hooks: dict[HookEvent, list[HookDefinition]] = field(default_factory=dict)

    def get_hooks(self, event: HookEvent, tool_name: str | None = None) -> list[HookDefinition]:
        """Get hooks for an event, optionally filtered by tool name."""
        defs = self.hooks.get(event, [])
        if tool_name is None:
            return defs
        return [d for d in defs if d.matches_tool(tool_name)]

    def merge(self, other: HookConfig) -> None:
        """Merge another config into this one (other's hooks appended)."""
        for event, defs in other.hooks.items():
            self.hooks.setdefault(event, []).extend(defs)


def _parse_event(raw: str) -> HookEvent | None:
    """Parse event name string to enum, case-insensitive."""
    for ev in HookEvent:
        if ev.value.lower() == raw.lower():
            return ev
    return None


def _parse_hooks_json(
    path: Path,
    env: dict[str, str] | None = None,
    source_plugin: str | None = None,
) -> HookConfig:
    """Parse a hooks.json file."""
    config = HookConfig()
    if not path.is_file():
        return config
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse hooks config: %s", path)
        return config

    env = env or {}
    hooks_raw = data.get("hooks", data)  # support both {hooks: {...}} and bare {...}
    if not isinstance(hooks_raw, dict):
        return config

    for event_name, entries in hooks_raw.items():
        event = _parse_event(event_name)
        if event is None:
            logger.warning("Unknown hook event '%s' in %s", event_name, path)
            continue
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            matcher = entry.get("matcher")
            inner_hooks = entry.get("hooks", [])
            if not isinstance(inner_hooks, list):
                continue

            for hook_spec in inner_hooks:
                if not isinstance(hook_spec, dict):
                    continue
                hook_type_raw = hook_spec.get("type", "command")
                if hook_type_raw != "command":
                    logger.debug("Unsupported hook type '%s', skipping", hook_type_raw)
                    continue
                command = hook_spec.get("command", "")
                command = _substitute_env_vars(str(command), env)
                timeout = int(hook_spec.get("timeout", 10))

                defn = HookDefinition(
                    event=event,
                    matcher=matcher,
                    hook_type=HookType.COMMAND,
                    command=command,
                    timeout_s=timeout,
                    source_plugin=source_plugin,
                )
                config.hooks.setdefault(event, []).append(defn)

    return config


def load_hook_configs(
    plugin_manifests: list[PluginManifest] | None = None,
    project_hooks_path: Path | None = None,
) -> HookConfig:
    """Load and merge hook configs from plugins and project.

    Project hooks are loaded first, then plugin hooks are appended.
    """
    merged = HookConfig()

    # Project-level hooks
    if project_hooks_path is not None:
        merged.merge(_parse_hooks_json(project_hooks_path))

    # Plugin hooks
    for manifest in (plugin_manifests or []):
        env = manifest.env_vars()
        plugin_hooks = _parse_hooks_json(
            manifest.hooks_file,
            env=env,
            source_plugin=manifest.name,
        )
        merged.merge(plugin_hooks)

    return merged
