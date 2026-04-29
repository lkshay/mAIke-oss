"""Plugin settings persistence — enable/disable state, install metadata, user config.

Reads and writes ``~/.config/maike/settings.json``.  Returns empty defaults
when the file is missing or invalid (graceful degradation).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InstalledPluginRecord:
    """Metadata about an installed plugin."""

    source: str          # local path or git URL
    version: str
    installed_at: str    # absolute path where installed
    installed_on: str    # ISO timestamp


@dataclass
class PluginSettings:
    """Plugin state read from / written to settings.json."""

    disabled: set[str] = field(default_factory=set)
    installed: dict[str, InstalledPluginRecord] = field(default_factory=dict)
    config: dict[str, dict[str, str]] = field(default_factory=dict)


def load_settings(path: Path | None = None) -> PluginSettings:
    """Load plugin settings from disk.  Returns empty defaults on any error."""
    from maike.constants import SETTINGS_PATH

    settings_path = path or SETTINGS_PATH
    if not settings_path.is_file():
        return PluginSettings()

    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read settings: %s", settings_path)
        return PluginSettings()

    plugins = raw.get("plugins", {})
    if not isinstance(plugins, dict):
        return PluginSettings()

    # Parse disabled
    disabled_raw = plugins.get("disabled", [])
    disabled = set(disabled_raw) if isinstance(disabled_raw, list) else set()

    # Parse installed
    installed: dict[str, InstalledPluginRecord] = {}
    for name, rec in plugins.get("installed", {}).items():
        if isinstance(rec, dict):
            installed[name] = InstalledPluginRecord(
                source=str(rec.get("source", "")),
                version=str(rec.get("version", "")),
                installed_at=str(rec.get("installed_at", "")),
                installed_on=str(rec.get("installed_on", "")),
            )

    # Parse config
    config_raw = plugins.get("config", {})
    config: dict[str, dict[str, str]] = {}
    if isinstance(config_raw, dict):
        for name, vals in config_raw.items():
            if isinstance(vals, dict):
                config[name] = {str(k): str(v) for k, v in vals.items()}

    return PluginSettings(disabled=disabled, installed=installed, config=config)


def save_settings(settings: PluginSettings, path: Path | None = None) -> None:
    """Write plugin settings to disk."""
    from maike.constants import SETTINGS_PATH

    settings_path = path or SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve any existing top-level keys we don't manage
    existing: dict[str, Any] = {}
    if settings_path.is_file():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    installed_dict = {
        name: {
            "source": rec.source,
            "version": rec.version,
            "installed_at": rec.installed_at,
            "installed_on": rec.installed_on,
        }
        for name, rec in settings.installed.items()
    }

    existing["plugins"] = {
        "disabled": sorted(settings.disabled),
        "installed": installed_dict,
        "config": settings.config,
    }

    settings_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def is_plugin_disabled(name: str, path: Path | None = None) -> bool:
    """Check if a plugin is disabled."""
    return name in load_settings(path).disabled
