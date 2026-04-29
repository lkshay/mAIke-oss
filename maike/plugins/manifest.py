"""Plugin manifest parsing.

A plugin directory can contain skills, agents, hooks, MCP servers, and LSP servers.
The manifest (``.maike-plugin/plugin.json``) declares metadata and optional component
path overrides.  When paths are omitted, defaults are auto-discovered.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Environment variable substitution pattern.
# Matches ${VAR_NAME} and ${user_config.KEY} (dots allowed for user config).
_ENV_VAR_PATTERN = re.compile(r"\$\{([\w.]+)\}")


def _substitute_env_vars(value: str, env: dict[str, str]) -> str:
    """Replace ``${VAR}`` placeholders in *value* using *env* dict."""
    def _replace(m: re.Match) -> str:
        return env.get(m.group(1), m.group(0))
    return _ENV_VAR_PATTERN.sub(_replace, value)


# ── User config field ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PluginUserConfigField:
    """A user-configurable value declared in plugin.json."""
    description: str
    sensitive: bool = False


# ── Plugin manifest ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PluginManifest:
    """Parsed plugin manifest from ``.maike-plugin/plugin.json``.

    Component paths are relative to the plugin root.  When ``None``, the
    default directory/file is used (auto-discovered if it exists).
    """

    name: str
    description: str
    version: str
    author: str
    path: Path  # directory containing .maike-plugin/

    # Component path overrides (all relative to plugin root)
    skills_path: str | None = None       # default: "skills/"
    agents_path: str | None = None       # default: "agents/"
    hooks_path: str | None = None        # default: "hooks/hooks.json"
    mcp_config_path: str | None = None   # default: ".mcp.json"
    lsp_config_path: str | None = None   # default: ".lsp.json"

    # User-configurable values prompted at enable time
    user_config: dict[str, PluginUserConfigField] = field(default_factory=dict)

    # Optional metadata
    homepage: str = ""
    repository: str = ""
    license: str = ""
    keywords: tuple[str, ...] = ()

    # ── Component directory/file accessors ────────────────────────────

    @property
    def skills_dir(self) -> Path:
        if self.skills_path is not None:
            return self.path / self.skills_path
        # Prefer skills/ but fall back to commands/.
        skills = self.path / "skills"
        if skills.is_dir():
            return skills
        commands = self.path / "commands"
        if commands.is_dir():
            return commands
        return skills  # default even if it doesn't exist

    @property
    def agents_dir(self) -> Path:
        if self.agents_path is not None:
            return self.path / self.agents_path
        return self.path / "agents"

    @property
    def hooks_file(self) -> Path:
        if self.hooks_path is not None:
            return self.path / self.hooks_path
        return self.path / "hooks" / "hooks.json"

    @property
    def mcp_config_file(self) -> Path:
        if self.mcp_config_path is not None:
            return self.path / self.mcp_config_path
        return self.path / ".mcp.json"

    @property
    def lsp_config_file(self) -> Path:
        if self.lsp_config_path is not None:
            return self.path / self.lsp_config_path
        return self.path / ".lsp.json"

    @property
    def data_dir(self) -> Path:
        """Persistent data directory for this plugin.

        Survives plugin updates. Suitable for caches, state files, etc.
        Created lazily on first access by the loader.
        """
        return Path.home() / ".config" / "maike" / "plugins" / "data" / self.name

    def env_vars(self, user_config_values: dict[str, str] | None = None) -> dict[str, str]:
        """Return environment variables available to plugin subprocesses.

        If *user_config_values* is provided, non-sensitive config fields
        are included as ``user_config.KEY`` entries.
        """
        env = {
            "MAIKE_PLUGIN_ROOT": str(self.path),
            "MAIKE_PLUGIN_DATA": str(self.data_dir),
        }
        if user_config_values:
            for key, value in user_config_values.items():
                field = self.user_config.get(key)
                # Only non-sensitive values are substituted into content.
                if field and not field.sensitive:
                    env[f"user_config.{key}"] = value
        return env


def _parse_user_config(raw: Any) -> dict[str, PluginUserConfigField]:
    """Parse the ``userConfig`` object from plugin.json."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, PluginUserConfigField] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        result[key] = PluginUserConfigField(
            description=str(value.get("description", "")),
            sensitive=bool(value.get("sensitive", False)),
        )
    return result


def _safe_component_path(path_str: str, plugin_root: Path) -> bool:
    """Return ``True`` if *path_str* stays within *plugin_root*.

    Blocks path traversal attacks (``..``) that would escape the plugin
    directory.
    """
    if ".." in path_str:
        return False
    try:
        resolved = (plugin_root / path_str).resolve()
        return resolved.is_relative_to(plugin_root.resolve())
    except (ValueError, OSError):
        return False


def parse_plugin_manifest(plugin_dir: Path) -> PluginManifest | None:
    """Parse a plugin manifest from a directory.

    Expects ``plugin_dir/.maike-plugin/plugin.json`` with required fields:
    ``name``, ``version``.  Optional: ``description``, ``author``, component
    paths, ``userConfig``, metadata.

    Returns ``None`` if the manifest is missing or invalid.
    """
    # Accept both .maike-plugin/ and .claude-plugin/.
    manifest_path = plugin_dir / ".maike-plugin" / "plugin.json"
    if not manifest_path.is_file():
        manifest_path = plugin_dir / ".claude-plugin" / "plugin.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    name = data.get("name")
    version = data.get("version")
    if not name or not version:
        return None

    author_raw = data.get("author", "")
    if isinstance(author_raw, dict):
        author = str(author_raw.get("name", ""))
    else:
        author = str(author_raw)

    # Parse optional keywords
    keywords_raw = data.get("keywords", ())
    if isinstance(keywords_raw, list):
        keywords = tuple(str(k) for k in keywords_raw)
    else:
        keywords = ()

    # Extract component path overrides.
    skills_path = data.get("skills") if isinstance(data.get("skills"), str) else None
    agents_path = data.get("agents") if isinstance(data.get("agents"), str) else None
    hooks_path = data.get("hooks") if isinstance(data.get("hooks"), str) else None
    mcp_config_path = data.get("mcpServers") if isinstance(data.get("mcpServers"), str) else None
    lsp_config_path = data.get("lspServers") if isinstance(data.get("lspServers"), str) else None

    # Path traversal validation: component paths must not escape plugin root.
    for label, path_str in [
        ("skills", skills_path), ("agents", agents_path),
        ("hooks", hooks_path), ("mcpServers", mcp_config_path),
        ("lspServers", lsp_config_path),
    ]:
        if path_str and not _safe_component_path(path_str, plugin_dir):
            logger.warning(
                "Plugin '%s' declares %s path '%s' that escapes plugin root — ignored",
                name, label, path_str,
            )
            return None

    return PluginManifest(
        name=str(name),
        description=str(data.get("description", "")),
        version=str(version),
        author=author,
        path=plugin_dir,
        skills_path=skills_path,
        agents_path=agents_path,
        hooks_path=hooks_path,
        mcp_config_path=mcp_config_path,
        lsp_config_path=lsp_config_path,
        # User config
        user_config=_parse_user_config(data.get("userConfig")),
        # Metadata
        homepage=str(data.get("homepage", "")),
        repository=str(data.get("repository", "")),
        license=str(data.get("license", "")),
        keywords=keywords,
    )
