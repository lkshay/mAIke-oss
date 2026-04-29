"""Tests for maike.plugins.manifest — plugin manifest parsing."""
from __future__ import annotations

import json
from pathlib import Path

from maike.plugins.manifest import (
    PluginManifest,
    PluginUserConfigField,
    _substitute_env_vars,
    parse_plugin_manifest,
)


def _write_manifest(plugin_dir: Path, data: dict) -> None:
    """Helper: write a plugin.json manifest inside plugin_dir/.maike-plugin/."""
    manifest_dir = plugin_dir / ".maike-plugin"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "plugin.json").write_text(json.dumps(data), encoding="utf-8")


# ── Basic parsing ─────────────────────────────────────────────────────


def test_parse_valid_manifest(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "my-plugin"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "my-plugin",
        "description": "A test plugin",
        "version": "1.0.0",
        "author": "Alice",
    })

    result = parse_plugin_manifest(plugin_dir)

    assert result is not None
    assert result.name == "my-plugin"
    assert result.description == "A test plugin"
    assert result.version == "1.0.0"
    assert result.author == "Alice"
    assert result.path == plugin_dir


def test_parse_manifest_missing_name(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "no-name"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {"version": "1.0.0"})

    assert parse_plugin_manifest(plugin_dir) is None


def test_parse_manifest_missing_version(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "no-version"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {"name": "no-version"})

    assert parse_plugin_manifest(plugin_dir) is None


def test_parse_manifest_missing_file(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "empty"
    plugin_dir.mkdir()

    assert parse_plugin_manifest(plugin_dir) is None


def test_parse_manifest_invalid_json(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "bad-json"
    manifest_dir = plugin_dir / ".maike-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "plugin.json").write_text("{invalid json!", encoding="utf-8")

    assert parse_plugin_manifest(plugin_dir) is None


def test_parse_manifest_author_as_dict(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "dict-author"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "dict-author",
        "version": "2.0.0",
        "author": {"name": "Bob", "email": "bob@example.com"},
    })

    result = parse_plugin_manifest(plugin_dir)

    assert result is not None
    assert result.author == "Bob"


def test_parse_manifest_author_as_string(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "str-author"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "str-author",
        "version": "1.0.0",
        "author": "Charlie",
    })

    result = parse_plugin_manifest(plugin_dir)

    assert result is not None
    assert result.author == "Charlie"


# ── Component path properties ─────────────────────────────────────────


def test_skills_dir_property(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "with-skills"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "with-skills",
        "version": "1.0.0",
    })

    result = parse_plugin_manifest(plugin_dir)

    assert result is not None
    assert result.skills_dir == plugin_dir / "skills"


def test_default_component_paths(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "defaults"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {"name": "defaults", "version": "1.0.0"})

    m = parse_plugin_manifest(plugin_dir)
    assert m is not None
    assert m.skills_dir == plugin_dir / "skills"
    assert m.agents_dir == plugin_dir / "agents"
    assert m.hooks_file == plugin_dir / "hooks" / "hooks.json"
    assert m.mcp_config_file == plugin_dir / ".mcp.json"
    assert m.lsp_config_file == plugin_dir / ".lsp.json"


def test_custom_component_paths(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "custom"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "custom",
        "version": "1.0.0",
        "skills": "./custom/skills/",
        "agents": "./my-agents/",
        "hooks": "./config/hooks.json",
        "mcpServers": "./mcp-config.json",
        "lspServers": "./lsp-config.json",
    })

    m = parse_plugin_manifest(plugin_dir)
    assert m is not None
    assert m.skills_dir == plugin_dir / "./custom/skills/"
    assert m.agents_dir == plugin_dir / "./my-agents/"
    assert m.hooks_file == plugin_dir / "./config/hooks.json"
    assert m.mcp_config_file == plugin_dir / "./mcp-config.json"
    assert m.lsp_config_file == plugin_dir / "./lsp-config.json"


# ── User config ───────────────────────────────────────────────────────


def test_user_config_parsed(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "with-config"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "with-config",
        "version": "1.0.0",
        "userConfig": {
            "api_endpoint": {"description": "API endpoint", "sensitive": False},
            "api_token": {"description": "API token", "sensitive": True},
        },
    })

    m = parse_plugin_manifest(plugin_dir)
    assert m is not None
    assert len(m.user_config) == 2
    assert m.user_config["api_endpoint"].description == "API endpoint"
    assert m.user_config["api_endpoint"].sensitive is False
    assert m.user_config["api_token"].sensitive is True


def test_user_config_empty_when_absent(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "no-config"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {"name": "no-config", "version": "1.0.0"})

    m = parse_plugin_manifest(plugin_dir)
    assert m is not None
    assert m.user_config == {}


# ── Metadata fields ───────────────────────────────────────────────────


def test_metadata_fields(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "meta"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "meta",
        "version": "2.0.0",
        "homepage": "https://example.com",
        "repository": "https://github.com/user/meta",
        "license": "MIT",
        "keywords": ["tool", "testing"],
    })

    m = parse_plugin_manifest(plugin_dir)
    assert m is not None
    assert m.homepage == "https://example.com"
    assert m.repository == "https://github.com/user/meta"
    assert m.license == "MIT"
    assert m.keywords == ("tool", "testing")


# ── Environment variable substitution ─────────────────────────────────


def test_substitute_env_vars() -> None:
    env = {"MAIKE_PLUGIN_ROOT": "/home/user/plugins/my-plugin"}
    assert _substitute_env_vars(
        "${MAIKE_PLUGIN_ROOT}/scripts/run.sh", env
    ) == "/home/user/plugins/my-plugin/scripts/run.sh"


def test_substitute_env_vars_missing_key() -> None:
    assert _substitute_env_vars("${MISSING}/foo", {}) == "${MISSING}/foo"


def test_env_vars_property(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "env-test"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {"name": "env-test", "version": "1.0.0"})

    m = parse_plugin_manifest(plugin_dir)
    assert m is not None
    env = m.env_vars()
    assert env["MAIKE_PLUGIN_ROOT"] == str(plugin_dir)


# ── Non-string component paths ignored ────────────────────────────────


def test_inline_object_component_paths_ignored(tmp_path: Path) -> None:
    """When mcpServers is an inline object (not a path string), it's ignored
    as a path override — the default .mcp.json path is used."""
    plugin_dir = tmp_path / "inline"
    plugin_dir.mkdir()
    _write_manifest(plugin_dir, {
        "name": "inline",
        "version": "1.0.0",
        "mcpServers": {"server-a": {"command": "npx", "args": []}},
    })

    m = parse_plugin_manifest(plugin_dir)
    assert m is not None
    # Inline objects are not treated as path overrides
    assert m.mcp_config_path is None
    assert m.mcp_config_file == plugin_dir / ".mcp.json"
