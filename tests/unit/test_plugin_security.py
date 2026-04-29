"""Tests for plugin security: path traversal validation, trust boundaries,
user config variable substitution, and data directory."""

import json
from pathlib import Path

from maike.plugins.manifest import (
    PluginManifest,
    PluginUserConfigField,
    _safe_component_path,
    parse_plugin_manifest,
)
from maike.plugins.agent_loader import parse_agent_file


# ── Path traversal validation ──────────────────────────────────────────────


class TestPathTraversalValidation:
    def test_safe_relative_path(self, tmp_path):
        assert _safe_component_path("skills/", tmp_path) is True

    def test_safe_nested_path(self, tmp_path):
        assert _safe_component_path("a/b/c/skills", tmp_path) is True

    def test_blocks_parent_traversal(self, tmp_path):
        assert _safe_component_path("../etc/passwd", tmp_path) is False

    def test_blocks_hidden_traversal(self, tmp_path):
        assert _safe_component_path("skills/../../secrets", tmp_path) is False

    def test_blocks_absolute_path(self, tmp_path):
        # Absolute paths resolve outside plugin root
        assert _safe_component_path("/etc/passwd", tmp_path) is False

    def test_manifest_rejected_on_traversal(self, tmp_path):
        """Manifest with path traversal in component paths is rejected."""
        plugin_dir = tmp_path / "evil-plugin"
        plugin_dir.mkdir()
        meta_dir = plugin_dir / ".maike-plugin"
        meta_dir.mkdir()
        (meta_dir / "plugin.json").write_text(json.dumps({
            "name": "evil",
            "version": "1.0.0",
            "skills": "../../../etc",
        }))
        manifest = parse_plugin_manifest(plugin_dir)
        assert manifest is None

    def test_manifest_accepted_with_safe_paths(self, tmp_path):
        """Manifest with safe component paths is accepted."""
        plugin_dir = tmp_path / "good-plugin"
        plugin_dir.mkdir()
        meta_dir = plugin_dir / ".maike-plugin"
        meta_dir.mkdir()
        (meta_dir / "plugin.json").write_text(json.dumps({
            "name": "good",
            "version": "1.0.0",
            "skills": "custom-skills/",
            "agents": "custom-agents/",
        }))
        manifest = parse_plugin_manifest(plugin_dir)
        assert manifest is not None
        assert manifest.skills_path == "custom-skills/"


# ── Plugin trust boundaries ────────────────────────────────────────────────


class TestPluginTrustBoundaries:
    def test_plugin_agent_hooks_ignored(self, tmp_path):
        """Plugin agents cannot declare hooks in frontmatter."""
        (tmp_path / "agent.md").write_text(
            "---\n"
            "name: my-agent\n"
            "description: An agent\n"
            "hooks: PostToolUse\n"
            "---\n"
            "System prompt"
        )
        # With namespace (plugin agent) — hooks field should trigger warning
        # but agent should still parse (hooks not applied to PluginAgent)
        agent = parse_agent_file(tmp_path / "agent.md", namespace="my-plugin")
        assert agent is not None
        assert agent.name == "my-agent"

    def test_non_plugin_agent_no_warning(self, tmp_path):
        """Non-plugin agents don't trigger trust boundary warnings."""
        (tmp_path / "agent.md").write_text(
            "---\n"
            "name: my-agent\n"
            "description: An agent\n"
            "---\n"
            "System prompt"
        )
        agent = parse_agent_file(tmp_path / "agent.md")
        assert agent is not None


# ── User config variable substitution ──────────────────────────────────────


class TestUserConfigEnvVars:
    def test_non_sensitive_included(self):
        manifest = PluginManifest(
            name="test", description="", version="1.0", author="", path=Path("/tmp/test"),
            user_config={
                "endpoint": PluginUserConfigField(description="API endpoint", sensitive=False),
            },
        )
        env = manifest.env_vars(user_config_values={"endpoint": "https://api.example.com"})
        assert env["user_config.endpoint"] == "https://api.example.com"

    def test_sensitive_excluded(self):
        manifest = PluginManifest(
            name="test", description="", version="1.0", author="", path=Path("/tmp/test"),
            user_config={
                "api_key": PluginUserConfigField(description="API key", sensitive=True),
            },
        )
        env = manifest.env_vars(user_config_values={"api_key": "secret-123"})
        assert "user_config.api_key" not in env

    def test_mixed_config(self):
        manifest = PluginManifest(
            name="test", description="", version="1.0", author="", path=Path("/tmp/test"),
            user_config={
                "endpoint": PluginUserConfigField(description="EP", sensitive=False),
                "api_key": PluginUserConfigField(description="Key", sensitive=True),
            },
        )
        env = manifest.env_vars(user_config_values={
            "endpoint": "https://example.com",
            "api_key": "secret",
        })
        assert "user_config.endpoint" in env
        assert "user_config.api_key" not in env

    def test_no_user_config(self):
        manifest = PluginManifest(
            name="test", description="", version="1.0", author="", path=Path("/tmp/test"),
        )
        env = manifest.env_vars()
        assert "MAIKE_PLUGIN_ROOT" in env
        assert "MAIKE_PLUGIN_DATA" in env


# ── Plugin data directory ──────────────────────────────────────────────────


class TestPluginDataDirectory:
    def test_data_dir_path(self):
        manifest = PluginManifest(
            name="my-plugin", description="", version="1.0", author="",
            path=Path("/tmp/my-plugin"),
        )
        assert manifest.data_dir.name == "my-plugin"
        assert "plugins/data" in str(manifest.data_dir)

    def test_data_dir_in_env_vars(self):
        manifest = PluginManifest(
            name="my-plugin", description="", version="1.0", author="",
            path=Path("/tmp/my-plugin"),
        )
        env = manifest.env_vars()
        assert "MAIKE_PLUGIN_DATA" in env
        assert "my-plugin" in env["MAIKE_PLUGIN_DATA"]

    def test_data_dir_different_from_plugin_root(self):
        manifest = PluginManifest(
            name="my-plugin", description="", version="1.0", author="",
            path=Path("/tmp/my-plugin"),
        )
        env = manifest.env_vars()
        assert env["MAIKE_PLUGIN_ROOT"] != env["MAIKE_PLUGIN_DATA"]
