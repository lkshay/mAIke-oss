"""Tests for plugin system components: MCP config, hooks, agents, LSP, bundle."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from maike.plugins.agent_loader import (
    PluginAgent,
    load_all_plugin_agents,
    load_plugin_agents,
    parse_agent_file,
)
from maike.plugins.bundle import PluginBundle
from maike.plugins.hooks import (
    HookConfig,
    HookDefinition,
    HookEvent,
    HookType,
    _parse_hooks_json,
    load_hook_configs,
)
from maike.plugins.lsp_config import LSPServerConfig, _parse_lsp_json, load_lsp_configs
from maike.plugins.manifest import PluginManifest, parse_plugin_manifest
from maike.plugins.mcp_config import MCPServerConfig, _parse_mcp_json, load_mcp_configs


# ── Helpers ───────────────────────────────────────────────────────────


def _make_plugin(parent: Path, name: str, manifest_data: dict | None = None) -> Path:
    plugin_dir = parent / name
    manifest_dir = plugin_dir / ".maike-plugin"
    manifest_dir.mkdir(parents=True)
    data = manifest_data or {"name": name, "version": "1.0.0"}
    (manifest_dir / "plugin.json").write_text(json.dumps(data), encoding="utf-8")
    return plugin_dir


# ── MCP Config Tests ──────────────────────────────────────────────────


class TestMCPConfig:
    def test_parse_mcp_json_basic(self, tmp_path: Path) -> None:
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@mcp/server-filesystem", "/tmp"],
                    "env": {"NODE_ENV": "production"},
                },
            },
        }))

        configs = _parse_mcp_json(mcp_file, source="project")
        assert len(configs) == 1
        assert configs[0].name == "filesystem"
        assert configs[0].command == "npx"
        assert configs[0].args == ("-y", "@mcp/server-filesystem", "/tmp")
        assert configs[0].env == {"NODE_ENV": "production"}
        assert configs[0].source == "project"

    def test_parse_mcp_json_missing_command(self, tmp_path: Path) -> None:
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {"bad": {"args": []}},
        }))
        configs = _parse_mcp_json(mcp_file, source="project")
        assert len(configs) == 0

    def test_parse_mcp_json_nonexistent(self, tmp_path: Path) -> None:
        configs = _parse_mcp_json(tmp_path / "nope.json", source="user")
        assert configs == []

    def test_parse_mcp_json_env_substitution(self, tmp_path: Path) -> None:
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {
                "db": {
                    "command": "${MAIKE_PLUGIN_ROOT}/server",
                    "args": ["--config", "${MAIKE_PLUGIN_ROOT}/config.json"],
                },
            },
        }))
        env = {"MAIKE_PLUGIN_ROOT": "/plugins/my-plugin"}
        configs = _parse_mcp_json(mcp_file, source="plugin", env=env)
        assert len(configs) == 1
        assert configs[0].command == "/plugins/my-plugin/server"
        assert configs[0].args[1] == "/plugins/my-plugin/config.json"

    def test_load_mcp_configs_merges_sources(self, tmp_path: Path, monkeypatch) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()

        # User config
        user_config = tmp_path / "user_mcp.json"
        user_config.write_text(json.dumps({
            "mcpServers": {"shared": {"command": "old-cmd", "args": []}},
        }))
        monkeypatch.setattr("maike.constants.MCP_USER_CONFIG", user_config)

        # Project config (overrides user)
        project_config = workspace / ".mcp.json"
        project_config.write_text(json.dumps({
            "mcpServers": {"shared": {"command": "new-cmd", "args": []}},
        }))

        configs = load_mcp_configs(workspace)
        assert len(configs) == 1
        assert configs[0].command == "new-cmd"

    def test_load_mcp_configs_from_plugin(self, tmp_path: Path, monkeypatch) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()

        # No user/project configs
        monkeypatch.setattr("maike.constants.MCP_USER_CONFIG", tmp_path / "none.json")

        # Plugin with .mcp.json
        plugin_dir = _make_plugin(tmp_path / "plugins", "db-tools")
        mcp_file = plugin_dir / ".mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {"sqlite": {"command": "uvx", "args": ["mcp-sqlite"]}},
        }))

        manifest = parse_plugin_manifest(plugin_dir)
        assert manifest is not None

        configs = load_mcp_configs(workspace, [manifest])
        assert len(configs) == 1
        assert configs[0].name == "db-tools:sqlite"
        assert configs[0].source == "plugin"


# ── Hook Tests ────────────────────────────────────────────────────────


class TestHooks:
    def test_parse_hooks_json(self, tmp_path: Path) -> None:
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Write|Edit",
                        "hooks": [
                            {"type": "command", "command": "echo formatted"},
                        ],
                    },
                ],
            },
        }))

        config = _parse_hooks_json(hooks_file)
        assert HookEvent.POST_TOOL_USE in config.hooks
        defs = config.hooks[HookEvent.POST_TOOL_USE]
        assert len(defs) == 1
        assert defs[0].matcher == "Write|Edit"
        assert defs[0].command == "echo formatted"

    def test_hook_matches_tool(self) -> None:
        hook = HookDefinition(
            event=HookEvent.POST_TOOL_USE,
            matcher="Write|Edit",
            command="echo ok",
        )
        assert hook.matches_tool("Write") is True
        assert hook.matches_tool("Edit") is True
        assert hook.matches_tool("Read") is False

    def test_hook_no_matcher_matches_all(self) -> None:
        hook = HookDefinition(event=HookEvent.SESSION_START, command="echo start")
        assert hook.matches_tool("anything") is True

    def test_hook_config_get_hooks_filtered(self) -> None:
        config = HookConfig(hooks={
            HookEvent.POST_TOOL_USE: [
                HookDefinition(event=HookEvent.POST_TOOL_USE, matcher="Write", command="cmd1"),
                HookDefinition(event=HookEvent.POST_TOOL_USE, matcher="Read", command="cmd2"),
            ],
        })
        result = config.get_hooks(HookEvent.POST_TOOL_USE, tool_name="Write")
        assert len(result) == 1
        assert result[0].command == "cmd1"

    def test_hook_config_merge(self) -> None:
        a = HookConfig(hooks={
            HookEvent.SESSION_START: [HookDefinition(event=HookEvent.SESSION_START, command="a")],
        })
        b = HookConfig(hooks={
            HookEvent.SESSION_START: [HookDefinition(event=HookEvent.SESSION_START, command="b")],
            HookEvent.STOP: [HookDefinition(event=HookEvent.STOP, command="c")],
        })
        a.merge(b)
        assert len(a.hooks[HookEvent.SESSION_START]) == 2
        assert len(a.hooks[HookEvent.STOP]) == 1

    def test_parse_hooks_json_nonexistent(self, tmp_path: Path) -> None:
        config = _parse_hooks_json(tmp_path / "nope.json")
        assert config.hooks == {}

    def test_parse_hooks_json_env_substitution(self, tmp_path: Path) -> None:
        hooks_file = tmp_path / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PostToolUse": [
                    {"hooks": [{"type": "command", "command": "${MAIKE_PLUGIN_ROOT}/format.sh"}]},
                ],
            },
        }))
        env = {"MAIKE_PLUGIN_ROOT": "/my/plugin"}
        config = _parse_hooks_json(hooks_file, env=env)
        defs = config.hooks[HookEvent.POST_TOOL_USE]
        assert defs[0].command == "/my/plugin/format.sh"

    def test_load_hook_configs_from_plugin(self, tmp_path: Path) -> None:
        plugin_dir = _make_plugin(tmp_path / "plugins", "fmt-plugin")
        hooks_dir = plugin_dir / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.json").write_text(json.dumps({
            "hooks": {
                "PostToolUse": [
                    {"hooks": [{"type": "command", "command": "echo ok"}]},
                ],
            },
        }))

        manifest = parse_plugin_manifest(plugin_dir)
        assert manifest is not None

        config = load_hook_configs([manifest])
        defs = config.hooks.get(HookEvent.POST_TOOL_USE, [])
        assert len(defs) == 1
        assert defs[0].source_plugin == "fmt-plugin"


# ── Hook Executor Tests ───────────────────────────────────────────────


class TestHookExecutor:
    def test_fire_command_hook(self, tmp_path: Path) -> None:
        from maike.plugins.hook_executor import HookExecutor

        config = HookConfig(hooks={
            HookEvent.POST_TOOL_USE: [
                HookDefinition(
                    event=HookEvent.POST_TOOL_USE,
                    matcher="Write",
                    command="echo 'hook ran'",
                    timeout_s=5,
                ),
            ],
        })

        executor = HookExecutor(config)
        results = asyncio.run(executor.fire(HookEvent.POST_TOOL_USE, tool_name="Write"))
        assert len(results) == 1
        assert results[0].success is True
        assert "hook ran" in results[0].stdout

    def test_fire_no_matching_hooks(self) -> None:
        from maike.plugins.hook_executor import HookExecutor

        config = HookConfig()
        executor = HookExecutor(config)
        results = asyncio.run(executor.fire(HookEvent.SESSION_START))
        assert results == []

    def test_fire_failing_command(self) -> None:
        from maike.plugins.hook_executor import HookExecutor

        config = HookConfig(hooks={
            HookEvent.STOP: [
                HookDefinition(event=HookEvent.STOP, command="exit 1", timeout_s=5),
            ],
        })
        executor = HookExecutor(config)
        results = asyncio.run(executor.fire(HookEvent.STOP))
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].exit_code == 1


# ── Agent Loader Tests ────────────────────────────────────────────────


class TestAgentLoader:
    def test_parse_agent_file(self, tmp_path: Path) -> None:
        agent_file = tmp_path / "reviewer.md"
        agent_file.write_text(
            "---\n"
            "name: security-reviewer\n"
            "description: Reviews code for vulnerabilities\n"
            "model: strong\n"
            "maxTurns: 15\n"
            "disallowedTools: Write, Edit\n"
            "---\n"
            "You are a security review agent.\n"
        )

        agent = parse_agent_file(agent_file, namespace="my-plugin")
        assert agent is not None
        assert agent.name == "security-reviewer"
        assert agent.description == "Reviews code for vulnerabilities"
        assert agent.model_tier == "strong"
        assert agent.max_turns == 15
        assert agent.disallowed_tools == ["Write", "Edit"]
        assert agent.namespace == "my-plugin"
        assert agent.qualified_name == "my-plugin:security-reviewer"
        assert "security review agent" in agent.system_prompt

    def test_parse_agent_file_minimal(self, tmp_path: Path) -> None:
        agent_file = tmp_path / "basic.md"
        agent_file.write_text(
            "---\nname: basic\ndescription: A basic agent\n---\nDo stuff.\n"
        )

        agent = parse_agent_file(agent_file)
        assert agent is not None
        assert agent.name == "basic"
        assert agent.model_tier == "default"
        assert agent.max_turns == 20
        assert agent.namespace is None
        assert agent.qualified_name == "basic"

    def test_parse_agent_file_missing_name(self, tmp_path: Path) -> None:
        agent_file = tmp_path / "bad.md"
        agent_file.write_text("---\ndescription: no name\n---\nContent.\n")

        assert parse_agent_file(agent_file) is None

    def test_load_plugin_agents(self, tmp_path: Path) -> None:
        plugin_dir = _make_plugin(tmp_path / "plugins", "test-plugin")
        agents_dir = plugin_dir / "agents"
        agents_dir.mkdir()
        (agents_dir / "alpha.md").write_text(
            "---\nname: alpha\ndescription: Agent A\n---\nPrompt A.\n"
        )
        (agents_dir / "beta.md").write_text(
            "---\nname: beta\ndescription: Agent B\n---\nPrompt B.\n"
        )

        manifest = parse_plugin_manifest(plugin_dir)
        assert manifest is not None

        agents = load_plugin_agents(manifest)
        assert len(agents) == 2
        names = {a.qualified_name for a in agents}
        assert "test-plugin:alpha" in names
        assert "test-plugin:beta" in names

    def test_load_plugin_agents_no_dir(self, tmp_path: Path) -> None:
        plugin_dir = _make_plugin(tmp_path / "plugins", "no-agents")
        manifest = parse_plugin_manifest(plugin_dir)
        assert manifest is not None
        assert load_plugin_agents(manifest) == []


# ── LSP Config Tests ──────────────────────────────────────────────────


class TestLSPConfig:
    def test_parse_lsp_json(self, tmp_path: Path) -> None:
        lsp_file = tmp_path / ".lsp.json"
        lsp_file.write_text(json.dumps({
            "pyright": {
                "command": "pyright-langserver",
                "args": ["--stdio"],
                "extensionToLanguage": {".py": "python"},
            },
        }))

        configs = _parse_lsp_json(lsp_file, source="project")
        assert len(configs) == 1
        assert configs[0].name == "pyright"
        assert configs[0].command == "pyright-langserver"
        assert configs[0].extension_to_language == {".py": "python"}
        assert configs[0].transport == "stdio"

    def test_parse_lsp_json_with_options(self, tmp_path: Path) -> None:
        lsp_file = tmp_path / ".lsp.json"
        lsp_file.write_text(json.dumps({
            "ts": {
                "command": "typescript-language-server",
                "args": ["--stdio"],
                "extensionToLanguage": {".ts": "typescript", ".tsx": "typescriptreact"},
                "restartOnCrash": True,
                "maxRestarts": 5,
                "startupTimeout": 15000,
            },
        }))

        configs = _parse_lsp_json(lsp_file, source="project")
        assert len(configs) == 1
        assert configs[0].restart_on_crash is True
        assert configs[0].max_restarts == 5
        assert configs[0].startup_timeout_ms == 15000

    def test_parse_lsp_json_nonexistent(self, tmp_path: Path) -> None:
        assert _parse_lsp_json(tmp_path / "nope.json", source="user") == []

    def test_load_lsp_configs_from_plugin(self, tmp_path: Path) -> None:
        plugin_dir = _make_plugin(tmp_path / "plugins", "py-lsp")
        lsp_file = plugin_dir / ".lsp.json"
        lsp_file.write_text(json.dumps({
            "pyright": {
                "command": "pyright-langserver",
                "args": ["--stdio"],
                "extensionToLanguage": {".py": "python"},
            },
        }))

        manifest = parse_plugin_manifest(plugin_dir)
        assert manifest is not None

        configs = load_lsp_configs([manifest])
        assert len(configs) == 1
        assert configs[0].name == "py-lsp:pyright"
        assert configs[0].source == "plugin"


# ── PluginBundle Tests ────────────────────────────────────────────────


class TestPluginBundle:
    def test_empty_bundle(self) -> None:
        bundle = PluginBundle()
        assert bundle.manifests == []
        assert bundle.skills == []
        assert bundle.agents == []
        assert bundle.hooks is None
        assert bundle.mcp_configs == []
        assert bundle.lsp_configs == []


# ── MCP Client Tests (unit — no subprocess) ──────────────────────────


class TestMCPClientProtocol:
    def test_mcp_tool_info_fields(self) -> None:
        from maike.plugins.mcp_client import MCPToolInfo

        info = MCPToolInfo(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        assert info.name == "read_file"
        assert info.description == "Read a file"
        assert "path" in info.input_schema["properties"]

    def test_mcp_server_config_fields(self) -> None:
        config = MCPServerConfig(
            name="test",
            command="echo",
            args=("hello",),
            env={"KEY": "val"},
            source="project",
        )
        assert config.name == "test"
        assert config.command == "echo"
        assert config.args == ("hello",)

    def test_extract_text_content(self) -> None:
        from maike.plugins.mcp_registry import _extract_text_content

        raw = {"content": [{"type": "text", "text": "Hello"}]}
        assert _extract_text_content(raw) == "Hello"

    def test_extract_text_content_multiple(self) -> None:
        from maike.plugins.mcp_registry import _extract_text_content

        raw = {"content": [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]}
        assert _extract_text_content(raw) == "Line 1\nLine 2"

    def test_extract_text_content_fallback(self) -> None:
        from maike.plugins.mcp_registry import _extract_text_content

        raw = {"some_key": "value"}
        result = _extract_text_content(raw)
        assert "some_key" in result  # JSON stringified


# ── LSP Diagnostic Tests ──────────────────────────────────────────────


class TestLSPDiagnostic:
    def test_diagnostic_str(self) -> None:
        from maike.plugins.lsp_client import Diagnostic

        d = Diagnostic(
            file_path="/src/main.py",
            line=10,
            character=5,
            severity=1,
            message="Undefined variable 'x'",
            source="pyright",
        )
        assert str(d) == "/src/main.py:11:6: error: Undefined variable 'x'"

    def test_diagnostic_severity_labels(self) -> None:
        from maike.plugins.lsp_client import Diagnostic

        assert Diagnostic("f", 0, 0, 1, "").severity_label == "error"
        assert Diagnostic("f", 0, 0, 2, "").severity_label == "warning"
        assert Diagnostic("f", 0, 0, 3, "").severity_label == "info"
        assert Diagnostic("f", 0, 0, 4, "").severity_label == "hint"


# ── LSP Manager Tests ────────────────────────────────────────────────


class TestLSPManager:
    def test_format_diagnostics_none(self) -> None:
        from maike.plugins.lsp_manager import LSPManager

        mgr = LSPManager()
        assert mgr.format_diagnostics_for_agent("/some/file.py") is None

    def test_supported_extensions_empty(self) -> None:
        from maike.plugins.lsp_manager import LSPManager

        mgr = LSPManager()
        assert mgr.supported_extensions == []
        assert mgr.server_names == []


# ── CLI Plugin Tests ──────────────────────────────────────────────────


class TestPluginCLI:
    def test_plugin_list_no_plugins(self, tmp_path: Path, capsys, monkeypatch) -> None:
        """maike plugin list with no plugins prints a message."""
        monkeypatch.setattr("maike.constants.PLUGIN_USER_DIR", tmp_path / "nonexistent")
        from maike.cli import _invoke_plugin_list
        _invoke_plugin_list(tmp_path)
        captured = capsys.readouterr()
        assert "No plugins" in captured.out

    def test_plugin_validate_valid(self, tmp_path: Path, capsys) -> None:
        plugin_dir = _make_plugin(tmp_path, "valid-plugin")
        from maike.cli import _invoke_plugin_validate
        _invoke_plugin_validate(plugin_dir)
        captured = capsys.readouterr()
        assert "valid-plugin" in captured.out
        assert "valid" in captured.out.lower()

    def test_plugin_validate_invalid(self, tmp_path: Path) -> None:
        import pytest
        from maike.cli import _invoke_plugin_validate
        with pytest.raises(SystemExit):
            _invoke_plugin_validate(tmp_path)
