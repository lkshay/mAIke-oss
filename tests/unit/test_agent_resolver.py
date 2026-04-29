"""Tests for maike.agents.agent_resolver — custom agent definition and resolution."""

from pathlib import Path

from maike.agents.agent_resolver import (
    AgentDefinition,
    AgentResolver,
    _load_agents_from_dir,
    _parse_extended_fields,
)
from maike.plugins.agent_loader import PluginAgent


def _write_agent(tmp_path: Path, name: str, description: str, body: str, **extra) -> Path:
    """Helper: write an agent markdown file."""
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    for key, value in extra.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    path = tmp_path / f"{name}.md"
    path.write_text("\n".join(lines))
    return path


class TestAgentDefinitionFromPluginAgent:
    def test_converts_basic_fields(self):
        pa = PluginAgent(
            name="reviewer",
            description="Code reviewer",
            system_prompt="You are a reviewer.",
            namespace="quality",
        )
        defn = AgentDefinition.from_plugin_agent(pa)
        assert defn.name == "quality:reviewer"
        assert defn.description == "Code reviewer"
        assert defn.system_prompt == "You are a reviewer."
        assert defn.source == "plugin:quality"

    def test_preserves_tool_restrictions(self):
        pa = PluginAgent(
            name="safe",
            description="Safe agent",
            system_prompt="You are safe.",
            allowed_tools=["Read", "Grep"],
            disallowed_tools=["Bash"],
        )
        defn = AgentDefinition.from_plugin_agent(pa)
        assert defn.allowed_tools == ["Read", "Grep"]
        assert defn.disallowed_tools == ["Bash"]

    def test_default_max_turns_becomes_none(self):
        pa = PluginAgent(name="x", description="x", system_prompt="x", max_turns=20)
        defn = AgentDefinition.from_plugin_agent(pa)
        assert defn.max_turns is None  # 20 is default, should be None

    def test_custom_max_turns_preserved(self):
        pa = PluginAgent(name="x", description="x", system_prompt="x", max_turns=40)
        defn = AgentDefinition.from_plugin_agent(pa)
        assert defn.max_turns == 40


class TestLoadAgentsFromDir:
    def test_loads_valid_agent(self, tmp_path):
        _write_agent(tmp_path, "my-agent", "My agent", "You are my agent.")
        agents = _load_agents_from_dir(tmp_path, "project")
        assert len(agents) == 1
        assert agents[0].name == "my-agent"
        assert agents[0].source == "project"
        assert "You are my agent." in agents[0].system_prompt

    def test_skips_invalid_file(self, tmp_path):
        (tmp_path / "bad.md").write_text("No frontmatter here")
        agents = _load_agents_from_dir(tmp_path, "user")
        assert agents == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        agents = _load_agents_from_dir(tmp_path / "nope", "user")
        assert agents == []

    def test_parses_tool_lists(self, tmp_path):
        _write_agent(
            tmp_path, "restricted", "Restricted agent", "Body",
            tools="Read, Grep, Bash",
            disallowedTools="Write, Edit",
        )
        agents = _load_agents_from_dir(tmp_path, "project")
        assert agents[0].allowed_tools == ["Read", "Grep", "Bash"]
        assert agents[0].disallowed_tools == ["Write", "Edit"]

    def test_parses_model_tier(self, tmp_path):
        _write_agent(tmp_path, "strong-agent", "Strong", "Body", model="strong")
        agents = _load_agents_from_dir(tmp_path, "project")
        assert agents[0].model_tier == "strong"

    def test_parses_max_turns(self, tmp_path):
        _write_agent(tmp_path, "long-agent", "Long", "Body", maxTurns="50")
        agents = _load_agents_from_dir(tmp_path, "project")
        assert agents[0].max_turns == 50

    def test_parses_skills(self, tmp_path):
        _write_agent(tmp_path, "skilled", "Skilled", "Body", skills="test-methodology, debugging")
        agents = _load_agents_from_dir(tmp_path, "project")
        assert agents[0].skills == ["test-methodology", "debugging"]


class TestParseExtendedFields:
    def test_parses_initial_prompt(self, tmp_path):
        _write_agent(
            tmp_path, "init", "Init agent", "Body",
            initialPrompt='"Read the README first."',
        )
        result = _parse_extended_fields(tmp_path / "init.md")
        assert result["initialPrompt"] == "Read the README first."

    def test_parses_background(self, tmp_path):
        _write_agent(tmp_path, "bg", "BG agent", "Body", background="true")
        result = _parse_extended_fields(tmp_path / "bg.md")
        assert result["background"] is True

    def test_parses_critical_reminder(self, tmp_path):
        _write_agent(
            tmp_path, "crit", "Crit agent", "Body",
            critical_reminder='"Always verify output."',
        )
        result = _parse_extended_fields(tmp_path / "crit.md")
        assert result["critical_reminder"] == "Always verify output."

    def test_missing_file_returns_empty(self, tmp_path):
        result = _parse_extended_fields(tmp_path / "nonexistent.md")
        assert result == {}


class TestAgentResolver:
    def test_resolves_plugin_agent(self):
        pa = PluginAgent(name="reviewer", description="Review", system_prompt="R", namespace="qa")
        resolver = AgentResolver(plugin_agents=[pa])
        defn = resolver.resolve("qa:reviewer")
        assert defn is not None
        assert defn.name == "qa:reviewer"

    def test_resolves_bare_name(self):
        pa = PluginAgent(name="reviewer", description="Review", system_prompt="R", namespace="qa")
        resolver = AgentResolver(plugin_agents=[pa])
        defn = resolver.resolve("reviewer")
        assert defn is not None

    def test_project_overrides_user(self, tmp_path):
        user_dir = tmp_path / "user"
        project_dir = tmp_path / "project"
        user_dir.mkdir()
        project_dir.mkdir()
        _write_agent(user_dir, "my-agent", "User version", "User body")
        _write_agent(project_dir, "my-agent", "Project version", "Project body")

        resolver = AgentResolver(user_dir=user_dir, project_dir=project_dir)
        defn = resolver.resolve("my-agent")
        assert defn is not None
        assert defn.source == "project"
        assert defn.description == "Project version"

    def test_user_overrides_plugin(self, tmp_path):
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        _write_agent(user_dir, "reviewer", "User reviewer", "User body")
        pa = PluginAgent(name="reviewer", description="Plugin reviewer", system_prompt="P")

        resolver = AgentResolver(plugin_agents=[pa], user_dir=user_dir)
        defn = resolver.resolve("reviewer")
        assert defn.source == "user"

    def test_returns_none_for_unknown(self):
        resolver = AgentResolver()
        assert resolver.resolve("nonexistent") is None

    def test_list_available(self, tmp_path):
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        _write_agent(user_dir, "alpha", "Alpha agent", "A body")
        _write_agent(user_dir, "beta", "Beta agent", "B body")
        pa = PluginAgent(name="gamma", description="Gamma", system_prompt="G", namespace="p")

        resolver = AgentResolver(plugin_agents=[pa], user_dir=user_dir)
        available = resolver.list_available()
        names = [d.name for d in available]
        assert "alpha" in names
        assert "beta" in names
        # gamma is available via both "gamma" and "p:gamma"
        assert any("gamma" in n for n in names)

    def test_list_names(self, tmp_path):
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        _write_agent(user_dir, "alpha", "Alpha", "A")

        resolver = AgentResolver(user_dir=user_dir)
        names = resolver.list_names()
        assert "alpha" in names

    def test_empty_resolver(self):
        resolver = AgentResolver()
        assert resolver.list_available() == []
        assert resolver.list_names() == []
        assert resolver.resolve("anything") is None
