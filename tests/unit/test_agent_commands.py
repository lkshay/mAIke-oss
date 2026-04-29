"""Tests for agent commands — shared logic used by REPL and TUI."""

from __future__ import annotations

from pathlib import Path

from maike.agents.agent_commands import (
    AgentWizardData,
    create_agent_file,
    create_agent_file_v2,
    format_agent_list,
    preview_agent_markdown,
    sanitize_agent_name,
)
from maike.agents.agent_resolver import AgentResolver


# ---------------------------------------------------------------------------
# sanitize_agent_name
# ---------------------------------------------------------------------------


def test_sanitize_basic():
    assert sanitize_agent_name("code-reviewer") == "code-reviewer"


def test_sanitize_uppercase():
    assert sanitize_agent_name("Code Reviewer") == "code-reviewer"


def test_sanitize_special_chars():
    assert sanitize_agent_name("my_agent@2.0!") == "my-agent-2-0"


def test_sanitize_empty_fallback():
    assert sanitize_agent_name("!!!") == "my-agent"


def test_sanitize_collapses_hyphens():
    assert sanitize_agent_name("a---b") == "a-b"


# ---------------------------------------------------------------------------
# format_agent_list
# ---------------------------------------------------------------------------


def test_format_agent_list_empty(tmp_path: Path):
    resolver = AgentResolver(user_dir=tmp_path / "no-such-dir")
    lines = format_agent_list(resolver)
    assert "No custom agents found" in lines[0]


def test_format_agent_list_with_agents(tmp_path: Path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews code\nmodel: default\n---\nYou review code.\n"
    )
    resolver = AgentResolver(project_dir=agents_dir)
    lines = format_agent_list(resolver)
    assert any("reviewer" in line for line in lines)
    assert any("(project)" in line for line in lines)


# ---------------------------------------------------------------------------
# create_agent_file (backward-compatible wrapper)
# ---------------------------------------------------------------------------


def test_create_agent_file_project_scope(tmp_path: Path):
    path = create_agent_file(
        name="test-agent",
        description="A test agent",
        model_tier="strong",
        tools="Read, Grep",
        max_turns=20,
        scope="project",
        workspace=tmp_path,
    )
    assert path.exists()
    assert path.name == "test-agent.md"
    content = path.read_text()
    assert "name: test-agent" in content
    assert "description: A test agent" in content
    assert "model: strong" in content
    assert "tools: Read, Grep" in content
    assert "maxTurns: 20" in content


def test_create_agent_file_user_scope(tmp_path: Path, monkeypatch):
    import maike.agents.agent_commands as mod
    monkeypatch.setattr(mod, "AGENTS_USER_DIR", tmp_path / "user-agents")

    path = create_agent_file(
        name="My Agent",
        description="User-scoped agent",
        scope="user",
    )
    assert path.exists()
    assert "my-agent" in path.name
    assert "User-scoped agent" in path.read_text()


def test_create_agent_file_name_sanitized(tmp_path: Path):
    path = create_agent_file(
        name="Bad Name!!",
        description="Testing sanitization",
        scope="project",
        workspace=tmp_path,
    )
    assert path.name == "bad-name.md"


# ---------------------------------------------------------------------------
# AgentWizardData + preview_agent_markdown
# ---------------------------------------------------------------------------


def test_preview_minimal():
    """Minimal data produces valid frontmatter with placeholder body."""
    data = AgentWizardData(name="my-agent", description="A test agent")
    md = preview_agent_markdown(data)
    assert "---" in md
    assert "name: my-agent" in md
    assert "description: A test agent" in md
    assert "model: default" in md
    assert "maxTurns: 30" in md
    # No tools field when allowed_tools is None.
    assert "tools:" not in md
    # Placeholder body when no system_prompt.
    assert "Guidelines" in md


def test_preview_full():
    """Full data emits all optional fields."""
    data = AgentWizardData(
        name="reviewer",
        description="Reviews code for security",
        system_prompt="You are a security reviewer.\nFocus on OWASP top 10.",
        model_tier="strong",
        max_turns=20,
        allowed_tools=["Read", "Grep", "Bash"],
        skills=["test-methodology"],
        initial_prompt="Read SECURITY.md first",
        background=True,
        critical_reminder="Never approve code with SQL injection",
    )
    md = preview_agent_markdown(data)
    assert "model: strong" in md
    assert "maxTurns: 20" in md
    assert "tools: Read, Grep, Bash" in md
    assert "skills: test-methodology" in md
    assert "initialPrompt: Read SECURITY.md first" in md
    assert "background: true" in md
    assert "critical_reminder: Never approve code with SQL injection" in md
    # Body is the system prompt, not the description.
    assert "You are a security reviewer." in md
    assert "OWASP top 10" in md
    # Placeholder should NOT appear when system_prompt is set.
    assert "Guidelines" not in md


def test_preview_disallowed_tools():
    data = AgentWizardData(
        name="safe-agent",
        description="Agent with restrictions",
        disallowed_tools=["Write", "Bash"],
    )
    md = preview_agent_markdown(data)
    assert "disallowedTools: Write, Bash" in md
    # No tools: line since allowed_tools is None.
    assert "\ntools:" not in md


def test_preview_description_with_special_chars():
    data = AgentWizardData(
        name="quoter",
        description="Agent for C++ and .NET projects",
    )
    md = preview_agent_markdown(data)
    assert "description: Agent for C++ and .NET projects" in md


def test_preview_empty_optional_fields_not_emitted():
    """When optional fields are at defaults, they should not appear."""
    data = AgentWizardData(name="basic", description="Basic agent")
    md = preview_agent_markdown(data)
    assert "disallowedTools" not in md
    assert "skills:" not in md
    assert "initialPrompt" not in md
    assert "background:" not in md
    assert "critical_reminder:" not in md


# ---------------------------------------------------------------------------
# create_agent_file_v2
# ---------------------------------------------------------------------------


def test_create_agent_file_v2_writes_preview(tmp_path: Path):
    """create_agent_file_v2 writes exactly what preview_agent_markdown returns."""
    data = AgentWizardData(
        name="reviewer",
        description="Code reviewer",
        system_prompt="You review code.",
        model_tier="strong",
        max_turns=20,
        allowed_tools=["Read", "Grep"],
    )
    path = create_agent_file_v2(data, workspace=tmp_path)
    assert path.exists()
    content = path.read_text()
    expected = preview_agent_markdown(data)
    assert content == expected


def test_create_agent_file_v2_parseable_by_resolver(tmp_path: Path):
    """Written agent file is correctly parsed by AgentResolver."""
    data = AgentWizardData(
        name="test-agent",
        description="An agent for testing",
        system_prompt="You are a test agent.",
        model_tier="default",
        max_turns=25,
        allowed_tools=["Read", "Grep", "Bash"],
    )
    agents_dir = tmp_path / ".maike" / "agents"
    data.scope = "project"
    path = create_agent_file_v2(data, workspace=tmp_path)

    resolver = AgentResolver(project_dir=agents_dir)
    defn = resolver.resolve("test-agent")
    assert defn is not None
    assert defn.name == "test-agent"
    assert defn.description == "An agent for testing"
    assert defn.model_tier == "default"
    assert defn.allowed_tools == ["Read", "Grep", "Bash"]
    # System prompt is the markdown body.
    assert "You are a test agent." in defn.system_prompt


# ---------------------------------------------------------------------------
# AgentResolver.build_catalog / reload
# ---------------------------------------------------------------------------


def test_build_catalog_empty(tmp_path: Path):
    resolver = AgentResolver(user_dir=tmp_path / "no-such-dir")
    assert resolver.build_catalog() == ""


def test_build_catalog_with_agents(tmp_path: Path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews code\nmodel: default\n---\nYou review code.\n"
    )
    resolver = AgentResolver(project_dir=agents_dir)
    catalog = resolver.build_catalog()
    assert "reviewer" in catalog
    assert "Reviews code" in catalog


def test_build_catalog_respects_cap(tmp_path: Path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for i in range(10):
        (agents_dir / f"agent-{i}.md").write_text(
            f"---\nname: agent-{i}\ndescription: Agent number {i} with a long description\nmodel: default\n---\nPrompt.\n"
        )
    resolver = AgentResolver(project_dir=agents_dir)
    catalog = resolver.build_catalog(cap=100)
    assert len(catalog) <= 100


def test_reload_picks_up_new_files(tmp_path: Path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    resolver = AgentResolver(project_dir=agents_dir)
    assert len(resolver.list_available()) == 0

    (agents_dir / "new-agent.md").write_text(
        "---\nname: new-agent\ndescription: Fresh agent\nmodel: default\n---\nHello.\n"
    )
    resolver.reload()
    assert len(resolver.list_available()) == 1
    assert resolver.resolve("new-agent") is not None
