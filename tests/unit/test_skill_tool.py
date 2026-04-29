"""Tests for the Skill tool — LLM-driven skill loading."""

import asyncio
from pathlib import Path

from maike.agents.skill import SkillLoader
from maike.tools.registry import ToolRegistry
from maike.tools.skill import register_skill_tool


def _write_skill(
    tmp_path: Path,
    name: str,
    description: str,
    triggers: list[str],
    content: str,
) -> Path:
    """Helper: write a flat skill .md file with frontmatter."""
    trigger_csv = ", ".join(triggers)
    text = (
        f"---\n"
        f"name: {name}\n"
        f'description: "{description}"\n'
        f"metadata:\n"
        f'  triggers: "{trigger_csv}"\n'
        f'  auto_inject: "false"\n'
        f"---\n\n"
        f"{content}\n"
    )
    path = tmp_path / f"{name}.md"
    path.write_text(text)
    return path


def _write_dir_skill(
    tmp_path: Path,
    name: str,
    description: str,
    triggers: list[str],
    content: str,
    references: dict[str, str] | None = None,
) -> Path:
    """Helper: write a directory-based skill with optional reference files."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    trigger_csv = ", ".join(triggers)
    text = (
        f"---\n"
        f"name: {name}\n"
        f'description: "{description}"\n'
        f"metadata:\n"
        f'  triggers: "{trigger_csv}"\n'
        f'  auto_inject: "false"\n'
        f"---\n\n"
        f"{content}\n"
    )
    (skill_dir / "SKILL.md").write_text(text)
    if references:
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        for ref_name, ref_content in references.items():
            (refs_dir / ref_name).write_text(ref_content)
    return skill_dir


def _make_registry(tmp_path: Path, **kwargs) -> tuple[ToolRegistry, SkillLoader]:
    """Build a ToolRegistry with the Skill tool registered."""
    loader = SkillLoader(builtin_dir=tmp_path, **kwargs)
    registry = ToolRegistry()
    register_skill_tool(registry, loader)
    return registry, loader


def test_skill_tool_returns_content(tmp_path):
    """Calling the Skill tool with a valid name returns the skill content."""
    _write_skill(tmp_path, "debugging", "Debug stuff", ["debug"], "Step 1: reproduce.")
    registry, _ = _make_registry(tmp_path)

    tool = registry.get("Skill")
    assert tool is not None
    result = asyncio.run(tool.fn(name="debugging"))
    assert result.success is True
    assert "Step 1: reproduce." in result.output
    assert result.tool_name == "Skill"


def test_skill_tool_unknown_name(tmp_path):
    """Calling with an unknown name returns failure and lists available skills."""
    _write_skill(tmp_path, "debugging", "Debug stuff", ["debug"], "Content.")
    registry, _ = _make_registry(tmp_path)

    tool = registry.get("Skill")
    result = asyncio.run(tool.fn(name="nonexistent"))
    assert result.success is False
    assert "nonexistent" in result.output
    assert "debugging" in result.output  # suggests available skills


def test_skill_tool_missing_name(tmp_path):
    """Calling without a name returns failure."""
    _write_skill(tmp_path, "debugging", "Debug stuff", ["debug"], "Content.")
    registry, _ = _make_registry(tmp_path)

    tool = registry.get("Skill")
    result = asyncio.run(tool.fn())
    assert result.success is False
    assert "debugging" in result.output  # lists available skills


def test_skill_tool_includes_supporting_content(tmp_path):
    """Directory-based skills include reference material in the output."""
    _write_dir_skill(
        tmp_path,
        "debugging",
        "Debug stuff",
        ["debug"],
        "Main debugging content.",
        references={"strategies.md": "# Advanced Strategies\n\nUse binary search."},
    )
    registry, _ = _make_registry(tmp_path)

    tool = registry.get("Skill")
    result = asyncio.run(tool.fn(name="debugging"))
    assert result.success is True
    assert "Main debugging content." in result.output
    assert "Advanced Strategies" in result.output
    assert "references/strategies.md" in result.output


def test_skill_tool_catalog_in_description(tmp_path):
    """The Skill tool description contains the skill catalog."""
    _write_skill(tmp_path, "debugging", "Systematic debugging methodology", ["debug"], "Content.")
    _write_skill(tmp_path, "refactoring", "Safe refactoring patterns", ["refactor"], "Content.")
    registry, _ = _make_registry(tmp_path)

    schemas = registry.get_all_schemas()
    skill_schema = next(s for s in schemas if s["name"] == "Skill")
    assert "debugging" in skill_schema["description"]
    assert "Systematic debugging methodology" in skill_schema["description"]
    assert "refactoring" in skill_schema["description"]
    assert "Safe refactoring patterns" in skill_schema["description"]


def test_skill_tool_excludes_model_disabled(tmp_path):
    """Skills with disable_model_invocation=True are hidden from catalog and loading."""
    # Write a skill with disable_model_invocation via the lower-level frontmatter.
    text = (
        "---\n"
        "name: hidden-skill\n"
        'description: "Should be hidden"\n'
        "disable_model_invocation: true\n"
        "metadata:\n"
        '  triggers: "hidden"\n'
        '  auto_inject: "false"\n'
        "---\n\n"
        "Hidden content.\n"
    )
    (tmp_path / "hidden-skill.md").write_text(text)
    _write_skill(tmp_path, "visible", "Visible skill", ["visible"], "Visible content.")

    registry, _ = _make_registry(tmp_path)

    # Hidden skill should not appear in the catalog/description.
    schemas = registry.get_all_schemas()
    skill_schema = next(s for s in schemas if s["name"] == "Skill")
    assert "hidden-skill" not in skill_schema["description"]
    assert "visible" in skill_schema["description"]

    # Attempting to load the hidden skill should fail.
    tool = registry.get("Skill")
    result = asyncio.run(tool.fn(name="hidden-skill"))
    assert result.success is False


def test_skill_tool_risk_level_is_read(tmp_path):
    """The Skill tool should have READ risk level."""
    from maike.atoms.tool import RiskLevel

    _write_skill(tmp_path, "debugging", "Debug stuff", ["debug"], "Content.")
    registry, _ = _make_registry(tmp_path)

    tool = registry.get("Skill")
    assert tool is not None
    assert tool.risk_level == RiskLevel.READ
