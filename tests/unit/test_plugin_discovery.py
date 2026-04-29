"""Tests for maike.plugins.discovery and maike.plugins.loader."""
from __future__ import annotations

import json
from pathlib import Path

from maike.agents.skill import SkillSource
from maike.plugins.discovery import PluginDiscovery
from maike.plugins.loader import PluginLoader
from maike.plugins.manifest import parse_plugin_manifest


def _make_plugin(
    parent: Path,
    name: str,
    version: str = "1.0.0",
    description: str = "",
) -> Path:
    """Helper: create a minimal plugin directory with manifest."""
    plugin_dir = parent / name
    manifest_dir = plugin_dir / ".maike-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "plugin.json").write_text(
        json.dumps({
            "name": name,
            "description": description,
            "version": version,
        }),
        encoding="utf-8",
    )
    return plugin_dir


def _add_skill(plugin_dir: Path, skill_name: str, description: str = "A skill") -> Path:
    """Helper: add a SKILL.md to a plugin's skills directory."""
    skill_dir = plugin_dir / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: {description}\ntriggers:\n- {skill_name}\n---\nSkill content for {skill_name}.\n",
        encoding="utf-8",
    )
    return skill_dir


# ── PluginDiscovery tests ──────────────────────────────────────────────


def test_discover_finds_plugins(tmp_path: Path) -> None:
    search_dir = tmp_path / "plugins"
    search_dir.mkdir()
    _make_plugin(search_dir, "alpha")
    _make_plugin(search_dir, "beta")

    result = PluginDiscovery.discover([search_dir])

    assert len(result) == 2
    assert result[0].name == "alpha"
    assert result[1].name == "beta"


def test_discover_skips_invalid(tmp_path: Path) -> None:
    search_dir = tmp_path / "plugins"
    search_dir.mkdir()
    _make_plugin(search_dir, "valid-plugin")
    # Create a directory without a manifest
    (search_dir / "not-a-plugin").mkdir()

    result = PluginDiscovery.discover([search_dir])

    assert len(result) == 1
    assert result[0].name == "valid-plugin"


def test_discover_empty_dir(tmp_path: Path) -> None:
    search_dir = tmp_path / "empty"
    search_dir.mkdir()

    assert PluginDiscovery.discover([search_dir]) == []


def test_discover_nonexistent_dir(tmp_path: Path) -> None:
    assert PluginDiscovery.discover([tmp_path / "does-not-exist"]) == []


def test_discover_multiple_search_dirs(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    _make_plugin(dir_a, "plugin-a")

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _make_plugin(dir_b, "plugin-b")

    result = PluginDiscovery.discover([dir_a, dir_b])

    assert len(result) == 2
    names = [m.name for m in result]
    assert "plugin-a" in names
    assert "plugin-b" in names


def test_discover_later_dir_overrides(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    _make_plugin(dir_a, "shared", version="1.0.0")

    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _make_plugin(dir_b, "shared", version="2.0.0")

    result = PluginDiscovery.discover([dir_a, dir_b])

    assert len(result) == 1
    assert result[0].name == "shared"
    assert result[0].version == "2.0.0"
    assert result[0].path == dir_b / "shared"


# ── PluginLoader tests ──────────────────────────────────────────────────


def test_load_plugin_skills_namespaced(tmp_path: Path) -> None:
    search_dir = tmp_path / "plugins"
    search_dir.mkdir()
    plugin_dir = _make_plugin(search_dir, "my-plugin")
    _add_skill(plugin_dir, "format")

    manifest = parse_plugin_manifest(plugin_dir)
    assert manifest is not None

    skills = PluginLoader.load_skills(manifest)

    assert len(skills) == 1
    assert skills[0].name == "my-plugin:format"


def test_load_plugin_skills_source_is_plugin(tmp_path: Path) -> None:
    search_dir = tmp_path / "plugins"
    search_dir.mkdir()
    plugin_dir = _make_plugin(search_dir, "src-test")
    _add_skill(plugin_dir, "lint")

    manifest = parse_plugin_manifest(plugin_dir)
    assert manifest is not None

    skills = PluginLoader.load_skills(manifest)

    assert len(skills) == 1
    assert skills[0].source == SkillSource.PLUGIN
    assert skills[0].namespace == "src-test"


def test_load_plugin_skills_empty_skills_dir(tmp_path: Path) -> None:
    search_dir = tmp_path / "plugins"
    search_dir.mkdir()
    plugin_dir = _make_plugin(search_dir, "empty-skills")
    # No skills/ directory at all

    manifest = parse_plugin_manifest(plugin_dir)
    assert manifest is not None

    assert PluginLoader.load_skills(manifest) == []


def test_load_all_plugin_skills(tmp_path: Path) -> None:
    search_dir = tmp_path / "plugins"
    search_dir.mkdir()

    plugin_a = _make_plugin(search_dir, "plugin-a")
    _add_skill(plugin_a, "skill-one")

    plugin_b = _make_plugin(search_dir, "plugin-b")
    _add_skill(plugin_b, "skill-two")
    _add_skill(plugin_b, "skill-three")

    manifests = PluginDiscovery.discover([search_dir])
    all_skills = PluginLoader.load_all_plugin_skills(manifests)

    assert len(all_skills) == 3
    names = {s.name for s in all_skills}
    assert "plugin-a:skill-one" in names
    assert "plugin-b:skill-two" in names
    assert "plugin-b:skill-three" in names
