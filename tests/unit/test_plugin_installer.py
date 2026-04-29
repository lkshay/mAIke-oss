"""Tests for maike.plugins.installer — plugin install/update."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from maike.plugins.installer import (
    PluginInstallError,
    _is_git_url,
    describe_plugin,
    install_plugin,
    install_skill,
    resolve_install_dir,
    uninstall_plugin,
    update_plugin,
)


def _make_source_plugin(parent: Path, name: str = "test-plugin") -> Path:
    """Create a minimal valid plugin source directory."""
    plugin_dir = parent / name
    manifest_dir = plugin_dir / ".maike-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "plugin.json").write_text(
        json.dumps({"name": name, "version": "1.0.0", "description": "A test"}),
        encoding="utf-8",
    )
    return plugin_dir


class TestIsGitUrl:
    def test_https_url(self) -> None:
        assert _is_git_url("https://github.com/user/repo.git") is True

    def test_ssh_url(self) -> None:
        assert _is_git_url("git@github.com:user/repo.git") is True

    def test_local_path(self) -> None:
        assert _is_git_url("/home/user/plugins/my-plugin") is False

    def test_relative_path(self) -> None:
        assert _is_git_url("./plugins/my-plugin") is False

    def test_git_protocol(self) -> None:
        assert _is_git_url("git://github.com/user/repo.git") is True


class TestResolveInstallDir:
    def test_user_scope_default(self) -> None:
        result = resolve_install_dir(None, "user")
        assert str(result).endswith(".config/maike/plugins")

    def test_project_scope(self) -> None:
        result = resolve_install_dir(None, "project")
        assert str(result).endswith(".maike/plugins")

    def test_custom_dir_treated_as_path(self) -> None:
        result = resolve_install_dir("/opt/plugins", "user")
        assert str(result) == "/opt/plugins"

    def test_custom_path(self) -> None:
        result = resolve_install_dir("/custom/path", "user")
        assert str(result) == "/custom/path"


class TestInstallLocal:
    def test_install_local_directory(self, tmp_path: Path, monkeypatch) -> None:
        source = _make_source_plugin(tmp_path / "source")
        target = tmp_path / "target"
        target.mkdir()

        # Avoid writing to real settings
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        manifest = install_plugin(str(source), target)
        assert manifest.name == "test-plugin"
        assert (target / "test-plugin" / ".maike-plugin" / "plugin.json").is_file()

        # Check settings recorded
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "test-plugin" in raw["plugins"]["installed"]

    def test_install_local_invalid_source(self, tmp_path: Path) -> None:
        with pytest.raises(PluginInstallError, match="not a directory"):
            install_plugin(str(tmp_path / "nonexistent"), tmp_path)

    def test_install_local_no_manifest(self, tmp_path: Path) -> None:
        source = tmp_path / "no-manifest"
        source.mkdir()
        with pytest.raises(PluginInstallError, match="No valid"):
            install_plugin(str(source), tmp_path)

    def test_install_local_already_exists(self, tmp_path: Path) -> None:
        source = _make_source_plugin(tmp_path / "source")
        target = tmp_path / "target"
        (target / "test-plugin").mkdir(parents=True)

        with pytest.raises(PluginInstallError, match="already exists"):
            install_plugin(str(source), target)

    def test_install_local_force_overwrite(self, tmp_path: Path, monkeypatch) -> None:
        source = _make_source_plugin(tmp_path / "source")
        target = tmp_path / "target"
        (target / "test-plugin").mkdir(parents=True)

        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        manifest = install_plugin(str(source), target, force=True)
        assert manifest.name == "test-plugin"


class TestInstallGit:
    def test_install_git_checks_git_available(self, tmp_path: Path) -> None:
        with patch("maike.plugins.installer.shutil.which", return_value=None):
            with pytest.raises(PluginInstallError, match="git is not installed"):
                install_plugin("https://github.com/user/repo.git", tmp_path)


class TestUpdatePlugin:
    def test_update_not_installed(self, tmp_path: Path, monkeypatch) -> None:
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        with pytest.raises(PluginInstallError, match="not recorded"):
            update_plugin("nonexistent")

    def test_update_local_returns_none(self, tmp_path: Path, monkeypatch) -> None:
        from maike.plugins.settings import InstalledPluginRecord, PluginSettings, save_settings

        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        save_settings(PluginSettings(installed={
            "local-plugin": InstalledPluginRecord(
                source="/home/user/plugins/local",
                version="1.0.0",
                installed_at=str(tmp_path / "local-plugin"),
                installed_on="2026-03-30T12:00:00Z",
            ),
        }), settings_path)

        assert update_plugin("local-plugin") is None


class TestUninstallPlugin:
    def test_uninstall_removes_directory(self, tmp_path: Path, monkeypatch) -> None:
        """Uninstalling removes the plugin directory."""
        source = _make_source_plugin(tmp_path / "source")
        target = tmp_path / "target"
        target.mkdir()
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        manifest = install_plugin(str(source), target)
        assert (target / "test-plugin").is_dir()

        uninstall_plugin("test-plugin")
        assert not (target / "test-plugin").exists()

    def test_uninstall_clears_settings(self, tmp_path: Path, monkeypatch) -> None:
        """Uninstalling removes the plugin from settings."""
        from maike.plugins.settings import load_settings

        source = _make_source_plugin(tmp_path / "source")
        target = tmp_path / "target"
        target.mkdir()
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        install_plugin(str(source), target)
        uninstall_plugin("test-plugin")

        settings = load_settings(settings_path)
        assert "test-plugin" not in settings.installed
        assert "test-plugin" not in settings.disabled

    def test_uninstall_not_installed(self, tmp_path: Path, monkeypatch) -> None:
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        with pytest.raises(PluginInstallError, match="not installed"):
            uninstall_plugin("nonexistent")

    def test_uninstall_with_remove_data(self, tmp_path: Path, monkeypatch) -> None:
        """remove_data=True removes the data directory too."""
        source = _make_source_plugin(tmp_path / "source")
        target = tmp_path / "target"
        target.mkdir()
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("maike.constants.SETTINGS_PATH", settings_path)

        install_plugin(str(source), target)

        # Create fake data dir
        data_dir = Path.home() / ".config" / "maike" / "plugins" / "data" / "test-plugin"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "state.json").write_text("{}")

        try:
            uninstall_plugin("test-plugin", remove_data=True)
            assert not data_dir.exists()
        finally:
            # Cleanup in case test fails
            if data_dir.exists():
                import shutil
                shutil.rmtree(data_dir)


class TestInstallSkill:
    def _make_skill_dir(self, parent: Path, name: str = "my-skill") -> Path:
        """Create a minimal valid skill directory with SKILL.md."""
        skill_dir = parent / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: A test skill\n---\n\nSkill content here.\n"
        )
        return skill_dir

    def test_install_single_skill(self, tmp_path: Path, monkeypatch) -> None:
        source = self._make_skill_dir(tmp_path / "source")
        target = tmp_path / "skills"
        target.mkdir()
        monkeypatch.setattr("maike.constants.SKILL_USER_DIR", target)

        names = install_skill(str(source))
        assert names == ["my-skill"]
        assert (target / "my-skill" / "SKILL.md").is_file()

    def test_install_multi_skills(self, tmp_path: Path, monkeypatch) -> None:
        source = tmp_path / "source"
        skills_dir = source / "skills"
        skills_dir.mkdir(parents=True)
        self._make_skill_dir(skills_dir, "skill-a")
        self._make_skill_dir(skills_dir, "skill-b")
        target = tmp_path / "skills"
        target.mkdir()
        monkeypatch.setattr("maike.constants.SKILL_USER_DIR", target)

        names = install_skill(str(source))
        assert sorted(names) == ["skill-a", "skill-b"]
        assert (target / "skill-a" / "SKILL.md").is_file()
        assert (target / "skill-b" / "SKILL.md").is_file()

    def test_install_no_skill_found(self, tmp_path: Path, monkeypatch) -> None:
        source = tmp_path / "empty"
        source.mkdir()
        monkeypatch.setattr("maike.constants.SKILL_USER_DIR", tmp_path / "target")

        with pytest.raises(PluginInstallError, match="No SKILL.md found"):
            install_skill(str(source))

    def test_install_already_exists(self, tmp_path: Path, monkeypatch) -> None:
        source = self._make_skill_dir(tmp_path / "source")
        target = tmp_path / "skills"
        (target / "my-skill").mkdir(parents=True)
        monkeypatch.setattr("maike.constants.SKILL_USER_DIR", target)

        with pytest.raises(PluginInstallError, match="already exists"):
            install_skill(str(source))

    def test_install_force_overwrite(self, tmp_path: Path, monkeypatch) -> None:
        source = self._make_skill_dir(tmp_path / "source")
        target = tmp_path / "skills"
        (target / "my-skill").mkdir(parents=True)
        monkeypatch.setattr("maike.constants.SKILL_USER_DIR", target)

        names = install_skill(str(source), force=True)
        assert names == ["my-skill"]

    def test_install_project_scope(self, tmp_path: Path, monkeypatch) -> None:
        source = self._make_skill_dir(tmp_path / "source")
        monkeypatch.chdir(tmp_path)

        names = install_skill(str(source), scope="project")
        assert names == ["my-skill"]
        assert (tmp_path / ".maike" / "skills" / "my-skill" / "SKILL.md").is_file()

    def test_install_git_checks_git_available(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("maike.constants.SKILL_USER_DIR", tmp_path)
        with patch("maike.plugins.installer.shutil.which", return_value=None):
            with pytest.raises(PluginInstallError, match="git is not installed"):
                install_skill("https://github.com/user/skill-repo.git")

    def test_name_from_frontmatter(self, tmp_path: Path, monkeypatch) -> None:
        """Skill name is extracted from SKILL.md frontmatter, not directory name."""
        skill_dir = tmp_path / "source" / "some-random-dirname"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: proper-name\ndescription: Test\n---\n\nContent\n"
        )
        target = tmp_path / "skills"
        target.mkdir()
        monkeypatch.setattr("maike.constants.SKILL_USER_DIR", target)

        names = install_skill(str(skill_dir))
        assert names == ["proper-name"]
        assert (target / "proper-name" / "SKILL.md").is_file()


class TestDescribePlugin:
    def test_describe_with_skills(self, tmp_path: Path) -> None:
        source = _make_source_plugin(tmp_path)
        skills_dir = source / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: X\n---\nBody")

        from maike.plugins.manifest import parse_plugin_manifest
        manifest = parse_plugin_manifest(source)
        desc = describe_plugin(manifest)
        assert "Skills: 1" in desc

    def test_describe_empty_plugin(self, tmp_path: Path) -> None:
        source = _make_source_plugin(tmp_path)
        from maike.plugins.manifest import parse_plugin_manifest
        manifest = parse_plugin_manifest(source)
        desc = describe_plugin(manifest)
        assert "no components" in desc
