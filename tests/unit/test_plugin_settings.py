"""Tests for maike.plugins.settings — plugin settings persistence."""
from __future__ import annotations

import json
from pathlib import Path

from maike.plugins.settings import (
    InstalledPluginRecord,
    PluginSettings,
    is_plugin_disabled,
    load_settings,
    save_settings,
)


def test_load_settings_missing_file(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "nonexistent.json")
    assert settings.disabled == set()
    assert settings.installed == {}
    assert settings.config == {}


def test_load_settings_empty_json(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    settings = load_settings(path)
    assert settings.disabled == set()


def test_load_settings_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{bad json!", encoding="utf-8")
    settings = load_settings(path)
    assert settings.disabled == set()


def test_roundtrip_save_load(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = PluginSettings(
        disabled={"plugin-a", "plugin-b"},
        installed={
            "plugin-a": InstalledPluginRecord(
                source="https://github.com/user/a.git",
                version="1.0.0",
                installed_at="/home/user/plugins/a",
                installed_on="2026-03-30T12:00:00Z",
            ),
        },
        config={"plugin-a": {"key": "value"}},
    )
    save_settings(original, path)

    loaded = load_settings(path)
    assert loaded.disabled == {"plugin-a", "plugin-b"}
    assert "plugin-a" in loaded.installed
    assert loaded.installed["plugin-a"].source == "https://github.com/user/a.git"
    assert loaded.installed["plugin-a"].version == "1.0.0"
    assert loaded.config["plugin-a"]["key"] == "value"


def test_save_creates_directory(tmp_path: Path) -> None:
    path = tmp_path / "subdir" / "settings.json"
    save_settings(PluginSettings(), path)
    assert path.is_file()


def test_save_preserves_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"other_section": {"foo": "bar"}}), encoding="utf-8")

    save_settings(PluginSettings(disabled={"x"}), path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["other_section"]["foo"] == "bar"
    assert "x" in raw["plugins"]["disabled"]


def test_is_plugin_disabled(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_settings(PluginSettings(disabled={"disabled-plugin"}), path)

    assert is_plugin_disabled("disabled-plugin", path) is True
    assert is_plugin_disabled("enabled-plugin", path) is False


def test_disabled_list_sorted_on_save(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_settings(PluginSettings(disabled={"z-plugin", "a-plugin", "m-plugin"}), path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["plugins"]["disabled"] == ["a-plugin", "m-plugin", "z-plugin"]
