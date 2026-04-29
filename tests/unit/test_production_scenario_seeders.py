"""Tests for production scenario seeders — verify files are created correctly."""

from __future__ import annotations

import shutil

import pytest

from maike.smoke.production_scenarios import PRODUCTION_SCENARIOS


class TestScenariosRegistered:
    def test_import_cycle_registered(self):
        assert "debugging-import-cycle" in PRODUCTION_SCENARIOS

    def test_plugin_system_registered(self):
        assert "editing-plugin-system" in PRODUCTION_SCENARIOS

    def test_js_parser_registered(self):
        assert "debugging-js-parser" in PRODUCTION_SCENARIOS


class TestImportCycleSeeder:
    def test_creates_expected_files(self, tmp_path):
        scenario = PRODUCTION_SCENARIOS["debugging-import-cycle"]
        scenario.setup_workspace(tmp_path)

        assert (tmp_path / "registry.py").exists()
        assert (tmp_path / "plugin_a.py").exists()
        assert (tmp_path / "plugin_b.py").exists()
        assert (tmp_path / "plugin_c.py").exists()
        assert (tmp_path / "test_registry.py").exists()

    def test_files_have_content(self, tmp_path):
        scenario = PRODUCTION_SCENARIOS["debugging-import-cycle"]
        scenario.setup_workspace(tmp_path)

        registry_content = (tmp_path / "registry.py").read_text()
        assert "def register" in registry_content
        assert "def all_names" in registry_content


class TestPluginSystemSeeder:
    def test_creates_expected_files(self, tmp_path):
        scenario = PRODUCTION_SCENARIOS["editing-plugin-system"]
        scenario.setup_workspace(tmp_path)

        assert (tmp_path / "plugin_loader.py").exists()
        assert (tmp_path / "test_plugin_loader.py").exists()

    def test_loader_has_load_all(self, tmp_path):
        scenario = PRODUCTION_SCENARIOS["editing-plugin-system"]
        scenario.setup_workspace(tmp_path)

        content = (tmp_path / "plugin_loader.py").read_text()
        assert "def load_all" in content
        assert "class PluginRegistry" in content

    def test_tests_include_dependency_ordering(self, tmp_path):
        scenario = PRODUCTION_SCENARIOS["editing-plugin-system"]
        scenario.setup_workspace(tmp_path)

        content = (tmp_path / "test_plugin_loader.py").read_text()
        assert "test_load_all_respects_dependency_order" in content
        assert "test_circular_dependency_detected" in content
        assert "test_diamond_dependency" in content


class TestJsParserSeeder:
    def test_creates_expected_files(self, tmp_path):
        scenario = PRODUCTION_SCENARIOS["debugging-js-parser"]
        scenario.setup_workspace(tmp_path)

        assert (tmp_path / "package.json").exists()
        assert (tmp_path / "parser.js").exists()
        assert (tmp_path / "parser.test.js").exists()

    def test_package_json_has_vitest(self, tmp_path):
        import json

        scenario = PRODUCTION_SCENARIOS["debugging-js-parser"]
        scenario.setup_workspace(tmp_path)

        pkg = json.loads((tmp_path / "package.json").read_text())
        assert "vitest" in pkg.get("devDependencies", {})
        assert pkg["scripts"]["test"] == "npx vitest run"

    def test_parser_has_bug(self, tmp_path):
        """The parser should have the escape sequence bug (\\n not handled)."""
        scenario = PRODUCTION_SCENARIOS["debugging-js-parser"]
        scenario.setup_workspace(tmp_path)

        content = (tmp_path / "parser.js").read_text()
        # The bug: no explicit case for 'n' or 't' — falls through to default
        assert "else result += ch;" in content

    @pytest.mark.skipif(not shutil.which("node"), reason="Node.js not available")
    def test_scenario_language_is_node(self):
        scenario = PRODUCTION_SCENARIOS["debugging-js-parser"]
        assert scenario.language == "node"
