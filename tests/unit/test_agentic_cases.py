"""Tests for agentic eval case seeders and verifiers."""

import subprocess
import sys

import pytest

from maike.smoke.workflow_cases.agentic_cases import (
    _seed_cross_module_debug,
    _seed_feature_addition,
    _seed_refactor_preservation,
    _verify_cross_module_debug,
    _verify_feature_addition,
    _verify_refactor_preservation,
    AGENTIC_EVAL_CASES,
)


class TestCaseRegistry:
    def test_all_cases_registered(self):
        assert "agentic-cross-module-debug" in AGENTIC_EVAL_CASES
        assert "agentic-feature-addition" in AGENTIC_EVAL_CASES
        assert "agentic-refactor-preservation" in AGENTIC_EVAL_CASES

    def test_all_cases_tagged_agentic(self):
        for name, case in AGENTIC_EVAL_CASES.items():
            assert "agentic" in case.tags, f"{name} missing 'agentic' tag"

    def test_difficulty_weights_set(self):
        for name, case in AGENTIC_EVAL_CASES.items():
            assert case.difficulty_weight > 0, f"{name} has no difficulty_weight"

    def test_expected_modified_files_set(self):
        for name, case in AGENTIC_EVAL_CASES.items():
            assert case.expected_modified_files, f"{name} has no expected_modified_files"


class TestCase1CrossModuleDebug:
    def test_seeder_creates_all_files(self, tmp_path):
        _seed_cross_module_debug(tmp_path)
        for name in ("models.py", "validators.py", "database.py", "api.py", "test_api.py", "conftest.py"):
            assert (tmp_path / name).exists(), f"Missing {name}"

    def test_seeded_tests_fail(self, tmp_path):
        _seed_cross_module_debug(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert result.returncode != 0, "Seeded workspace should have failing tests"

    def test_verifier_rejects_buggy_workspace(self, tmp_path):
        _seed_cross_module_debug(tmp_path)
        with pytest.raises((AssertionError, Exception)):
            _verify_cross_module_debug(tmp_path)

    def test_verifier_accepts_fixed_workspace(self, tmp_path):
        _seed_cross_module_debug(tmp_path)
        # Apply the fix: change >= to >
        source = (tmp_path / "validators.py").read_text()
        source = source.replace("len(name) >= max_length", "len(name) > max_length")
        (tmp_path / "validators.py").write_text(source)
        _verify_cross_module_debug(tmp_path)


class TestCase2FeatureAddition:
    def test_seeder_creates_all_files(self, tmp_path):
        _seed_feature_addition(tmp_path)
        for name in ("report.py", "formatters.py", "cli.py", "test_report.py", "README.md"):
            assert (tmp_path / name).exists(), f"Missing {name}"

    def test_seeded_tests_pass(self, tmp_path):
        """Existing tests should pass before adding the feature."""
        _seed_feature_addition(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Seeded tests should pass: {result.stdout}"

    def test_verifier_rejects_unfeatured_workspace(self, tmp_path):
        """Verifier should fail because 'json' formatter doesn't exist yet."""
        _seed_feature_addition(tmp_path)
        with pytest.raises((AssertionError, KeyError, Exception)):
            _verify_feature_addition(tmp_path)


class TestCase3RefactorPreservation:
    def test_seeder_creates_all_files(self, tmp_path):
        _seed_refactor_preservation(tmp_path)
        for name in ("utils.py", "processor.py", "exporter.py", "analyzer.py", "pipeline.py", "test_utils.py", "test_pipeline.py"):
            assert (tmp_path / name).exists(), f"Missing {name}"

    def test_seeded_tests_pass(self, tmp_path):
        """Existing tests should pass before refactoring."""
        _seed_refactor_preservation(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Seeded tests should pass: {result.stdout}"

    def test_verifier_rejects_unrefactored(self, tmp_path):
        """Verifier should fail because format_output doesn't require encoding yet."""
        _seed_refactor_preservation(tmp_path)
        with pytest.raises((AssertionError, TypeError, Exception)):
            _verify_refactor_preservation(tmp_path)
