"""Tests for environment error diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from maike.atoms.tool import ToolResult
from maike.runtime.diagnostics import EnvironmentDiagnostics, ErrorDiagnosis


# ---------------------------------------------------------------------------
# Helpers — lightweight manifest stub
# ---------------------------------------------------------------------------

@dataclass
class _StubManifest:
    language: str = "python"
    install_command: str | None = "pip install {package}"
    install_all_command: str | None = "pip install -e ."
    package_manager: str | None = "pip"


def _make_diagnostics(language: str = "python", **kwargs) -> EnvironmentDiagnostics:
    manifest = _StubManifest(language=language, **kwargs)
    return EnvironmentDiagnostics(manifest)


def _failed_result(output: str) -> ToolResult:
    return ToolResult(
        tool_name="execute_bash",
        success=False,
        output=output,
        raw_output=output,
        error="Command exited with 1",
    )


def _success_result(output: str = "OK") -> ToolResult:
    return ToolResult(
        tool_name="execute_bash",
        success=True,
        output=output,
        raw_output=output,
    )


# ---------------------------------------------------------------------------
# Python error patterns
# ---------------------------------------------------------------------------

class TestPythonDiagnostics:
    def test_module_not_found(self):
        diag = _make_diagnostics()
        result = diag.diagnose("ModuleNotFoundError: No module named 'markdown'")
        assert len(result) == 1
        assert result[0].category == "missing_dependency"
        assert "markdown" in result[0].remediation
        assert result[0].install_command == "pip install markdown"

    def test_module_not_found_dotted(self):
        diag = _make_diagnostics()
        result = diag.diagnose("ModuleNotFoundError: No module named 'yaml.parser'")
        assert len(result) == 1
        assert "yaml.parser" in result[0].remediation

    def test_import_error(self):
        diag = _make_diagnostics()
        result = diag.diagnose(
            "ImportError: cannot import name 'Builder' from 'forge.builder'"
        )
        assert len(result) == 1
        assert result[0].category == "missing_dependency"
        assert "Builder" in result[0].remediation

    def test_missing_venv(self):
        diag = _make_diagnostics()
        result = diag.diagnose("No such file or directory: '.venv/bin/python'")
        assert len(result) == 1
        assert result[0].category == "missing_venv"
        assert "python3 -m venv" in result[0].remediation


# ---------------------------------------------------------------------------
# Node error patterns
# ---------------------------------------------------------------------------

class TestNodeDiagnostics:
    def test_cannot_find_module(self):
        diag = _make_diagnostics(
            language="node",
            install_command="npm install {package}",
            install_all_command="npm install",
        )
        result = diag.diagnose("Error: Cannot find module 'express'")
        assert len(result) == 1
        assert result[0].category == "missing_dependency"
        assert "npm install" in result[0].remediation

    def test_err_module_not_found(self):
        diag = _make_diagnostics(
            language="node",
            install_all_command="npm install",
        )
        result = diag.diagnose("ERR_MODULE_NOT_FOUND something went wrong")
        assert len(result) == 1
        assert result[0].category == "missing_dependency"


# ---------------------------------------------------------------------------
# Rust error patterns
# ---------------------------------------------------------------------------

class TestRustDiagnostics:
    def test_failed_to_resolve(self):
        diag = _make_diagnostics(language="rust", install_command=None)
        result = diag.diagnose("error[E0433]: failed to resolve: use of `serde`")
        assert len(result) == 1
        assert result[0].category == "missing_dependency"
        assert "serde" in result[0].remediation


# ---------------------------------------------------------------------------
# Universal patterns
# ---------------------------------------------------------------------------

class TestUniversalDiagnostics:
    def test_command_not_found(self):
        diag = _make_diagnostics()
        result = diag.diagnose("bash: command not found: ruff")
        assert len(result) == 1
        assert result[0].category == "missing_command"
        assert "ruff" in result[0].remediation

    def test_permission_denied(self):
        diag = _make_diagnostics()
        result = diag.diagnose("Permission denied: /usr/local/bin/something")
        assert len(result) == 1
        assert result[0].category == "permission_error"
        assert result[0].severity == "system"


# ---------------------------------------------------------------------------
# ToolResult annotation
# ---------------------------------------------------------------------------

class TestAnnotateToolResult:
    def test_successful_result_unchanged(self):
        diag = _make_diagnostics()
        result = _success_result("all tests passed")
        annotated = diag.annotate_tool_result(result)
        assert annotated is result  # exact same object

    def test_failed_without_pattern_unchanged(self):
        diag = _make_diagnostics()
        result = _failed_result("AssertionError: 1 != 2")
        annotated = diag.annotate_tool_result(result)
        assert "ENVIRONMENT DIAGNOSTIC" not in annotated.output

    def test_failed_with_module_not_found_annotated(self):
        diag = _make_diagnostics()
        result = _failed_result(
            "ModuleNotFoundError: No module named 'markdown'\n"
            "Error: subprocess exited with 1"
        )
        annotated = diag.annotate_tool_result(result)
        assert "ENVIRONMENT DIAGNOSTIC" in annotated.output
        assert "not a code bug" in annotated.output
        assert "pip install markdown" in annotated.output
        assert "pip install -e ." in annotated.output
        assert annotated.metadata.get("environment_diagnoses")
        assert annotated.metadata["environment_diagnoses"][0]["category"] == "missing_dependency"

    def test_failed_result_preserves_original_output(self):
        diag = _make_diagnostics()
        original_output = "ModuleNotFoundError: No module named 'jinja2'"
        result = _failed_result(original_output)
        annotated = diag.annotate_tool_result(result)
        assert original_output in annotated.output

    def test_multiple_patterns_deduplicated(self):
        """Multiple matches of the same category should produce only one diagnosis."""
        diag = _make_diagnostics()
        result = _failed_result(
            "ModuleNotFoundError: No module named 'markdown'\n"
            "ModuleNotFoundError: No module named 'jinja2'"
        )
        annotated = diag.annotate_tool_result(result)
        diagnoses = annotated.metadata.get("environment_diagnoses", [])
        # Only one diagnosis per category
        assert len(diagnoses) == 1

    def test_no_patterns_for_unknown_language(self):
        """Unknown language should still get universal patterns."""
        diag = _make_diagnostics(language="unknown")
        result = _failed_result("bash: command not found: mycommand")
        annotated = diag.annotate_tool_result(result)
        assert "ENVIRONMENT DIAGNOSTIC" in annotated.output

    def test_custom_install_command_used(self):
        diag = _make_diagnostics(
            install_command="uv pip install {package}",
            install_all_command="uv sync",
        )
        result = _failed_result("ModuleNotFoundError: No module named 'flask'")
        annotated = diag.annotate_tool_result(result)
        assert "uv pip install flask" in annotated.output
        assert "uv sync" in annotated.output
