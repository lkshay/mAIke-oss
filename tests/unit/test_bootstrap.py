"""Tests for the dependency bootstrapper."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from maike.atoms.tool import ToolResult
from maike.runtime.bootstrap import (
    BootstrapPolicy,
    BootstrapResult,
    DependencyBootstrapper,
    GoBootstrapStrategy,
    NodeBootstrapStrategy,
    PythonBootstrapStrategy,
    RustBootstrapStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _StubManifest:
    language: str = "python"
    install_command: str | None = "pip install {package}"
    install_all_command: str | None = "pip install -e ."
    package_manager: str | None = "pip"
    interpreter_command: str | None = "python3"
    source: str = "probe"
    test_command: str | None = None
    list_packages_command: str | None = None
    lint_command: str | None = None
    typecheck_command: str | None = None
    project_root: str | None = None
    shell_env: dict = field(default_factory=dict)
    available_tools: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    command_sources: dict = field(default_factory=dict)
    confidence: str = "medium"
    ready: bool = True
    conventions: list = field(default_factory=list)
    structure_summary: list = field(default_factory=list)
    deps_installed: bool | None = None
    venv_managed: bool = False
    env_preferences: dict = field(default_factory=dict)


def _ok_result() -> ToolResult:
    return ToolResult(tool_name="execute_bash", success=True, output="OK")


def _fail_result(msg: str = "failed") -> ToolResult:
    return ToolResult(tool_name="execute_bash", success=False, output=msg, error=msg)


# ---------------------------------------------------------------------------
# PythonBootstrapStrategy
# ---------------------------------------------------------------------------

class TestPythonBootstrapStrategy:
    def test_needs_isolation_no_venv_no_deps_file(self, tmp_path):
        """No venv and no deps file — don't create a venv."""
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest()
        assert strategy.needs_isolation(tmp_path, manifest) is False

    def test_needs_isolation_no_venv_with_pyproject(self, tmp_path):
        """No venv but has pyproject.toml — needs isolation."""
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
        assert strategy.needs_isolation(tmp_path, manifest) is True

    def test_needs_isolation_with_venv(self, tmp_path):
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest()
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").touch()
        assert strategy.needs_isolation(tmp_path, manifest) is False

    def test_deps_installed_no_deps_file(self, tmp_path):
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest()
        result = strategy.deps_installed(tmp_path, manifest)
        assert result is None  # Unknown

    def test_deps_installed_with_pyproject_no_venv(self, tmp_path):
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
        result = strategy.deps_installed(tmp_path, manifest)
        assert result is False

    def test_deps_installed_with_populated_venv(self, tmp_path):
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
        # Create fake venv with dist-info dirs
        lib = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages"
        lib.mkdir(parents=True)
        for pkg in ("pip-24.0.dist-info", "setuptools-69.0.dist-info", "markdown-3.6.dist-info"):
            (lib / pkg).mkdir()
        result = strategy.deps_installed(tmp_path, manifest)
        assert result is True

    def test_install_command_from_manifest(self):
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest(install_all_command="uv sync")
        assert strategy.install_command(manifest) == "uv sync"

    def test_install_command_fallback(self):
        strategy = PythonBootstrapStrategy()
        manifest = _StubManifest(install_all_command=None)
        cmd = strategy.install_command(manifest)
        assert "pip install -e ." in cmd

    def test_isolation_label(self):
        strategy = PythonBootstrapStrategy()
        assert "venv" in strategy.isolation_label().lower()


# ---------------------------------------------------------------------------
# NodeBootstrapStrategy
# ---------------------------------------------------------------------------

class TestNodeBootstrapStrategy:
    def test_needs_isolation_always_false(self, tmp_path):
        strategy = NodeBootstrapStrategy()
        assert strategy.needs_isolation(tmp_path, _StubManifest(language="node")) is False

    def test_deps_installed_no_package_json(self, tmp_path):
        strategy = NodeBootstrapStrategy()
        result = strategy.deps_installed(tmp_path, _StubManifest(language="node"))
        assert result is None

    def test_deps_installed_no_node_modules(self, tmp_path):
        strategy = NodeBootstrapStrategy()
        (tmp_path / "package.json").write_text("{}")
        result = strategy.deps_installed(tmp_path, _StubManifest(language="node"))
        assert result is False

    def test_deps_installed_with_node_modules(self, tmp_path):
        strategy = NodeBootstrapStrategy()
        (tmp_path / "package.json").write_text("{}")
        nm = tmp_path / "node_modules" / "express"
        nm.mkdir(parents=True)
        result = strategy.deps_installed(tmp_path, _StubManifest(language="node"))
        assert result is True

    def test_install_command(self):
        strategy = NodeBootstrapStrategy()
        manifest = _StubManifest(language="node", package_manager="yarn", install_all_command=None)
        assert strategy.install_command(manifest) == "yarn install"


# ---------------------------------------------------------------------------
# Rust & Go — trivial strategies
# ---------------------------------------------------------------------------

class TestRustGoStrategies:
    def test_rust_deps_always_installed(self, tmp_path):
        assert RustBootstrapStrategy().deps_installed(tmp_path, _StubManifest()) is True

    def test_go_deps_always_installed(self, tmp_path):
        assert GoBootstrapStrategy().deps_installed(tmp_path, _StubManifest()) is True

    def test_rust_no_isolation_needed(self, tmp_path):
        assert RustBootstrapStrategy().needs_isolation(tmp_path, _StubManifest()) is False

    def test_go_no_isolation_needed(self, tmp_path):
        assert GoBootstrapStrategy().needs_isolation(tmp_path, _StubManifest()) is False


# ---------------------------------------------------------------------------
# DependencyBootstrapper integration
# ---------------------------------------------------------------------------

class TestDependencyBootstrapper:
    def _make_bootstrapper(self, runtime=None, approval_gate=None):
        if runtime is None:
            runtime = AsyncMock()
            runtime.execute_bash = AsyncMock(return_value=_ok_result())
        if approval_gate is None:
            approval_gate = MagicMock()
            approval_gate.confirm = AsyncMock(return_value=True)
        return DependencyBootstrapper(runtime, approval_gate)

    def test_unknown_language_does_nothing(self):
        boot = self._make_bootstrapper()
        manifest = _StubManifest(language="cobol")
        result = asyncio.run(boot.bootstrap(Path("/tmp"), manifest))
        assert result.action_taken == "none"
        assert result.success is True

    def test_rust_does_nothing(self):
        boot = self._make_bootstrapper()
        manifest = _StubManifest(language="rust")
        result = asyncio.run(boot.bootstrap(Path("/tmp"), manifest))
        assert result.action_taken == "none"

    def test_user_decline_skips_gracefully(self, tmp_path):
        approval = MagicMock()
        approval.confirm = AsyncMock(return_value=False)
        boot = self._make_bootstrapper(approval_gate=approval)
        manifest = _StubManifest(language="python")
        # Need a deps file for the strategy to consider isolation
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
        result = asyncio.run(boot.bootstrap(tmp_path, manifest))
        assert result.action_taken == "none"
        assert result.success is True
        assert any("declined" in d for d in result.diagnostics)

    def test_policy_no_install_skips_deps(self, tmp_path):
        boot = self._make_bootstrapper()
        manifest = _StubManifest(language="python")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
        policy = BootstrapPolicy(create_venv=False, install_deps=False)
        result = asyncio.run(boot.bootstrap(tmp_path, manifest, policy))
        assert result.action_taken == "none"

    def test_no_approval_needed_when_auto(self, tmp_path):
        approval = MagicMock()
        approval.confirm = AsyncMock(return_value=True)
        boot = self._make_bootstrapper(approval_gate=approval)
        manifest = _StubManifest(language="python")
        policy = BootstrapPolicy(require_approval=False)
        result = asyncio.run(boot.bootstrap(tmp_path, manifest, policy))
        assert result.success is True
