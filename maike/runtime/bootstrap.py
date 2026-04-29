"""Dependency bootstrapper — ensures the workspace environment is ready.

Called between environment probe and preflight check.  Detects missing
isolation environments (venvs) and uninstalled dependencies, creates/installs
them with user consent, and re-probes to update the manifest.

Language-specific logic is delegated to BootstrapStrategy implementations
registered in _BOOTSTRAP_STRATEGIES.  Adding a new language requires only
writing a strategy class and registering it — no changes to the bootstrapper
or orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from maike.runtime.probe import EnvironmentManifest
    from maike.runtime.protocol import ExecutionRuntime
    from maike.safety.approval import ApprovalGate

log = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 300


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapPolicy:
    """Controls what the bootstrapper is allowed to do."""

    create_venv: bool = True        # Create .venv if missing (Python)
    install_deps: bool = True       # Install deps from manifest files
    require_approval: bool = True   # Ask user before acting (False with --yes)


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a bootstrap attempt."""

    action_taken: str                   # "none" | "venv_created" | "deps_installed" | "both"
    success: bool
    manifest: EnvironmentManifest       # Updated manifest after bootstrap
    diagnostics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy protocol — implement per language
# ---------------------------------------------------------------------------

@runtime_checkable
class BootstrapStrategy(Protocol):
    """Language-specific bootstrap behavior."""

    def needs_isolation(self, workspace: Path, manifest: EnvironmentManifest) -> bool:
        """Does this language need an isolated environment (venv, etc.)?"""
        ...

    async def create_isolation(self, runtime: ExecutionRuntime, workspace: Path) -> bool:
        """Create the isolation env (venv, virtualenv, etc.). Returns success."""
        ...

    def deps_installed(self, workspace: Path, manifest: EnvironmentManifest) -> bool | None:
        """Quick check: are deps installed? None = unknown."""
        ...

    def install_command(self, manifest: EnvironmentManifest) -> str | None:
        """Command to install all deps. None = not applicable."""
        ...

    def isolation_label(self) -> str:
        """Human-readable label: 'virtual environment', 'node_modules', etc."""
        ...


# ---------------------------------------------------------------------------
# Language strategies
# ---------------------------------------------------------------------------

class PythonBootstrapStrategy:
    """Bootstrap for Python projects."""

    def needs_isolation(self, workspace: Path, manifest: EnvironmentManifest) -> bool:
        # Only create a venv if there's a dependency file — otherwise
        # we'd create venvs in empty workspaces or test directories.
        has_deps_file = any(
            (workspace / f).exists()
            for f in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg")
        )
        if not has_deps_file:
            return False

        # Check for common venv locations
        for venv_dir in (".venv", "venv"):
            venv_path = workspace / venv_dir
            if venv_path.is_dir() and (venv_path / "bin" / "python").exists():
                return False
            if venv_path.is_dir() and (venv_path / "Scripts" / "python.exe").exists():
                return False
        return True

    async def create_isolation(self, runtime: ExecutionRuntime, workspace: Path) -> bool:
        result = await runtime.execute_bash(
            "python3 -m venv .venv",
            timeout=DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS,
        )
        return result.success

    def deps_installed(self, workspace: Path, manifest: EnvironmentManifest) -> bool | None:
        # If there's no manifest file, we can't check deps
        has_deps_file = any(
            (workspace / f).exists()
            for f in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg")
        )
        if not has_deps_file:
            return None  # Unknown — no deps file to check against

        # Quick heuristic: if a venv exists and has site-packages with content,
        # assume deps are probably installed.  The full check (pip check) happens
        # at runtime and the diagnostics layer catches failures.
        for venv_dir in (".venv", "venv"):
            site_packages = workspace / venv_dir / "lib"
            if site_packages.is_dir():
                # Has a venv with lib/ — check if anything is installed beyond pip
                try:
                    pkg_count = sum(
                        1 for p in site_packages.rglob("*.dist-info")
                    )
                    if pkg_count > 2:  # pip + setuptools at minimum
                        return True
                except OSError:
                    pass
                return False  # Venv exists but appears empty

        return False  # No venv at all

    def install_command(self, manifest: EnvironmentManifest) -> str | None:
        if manifest.install_all_command:
            return manifest.install_all_command
        # Fall back to standard pip install
        interpreter = manifest.interpreter_command or "python3"
        return f"{interpreter} -m pip install -e ."

    def fallback_install_commands(
        self, workspace: Path, manifest: EnvironmentManifest
    ) -> list[str]:
        """Return fallback install commands when the primary fails.

        Ordered from most faithful to original intent to least:
        1. Non-editable install (``pip install .``) — works for C extensions
        2. PyPI install by name — prebuilt wheel, avoids compilation entirely
        """
        fallbacks: list[str] = []
        interpreter = manifest.interpreter_command or "python3"
        primary = self.install_command(manifest) or ""

        # If primary is editable, try non-editable
        if "-e" in primary or "--editable" in primary:
            fallbacks.append(f"{interpreter} -m pip install .")

        # Try installing by package name from PyPI (prebuilt wheel)
        pkg_name = self._extract_package_name(workspace)
        if pkg_name:
            fallbacks.append(f"{interpreter} -m pip install {pkg_name}")

        return fallbacks

    @staticmethod
    def _extract_package_name(workspace: Path) -> str | None:
        """Extract package name from pyproject.toml or setup.py."""
        import re as _re

        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                match = _re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
            except OSError:
                pass

        setup_py = workspace / "setup.py"
        if setup_py.exists():
            try:
                content = setup_py.read_text(encoding="utf-8")
                match = _re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
            except OSError:
                pass

        return None

    def isolation_label(self) -> str:
        return "virtual environment (.venv)"


class NodeBootstrapStrategy:
    """Bootstrap for Node.js projects."""

    def needs_isolation(self, workspace: Path, manifest: EnvironmentManifest) -> bool:
        return False  # Node doesn't need a separate isolation env

    async def create_isolation(self, runtime: ExecutionRuntime, workspace: Path) -> bool:
        return True  # N/A

    def deps_installed(self, workspace: Path, manifest: EnvironmentManifest) -> bool | None:
        has_package_json = (workspace / "package.json").exists()
        if not has_package_json:
            return None
        node_modules = workspace / "node_modules"
        if node_modules.is_dir():
            # Check it's not empty
            try:
                has_contents = any(node_modules.iterdir())
                return has_contents
            except OSError:
                return False
        return False

    def install_command(self, manifest: EnvironmentManifest) -> str | None:
        if manifest.install_all_command:
            return manifest.install_all_command
        pm = manifest.package_manager or "npm"
        return f"{pm} install"

    def isolation_label(self) -> str:
        return "node_modules"


class RustBootstrapStrategy:
    """Bootstrap for Rust projects — cargo fetches deps on demand."""

    def needs_isolation(self, workspace: Path, manifest: EnvironmentManifest) -> bool:
        return False

    async def create_isolation(self, runtime: ExecutionRuntime, workspace: Path) -> bool:
        return True

    def deps_installed(self, workspace: Path, manifest: EnvironmentManifest) -> bool | None:
        return True  # Cargo fetches on demand

    def install_command(self, manifest: EnvironmentManifest) -> str | None:
        return None

    def isolation_label(self) -> str:
        return "cargo registry"


class GoBootstrapStrategy:
    """Bootstrap for Go projects — go fetches deps on demand."""

    def needs_isolation(self, workspace: Path, manifest: EnvironmentManifest) -> bool:
        return False

    async def create_isolation(self, runtime: ExecutionRuntime, workspace: Path) -> bool:
        return True

    def deps_installed(self, workspace: Path, manifest: EnvironmentManifest) -> bool | None:
        return True  # Go fetches on demand

    def install_command(self, manifest: EnvironmentManifest) -> str | None:
        return None

    def isolation_label(self) -> str:
        return "go module cache"


# ---------------------------------------------------------------------------
# Strategy registry — add new languages here
# ---------------------------------------------------------------------------

_BOOTSTRAP_STRATEGIES: dict[str, BootstrapStrategy] = {
    "python": PythonBootstrapStrategy(),
    "node": NodeBootstrapStrategy(),
    "rust": RustBootstrapStrategy(),
    "go": GoBootstrapStrategy(),
}


def register_bootstrap_strategy(language: str, strategy: BootstrapStrategy) -> None:
    """Register a custom bootstrap strategy for a language."""
    _BOOTSTRAP_STRATEGIES[language] = strategy


# ---------------------------------------------------------------------------
# Main bootstrapper
# ---------------------------------------------------------------------------

class DependencyBootstrapper:
    """Ensures workspace dependencies are available before agents run.

    Called by the orchestrator between EnvironmentProbe.resolve() and
    PreflightChecker.ensure_ready().
    """

    def __init__(
        self,
        runtime: ExecutionRuntime,
        approval_gate: ApprovalGate,
    ) -> None:
        self._runtime = runtime
        self._approval_gate = approval_gate

    async def bootstrap(
        self,
        workspace: Path,
        manifest: EnvironmentManifest,
        policy: BootstrapPolicy | None = None,
    ) -> BootstrapResult:
        """Check and set up the workspace environment.

        Returns a BootstrapResult with the (possibly updated) manifest.
        """
        if policy is None:
            policy = BootstrapPolicy()

        language = manifest.language.lower()
        strategy = _BOOTSTRAP_STRATEGIES.get(language)

        if strategy is None:
            log.debug("No bootstrap strategy for language=%s", language)
            return BootstrapResult(
                action_taken="none",
                success=True,
                manifest=manifest,
                diagnostics=[f"No bootstrap strategy for language '{language}'"],
            )

        diagnostics: list[str] = []
        created_isolation = False
        installed_deps = False

        # Step 1: Create isolation environment if needed
        if strategy.needs_isolation(workspace, manifest) and policy.create_venv:
            label = strategy.isolation_label()
            prompt = f"No {label} found. Create one and install dependencies? [y/N]: "

            if policy.require_approval:
                approved = await self._approval_gate.confirm(prompt)
            else:
                approved = True

            if approved:
                log.info("Creating %s in %s", label, workspace)
                success = await strategy.create_isolation(self._runtime, workspace)
                if success:
                    created_isolation = True
                    diagnostics.append(f"Created {label}")
                else:
                    diagnostics.append(f"Failed to create {label}")
                    return BootstrapResult(
                        action_taken="none",
                        success=False,
                        manifest=manifest,
                        diagnostics=diagnostics,
                    )
            else:
                diagnostics.append(f"User declined {label} creation")

        # If we just created a venv, re-probe so that the install command
        # uses the venv's python interpreter (not the system one).
        if created_isolation:
            from maike.runtime.probe import EnvironmentProbe
            manifest = EnvironmentProbe().resolve(workspace)

        # Step 2: Check and install dependencies
        deps_status = strategy.deps_installed(workspace, manifest)
        if deps_status is False and policy.install_deps:
            cmd = strategy.install_command(manifest)
            if cmd:
                prompt = f"Dependencies not installed. Run '{cmd}'? [y/N]: "
                if policy.require_approval and not created_isolation:
                    # If we already got approval for isolation creation,
                    # the user implicitly approved dep installation too.
                    approved = await self._approval_gate.confirm(prompt)
                else:
                    approved = True

                if approved:
                    log.info("Installing dependencies: %s", cmd)
                    # If we just created a venv, rebuild the runtime so
                    # execute_bash uses the venv's PATH and interpreter.
                    install_runtime = self._runtime
                    if created_isolation:
                        from maike.runtime.local import LocalRuntime
                        install_runtime = LocalRuntime(workspace, manifest=manifest)
                    result = await install_runtime.execute_bash(
                        cmd, timeout=DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
                    )
                    if result.success:
                        installed_deps = True
                        diagnostics.append(f"Installed dependencies: {cmd}")
                    else:
                        diagnostics.append(
                            f"Primary install failed: {result.error or 'unknown error'}"
                        )
                        # Try fallback install commands (non-editable, PyPI, etc.)
                        fallback_cmds = (
                            strategy.fallback_install_commands(workspace, manifest)
                            if hasattr(strategy, "fallback_install_commands")
                            else []
                        )
                        for fb_cmd in fallback_cmds:
                            log.info("Trying fallback install: %s", fb_cmd)
                            fb_result = await install_runtime.execute_bash(
                                fb_cmd, timeout=DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
                            )
                            if fb_result.success:
                                installed_deps = True
                                diagnostics.append(f"Installed via fallback: {fb_cmd}")
                                break
                            diagnostics.append(f"Fallback failed: {fb_cmd}")

                        if not installed_deps:
                            diagnostics.append(
                                "All install attempts failed. The agent will start "
                                "without dependencies — it may need to install them manually."
                            )
                else:
                    diagnostics.append("User declined dependency installation")

        # Step 3: Re-probe if anything changed
        if created_isolation or installed_deps:
            from maike.runtime.probe import EnvironmentProbe
            updated_manifest = EnvironmentProbe().resolve(workspace)
            updated_manifest.venv_managed = created_isolation
            action = "both" if (created_isolation and installed_deps) else (
                "venv_created" if created_isolation else "deps_installed"
            )

            return BootstrapResult(
                action_taken=action,
                success=True,
                manifest=updated_manifest,
                diagnostics=diagnostics,
            )

        return BootstrapResult(
            action_taken="none",
            success=True,
            manifest=manifest,
            diagnostics=diagnostics,
        )
