from __future__ import annotations

import os
import platform as _platform
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil


@dataclass
class EnvironmentManifest:
    language: str = "unknown"
    source: str = "probe"          # "MAIKE.md" | "AGENTS.md" | "probe" | "fallback"
    test_command: str | None = None
    install_command: str | None = None
    list_packages_command: str | None = None
    install_all_command: str | None = None
    lint_command: str | None = None
    typecheck_command: str | None = None
    package_manager: str | None = None
    interpreter_command: str | None = None
    project_root: str | None = None
    shell_env: dict[str, str] = field(default_factory=dict)
    available_tools: dict[str, bool] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    command_sources: dict[str, str] = field(default_factory=dict)
    confidence: str = "low"
    ready: bool = True
    conventions: list[str] = field(default_factory=list)
    structure_summary: list[str] = field(default_factory=list)
    # Bootstrap state — set by DependencyBootstrapper after checking/installing.
    deps_installed: bool | None = None       # None = unknown, True/False = checked
    venv_managed: bool = False               # True if mAIke created the venv
    env_preferences: dict[str, str] = field(default_factory=dict)  # from MAIKE.md ## Environment
    # Framework detection.
    frameworks: list[str] = field(default_factory=list)  # e.g. ["django", "react", "fastapi"]
    # Target language version (e.g. ">=3.7,<3.11" for a Python project's
    # python_requires).  Surfaced to the agent so it doesn't waste budget
    # "fixing" stdlib import errors that are actually host-vs-target
    # version mismatches.  See django__django-11400 (24 Apr 2026): agent
    # spent budget patching cgi removal in Python 3.13 even though the
    # harness runs Python 3.7.  None when not detected.
    target_language_version: str | None = None

    def to_runtime_config(self) -> "RuntimeConfig":
        """Bridge to existing LocalRuntime interface."""
        from maike.runtime.local import RuntimeConfig
        return RuntimeConfig(
            language=self.language,
            test_command=self.test_command or "pytest {path} --tb=short -q",
            install_command=self.install_command or "python3 -m pip install {package}",
            list_packages_command=self.list_packages_command or "python3 -m pip list --format=columns",
            install_all_command=self.install_all_command,
            lint_command=self.lint_command,
            package_manager=self.package_manager,
            interpreter_command=self.interpreter_command,
            project_root=self.project_root,
            shell_env=dict(self.shell_env),
            diagnostics=list(self.diagnostics),
            environment_confidence=self.confidence,
            environment_ready=self.ready,
        )

    def to_context_block(self) -> str:
        """Injected into agent context at session start.

        Descriptive only — reports what files and tools were detected, not
        what commands to run.  The agent reads project config files (README,
        Makefile, pyproject.toml, package.json) to determine exact commands.
        """
        now_local = datetime.now().astimezone()
        lines = [
            "## Session Info",
            f"Date: {now_local.strftime('%Y-%m-%d')} ({now_local.strftime('%A')})",
            f"Local time: {now_local.strftime('%H:%M %Z')}",
            f"Platform: {_platform.system()} {_platform.release()}",
            "",
            "## Project Environment",
            f"Language: {self.language}",
        ]
        if self.package_manager:
            lines.append(f"Package manager: {self.package_manager}")
        if self.target_language_version:
            lines.append(
                f"Target {self.language} version: {self.target_language_version}"
            )
        if self.frameworks:
            lines.append(f"Frameworks: {', '.join(self.frameworks)}")
        if self.structure_summary:
            lines.append("\n## Project Structure")
            lines.extend(self.structure_summary)
        if self.diagnostics:
            lines.append("\n## Runtime Diagnostics")
            lines.extend(f"- {item}" for item in self.diagnostics)
        if self.conventions:
            lines.append("\n## Conventions")
            lines.extend(f"- {c}" for c in self.conventions)

        # Config files detected — tell the agent where to look for commands.
        config_hints = self._detected_config_files()
        if config_hints:
            lines.append("\n## Detected Config Files")
            lines.extend(f"- {f}" for f in config_hints)

        if self.confidence == "low" or self.language == "unknown":
            lines.append(
                "\nEnvironment detection confidence is low. Read the project's "
                "README, Makefile, or build config files to determine the correct "
                "build, test, and install commands before proceeding."
            )

        return "\n".join(lines)

    def _detected_config_files(self) -> list[str]:
        """List config files found in the project root."""
        if not self.project_root:
            return []
        root = Path(self.project_root)
        config_files = [
            "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
            "Pipfile", "uv.lock", "poetry.lock",
            "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            "Cargo.toml", "go.mod", "Gemfile",
            "Makefile", "Dockerfile", "docker-compose.yml",
            "tsconfig.json", "jest.config.js", "vitest.config.ts",
            "pytest.ini", "tox.ini", "ruff.toml", "mypy.ini",
            ".eslintrc.js", ".eslintrc.json",
        ]
        found = []
        for f in config_files:
            if (root / f).exists():
                found.append(f)
        return found



class EnvironmentProbe:
    """
    Reads the workspace at session start and produces an EnvironmentManifest.
    Priority order:
      1. MAIKE.md   — explicit project config for mAIke
      2. AGENTS.md  — alternate project config format
      3. CLAUDE.md  — alternate project config format
      4. Filesystem signal detection — implicit, covers most projects
      5. Unknown    — agents are told to probe further themselves
    """

    # Language detection — checked in order, first match wins
    LANGUAGE_SIGNALS: list[tuple[str, str]] = [
        ("pyproject.toml",   "python"),
        ("requirements.txt", "python"),
        ("setup.py",         "python"),
        ("Cargo.toml",       "rust"),
        ("go.mod",           "go"),
        ("package.json",     "node"),
        ("pom.xml",          "java"),
        ("build.gradle",     "java"),
        ("Gemfile",          "ruby"),
        ("*.csproj",         "csharp"),
    ]

    # Package manager signals — more specific wins over language default
    # Checked in priority order: more specific lock files first
    PYTHON_PM_SIGNALS: list[tuple[str, str, str, str, str]] = [
        # (signal_file, package_manager, install_command, install_all_command, list_packages_command)
        ("uv.lock",          "uv",      "uv add {package}",         "uv sync",                     "uv pip list"),
        ("poetry.lock",      "poetry",  "poetry add {package}",     "poetry install",             "poetry show"),
        ("Pipfile.lock",     "pipenv",  "pipenv install {package}", "pipenv install",             "pipenv run python -m pip list --format=columns"),
        ("requirements.txt", "pip",     "pip install {package}",    "pip install -r requirements.txt", "python3 -m pip list --format=columns"),
        ("pyproject.toml",   "pip",     "pip install {package}",    "pip install -e .",           "python3 -m pip list --format=columns"),
    ]

    NODE_PM_SIGNALS: list[tuple[str, str, str, str, str]] = [
        ("yarn.lock",         "yarn", "yarn add {package}",    "yarn install",  "yarn list --depth=0"),
        ("pnpm-lock.yaml",    "pnpm", "pnpm add {package}",    "pnpm install",  "pnpm list --depth=0"),
        ("package-lock.json", "npm",  "npm install {package}", "npm install",   "npm ls --depth=0"),
        ("package.json",      "npm",  "npm install {package}", "npm install",   "npm ls --depth=0"),
    ]

    # Test runner signals per language
    PYTHON_TEST_SIGNALS: list[tuple[str, str]] = [
        ("pytest.ini",       "pytest {path} --tb=short -q"),
        ("pyproject.toml",   "pytest {path} --tb=short -q"),   # check for [tool.pytest] inside
        ("setup.cfg",        "pytest {path} --tb=short -q"),
        ("tox.ini",          "python -m pytest {path} --tb=short -q"),
    ]

    NODE_TEST_SIGNALS: list[tuple[str, str]] = [
        ("vitest.config.ts",  "npx vitest run {path}"),
        ("vitest.config.js",  "npx vitest run {path}"),
        ("jest.config.ts",    "npx jest {path}"),
        ("jest.config.js",    "npx jest {path}"),
        ("package.json",      "npm test"),                    # fallback
    ]

    # Framework detection — maps dependency names to canonical framework labels.
    # Tuples: (dependency_pattern, label)
    PYTHON_FRAMEWORK_SIGNALS: list[tuple[str, str]] = [
        ("django", "django"),
        ("fastapi", "fastapi"),
        ("flask", "flask"),
        ("starlette", "starlette"),
        ("celery", "celery"),
        ("sqlalchemy", "sqlalchemy"),
        ("alembic", "alembic"),
        ("pydantic", "pydantic"),
    ]

    JS_FRAMEWORK_SIGNALS: list[tuple[str, str]] = [
        ("react", "react"),
        ("next", "nextjs"),
        ("express", "express"),
        ("vue", "vue"),
        ("@angular/core", "angular"),
        ("angular", "angular"),
        ("svelte", "svelte"),
    ]

    GO_FRAMEWORK_SIGNALS: list[tuple[str, str]] = [
        ("github.com/gin-gonic/gin", "gin"),
        ("github.com/gorilla/mux", "gorilla"),
    ]

    RUST_FRAMEWORK_SIGNALS: list[tuple[str, str]] = [
        ("actix-web", "actix"),
        ("axum", "axum"),
        ("rocket", "rocket"),
    ]

    CONFIG_FILES = ["MAIKE.md", "AGENTS.md", "CLAUDE.md"]

    # TODO(env-runtime-planner): add a repair/bootstrap planner on top of the
    # resolved manifest so the runtime can choose and verify safe environment
    # mutations when the workspace is detected correctly but not yet runnable.

    def resolve(
        self,
        workspace: Path,
        *,
        language_override: str | None = None,
        default_config: "RuntimeConfig" | None = None,
    ) -> EnvironmentManifest:
        manifest = self.probe(workspace)
        if manifest.language == "unknown":
            manifest = replace(manifest, source="fallback")
        if language_override is not None:
            normalized = language_override.lower()
            if manifest.language != normalized:
                manifest = replace(
                    manifest,
                    language=normalized,
                    source="override",
                    test_command=None,
                    install_command=None,
                    list_packages_command=None,
                    install_all_command=None,
                    lint_command=None,
                    package_manager=None,
                    interpreter_command=None,
                    shell_env={},
                    diagnostics=[],
                    command_sources={},
                )
            else:
                manifest = replace(manifest, language=normalized)
        defaults = self._defaults_for_language(manifest.language, default_config=default_config)
        resolved = EnvironmentManifest(
            language=manifest.language if manifest.language != "unknown" else defaults.language,
            source=manifest.source,
            test_command=manifest.test_command or defaults.test_command,
            install_command=manifest.install_command or defaults.install_command,
            list_packages_command=manifest.list_packages_command or defaults.list_packages_command,
            install_all_command=manifest.install_all_command,
            lint_command=manifest.lint_command,
            package_manager=manifest.package_manager or defaults.package_manager,
            interpreter_command=manifest.interpreter_command or defaults.interpreter_command,
            project_root=manifest.project_root or str(workspace),
            shell_env=dict(manifest.shell_env or defaults.shell_env),
            available_tools=dict(manifest.available_tools),
            diagnostics=list(manifest.diagnostics or defaults.diagnostics),
            command_sources=dict(manifest.command_sources),
            confidence=manifest.confidence or defaults.environment_confidence,
            ready=manifest.ready if manifest.ready is not None else defaults.environment_ready,
            conventions=list(manifest.conventions),
            frameworks=list(manifest.frameworks),
        )
        return self._resolve_runtime_details(workspace, resolved)

    def probe(self, workspace: Path) -> EnvironmentManifest:
        # 1. Explicit config file — highest trust
        for config_name in self.CONFIG_FILES:
            config_path = workspace / config_name
            if config_path.exists():
                manifest = self._parse_config_file(config_path, config_name)
                if manifest:
                    return manifest

        manifest = EnvironmentManifest(project_root=str(workspace))

        # 2. Detect language
        for filename, language in self.LANGUAGE_SIGNALS:
            if "*" in filename:
                matches = list(workspace.glob(filename))
                found = bool(matches)
            else:
                found = (workspace / filename).exists()
            if found:
                manifest.language = language
                break

        # 3. Detect package manager + test runner based on language
        if manifest.language == "python":
            self._probe_python(workspace, manifest)
        elif manifest.language == "node":
            self._probe_node(workspace, manifest)
        elif manifest.language == "rust":
            manifest.package_manager = "cargo"
            manifest.test_command = "cargo test {path}"
            manifest.install_command = "cargo add {package}"
            manifest.list_packages_command = "cargo tree --depth 1"
            manifest.install_all_command = "cargo build"
        elif manifest.language == "go":
            manifest.package_manager = "go"
            manifest.test_command = "go test {path}/..."
            manifest.install_command = "go get {package}"
            manifest.list_packages_command = "go list -m all"
            manifest.install_all_command = "go mod tidy"

        # 4. Detect frameworks
        manifest.frameworks = self._detect_frameworks(workspace, manifest.language)

        manifest.source = "probe"
        return manifest

    def _probe_python(self, workspace: Path, manifest: EnvironmentManifest) -> None:
        for filename, package_manager, install_cmd, install_all_cmd, list_packages_cmd in self.PYTHON_PM_SIGNALS:
            if (workspace / filename).exists():
                manifest.package_manager = package_manager
                manifest.install_command = install_cmd
                manifest.install_all_command = install_all_cmd
                manifest.list_packages_command = list_packages_cmd
                break

        for filename, test_cmd in self.PYTHON_TEST_SIGNALS:
            if (workspace / filename).exists():
                manifest.test_command = test_cmd
                break

        # Detect ruff vs flake8 vs pylint
        if (workspace / "ruff.toml").exists() or self._pyproject_has(workspace, "ruff"):
            manifest.lint_command = "ruff check ."
        elif (workspace / ".flake8").exists():
            manifest.lint_command = "flake8 ."

        # Detect type checker
        if (workspace / "mypy.ini").exists() or self._pyproject_has(workspace, "mypy"):
            manifest.typecheck_command = "mypy ."
        elif (workspace / "pyrightconfig.json").exists() or (workspace / "pyproject.toml").exists() and self._pyproject_has(workspace, "pyright"):
            manifest.typecheck_command = "pyright ."

        # Extract Python version constraint (python_requires from setup.py
        # or requires-python from pyproject.toml).  Used to warn the agent
        # about host-vs-target Python mismatches in SWE-bench eval, where
        # the host runs a newer Python than the target instance and
        # ImportError on removed stdlib modules (cgi, asyncore, imp) gets
        # misread as a real bug.
        manifest.target_language_version = _extract_python_requires(workspace)

    def _probe_node(self, workspace: Path, manifest: EnvironmentManifest) -> None:
        for filename, package_manager, install_cmd, install_all_cmd, list_packages_cmd in self.NODE_PM_SIGNALS:
            if (workspace / filename).exists():
                manifest.package_manager = package_manager
                manifest.install_command = install_cmd
                manifest.install_all_command = install_all_cmd
                manifest.list_packages_command = list_packages_cmd
                break

        for filename, test_cmd in self.NODE_TEST_SIGNALS:
            if (workspace / filename).exists():
                manifest.test_command = test_cmd
                break

        # Detect TypeScript type checker
        if (workspace / "tsconfig.json").exists():
            manifest.typecheck_command = "npx tsc --noEmit"

    # Framework detection signals: (file_or_import, framework_name)
    _PYTHON_FRAMEWORK_SIGNALS: tuple[tuple[str, str], ...] = (
        ("manage.py", "django"),
        ("django", "django"),         # import in requirements
        ("fastapi", "fastapi"),
        ("flask", "flask"),
        ("starlette", "starlette"),
        ("celery", "celery"),
        ("sqlalchemy", "sqlalchemy"),
        ("alembic", "alembic"),
        ("pydantic", "pydantic"),
    )

    _NODE_FRAMEWORK_SIGNALS: tuple[tuple[str, str], ...] = (
        ("next", "nextjs"),
        ("nuxt", "nuxtjs"),
        ("angular", "angular"),
        ("svelte", "svelte"),
        ("vite", "vite"),
        ("react", "react"),           # in package.json deps
        ("vue", "vue"),
        ("express", "express"),
        ("nestjs", "nestjs"),
    )

    def _detect_frameworks(self, workspace: Path, language: str) -> list[str]:
        """Detect web frameworks and libraries in the workspace."""
        detected: list[str] = []

        if language == "python":
            # Check for manage.py (Django)
            if (workspace / "manage.py").exists():
                detected.append("django")
            # Scan requirements files for framework imports
            req_content = ""
            for req_file in ("requirements.txt", "pyproject.toml", "Pipfile"):
                req_path = workspace / req_file
                if req_path.exists():
                    try:
                        req_content = req_path.read_text(encoding="utf-8").lower()
                    except OSError:
                        continue
                    break
            for signal, framework in self._PYTHON_FRAMEWORK_SIGNALS:
                if framework not in detected and signal in req_content:
                    detected.append(framework)

        elif language == "node":
            # Check config files
            for config_file in ("next.config.js", "next.config.mjs", "next.config.ts"):
                if (workspace / config_file).exists():
                    detected.append("nextjs")
                    break
            if (workspace / "angular.json").exists():
                detected.append("angular")
            if (workspace / "svelte.config.js").exists():
                detected.append("svelte")
            # Scan package.json dependencies
            pkg_path = workspace / "package.json"
            if pkg_path.exists():
                try:
                    pkg_content = pkg_path.read_text(encoding="utf-8").lower()
                    for signal, framework in self._NODE_FRAMEWORK_SIGNALS:
                        if framework not in detected and f'"{signal}"' in pkg_content:
                            detected.append(framework)
                except OSError:
                    pass

        elif language == "go":
            go_mod = workspace / "go.mod"
            if go_mod.exists():
                try:
                    content = go_mod.read_text(encoding="utf-8").lower()
                    for signal, framework in self.GO_FRAMEWORK_SIGNALS:
                        if signal.lower() in content and framework not in detected:
                            detected.append(framework)
                except OSError:
                    pass

        elif language == "rust":
            cargo_toml = workspace / "Cargo.toml"
            if cargo_toml.exists():
                try:
                    content = cargo_toml.read_text(encoding="utf-8").lower()
                    for signal, framework in self.RUST_FRAMEWORK_SIGNALS:
                        if signal.lower() in content and framework not in detected:
                            detected.append(framework)
                except OSError:
                    pass

        return detected

    def _pyproject_has(self, workspace: Path, section: str) -> bool:
        """Check if pyproject.toml contains a [tool.{section}] table."""
        pyproject = workspace / "pyproject.toml"
        if not pyproject.exists():
            return False
        try:
            content = pyproject.read_text(encoding="utf-8")
            return f"[tool.{section}]" in content
        except OSError:
            return False

    def _parse_config_file(
        self, path: Path, source: str
    ) -> EnvironmentManifest | None:
        """
        Parse MAIKE.md / AGENTS.md / CLAUDE.md.
        Looks for a ## Commands section with key: value pairs.
        Returns None if the file doesn't contain recognizable environment config.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None

        manifest = EnvironmentManifest(source=source)
        found_commands = False

        in_commands = False
        in_environment = False
        for line in content.splitlines():
            if re.match(r"^##\s+Commands", line, re.IGNORECASE):
                in_commands = True
                in_environment = False
                found_commands = True
                continue
            if re.match(r"^##\s+Environment", line, re.IGNORECASE):
                in_environment = True
                in_commands = False
                continue
            if re.match(r"^##\s+", line):
                in_commands = False
                in_environment = False

            if in_commands and ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower().lstrip("-").strip()
                value = value.strip()
                if key in ("test", "run tests"):
                    manifest.test_command = value
                elif key in ("install", "install package"):
                    manifest.install_command = value
                elif key in ("list packages", "packages"):
                    manifest.list_packages_command = value
                elif key in ("install all", "setup"):
                    manifest.install_all_command = value
                elif key in ("lint", "check"):
                    manifest.lint_command = value
                elif key in ("package manager", "package_manager"):
                    manifest.package_manager = value.lower()
                elif key in ("interpreter", "python"):
                    manifest.interpreter_command = value

            # Parse ## Environment section for bootstrap preferences
            if in_environment and ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower().lstrip("-").strip()
                value = value.strip().lower()
                if key in ("venv", "virtualenv", "virtual_env"):
                    manifest.env_preferences["venv"] = value
                elif key in ("install_deps", "install deps", "dependencies"):
                    manifest.env_preferences["install_deps"] = value
                elif key in ("python_version", "python version"):
                    manifest.env_preferences["python_version"] = value

            # Language line anywhere in file
            lang_match = re.match(r"^\s*[*-]?\s*[Ll]anguage\s*:\s*(\w+)", line)
            if lang_match:
                manifest.language = lang_match.group(1).lower()

        if manifest.package_manager is None:
            manifest.package_manager = self._infer_package_manager(manifest)
        return manifest if found_commands or manifest.language != "unknown" else None

    def _defaults_for_language(
        self,
        language: str,
        *,
        default_config: "RuntimeConfig" | None = None,
    ) -> "RuntimeConfig":
        from maike.runtime.local import RuntimeConfig

        if language == "unknown":
            return default_config or RuntimeConfig.for_language("python")
        return RuntimeConfig.for_language(language)

    def _infer_package_manager(self, manifest: EnvironmentManifest) -> str | None:
        commands = " ".join(
            item
            for item in (
                manifest.install_command,
                manifest.install_all_command,
                manifest.list_packages_command,
                manifest.test_command,
            )
            if item
        ).lower()
        if "uv " in commands:
            return "uv"
        if "poetry" in commands:
            return "poetry"
        if "pipenv" in commands:
            return "pipenv"
        if "pnpm" in commands:
            return "pnpm"
        if "yarn" in commands:
            return "yarn"
        if "npm" in commands:
            return "npm"
        if "cargo" in commands:
            return "cargo"
        if "go " in commands:
            return "go"
        if "pip" in commands:
            return "pip"
        return None

    def _resolve_runtime_details(self, workspace: Path, manifest: EnvironmentManifest) -> EnvironmentManifest:
        available_tools = self._discover_available_tools()
        diagnostics = list(manifest.diagnostics)
        shell_env = dict(manifest.shell_env)
        command_sources = dict(manifest.command_sources)
        package_manager = manifest.package_manager
        interpreter_command = manifest.interpreter_command
        confidence = "high" if manifest.source in {"MAIKE.md", "AGENTS.md", "CLAUDE.md"} else "medium"
        ready = manifest.ready
        updates: dict[str, object] = {
            "available_tools": available_tools,
            "project_root": manifest.project_root or str(workspace),
        }

        if manifest.language == "python":
            resolved = self._resolve_python_runtime(workspace, manifest, available_tools)
            package_manager = resolved["package_manager"]
            interpreter_command = resolved["interpreter_command"]
            shell_env.update(resolved["shell_env"])
            diagnostics.extend(resolved["diagnostics"])
            command_sources.update(resolved["command_sources"])
            ready = bool(resolved["ready"])
            updates.update(
                {
                    "test_command": resolved["test_command"],
                    "install_command": resolved["install_command"],
                    "list_packages_command": resolved["list_packages_command"],
                    "install_all_command": resolved["install_all_command"],
                }
            )
            confidence = resolved["confidence"]
        elif manifest.language == "node":
            resolved = self._resolve_node_runtime(manifest, available_tools)
            package_manager = resolved["package_manager"]
            diagnostics.extend(resolved["diagnostics"])
            command_sources.update(resolved["command_sources"])
            ready = bool(resolved["ready"])
            updates.update(
                {
                    "test_command": resolved["test_command"],
                    "install_command": resolved["install_command"],
                    "list_packages_command": resolved["list_packages_command"],
                    "install_all_command": resolved["install_all_command"],
                }
            )
            confidence = resolved["confidence"]
        elif manifest.language == "rust":
            if not available_tools.get("cargo", False):
                diagnostics.append("cargo is not available on PATH.")
                confidence = "low"
                ready = False
            else:
                command_sources.update(
                    {
                        "test_command": "cargo",
                        "install_command": "cargo",
                        "list_packages_command": "cargo",
                        "install_all_command": "cargo",
                    }
                )
        elif manifest.language == "go":
            if not available_tools.get("go", False):
                diagnostics.append("go is not available on PATH.")
                confidence = "low"
                ready = False
            else:
                command_sources.update(
                    {
                        "test_command": "go",
                        "install_command": "go",
                        "list_packages_command": "go",
                        "install_all_command": "go",
                    }
                )

        structure_summary = self._build_structure_summary(Path(manifest.project_root or "."))

        updates.update(
            {
                "package_manager": package_manager,
                "interpreter_command": interpreter_command,
                "shell_env": shell_env,
                "diagnostics": diagnostics,
                "command_sources": command_sources,
                "confidence": confidence,
                "ready": ready,
                "structure_summary": structure_summary,
            }
        )
        return replace(manifest, **updates)

    def _resolve_python_runtime(
        self,
        workspace: Path,
        manifest: EnvironmentManifest,
        available_tools: dict[str, bool],
    ) -> dict[str, object]:
        diagnostics: list[str] = []
        command_sources: dict[str, str] = {}
        shell_env: dict[str, str] = {}
        package_manager = manifest.package_manager or "pip"
        interpreter_command = manifest.interpreter_command
        confidence = "medium"
        ready = True
        test_command = manifest.test_command
        install_command = manifest.install_command
        list_packages_command = manifest.list_packages_command
        install_all_command = manifest.install_all_command

        venv_python = self._workspace_venv_python(workspace)
        src_dir = workspace / "src"
        if venv_python is not None:
            quoted_python = shlex.quote(str(venv_python))
            interpreter_command = quoted_python
            shell_env = self._venv_env(venv_python)
            confidence = "high"
            if manifest.source not in {"MAIKE.md", "AGENTS.md", "CLAUDE.md"}:
                test_command = f"{quoted_python} -m pytest {{path}} --tb=short -q"
                install_command = f"{quoted_python} -m pip install {{package}}"
                list_packages_command = f"{quoted_python} -m pip list --format=columns"
                if install_all_command is None:
                    if (workspace / "requirements.txt").exists():
                        install_all_command = f"{quoted_python} -m pip install -r requirements.txt"
                    elif (workspace / "pyproject.toml").exists():
                        install_all_command = f"{quoted_python} -m pip install -e ."
                command_sources.update(
                    {
                        "test_command": "workspace_venv",
                        "install_command": "workspace_venv",
                        "list_packages_command": "workspace_venv",
                    }
                )
            if src_dir.is_dir():
                shell_env["PYTHONPATH"] = str(src_dir)
        elif package_manager == "uv" and available_tools.get("uv", False):
            interpreter_command = "uv run python"
            confidence = "high"
            if manifest.source not in {"MAIKE.md", "AGENTS.md", "CLAUDE.md"}:
                test_command = "uv run pytest {path} --tb=short -q"
                install_command = "uv add {package}"
                list_packages_command = "uv pip list"
                install_all_command = install_all_command or "uv sync"
                command_sources.update(
                    {
                        "test_command": "uv",
                        "install_command": "uv",
                        "list_packages_command": "uv",
                    }
                )
        elif package_manager == "poetry" and available_tools.get("poetry", False):
            interpreter_command = "poetry run python"
            confidence = "high"
            if manifest.source not in {"MAIKE.md", "AGENTS.md", "CLAUDE.md"}:
                test_command = "poetry run pytest {path} --tb=short -q"
                install_command = "poetry add {package}"
                list_packages_command = "poetry show"
                install_all_command = install_all_command or "poetry install"
                command_sources.update(
                    {
                        "test_command": "poetry",
                        "install_command": "poetry",
                        "list_packages_command": "poetry",
                    }
                )
        elif package_manager == "pipenv" and available_tools.get("pipenv", False):
            interpreter_command = "pipenv run python"
            confidence = "high"
            if manifest.source not in {"MAIKE.md", "AGENTS.md", "CLAUDE.md"}:
                test_command = "pipenv run python -m pytest {path} --tb=short -q"
                install_command = "pipenv install {package}"
                list_packages_command = "pipenv run python -m pip list --format=columns"
                install_all_command = install_all_command or "pipenv install"
                command_sources.update(
                    {
                        "test_command": "pipenv",
                        "install_command": "pipenv",
                        "list_packages_command": "pipenv",
                    }
                )
        else:
            interpreter = self._first_available(available_tools, ("python3", "python"))
            if interpreter is not None:
                quoted_python = shlex.quote(interpreter)
                interpreter_command = quoted_python
                confidence = "medium"
                if manifest.source not in {"MAIKE.md", "AGENTS.md", "CLAUDE.md"}:
                    test_command = f"{quoted_python} -m pytest {{path}} --tb=short -q"
                    install_command = f"{quoted_python} -m pip install {{package}}"
                    list_packages_command = f"{quoted_python} -m pip list --format=columns"
                    if install_all_command is None:
                        if (workspace / "requirements.txt").exists():
                            install_all_command = f"{quoted_python} -m pip install -r requirements.txt"
                        elif (workspace / "pyproject.toml").exists():
                            install_all_command = f"{quoted_python} -m pip install -e ."
                    command_sources.update(
                        {
                            "test_command": interpreter,
                            "install_command": interpreter,
                            "list_packages_command": interpreter,
                        }
                    )
                if src_dir.is_dir():
                    shell_env["PYTHONPATH"] = str(src_dir)
                    diagnostics.append("Detected src/ layout; exporting PYTHONPATH=src for workspace commands.")
            else:
                diagnostics.append("No Python interpreter was found on PATH and no workspace .venv was detected.")
                confidence = "low"
                ready = False

        if package_manager in {"uv", "poetry", "pipenv"} and not available_tools.get(package_manager, False):
            # Fall back to pip instead of failing — the project prefers uv/poetry/pipenv
            # but pip can handle most installs.
            diagnostics.append(
                f"Configured package manager '{package_manager}' is not available; falling back to pip."
            )
            package_manager = "pip"
            interpreter = interpreter_command or "python3"
            install_command = f"{interpreter} -m pip install"
            install_all_command = f"{interpreter} -m pip install -e ."
            list_packages_command = f"{interpreter} -m pip list"

        return {
            "package_manager": package_manager,
            "interpreter_command": interpreter_command,
            "test_command": test_command,
            "install_command": install_command,
            "list_packages_command": list_packages_command,
            "install_all_command": install_all_command,
            "shell_env": shell_env,
            "diagnostics": diagnostics,
            "command_sources": command_sources,
            "confidence": confidence,
            "ready": ready,
        }

    def _resolve_node_runtime(
        self,
        manifest: EnvironmentManifest,
        available_tools: dict[str, bool],
    ) -> dict[str, object]:
        diagnostics: list[str] = []
        command_sources: dict[str, str] = {}
        package_manager = manifest.package_manager or "npm"
        install_command = manifest.install_command
        list_packages_command = manifest.list_packages_command
        install_all_command = manifest.install_all_command
        test_command = manifest.test_command
        confidence = "medium"
        ready = True

        manager = package_manager if available_tools.get(package_manager, False) else self._first_available(
            available_tools,
            ("pnpm", "yarn", "npm"),
        )
        if manager is None:
            diagnostics.append("No Node package manager (pnpm, yarn, npm) is available on PATH.")
            confidence = "low"
            ready = False
            return {
                "package_manager": package_manager,
                "install_command": install_command,
                "list_packages_command": list_packages_command,
                "install_all_command": install_all_command,
                "test_command": test_command,
                "diagnostics": diagnostics,
                "command_sources": command_sources,
                "confidence": confidence,
                "ready": ready,
            }

        if manager != package_manager:
            diagnostics.append(
                f"Preferred Node package manager '{package_manager}' is unavailable; falling back to '{manager}'."
            )
            package_manager = manager
            confidence = "medium"
        else:
            confidence = "high"

        if manifest.source not in {"MAIKE.md", "AGENTS.md", "CLAUDE.md"}:
            if manager == "pnpm":
                install_command = "pnpm add {package}"
                install_all_command = install_all_command or "pnpm install"
                list_packages_command = "pnpm list --depth=0"
                test_command = test_command or "pnpm test"
            elif manager == "yarn":
                install_command = "yarn add {package}"
                install_all_command = install_all_command or "yarn install"
                list_packages_command = "yarn list --depth=0"
                test_command = test_command or "yarn test"
            else:
                install_command = "npm install {package}"
                install_all_command = install_all_command or "npm install"
                list_packages_command = "npm ls --depth=0"
                test_command = test_command or "npm test"
            command_sources.update(
                {
                    "test_command": manager,
                    "install_command": manager,
                    "list_packages_command": manager,
                }
            )

        return {
            "package_manager": package_manager,
            "install_command": install_command,
            "list_packages_command": list_packages_command,
            "install_all_command": install_all_command,
            "test_command": test_command,
            "diagnostics": diagnostics,
            "command_sources": command_sources,
            "confidence": confidence,
            "ready": ready,
        }

    _STRUCTURE_SKIP_DIRS = {
        ".git", "node_modules", ".venv", "__pycache__", "dist", "build",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "venv",
        ".maike", ".eggs", "*.egg-info",
    }

    def _build_structure_summary(self, workspace: Path, max_depth: int = 2) -> list[str]:
        """Walk the top levels of the workspace and return a compact tree."""
        lines: list[str] = []
        try:
            self._walk_structure(workspace, workspace, 0, max_depth, lines)
        except OSError:
            pass
        return lines[:40]  # cap to prevent bloat

    def _walk_structure(
        self, root: Path, current: Path, depth: int, max_depth: int, lines: list[str],
    ) -> None:
        if depth > max_depth or len(lines) > 40:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if name.startswith(".") and name in self._STRUCTURE_SKIP_DIRS:
                continue
            if name in self._STRUCTURE_SKIP_DIRS:
                continue
            indent = "  " * depth
            if entry.is_dir():
                lines.append(f"{indent}{name}/")
                self._walk_structure(root, entry, depth + 1, max_depth, lines)
            elif depth < max_depth:
                lines.append(f"{indent}{name}")

    def _discover_available_tools(self) -> dict[str, bool]:
        names = (
            "python3",
            "python",
            "uv",
            "poetry",
            "pipenv",
            "npm",
            "pnpm",
            "yarn",
            "cargo",
            "go",
            "git",
        )
        return {name: shutil.which(name) is not None for name in names}

    def _workspace_venv_python(self, workspace: Path) -> Path | None:
        candidates = [
            workspace / ".venv" / "bin" / "python",
            workspace / ".venv" / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                # Do NOT resolve() — following the symlink chain would
                # return the system Python path instead of the venv path,
                # causing _venv_env to build incorrect VIRTUAL_ENV/PATH.
                return candidate
        return None

    def _venv_env(self, python_path: Path) -> dict[str, str]:
        venv_root = python_path.parent.parent
        scripts_dir = python_path.parent
        path_value = os.environ.get("PATH", "")
        path_parts = [str(scripts_dir)]
        if path_value:
            path_parts.append(path_value)
        return {
            "VIRTUAL_ENV": str(venv_root),
            "PATH": os.pathsep.join(path_parts),
        }

    def _first_available(
        self,
        available_tools: dict[str, bool],
        names: tuple[str, ...],
    ) -> str | None:
        for name in names:
            if available_tools.get(name, False):
                return name
        return None


# ── Framework-specific guidance ──────────────────────────────────────

_FRAMEWORK_HINTS: dict[str, str] = {
    "fastapi": "FastAPI project — use `uvicorn` for dev server, `pytest-asyncio` for async tests, routes in routers/",
    "django": "Django project — use `manage.py runserver`, `manage.py test`, models in models.py, views in views.py",
    "flask": "Flask project — use `flask run` for dev server, app factory pattern likely in __init__.py",
    "react": "React frontend — use `npm run dev` or `npm start`, components in src/components/",
    "nextjs": "Next.js project — use `npm run dev`, pages in app/ or pages/, API routes in api/",
    "express": "Express.js backend — routes in routes/, middleware in middleware/",
    "vue": "Vue.js frontend — use `npm run dev`, components in src/components/",
    "angular": "Angular project — use `ng serve`, components in src/app/",
    "sqlalchemy": "Uses SQLAlchemy ORM — models define tables, use alembic for migrations if present",
    "gin": "Go Gin framework — handlers in handlers/, routes registered in main.go or router.go",
    "actix": "Rust Actix-Web — handlers in src/handlers/, routes in main.rs or config",
    "axum": "Rust Axum — handlers + routers in src/, state shared via Extension",
}


_PYTHON_REQUIRES_PATTERNS = (
    # setup.py: python_requires="..."
    re.compile(r"""python_requires\s*=\s*["']([^"']+)["']"""),
    # setup.cfg: python_requires = ...
    re.compile(r"""(?m)^\s*python_requires\s*=\s*(.+)\s*$"""),
    # pyproject.toml: requires-python = "..."
    re.compile(r"""requires-python\s*=\s*["']([^"']+)["']"""),
)

# Tuple-style version constants near python_requires.  Django et al. do:
#   REQUIRED_PYTHON = (3, 6)
#   python_requires='>={}.{}'.format(*REQUIRED_PYTHON)
_PYTHON_VERSION_TUPLE = re.compile(
    r"""\b([A-Z][A-Z_]*PYTHON[A-Z_]*|MIN_VERSION)\s*=\s*"""
    r"""\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*\d+)?\s*\)""",
)

# Format-marker characters that indicate a template, not a literal string.
_TEMPLATE_MARKERS = frozenset("{}()%+")


def _extract_python_requires(workspace: Path) -> str | None:
    """Extract a Python version constraint from project config.

    Reads setup.py, setup.cfg, and pyproject.toml and returns the first
    ``python_requires`` / ``requires-python`` value found.  Resolves
    Django-style format templates (``'>={}.{}'.format(*REQUIRED_PYTHON)``)
    by locating the tuple constant in the same file.  Returns ``None``
    if no constraint can be determined.

    Used to warn agents about host-vs-target Python mismatches in
    SWE-bench eval, where a Python 3.13/3.14 host hitting ``cgi`` import
    errors gets misread as a real bug needing a fix.
    """
    candidates = ("setup.py", "setup.cfg", "pyproject.toml")
    for fname in candidates:
        path = workspace / fname
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in _PYTHON_REQUIRES_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            value = m.group(1).strip().strip("'\"")
            if not any(c in value for c in _TEMPLATE_MARKERS):
                return value  # plain literal — done
            resolved = _resolve_python_version_template(text, value)
            if resolved:
                return resolved
            # Template not resolvable from this file; keep scanning others.
        # Fall back: any ``X_PYTHON = (M, N)`` tuple even without an explicit
        # python_requires line.  Better than nothing for projects that
        # define the version programmatically.
        m = _PYTHON_VERSION_TUPLE.search(text)
        if m:
            return f">={m.group(2)}.{m.group(3)}"
    return None


def _resolve_python_version_template(text: str, template: str) -> str | None:
    """Substitute a ``>={}.{}`` template using a tuple constant in ``text``."""
    m = _PYTHON_VERSION_TUPLE.search(text)
    if not m:
        return None
    major, minor = m.group(2), m.group(3)
    if "{}.{}" in template:
        return template.replace("{}.{}", f"{major}.{minor}", 1)
    if template.count("{}") >= 2:
        return template.replace("{}", major, 1).replace("{}", minor, 1)
    if template.count("{}") == 1:
        return template.replace("{}", f"{major}.{minor}")
    return None


def _framework_guidance(frameworks: list[str]) -> str:
    """Return practical hints for detected frameworks."""
    hints = []
    for fw in frameworks:
        hint = _FRAMEWORK_HINTS.get(fw)
        if hint:
            hints.append(f"- {hint}")
    if not hints:
        return ""
    return "Framework hints:\n" + "\n".join(hints)
