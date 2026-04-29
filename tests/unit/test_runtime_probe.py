from pathlib import Path

from maike.runtime.probe import EnvironmentProbe


def test_environment_probe_parses_maike_commands_file(tmp_path):
    (tmp_path / "MAIKE.md").write_text(
        "\n".join(
            [
                "# Project",
                "Language: python",
                "## Commands",
                "- test: poetry run pytest -q",
                "- install package: poetry add {package}",
                "- list packages: poetry show",
                "- install all: poetry install",
                "- lint: ruff check .",
            ]
        ),
        encoding="utf-8",
    )

    manifest = EnvironmentProbe().probe(tmp_path)

    assert manifest.source == "MAIKE.md"
    assert manifest.language == "python"
    assert manifest.test_command == "poetry run pytest -q"
    assert manifest.install_command == "poetry add {package}"
    assert manifest.list_packages_command == "poetry show"
    assert manifest.install_all_command == "poetry install"
    assert manifest.lint_command == "ruff check ."


def test_environment_probe_cli_language_override_wins_over_config_file(tmp_path):
    (tmp_path / "MAIKE.md").write_text(
        "\n".join(
            [
                "# Project",
                "Language: python",
                "## Commands",
                "- test: poetry run pytest -q",
                "- install package: poetry add {package}",
            ]
        ),
        encoding="utf-8",
    )

    manifest = EnvironmentProbe().resolve(tmp_path, language_override="node")

    assert manifest.language == "node"
    assert manifest.source == "override"
    assert manifest.install_command == "npm install {package}"
    assert manifest.list_packages_command == "npm ls --depth=0"
    assert manifest.package_manager == "npm"


def test_environment_probe_empty_workspace_falls_back_to_python_defaults(tmp_path):
    manifest = EnvironmentProbe().resolve(tmp_path)

    assert manifest.source == "fallback"
    assert manifest.language == "python"
    assert manifest.interpreter_command in {"python3", "python"}
    assert manifest.test_command == f"{manifest.interpreter_command} -m pytest {{path}} --tb=short -q"
    assert manifest.install_command == f"{manifest.interpreter_command} -m pip install {{package}}"
    assert manifest.list_packages_command == f"{manifest.interpreter_command} -m pip list --format=columns"


def test_environment_probe_detects_node_workspace_from_lockfile_and_package_json(tmp_path):
    (tmp_path / "package.json").write_text('{"name":"demo","scripts":{"test":"npm test"}}\n', encoding="utf-8")
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")

    manifest = EnvironmentProbe().resolve(tmp_path)

    assert manifest.source == "probe"
    assert manifest.language == "node"
    if manifest.package_manager == "pnpm":
        assert manifest.install_command == "pnpm add {package}"
        assert manifest.install_all_command == "pnpm install"
        assert manifest.list_packages_command == "pnpm list --depth=0"
        assert manifest.test_command == "npm test"
    else:
        assert manifest.package_manager in {"npm", "yarn"}
        assert manifest.ready is True
        assert manifest.diagnostics


def test_environment_probe_prefers_workspace_venv_for_python(tmp_path):
    python_path = tmp_path / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()

    manifest = EnvironmentProbe().resolve(tmp_path)

    assert manifest.interpreter_command == str(python_path.resolve())
    assert manifest.test_command.startswith(str(python_path.resolve()))
    assert manifest.shell_env["VIRTUAL_ENV"] == str((tmp_path / ".venv").resolve())
    assert manifest.shell_env["PATH"].split(":")[0] == str((tmp_path / ".venv" / "bin").resolve())
    assert manifest.shell_env["PYTHONPATH"] == str(tmp_path / "src")
    assert manifest.ready is True


def test_environment_probe_marks_python_runtime_unready_when_no_interpreter(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setattr("maike.runtime.probe.shutil.which", lambda name: None)

    manifest = EnvironmentProbe().resolve(tmp_path)

    assert manifest.language == "python"
    assert manifest.ready is False
    assert "No Python interpreter was found on PATH" in manifest.diagnostics[0]


def test_environment_probe_falls_back_when_preferred_node_manager_missing(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text('{"name":"demo"}\n', encoding="utf-8")
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")

    def fake_which(name: str):
        return "/usr/bin/npm" if name == "npm" else None

    monkeypatch.setattr("maike.runtime.probe.shutil.which", fake_which)

    manifest = EnvironmentProbe().resolve(tmp_path)

    assert manifest.package_manager == "npm"
    assert manifest.install_command == "npm install {package}"
    assert manifest.ready is True
    assert "falling back to 'npm'" in manifest.diagnostics[0]


# --- Framework detection tests ---


def test_detects_django_from_requirements_txt(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "django>=4.2\ncelery>=5.3\n", encoding="utf-8"
    )
    manifest = EnvironmentProbe().probe(tmp_path)
    assert "django" in manifest.frameworks
    assert "celery" in manifest.frameworks


def test_detects_react_from_package_json(tmp_path):
    import json

    pkg = {
        "name": "myapp",
        "dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"},
        "devDependencies": {"next": "^14.0.0"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
    manifest = EnvironmentProbe().probe(tmp_path)
    assert "react" in manifest.frameworks
    assert "nextjs" in manifest.frameworks


def test_detects_gin_from_go_mod(tmp_path):
    (tmp_path / "go.mod").write_text(
        "module example.com/myapp\n\ngo 1.21\n\n"
        "require (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n",
        encoding="utf-8",
    )
    manifest = EnvironmentProbe().probe(tmp_path)
    assert "gin" in manifest.frameworks


def test_empty_workspace_returns_no_frameworks(tmp_path):
    manifest = EnvironmentProbe().probe(tmp_path)
    assert manifest.frameworks == []


def test_missing_dependency_files_returns_no_frameworks(tmp_path):
    # Only a README, no dependency manifests
    (tmp_path / "README.md").write_text("# Hello\n", encoding="utf-8")
    manifest = EnvironmentProbe().probe(tmp_path)
    assert manifest.frameworks == []


def test_detects_multiple_frameworks_in_same_project(tmp_path):
    # Framework detection is scoped to the detected primary language.
    # A Python project with requirements.txt detects Python frameworks.
    (tmp_path / "requirements.txt").write_text(
        "fastapi>=0.100\ncelery>=5.3\n", encoding="utf-8"
    )
    manifest = EnvironmentProbe().probe(tmp_path)
    assert "fastapi" in manifest.frameworks
    assert "celery" in manifest.frameworks


def test_detects_rust_framework_from_cargo_toml(tmp_path):
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "myapp"\nversion = "0.1.0"\n\n'
        '[dependencies]\naxum = "0.7"\ntokio = { version = "1", features = ["full"] }\n',
        encoding="utf-8",
    )
    manifest = EnvironmentProbe().probe(tmp_path)
    assert "axum" in manifest.frameworks


def test_frameworks_survive_resolve(tmp_path):
    """Verify that frameworks detected in probe() are carried through resolve()."""
    (tmp_path / "requirements.txt").write_text(
        "flask>=3.0\n", encoding="utf-8"
    )
    manifest = EnvironmentProbe().resolve(tmp_path)
    assert "flask" in manifest.frameworks


def test_frameworks_appear_in_context_block(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "django>=4.2\nfastapi>=0.100\n", encoding="utf-8"
    )
    manifest = EnvironmentProbe().probe(tmp_path)
    block = manifest.to_context_block()
    assert "Frameworks: django, fastapi" in block
