"""Tests for environment probe enrichments: typecheck_command and structure_summary."""

from maike.runtime.probe import EnvironmentManifest, EnvironmentProbe


def test_context_block_is_descriptive_not_prescriptive():
    """Context block should report what was detected, not prescribe commands."""
    manifest = EnvironmentManifest(
        language="python",
        package_manager="poetry",
        lint_command="ruff check .",
        typecheck_command="mypy .",
    )
    block = manifest.to_context_block()
    # Should include descriptive info
    assert "Language: python" in block
    assert "Package manager: poetry" in block
    # Should NOT include prescriptive commands
    assert "Lint:" not in block
    assert "Type check:" not in block
    assert "Test:" not in block
    assert "Install package:" not in block


def test_context_block_low_confidence_shows_fallback_hint():
    """When confidence is low, tell the agent to read config files."""
    manifest = EnvironmentManifest(language="unknown", confidence="low")
    block = manifest.to_context_block()
    assert "confidence is low" in block
    assert "README" in block


def test_structure_summary_included_in_context_block():
    manifest = EnvironmentManifest(
        language="python",
        structure_summary=["src/", "  app.py", "tests/", "  test_app.py"],
    )
    block = manifest.to_context_block()
    assert "## Project Structure" in block
    assert "src/" in block
    assert "  app.py" in block


def test_structure_summary_empty_omitted_from_context_block():
    manifest = EnvironmentManifest(language="python")
    block = manifest.to_context_block()
    assert "## Project Structure" not in block


def test_probe_detects_mypy_ini_typecheck(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "mypy.ini").write_text("[mypy]\n")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert manifest.typecheck_command == "mypy ."


def test_probe_detects_pyright_typecheck(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "pyrightconfig.json").write_text("{}")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert manifest.typecheck_command == "pyright ."


def test_probe_detects_pyproject_mypy_typecheck(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n\n[tool.mypy]\nstrict = true\n")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert manifest.typecheck_command == "mypy ."


def test_probe_detects_tsconfig_typecheck(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "x"}')
    (tmp_path / "tsconfig.json").write_text("{}")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert manifest.typecheck_command == "npx tsc --noEmit"


def test_probe_no_typecheck_when_absent(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert manifest.typecheck_command is None


def test_structure_summary_built_during_resolve(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("print('hi')")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("def test_hi(): pass")

    probe = EnvironmentProbe()
    manifest = probe.resolve(tmp_path)
    assert len(manifest.structure_summary) > 0
    joined = "\n".join(manifest.structure_summary)
    assert "src/" in joined
    assert "tests/" in joined


def test_structure_summary_skips_hidden_and_venv(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "app.py").write_text("print('hi')")

    probe = EnvironmentProbe()
    manifest = probe.resolve(tmp_path)
    joined = "\n".join(manifest.structure_summary)
    assert ".git" not in joined
    assert ".venv" not in joined
    assert "node_modules" not in joined
    assert "__pycache__" not in joined
    assert "app.py" in joined


def test_structure_summary_capped_at_40_lines(tmp_path):
    """Even with many files, the summary doesn't exceed 40 lines."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    for i in range(60):
        (tmp_path / f"file_{i:03d}.py").write_text(f"# {i}")

    probe = EnvironmentProbe()
    manifest = probe.resolve(tmp_path)
    assert len(manifest.structure_summary) <= 40
