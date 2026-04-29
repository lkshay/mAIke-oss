"""Tests for framework detection in EnvironmentProbe."""

from pathlib import Path

from maike.runtime.probe import EnvironmentProbe


def test_django_detection_via_manage_py(tmp_path: Path):
    (tmp_path / "manage.py").write_text("#!/usr/bin/env python\nimport django\n")
    (tmp_path / "requirements.txt").write_text("django==5.0\n")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert "django" in manifest.frameworks


def test_fastapi_detection_via_requirements(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("fastapi==0.100.0\nuvicorn\n")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert "fastapi" in manifest.frameworks


def test_react_detection_via_package_json(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"dependencies": {"react": "^18.0.0"}}')
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert "react" in manifest.frameworks


def test_nextjs_detection_via_config(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"dependencies": {"next": "^14.0.0"}}')
    (tmp_path / "next.config.js").write_text("module.exports = {}")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert "nextjs" in manifest.frameworks


def test_no_framework_detected(tmp_path: Path):
    (tmp_path / "main.py").write_text("print('hello')\n")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert manifest.frameworks == []


def test_multiple_frameworks(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text("fastapi\nsqlalchemy\npydantic\n")
    probe = EnvironmentProbe()
    manifest = probe.probe(tmp_path)
    assert "fastapi" in manifest.frameworks
    assert "sqlalchemy" in manifest.frameworks
    assert "pydantic" in manifest.frameworks
