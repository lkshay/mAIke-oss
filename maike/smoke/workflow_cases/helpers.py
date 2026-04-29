"""Shared helpers for seeded workflow cases."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def import_module(workspace: Path, module_name: str):
    root_name = module_name.split(".", 1)[0]
    for loaded_name in tuple(sys.modules):
        if loaded_name == root_name or loaded_name.startswith(f"{root_name}."):
            sys.modules.pop(loaded_name, None)
    importlib.invalidate_caches()
    sys.path.insert(0, str(workspace))
    try:
        return importlib.import_module(module_name)
    finally:
        if sys.path and sys.path[0] == str(workspace):
            sys.path.pop(0)


def run_checked(args: list[str], workspace: Path, *, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def run_expect_failure(args: list[str], workspace: Path, *, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        raise AssertionError(
            f"{label} unexpectedly succeeded.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _workspace_python(workspace: Path) -> str:
    """Return the Python interpreter for the workspace's venv, falling back to sys.executable."""
    venv_python = workspace / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_pytest(workspace: Path, *, label: str = "pytest") -> None:
    run_checked([_workspace_python(workspace), "-m", "pytest", "-q"], workspace, label=label)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_readme_mentions(workspace: Path, *phrases: str) -> None:
    text = (workspace / "README.md").read_text(encoding="utf-8").lower()
    for phrase in phrases:
        assert phrase.lower() in text, f"README.md does not mention {phrase!r}"


def snapshot_workspace_files(workspace: Path) -> tuple[str, ...]:
    ignored_dirs = {".git", ".maike", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}
    files: list[str] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.relative_to(workspace).parts):
            continue
        files.append(path.relative_to(workspace).as_posix())
    return tuple(sorted(files))


def snapshot_workspace_hashes(workspace: Path) -> dict[str, str]:
    """Return {relative_path: content_hash} for all source files.

    Uses MD5 for speed — this is for change detection, not security.
    """
    import hashlib
    ignored_dirs = {".git", ".maike", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}
    result: dict[str, str] = {}
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.relative_to(workspace).parts):
            continue
        rel = path.relative_to(workspace).as_posix()
        try:
            content = path.read_bytes()
            result[rel] = hashlib.md5(content).hexdigest()
        except OSError:
            pass
    return result


def reset_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
