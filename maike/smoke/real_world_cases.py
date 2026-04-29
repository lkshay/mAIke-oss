"""Real-world eval cases using actual open source repositories.

These scenarios go beyond toy examples and test the agent against real
codebases with real test suites, real architecture, and real complexity.

Each scenario:
1. Clones a specific commit of an open source repo
2. Poses a realistic task (bug fix, feature addition, test improvement)
3. Verifies the outcome using the repo's own test suite + custom checks
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RealWorldCase:
    """A real-world eval case backed by an actual open source repo."""

    name: str
    description: str
    repo_url: str
    commit_sha: str            # Pin to specific commit for reproducibility
    task: str                   # What the agent is asked to do
    setup: Callable[[Path], None] | None = None  # Post-clone setup (inject bug, etc.)
    verify: Callable[[Path], None] | None = None  # Verification (tests + custom)
    budget: float = 10.0
    tags: tuple[str, ...] = ()
    language: str = "python"
    install_cmd: str | None = None  # How to install deps (pip install -e ., etc.)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_repo(url: str, sha: str, workspace: Path) -> None:
    """Clone at a specific commit.  Uses shallow clone for speed."""
    # Clone with enough depth to find the target commit
    subprocess.run(
        ["git", "clone", url, str(workspace)],
        capture_output=True, check=True, timeout=180,
    )
    subprocess.run(
        ["git", "checkout", sha],
        cwd=workspace, capture_output=True, check=True,
    )


def _run_pytest(workspace: Path, test_path: str = "tests/", extra_args: list[str] | None = None) -> tuple[int, int, str]:
    """Run pytest in the workspace's venv and return (passed, failed, output)."""
    venv_python = workspace / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    cmd = [str(venv_python), "-m", "pytest", test_path, "-v", "--tb=short", "-q"]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd, cwd=workspace, capture_output=True, text=True, timeout=300,
    )
    output = result.stdout + result.stderr

    # Parse pytest summary line: "X passed, Y failed"
    passed = failed = 0
    for line in output.splitlines():
        if "passed" in line or "failed" in line:
            import re
            m_passed = re.search(r"(\d+) passed", line)
            m_failed = re.search(r"(\d+) failed", line)
            if m_passed:
                passed = int(m_passed.group(1))
            if m_failed:
                failed = int(m_failed.group(1))

    return passed, failed, output


def _install_in_venv(workspace: Path, install_cmd: str | None = None) -> None:
    """Create venv and install the project."""
    subprocess.run(
        [sys.executable, "-m", "venv", str(workspace / ".venv")],
        check=True, capture_output=True, timeout=60,
    )
    pip = str(workspace / ".venv" / "bin" / "pip")
    cmd = install_cmd or "pip install -e '.[test]'"
    # Use the venv's pip
    subprocess.run(
        [pip, "install", "-e", ".[test]"],
        cwd=workspace, capture_output=True, timeout=300,
    )
    # Also try without [test] extra
    subprocess.run(
        [pip, "install", "-e", "."],
        cwd=workspace, capture_output=True, timeout=300,
    )
    # Install pytest separately
    subprocess.run(
        [pip, "install", "pytest"],
        cwd=workspace, capture_output=True, timeout=120,
    )


# ---------------------------------------------------------------------------
# Case 1: prettytable — Add CSV export feature
# ---------------------------------------------------------------------------

def _setup_prettytable_feature(workspace: Path) -> None:
    """prettytable has no built-in CSV export. The agent should add one."""
    _install_in_venv(workspace)
    # prettytable's test suite needs pytest_lazy_fixtures which isn't in [test] extras
    pip = str(workspace / ".venv" / "bin" / "pip")
    subprocess.run(
        [pip, "install", "pytest_lazy_fixtures"],
        cwd=workspace, capture_output=True, timeout=120,
    )


def _verify_prettytable_feature(workspace: Path) -> None:
    """Verify CSV export was added and existing tests still pass."""
    # Check existing tests still pass (PASS_TO_PASS)
    passed, failed, output = _run_pytest(workspace, "tests/")
    assert failed == 0, f"Existing tests broke: {failed} failures\n{output[-500:]}"
    assert passed > 0, f"No tests ran\n{output[-500:]}"

    # Check the feature exists (FAIL_TO_PASS equivalent)
    venv_python = str(workspace / ".venv" / "bin" / "python")
    result = subprocess.run(
        [venv_python, "-c", """
from prettytable import PrettyTable
t = PrettyTable()
t.field_names = ["Name", "Age", "City"]
t.add_row(["Alice", 30, "NYC"])
t.add_row(["Bob", 25, "SF"])
csv_output = t.get_csv_string()
assert "Name" in csv_output, f"CSV header missing: {csv_output}"
assert "Alice" in csv_output, f"CSV data missing: {csv_output}"
assert "," in csv_output, f"No commas in CSV: {csv_output}"
lines = [l for l in csv_output.strip().splitlines() if l.strip()]
assert len(lines) == 3, f"Expected 3 lines (header + 2 rows), got {len(lines)}: {csv_output}"
print("CSV export verified successfully")
"""],
        cwd=workspace, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"CSV export verification failed:\n{result.stdout}\n{result.stderr}"


PRETTYTABLE_CSV = RealWorldCase(
    name="prettytable-csv-export",
    description="Add CSV export to prettytable (get_csv_string method)",
    repo_url="https://github.com/jazzband/prettytable.git",
    commit_sha="a5ec071",  # Recent stable commit
    task=(
        "Add a get_csv_string() method to PrettyTable that returns the table "
        "data as a CSV-formatted string using Python's csv module. "
        "The output should include a header row with field names and one data "
        "row per table row. Add tests for this new method in the test file. "
        "Make sure all existing tests still pass."
    ),
    setup=_setup_prettytable_feature,
    verify=_verify_prettytable_feature,
    budget=10.0,
    tags=("real-world", "feature", "medium"),
)


# ---------------------------------------------------------------------------
# Case 2: colorama — Fix a seeded regression bug
# ---------------------------------------------------------------------------

def _setup_colorama_bug(workspace: Path) -> None:
    """Inject a subtle bug into colorama's AnsiToWin32 stream wrapper."""
    _install_in_venv(workspace)

    # Inject a bug: make Style.RESET_ALL not actually reset
    init_path = workspace / "colorama" / "ansi.py"
    content = init_path.read_text()
    # Change RESET_ALL from '\033[0m' to '\033[1m' (bold instead of reset)
    bugged = content.replace("'\\033[0m'", "'\\033[1m'")
    if bugged != content:
        init_path.write_text(bugged)
        subprocess.run(
            ["git", "add", "-A"], cwd=workspace, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "introduced regression"],
            cwd=workspace, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
        )


def _verify_colorama_bug(workspace: Path) -> None:
    """Verify the RESET_ALL value is restored.

    We check the runtime value, not the source literal, because colorama
    stores ANSI codes as integers that get formatted into escape sequences
    by the AnsiCodes metaclass.
    """
    venv_python = str(workspace / ".venv" / "bin" / "python")
    result = subprocess.run(
        [venv_python, "-c", """
from colorama import Style
actual = Style.RESET_ALL
# Accept both string representations — they're the same bytes
assert actual in ('\\033[0m', '\\x1b[0m'), f"RESET_ALL is wrong: {repr(actual)}"
print("RESET_ALL verified correctly")
"""],
        cwd=workspace, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"Bug not fixed:\n{result.stdout}\n{result.stderr}"


COLORAMA_BUG = RealWorldCase(
    name="colorama-reset-bug",
    description="Fix a seeded regression where Style.RESET_ALL sends bold instead of reset",
    repo_url="https://github.com/tartley/colorama.git",
    commit_sha="406153f",  # Recent stable
    task=(
        "The test suite is failing. Style.RESET_ALL is supposed to send the "
        "ANSI reset sequence '\\033[0m' but it's currently sending '\\033[1m' "
        "(bold). Find and fix the bug. Run the tests to confirm the fix."
    ),
    setup=_setup_colorama_bug,
    verify=_verify_colorama_bug,
    budget=5.0,
    tags=("real-world", "bug-fix", "small"),
)


# ---------------------------------------------------------------------------
# Case 3: Build a CLI tool on top of an existing library (markupsafe)
# ---------------------------------------------------------------------------

def _setup_markupsafe_cli(workspace: Path) -> None:
    """Clone markupsafe and install it."""
    # markupsafe uses C extensions; install the wheel from PyPI instead
    subprocess.run(
        [sys.executable, "-m", "venv", str(workspace / ".venv")],
        check=True, capture_output=True, timeout=60,
    )
    pip = str(workspace / ".venv" / "bin" / "pip")
    subprocess.run([pip, "install", "markupsafe", "pytest"], cwd=workspace, capture_output=True, timeout=120)


def _verify_markupsafe_cli(workspace: Path) -> None:
    """Verify the CLI tool works."""
    venv_python = str(workspace / ".venv" / "bin" / "python")

    # Test the CLI: escape HTML from stdin
    result = subprocess.run(
        [venv_python, "-m", "markupsafe_cli", "escape"],
        input="<script>alert('xss')</script>",
        cwd=workspace, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert "&lt;script&gt;" in result.stdout, f"HTML not escaped: {result.stdout}"
    assert "<script>" not in result.stdout, f"Raw HTML in output: {result.stdout}"

    # Test unescape
    result = subprocess.run(
        [venv_python, "-m", "markupsafe_cli", "unescape"],
        input="&lt;b&gt;hello&lt;/b&gt;",
        cwd=workspace, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"Unescape CLI failed:\n{result.stderr}"
    assert "<b>hello</b>" in result.stdout, f"Not unescaped: {result.stdout}"

    # Run the agent's own tests if they exist
    for test_path in ["test_markupsafe_cli.py", "tests/test_markupsafe_cli.py", "tests/test_cli.py"]:
        if (workspace / test_path).exists():
            passed, failed, output = _run_pytest(workspace, test_path)
            assert passed > 0, f"Agent tests didn't pass\n{output[-500:]}"
            break


MARKUPSAFE_CLI = RealWorldCase(
    name="markupsafe-cli-wrapper",
    description="Build a CLI wrapper around markupsafe for HTML escaping/unescaping from stdin",
    repo_url="https://github.com/pallets/markupsafe.git",
    commit_sha="b2e4d9c",  # Recent stable
    task=(
        "Create a CLI tool called markupsafe_cli (as a Python module that can "
        "be run with `python -m markupsafe_cli`) that wraps the existing "
        "markupsafe library. It should support two commands:\n"
        "  1. `python -m markupsafe_cli escape` — reads stdin and outputs "
        "HTML-escaped text\n"
        "  2. `python -m markupsafe_cli unescape` — reads stdin and outputs "
        "HTML-unescaped text\n"
        "Use argparse for the CLI. Write tests for the new CLI module. "
        "Do NOT modify any existing markupsafe code. Make sure all existing "
        "tests still pass."
    ),
    setup=_setup_markupsafe_cli,
    verify=_verify_markupsafe_cli,
    budget=10.0,
    tags=("real-world", "feature", "medium"),
)


# ---------------------------------------------------------------------------
# Case 4: Fix a real multi-file bug (Box nested merge)
# ---------------------------------------------------------------------------

def _setup_box_merge_bug(workspace: Path) -> None:
    """Seed Box with a bug in merge_update for nested dicts."""
    _install_in_venv(workspace)

    # Add a failing test that demonstrates the bug
    test_content = '''"""Test for nested merge_update behavior."""
import pytest
from box import Box


def test_merge_update_nested_dicts():
    """merge_update should recursively merge nested dictionaries."""
    d1 = Box({"app": {"db": {"host": "localhost", "port": 5432}}})
    d2 = Box({"app": {"db": {"port": 5433, "name": "mydb"}}})
    d1.merge_update(d2)
    assert d1.app.db.host == "localhost", "Existing nested key was lost"
    assert d1.app.db.port == 5433, "Nested key was not updated"
    assert d1.app.db.name == "mydb", "New nested key was not added"


def test_merge_update_preserves_top_level():
    """merge_update should preserve unrelated top-level keys."""
    d1 = Box({"name": "app", "version": "1.0", "config": {"debug": True}})
    d2 = Box({"config": {"log_level": "INFO"}})
    d1.merge_update(d2)
    assert d1.name == "app"
    assert d1.version == "1.0"
    assert d1.config.debug is True
    assert d1.config.log_level == "INFO"
'''
    (workspace / "test_nested_merge.py").write_text(test_content)
    subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add failing merge test"],
        cwd=workspace, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
    )


def _verify_box_merge(workspace: Path) -> None:
    """Verify merge_update works for nested dicts."""
    # The new tests should pass
    passed, failed, output = _run_pytest(workspace, "test_nested_merge.py")
    assert failed == 0, f"Nested merge tests still failing:\n{output[-500:]}"
    assert passed >= 2, f"Expected at least 2 passing tests\n{output[-500:]}"

    # Existing tests should still pass
    passed, failed, output = _run_pytest(workspace, "test/")
    assert failed == 0, f"Existing tests broke: {failed} failures\n{output[-500:]}"


BOX_MERGE = RealWorldCase(
    name="box-nested-merge",
    description="Fix Box.merge_update to recursively merge nested dictionaries",
    repo_url="https://github.com/cdgriffith/Box.git",
    commit_sha="a4c10e9",  # v7.x
    task=(
        "There's a test file test_nested_merge.py that demonstrates a bug: "
        "Box.merge_update() doesn't recursively merge nested dictionaries — "
        "it replaces them entirely. Fix the merge_update method in box.py "
        "to recursively merge when both the existing and new values are "
        "dictionaries. Run both the new tests AND the existing test suite "
        "to make sure nothing is broken."
    ),
    setup=_setup_box_merge_bug,
    verify=_verify_box_merge,
    budget=10.0,
    tags=("real-world", "bug-fix", "medium"),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REAL_WORLD_CASES: dict[str, RealWorldCase] = {
    "prettytable-csv-export": PRETTYTABLE_CSV,
    "colorama-reset-bug": COLORAMA_BUG,
    "markupsafe-cli-wrapper": MARKUPSAFE_CLI,
    "box-nested-merge": BOX_MERGE,
}
