"""Legacy lightweight workflow cases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from maike.smoke.workflow_cases.helpers import assert_readme_mentions, import_module, run_pytest, write_text


def _verify_greenfield_workspace(workspace: Path) -> None:
    readme_path = workspace / "README.md"
    assert readme_path.exists(), "README.md was not created"
    readme_text = readme_path.read_text(encoding="utf-8").lower()
    assert "hello world" in readme_text, "README.md does not mention Hello World"

    source_files = [
        path
        for path in workspace.iterdir()
        if path.is_file() and path.suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs"}
    ]
    assert source_files, "No top-level source file was generated"
    combined_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore").lower() for path in source_files)
    assert "hello" in combined_text, "Generated source files do not contain a hello-style implementation"


def _seed_editing_workspace(workspace: Path) -> None:
    write_text(
        workspace / "calculator.py",
        """def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    print(add(2, 3))
""",
    )
    write_text(
        workspace / "README.md",
        """# Calculator

This small calculator currently supports addition only.
""",
    )


def _verify_editing_workspace(workspace: Path) -> None:
    module = import_module(workspace, "calculator")
    assert hasattr(module, "subtract"), "calculator.subtract was not added"
    assert module.subtract(7, 2) == 5, "calculator.subtract returned the wrong value"
    assert_readme_mentions(workspace, "subtract")


def _verify_multifile_workspace(workspace: Path) -> None:
    required_files = [
        workspace / "task_store.py",
        workspace / "task_cli.py",
        workspace / "test_task_store.py",
        workspace / "README.md",
    ]
    for path in required_files:
        assert path.exists(), f"Missing expected file: {path.name}"

    add_result = subprocess.run(
        [sys.executable, "task_cli.py", "add", "buy milk"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if add_result.returncode != 0:
        raise AssertionError(
            "task_cli.py add failed.\n"
            f"stdout:\n{add_result.stdout}\n"
            f"stderr:\n{add_result.stderr}"
        )

    tasks_path = workspace / "tasks.json"
    assert tasks_path.exists(), "tasks.json was not created by the CLI"
    tasks_text = tasks_path.read_text(encoding="utf-8").lower()
    assert "buy milk" in tasks_text, "tasks.json does not contain the added task"

    list_result = subprocess.run(
        [sys.executable, "task_cli.py", "list"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if list_result.returncode != 0:
        raise AssertionError(
            "task_cli.py list failed.\n"
            f"stdout:\n{list_result.stdout}\n"
            f"stderr:\n{list_result.stderr}"
        )
    assert "buy milk" in list_result.stdout.lower(), "CLI list output does not include the saved task"
    run_pytest(workspace, label="generated workspace pytest")
    assert_readme_mentions(workspace, "task_cli.py", "pytest")


def _seed_debugging_workspace(workspace: Path) -> None:
    write_text(
        workspace / "calculator.py",
        """def divide(a: int, b: int) -> float:
    if a == 0:
        return 1
    return a / b
""",
    )
    write_text(
        workspace / "test_calculator.py",
        """from calculator import divide


def test_divide_zero_numerator():
    assert divide(0, 5) == 0


def test_divide_normal():
    assert divide(8, 2) == 4
""",
    )


def _verify_debugging_workspace(workspace: Path) -> None:
    module = import_module(workspace, "calculator")
    assert module.divide(0, 5) == 0, "divide(0, 5) is still incorrect"
    run_pytest(workspace, label="debugging workspace pytest")


def _seed_editing_multifile_workspace(workspace: Path) -> None:
    write_text(
        workspace / "task_store.py",
        """import json
from pathlib import Path


class TaskStore:
    def __init__(self, path: str = "tasks.json") -> None:
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, tasks: list[dict]) -> None:
        self.path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    def add(self, title: str) -> dict:
        tasks = self.load()
        task = {"title": title, "completed": False}
        tasks.append(task)
        self.save(tasks)
        return task
""",
    )
    write_text(
        workspace / "task_cli.py",
        """import sys

from task_store import TaskStore


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    store = TaskStore()
    if not args:
        print("usage: task_cli.py <add|list> [args]")
        return 1
    if args[0] == "add" and len(args) > 1:
        store.add(" ".join(args[1:]))
        print("added")
        return 0
    if args[0] == "list":
        for index, task in enumerate(store.load(), start=1):
            print(f"{index}. {task['title']}")
        return 0
    print("unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
""",
    )
    write_text(
        workspace / "test_task_store.py",
        """from task_store import TaskStore


def test_add_persists_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.json")
    task = store.add("write docs")

    assert task["title"] == "write docs"
    assert store.load()[0]["completed"] is False
""",
    )
    write_text(
        workspace / "README.md",
        """# Task Tracker

Supports:
- `python task_cli.py add <title>`
- `python task_cli.py list`
""",
    )


def _verify_editing_multifile_workspace(workspace: Path) -> None:
    subprocess.run([sys.executable, "task_cli.py", "add", "buy milk"], cwd=workspace, check=True, capture_output=True)
    subprocess.run([sys.executable, "task_cli.py", "done", "1"], cwd=workspace, check=True, capture_output=True)
    tasks_text = (workspace / "tasks.json").read_text(encoding="utf-8").lower()
    assert '"completed": true' in tasks_text, "Completed task state was not persisted"
    run_pytest(workspace, label="editing-multifile pytest")
    assert_readme_mentions(workspace, "done")


def _seed_specialist_debugging_workspace(workspace: Path) -> None:
    write_text(
        workspace / "config_loader.py",
        """def load_port(env: dict[str, str | None]) -> int:
    raw = env.get("APP_PORT")
    if not raw:
        return 8080
    if raw == "0":
        return 8080
    return int(raw)
""",
    )
    write_text(
        workspace / "test_config_loader.py",
        """from config_loader import load_port


def test_missing_port_uses_default():
    assert load_port({}) == 8080


def test_zero_port_is_preserved():
    assert load_port({"APP_PORT": "0"}) == 0


def test_explicit_port_is_parsed():
    assert load_port({"APP_PORT": "9000"}) == 9000
""",
    )


def _verify_specialist_debugging_workspace(workspace: Path) -> None:
    module = import_module(workspace, "config_loader")
    assert module.load_port({"APP_PORT": "0"}) == 0, "APP_PORT=0 is still treated incorrectly"
    run_pytest(workspace, label="specialist debugging pytest")


def _verify_reflection_greenfield_workspace(workspace: Path) -> None:
    module = import_module(workspace, "slugify")
    assert module.slugify("  Hello, World!  ") == "hello-world", "slugify did not normalize punctuation and spacing"
    assert module.slugify("Version 2.0 release") == "version-20-release", "slugify did not preserve digits correctly"
    assert module.slugify("!!!") == "", "slugify should return an empty slug for punctuation-only input"
    cli_result = subprocess.run(
        [sys.executable, "slugify.py", "  Launch Day!  "],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if cli_result.returncode != 0:
        raise AssertionError(
            "slugify.py CLI failed.\n"
            f"stdout:\n{cli_result.stdout}\n"
            f"stderr:\n{cli_result.stderr}"
        )
    assert cli_result.stdout.strip() == "launch-day", "CLI output does not match the slugified value"
    run_pytest(workspace, label="reflection greenfield pytest")
    assert_readme_mentions(workspace, "slugify.py", "pytest")
