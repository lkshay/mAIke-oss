"""Tier 3 production workflow cases."""

from __future__ import annotations

import ast
from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.helpers import assert_readme_mentions, run_checked, run_pytest, write_bytes, write_text


def _seed_binary_workspace(workspace: Path) -> None:
    write_bytes(workspace / "image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    write_bytes(workspace / "data.bin", b"\xff\x00\x10binary")
    write_text(workspace / ".gitignore", "__pycache__/\n")


def _verify_binary_workspace(workspace: Path) -> None:
    assert (workspace / "README.md").exists()
    run_checked(["python3", "summarize_workspace.py"], workspace, label="binary workspace script")


def _seed_nested_workspace(workspace: Path) -> None:
    files = {
        "src/core/math.py": ["add", "subtract"],
        "src/core/strings.py": ["slugify", "titleize"],
        "src/api/handlers.py": ["handle_ping", "handle_status"],
        "src/api/auth.py": ["login", "logout"],
        "src/utils/files.py": ["read_text", "write_text"],
        "src/utils/ids.py": ["parse_id", "format_id"],
        "src/models/user.py": ["build_user", "serialize_user"],
        "src/models/order.py": ["build_order", "serialize_order"],
        "tests/test_math.py": [],
        "tests/test_strings.py": [],
        "tests/test_handlers.py": [],
        "tests/test_auth.py": [],
        "tests/test_files.py": [],
        "tests/test_ids.py": [],
        "tests/test_user.py": [],
        "tests/test_order.py": [],
        "scripts/sync.py": ["run"],
        "scripts/export.py": ["run"],
        "tools/checks.py": ["main"],
        "tools/report.py": ["main"],
    }
    for relative_path, functions in files.items():
        path = workspace / relative_path
        if relative_path.startswith("tests/"):
            write_text(
                path,
                "def test_smoke():\n    assert True\n",
            )
            continue
        body = []
        for function_name in functions:
            body.extend(
                [
                    f"def {function_name}(value, other=1):",
                    "    return value if other == 1 else other",
                    "",
                ]
            )
        write_text(path, "\n".join(body) or "VALUE = 1\n")


def _verify_nested_workspace(workspace: Path) -> None:
    for path in workspace.rglob("*.py"):
        if ".maike" in path.parts or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            assert node.returns is not None, f"{path.name}:{node.name} is missing a return annotation"
            for arg in node.args.args:
                if arg.arg in {"self", "cls"}:
                    continue
                assert arg.annotation is not None, f"{path.name}:{node.name} missing annotation for {arg.arg}"
    run_pytest(workspace, label="nested workspace pytest")


def _verify_conflicting_workspace(workspace: Path) -> None:
    assert (workspace / "sorting.py").exists()
    run_pytest(workspace, label="conflicting requirements pytest")
    assert_readme_mentions(workspace, "sorting.py")


def _seed_node_workspace(workspace: Path) -> None:
    write_text(
        workspace / "package.json",
        """{
  "name": "node-debug-case",
  "version": "1.0.0",
  "scripts": {
    "test": "node test.js"
  }
}
""",
    )
    write_text(workspace / "index.js", "function multiply(a, b) { return a + b; }\nmodule.exports = { multiply };\n")
    write_text(
        workspace / "test.js",
        """const assert = require("assert");
const { multiply } = require("./index");

assert.strictEqual(multiply(3, 4), 12);
console.log("ok");
""",
    )


def _verify_node_workspace(workspace: Path) -> None:
    run_checked(["npm", "test"], workspace, label="node workspace npm test")


TIER3_EVAL_CASES: dict[str, EvalCase] = {
    "workspace-with-binary-files": EvalCase(
        name="workspace-with-binary-files",
        phases=(
            EvalPhase(
                task="Add README.md and summarize_workspace.py to this workspace. summarize_workspace.py should print a short summary when run.",
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_binary_workspace,
        verify_workspace=_verify_binary_workspace,
        tags=("tier3",),
    ),
    "deeply-nested-workspace": EvalCase(
        name="deeply-nested-workspace",
        phases=(
            EvalPhase(
                task="Add type hints (parameter and return annotations) to every function in every .py file across this project — including src/, scripts/, and tools/ directories. Keep the existing tests passing.",
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_nested_workspace,
        verify_workspace=_verify_nested_workspace,
        tags=("tier3",),
    ),
    "conflicting-requirements": EvalCase(
        name="conflicting-requirements",
        phases=(
            EvalPhase(
                task=(
                    "Create sorting.py, test_sorting.py, and README.md for a comparison-based sorting function. "
                    "The request claims the algorithm must be O(n) for all inputs. "
                    "Implement the best valid solution and clearly document the impossible constraint."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_conflicting_workspace,
        tags=("tier3",),
    ),
    "cross-language-workspace": EvalCase(
        name="cross-language-workspace",
        phases=(
            EvalPhase(
                task="Fix the failing test in this Node.js project.",
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_node_workspace,
        verify_workspace=_verify_node_workspace,
        language="node",
        tags=("tier3",),
    ),
}
