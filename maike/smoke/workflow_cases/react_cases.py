"""React-mode eval cases.

These cases reuse the same tasks and verifiers from the deterministic
pipeline cases and test that the react loop produces functionally
correct outcomes.

Principle: grade outcomes, not paths.
"""

from __future__ import annotations

from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.base import (
    _seed_debugging_workspace,
    _seed_editing_multifile_workspace,
    _seed_editing_workspace,
    _seed_specialist_debugging_workspace,
    _verify_debugging_workspace,
    _verify_editing_multifile_workspace,
    _verify_editing_workspace,
    _verify_greenfield_workspace,
    _verify_multifile_workspace,
    _verify_reflection_greenfield_workspace,
    _verify_specialist_debugging_workspace,
)


# ---------------------------------------------------------------------------
# Verifiers specific to react mode
# ---------------------------------------------------------------------------

def _verify_react_question_workspace(workspace: Path) -> None:
    """Verify that a question/explanation task did NOT create or modify code.

    This checks that `*.py` files in the workspace are unchanged.
    We only verify that no *new* Python files appeared.  The react
    question case runs on the debugging workspace (seeded with
    calculator.py + test_calculator.py), so those two files are expected.
    """
    py_files = {
        p.relative_to(workspace)
        for p in workspace.rglob("*.py")
        if ".maike" not in p.parts and ".venv" not in p.parts
    }
    expected = {Path("calculator.py"), Path("test_calculator.py")}
    unexpected = py_files - expected
    assert not unexpected, (
        f"Read-only phase created unexpected Python files: {unexpected}"
    )


# ---------------------------------------------------------------------------
# React EvalCases
# ---------------------------------------------------------------------------

REACT_EVAL_CASES: dict[str, EvalCase] = {
    "react-greenfield": EvalCase(
        name="react-greenfield",
        phases=(
            EvalPhase(
                task=(
                    "Create a README and a single-file Python CLI app that "
                    "prints Hello World. Keep it dependency-free and document "
                    "how to run it."
                ),
            ),
        ),
        # React mode — no pipeline/stage/artifact expectations
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        verify_workspace=_verify_greenfield_workspace,
        tags=("react", "core"),
        budget=5.0,
    ),
    "react-editing": EvalCase(
        name="react-editing",
        phases=(
            EvalPhase(
                task=(
                    "Update the existing calculator app to support subtraction, "
                    "keep it as a simple one-file Python module, and refresh the "
                    "README with the new behavior."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_editing_workspace,
        verify_workspace=_verify_editing_workspace,
        tags=("react", "core"),
        budget=5.0,
    ),
    "react-debugging": EvalCase(
        name="react-debugging",
        phases=(
            EvalPhase(
                task=(
                    "Fix the bug in the existing calculator module where "
                    "divide(0, n) returns the wrong value, and make the tests pass."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_debugging_workspace,
        verify_workspace=_verify_debugging_workspace,
        tags=("react", "core"),
        budget=5.0,
    ),
    "react-question": EvalCase(
        name="react-question",
        phases=(
            EvalPhase(
                task=(
                    "Can you explain how the divide function works in "
                    "calculator.py? What edge cases does it handle?"
                ),
                expect_read_only=True,
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_debugging_workspace,
        verify_workspace=_verify_react_question_workspace,
        tags=("react", "read-only"),
        budget=3.0,
    ),
    "react-multifile": EvalCase(
        name="react-multifile",
        phases=(
            EvalPhase(
                task=(
                    "Create a dependency-free Python task tracker with these exact files: "
                    "task_store.py, task_cli.py, test_task_store.py, and README.md. "
                    "task_store.py must load and save tasks in tasks.json. "
                    "task_cli.py must support 'add <title>' and 'list'. "
                    "test_task_store.py must validate the core storage behavior with pytest. "
                    "README.md must explain how to run the CLI and the tests."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        verify_workspace=_verify_multifile_workspace,
        tags=("react",),
        budget=5.0,
    ),
    "react-editing-multifile": EvalCase(
        name="react-editing-multifile",
        phases=(
            EvalPhase(
                task=(
                    "Update the existing dependency-free task tracker to support marking tasks complete. "
                    "Add a `done <index>` CLI command, persist completion state in the store, refresh the README, "
                    "and extend the pytest coverage. Keep the file structure as task_store.py, task_cli.py, "
                    "test_task_store.py, and README.md."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_editing_multifile_workspace,
        verify_workspace=_verify_editing_multifile_workspace,
        tags=("react",),
        budget=5.0,
    ),
    "react-debugging-specialist": EvalCase(
        name="react-debugging-specialist",
        phases=(
            EvalPhase(
                task=(
                    "Fix the bug in the existing config loader where APP_PORT='0' is treated as missing. "
                    "Keep the fix narrowly scoped."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_specialist_debugging_workspace,
        verify_workspace=_verify_specialist_debugging_workspace,
        tags=("react",),
        budget=5.0,
    ),
    "react-greenfield-reflection": EvalCase(
        name="react-greenfield-reflection",
        phases=(
            EvalPhase(
                task=(
                    "Create a dependency-free Python slugify module in slugify.py with a tiny CLI in the same file. "
                    "`slugify(text)` must lowercase input, trim surrounding whitespace, collapse internal whitespace "
                    "to single dashes, strip punctuation, preserve digits, and never leave leading or trailing dashes. "
                    "Add pytest coverage and document how to run the CLI and tests."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        verify_workspace=_verify_reflection_greenfield_workspace,
        tags=("react",),
        budget=5.0,
    ),
    "react-ssg-workflow": EvalCase(
        name="react-ssg-workflow",
        phases=(
            EvalPhase(
                task=(
                    "Build a Python static site generator CLI called 'forge'. "
                    "1) Read markdown files from content/ directory. "
                    "2) Parse YAML frontmatter (title, date, tags, template). "
                    "3) Convert markdown to HTML. "
                    "4) Render through Jinja2 templates from templates/ directory. "
                    "5) Output to build/ preserving structure. "
                    "6) CLI: 'forge build' and 'forge clean'. "
                    "Use stdlib + markdown + jinja2 + pyyaml. "
                    "Include pyproject.toml with [tool.setuptools] packages=['forge']. "
                    "Write tests and verify end-to-end."
                ),
                budget=10.0,
            ),
            EvalPhase(
                task=(
                    "Can you explain the code we have generated so far? "
                    "Walk me through the architecture and how the build process works."
                ),
                budget=5.0,
                expect_read_only=True,
            ),
            EvalPhase(
                task=(
                    "Add comprehensive unit tests for the forge project. "
                    "Test the frontmatter parser edge cases, markdown-to-HTML conversion, "
                    "template rendering, directory structure preservation in build output, "
                    "and the CLI commands. Make sure all tests pass."
                ),
                budget=10.0,
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        verify_workspace=None,  # Multi-phase — each phase self-verifies
        tags=("react", "multi-phase", "ssg"),
        budget=10.0,
    ),
}
