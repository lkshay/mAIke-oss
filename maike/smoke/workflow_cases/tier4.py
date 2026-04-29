"""Tier 4 production workflow cases."""

from __future__ import annotations

from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.helpers import import_module, run_pytest, write_text


def _seed_partition_conflict_workspace(workspace: Path) -> None:
    write_text(workspace / "shared.py", "def make_task(title: str) -> dict[str, object]:\n    return {'title': title}\n")
    write_text(
        workspace / "frontend.py",
        """from shared import make_task


def render_task(title: str) -> str:
    task = make_task(title)
    return task["title"]
""",
    )
    write_text(
        workspace / "backend.py",
        """from shared import make_task


def build_task(title: str) -> dict[str, object]:
    return make_task(title)
""",
    )
    write_text(
        workspace / "test_frontend.py",
        """from frontend import render_task


def test_render_task():
    assert render_task("demo") == "demo"
""",
    )
    write_text(
        workspace / "test_backend.py",
        """from backend import build_task


def test_build_task():
    assert build_task("demo")["title"] == "demo"
""",
    )


def _verify_partition_conflict_workspace(workspace: Path) -> None:
    backend = import_module(workspace, "backend")
    frontend = import_module(workspace, "frontend")

    built = backend.build_task("demo", priority="high")
    assert built["priority"] == "high"
    assert frontend.render_task("demo", priority="high").endswith("[high]")
    run_pytest(workspace, label="partition conflict pytest")


def _seed_specialist_chain_workspace(workspace: Path) -> None:
    write_text(
        workspace / "data_pipeline.py",
        """def normalize_record(record: dict[str, str]) -> dict[str, str]:
    return {
        "name": record["name"].strip(),
        "status": record.get("status") or "unknown",
    }
""",
    )
    write_text(
        workspace / "test_data_pipeline.py",
        """from data_pipeline import normalize_record


def test_status_zero_string_is_preserved():
    assert normalize_record({"name": " Jane ", "status": "0"})["status"] == "0"
""",
    )


def _verify_specialist_chain_workspace(workspace: Path) -> None:
    module = import_module(workspace, "data_pipeline")

    assert module.normalize_record({"name": " Jane ", "status": "0"})["status"] == "0"
    run_pytest(workspace, label="specialist chain pytest")


def _seed_reflection_workspace(workspace: Path) -> None:
    del workspace


def _verify_reflection_workspace(workspace: Path) -> None:
    assert (workspace / "validators.py").exists(), "validators.py was not created"
    assert (workspace / "test_validators.py").exists(), "test_validators.py was not created"
    assert (workspace / "README.md").exists(), "README.md was not created"
    import_module(workspace, "validators")
    run_pytest(workspace, label="reflection acceptance pytest")


TIER4_EVAL_CASES: dict[str, EvalCase] = {
    "parallel-partition-file-conflict": EvalCase(
        name="parallel-partition-file-conflict",
        phases=(
            EvalPhase(
                task=(
                    "Update both frontend.py and backend.py to support a new priority field for tasks. "
                    "shared.py may be updated if needed, and the tests must keep passing."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_partition_conflict_workspace,
        verify_workspace=_verify_partition_conflict_workspace,
        tags=("tier4",),
    ),
    "specialist-chain-depth-limit": EvalCase(
        name="specialist-chain-depth-limit",
        phases=(
            EvalPhase(
                task=(
                    "Fix the bug in this data pipeline where the string status '0' is treated as missing. "
                    "Keep the fix narrowly scoped."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_specialist_chain_workspace,
        verify_workspace=_verify_specialist_chain_workspace,
        tags=("tier4",),
    ),
    "reflection-disagrees-with-acceptance": EvalCase(
        name="reflection-disagrees-with-acceptance",
        phases=(
            EvalPhase(
                task=(
                    "Create validators.py, test_validators.py, and README.md for a small data validation library with strict input checking. "
                    "Document the API and make the pytest suite pass."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_reflection_workspace,
        verify_workspace=_verify_reflection_workspace,
        tags=("tier4",),
    ),
}
