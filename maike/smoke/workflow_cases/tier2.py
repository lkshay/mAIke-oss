"""Tier 2 production workflow cases."""

from __future__ import annotations

import subprocess
from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.helpers import assert_readme_mentions, import_module, run_pytest, write_text


def _verify_fibonacci_workspace(workspace: Path) -> None:
    module = import_module(workspace, "fib")
    assert module.fib_recursive(7) == 13
    assert module.fib_iterative(7) == 13
    run_pytest(workspace, label="fibonacci pytest")


def _verify_shortener_workspace(workspace: Path) -> None:
    module = import_module(workspace, "shortener")
    encoded = module.encode("https://example.com/demo")
    decoded = module.decode(encoded)
    assert decoded == "https://example.com/demo"
    assert module.encode(decoded) == encoded
    run_pytest(workspace, label="shortener pytest")


def _verify_budget_workspace(workspace: Path) -> None:
    assert workspace.exists(), "Workspace does not exist"


def _seed_monolith_workspace(workspace: Path) -> None:
    write_text(
        workspace / "monolith.py",
        "\n".join(
            [
                "def build_user(name: str) -> dict[str, str]:",
                "    return {'name': name}",
                "",
                "def build_order(order_id: int) -> dict[str, int]:",
                "    return {'id': order_id}",
                "",
                "def render_cli(name: str, order_id: int) -> str:",
                "    user = build_user(name)",
                "    order = build_order(order_id)",
                "    return f\"{user['name']}:{order['id']}\"",
                "",
                "def main() -> None:",
                "    print(render_cli('jane', 9))",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
            + [f"HELPER_{index} = {index}" for index in range(80)]
        ),
    )
    write_text(
        workspace / "test_monolith.py",
        """from monolith import render_cli


def test_render_cli():
    assert render_cli("jane", 9) == "jane:9"
""",
    )


def _verify_checkpoint_workspace(workspace: Path) -> None:
    run_pytest(workspace, label="checkpoint refactor pytest")
    assert (workspace / "models.py").exists()
    assert (workspace / "cli.py").exists()
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "maike-checkpoint:" in result.stdout, "No checkpoint commit was recorded"


def _seed_idempotent_workspace(workspace: Path) -> None:
    write_text(
        workspace / "calculator.py",
        """def add(a: int, b: int) -> int:
    return a + b
""",
    )
    write_text(workspace / "README.md", "# Calculator\n\nSupports add only.\n")


def _verify_idempotent_workspace(workspace: Path) -> None:
    module = import_module(workspace, "calculator")
    assert module.multiply(3, 4) == 12
    assert_readme_mentions(workspace, "multiply")


TIER2_EVAL_CASES: dict[str, EvalCase] = {
    "forced-test-failure-retry": EvalCase(
        name="forced-test-failure-retry",
        phases=(
            EvalPhase(
                task=(
                    "Create fib.py, test_fib.py, and README.md. "
                    "fib.py must expose fib_recursive(n) and fib_iterative(n). "
                    "The tests must verify both implementations, and the workspace must pass pytest."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_fibonacci_workspace,
        agent_token_budget=15_000,
        tags=("tier2",),
    ),
    "acceptance-gate-repair": EvalCase(
        name="acceptance-gate-repair",
        phases=(
            EvalPhase(
                task=(
                    "Create shortener.py, test_shortener.py, and README.md. "
                    "Expose encode(url) and decode(short_code) with stable round-trip behavior, "
                    "document the contract, and make the pytest suite pass."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_shortener_workspace,
        tags=("tier2",),
    ),
    "budget-exhaustion-graceful": EvalCase(
        name="budget-exhaustion-graceful",
        phases=(
            EvalPhase(
                task="Create a comprehensive REST API design document with 10 endpoints, request and response payloads, edge cases, and testing notes.",
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_budget_workspace,
        budget=0.30,
        expected_session_statuses=("failed",),
        tags=("tier2",),
    ),
    "checkpoint-restore-after-destructive-op": EvalCase(
        name="checkpoint-restore-after-destructive-op",
        phases=(
            EvalPhase(
                task=(
                    "Refactor the existing monolith.py module into models.py and cli.py while keeping the public render_cli behavior the same. "
                    "Keep the tests passing and document the new structure in README.md."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_monolith_workspace,
        verify_workspace=_verify_checkpoint_workspace,
        tags=("tier2",),
    ),
    "idempotent-rerun-same-workspace": EvalCase(
        name="idempotent-rerun-same-workspace",
        phases=(
            EvalPhase(
                task="Update the existing calculator module to add multiply(a, b) and refresh the README with the new behavior.",
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_idempotent_workspace,
        verify_workspace=_verify_idempotent_workspace,
        repeat_runs=2,
        reuse_workspace_between_runs=True,
        tags=("tier2",),
    ),
}
