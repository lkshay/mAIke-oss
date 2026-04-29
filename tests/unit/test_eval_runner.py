import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import maike.cli
import pytest
from maike.eval.contracts import (
    DynamicEvalMetrics,
    EvalBaseline,
    EvalCaseReport,
    EvalCostSummary,
    EvalMetrics,
    EvalMode,
    EvalReport,
    EvalRequest,
)
from maike.eval.baselines import load_baseline
from maike.eval.runner import EvalRunner
from maike.smoke.workflow_cases.common import DynamicExpectations, WorkflowCase


def test_eval_runner_scores_successful_workflow_and_cleans_up_success_workspace(tmp_path, monkeypatch):
    def setup_workspace(workspace: Path) -> None:
        (workspace / "seed.txt").write_text("seed", encoding="utf-8")

    def verify_workspace(workspace: Path) -> None:
        assert (workspace / "seed.txt").read_text(encoding="utf-8") == "seed"

    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=setup_workspace,
        verify_workspace=verify_workspace,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(
        maike.cli,
        "run_command",
        fake_run_command,
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md", "plan.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_session_cost_summary",
        lambda workspace, session_id: {
            "total_cost_usd": 0.15,
            "total_tokens": 250,
            "input_tokens": 150,
            "output_tokens": 100,
            "llm_calls": 2,
            "agent_runs": 1,
            "has_token_breakdown": True,
            "per_stage": [
                {
                    "stage_name": "analysis",
                    "cost_usd": 0.15,
                    "tokens_used": 250,
                    "input_tokens": 150,
                    "output_tokens": 100,
                    "llm_calls": 2,
                    "agent_runs": 1,
                    "failed_runs": 0,
                    "has_token_breakdown": True,
                }
            ],
        },
    )

    summary = EvalRunner().run_suite("all", tmp_path)

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.failed == 0
    assert summary.average_score == 1.0
    result = summary.results[0]
    assert result.passed is True
    assert result.score == 1.0
    assert result.metrics.workspace_verified is True
    assert result.cost_summary is not None
    assert result.cost_summary.total_tokens == 250
    assert result.cost_summary.per_stage[0].stage_name == "analysis"
    assert Path(result.workspace).exists() is False


def test_eval_runner_records_partial_failures_and_keeps_failed_workspace(tmp_path, monkeypatch):
    def setup_workspace(workspace: Path) -> None:
        (workspace / "seed.txt").write_text("seed", encoding="utf-8")

    def verify_workspace(workspace: Path) -> None:
        raise AssertionError("workspace output was incomplete")

    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=setup_workspace,
        verify_workspace=verify_workspace,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="debugging",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(
        maike.cli,
        "run_command",
        fake_run_command,
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )

    summary = EvalRunner().run_suite("all", tmp_path)

    assert summary.total == 1
    assert summary.passed == 0
    result = summary.results[0]
    assert result.passed is False
    assert 0.0 < result.score < 1.0
    assert result.metrics.session_completed is True
    assert result.metrics.pipeline_match is False
    assert result.metrics.required_artifacts_present is False
    assert result.metrics.workspace_verified is False
    assert "expected pipeline" in (result.error or "")
    assert "workspace verification failed" in (result.error or "")
    assert Path(result.workspace).exists() is True


def test_eval_runner_uses_isolated_git_workspace_when_root_is_inside_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True, capture_output=True)

    runner = EvalRunner()
    workspace = runner._create_case_workspace(tmp_path, "editing")

    try:
        assert workspace.exists()
        assert tmp_path not in workspace.parents
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        assert Path(result.stdout.strip()).resolve() == workspace.resolve()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_eval_runner_compares_dynamic_case_against_persisted_baseline(tmp_path, monkeypatch):
    case = WorkflowCase(
        name="editing-multifile",
        task="Update the task tracker",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=("analysis.md", "plan.md", "code-summary.md", "test-results.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=lambda workspace: None,
        dynamic_agents_enabled=True,
        parallel_coding_enabled=True,
        dynamic_expectations=DynamicExpectations(
            min_partitions=1,
            require_fan_in=True,
            compare_to_deterministic_baseline=True,
        ),
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing-multifile": case})
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)

    async def fake_run_command(**kwargs):
        return SimpleNamespace(
            session_id="dynamic-session",
            pipeline="editing",
            stage_results={"analysis": [], "planning": [], "coding": [], "testing": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning", "coding", "testing"],
            artifact_names=["analysis.md", "plan.md", "code-summary.md", "test-results.md"],
                agent_runs=[
                    {
                        "stage_name": "coding",
                        "agent_id": "partition-1",
                        "role": "coder",
                    "success": True,
                    "output": "partition",
                    "cost_usd": 0.12,
                    "tokens_used": 15,
                    "metadata": {
                        "spawn_reason": "parallel_partition",
                        "input_artifact_ids": [],
                        "produced_artifact_ids": ["partition-artifact-1"],
                        "stage_attempt": 1,
                    },
                    "created_at": "2025-01-01T00:00:00+00:00",
                },
                {
                    "stage_name": "coding",
                    "agent_id": "fan-in-1",
                    "role": "coder",
                    "success": True,
                    "output": "integrated",
                    "cost_usd": 0.2,
                    "tokens_used": 20,
                    "metadata": {
                        "spawn_reason": "delegation",
                        "coordination_mode": "fan_in",
                        "input_artifact_ids": ["partition-artifact-1"],
                        "produced_artifact_ids": ["code-summary-artifact"],
                        "stage_attempt": 1,
                        },
                        "created_at": "2025-01-01T00:00:01+00:00",
                    },
                    {
                        "stage_name": "testing",
                        "agent_id": "tester-1",
                        "role": "tester",
                        "success": True,
                        "output": "Validation Performed\n- Ran pytest\n\nOverall Status: PASS",
                        "cost_usd": 0.0,
                        "tokens_used": 0,
                        "metadata": {
                            "stage_gate": {
                                "stage": "testing",
                                "overall_status": "PASS",
                                "summary": "Validation passed.",
                                "blocking_findings": [],
                                "contract_checks": [
                                    {"id": "pytest", "status": "PASS", "evidence": "Pytest passed."}
                                ],
                                "source": "json",
                                "json_present": True,
                            }
                        },
                        "created_at": "2025-01-01T00:00:02+00:00",
                    },
                ],
                spawn_requests=[],
            ),
        )
    monkeypatch.setattr(
        "maike.eval.runner.load_session_cost_summary",
        lambda workspace, session_id: {
            "total_cost_usd": 0.32,
            "total_tokens": 35,
            "input_tokens": 20,
            "output_tokens": 15,
            "llm_calls": 2,
            "agent_runs": 2,
            "has_token_breakdown": True,
            "per_stage": [],
        },
    )
    baseline_result = EvalCaseReport(
        workflow_name="editing-multifile",
        provider="gemini",
        model="gemini-2.5-flash",
        workspace="/tmp/baseline",
        session_id="baseline-session",
        pipeline="editing",
        passed=True,
        execution_passed=True,
        comparison_passed=True,
        score=1.0,
            metrics=EvalMetrics(
                session_completed=True,
                session_status_match=True,
                pipeline_match=True,
                stage_sequence_match=True,
                persisted_stage_sequence_match=True,
                required_artifacts_present=True,
                required_stage_gates_present=True,
                workspace_verified=True,
                dynamic=DynamicEvalMetrics(
                    partition_count=1,
                    completed_partition_count=1,
                    fan_in_completed=True,
                total_cost_usd=0.3,
                total_tokens=30,
            ),
        ),
        cost_summary=EvalCostSummary(total_cost_usd=0.3, total_tokens=30),
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_baseline",
        lambda *args, **kwargs: EvalBaseline(
            schema_version=1,
            captured_at="2025-01-01T00:00:00+00:00",
            suite="dynamic",
            suite_key="dynamic",
            provider="gemini",
            model="gemini-2.5-flash",
            baseline_path=str(tmp_path / ".maike-eval" / "baselines" / "dynamic" / "gemini" / "gemini-2.5-flash.json"),
            policy={},
            report=EvalReport(
                schema_version=1,
                mode="baseline",
                suite="dynamic",
                suite_key="dynamic",
                provider="gemini",
                model="gemini-2.5-flash",
                budget=5.0,
                created_at="2025-01-01T00:00:00+00:00",
                git_sha=None,
                report_path=None,
                baseline_path=None,
                total=1,
                passed=1,
                failed=0,
                average_score=1.0,
                execution_failed=False,
                regression_failed=False,
                warning_count=0,
                results=[baseline_result],
            ),
        ),
    )

    summary = EvalRunner().run(
        EvalRequest(
            suite="dynamic",
            workspace_root=tmp_path,
            mode=EvalMode.COMPARE,
        )
    )

    assert summary.total == 1
    result = summary.results[0]
    assert result.metrics.dynamic.partition_count == 1
    assert result.metrics.dynamic.fan_in_completed is True
    assert result.metrics.dynamic.baseline_cost_usd == 0.3
    assert result.metrics.dynamic.cost_delta_usd == pytest.approx(0.02)
    assert result.comparison_passed is True
    assert result.passed is True


def test_eval_runner_compare_marks_cost_regression_failure(tmp_path, monkeypatch):
    def setup_workspace(workspace: Path) -> None:
        (workspace / "seed.txt").write_text("seed", encoding="utf-8")

    def verify_workspace(workspace: Path) -> None:
        assert (workspace / "seed.txt").exists()

    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=setup_workspace,
        verify_workspace=verify_workspace,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_cost_summary",
        lambda workspace, session_id: {
            "total_cost_usd": 0.7,
            "total_tokens": 250,
            "input_tokens": 150,
            "output_tokens": 100,
            "llm_calls": 2,
            "agent_runs": 1,
            "has_token_breakdown": True,
            "per_stage": [],
        },
    )

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md", "plan.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_baseline",
        lambda *args, **kwargs: EvalBaseline(
            schema_version=1,
            captured_at="2025-01-01T00:00:00+00:00",
            suite="smoke",
            suite_key="smoke",
            provider="gemini",
            model="gemini-2.5-flash",
            baseline_path=str(tmp_path / ".maike-eval" / "baselines" / "smoke" / "gemini" / "gemini-2.5-flash.json"),
            policy={},
            report=EvalReport(
                schema_version=1,
                mode="baseline",
                suite="smoke",
                suite_key="smoke",
                provider="gemini",
                model="gemini-2.5-flash",
                budget=5.0,
                created_at="2025-01-01T00:00:00+00:00",
                git_sha=None,
                report_path=None,
                baseline_path=None,
                total=1,
                passed=1,
                failed=0,
                average_score=1.0,
                execution_failed=False,
                regression_failed=False,
                warning_count=0,
                results=[
                    EvalCaseReport(
                        workflow_name="editing",
                        provider="gemini",
                        model="gemini-2.5-flash",
                        workspace="/tmp/baseline",
                        session_id="baseline-session",
                        pipeline="editing",
                        passed=True,
                        execution_passed=True,
                        comparison_passed=True,
                        score=1.0,
                        metrics=EvalMetrics(
                            session_completed=True,
                            session_status_match=True,
                            pipeline_match=True,
                            stage_sequence_match=True,
                            persisted_stage_sequence_match=True,
                            required_artifacts_present=True,
                            workspace_verified=True,
                        ),
                        cost_summary=EvalCostSummary(total_cost_usd=0.3, total_tokens=120),
                    )
                ],
            ),
        ),
    )

    report = EvalRunner().run(
        EvalRequest(
            suite="smoke",
            workspace_root=tmp_path,
            mode=EvalMode.COMPARE,
        )
    )

    assert report.regression_failed is True
    result = report.results[0]
    assert result.passed is False
    assert result.comparison_passed is False
    assert any(item.metric == "total_cost_usd" for item in result.comparison.regressions)


def test_eval_runner_compare_missing_baseline_marks_hard_failure(tmp_path, monkeypatch):
    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=lambda workspace: None,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr(
        "maike.eval.runner.load_baseline",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing baseline")),
    )

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md", "plan.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )

    report = EvalRunner().run(
        EvalRequest(
            suite="smoke",
            workspace_root=tmp_path,
            mode=EvalMode.COMPARE,
        )
    )

    assert report.regression_failed is True
    result = report.results[0]
    assert result.comparison_passed is False
    assert result.passed is False
    assert result.comparison is not None
    assert result.comparison.regressions[0].metric == "baseline"


def test_eval_runner_baseline_mode_writes_baseline_and_custom_report_path(tmp_path, monkeypatch):
    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=lambda workspace: None,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_cost_summary",
        lambda workspace, session_id: {
            "total_cost_usd": 0.15,
            "total_tokens": 250,
            "input_tokens": 150,
            "output_tokens": 100,
            "llm_calls": 2,
            "agent_runs": 1,
            "has_token_breakdown": True,
            "per_stage": [],
        },
    )

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md", "plan.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )

    output_path = tmp_path / "custom-report.json"
    report = EvalRunner().run(
        EvalRequest(
            suite="smoke",
            workspace_root=tmp_path,
            mode=EvalMode.BASELINE,
            output_path=output_path,
        )
    )

    assert output_path.exists()
    assert report.report_path == str(output_path)
    assert report.baseline_path is not None
    assert Path(report.baseline_path).exists()
    baseline = load_baseline(tmp_path, suite_key="smoke", provider="gemini", model="gemini-2.5-flash")
    assert baseline.report is not None
    assert baseline.report.report_path == str(output_path)


def test_eval_runner_aggregates_multi_case_suite_summary(tmp_path, monkeypatch):
    def setup_workspace(workspace: Path) -> None:
        (workspace / "seed.txt").write_text("seed", encoding="utf-8")

    cases = {
        "editing": WorkflowCase(
            name="editing",
            task="Update the calculator",
            expected_pipeline="editing",
            expected_stages=("analysis", "planning"),
            expected_stage_artifacts=("analysis.md", "plan.md"),
            setup_workspace=setup_workspace,
            verify_workspace=lambda workspace: None,
        ),
        "debugging": WorkflowCase(
            name="debugging",
            task="Fix the calculator bug",
            expected_pipeline="debugging",
            expected_stages=("analysis", "planning"),
            expected_stage_artifacts=("analysis.md", "plan.md"),
            setup_workspace=setup_workspace,
            verify_workspace=lambda workspace: (_ for _ in ()).throw(AssertionError("broken workspace")),
        ),
    }

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: cases)
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)

    async def fake_run_command(**kwargs):
        if "Fix the calculator bug" in kwargs["task"]:
            return SimpleNamespace(
                session_id="debug-session",
                pipeline="editing",
                stage_results={"analysis": [], "planning": []},
            )
        return SimpleNamespace(
            session_id="edit-session",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md", "plan.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )

    report = EvalRunner().run_suite("all", tmp_path)

    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.execution_failed is True
    assert [item.workflow_name for item in report.results] == ["editing", "debugging"]


def test_eval_runner_all_suite_covers_every_declared_workflow(monkeypatch):
    cases = {
        "greenfield": WorkflowCase(
            name="greenfield",
            task="task",
            expected_pipeline="greenfield",
            expected_stages=("requirements",),
            expected_stage_artifacts=("spec.md",),
            setup_workspace=lambda workspace: None,
            verify_workspace=lambda workspace: None,
        ),
        "multifile": WorkflowCase(
            name="multifile",
            task="task",
            expected_pipeline="greenfield",
            expected_stages=("requirements",),
            expected_stage_artifacts=("spec.md",),
            setup_workspace=lambda workspace: None,
            verify_workspace=lambda workspace: None,
        ),
        "editing": WorkflowCase(
            name="editing",
            task="task",
            expected_pipeline="editing",
            expected_stages=("analysis",),
            expected_stage_artifacts=("analysis.md",),
            setup_workspace=lambda workspace: None,
            verify_workspace=lambda workspace: None,
        ),
        "editing-multifile": WorkflowCase(
            name="editing-multifile",
            task="task",
            expected_pipeline="editing",
            expected_stages=("analysis",),
            expected_stage_artifacts=("analysis.md",),
            setup_workspace=lambda workspace: None,
            verify_workspace=lambda workspace: None,
        ),
    }
    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: cases)

    assert EvalRunner()._workflow_names_for_suite("all") == list(cases)


def test_eval_runner_uses_latest_historical_stage_artifacts_for_required_outputs(
    tmp_path,
    monkeypatch,
):
    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=lambda workspace: None,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr("maike.eval.runner.load_session_cost_summary", lambda workspace, session_id: None)

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md"],
            latest_artifact_names=["analysis.md", "plan.md"],
            active_artifact_names=["analysis.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )

    report = EvalRunner().run_suite("all", tmp_path)

    assert report.total == 1
    result = report.results[0]
    assert result.passed is True
    assert result.metrics.required_artifacts_present is True


def test_eval_runner_grades_historical_gated_run_as_pass_even_when_latest_artifacts_are_missing(
    tmp_path,
    monkeypatch,
):
    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=("analysis.md", "plan.md", "code-summary.md", "test-results.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=lambda workspace: None,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr("maike.eval.runner.load_session_cost_summary", lambda workspace, session_id: None)

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": [], "coding": [], "testing": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning", "coding", "testing"],
            artifact_names=["analysis.md", "plan.md"],
            latest_artifact_names=["analysis.md", "plan.md"],
            active_artifact_names=["analysis.md", "plan.md"],
            agent_runs=[
                {
                    "stage_name": "testing",
                    "success": True,
                    "output": "Validation Performed\n- Ran pytest\n\nOverall Status: PASS",
                    "cost_usd": 0.0,
                    "tokens_used": 0,
                    "metadata": {},
                }
            ],
            spawn_requests=[],
        ),
    )

    report = EvalRunner().run_suite("all", tmp_path)

    assert report.total == 1
    result = report.results[0]
    assert result.passed is True
    assert result.metrics.required_artifacts_present is False
    assert result.metrics.required_stage_gates_present is True


def test_eval_runner_ignores_stage_gate_metadata_after_refactor(
    tmp_path,
    monkeypatch,
):
    """Stage gates have been removed; legacy stage-gate metadata in agent_runs is ignored."""
    case = WorkflowCase(
        name="editing",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=("analysis.md", "plan.md", "code-summary.md", "test-results.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=lambda workspace: None,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr("maike.eval.runner.load_session_cost_summary", lambda workspace, session_id: None)

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": [], "coding": [], "testing": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning", "coding", "testing"],
            artifact_names=["analysis.md", "plan.md", "code-summary.md", "test-results.md"],
            latest_artifact_names=["analysis.md", "plan.md", "code-summary.md", "test-results.md"],
            active_artifact_names=["analysis.md", "plan.md", "code-summary.md", "test-results.md"],
            agent_runs=[
                {
                    "stage_name": "testing",
                    "success": True,
                    "output": "Validation Performed\n- Ran pytest\n\nOverall Status: PASS",
                    "cost_usd": 0.0,
                    "tokens_used": 0,
                    "metadata": {
                        "stage_gate": {
                            "stage": "testing",
                            "overall_status": "FAIL",
                            "summary": "Regression remains.",
                            "blocking_findings": [
                                {
                                    "severity": "HIGH",
                                    "target": "workspace",
                                    "summary": "Regression remains.",
                                    "repair_instruction": "Fix the regression.",
                                }
                            ],
                            "contract_checks": [
                                {
                                    "id": "pytest",
                                    "status": "FAIL",
                                    "evidence": "The regression path still fails.",
                                }
                            ],
                            "source": "json",
                            "json_present": True,
                        }
                    },
                }
            ],
            spawn_requests=[],
        ),
    )

    report = EvalRunner().run_suite("all", tmp_path)

    assert report.total == 1
    result = report.results[0]
    # Stage gates removed: legacy metadata is ignored, case passes.
    assert result.passed is True
    assert result.metrics.required_stage_gates_present is True


def test_eval_runner_supports_tier_suite_selection(monkeypatch):
    cases = {
        "tier-one": WorkflowCase(
            name="tier-one",
            task="task",
            expected_pipeline="editing",
            expected_stages=("analysis",),
            expected_stage_artifacts=("analysis.md",),
            setup_workspace=lambda workspace: None,
            verify_workspace=lambda workspace: None,
            tags=("tier1",),
        ),
        "tier-two": WorkflowCase(
            name="tier-two",
            task="task",
            expected_pipeline="editing",
            expected_stages=("analysis",),
            expected_stage_artifacts=("analysis.md",),
            setup_workspace=lambda workspace: None,
            verify_workspace=lambda workspace: None,
            tags=("tier2",),
        ),
    }
    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: cases)

    assert EvalRunner()._workflow_names_for_suite("tier1") == ["tier-one"]
    assert EvalRunner()._workflow_names_for_suite("tier2") == ["tier-two"]


def test_eval_runner_accepts_expected_failed_status_and_calls_verify_run_after_workspace(
    tmp_path,
    monkeypatch,
):
    events: list[str] = []

    def verify_workspace(workspace: Path) -> None:
        events.append("workspace")
        assert workspace.exists()

    def verify_run(context) -> None:
        assert events == ["workspace"]
        events.append("run")
        assert len(context.runs) == 1
        assert context.latest_run.run_result.session_id == "session-1"

    case = WorkflowCase(
        name="budget-exhaustion-graceful",
        task="Produce a large design document",
        expected_pipeline="greenfield",
        expected_stages=("requirements", "planning", "architecture", "coding"),
        expected_stage_artifacts=(),
        setup_workspace=lambda workspace: None,
        verify_workspace=verify_workspace,
        verify_run=verify_run,
        expected_session_statuses=("failed",),
        observed_stage_sequence_mode="prefix",
        persisted_stage_sequence_mode="prefix",
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {case.name: case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: [case.name])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr("maike.eval.runner.load_session_cost_summary", lambda workspace, session_id: None)

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="greenfield",
            stage_results={"requirements": [], "planning": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="failed",
            stage_names=["requirements", "planning"],
            artifact_names=[],
            latest_artifact_names=[],
            active_artifact_names=[],
            agent_runs=[],
            spawn_requests=[],
        ),
    )

    report = EvalRunner().run_suite("all", tmp_path)

    assert report.total == 1
    result = report.results[0]
    assert result.passed is True
    assert result.metrics.session_completed is False
    assert result.metrics.session_status_match is True
    assert result.metrics.stage_sequence_match is True
    assert result.metrics.persisted_stage_sequence_match is True
    assert events == ["workspace", "run"]


def test_eval_runner_repeat_runs_reuse_workspace_and_enforce_spawn_depth_limit(
    tmp_path,
    monkeypatch,
):
    setup_calls = 0

    def setup_workspace(workspace: Path) -> None:
        nonlocal setup_calls
        setup_calls += 1
        (workspace / "seed.txt").write_text("seed", encoding="utf-8")

    verify_calls: list[int] = []

    def verify_workspace(workspace: Path) -> None:
        verify_calls.append(1)
        assert (workspace / "seed.txt").read_text(encoding="utf-8") == "seed"

    case = WorkflowCase(
        name="repeat-dynamic",
        task="Update the seeded workspace",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=setup_workspace,
        verify_workspace=verify_workspace,
        repeat_runs=2,
        reuse_workspace_between_runs=True,
        dynamic_expectations=DynamicExpectations(min_retries=1, max_spawn_depth=2),
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {case.name: case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: [case.name])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr("maike.eval.runner.load_session_cost_summary", lambda workspace, session_id: None)

    async def fake_run_command(**kwargs):
        workspace = kwargs["workspace"]
        assert (workspace / "seed.txt").exists()
        run_number = 1 if "session-1" not in getattr(fake_run_command, "sessions", ()) else 2
        sessions = list(getattr(fake_run_command, "sessions", ()))
        sessions.append(f"session-{run_number}")
        fake_run_command.sessions = tuple(sessions)
        return SimpleNamespace(
            session_id=f"session-{run_number}",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)

    def fake_snapshot(workspace: Path, session_id: str):
        del workspace
        spawn_depth = 1 if session_id == "session-1" else 3
        stage_attempt = 1 if session_id == "session-1" else 2
        return SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            artifact_names=["analysis.md", "plan.md"],
            latest_artifact_names=["analysis.md", "plan.md"],
            active_artifact_names=["analysis.md", "plan.md"],
            agent_runs=[
                {
                    "stage_name": "analysis",
                    "success": True,
                    "cost_usd": 0.0,
                    "tokens_used": 0,
                    "metadata": {
                        "spawn_reason": "specialist_needed",
                        "produced_artifact_ids": ["analysis-artifact"],
                        "input_artifact_ids": [],
                        "stage_attempt": stage_attempt,
                        "spawn_depth": spawn_depth,
                    },
                }
            ],
            spawn_requests=[],
        )

    monkeypatch.setattr("maike.eval.runner.load_session_snapshot", fake_snapshot)

    report = EvalRunner().run_suite("all", tmp_path)

    assert report.total == 1
    result = report.results[0]
    assert result.passed is False
    assert result.metrics.dynamic.retry_count == 1
    assert result.metrics.dynamic.max_observed_spawn_depth == 3
    assert "expected spawn depth <= 2" in (result.error or "")
    assert setup_calls == 1
    assert len(verify_calls) == 1
