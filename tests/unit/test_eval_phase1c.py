"""Tests for Phase 1C: EvalCase wiring into EvalRunner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import maike.cli
from maike.eval.case_protocol import EvalCase, EvalPhase, PhaseReport
from maike.eval.adapters import workflow_case_to_eval_case, production_scenario_to_eval_case
from maike.eval.contracts import EvalCaseReport
from maike.eval.runner import EvalRunner
from maike.smoke.workflow_cases.common import WorkflowCase


# ---------------------------------------------------------------------------
# EvalCase property tests
# ---------------------------------------------------------------------------


class TestEvalCaseProperties:
    def test_is_react_always_true(self):
        case = EvalCase(
            name="react-test",
            phases=(EvalPhase(task="do something"),),
        )
        assert case.is_react is True

    def test_is_multi_phase(self):
        case = EvalCase(
            name="multi",
            phases=(
                EvalPhase(task="phase 1"),
                EvalPhase(task="phase 2"),
            ),
        )
        assert case.is_multi_phase is True

    def test_is_not_multi_phase(self):
        case = EvalCase(
            name="single",
            phases=(EvalPhase(task="only phase"),),
        )
        assert case.is_multi_phase is False

    def test_task_property(self):
        case = EvalCase(
            name="test",
            phases=(EvalPhase(task="the primary task"),),
        )
        assert case.task == "the primary task"

    def test_task_property_empty_phases(self):
        case = EvalCase(name="empty", phases=())
        assert case.task == ""


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_workflow_case_to_eval_case_preserves_fields(self):
        wc = WorkflowCase(
            name="editing",
            task="Update calculator",
            expected_pipeline="editing",
            expected_stages=("analysis", "planning"),
            expected_stage_artifacts=("analysis.md", "plan.md"),
            setup_workspace=lambda _: None,
            verify_workspace=lambda _: None,
            language="python",
            tags=("core",),
        )
        ec = workflow_case_to_eval_case(wc)
        assert ec.name == "editing"
        assert ec.task == "Update calculator"
        assert ec.expected_pipeline == "editing"
        assert ec.expected_stages == ("analysis", "planning")
        assert ec.expected_stage_artifacts == ("analysis.md", "plan.md")
        assert ec.language == "python"
        assert ec.tags == ("core",)
        assert len(ec.phases) == 1
        assert ec.phases[0].task == "Update calculator"
        assert ec.is_react is True
        assert ec.is_multi_phase is False


# ---------------------------------------------------------------------------
# EvalRunner suite selection tests
# ---------------------------------------------------------------------------


class TestSuiteSelection:
    def test_react_suite_returns_react_cases(self, monkeypatch):
        react_case = EvalCase(
            name="react-greenfield",
            phases=(EvalPhase(task="test"),),
            tags=("react", "core"),
        )
        workflow_case = WorkflowCase(
            name="editing",
            task="edit",
            expected_pipeline="editing",
            expected_stages=("analysis",),
            expected_stage_artifacts=("a.md",),
            setup_workspace=lambda _: None,
            verify_workspace=lambda _: None,
        )
        cases = {"react-greenfield": react_case, "editing": workflow_case}
        monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: cases)

        runner = EvalRunner()
        names = runner._workflow_names_for_suite("react")
        assert names == ["react-greenfield"]
        assert "editing" not in names


# ---------------------------------------------------------------------------
# EvalRunner react case execution tests
# ---------------------------------------------------------------------------


def _make_fake_run_command():
    """Create a fake run_command."""
    async def fake_run_command(**kwargs):
        return SimpleNamespace(
            session_id="session-react-1",
            pipeline="react",
            stage_results={},
        )
    return fake_run_command


def _setup_runner_monkeypatches(monkeypatch, cases):
    """Common monkeypatch setup for runner tests."""
    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: cases)
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: list(cases.keys()))
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr(maike.cli, "run_command", _make_fake_run_command())
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=[],
            artifact_names=[],
            agent_runs=[],
            spawn_requests=[],
        ),
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_session_cost_summary",
        lambda workspace, session_id: {
            "total_cost_usd": 0.10,
            "total_tokens": 100,
            "input_tokens": 60,
            "output_tokens": 40,
            "llm_calls": 1,
            "agent_runs": 1,
            "has_token_breakdown": True,
            "per_stage": [],
        },
    )


def test_react_case_uses_react_scoring(tmp_path, monkeypatch):
    """React cases should use react_eval_score (0.3 + 0.5 = 0.8 with completed + verified)."""
    react_case = EvalCase(
        name="react-greenfield",
        phases=(EvalPhase(task="Create a hello world app"),),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        verify_workspace=lambda _: None,
        tags=("react",),
        budget=5.0,
    )
    _setup_runner_monkeypatches(monkeypatch, {"react-greenfield": react_case})

    report = EvalRunner().run_suite("all", tmp_path)

    result = report.results[0]
    assert result.passed is True
    assert result.case_type == "react"
    # react_eval_score: 0.3 (session_completed) + 0.5 (workspace_verified) = 0.8
    assert result.score == pytest.approx(0.8)


def test_workflow_case_still_uses_workflow_scoring(tmp_path, monkeypatch):
    """WorkflowCase should continue to use workflow_eval_score (backward compat)."""
    wf_case = WorkflowCase(
        name="editing",
        task="Update calculator",
        expected_pipeline="editing",
        expected_stages=("analysis",),
        expected_stage_artifacts=("analysis.md",),
        setup_workspace=lambda _: None,
        verify_workspace=lambda _: None,
    )

    async def fake_run_command(**kwargs):
        return SimpleNamespace(
            session_id="session-wf",
            pipeline="editing",
            stage_results={"analysis": []},
        )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"editing": wf_case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["editing"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis"],
            artifact_names=["analysis.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_session_cost_summary",
        lambda workspace, session_id: {
            "total_cost_usd": 0.10,
            "total_tokens": 100,
            "input_tokens": 60,
            "output_tokens": 40,
            "llm_calls": 1,
            "agent_runs": 1,
            "has_token_breakdown": True,
            "per_stage": [],
        },
    )

    report = EvalRunner().run_suite("all", tmp_path)
    result = report.results[0]
    assert result.case_type == "workflow"
    # workflow_eval_score with all checks passing = 1.0
    assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multi-phase tests
# ---------------------------------------------------------------------------


def test_multi_phase_case_runs_all_phases(tmp_path, monkeypatch):
    """Multi-phase EvalCase should execute each phase sequentially."""
    tasks_seen: list[str] = []

    async def fake_run_command(**kwargs):
        tasks_seen.append(kwargs["task"])
        return SimpleNamespace(
            session_id=f"session-{len(tasks_seen)}",
            pipeline="react",
            stage_results={},
        )

    multi_case = EvalCase(
        name="multi-phase-test",
        phases=(
            EvalPhase(task="Phase 1: build it"),
            EvalPhase(task="Phase 2: explain it", expect_read_only=True),
            EvalPhase(task="Phase 3: test it"),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        tags=("react", "multi-phase"),
        budget=10.0,
    )

    monkeypatch.setattr("maike.eval.runner.load_workflow_cases", lambda: {"multi-phase-test": multi_case})
    monkeypatch.setattr("maike.eval.runner.select_workflow_names", lambda values: ["multi-phase-test"])
    monkeypatch.setattr("maike.eval.runner.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.eval.runner.provider_has_key", lambda provider: True)
    monkeypatch.setattr("maike.eval.runner.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")
    monkeypatch.setattr(EvalRunner, "_initialize_git_repo", lambda self, workspace_path: None)
    monkeypatch.setattr(maike.cli, "run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.eval.runner.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=[],
            artifact_names=[],
            agent_runs=[],
            spawn_requests=[],
        ),
    )
    monkeypatch.setattr(
        "maike.eval.runner.load_session_cost_summary",
        lambda workspace, session_id: {
            "total_cost_usd": 0.05,
            "total_tokens": 50,
            "input_tokens": 30,
            "output_tokens": 20,
            "llm_calls": 1,
            "agent_runs": 1,
            "has_token_breakdown": True,
            "per_stage": [],
        },
    )

    report = EvalRunner().run_suite("all", tmp_path)

    assert len(tasks_seen) == 3
    assert tasks_seen[0] == "Phase 1: build it"
    assert tasks_seen[1] == "Phase 2: explain it"
    assert tasks_seen[2] == "Phase 3: test it"

    result = report.results[0]
    assert result.case_type == "multi_phase"
    assert result.phase_reports is not None
    assert len(result.phase_reports) == 3
    assert result.phase_reports[0].phase_name == "phase-0"
    assert result.phase_reports[1].phase_name == "phase-1"
    assert result.phase_reports[2].phase_name == "phase-2"
    # All phases completed, no verifier → 0.3 + 0.5 = 0.8
    assert result.score == pytest.approx(0.8)


