import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from maike.atoms.artifact import Artifact, ArtifactType
from maike.memory.session import SessionStore
from maike.smoke.workflow_cases.common import (
    WorkflowCase,
    WorkflowRunRecord,
    WorkflowVerificationContext,
)
from maike.smoke.workflows import (
    DEFAULT_WORKFLOW_NAMES,
    WORKFLOW_CASES,
    default_live_provider,
    load_session_snapshot,
    run_workflow_case,
    select_workflow_names,
)


def test_select_workflow_names_supports_all_and_explicit_values():
    assert tuple(select_workflow_names(["all"])) == DEFAULT_WORKFLOW_NAMES
    assert select_workflow_names(["react-greenfield"]) == ["react-greenfield"]
    assert select_workflow_names(["react-greenfield", "react-debugging"]) == ["react-greenfield", "react-debugging"]


def test_select_workflow_names_supports_tier_selectors():
    names = select_workflow_names(["tier1"])
    assert "ambiguous-pipeline-selection" in names
    assert "editing-on-large-codebase" in names
    assert "debugging-multi-symptom" in names
    assert "greenfield-with-dependencies" in names


def test_select_workflow_names_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown workflow"):
        select_workflow_names(["unknown"])


def test_default_workflow_names_are_react_cases():
    assert DEFAULT_WORKFLOW_NAMES == ("react-greenfield", "react-editing", "react-debugging")


def test_default_live_provider_prefers_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert default_live_provider() == "gemini"


def test_load_session_snapshot_keeps_latest_historical_stage_outputs(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)

        analysis_v1 = await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.ANALYSIS,
                logical_name="analysis.md",
                content="analysis v1",
                produced_by="agent-a",
                stage_name="analysis",
            ),
        )
        await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.PLAN,
                logical_name="plan.md",
                content="plan v1",
                produced_by="agent-b",
                stage_name="planning",
                depends_on=[analysis_v1.id],
            ),
        )
        await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.ANALYSIS,
                logical_name="analysis.md",
                content="analysis v2",
                produced_by="agent-c",
                stage_name="analysis",
            ),
        )
        return session_id

    session_id = asyncio.run(scenario())

    snapshot = load_session_snapshot(tmp_path, session_id)

    assert snapshot.active_artifact_names == ["analysis.md"]
    assert snapshot.latest_artifact_names == ["analysis.md", "plan.md"]
    assert snapshot.artifact_names == ["analysis.md", "plan.md"]


def test_run_workflow_case_reuses_workspace_and_exposes_repeat_runs(tmp_path, monkeypatch):
    setup_calls: list[Path] = []
    verification_events: list[str] = []

    def setup_workspace(workspace: Path) -> None:
        setup_calls.append(workspace)
        if not (workspace / "seed.txt").exists():
            (workspace / "seed.txt").write_text("seed", encoding="utf-8")

    def verify_workspace(workspace: Path) -> None:
        verification_events.append("workspace")
        assert (workspace / "seed.txt").read_text(encoding="utf-8") == "seed"

    def verify_run(context: WorkflowVerificationContext) -> None:
        verification_events.append("run")
        assert len(context.runs) == 2
        assert context.latest_run.run_result.session_id == "session-2"
        assert context.runs[0].workspace_files == context.runs[1].workspace_files

    case = WorkflowCase(
        name="repeat-workspace-case",
        task="Update the seeded workspace",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning"),
        expected_stage_artifacts=("analysis.md", "plan.md"),
        setup_workspace=setup_workspace,
        verify_workspace=verify_workspace,
        verify_run=verify_run,
        repeat_runs=2,
        reuse_workspace_between_runs=True,
    )

    monkeypatch.setitem(WORKFLOW_CASES, case.name, case)
    monkeypatch.setattr("maike.smoke.workflows.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.smoke.workflows.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")

    async def fake_run_command(**kwargs):
        workspace = kwargs["workspace"]
        run_index = len([event for event in verification_events if event == "command"]) + 1
        verification_events.append("command")
        assert (workspace / "seed.txt").exists()
        return SimpleNamespace(
            session_id=f"session-{run_index}",
            pipeline="editing",
            stage_results={"analysis": [], "planning": []},
        )

    async def fake_cost_summary(workspace: Path, session_id: str):
        del workspace, session_id
        return None

    monkeypatch.setattr("maike.smoke.workflows.run_command", fake_run_command)
    monkeypatch.setattr(
        "maike.smoke.workflows.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning"],
            latest_artifact_names=["analysis.md", "plan.md"],
            artifact_names=["analysis.md", "plan.md"],
            agent_runs=[],
            spawn_requests=[],
        ),
    )
    monkeypatch.setattr("maike.smoke.workflows._load_session_cost_summary", fake_cost_summary)

    outcome = run_workflow_case(case.name, workspace=tmp_path / "workspace")

    assert len(outcome.runs) == 2
    assert len(setup_calls) == 1
    assert verification_events[-2:] == ["workspace", "run"]


def test_run_workflow_case_uses_stage_gate_truth_instead_of_latest_stage_artifacts(tmp_path, monkeypatch):
    case = WorkflowCase(
        name="gated-artifact-history",
        task="Update the calculator",
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=("analysis.md", "plan.md", "code-summary.md", "test-results.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=lambda workspace: None,
    )

    monkeypatch.setitem(WORKFLOW_CASES, case.name, case)
    monkeypatch.setattr("maike.smoke.workflows.default_live_provider", lambda: "gemini")
    monkeypatch.setattr("maike.smoke.workflows.resolve_model_name", lambda provider, model: model or "gemini-2.5-flash")

    async def fake_run_command(**kwargs):
        del kwargs
        return SimpleNamespace(
            session_id="session-1",
            pipeline="editing",
            stage_results={"analysis": [], "planning": [], "coding": [], "testing": []},
        )

    async def fake_cost_summary(workspace: Path, session_id: str):
        del workspace, session_id
        return None

    monkeypatch.setattr("maike.smoke.workflows.run_command", fake_run_command)
    monkeypatch.setattr("maike.smoke.workflows._load_session_cost_summary", fake_cost_summary)
    monkeypatch.setattr(
        "maike.smoke.workflows.load_session_snapshot",
        lambda workspace, session_id: SimpleNamespace(
            status="completed",
            stage_names=["analysis", "planning", "coding", "testing"],
            latest_artifact_names=["analysis.md", "plan.md"],
            artifact_names=["analysis.md", "plan.md"],
            agent_runs=[
                {
                    "stage_name": "testing",
                    "output": "Validation Performed\n- Ran pytest\n\nOverall Status: PASS",
                    "metadata": {},
                }
            ],
            spawn_requests=[],
        ),
    )

    outcome = run_workflow_case(case.name, workspace=tmp_path / "workspace")

    assert outcome.pipeline == "editing"
