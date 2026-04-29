"""Live workflow smoke harness for real provider runs."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from maike.cli import run_command
from maike.constants import DEFAULT_PROVIDER, DEFAULT_RUN_BUDGET_USD
from maike.gateway.providers import resolve_model_name
from maike.memory.session import SessionStore
from maike.smoke.workflow_cases.common import (
    SessionSnapshot,
    WorkflowExecutionError,
    WorkflowOutcome,
    WorkflowRunRecord,
    WorkflowVerificationContext,
)
from maike.smoke.workflow_cases.helpers import snapshot_workspace_files
from maike.smoke.workflow_cases.registry import (
    DEFAULT_WORKFLOW_NAMES,
    TIER_WORKFLOW_NAMES,
    WORKFLOW_CASES,
)
from maike.utils import dedupe_preserve_order


def provider_has_key(provider: str) -> bool:
    load_dotenv()
    env_keys = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    }
    return any(os.getenv(key) for key in env_keys.get(provider, ()))


def default_live_provider() -> str:
    for provider in ("gemini", "openai", "anthropic"):
        if provider_has_key(provider):
            return provider
    return DEFAULT_PROVIDER


def select_workflow_names(values: Iterable[str]) -> list[str]:
    from maike.smoke.workflow_cases.registry import _ensure_all_cases
    _ensure_all_cases()
    requested = [item.strip().lower() for item in values if item.strip()]
    if not requested or "all" in requested:
        return list(DEFAULT_WORKFLOW_NAMES)
    resolved: list[str] = []
    invalid: list[str] = []
    for item in requested:
        if item in TIER_WORKFLOW_NAMES:
            resolved.extend(TIER_WORKFLOW_NAMES[item])
            continue
        if item in WORKFLOW_CASES:
            resolved.append(item)
            continue
        invalid.append(item)
    if invalid:
        valid = ", ".join(dedupe_preserve_order(["all", *WORKFLOW_CASES.keys(), *TIER_WORKFLOW_NAMES.keys()]))
        raise ValueError(f"Unknown workflow(s): {', '.join(invalid)}. Valid options: {valid}")
    return dedupe_preserve_order(resolved)


async def _load_session_snapshot(workspace: Path, session_id: str) -> SessionSnapshot:
    from maike.atoms.artifact import ArtifactKind

    store = SessionStore(workspace)
    await store.initialize()
    session_row = await store.get_session(session_id)
    if session_row is None:
        raise AssertionError(f"Missing session row for {session_id}")
    agent_runs = await store.get_agent_runs(session_id)
    active_artifacts = await store.list_artifacts(session_id, active_only=True, kind=ArtifactKind.STAGE)
    historical_artifacts = await store.list_artifacts(session_id, active_only=False, kind=ArtifactKind.STAGE)
    spawn_requests = await store.list_spawn_requests(session_id)
    latest_artifact_names = _latest_stage_artifact_names(historical_artifacts)
    active_artifact_names = [artifact.logical_name for artifact in active_artifacts]
    return SessionSnapshot(
        status=session_row["status"],
        stage_names=dedupe_preserve_order([row["stage_name"] for row in agent_runs]),
        artifact_names=latest_artifact_names,
        agent_runs=agent_runs,
        spawn_requests=spawn_requests,
        latest_artifact_names=latest_artifact_names,
        active_artifact_names=active_artifact_names,
    )


def load_session_snapshot(workspace: Path, session_id: str) -> SessionSnapshot:
    return asyncio.run(_load_session_snapshot(workspace, session_id))


async def _load_session_cost_summary(workspace: Path, session_id: str) -> dict[str, Any] | None:
    store = SessionStore(workspace)
    await store.initialize()
    return await store.get_session_cost(session_id)


def _latest_session(workspace: Path, session_id: str) -> tuple[str, list[str], list[str]]:
    snapshot = load_session_snapshot(workspace, session_id)
    artifact_names = snapshot.latest_artifact_names or snapshot.artifact_names
    return snapshot.status, snapshot.stage_names, artifact_names


def _latest_stage_artifact_names(artifacts: list[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.logical_name in seen:
            continue
        seen.add(artifact.logical_name)
        names.append(artifact.logical_name)
    return names


def _matches_stage_sequence(observed: tuple[str, ...], expected: tuple[str, ...], mode: str) -> bool:
    if mode == "exact":
        return observed == expected
    if mode == "prefix":
        return observed == expected[: len(observed)]
    raise ValueError(f"Unsupported stage sequence mode: {mode}")


def run_workflow_case(
    workflow_name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    budget: float = DEFAULT_RUN_BUDGET_USD,
    workspace: Path | None = None,
) -> WorkflowOutcome:
    from maike.smoke.workflow_cases.registry import _ensure_all_cases
    _ensure_all_cases()
    load_dotenv()
    case = WORKFLOW_CASES[workflow_name]
    resolved_provider = provider or default_live_provider()
    resolved_model = resolve_model_name(resolved_provider, model)
    workspace_path = workspace or Path(tempfile.mkdtemp(prefix=f"maike-{case.name}-"))
    workspace_path.mkdir(parents=True, exist_ok=True)

    try:
        runs: list[WorkflowRunRecord] = []
        seeded = False
        for run_index in range(case.repeat_runs):
            if run_index == 0 or not case.reuse_workspace_between_runs:
                if run_index > 0 and not case.reuse_workspace_between_runs:
                    for child in workspace_path.iterdir():
                        if child.is_dir():
                            import shutil
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink()
                case.setup_workspace(workspace_path)
                seeded = True
            elif not seeded:
                case.setup_workspace(workspace_path)
                seeded = True

            run_result = asyncio.run(
                run_command(
                    task=case.task,
                    workspace=workspace_path,
                    provider=resolved_provider,
                    model=resolved_model,
                    language=case.language,
                    budget=case.budget if case.budget is not None else budget,
                    agent_token_budget=case.agent_token_budget,
                    yes=True,
                    dynamic_agents_enabled=case.dynamic_agents_enabled,
                    parallel_coding_enabled=case.parallel_coding_enabled,
                )
            )
            snapshot = load_session_snapshot(workspace_path, run_result.session_id)
            cost_summary = asyncio.run(_load_session_cost_summary(workspace_path, run_result.session_id))
            runs.append(
                WorkflowRunRecord(
                    run_result=run_result,
                    snapshot=snapshot,
                    cost_summary=cost_summary,
                    workspace_files=snapshot_workspace_files(workspace_path),
                )
            )

        latest = runs[-1]
        observed_stages = tuple(latest.run_result.stage_results)
        persisted_stages = tuple(latest.snapshot.stage_names)
        if latest.run_result.pipeline != case.expected_pipeline:
            raise AssertionError(f"Expected pipeline {case.expected_pipeline}, got {latest.run_result.pipeline}")
        if latest.snapshot.status not in case.expected_session_statuses:
            raise AssertionError(
                f"Expected session status in {case.expected_session_statuses}, got {latest.snapshot.status!r}"
            )
        if not _matches_stage_sequence(observed_stages, case.expected_stages, case.observed_stage_sequence_mode):
            raise AssertionError(f"Expected observed stages {case.expected_stages}, got {observed_stages}")
        if not _matches_stage_sequence(persisted_stages, case.expected_stages, case.persisted_stage_sequence_mode):
            raise AssertionError(f"Expected persisted stages {case.expected_stages}, got {persisted_stages}")
        case.verify_workspace(workspace_path)
        if case.verify_run is not None:
            context = WorkflowVerificationContext(workspace=workspace_path, runs=tuple(runs), latest_run=latest)
            case.verify_run(context)

        return WorkflowOutcome(
            case_name=case.name,
            workspace=workspace_path,
            provider=resolved_provider,
            model=resolved_model,
            session_id=latest.run_result.session_id,
            pipeline=latest.run_result.pipeline,
            runs=tuple(runs),
        )
    except Exception as exc:
        raise WorkflowExecutionError(case.name, workspace_path, str(exc)) from exc
