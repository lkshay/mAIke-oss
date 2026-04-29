"""Shared workflow-case contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


WorkflowVerifier = Callable[[Path], None]
WorkflowRunVerifier = Callable[["WorkflowVerificationContext"], None]
WorkflowSeeder = Callable[[Path], None]


@dataclass(frozen=True)
class DynamicExpectations:
    min_partitions: int = 0
    require_fan_in: bool = False
    min_specialists: int = 0
    require_specialist_useful: bool = False
    min_reflections: int = 0
    require_repair: bool = False
    compare_to_deterministic_baseline: bool = False
    min_retries: int = 0
    max_spawn_depth: int | None = None


@dataclass(frozen=True)
class SessionSnapshot:
    status: str
    stage_names: list[str]
    artifact_names: list[str]
    agent_runs: list[dict[str, Any]]
    spawn_requests: list[dict[str, Any]]
    latest_artifact_names: list[str] = field(default_factory=list)
    active_artifact_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowRunRecord:
    run_result: Any
    snapshot: SessionSnapshot
    cost_summary: Any
    workspace_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowVerificationContext:
    workspace: Path
    runs: tuple[WorkflowRunRecord, ...]
    latest_run: WorkflowRunRecord


@dataclass(frozen=True)
class WorkflowCase:
    name: str
    task: str
    expected_pipeline: str
    expected_stages: tuple[str, ...]
    expected_stage_artifacts: tuple[str, ...]
    setup_workspace: WorkflowSeeder
    verify_workspace: WorkflowVerifier
    language: str = "python"
    dynamic_agents_enabled: bool = False
    parallel_coding_enabled: bool = False
    dynamic_expectations: DynamicExpectations | None = None
    tags: tuple[str, ...] = ()
    budget: float | None = None
    agent_token_budget: int | None = None
    expected_session_statuses: tuple[str, ...] = ("completed",)
    observed_stage_sequence_mode: str = "exact"
    persisted_stage_sequence_mode: str = "exact"
    repeat_runs: int = 1
    reuse_workspace_between_runs: bool = False
    verify_run: WorkflowRunVerifier | None = None


@dataclass(frozen=True)
class WorkflowOutcome:
    case_name: str
    workspace: Path
    provider: str
    model: str
    session_id: str
    pipeline: str
    runs: tuple[WorkflowRunRecord, ...] = ()


class WorkflowExecutionError(RuntimeError):
    """Raised when a live smoke workflow fails."""

    def __init__(self, case_name: str, workspace: Path, message: str) -> None:
        super().__init__(f"{case_name} failed in {workspace}: {message}")
        self.case_name = case_name
        self.workspace = workspace
