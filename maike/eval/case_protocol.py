"""Unified eval case protocol.

EvalCase is the universal type that all eval patterns (WorkflowCase,
ProductionScenario, SSGScenario) adapt into.  EvalRunner operates on
EvalCase exclusively, eliminating three separate execution paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from maike.eval.contracts import CostBudget
from maike.smoke.workflow_cases.common import (
    DynamicExpectations,
    WorkflowSeeder,
    WorkflowVerifier,
)


# ---------------------------------------------------------------------------
# Grading configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GradingConfig:
    """Scoring rubric and pass/fail thresholds for an eval case.

    Weight factors control how the composite score is computed.
    They must sum to 1.0.
    """

    pass_threshold: float = 0.7
    correctness_weight: float = 0.5
    cost_weight: float = 0.3
    latency_weight: float = 0.2


# ---------------------------------------------------------------------------
# Phase-level contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalPhase:
    """One step within a possibly multi-phase eval case."""

    task: str
    budget: float | None = None            # per-phase budget override
    expect_read_only: bool = False         # True → fail if files are mutated


@dataclass(frozen=True)
class PhaseReport:
    """Result of executing a single phase."""

    phase_index: int
    phase_name: str
    session_id: str | None = None
    status: str = "pending"
    cost_usd: float = 0.0
    tokens_used: int = 0
    files_written: int = 0
    read_only_violation: bool = False
    duration_seconds: float = 0.0
    wall_clock_seconds: float = 0.0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None
    error_type: str | None = None


# ---------------------------------------------------------------------------
# The universal eval case
# ---------------------------------------------------------------------------

PhaseVerifier = Callable[[list[PhaseReport], Path], None]


@dataclass(frozen=True)
class EvalCase:
    """Universal eval case — single-phase or multi-phase.

    Single-phase cases (adapted from WorkflowCase or ProductionScenario)
    have ``phases`` of length 1.  Multi-phase cases (adapted from
    SSGScenario) have multiple entries that execute sequentially in the
    same workspace.
    """

    name: str
    phases: tuple[EvalPhase, ...]

    # Pipeline expectations (None = don't check; react cases skip these)
    expected_pipeline: str | None = None
    expected_stages: tuple[str, ...] | None = None
    expected_stage_artifacts: tuple[str, ...] | None = None
    expected_session_statuses: tuple[str, ...] = ("completed",)

    # Workspace management
    setup_workspace: WorkflowSeeder = lambda _: None
    verify_workspace: WorkflowVerifier | None = None
    verify_phases: PhaseVerifier | None = None

    # Agent configuration
    language: str = "python"
    dynamic_agents_enabled: bool = False
    parallel_coding_enabled: bool = False
    budget: float | None = None
    agent_token_budget: int | None = None

    # Dynamic behavior expectations
    dynamic_expectations: DynamicExpectations | None = None

    # Metadata
    tags: tuple[str, ...] = ()

    # Eval-specific settings
    grading: GradingConfig | None = None
    cost_budget: CostBudget | None = None

    # Agentic eval settings
    difficulty_weight: float = 1.0             # 1.0=easy, 2.0=medium, 3.0=hard (for cost normalization)
    expected_modified_files: tuple[str, ...] = ()  # Files the agent SHOULD modify (for change verification)

    # WorkflowCase compat fields (used by EvalRunner for stage-mode scoring)
    observed_stage_sequence_mode: str = "exact"
    persisted_stage_sequence_mode: str = "exact"
    repeat_runs: int = 1
    reuse_workspace_between_runs: bool = False

    @property
    def is_multi_phase(self) -> bool:
        return len(self.phases) > 1

    @property
    def is_react(self) -> bool:
        """Always True — only the react pipeline is supported."""
        return True

    @property
    def task(self) -> str:
        """Primary task (first phase).  Useful for display / logging."""
        return self.phases[0].task if self.phases else ""
