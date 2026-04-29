"""Adapters converting legacy eval types to the unified EvalCase protocol.

Each adapter is a pure function — zero changes to the original type.
"""

from __future__ import annotations

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.common import WorkflowCase


# ---------------------------------------------------------------------------
# WorkflowCase → EvalCase
# ---------------------------------------------------------------------------

def workflow_case_to_eval_case(wc: WorkflowCase) -> EvalCase:
    """Wrap a WorkflowCase in the universal EvalCase protocol."""
    return EvalCase(
        name=wc.name,
        phases=(EvalPhase(task=wc.task),),
        expected_pipeline=wc.expected_pipeline,
        expected_stages=wc.expected_stages,
        expected_stage_artifacts=wc.expected_stage_artifacts,
        expected_session_statuses=wc.expected_session_statuses,
        setup_workspace=wc.setup_workspace,
        verify_workspace=wc.verify_workspace,
        language=wc.language,
        dynamic_agents_enabled=wc.dynamic_agents_enabled,
        parallel_coding_enabled=wc.parallel_coding_enabled,
        budget=wc.budget,
        agent_token_budget=wc.agent_token_budget,
        dynamic_expectations=wc.dynamic_expectations,
        tags=wc.tags,
        observed_stage_sequence_mode=wc.observed_stage_sequence_mode,
        persisted_stage_sequence_mode=wc.persisted_stage_sequence_mode,
        repeat_runs=wc.repeat_runs,
        reuse_workspace_between_runs=wc.reuse_workspace_between_runs,
    )


# ---------------------------------------------------------------------------
# ProductionScenario → EvalCase
# ---------------------------------------------------------------------------

def production_scenario_to_eval_case(ps: "ProductionScenario") -> EvalCase:  # noqa: F821
    """Wrap a ProductionScenario in the universal EvalCase protocol.

    Uses string annotation for ProductionScenario to avoid a hard import
    of the heavy production_scenarios module at import time.
    """
    return EvalCase(
        name=ps.name,
        phases=(EvalPhase(task=ps.task),),
        expected_pipeline=ps.expected_pipeline,
        expected_stages=ps.expected_stages,
        expected_stage_artifacts=ps.expected_stage_artifacts,
        setup_workspace=ps.setup_workspace,
        verify_workspace=ps.verify_workspace,
        language=ps.language,
        dynamic_agents_enabled=ps.dynamic_agents_enabled,
        parallel_coding_enabled=ps.parallel_coding_enabled,
        agent_token_budget=ps.agent_token_budget,
        tags=("production",),
    )


# ---------------------------------------------------------------------------
# SSG multi-phase scenario → EvalCase
# ---------------------------------------------------------------------------

def ssg_scenario_to_eval_case() -> EvalCase:
    """Build the SSG multi-phase eval case from the canonical phase list.

    Imports the phase definitions lazily to avoid circular imports.
    """
    from maike.smoke.ssg_scenario import _PHASES

    phases: list[EvalPhase] = []
    for phase_def in _PHASES:
        phases.append(
            EvalPhase(
                task=phase_def["task"],
                budget=phase_def.get("budget"),
                expect_read_only=phase_def.get("expect_read_only", False),
            )
        )

    return EvalCase(
        name="ssg-workflow",
        phases=tuple(phases),
        # React mode — no pipeline/stage expectations
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        # The SSG scenario does its own verification in verify_phases
        verify_workspace=None,
        language="python",
        tags=("react", "multi-phase", "ssg"),
        budget=10.0,
    )
