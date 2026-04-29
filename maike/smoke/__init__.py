"""Workflow smoke helpers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PRODUCTION_SCENARIOS",
    "DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT",
    "ProductionScenario",
    "ProductionScenarioExecutionError",
    "ProductionScenarioOutcome",
    "WORKFLOW_CASES",
    "DEFAULT_WORKFLOW_NAMES",
    "WorkflowCase",
    "WorkflowExecutionError",
    "WorkflowOutcome",
    "WorkflowRunRecord",
    "WorkflowVerificationContext",
    "default_live_provider",
    "path_is_within",
    "provider_has_key",
    "run_production_scenario",
    "run_workflow_case",
    "select_production_scenario_names",
    "select_workflow_names",
]

_EXPORTS = {
    "PRODUCTION_SCENARIOS": ("maike.smoke.production_scenarios", "PRODUCTION_SCENARIOS"),
    "DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT": (
        "maike.smoke.production_scenarios",
        "DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT",
    ),
    "ProductionScenario": ("maike.smoke.production_scenarios", "ProductionScenario"),
    "ProductionScenarioExecutionError": (
        "maike.smoke.production_scenarios",
        "ProductionScenarioExecutionError",
    ),
    "ProductionScenarioOutcome": ("maike.smoke.production_scenarios", "ProductionScenarioOutcome"),
    "path_is_within": ("maike.smoke.production_scenarios", "path_is_within"),
    "run_production_scenario": ("maike.smoke.production_scenarios", "run_production_scenario"),
    "select_production_scenario_names": (
        "maike.smoke.production_scenarios",
        "select_production_scenario_names",
    ),
    "WORKFLOW_CASES": ("maike.smoke.workflows", "WORKFLOW_CASES"),
    "DEFAULT_WORKFLOW_NAMES": ("maike.smoke.workflows", "DEFAULT_WORKFLOW_NAMES"),
    "WorkflowCase": ("maike.smoke.workflows", "WorkflowCase"),
    "WorkflowExecutionError": ("maike.smoke.workflows", "WorkflowExecutionError"),
    "WorkflowOutcome": ("maike.smoke.workflows", "WorkflowOutcome"),
    "WorkflowRunRecord": ("maike.smoke.workflows", "WorkflowRunRecord"),
    "WorkflowVerificationContext": ("maike.smoke.workflows", "WorkflowVerificationContext"),
    "default_live_provider": ("maike.smoke.workflows", "default_live_provider"),
    "provider_has_key": ("maike.smoke.workflows", "provider_has_key"),
    "run_workflow_case": ("maike.smoke.workflows", "run_workflow_case"),
    "select_workflow_names": ("maike.smoke.workflows", "select_workflow_names"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
