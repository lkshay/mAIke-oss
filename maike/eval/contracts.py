"""Shared contracts for formal eval reports and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class EvalMode(str, Enum):
    RUN = "run"
    COMPARE = "compare"
    BASELINE = "baseline"


@dataclass(frozen=True)
class EvalRequest:
    suite: str
    workspace_root: Path
    provider: str | None = None
    model: str | None = None
    budget: float = 5.0
    keep_workspaces: bool = False
    mode: EvalMode = EvalMode.RUN
    output_path: Path | None = None
    k: int = 1                    # Number of trials per case (pass@k)
    adaptive_model: bool = False  # Disabled by default in evals


@dataclass(frozen=True)
class EvalRunMetadata:
    suite: str
    suite_key: str
    provider: str
    model: str
    budget: float
    created_at: str
    git_sha: str | None = None
    report_path: str | None = None
    baseline_path: str | None = None


@dataclass(frozen=True)
class EvalStageCostSummary:
    stage_name: str
    cost_usd: float = 0.0
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    agent_runs: int = 0
    failed_runs: int = 0
    has_token_breakdown: bool = False


@dataclass(frozen=True)
class EvalCostSummary:
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    agent_runs: int = 0
    has_token_breakdown: bool = False
    per_stage: list[EvalStageCostSummary] = field(default_factory=list)


@dataclass(frozen=True)
class DynamicEvalMetrics:
    partition_count: int = 0
    completed_partition_count: int = 0
    fan_in_completed: bool = False
    specialist_spawn_count: int = 0
    useful_specialist_count: int = 0
    reflection_count: int = 0
    blocking_reflection_count: int = 0
    repair_count: int = 0
    retry_count: int = 0
    max_observed_spawn_depth: int = 0
    final_convergence: bool = False
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    baseline_cost_usd: float | None = None
    baseline_tokens: int | None = None
    cost_delta_usd: float | None = None
    token_delta: int | None = None


@dataclass(frozen=True)
class EvalMetrics:
    session_completed: bool = False
    session_status_match: bool = False
    pipeline_match: bool = False
    stage_sequence_match: bool = False
    persisted_stage_sequence_match: bool = False
    required_artifacts_present: bool = False
    required_stage_gates_present: bool = True
    workspace_verified: bool = False
    dynamic: DynamicEvalMetrics = field(default_factory=DynamicEvalMetrics)


@dataclass(frozen=True)
class EvalRegression:
    metric: str
    severity: str
    message: str
    current_value: Any = None
    baseline_value: Any = None
    delta: float | int | None = None
    delta_ratio: float | None = None


@dataclass(frozen=True)
class EvalComparison:
    baseline_workflow_name: str | None = None
    passed: bool = True
    regressions: list[EvalRegression] = field(default_factory=list)
    warnings: list[EvalRegression] = field(default_factory=list)


@dataclass(frozen=True)
class PhaseReport:
    """Result of a single phase in a multi-phase eval case."""
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


@dataclass(frozen=True)
class TrialStatistics:
    """Aggregated statistics from pass@k trials."""
    k: int
    pass_count: int
    pass_at_k: float
    mean_score: float
    score_std_dev: float
    mean_cost_usd: float
    cost_std_dev: float
    mean_tokens: int = 0
    trial_scores: tuple[float, ...] = ()
    trial_passed: tuple[bool, ...] = ()


@dataclass(frozen=True)
class CostBudget:
    """Expected cost parameters for efficiency scoring and resource limits."""
    expected_cost_usd: float
    max_cost_usd: float
    expected_tokens: int | None = None
    max_tokens: int | None = None
    max_latency_seconds: float | None = None


@dataclass(frozen=True)
class CostEfficiencyMetrics:
    """Cost efficiency evaluation result."""
    cost_per_successful_task: float | None = None
    cost_ratio: float = 0.0
    token_ratio: float | None = None
    within_budget: bool = True
    efficiency_score: float = 1.0


class ErrorCategory(str, Enum):
    """Classifies the root cause of a failed eval case."""
    TIMEOUT = "timeout"              # Agent hit time/iteration limit
    BUDGET_EXHAUSTED = "budget"      # Cost or token budget exceeded
    PROVIDER_ERROR = "provider"      # LLM API error (auth, rate limit, server)
    TOOL_ERROR = "tool"              # Tool execution failure (bash, file I/O)
    TEST_FAILURE = "test_failure"    # Tests written but failing
    SPEC_MISMATCH = "spec_mismatch" # Output doesn't match spec/acceptance criteria
    EMPTY_OUTPUT = "empty_output"    # Agent produced no artifacts
    CRASH = "crash"                  # Unhandled exception
    UNKNOWN = "unknown"


def classify_error(error: str | None, failure_reasons: list[str] | None = None) -> ErrorCategory:
    """Classify an eval failure into a category based on error text and failure reasons."""
    texts = []
    if error:
        texts.append(error.lower())
    for reason in failure_reasons or []:
        texts.append(reason.lower())
    combined = " ".join(texts)
    if not combined.strip():
        return ErrorCategory.UNKNOWN
    if any(kw in combined for kw in ("timeout", "timed out", "iteration limit", "max_iterations")):
        return ErrorCategory.TIMEOUT
    if any(kw in combined for kw in ("budget", "cost limit", "token limit", "exceeded")):
        return ErrorCategory.BUDGET_EXHAUSTED
    if any(kw in combined for kw in ("401", "403", "rate limit", "429", "provider", "api error", "authentication")):
        return ErrorCategory.PROVIDER_ERROR
    if any(kw in combined for kw in ("tool error", "bash", "command failed", "file not found", "permission denied")):
        return ErrorCategory.TOOL_ERROR
    if any(kw in combined for kw in ("test fail", "tests fail", "assertion", "pytest", "jest")):
        return ErrorCategory.TEST_FAILURE
    if any(kw in combined for kw in ("spec", "mismatch", "workspace_verified", "acceptance")):
        return ErrorCategory.SPEC_MISMATCH
    if any(kw in combined for kw in ("empty", "no artifact", "no output", "no files")):
        return ErrorCategory.EMPTY_OUTPUT
    if any(kw in combined for kw in ("crash", "traceback", "unhandled", "exception")):
        return ErrorCategory.CRASH
    return ErrorCategory.UNKNOWN


@dataclass(frozen=True)
class AgenticMetrics:
    """Post-run behavioral metrics for agentic eval cases."""

    # Iteration tracking
    iteration_count: int = 0
    fix_test_fix_cycles: int = 0
    # Number of test runs (pytest/jest/etc) the agent issued during the
    # session.  Leading indicator of test-driven self-correction; sessions
    # that never run tests are likely flying blind on SWE-bench.
    test_iteration_count: int = 0

    # Cost normalization
    cost_per_resolved: float | None = None
    difficulty_weight: float = 1.0

    # Wasted call detection
    total_tool_calls: int = 0
    wasted_read_calls: int = 0
    wasted_grep_calls: int = 0
    wasted_call_ratio: float = 0.0

    # Error recovery
    self_corrections: int = 0
    unrecovered_errors: int = 0
    error_recovery_rate: float = 0.0

    # Delegation usage
    sync_delegates: int = 0
    async_delegates: int = 0

    # Multi-file change verification
    files_modified: tuple[str, ...] = ()
    expected_files_modified: tuple[str, ...] = ()
    correct_files_touched: bool = True
    unnecessary_files_touched: tuple[str, ...] = ()
    change_minimality_score: float = 1.0


@dataclass(frozen=True)
class EvalCaseReport:
    workflow_name: str
    provider: str
    model: str
    workspace: str
    session_id: str | None
    pipeline: str | None
    passed: bool
    execution_passed: bool
    comparison_passed: bool
    score: float
    metrics: EvalMetrics
    cost_summary: EvalCostSummary | None = None
    failure_reasons: list[str] = field(default_factory=list)
    warnings: list[EvalRegression] = field(default_factory=list)
    comparison: EvalComparison | None = None
    error: str | None = None
    error_category: ErrorCategory | None = None
    duration_seconds: float = 0.0
    # New: case type and multi-phase/trial support
    case_type: str = "workflow"    # workflow | production | react | multi_phase
    phase_reports: list[PhaseReport] | None = None
    trial_statistics: TrialStatistics | None = None
    cost_efficiency: CostEfficiencyMetrics | None = None
    agentic_metrics: AgenticMetrics | None = None


@dataclass(frozen=True)
class EvalReport:
    schema_version: int
    mode: str
    suite: str
    suite_key: str
    provider: str
    model: str
    budget: float
    created_at: str
    git_sha: str | None
    report_path: str | None
    baseline_path: str | None
    total: int
    passed: int
    failed: int
    average_score: float
    execution_failed: bool
    regression_failed: bool
    warning_count: int
    results: list[EvalCaseReport] = field(default_factory=list)


@dataclass(frozen=True)
class EvalBaseline:
    schema_version: int
    captured_at: str
    suite: str
    suite_key: str
    provider: str
    model: str
    baseline_path: str
    policy: dict[str, Any] = field(default_factory=dict)
    report: EvalReport | None = None


def eval_stage_cost_summary_from_dict(data: dict[str, Any]) -> EvalStageCostSummary:
    return EvalStageCostSummary(
        stage_name=data["stage_name"],
        cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
        tokens_used=int(data.get("tokens_used", 0) or 0),
        input_tokens=int(data.get("input_tokens", 0) or 0),
        output_tokens=int(data.get("output_tokens", 0) or 0),
        llm_calls=int(data.get("llm_calls", 0) or 0),
        agent_runs=int(data.get("agent_runs", 0) or 0),
        failed_runs=int(data.get("failed_runs", 0) or 0),
        has_token_breakdown=bool(data.get("has_token_breakdown", False)),
    )


def eval_cost_summary_from_dict(data: dict[str, Any] | None) -> EvalCostSummary | None:
    if data is None:
        return None
    return EvalCostSummary(
        total_cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
        total_tokens=int(data.get("total_tokens", 0) or 0),
        input_tokens=int(data.get("input_tokens", 0) or 0),
        output_tokens=int(data.get("output_tokens", 0) or 0),
        llm_calls=int(data.get("llm_calls", 0) or 0),
        agent_runs=int(data.get("agent_runs", 0) or 0),
        has_token_breakdown=bool(data.get("has_token_breakdown", False)),
        per_stage=[
            eval_stage_cost_summary_from_dict(item)
            for item in data.get("per_stage", [])
        ],
    )


def dynamic_eval_metrics_from_dict(data: dict[str, Any] | None) -> DynamicEvalMetrics:
    if data is None:
        return DynamicEvalMetrics()
    return DynamicEvalMetrics(
        partition_count=int(data.get("partition_count", 0) or 0),
        completed_partition_count=int(data.get("completed_partition_count", 0) or 0),
        fan_in_completed=bool(data.get("fan_in_completed", False)),
        specialist_spawn_count=int(data.get("specialist_spawn_count", 0) or 0),
        useful_specialist_count=int(data.get("useful_specialist_count", 0) or 0),
        reflection_count=int(data.get("reflection_count", 0) or 0),
        blocking_reflection_count=int(data.get("blocking_reflection_count", 0) or 0),
        repair_count=int(data.get("repair_count", 0) or 0),
        retry_count=int(data.get("retry_count", 0) or 0),
        max_observed_spawn_depth=int(data.get("max_observed_spawn_depth", 0) or 0),
        final_convergence=bool(data.get("final_convergence", False)),
        total_cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
        total_tokens=int(data.get("total_tokens", 0) or 0),
        baseline_cost_usd=(
            float(data["baseline_cost_usd"])
            if data.get("baseline_cost_usd") is not None
            else None
        ),
        baseline_tokens=(
            int(data["baseline_tokens"])
            if data.get("baseline_tokens") is not None
            else None
        ),
        cost_delta_usd=(
            float(data["cost_delta_usd"])
            if data.get("cost_delta_usd") is not None
            else None
        ),
        token_delta=(
            int(data["token_delta"])
            if data.get("token_delta") is not None
            else None
        ),
    )


def eval_metrics_from_dict(data: dict[str, Any]) -> EvalMetrics:
    return EvalMetrics(
        session_completed=bool(data.get("session_completed", False)),
        session_status_match=bool(data.get("session_status_match", data.get("session_completed", False))),
        pipeline_match=bool(data.get("pipeline_match", False)),
        stage_sequence_match=bool(data.get("stage_sequence_match", False)),
        persisted_stage_sequence_match=bool(data.get("persisted_stage_sequence_match", False)),
        required_artifacts_present=bool(data.get("required_artifacts_present", False)),
        required_stage_gates_present=bool(data.get("required_stage_gates_present", True)),
        workspace_verified=bool(data.get("workspace_verified", False)),
        dynamic=dynamic_eval_metrics_from_dict(data.get("dynamic")),
    )


def eval_regression_from_dict(data: dict[str, Any]) -> EvalRegression:
    return EvalRegression(
        metric=data["metric"],
        severity=data["severity"],
        message=data["message"],
        current_value=data.get("current_value"),
        baseline_value=data.get("baseline_value"),
        delta=data.get("delta"),
        delta_ratio=data.get("delta_ratio"),
    )


def eval_comparison_from_dict(data: dict[str, Any] | None) -> EvalComparison | None:
    if data is None:
        return None
    return EvalComparison(
        baseline_workflow_name=data.get("baseline_workflow_name"),
        passed=bool(data.get("passed", True)),
        regressions=[eval_regression_from_dict(item) for item in data.get("regressions", [])],
        warnings=[eval_regression_from_dict(item) for item in data.get("warnings", [])],
    )


def eval_case_report_from_dict(data: dict[str, Any]) -> EvalCaseReport:
    execution_passed = bool(data.get("execution_passed", data.get("passed", False)))
    comparison_passed = bool(data.get("comparison_passed", True))
    failure_reasons = list(data.get("failure_reasons", []))
    error = data.get("error")
    error_cat_raw = data.get("error_category")
    error_category = ErrorCategory(error_cat_raw) if error_cat_raw else (
        classify_error(error, failure_reasons) if (error or failure_reasons) and not execution_passed else None
    )
    return EvalCaseReport(
        workflow_name=data["workflow_name"],
        provider=data["provider"],
        model=data["model"],
        workspace=data["workspace"],
        session_id=data.get("session_id"),
        pipeline=data.get("pipeline"),
        passed=bool(data.get("passed", execution_passed and comparison_passed)),
        execution_passed=execution_passed,
        comparison_passed=comparison_passed,
        score=float(data.get("score", 0.0) or 0.0),
        metrics=eval_metrics_from_dict(data["metrics"]),
        cost_summary=eval_cost_summary_from_dict(data.get("cost_summary")),
        failure_reasons=failure_reasons,
        warnings=[eval_regression_from_dict(item) for item in data.get("warnings", [])],
        comparison=eval_comparison_from_dict(data.get("comparison")),
        error=error,
        error_category=error_category,
        duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
    )


def eval_report_from_dict(data: dict[str, Any]) -> EvalReport:
    results = [eval_case_report_from_dict(item) for item in data.get("results", [])]
    execution_failed = bool(
        data.get("execution_failed", any(not result.execution_passed for result in results))
    )
    regression_failed = bool(
        data.get(
            "regression_failed",
            any(not result.comparison_passed for result in results),
        )
    )
    warning_count = int(
        data.get(
            "warning_count",
            sum(len(result.warnings) for result in results),
        )
    )
    return EvalReport(
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        mode=data.get("mode", EvalMode.RUN.value),
        suite=data["suite"],
        suite_key=data.get("suite_key", data["suite"]),
        provider=data["provider"],
        model=data["model"],
        budget=float(data.get("budget", 0.0) or 0.0),
        created_at=data["created_at"],
        git_sha=data.get("git_sha"),
        report_path=data.get("report_path"),
        baseline_path=data.get("baseline_path"),
        total=int(data.get("total", len(results))),
        passed=int(data.get("passed", sum(1 for result in results if result.passed))),
        failed=int(data.get("failed", sum(1 for result in results if not result.passed))),
        average_score=float(data.get("average_score", 0.0) or 0.0),
        execution_failed=execution_failed,
        regression_failed=regression_failed,
        warning_count=warning_count,
        results=results,
    )


def eval_baseline_from_dict(data: dict[str, Any]) -> EvalBaseline:
    baseline_path = data.get("baseline_path")
    if not baseline_path:
        raise ValueError("Baseline payload missing baseline_path")
    report_data = data.get("report")
    return EvalBaseline(
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        captured_at=data["captured_at"],
        suite=data["suite"],
        suite_key=data.get("suite_key", data["suite"]),
        provider=data["provider"],
        model=data["model"],
        baseline_path=baseline_path,
        policy=dict(data.get("policy", {})),
        report=eval_report_from_dict(report_data) if report_data else None,
    )
