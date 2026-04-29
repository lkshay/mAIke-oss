"""Simple eval metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdEvaluation:
    delta: float
    delta_ratio: float | None
    warning: bool
    failure: bool


SCORE_DROP_WARN = 0.05
SCORE_DROP_FAIL = 0.10
COST_INCREASE_WARN_RATIO = 0.10
COST_INCREASE_WARN_ABS = 0.02
COST_INCREASE_FAIL_RATIO = 0.25
COST_INCREASE_FAIL_ABS = 0.05
TOKEN_INCREASE_WARN_RATIO = 0.10
TOKEN_INCREASE_WARN_ABS = 50
TOKEN_INCREASE_FAIL_RATIO = 0.25
TOKEN_INCREASE_FAIL_ABS = 100
RETRY_INCREASE_WARN = 1
RETRY_INCREASE_FAIL = 2


import math


def react_eval_score(
    *,
    session_completed: bool,
    workspace_verified: bool,
    tests_pass: bool = False,
) -> float:
    """Scoring function for react-mode eval cases.

    React cases don't have pipeline/stage/artifact checks, so scoring
    weights shift entirely to functional verification:
      - 0.3 session completed
      - 0.5 workspace verified (outcomes correct)
      - 0.2 tests passing
    """
    score = 0.0
    if session_completed:
        score += 0.3
    if workspace_verified:
        score += 0.5
    if tests_pass:
        score += 0.2
    return score


_VERDICT_CAPS: dict[str, float] = {
    "satisfied": 1.0,
    "partial": 0.7,
    "unproductive_budget_exhaustion": 0.15,
    "unproductive_loop": 0.15,
    # cancelled and unknown intentionally omitted → no cap applied (passthrough)
}


def agentic_eval_score(
    *,
    session_completed: bool,
    workspace_verified: bool,
    tests_pass: bool = False,
    error_recovery_rate: float = 0.0,
    wasted_call_ratio: float = 0.0,
    change_minimality_score: float = 1.0,
    verdict_label: str | None = None,
) -> float:
    """Scoring function for agentic eval cases.

    Extends react_eval_score with agentic quality signals:
      - 0.25 session completed
      - 0.35 workspace verified (outcomes correct)
      - 0.15 tests passing
      - 0.10 error recovery effectiveness
      - 0.05 low wasted calls (penalizes inefficiency)
      - 0.10 change minimality (only touched necessary files)

    The optional ``verdict_label`` (from ``SessionVerdict.label``) acts as an
    UPPER CAP on the final score — it never raises it.  Intent: a session
    that hit budget with zero edits cannot score higher than 0.15 even if
    other signals are noisy.  ``cancelled`` / ``unknown`` / ``None`` have no
    cap (passthrough).  This is purely additive / monotonic-tightening and
    preserves all existing score behavior for callers that don't pass
    ``verdict_label``.
    """
    score = 0.0
    if session_completed:
        score += 0.25
    if workspace_verified:
        score += 0.35
    if tests_pass:
        score += 0.15
    score += 0.10 * error_recovery_rate
    score += 0.05 * max(0.0, 1.0 - wasted_call_ratio)
    score += 0.10 * change_minimality_score
    if verdict_label in _VERDICT_CAPS:
        score = min(score, _VERDICT_CAPS[verdict_label])
    return score


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Given *n* total samples with *c* correct, estimates the probability
    that at least one of *k* random draws is correct.

    Reference: Chen et al., "Evaluating Large Language Models Trained
    on Code", 2021.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def cost_efficiency_score(
    actual: float,
    expected: float,
    max_allowed: float,
) -> float:
    """Score cost efficiency on a 0-1 scale.

    Returns 1.0 at or under *expected*, linearly degrades to 0.0 at
    *max_allowed*.
    """
    if actual <= expected:
        return 1.0
    if actual >= max_allowed:
        return 0.0
    return 1.0 - (actual - expected) / (max_allowed - expected)


def pass_fail_score(*, tests_passed: bool, matched_spec: bool) -> float:
    if tests_passed and matched_spec:
        return 1.0
    if tests_passed or matched_spec:
        return 0.5
    return 0.0


def base_workflow_eval_score(
    *,
    session_status_match: bool,
    pipeline_match: bool,
    stage_sequence_match: bool,
    persisted_stage_sequence_match: bool,
    required_stage_gates_present: bool,
    workspace_verified: bool,
) -> float:
    score = 0.0
    if session_status_match:
        score += 0.1
    if pipeline_match:
        score += 0.15
    if stage_sequence_match:
        score += 0.2
    if persisted_stage_sequence_match:
        score += 0.15
    if required_stage_gates_present:
        score += 0.15
    if workspace_verified:
        score += 0.25
    return score


def dynamic_behavior_score(checks: list[bool]) -> float:
    if not checks:
        return 1.0
    satisfied = sum(1 for item in checks if item)
    return satisfied / len(checks)


def workflow_eval_score(
    *,
    session_status_match: bool,
    pipeline_match: bool,
    stage_sequence_match: bool,
    persisted_stage_sequence_match: bool,
    required_stage_gates_present: bool,
    workspace_verified: bool,
    dynamic_score: float | None = None,
) -> float:
    base_score = base_workflow_eval_score(
        session_status_match=session_status_match,
        pipeline_match=pipeline_match,
        stage_sequence_match=stage_sequence_match,
        persisted_stage_sequence_match=persisted_stage_sequence_match,
        required_stage_gates_present=required_stage_gates_present,
        workspace_verified=workspace_verified,
    )
    if dynamic_score is None:
        return base_score
    return (base_score * 0.7) + (dynamic_score * 0.3)


def policy_snapshot() -> dict[str, object]:
    return {
        "score_drop": {
            "warning": SCORE_DROP_WARN,
            "failure": SCORE_DROP_FAIL,
        },
        "cost_increase": {
            "warning_ratio": COST_INCREASE_WARN_RATIO,
            "warning_absolute": COST_INCREASE_WARN_ABS,
            "failure_ratio": COST_INCREASE_FAIL_RATIO,
            "failure_absolute": COST_INCREASE_FAIL_ABS,
        },
        "token_increase": {
            "warning_ratio": TOKEN_INCREASE_WARN_RATIO,
            "warning_absolute": TOKEN_INCREASE_WARN_ABS,
            "failure_ratio": TOKEN_INCREASE_FAIL_RATIO,
            "failure_absolute": TOKEN_INCREASE_FAIL_ABS,
        },
        "retry_increase": {
            "warning": RETRY_INCREASE_WARN,
            "failure": RETRY_INCREASE_FAIL,
        },
        "strict_compare_mode": True,
    }


def evaluate_increase(
    current: float | int | None,
    baseline: float | int | None,
    *,
    warning_ratio: float,
    warning_absolute: float,
    failure_ratio: float,
    failure_absolute: float,
) -> ThresholdEvaluation | None:
    if current is None or baseline is None:
        return None
    current_value = float(current)
    baseline_value = float(baseline)
    delta = current_value - baseline_value
    if delta <= 0:
        return ThresholdEvaluation(delta=delta, delta_ratio=0.0, warning=False, failure=False)
    warning_threshold = max(baseline_value * warning_ratio, warning_absolute)
    failure_threshold = max(baseline_value * failure_ratio, failure_absolute)
    delta_ratio = None if baseline_value == 0 else delta / baseline_value
    return ThresholdEvaluation(
        delta=delta,
        delta_ratio=delta_ratio,
        warning=delta > warning_threshold,
        failure=delta > failure_threshold,
    )


def evaluate_score_drop(current: float | None, baseline: float | None) -> ThresholdEvaluation | None:
    if current is None or baseline is None:
        return None
    delta = float(baseline) - float(current)
    if delta <= 0:
        return ThresholdEvaluation(delta=delta, delta_ratio=0.0, warning=False, failure=False)
    return ThresholdEvaluation(
        delta=delta,
        delta_ratio=None if baseline == 0 else delta / float(baseline),
        warning=delta > SCORE_DROP_WARN,
        failure=delta > SCORE_DROP_FAIL,
    )


def composite_eval_score(
    *,
    functional_score: float,
    cost_efficiency_score_val: float = 1.0,
    latency_score: float = 1.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted composite score from functional, cost, and latency components.

    Default weights: functional=0.6, cost=0.25, latency=0.15
    """
    w = weights or {"functional": 0.6, "cost": 0.25, "latency": 0.15}
    return (
        functional_score * w.get("functional", 0.6)
        + cost_efficiency_score_val * w.get("cost", 0.25)
        + latency_score * w.get("latency", 0.15)
    )


def latency_score(duration_seconds: float, expected_seconds: float, max_seconds: float) -> float:
    """Score latency on a 0-1 scale. 1.0 at or under expected, 0.0 at max."""
    if duration_seconds <= expected_seconds:
        return 1.0
    if duration_seconds >= max_seconds:
        return 0.0
    return 1.0 - (duration_seconds - expected_seconds) / (max_seconds - expected_seconds)


def evaluate_retry_increase(current: int | None, baseline: int | None) -> ThresholdEvaluation | None:
    if current is None or baseline is None:
        return None
    delta = int(current) - int(baseline)
    if delta <= 0:
        return ThresholdEvaluation(delta=delta, delta_ratio=0.0, warning=False, failure=False)
    baseline_value = int(baseline)
    return ThresholdEvaluation(
        delta=float(delta),
        delta_ratio=None if baseline_value == 0 else delta / baseline_value,
        warning=delta >= RETRY_INCREASE_WARN,
        failure=delta >= RETRY_INCREASE_FAIL,
    )


# ---------------------------------------------------------------------------
# Error type classification
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("syntax", re.compile(r"SyntaxError|IndentationError|TabError", re.IGNORECASE)),
    ("import", re.compile(r"ImportError|ModuleNotFoundError|No module named", re.IGNORECASE)),
    ("timeout", re.compile(r"TimeoutError|timed?\s*out|deadline exceeded", re.IGNORECASE)),
    ("assertion", re.compile(r"AssertionError|assert\s+.*failed", re.IGNORECASE)),
    ("runtime", re.compile(
        r"TypeError|ValueError|KeyError|IndexError|AttributeError"
        r"|ZeroDivisionError|FileNotFoundError|PermissionError"
        r"|RuntimeError|OSError|NameError",
        re.IGNORECASE,
    )),
    ("logic", re.compile(r"expected .* but got|mismatch|incorrect|wrong (output|result|answer)", re.IGNORECASE)),
]


def classify_eval_error(error_output: str) -> str:
    """Classify an error string into a category.

    Returns one of: "syntax", "runtime", "logic", "timeout",
    "import", "assertion", "unknown".
    """
    if not error_output:
        return "unknown"
    for category, pattern in _ERROR_PATTERNS:
        if pattern.search(error_output):
            return category
    return "unknown"


# ---------------------------------------------------------------------------
# Latency scoring (reuses the same linear-degradation logic as cost)
# ---------------------------------------------------------------------------

def latency_score(
    actual_seconds: float,
    expected_seconds: float,
    max_allowed_seconds: float,
) -> float:
    """Score latency on a 0-1 scale.

    Delegates to ``cost_efficiency_score`` since the math is identical:
    1.0 at or under *expected_seconds*, linearly degrades to 0.0 at
    *max_allowed_seconds*.
    """
    return cost_efficiency_score(actual_seconds, expected_seconds, max_allowed_seconds)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

# Default weight factors (must sum to 1.0)
_DEFAULT_CORRECTNESS_WEIGHT = 0.5
_DEFAULT_COST_WEIGHT = 0.3
_DEFAULT_LATENCY_WEIGHT = 0.2


def composite_eval_score(
    *,
    correctness: float,
    cost_efficiency: float,
    latency: float,
    correctness_weight: float = _DEFAULT_CORRECTNESS_WEIGHT,
    cost_weight: float = _DEFAULT_COST_WEIGHT,
    latency_weight: float = _DEFAULT_LATENCY_WEIGHT,
) -> float:
    """Weighted composite of correctness, cost, and latency scores.

    Each input score should be in [0, 1].  If a ``GradingConfig`` is
    available, pass its weight fields as the ``*_weight`` arguments.
    """
    return (
        correctness * correctness_weight
        + cost_efficiency * cost_weight
        + latency * latency_weight
    )
