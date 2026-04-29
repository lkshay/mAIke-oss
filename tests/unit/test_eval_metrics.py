from datetime import datetime, timezone

from maike.eval.case_protocol import GradingConfig, PhaseReport
from maike.eval.contracts import CostBudget
from maike.eval.metrics import (
    COST_INCREASE_FAIL_ABS,
    COST_INCREASE_FAIL_RATIO,
    COST_INCREASE_WARN_ABS,
    COST_INCREASE_WARN_RATIO,
    TOKEN_INCREASE_FAIL_ABS,
    TOKEN_INCREASE_FAIL_RATIO,
    TOKEN_INCREASE_WARN_ABS,
    TOKEN_INCREASE_WARN_RATIO,
    classify_eval_error,
    composite_eval_score,
    cost_efficiency_score,
    evaluate_increase,
    evaluate_retry_increase,
    evaluate_score_drop,
    latency_score,
)


def test_evaluate_score_drop_warns_and_fails_at_expected_thresholds():
    warning = evaluate_score_drop(0.93, 1.0)
    failure = evaluate_score_drop(0.85, 1.0)

    assert warning is not None
    assert warning.warning is True
    assert warning.failure is False
    assert failure is not None
    assert failure.warning is True
    assert failure.failure is True


def test_evaluate_token_increase_warns_and_fails_at_expected_thresholds():
    warning = evaluate_increase(
        170,
        100,
        warning_ratio=TOKEN_INCREASE_WARN_RATIO,
        warning_absolute=TOKEN_INCREASE_WARN_ABS,
        failure_ratio=TOKEN_INCREASE_FAIL_RATIO,
        failure_absolute=TOKEN_INCREASE_FAIL_ABS,
    )
    failure = evaluate_increase(
        230,
        100,
        warning_ratio=TOKEN_INCREASE_WARN_RATIO,
        warning_absolute=TOKEN_INCREASE_WARN_ABS,
        failure_ratio=TOKEN_INCREASE_FAIL_RATIO,
        failure_absolute=TOKEN_INCREASE_FAIL_ABS,
    )

    assert warning is not None
    assert warning.warning is True
    assert warning.failure is False
    assert failure is not None
    assert failure.warning is True
    assert failure.failure is True


def test_evaluate_retry_increase_warns_then_fails():
    warning = evaluate_retry_increase(1, 0)
    failure = evaluate_retry_increase(2, 0)

    assert warning is not None
    assert warning.warning is True
    assert warning.failure is False
    assert failure is not None
    assert failure.warning is True
    assert failure.failure is True


def test_evaluate_cost_increase_uses_ratio_or_absolute_floor():
    warning = evaluate_increase(
        0.34,
        0.30,
        warning_ratio=COST_INCREASE_WARN_RATIO,
        warning_absolute=COST_INCREASE_WARN_ABS,
        failure_ratio=COST_INCREASE_FAIL_RATIO,
        failure_absolute=COST_INCREASE_FAIL_ABS,
    )
    failure = evaluate_increase(
        0.39,
        0.30,
        warning_ratio=COST_INCREASE_WARN_RATIO,
        warning_absolute=COST_INCREASE_WARN_ABS,
        failure_ratio=COST_INCREASE_FAIL_RATIO,
        failure_absolute=COST_INCREASE_FAIL_ABS,
    )

    assert warning is not None
    assert warning.warning is True
    assert warning.failure is False
    assert failure is not None
    assert failure.warning is True
    assert failure.failure is True


# ---------------------------------------------------------------------------
# GradingConfig tests
# ---------------------------------------------------------------------------


def test_grading_config_defaults():
    cfg = GradingConfig()
    assert cfg.pass_threshold == 0.7
    assert cfg.correctness_weight == 0.5
    assert cfg.cost_weight == 0.3
    assert cfg.latency_weight == 0.2


def test_grading_config_custom_weights():
    cfg = GradingConfig(
        pass_threshold=0.8,
        correctness_weight=0.6,
        cost_weight=0.2,
        latency_weight=0.2,
    )
    assert cfg.pass_threshold == 0.8
    assert cfg.correctness_weight == 0.6
    assert cfg.cost_weight + cfg.latency_weight == 0.4


def test_grading_config_is_frozen():
    cfg = GradingConfig()
    try:
        cfg.pass_threshold = 0.9  # type: ignore[misc]
        assert False, "should have raised"
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# CostBudget tests
# ---------------------------------------------------------------------------


def test_cost_budget_fields():
    budget = CostBudget(expected_cost_usd=1.0, max_cost_usd=5.0)
    assert budget.expected_cost_usd == 1.0
    assert budget.max_cost_usd == 5.0
    assert budget.expected_tokens is None
    assert budget.max_tokens is None
    assert budget.max_latency_seconds is None


def test_cost_budget_with_resource_limits():
    budget = CostBudget(
        expected_cost_usd=0.5,
        max_cost_usd=1.0,
        max_tokens=1000,
        max_latency_seconds=60.0,
    )
    assert budget.max_tokens == 1000
    assert budget.max_latency_seconds == 60.0


def test_cost_budget_enforcement_exceeded():
    budget = CostBudget(
        expected_cost_usd=0.5,
        max_cost_usd=1.0,
        max_tokens=1000,
        max_latency_seconds=60.0,
    )
    actual_cost = 1.5
    actual_tokens = 1200
    actual_latency = 90.0
    assert actual_cost > budget.max_cost_usd
    assert actual_tokens > budget.max_tokens
    assert actual_latency > budget.max_latency_seconds


def test_cost_budget_efficiency_scoring():
    """CostBudget values feed into cost_efficiency_score correctly."""
    budget = CostBudget(expected_cost_usd=1.0, max_cost_usd=2.0)
    # Under expected: perfect
    assert cost_efficiency_score(0.5, budget.expected_cost_usd, budget.max_cost_usd) == 1.0
    # At max: zero
    assert cost_efficiency_score(budget.max_cost_usd, budget.expected_cost_usd, budget.max_cost_usd) == 0.0


# ---------------------------------------------------------------------------
# PhaseReport latency tracking tests
# ---------------------------------------------------------------------------


def test_phase_report_latency_fields():
    now = datetime.now(timezone.utc)
    report = PhaseReport(
        phase_index=0,
        phase_name="phase-0",
        status="completed",
        duration_seconds=12.5,
        wall_clock_seconds=12.5,
        started_at=now,
        ended_at=now,
    )
    assert report.wall_clock_seconds == 12.5
    assert report.started_at is not None
    assert report.ended_at is not None
    assert report.started_at == report.ended_at  # same instant in this test


def test_phase_report_default_latency_fields():
    report = PhaseReport(phase_index=0, phase_name="phase-0")
    assert report.wall_clock_seconds == 0.0
    assert report.started_at is None
    assert report.ended_at is None


def test_phase_report_error_type_field():
    report = PhaseReport(
        phase_index=0,
        phase_name="phase-0",
        status="error",
        error="SyntaxError: unexpected EOF",
        error_type="syntax",
    )
    assert report.error_type == "syntax"


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------


def test_classify_eval_error_syntax():
    assert classify_eval_error("SyntaxError: invalid syntax at line 5") == "syntax"
    assert classify_eval_error("IndentationError: unexpected indent") == "syntax"


def test_classify_eval_error_import():
    assert classify_eval_error("ImportError: cannot import name 'foo'") == "import"
    assert classify_eval_error("ModuleNotFoundError: No module named 'bar'") == "import"


def test_classify_eval_error_timeout():
    assert classify_eval_error("TimeoutError: operation timed out") == "timeout"
    assert classify_eval_error("deadline exceeded after 30s") == "timeout"


def test_classify_eval_error_assertion():
    assert classify_eval_error("AssertionError: 1 != 2") == "assertion"


def test_classify_eval_error_runtime():
    assert classify_eval_error("TypeError: unsupported operand") == "runtime"
    assert classify_eval_error("ValueError: invalid literal") == "runtime"
    assert classify_eval_error("KeyError: 'missing'") == "runtime"


def test_classify_eval_error_logic():
    assert classify_eval_error("expected 42 but got 0") == "logic"
    assert classify_eval_error("output mismatch on line 3") == "logic"


def test_classify_eval_error_unknown():
    assert classify_eval_error("something completely different happened") == "unknown"
    assert classify_eval_error("") == "unknown"


# ---------------------------------------------------------------------------
# Latency score tests
# ---------------------------------------------------------------------------


def test_latency_score_under_expected():
    assert latency_score(5.0, 10.0, 30.0) == 1.0


def test_latency_score_at_max():
    assert latency_score(30.0, 10.0, 30.0) == 0.0


def test_latency_score_midpoint():
    score = latency_score(20.0, 10.0, 30.0)
    assert abs(score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Composite scoring tests
# ---------------------------------------------------------------------------


def test_composite_eval_score_defaults():
    score = composite_eval_score(
        correctness=1.0,
        cost_efficiency=1.0,
        latency=1.0,
    )
    assert abs(score - 1.0) < 1e-9


def test_composite_eval_score_zero():
    score = composite_eval_score(
        correctness=0.0,
        cost_efficiency=0.0,
        latency=0.0,
    )
    assert abs(score - 0.0) < 1e-9


def test_composite_eval_score_weighted():
    score = composite_eval_score(
        correctness=1.0,
        cost_efficiency=0.0,
        latency=0.0,
    )
    assert abs(score - 0.5) < 1e-9


def test_composite_eval_score_custom_weights():
    score = composite_eval_score(
        correctness=1.0,
        cost_efficiency=1.0,
        latency=0.0,
        correctness_weight=0.6,
        cost_weight=0.2,
        latency_weight=0.2,
    )
    assert abs(score - 0.8) < 1e-9


def test_composite_eval_score_with_grading_config():
    cfg = GradingConfig(
        correctness_weight=0.4,
        cost_weight=0.4,
        latency_weight=0.2,
    )
    score = composite_eval_score(
        correctness=0.8,
        cost_efficiency=0.6,
        latency=1.0,
        correctness_weight=cfg.correctness_weight,
        cost_weight=cfg.cost_weight,
        latency_weight=cfg.latency_weight,
    )
    expected = 0.8 * 0.4 + 0.6 * 0.4 + 1.0 * 0.2
    assert abs(score - expected) < 1e-9
