import pytest

from maike.atoms.context import AgentContext
from maike.atoms.llm import LLMCallRecord
from maike.constants import DEFAULT_MODEL, DISABLED_AGENT_TOKEN_BUDGET
from maike.cost.tracker import BudgetEnforcer, BudgetExceededError, CostTracker


def make_record(*, cost_usd: float, input_tokens: int = 100, output_tokens: int = 50) -> LLMCallRecord:
    return LLMCallRecord(
        provider="anthropic",
        model=DEFAULT_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=12,
        stop_reason="end_turn",
    )


def test_cost_tracker_retains_records_and_updates_totals():
    tracker = CostTracker()
    first = make_record(cost_usd=0.25)
    second = make_record(cost_usd=0.30, input_tokens=80, output_tokens=20)

    tracker.record(first)
    tracker.record(second)

    assert tracker.session_total == 0.55
    assert tracker.records == [first, second]


def test_cost_tracker_checks_budget_after_recording_over_budget_call():
    tracker = CostTracker(session_budget_usd=0.50)
    tracker.record(make_record(cost_usd=0.30))
    tracker.check_session_budget()

    over_budget_call = make_record(cost_usd=0.25)
    tracker.record(over_budget_call)

    with pytest.raises(BudgetExceededError, match="Session cost budget exceeded"):
        tracker.check_session_budget()

    assert tracker.session_total == 0.55
    assert tracker.records[-1] == over_budget_call


def test_cost_tracker_checks_projected_budget_before_call():
    tracker = CostTracker(session_budget_usd=1.00)
    tracker.record(make_record(cost_usd=0.80))

    with pytest.raises(BudgetExceededError, match="Projected session cost budget exceeded before LLM call"):
        tracker.check_projected_session_budget(0.20, safety_margin=0.95)


def test_budget_enforcer_reserves_next_call_budget():
    ctx = AgentContext(
        role="coder",
        task="build",
        stage_name="coding",
        tool_profile="coding",
        tokens_used=900,
        token_budget=1_000,
        cost_used_usd=0.80,
        cost_budget_usd=1.00,
    )

    enforcer = BudgetEnforcer()

    with pytest.raises(BudgetExceededError, match="token budget"):
        enforcer.check(ctx, reserved_tokens=101)

    with pytest.raises(BudgetExceededError, match="cost budget"):
        enforcer.check(ctx, reserved_cost_usd=0.25)


def test_budget_enforcer_skips_token_limit_when_disabled():
    ctx = AgentContext(
        role="coder",
        task="build",
        stage_name="coding",
        tool_profile="coding",
        tokens_used=90_682,
        token_budget=DISABLED_AGENT_TOKEN_BUDGET,
        cost_used_usd=0.10,
        cost_budget_usd=1.00,
    )

    enforcer = BudgetEnforcer()

    enforcer.check(ctx, reserved_tokens=17_912)
