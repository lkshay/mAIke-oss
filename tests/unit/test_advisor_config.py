"""Unit tests for Phase 1 of the advisor pattern: config, AdvisorSession scaffold."""

from __future__ import annotations

import asyncio

import pytest

from maike.agents.advisor import (
    AdvisorConfig,
    AdvisorSession,
    AdvisorTrigger,
    AdvisorUrgency,
    AdvisorVerdict,
    exploration_threshold_met,
    load_advisor_prompt,
    resolve_advisor_config,
)


# ── resolve_advisor_config ──────────────────────────────────────────


def test_disabled_returns_disabled_config():
    cfg = resolve_advisor_config(
        enabled=False,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        session_budget_usd=5.0,
        budget_fraction=0.2,
    )
    assert cfg.enabled is False
    assert cfg.provider == ""
    assert cfg.model == ""


def test_enabled_defaults_to_strong_tier_same_provider():
    cfg = resolve_advisor_config(
        enabled=True,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        session_budget_usd=5.0,
        budget_fraction=0.2,
    )
    assert cfg.enabled is True
    assert cfg.provider == "gemini"
    # Strong tier of gemini is gemini-2.5-pro (see constants.py).
    assert "pro" in cfg.model.lower() or "2.5" in cfg.model


def test_explicit_overrides_win():
    cfg = resolve_advisor_config(
        enabled=True,
        executor_provider="gemini",
        advisor_provider="anthropic",
        advisor_model="claude-opus-4-6",
        session_budget_usd=3.0,
        budget_fraction=0.3,
    )
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-opus-4-6"


def test_budget_computed_as_fraction():
    cfg = resolve_advisor_config(
        enabled=True,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        session_budget_usd=10.0,
        budget_fraction=0.2,
    )
    assert cfg.budget_usd == pytest.approx(2.0)


def test_zero_session_budget_gives_unlimited_advisor_budget():
    cfg = resolve_advisor_config(
        enabled=True,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        session_budget_usd=0.0,
        budget_fraction=0.2,
    )
    # 0.0 is the "unlimited" sentinel used elsewhere in the codebase.
    assert cfg.budget_usd == 0.0


def test_fraction_clamped_to_sensible_range():
    cfg_high = resolve_advisor_config(
        enabled=True,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        session_budget_usd=10.0,
        budget_fraction=5.0,  # nonsense
    )
    assert cfg_high.budget_usd == pytest.approx(10.0)  # capped at 100%

    cfg_neg = resolve_advisor_config(
        enabled=True,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        session_budget_usd=10.0,
        budget_fraction=-0.5,  # nonsense
    )
    assert cfg_neg.budget_usd == 0.0


# ── AdvisorSession scaffold ──────────────────────────────────────────


def test_advisor_session_disabled_when_gateway_none():
    cfg = AdvisorConfig(enabled=True)
    session = AdvisorSession(gateway=None, config=cfg)
    assert session.enabled is False  # disabled because gateway is None


def test_advisor_session_disabled_when_config_disabled():
    cfg = AdvisorConfig(enabled=False)

    class _DummyGw:
        pass

    session = AdvisorSession(gateway=_DummyGw(), config=cfg)
    assert session.enabled is False


def test_advisor_session_enabled_when_both_set():
    cfg = AdvisorConfig(enabled=True, provider="gemini", model="x", budget_usd=1.0)

    class _DummyGw:
        pass

    session = AdvisorSession(gateway=_DummyGw(), config=cfg)
    assert session.enabled is True


def test_advisor_budget_remaining():
    cfg = AdvisorConfig(enabled=True, budget_usd=1.0)

    class _DummyGw:
        pass

    session = AdvisorSession(gateway=_DummyGw(), config=cfg)
    assert session.budget_remaining_usd == 1.0
    session.cost_spent_usd = 0.7
    assert session.budget_remaining_usd == pytest.approx(0.3)
    session.cost_spent_usd = 2.0
    assert session.budget_remaining_usd == 0.0  # never negative


def test_advise_returns_throttled_when_disabled():
    cfg = AdvisorConfig(enabled=False)
    session = AdvisorSession(gateway=None, config=cfg)

    async def _go():
        return await session.advise(
            question="help",
            urgency=AdvisorUrgency.NORMAL,
            trigger=AdvisorTrigger.TOOL,
            conversation=[],
            ctx=_DummyCtx(),
            iteration_count=5,
        )

    verdict = asyncio.run(_go())
    assert verdict.throttled is True
    assert verdict.throttle_reason == "disabled"


def test_advise_throttles_on_cooldown():
    cfg = AdvisorConfig(enabled=True, cooldown_iterations=3)

    class _DummyGw:
        pass

    session = AdvisorSession(gateway=_DummyGw(), config=cfg)
    session.last_call_iteration = 5  # just called at iteration 5

    async def _go():
        return await session.advise(
            question="help",
            urgency=AdvisorUrgency.NORMAL,
            trigger=AdvisorTrigger.TOOL,
            conversation=[],
            ctx=_DummyCtx(),
            iteration_count=6,  # only 1 iter since last call
        )

    verdict = asyncio.run(_go())
    assert verdict.throttled is True
    assert "cooldown" in verdict.throttle_reason


def test_advise_throttles_on_max_calls():
    cfg = AdvisorConfig(enabled=True, max_calls=2, cooldown_iterations=0)

    class _DummyGw:
        pass

    session = AdvisorSession(gateway=_DummyGw(), config=cfg)
    session.call_count = 2

    async def _go():
        return await session.advise(
            question="help",
            urgency=AdvisorUrgency.NORMAL,
            trigger=AdvisorTrigger.TOOL,
            conversation=[],
            ctx=_DummyCtx(),
            iteration_count=100,
        )

    verdict = asyncio.run(_go())
    assert verdict.throttled is True
    assert verdict.throttle_reason == "max_calls_reached"


def test_advise_throttles_on_budget_exhausted():
    cfg = AdvisorConfig(enabled=True, budget_usd=1.0, cooldown_iterations=0)

    class _DummyGw:
        pass

    session = AdvisorSession(gateway=_DummyGw(), config=cfg)
    session.cost_spent_usd = 1.5  # over budget

    async def _go():
        return await session.advise(
            question="help",
            urgency=AdvisorUrgency.NORMAL,
            trigger=AdvisorTrigger.TOOL,
            conversation=[],
            ctx=_DummyCtx(),
            iteration_count=100,
        )

    verdict = asyncio.run(_go())
    assert verdict.throttled is True
    assert verdict.throttle_reason == "budget_exhausted"


def test_record_verdict_updates_state():
    cfg = AdvisorConfig(enabled=True, budget_usd=2.0)
    session = AdvisorSession(gateway=object(), config=cfg)
    verdict = AdvisorVerdict(
        advice="try X",
        urgency=AdvisorUrgency.STUCK,
        trigger=AdvisorTrigger.ON_STUCK,
        cost_usd=0.05,
        tokens_used=120,
    )
    session.record_verdict(verdict, iteration_count=7)
    assert session.call_count == 1
    assert session.cost_spent_usd == pytest.approx(0.05)
    assert session.last_call_iteration == 7
    assert "on_stuck" in session.triggers_fired
    assert verdict in session.previous_verdicts


def test_record_verdict_ignores_throttled():
    cfg = AdvisorConfig(enabled=True)
    session = AdvisorSession(gateway=object(), config=cfg)
    throttled = AdvisorVerdict.throttled_verdict(
        AdvisorUrgency.NORMAL, AdvisorTrigger.TOOL, "disabled",
    )
    session.record_verdict(throttled, iteration_count=3)
    assert session.call_count == 0
    assert session.previous_verdicts == []


# ── exploration_threshold_met ────────────────────────────────────


def test_exploration_empty_conversation_false():
    assert exploration_threshold_met([], threshold=3) is False


def _tool_use_msg(tool_name: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "x", "name": tool_name, "input": {}},
        ],
    }


def test_exploration_threshold_met_after_n_reads():
    conv = [
        _tool_use_msg("Read"),
        _tool_use_msg("Grep"),
        _tool_use_msg("Read"),
    ]
    assert exploration_threshold_met(conv, threshold=3) is True


def test_exploration_threshold_not_met_with_fewer_reads():
    conv = [_tool_use_msg("Read"), _tool_use_msg("Grep")]
    assert exploration_threshold_met(conv, threshold=3) is False


def test_exploration_false_once_agent_has_written():
    conv = [
        _tool_use_msg("Read"),
        _tool_use_msg("Grep"),
        _tool_use_msg("Read"),
        _tool_use_msg("Edit"),  # agent has already started acting
    ]
    assert exploration_threshold_met(conv, threshold=3) is False


def test_exploration_still_true_after_bash():
    """Bash counts as exploration (running tests, ls, which), not a write."""
    conv = [
        _tool_use_msg("Read"),
        _tool_use_msg("Grep"),
        _tool_use_msg("Read"),
        _tool_use_msg("Bash"),
    ]
    assert exploration_threshold_met(conv, threshold=3) is True


def test_exploration_false_after_write():
    conv = [
        _tool_use_msg("Read"),
        _tool_use_msg("Grep"),
        _tool_use_msg("Read"),
        _tool_use_msg("Write"),
    ]
    assert exploration_threshold_met(conv, threshold=3) is False


# ── before_first_edit_condition (behavioral, no keywords) ─────────


def test_before_first_edit_fires_on_empty_conversation():
    """Clean session — no edits yet — trigger fires regardless of task text."""
    from maike.agents.advisor import before_first_edit_condition

    assert before_first_edit_condition([], None) is True


def test_before_first_edit_fires_short_task():
    """Task wording is irrelevant — short 'fix it' tasks fire too."""
    from maike.agents.advisor import before_first_edit_condition

    assert before_first_edit_condition([], None) is True


def test_before_first_edit_fires_non_planning_wording():
    """'Please debug this' doesn't contain add/create/implement but still fires."""
    from maike.agents.advisor import before_first_edit_condition

    assert before_first_edit_condition([], None) is True


def test_before_first_edit_skips_after_edit():
    """Once Edit has been emitted, the trigger should not fire."""
    from maike.agents.advisor import before_first_edit_condition

    conv = [_tool_use_msg("Edit")]
    assert before_first_edit_condition(conv, None) is False


def test_before_first_edit_skips_after_write():
    from maike.agents.advisor import before_first_edit_condition

    conv = [_tool_use_msg("Write")]
    assert before_first_edit_condition(conv, None) is False


def test_before_first_edit_allows_reads_bash_grep():
    """Reads, Bash, and Grep don't prevent the trigger; only Edit/Write do."""
    from maike.agents.advisor import before_first_edit_condition

    conv = [_tool_use_msg("Read"), _tool_use_msg("Bash"), _tool_use_msg("Grep")]
    assert before_first_edit_condition(conv, None) is True


def test_before_first_write_alias_exists():
    """The old name is preserved as an alias for backwards compatibility."""
    from maike.agents.advisor import (
        before_first_edit_condition,
        before_first_write_condition,
    )

    assert before_first_write_condition is before_first_edit_condition


# ── before_completion_condition ────────────────────────────────────


def test_before_completion_skips_when_last_llm_had_tools():
    """If the LLM just called a tool, the turn isn't ending — no advisor."""
    from maike.agents.advisor import before_completion_condition

    should, reason = before_completion_condition([], last_llm_had_tool_calls=True)
    assert should is False
    assert reason == ""


def test_before_completion_fires_zero_edits():
    """Text-only termination with no edits in session → 'zero_edits'."""
    from maike.agents.advisor import before_completion_condition

    conv = [_tool_use_msg("Read"), _tool_use_msg("Bash")]
    should, reason = before_completion_condition(conv, last_llm_had_tool_calls=False)
    assert should is True
    assert reason == "zero_edits"


def test_before_completion_fires_unverified_edit():
    """Edit was the last tool — no Bash verification after it."""
    from maike.agents.advisor import before_completion_condition

    conv = [_tool_use_msg("Read"), _tool_use_msg("Edit")]
    should, reason = before_completion_condition(conv, last_llm_had_tool_calls=False)
    assert should is True
    assert reason == "unverified_edit"


def test_before_completion_silent_when_edit_then_bash():
    """Edit followed by Bash — verification happened, agent is done cleanly."""
    from maike.agents.advisor import before_completion_condition

    conv = [_tool_use_msg("Read"), _tool_use_msg("Edit"), _tool_use_msg("Bash")]
    should, reason = before_completion_condition(conv, last_llm_had_tool_calls=False)
    assert should is False


# ── failures_seen monotonic counter ────────────────────────────────


def test_advisor_session_failures_seen_starts_at_zero():
    from maike.agents.advisor import AdvisorConfig, AdvisorSession

    session = AdvisorSession(gateway=object(), config=AdvisorConfig(enabled=True))
    assert session.failures_seen == 0


def test_record_failure_increments_monotonically():
    from maike.agents.advisor import AdvisorConfig, AdvisorSession

    session = AdvisorSession(gateway=object(), config=AdvisorConfig(enabled=True))
    session.record_failure()
    session.record_failure()
    session.record_failure()
    assert session.failures_seen == 3


def test_record_failure_never_resets():
    """Unlike RepeatedFailureTracker._failure_hashes, this counter never
    gets cleared — even after many calls and advisor verdicts."""
    from maike.agents.advisor import (
        AdvisorConfig,
        AdvisorSession,
        AdvisorTrigger,
        AdvisorUrgency,
        AdvisorVerdict,
    )

    session = AdvisorSession(gateway=object(), config=AdvisorConfig(enabled=True))
    for _ in range(5):
        session.record_failure()
    # Recording a verdict must NOT reset the failure counter.
    verdict = AdvisorVerdict(
        advice="test",
        urgency=AdvisorUrgency.STUCK,
        trigger=AdvisorTrigger.ON_STUCK,
        cost_usd=0.01,
        tokens_used=50,
    )
    session.record_verdict(verdict, iteration_count=3)
    assert session.failures_seen == 5


# ── Advisor prompt loading ───────────────────────────────────────


def test_load_advisor_prompt_returns_nonempty():
    prompt = load_advisor_prompt()
    assert prompt  # non-empty string
    assert isinstance(prompt, str)


# ── Test helpers ─────────────────────────────────────────────────


class _DummyCtx:
    """Minimal AgentContext-like stand-in for scaffold tests."""

    task = "test task"
    max_iterations = 50
    role = "react"
