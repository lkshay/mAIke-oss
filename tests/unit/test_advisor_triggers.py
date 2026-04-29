"""Unit tests for Phase 4: AgentCore auto-trigger detection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from maike.agents.advisor import (
    AdvisorConfig,
    AdvisorSession,
    AdvisorTrigger,
    AdvisorUrgency,
    AdvisorVerdict,
)


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _Ctx:
    task: str = "test task"
    max_iterations: int = 50
    role: str = "react"
    agent_id: str = "agent-x"


class _RecordingSession(AdvisorSession):
    """Test double: records advise() calls and returns scripted verdicts."""

    def __init__(self, scripted_verdicts: list[AdvisorVerdict]):
        cfg = AdvisorConfig(enabled=True, model="m", budget_usd=10.0, cooldown_iterations=0)
        super().__init__(gateway=object(), config=cfg)
        self._scripted = list(scripted_verdicts)
        self.calls: list[dict[str, Any]] = []

    async def advise(self, question, urgency, trigger, conversation, ctx, iteration_count):
        self.calls.append({
            "question": question,
            "urgency": urgency,
            "trigger": trigger,
            "iteration_count": iteration_count,
            "conversation_len": len(conversation),
        })
        if not self._scripted:
            return AdvisorVerdict.throttled_verdict(urgency, trigger, "no_more_scripts")
        return self._scripted.pop(0)


class _FailureTracker:
    def __init__(self, hashes: list[str] | None = None):
        self._failure_hashes = hashes or []


class _StubAgentCore:
    """Bind only the methods we need to exercise the trigger logic."""

    from maike.agents.core import AgentCore

    def __init__(self, advisor_session, tracer=None):
        from maike.observability.tracer import Tracer
        self._advisor_session = advisor_session
        self.tracer = tracer or Tracer()

    # Pull the unbound method from AgentCore so we test the real impl.
    _maybe_fire_advisor = AgentCore._maybe_fire_advisor


def _read_msg(name: str = "Read", path: str | None = None) -> dict:
    """A single Read/Grep/etc. tool_use. Distinct paths avoid triggering
    detect_spinning_v2 in tests that want to exercise after_exploration."""
    if path is None:
        # Use object id to vary per call when no path supplied.
        path = f"f_{id(name) % 1000}_{name}.py"
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "x", "name": name, "input": {"file_path": path}},
        ],
    }


def _edit_msg() -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "y", "name": "Edit", "input": {}},
        ],
    }


def _good_verdict(trigger: AdvisorTrigger, advice: str = "do X") -> AdvisorVerdict:
    return AdvisorVerdict(
        advice=advice,
        urgency=AdvisorUrgency.NORMAL if trigger == AdvisorTrigger.AFTER_EXPLORATION else AdvisorUrgency.STUCK,
        trigger=trigger,
        cost_usd=0.02,
        tokens_used=80,
    )


# ── after_exploration trigger ────────────────────────────────────────


def test_after_exploration_fires_after_n_reads():
    session = _RecordingSession([_good_verdict(AdvisorTrigger.AFTER_EXPLORATION)])
    core = _StubAgentCore(session)
    # Distinct paths so detect_spinning_v2 doesn't classify this as stuck.
    conversation = [
        _read_msg("Read", path="a.py"),
        _read_msg("Grep", path="b.py"),
        _read_msg("Read", path="c.py"),
    ]

    _run(core._maybe_fire_advisor(
        conversation=conversation,
        ctx=_Ctx(),
        iteration_count=4,
        failure_tracker=_FailureTracker(),
    ))
    assert len(session.calls) == 1
    assert session.calls[0]["trigger"] == AdvisorTrigger.AFTER_EXPLORATION
    # Verdict was injected as a maike-advisor block.
    last = conversation[-1]
    assert last["role"] == "user"
    assert "<maike-advisor" in last["content"]
    assert "do X" in last["content"]


def test_after_exploration_fires_only_once():
    session = _RecordingSession([
        _good_verdict(AdvisorTrigger.AFTER_EXPLORATION, "advice 1"),
        _good_verdict(AdvisorTrigger.AFTER_EXPLORATION, "advice 2"),
    ])
    core = _StubAgentCore(session)
    conversation = [
        _read_msg("Read", path="a.py"),
        _read_msg("Grep", path="b.py"),
        _read_msg("Read", path="c.py"),
    ]

    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=4,
        failure_tracker=_FailureTracker(),
    ))
    # Mark after_exploration as fired — the real AdvisorSession does this via
    # record_verdict; our RecordingSession stub doesn't.
    session.triggers_fired.add("after_exploration")

    # Add another read — after_exploration should NOT re-fire (already in
    # triggers_fired). before_first_edit MAY still fire the first time
    # through, but we're asserting here about after_exploration specifically.
    conversation.append(_read_msg("Read", path="d.py"))
    session.triggers_fired.add("before_first_edit")  # suppress that too
    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=5,
        failure_tracker=_FailureTracker(),
    ))
    # The RecordingSession sees both triggers reusing the AFTER_EXPLORATION
    # enum. What we care about: after_exploration fired exactly once in the
    # first call, and did NOT fire again in the second call.
    assert len(session.calls) == 1, "after_exploration should fire exactly once"


def test_after_exploration_does_not_fire_after_writes():
    session = _RecordingSession([_good_verdict(AdvisorTrigger.AFTER_EXPLORATION)])
    core = _StubAgentCore(session)
    # 3 reads, but agent has already started editing — too late for plan check.
    conversation = [_read_msg(), _read_msg(), _read_msg(), _edit_msg()]

    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=5,
        failure_tracker=_FailureTracker(),
    ))
    after_explore_calls = [c for c in session.calls if c["trigger"] == AdvisorTrigger.AFTER_EXPLORATION]
    assert after_explore_calls == []


# ── on_stuck trigger ─────────────────────────────────────────────────


def test_on_stuck_fires_when_failures_seen_reaches_two():
    """The on_stuck check reads AdvisorSession.failures_seen (monotonic
    counter), not failure_tracker._failure_hashes. The counter survives
    the tracker's own reset-after-nudge."""
    session = _RecordingSession([_good_verdict(AdvisorTrigger.ON_STUCK, "try Y instead")])
    # Simulate 2 observed failures (how the real loop increments it).
    session.record_failure()
    session.record_failure()

    core = _StubAgentCore(session)
    conversation = [_read_msg()]
    # failure_tracker._failure_hashes is EMPTY — this is the blind spot we
    # fixed. Without the monotonic counter, this test scenario would miss.
    failure_tracker = _FailureTracker(hashes=[])

    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=10,
        failure_tracker=failure_tracker,
    ))
    assert len(session.calls) == 1
    assert session.calls[0]["trigger"] == AdvisorTrigger.ON_STUCK
    assert "try Y instead" in conversation[-1]["content"]


def test_on_stuck_takes_precedence_over_after_exploration():
    """Both conditions met simultaneously — on_stuck wins."""
    session = _RecordingSession([_good_verdict(AdvisorTrigger.ON_STUCK)])
    session.record_failure()
    session.record_failure()
    core = _StubAgentCore(session)
    conversation = [_read_msg(), _read_msg(), _read_msg()]

    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=8,
        failure_tracker=_FailureTracker(),
    ))
    assert session.calls[0]["trigger"] == AdvisorTrigger.ON_STUCK


# ── delegate exemption ──────────────────────────────────────────────


def test_delegate_role_skips_advisor():
    """Delegates never invoke advisor — caller-side guard in run()."""
    # The guard lives in the AgentCore.run() loop, not in _maybe_fire_advisor.
    # So _maybe_fire_advisor itself does fire when called — the guard is the
    # "if ctx.role != 'delegate'" check in the caller.  Verify the caller
    # contract by reading it directly.
    import inspect
    from maike.agents.core import AgentCore
    src = inspect.getsource(AgentCore.run)
    assert "ctx.role != \"delegate\"" in src
    assert "_maybe_fire_advisor" in src


# ── disabled / throttled ────────────────────────────────────────────


def test_no_action_when_session_disabled():
    cfg = AdvisorConfig(enabled=False)
    session = AdvisorSession(gateway=None, config=cfg)
    session.record_failure()
    session.record_failure()  # even with failures, disabled session stays silent
    core = _StubAgentCore(session)
    conversation = [_read_msg(), _read_msg(), _read_msg()]

    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=5,
        failure_tracker=_FailureTracker(),
    ))
    # No advisor block injected.
    assert all("<maike-advisor" not in (m.get("content") if isinstance(m.get("content"), str) else "")
               for m in conversation)


def test_throttled_verdict_not_injected():
    """If the session returns a throttled verdict, nothing is appended."""
    throttled = AdvisorVerdict.throttled_verdict(
        AdvisorUrgency.STUCK, AdvisorTrigger.ON_STUCK, "budget_exhausted",
    )
    session = _RecordingSession([throttled])
    session.record_failure()
    session.record_failure()
    core = _StubAgentCore(session)
    conversation = [_read_msg()]

    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=5,
        failure_tracker=_FailureTracker(),
    ))
    # Conversation unchanged in length; no maike-advisor block.
    assert len(conversation) == 1
    assert "<maike-advisor" not in str(conversation[-1].get("content"))
    # And session's call_count is NOT bumped (record_verdict skips throttled).
    assert session.call_count == 0


# ── nudge tag attributes ────────────────────────────────────────────


def test_advisor_nudge_uses_high_priority_tag():
    session = _RecordingSession([_good_verdict(AdvisorTrigger.AFTER_EXPLORATION)])
    core = _StubAgentCore(session)
    conversation = [
        _read_msg("Read", path="a.py"),
        _read_msg("Grep", path="b.py"),
        _read_msg("Read", path="c.py"),
    ]

    _run(core._maybe_fire_advisor(
        conversation=conversation, ctx=_Ctx(), iteration_count=4,
        failure_tracker=_FailureTracker(),
    ))
    last = conversation[-1]
    content = last["content"]
    assert 'priority="high"' in content
    assert 'source="after_exploration"' in content
