"""Tests for TrajectoryAuditor — markdown-or-empty gating (no JSON)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from maike.agents.stuck_detectors import CandidateNudge
from maike.agents.trajectory_auditor import (
    AuditorConfig,
    TrajectoryAuditor,
)


# ---------------------------------------------------------------------------
# Fake gateway scaffolding
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    content: str


class _FakeGateway:
    def __init__(self, responses: list[str] | str | None = None):
        if isinstance(responses, str):
            responses = [responses]
        self.responses: list[str] = list(responses or ["Stop reading, try an Edit on widgets.py."])
        self.calls: list[dict] = []
        self.cost_tracker = _FakeCostTracker()

    async def call(self, *, system, messages, model, temperature, max_tokens, tools):
        self.calls.append({"system": system, "messages": messages, "model": model})
        r = self.responses.pop(0) if self.responses else self.responses[-1] if hasattr(self.responses, "__getitem__") else ""
        self.cost_tracker._session_total += 0.001
        return _FakeResult(content=r)


@dataclass
class _FakeCostTracker:
    _session_total: float = 0.0

    @property
    def session_total(self) -> float:
        return self._session_total


@pytest.fixture(autouse=True)
def _clear_ollama_cache():
    from maike.agents import constraints as cmod
    cmod._OLLAMA_PROBE_CACHE.clear()
    yield
    cmod._OLLAMA_PROBE_CACHE.clear()


def _returns_gateway(gw):
    async def _inner(bg, provider):
        return gw, provider, "fake-cheap-model"
    return _inner


def _run(coro):
    return asyncio.run(coro)


def _candidate(kind: str = "reads_without_edit", text: str = "Stop reading, edit now") -> CandidateNudge:
    return CandidateNudge(kind=kind, text=text, evidence={"count": 20})


# ---------------------------------------------------------------------------
# Approve (non-empty text) / veto (empty) behavior
# ---------------------------------------------------------------------------


class TestAuditApproveVeto:
    def _make(self, gw=None, enabled=True, budget=1.0, max_calls=6, cooldown=0):
        return TrajectoryAuditor(
            gateway=gw or _FakeGateway(),
            provider="gemini",
            config=AuditorConfig(
                enabled=enabled, budget_usd=budget,
                max_calls=max_calls, cooldown_iterations=cooldown,
            ),
        )

    def test_non_empty_response_is_approve(self):
        gw = _FakeGateway(responses=["Focus on widgets.py — you've read it 20 times, try editing the merge function."])
        auditor = self._make(gw=gw)
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v = _run(auditor.audit(
                candidate=_candidate(text="original"),
                task="Fix the merge bug.", iteration=1,
            ))
        assert v.decision == "approve"
        assert "widgets.py" in v.text
        assert v.source == "llm"

    def test_empty_response_is_veto(self):
        gw = _FakeGateway(responses=[""])
        auditor = self._make(gw=gw)
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v = _run(auditor.audit(
                candidate=_candidate(text="edit now"),
                task="Summarize this codebase.", iteration=1,
            ))
        assert v.decision == "veto"
        assert v.text is None
        assert v.source == "llm"

    def test_whitespace_only_is_veto(self):
        gw = _FakeGateway(responses=["   \n  "])
        auditor = self._make(gw=gw)
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v = _run(auditor.audit(
                candidate=_candidate(), task="t", iteration=1,
            ))
        assert v.decision == "veto"
        assert v.text is None

    def test_fenced_markdown_stripped(self):
        gw = _FakeGateway(responses=["```\nStop reading widgets.py, edit it.\n```"])
        auditor = self._make(gw=gw)
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v = _run(auditor.audit(
                candidate=_candidate(), task="t", iteration=1,
            ))
        assert v.decision == "approve"
        assert "```" not in v.text
        assert "Stop reading" in v.text

    def test_long_response_truncated(self):
        long_text = "Long nudge.  " * 200
        gw = _FakeGateway(responses=[long_text])
        auditor = self._make(gw=gw)
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v = _run(auditor.audit(
                candidate=_candidate(), task="t", iteration=1,
            ))
        assert v.decision == "approve"
        assert len(v.text) <= 1200

    def test_gateway_exception_fail_safe_approves(self):
        class _BadGw(_FakeGateway):
            async def call(self, **kwargs):
                raise RuntimeError("boom")
        gw = _BadGw()
        auditor = self._make(gw=gw)
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v = _run(auditor.audit(
                candidate=_candidate(text="original"), task="t", iteration=1,
            ))
        assert v.decision == "approve"
        assert v.text == "original"
        assert v.source == "fail_safe"

    def test_timeout_fail_safe_approves(self):
        class _SlowGw(_FakeGateway):
            async def call(self, **kwargs):
                await asyncio.sleep(5)
                return _FakeResult("ignored")
        gw = _SlowGw()
        auditor = TrajectoryAuditor(
            gateway=gw, provider="gemini",
            config=AuditorConfig(enabled=True, budget_usd=1.0,
                                 max_calls=6, cooldown_iterations=0,
                                 timeout_s=0.1),
        )
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v = _run(auditor.audit(
                candidate=_candidate(text="original"), task="t", iteration=1,
            ))
        assert v.decision == "approve"
        assert v.source == "fail_safe"
        assert "timeout" in v.reason.lower()


# ---------------------------------------------------------------------------
# Throttle paths (unchanged semantics)
# ---------------------------------------------------------------------------


class TestAuditThrottle:
    def test_disabled_config_throttles(self):
        gw = _FakeGateway()
        auditor = TrajectoryAuditor(
            gateway=gw, provider="gemini",
            config=AuditorConfig(enabled=False, budget_usd=1.0, max_calls=6),
        )
        v = _run(auditor.audit(candidate=_candidate(), task="t", iteration=1))
        assert v.throttled is True
        assert v.decision == "approve"
        assert gw.calls == []

    def test_max_calls_throttles(self):
        gw = _FakeGateway()
        auditor = TrajectoryAuditor(
            gateway=gw, provider="gemini",
            config=AuditorConfig(enabled=True, budget_usd=10.0, max_calls=2,
                                 cooldown_iterations=0),
        )
        auditor.calls_made = 2
        v = _run(auditor.audit(candidate=_candidate(), task="t", iteration=10))
        assert v.throttled is True
        assert gw.calls == []

    def test_budget_exhausted_throttles(self):
        gw = _FakeGateway()
        auditor = TrajectoryAuditor(
            gateway=gw, provider="gemini",
            config=AuditorConfig(enabled=True, budget_usd=0.001, max_calls=100,
                                 cooldown_iterations=0),
        )
        auditor.cost_used_usd = 0.002
        v = _run(auditor.audit(candidate=_candidate(), task="t", iteration=1))
        assert v.throttled is True

    def test_cooldown_throttles(self):
        gw = _FakeGateway()
        auditor = TrajectoryAuditor(
            gateway=gw, provider="gemini",
            config=AuditorConfig(enabled=True, budget_usd=1.0, max_calls=10,
                                 cooldown_iterations=5),
        )
        auditor.last_call_iteration = 10
        v = _run(auditor.audit(candidate=_candidate(), task="t", iteration=12))
        assert v.throttled is True


# ---------------------------------------------------------------------------
# Cache (unchanged semantics — same (kind, task_hash) is memoized)
# ---------------------------------------------------------------------------


class TestAuditCache:
    def test_same_kind_and_task_reuses_verdict(self):
        gw = _FakeGateway(responses=[""])  # veto on first call
        auditor = TrajectoryAuditor(
            gateway=gw, provider="gemini",
            config=AuditorConfig(enabled=True, budget_usd=1.0,
                                 max_calls=10, cooldown_iterations=0),
        )
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            v1 = _run(auditor.audit(
                candidate=_candidate(), task="summarize codebase", iteration=1,
            ))
            v2 = _run(auditor.audit(
                candidate=_candidate(), task="summarize codebase", iteration=2,
            ))
        assert v1.decision == "veto"
        assert v2.decision == "veto"
        assert v2.source == "cache"
        assert len(gw.calls) == 1

    def test_different_task_causes_fresh_call(self):
        gw = _FakeGateway(responses=["Approve text 1.", ""])
        auditor = TrajectoryAuditor(
            gateway=gw, provider="gemini",
            config=AuditorConfig(enabled=True, budget_usd=1.0,
                                 max_calls=10, cooldown_iterations=0),
        )
        with patch("maike.agents.constraints._select_extractor_gateway",
                   new=_returns_gateway(gw)):
            _run(auditor.audit(candidate=_candidate(), task="Fix bug", iteration=1))
            v2 = _run(auditor.audit(candidate=_candidate(), task="Different task", iteration=2))
        assert v2.decision == "veto"
        assert len(gw.calls) == 2
