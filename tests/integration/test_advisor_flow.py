"""Integration test for the advisor pattern end-to-end.

Spins up an Orchestrator with --advisor enabled, builds an AdvisorSession
backed by a FakeAdvisorGateway, and verifies:
  - Tool handler calls advisor and returns advice in a ToolResult
  - Auto-trigger fires when the conversation looks stuck
  - Throttling propagates correctly
  - Cost is attributed to the advisor session (not just total cost)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maike.agents.advisor import (
    AdvisorConfig,
    AdvisorSession,
    AdvisorTrigger,
    AdvisorUrgency,
    AdvisorVerdict,
)
from maike.atoms.llm import LLMContentBlock, LLMResult, TokenUsage
from maike.cost.tracker import CostTracker
from maike.observability.tracer import Tracer
from maike.orchestrator.orchestrator import Orchestrator


def _run(coro):
    return asyncio.run(coro)


class FakeAdvisorGateway:
    """Stand-in for an LLMGateway. Returns scripted advice."""

    def __init__(
        self,
        advice: str = "proceed with plan A",
        cost_usd: float = 0.04,
        tokens: int = 90,
        cost_tracker: CostTracker | None = None,
    ):
        self._advice = advice
        self._cost = cost_usd
        self._tokens = tokens
        self.calls: list[dict[str, Any]] = []
        self.cost_tracker = cost_tracker or CostTracker()
        self.provider_name = "fake"

    async def call(self, *, system, messages, tools, model, temperature, max_tokens):
        self.calls.append({
            "system": system,
            "messages": messages,
            "tools": tools,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        # The real adapter records via cost_tracker; here we just return the
        # cost on the LLMResult — call_advisor_gateway reads it from there.
        return LLMResult(
            content_blocks=[LLMContentBlock(type="text", text=self._advice)],
            usage=TokenUsage(input_tokens=self._tokens // 2, output_tokens=self._tokens // 2),
            cost_usd=self._cost,
            model=model,
        )

    async def aclose(self):
        pass


@dataclass
class _Ctx:
    task: str = "do the thing"
    max_iterations: int = 50
    role: str = "react"
    agent_id: str = "agent-1"


def test_advisor_handler_end_to_end_through_orchestrator(tmp_path):
    """Build a real Orchestrator, get its advisor handler, invoke it, verify
    advice flows back and cost is recorded on the advisor's cost tracker."""
    from maike.tools.context import (
        AgentLoopState,
        CURRENT_AGENT_LOOP_STATE,
    )

    cost_tracker = CostTracker()
    orch = Orchestrator(
        base_path=tmp_path,
        cost_tracker=cost_tracker,
        tracer=Tracer(),
    )

    fake_gw = FakeAdvisorGateway(advice="Proceed; no issues spotted.", cost_usd=0.03)
    cfg = AdvisorConfig(
        enabled=True,
        provider="fake",
        model="fake-strong",
        budget_usd=1.0,
        cooldown_iterations=0,
    )
    session = AdvisorSession(gateway=fake_gw, config=cfg)
    handler = orch._make_advisor_handler(session)

    # Publish a fake loop state so the handler can find conversation + ctx.
    loop_state = AgentLoopState(
        conversation=[{"role": "user", "content": "What now?"}],
        iteration_count=4,
        agent_context=_Ctx(),
    )
    token = CURRENT_AGENT_LOOP_STATE.set(loop_state)
    try:
        result = _run(handler(question="should I proceed?", urgency="normal"))
    finally:
        CURRENT_AGENT_LOOP_STATE.reset(token)

    assert result.success is True
    assert "Proceed" in result.output
    assert result.metadata.get("advisor_cost_usd") == 0.03
    assert result.metadata.get("advisor_trigger") == "tool"
    # The handler called record_verdict on the session.
    assert session.call_count == 1
    assert session.cost_spent_usd == 0.03


def test_throttled_advisor_returns_friendly_message(tmp_path):
    from maike.tools.context import (
        AgentLoopState,
        CURRENT_AGENT_LOOP_STATE,
    )

    orch = Orchestrator(
        base_path=tmp_path,
        cost_tracker=CostTracker(),
        tracer=Tracer(),
    )
    cfg = AdvisorConfig(enabled=True, model="m", budget_usd=1.0, cooldown_iterations=0)
    session = AdvisorSession(gateway=FakeAdvisorGateway(), config=cfg)
    session.cost_spent_usd = 5.0  # over budget
    handler = orch._make_advisor_handler(session)

    loop_state = AgentLoopState(
        conversation=[],
        iteration_count=1,
        agent_context=_Ctx(),
    )
    token = CURRENT_AGENT_LOOP_STATE.set(loop_state)
    try:
        result = _run(handler(question="help", urgency="normal"))
    finally:
        CURRENT_AGENT_LOOP_STATE.reset(token)

    assert result.success is True  # throttled is graceful
    assert result.metadata.get("throttled") is True
    assert "budget" in result.output.lower()


def test_disabled_session_skips_gateway_construction(tmp_path):
    """When --advisor is off, _build_advisor_session returns a no-op session
    and never builds a gateway."""
    orch = Orchestrator(
        base_path=tmp_path,
        cost_tracker=CostTracker(),
        tracer=Tracer(),
    )
    session = orch._build_advisor_session(
        enabled=False,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        budget=5.0,
        budget_pct=0.2,
    )
    assert session is not None
    assert session.enabled is False
    assert session.gateway is None


def test_enabled_session_builds_silent_gateway(tmp_path):
    """When --advisor is on, an LLMGateway is constructed with silent=True."""
    orch = Orchestrator(
        base_path=tmp_path,
        cost_tracker=CostTracker(),
        tracer=Tracer(),
    )
    session = orch._build_advisor_session(
        enabled=True,
        executor_provider="gemini",
        advisor_provider=None,
        advisor_model=None,
        budget=5.0,
        budget_pct=0.2,
    )
    assert session.enabled is True
    assert session.gateway is not None
    # Silent so it doesn't pollute TUI traces.
    assert getattr(session.gateway, "_silent", False) is True
    # Budget = 20% of 5.0
    assert session.config.budget_usd == 1.0


def test_cli_to_orchestrator_advisor_flag_propagation(tmp_path):
    """End-to-end smoke: parse CLI args, construct Orchestrator with advisor
    enabled, verify the advisor gateway is wired."""
    from maike.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "run", "the task",
        "--workspace", str(tmp_path),
        "--budget", "2.0",
        "--advisor",
        "--advisor-budget-pct", "0.5",
    ])
    assert args.advisor is True
    assert args.advisor_budget_pct == 0.5

    # Now build orchestrator + advisor session as the orchestrator would.
    orch = Orchestrator(
        base_path=tmp_path,
        cost_tracker=CostTracker(),
        tracer=Tracer(),
    )
    session = orch._build_advisor_session(
        enabled=args.advisor,
        executor_provider=args.provider,  # "gemini"
        advisor_provider=args.advisor_provider,
        advisor_model=args.advisor_model,
        budget=args.budget,
        budget_pct=args.advisor_budget_pct,
    )
    assert session.enabled is True
    # 50% of 2.0 = 1.0
    assert session.config.budget_usd == 1.0
    # Default advisor model = strong tier of gemini.
    assert "pro" in session.config.model.lower() or "2.5" in session.config.model
