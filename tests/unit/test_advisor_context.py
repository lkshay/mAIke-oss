"""Unit tests for Phase 3: transcript compression + advisor gateway call."""

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
    build_advisor_context,
    call_advisor_gateway,
    load_advisor_prompt,
)


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _Ctx:
    task: str = "test task"
    max_iterations: int = 50
    role: str = "react"


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _tool_use_msg(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": name, "input": args},
        ],
    }


def _tool_result_msg(text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": text},
        ],
    }


# ── build_advisor_context ──────────────────────────────────────────


def test_context_includes_task_framing():
    ctx_text = build_advisor_context(
        question="Should I refactor X?",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=[],
        ctx=_Ctx(),
        previous_verdicts=[],
        iteration_count=5,
    )
    assert "test task" in ctx_text
    assert "Iteration: 5" in ctx_text
    assert "Urgency: normal" in ctx_text
    assert "Trigger: tool" in ctx_text


def test_context_includes_question_at_end():
    ctx_text = build_advisor_context(
        question="The specific question",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=[],
        ctx=_Ctx(),
        previous_verdicts=[],
        iteration_count=1,
    )
    # The question section is the last in the rendered text — important so
    # the LLM doesn't lose it under truncation.
    assert ctx_text.rstrip().endswith("The specific question")


def test_context_includes_recent_conversation_tail():
    conv = [
        _user("first user message"),
        _tool_use_msg("Read", {"file_path": "x.py"}),
        _tool_result_msg("file contents here"),
    ]
    ctx_text = build_advisor_context(
        question="Q?",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=conv,
        ctx=_Ctx(),
        previous_verdicts=[],
        iteration_count=2,
    )
    assert "Recent conversation" in ctx_text
    assert "Read" in ctx_text or "file_path" in ctx_text
    assert "file contents here" in ctx_text


def test_context_includes_previous_verdicts():
    prev = [
        AdvisorVerdict(
            advice="Try X first",
            urgency=AdvisorUrgency.STUCK,
            trigger=AdvisorTrigger.ON_STUCK,
            cost_usd=0.01,
            tokens_used=50,
        ),
        AdvisorVerdict(
            advice="Now try Y",
            urgency=AdvisorUrgency.STUCK,
            trigger=AdvisorTrigger.ON_STUCK,
            cost_usd=0.01,
            tokens_used=50,
        ),
    ]
    ctx_text = build_advisor_context(
        question="Q?",
        urgency=AdvisorUrgency.STUCK,
        trigger=AdvisorTrigger.ON_STUCK,
        conversation=[],
        ctx=_Ctx(),
        previous_verdicts=prev,
        iteration_count=10,
    )
    assert "Previous advice" in ctx_text
    assert "Try X first" in ctx_text
    assert "Now try Y" in ctx_text


def test_context_omits_previous_advice_section_when_empty():
    ctx_text = build_advisor_context(
        question="Q?",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=[],
        ctx=_Ctx(),
        previous_verdicts=[],
        iteration_count=1,
    )
    assert "Previous advice" not in ctx_text


def test_context_respects_char_cap():
    """A pathologically long conversation must be truncated to max_chars."""
    huge_text = "x" * 100_000
    conv = [_user(huge_text)]
    ctx_text = build_advisor_context(
        question="Q?",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=conv,
        ctx=_Ctx(),
        previous_verdicts=[],
        iteration_count=1,
        max_chars=2_000,
        recent_tail_chars=1_000,
    )
    # Some slack for framing overhead (the cap applies to the total).
    assert len(ctx_text) <= 2_500
    # Question survives truncation.
    assert "Q?" in ctx_text


def test_context_truncation_keeps_question_intact():
    """Even with a tiny budget, the question must NOT be cut off."""
    huge_text = "y" * 50_000
    conv = [_user(huge_text)]
    ctx_text = build_advisor_context(
        question="Critical question that must survive",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=conv,
        ctx=_Ctx(),
        previous_verdicts=[],
        iteration_count=1,
        max_chars=1_000,
        recent_tail_chars=500,
    )
    assert "Critical question that must survive" in ctx_text


# ── load_advisor_prompt ────────────────────────────────────────────


def test_load_advisor_prompt_includes_core_and_guidance():
    prompt = load_advisor_prompt()
    # Core instructions are wrapped at high priority.
    assert "strategic advisor" in prompt.lower()
    # Guidance is wrapped at low priority — should be present.
    assert "transcript" in prompt.lower()


# ── call_advisor_gateway ───────────────────────────────────────────


class _FakeGateway:
    """Captures call args and returns a controlled LLMResult."""

    def __init__(self, advice_text: str, cost_usd: float = 0.05, tokens: int = 120):
        self._advice_text = advice_text
        self._cost_usd = cost_usd
        self._tokens = tokens
        self.calls: list[dict[str, Any]] = []

    async def call(self, *, system, messages, tools, model, temperature, max_tokens):
        self.calls.append({
            "system": system,
            "messages": messages,
            "tools": tools,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        # Build a result that walks like LLMResult enough for the extractor.
        from maike.atoms.llm import LLMContentBlock, LLMResult, TokenUsage
        return LLMResult(
            content_blocks=[LLMContentBlock(type="text", text=self._advice_text)],
            usage=TokenUsage(input_tokens=self._tokens // 2, output_tokens=self._tokens // 2),
            cost_usd=self._cost_usd,
            model=model,
        )


def test_gateway_call_returns_advice_tokens_cost():
    cfg = AdvisorConfig(enabled=True, model="strong-model")
    gw = _FakeGateway(advice_text="Proceed with plan A; it covers the edge case.")
    session = AdvisorSession(gateway=gw, config=cfg)

    advice, tokens, cost = _run(call_advisor_gateway(
        session,
        system_prompt="sys",
        user_context="user",
    ))
    assert advice == "Proceed with plan A; it covers the edge case."
    assert tokens == 120
    assert cost == pytest.approx(0.05)


def test_gateway_call_passes_through_advisor_config():
    cfg = AdvisorConfig(enabled=True, model="claude-opus-4-6")
    gw = _FakeGateway(advice_text="ok")
    session = AdvisorSession(gateway=gw, config=cfg)

    _run(call_advisor_gateway(session, system_prompt="SYS", user_context="USER"))
    call = gw.calls[0]
    assert call["model"] == "claude-opus-4-6"
    assert call["tools"] == []  # advisor must never get tools
    from maike.constants import ADVISOR_MAX_OUTPUT_TOKENS
    assert call["max_tokens"] == ADVISOR_MAX_OUTPUT_TOKENS
    assert call["system"] == "SYS"
    assert call["messages"] == [{"role": "user", "content": "USER"}]


def test_gateway_call_falls_back_to_flat_content_field():
    """Adapters that return content as a flat string instead of blocks."""
    cfg = AdvisorConfig(enabled=True, model="m")

    class _FlatGateway(_FakeGateway):
        async def call(self, *, system, messages, tools, model, temperature, max_tokens):
            self.calls.append(locals())
            from maike.atoms.llm import LLMResult, TokenUsage
            return LLMResult(
                content="flat advice text",
                content_blocks=[],  # empty blocks
                usage=TokenUsage(input_tokens=20, output_tokens=20),
                cost_usd=0.01,
                model=model,
            )

    session = AdvisorSession(gateway=_FlatGateway("ignored"), config=cfg)
    advice, tokens, cost = _run(call_advisor_gateway(
        session, system_prompt="sys", user_context="user",
    ))
    assert advice == "flat advice text"
    assert tokens == 40


def test_gateway_call_raises_when_session_has_no_gateway():
    cfg = AdvisorConfig(enabled=False)
    session = AdvisorSession(gateway=None, config=cfg)
    with pytest.raises(RuntimeError):
        _run(call_advisor_gateway(session, system_prompt="x", user_context="y"))


# ── AdvisorSession.advise() end-to-end with FakeGateway ────────────


def test_advise_records_verdict_and_returns_advice():
    cfg = AdvisorConfig(enabled=True, model="m", budget_usd=1.0, cooldown_iterations=0)
    gw = _FakeGateway(advice_text="Do X, then verify with Y.")
    session = AdvisorSession(gateway=gw, config=cfg)

    verdict = _run(session.advise(
        question="What now?",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=[],
        ctx=_Ctx(),
        iteration_count=3,
    ))
    assert verdict.throttled is False
    assert "Do X" in verdict.advice
    assert verdict.cost_usd == pytest.approx(0.05)
    # The session does NOT auto-record — orchestrator handler does.
    assert session.call_count == 0
    session.record_verdict(verdict, iteration_count=3)
    assert session.call_count == 1


def test_advise_returns_throttled_on_gateway_exception():
    cfg = AdvisorConfig(enabled=True, model="m", budget_usd=1.0, cooldown_iterations=0)

    class _BoomGateway:
        async def call(self, **kw):
            raise RuntimeError("upstream blew up")

    session = AdvisorSession(gateway=_BoomGateway(), config=cfg)
    verdict = _run(session.advise(
        question="?",
        urgency=AdvisorUrgency.NORMAL,
        trigger=AdvisorTrigger.TOOL,
        conversation=[],
        ctx=_Ctx(),
        iteration_count=3,
    ))
    assert verdict.throttled is True
    assert verdict.throttle_reason.startswith("call_failed")
