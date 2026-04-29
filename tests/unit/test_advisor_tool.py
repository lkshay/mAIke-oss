"""Unit tests for Phase 2 of the advisor pattern: the Advisor tool."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from maike.atoms.tool import RiskLevel, ToolResult
from maike.tools.advisor import register_advisor_tool
from maike.tools.registry import ToolRegistry


def _run(coro):
    return asyncio.run(coro)


def _captured_handler():
    """Return (handler, calls_list) — calls_list records every invocation."""
    calls: list[dict[str, Any]] = []

    async def handler(**kwargs):
        calls.append(kwargs)
        return ToolResult(
            tool_name="Advisor",
            success=True,
            output="pretend advice",
        )

    return handler, calls


# ── Schema ──────────────────────────────────────────────────────────


def test_advisor_tool_registers_with_schema():
    handler, _ = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)
    assert "Advisor" in registry.list_tool_names()


def test_advisor_tool_schema_shape():
    handler, _ = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)
    schemas = registry.get_all_schemas()
    advisor = next(s for s in schemas if s["name"] == "Advisor")
    assert advisor["name"] == "Advisor"
    assert "question" in advisor["input_schema"]["properties"]
    assert "urgency" in advisor["input_schema"]["properties"]
    assert advisor["input_schema"]["required"] == ["question"]
    # Urgency is enum-constrained.
    assert advisor["input_schema"]["properties"]["urgency"]["enum"] == ["normal", "stuck"]


def test_advisor_tool_is_read_risk_level():
    """Advisor never mutates state — must be READ so it bypasses WRITE gates."""
    handler, _ = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)
    # ToolRegistry stores tools internally; we can iterate via get_all_tools.
    for tool in registry.get_all_tools():
        if tool.schema.name == "Advisor":
            assert tool.risk_level == RiskLevel.READ
            return
    pytest.fail("Advisor tool not found in registry")


# ── Handler invocation ─────────────────────────────────────────────


def test_advisor_handler_receives_question_and_urgency():
    handler, calls = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)

    invoke = _get_tool_fn(registry, "Advisor")
    result = _run(invoke(question="Is my plan sound?", urgency="stuck"))
    assert result.success is True
    assert calls == [{"question": "Is my plan sound?", "urgency": "stuck"}]


def test_advisor_handler_defaults_urgency_to_normal():
    handler, calls = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)

    invoke = _get_tool_fn(registry, "Advisor")
    result = _run(invoke(question="help"))
    assert result.success is True
    assert calls[0]["urgency"] == "normal"


def test_advisor_handler_normalizes_invalid_urgency_to_normal():
    handler, calls = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)

    invoke = _get_tool_fn(registry, "Advisor")
    result = _run(invoke(question="help", urgency="PANIC"))
    assert result.success is True
    # The tool sanitizes unknown urgency values to "normal".
    assert calls[0]["urgency"] == "normal"


def test_advisor_handler_rejects_empty_question():
    handler, calls = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)

    invoke = _get_tool_fn(registry, "Advisor")
    result = _run(invoke(question=""))
    assert result.success is False
    # Handler must NOT be called when question is missing.
    assert calls == []


def test_advisor_handler_strips_whitespace_question():
    handler, calls = _captured_handler()
    registry = ToolRegistry()
    register_advisor_tool(registry, handler)

    invoke = _get_tool_fn(registry, "Advisor")
    result = _run(invoke(question="   "))
    assert result.success is False  # whitespace-only still counts as empty
    assert calls == []


def test_advisor_handler_propagates_throttled_result():
    """When the orchestrator's handler returns a throttled ToolResult, the
    tool wrapper must pass it through unchanged."""
    async def throttled_handler(**kwargs):
        return ToolResult(
            tool_name="Advisor",
            success=True,
            output="Advisor budget exhausted — proceed with your own judgement.",
            metadata={"throttled": True, "throttle_reason": "budget_exhausted"},
        )

    registry = ToolRegistry()
    register_advisor_tool(registry, throttled_handler)

    invoke = _get_tool_fn(registry, "Advisor")
    result = _run(invoke(question="Help"))
    assert result.success is True
    assert result.metadata.get("throttled") is True
    assert result.metadata.get("throttle_reason") == "budget_exhausted"


# ── register_default_tools wiring ─────────────────────────────────────


def test_register_default_tools_wires_advisor_when_handler_provided():
    from maike.runtime.local import LocalRuntime
    from maike.tools import register_default_tools
    from pathlib import Path
    import tempfile

    handler, _ = _captured_handler()
    registry = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        runtime = LocalRuntime(Path(tmp))
        register_default_tools(registry, runtime, advisor_handler=handler)
    assert "Advisor" in registry.list_tool_names()


def test_register_default_tools_omits_advisor_when_no_handler():
    from maike.runtime.local import LocalRuntime
    from maike.tools import register_default_tools
    from pathlib import Path
    import tempfile

    registry = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        runtime = LocalRuntime(Path(tmp))
        register_default_tools(registry, runtime)  # no advisor_handler
    assert "Advisor" not in registry.list_tool_names()


# ── Orchestrator._make_advisor_handler integration ────────────────


def test_make_advisor_handler_returns_error_when_no_loop_state():
    """If the handler is somehow called outside an active agent turn, it must
    fail cleanly instead of crashing.
    """
    from maike.agents.advisor import AdvisorConfig, AdvisorSession
    from maike.cost.tracker import CostTracker
    from maike.observability.tracer import Tracer
    from maike.orchestrator.orchestrator import Orchestrator
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        orch = Orchestrator(
            base_path=Path(tmp),
            cost_tracker=CostTracker(),
            tracer=Tracer(),
        )
        session = AdvisorSession(
            gateway=object(),
            config=AdvisorConfig(enabled=True, budget_usd=1.0),
        )
        handler = orch._make_advisor_handler(session)
        # No CURRENT_AGENT_LOOP_STATE set → graceful error.
        result = _run(handler(question="help", urgency="normal"))
        assert result.success is False
        assert "outside an active agent turn" in result.output


# ── Helpers ────────────────────────────────────────────────────────


def _get_tool_fn(registry: ToolRegistry, name: str):
    """Pull the async `fn` out of the registry for *name*."""
    for tool in registry.get_all_tools():
        if tool.schema.name == name:
            return tool.fn
    raise KeyError(name)
