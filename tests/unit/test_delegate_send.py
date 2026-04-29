"""Tests for Delegate action='send' (agent communication)."""

import asyncio
from types import SimpleNamespace

from maike.atoms.tool import ToolResult
from maike.tools.context import CURRENT_AGENT_CONTEXT
from maike.tools.delegate import register_delegate_tool
from maike.tools.registry import ToolRegistry


def _make_ctx(**overrides):
    defaults = dict(
        agent_id="parent-001",
        role="react_agent",
        task="build something",
        stage_name="react",
        tool_profile="react",
        token_budget=1_000_000,
        cost_budget_usd=5.0,
        tokens_used=200_000,
        cost_used_usd=1.0,
        children_ids=[],
        metadata={"session_id": "session-1"},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestDelegateSendSchema:
    def test_schema_includes_send_action(self):
        registry = ToolRegistry()

        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        register_delegate_tool(registry, fake_handler)
        tool = registry.get("Delegate")
        enum_values = tool.schema.input_schema["properties"]["action"]["enum"]
        assert "send" in enum_values

    def test_send_requires_handle(self):
        captured = {}

        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        async def fake_send(**kwargs):
            captured.update(kwargs)
            return ToolResult(tool_name="Delegate", success=True, output="sent")

        registry = ToolRegistry()
        register_delegate_tool(registry, fake_handler, send_handler=fake_send)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(tool.fn(action="send", task="do something"))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is False
        assert "handle" in result.output.lower()

    def test_send_requires_task(self):
        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        async def fake_send(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="sent")

        registry = ToolRegistry()
        register_delegate_tool(registry, fake_handler, send_handler=fake_send)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(tool.fn(action="send", handle="delegate-001"))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is False
        assert "task" in result.output.lower() or "message" in result.output.lower()

    def test_send_calls_handler(self):
        captured = {}

        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        async def fake_send(**kwargs):
            captured.update(kwargs)
            return ToolResult(tool_name="Delegate", success=True, output="sent")

        registry = ToolRegistry()
        register_delegate_tool(registry, fake_handler, send_handler=fake_send)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(
                tool.fn(action="send", handle="delegate-001", task="also check errors")
            )
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is True
        assert captured["handle"] == "delegate-001"
        assert captured["message"] == "also check errors"
