"""Tests for the Delegate tool and delegation system."""

import asyncio
from types import SimpleNamespace

from maike.atoms.blueprint import SpawnReason
from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.constants import (
    DELEGATION_BUDGET_CAP_USD,
    DELEGATION_BUDGET_FRACTION,
    DELEGATION_MAX_DEPTH,
    DELEGATION_MAX_ITERATIONS,
    DELEGATION_TOKEN_BUDGET,
)
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


class TestDelegateToolRegistration:
    def test_delegate_tool_is_registered(self):
        registry = ToolRegistry()

        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        register_delegate_tool(registry, fake_handler)
        tool = registry.get("Delegate")
        assert tool is not None
        assert tool.risk_level == RiskLevel.WRITE

    def test_delegate_tool_schema_has_required_fields(self):
        registry = ToolRegistry()

        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        register_delegate_tool(registry, fake_handler)
        tool = registry.get("Delegate")
        schema = tool.schema.input_schema
        assert "task" in schema["properties"]
        assert "context" in schema["properties"]
        assert "model_tier" in schema["properties"]
        assert schema["properties"]["model_tier"]["enum"] == ["default", "cheap", "strong"]
        # No fields are globally required since background mode doesn't need all params.
        assert schema["required"] == []

    def test_delegate_tool_schema_has_action_and_handle(self):
        """The Delegate schema includes action and handle for check/wait."""
        registry = ToolRegistry()

        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        register_delegate_tool(registry, fake_handler)
        tool = registry.get("Delegate")
        schema = tool.schema.input_schema
        assert "action" in schema["properties"]
        assert "handle" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["check", "wait", "send"]


class TestDelegateToolExecution:
    def test_delegate_calls_handler_with_budget_info(self):
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="sub-agent completed",
                metadata={"agent_id": "delegate-001"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(tool.fn(task="fix the auth bug", context="KeyError on login", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is True
        assert "sub-agent completed" in result.output
        assert captured_kwargs["task"] == "fix the auth bug"
        assert captured_kwargs["context"] == "KeyError on login"
        assert captured_kwargs["parent_agent_id"] == "parent-001"
        assert captured_kwargs["remaining_token_budget"] == 800_000
        assert captured_kwargs["remaining_cost_budget_usd"] == 4.0

    def test_delegate_tracks_child_agent_id(self):
        async def handler(**kwargs):
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-002"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="do something", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert "delegate-002" in ctx.children_ids

    def test_delegate_passes_model_tier_to_handler(self):
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-tier"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="hard task", model_tier="strong", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert captured_kwargs["model_tier"] == "strong"

    def test_delegate_defaults_model_tier_to_default(self):
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-default"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="normal task", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        # Default agent_type is "explore", which auto-selects "cheap" model tier.
        assert captured_kwargs["model_tier"] == "cheap"

    def test_delegate_sanitizes_invalid_model_tier(self):
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-sanitize"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="task", model_tier="invalid", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert captured_kwargs["model_tier"] == "default"

    def test_delegate_handles_failure(self):
        async def failing_handler(**kwargs):
            return ToolResult(
                tool_name="Delegate",
                success=False,
                output="Sub-agent failed: timeout",
                error="timeout",
                metadata={"agent_id": "delegate-003"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, failing_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(tool.fn(task="impossible task", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is False
        assert "failed" in result.output.lower()


class TestDelegationConstants:
    def test_delegation_max_depth_prevents_recursion(self):
        assert DELEGATION_MAX_DEPTH == 1

    def test_delegation_budget_fraction_is_reasonable(self):
        assert 0 < DELEGATION_BUDGET_FRACTION <= 0.5

    def test_delegation_budget_cap_exists(self):
        assert DELEGATION_BUDGET_CAP_USD > 0

    def test_delegation_token_budget_is_set(self):
        assert DELEGATION_TOKEN_BUDGET > 0
        assert DELEGATION_TOKEN_BUDGET <= 1_000_000

    def test_delegation_max_iterations_is_set(self):
        assert DELEGATION_MAX_ITERATIONS > 0


class TestSpawnReasonEnum:
    def test_delegation_spawn_reason_exists(self):
        assert SpawnReason.DELEGATION == "delegation"
        assert SpawnReason.DELEGATION.value == "delegation"


class TestAgentTypeSchema:
    def test_schema_has_all_seven_agent_types(self):
        registry = ToolRegistry()

        async def fake_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        register_delegate_tool(registry, fake_handler)
        tool = registry.get("Delegate")
        enum_values = tool.schema.input_schema["properties"]["agent_type"]["enum"]
        assert set(enum_values) == {"explore", "plan", "implement", "review", "verify", "debug", "test"}

    def test_delegate_debug_defaults_to_strong_model(self):
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-debug"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="debug a failing test", agent_type="debug", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert captured_kwargs["model_tier"] == "strong"

    def test_delegate_explore_defaults_to_cheap_model(self):
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-explore"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="find usages", agent_type="explore", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert captured_kwargs["model_tier"] == "cheap"

    def test_delegate_plan_keeps_default_model(self):
        """Plan agents should NOT be downgraded — planning needs reasoning quality."""
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-plan"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="plan the implementation", agent_type="plan", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert captured_kwargs["model_tier"] == "default"

    def test_delegate_unknown_type_falls_back_to_implement(self):
        captured_kwargs = {}

        async def capture_handler(**kwargs):
            captured_kwargs.update(kwargs)
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="done",
                metadata={"agent_id": "delegate-fallback"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, capture_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            asyncio.run(tool.fn(task="do something", agent_type="nonexistent", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        # Unknown types fall back to "explore" which auto-selects "cheap" model tier.
        assert captured_kwargs["model_tier"] == "cheap"
