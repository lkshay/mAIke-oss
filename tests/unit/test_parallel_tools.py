"""Tests for parallel tool execution in AgentCore."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from maike.agents.core import AgentCore
from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.tools.registry import ToolRegistry


def _make_registry_with_grep(delay: float = 0.1) -> ToolRegistry:
    """Create a registry with a fake Grep tool that sleeps for *delay* seconds."""
    registry = ToolRegistry()

    async def fake_grep(**kwargs) -> ToolResult:
        await asyncio.sleep(delay)
        return ToolResult(
            tool_name="Grep",
            success=True,
            output=f"matched: {kwargs.get('pattern', '?')}",
        )

    registry.register(
        ToolSchema(
            name="Grep",
            description="Search",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        ),
        fn=fake_grep,
        risk_level=RiskLevel.READ,
    )
    return registry


def _make_registry_with_grep_and_write(delay: float = 0.1) -> ToolRegistry:
    """Registry with both Grep (parallel-safe) and Write (sequential)."""
    registry = _make_registry_with_grep(delay)

    async def fake_write(**kwargs) -> ToolResult:
        await asyncio.sleep(delay)
        return ToolResult(tool_name="Write", success=True, output="written")

    registry.register(
        ToolSchema(
            name="Write",
            description="Write file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        ),
        fn=fake_write,
        risk_level=RiskLevel.WRITE,
    )
    return registry


def _make_core(registry: ToolRegistry) -> AgentCore:
    """Build a minimal AgentCore with mocked dependencies."""
    gateway = MagicMock()
    tracer = MagicMock()
    tracer.log_tool_result = MagicMock()
    tracer.log_tool_start = MagicMock()
    safety = MagicMock()
    safety.assess = MagicMock(return_value=MagicMock(decision="ALLOW"))
    working_memory = MagicMock()
    approval_gate = MagicMock()
    core = AgentCore(
        llm_gateway=gateway,
        tool_registry=registry,
        runtime=MagicMock(),
        safety_layer=safety,
        working_memory=working_memory,
        tracer=tracer,
        approval_gate=approval_gate,
    )
    return core


def test_all_grep_calls_run_in_parallel():
    """When all tool calls are parallel-safe (Grep), they should run concurrently."""
    delay = 0.2
    registry = _make_registry_with_grep(delay=delay)
    core = _make_core(registry)

    tool_calls = [
        {"id": "t1", "name": "Grep", "input": {"pattern": "foo"}},
        {"id": "t2", "name": "Grep", "input": {"pattern": "bar"}},
        {"id": "t3", "name": "Grep", "input": {"pattern": "baz"}},
    ]

    async def _run():
        start = time.monotonic()
        results = list(await asyncio.gather(
            *(core._execute_tool(tc, MagicMock()) for tc in tool_calls)
        ))
        elapsed = time.monotonic() - start
        return results, elapsed

    # Verify the parallel-safe check works
    assert all(
        core._resolve_tool_name(tc["name"]).resolved_name in core._PARALLEL_SAFE_TOOLS
        for tc in tool_calls
    )

    results, elapsed = asyncio.run(_run())
    assert len(results) == 3
    # 3 tasks at 0.2s each should complete in ~0.2s (parallel), not ~0.6s (sequential)
    assert elapsed < delay * 2, f"Expected parallel execution (<{delay * 2}s), got {elapsed:.2f}s"


def test_mixed_batch_is_not_parallel_safe():
    """A batch with both Grep and Write should NOT be identified as parallel-safe."""
    registry = _make_registry_with_grep_and_write()
    core = _make_core(registry)

    tool_calls = [
        {"id": "t1", "name": "Grep", "input": {"pattern": "foo"}},
        {"id": "t2", "name": "Write", "input": {"path": "x.py", "content": "hi"}},
    ]

    all_parallel = len(tool_calls) > 1 and all(
        core._resolve_tool_name(tc["name"]).resolved_name in core._PARALLEL_SAFE_TOOLS
        for tc in tool_calls
    )
    assert all_parallel is False


def test_single_tool_call_is_not_parallel():
    """A single tool call should not attempt parallelization."""
    registry = _make_registry_with_grep()
    core = _make_core(registry)

    tool_calls = [
        {"id": "t1", "name": "Grep", "input": {"pattern": "foo"}},
    ]

    all_parallel = len(tool_calls) > 1 and all(
        core._resolve_tool_name(tc["name"]).resolved_name in core._PARALLEL_SAFE_TOOLS
        for tc in tool_calls
    )
    assert all_parallel is False


def test_result_ordering_preserved():
    """Results must be returned in the same order as the original tool calls."""
    registry = ToolRegistry()
    call_order = []

    async def grep_with_tracking(**kwargs) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        # Stagger execution to test ordering
        delay = {"first": 0.15, "second": 0.05, "third": 0.10}.get(pattern, 0.01)
        await asyncio.sleep(delay)
        call_order.append(pattern)
        return ToolResult(tool_name="Grep", success=True, output=f"result:{pattern}")

    registry.register(
        ToolSchema(
            name="Grep",
            description="Search",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        ),
        fn=grep_with_tracking,
        risk_level=RiskLevel.READ,
    )
    core = _make_core(registry)

    tool_calls = [
        {"id": "t1", "name": "Grep", "input": {"pattern": "first"}},
        {"id": "t2", "name": "Grep", "input": {"pattern": "second"}},
        {"id": "t3", "name": "Grep", "input": {"pattern": "third"}},
    ]

    async def _run():
        return list(await asyncio.gather(
            *(core._execute_tool(tc, MagicMock()) for tc in tool_calls)
        ))

    results = asyncio.run(_run())
    # Results should be in original order (first, second, third)
    # even though "second" completes fastest
    assert results[0].output == "result:first"
    assert results[1].output == "result:second"
    assert results[2].output == "result:third"
