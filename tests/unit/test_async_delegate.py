"""Tests for unified Delegate tool with background support, and AsyncDelegateManager."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from maike.atoms.agent import AgentResult
from maike.atoms.context import AgentProgress
from maike.atoms.tool import RiskLevel, ToolResult
from maike.constants import MAX_ASYNC_DELEGATES
from maike.orchestrator.orchestrator import AsyncDelegateManager
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


def _make_agent_result(**overrides):
    defaults = dict(
        agent_id="test",
        role="delegate",
        stage_name="test",
        output="Done",
        messages=[],
        cost_usd=0.1,
        tokens_used=100,
        success=True,
    )
    defaults.update(overrides)
    return AgentResult(**defaults)


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------


class TestDelegateRegistration:
    def test_delegate_registered(self):
        """Verify Delegate is registered with EXECUTE risk level."""
        registry = ToolRegistry()

        async def fake_sync_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        register_delegate_tool(registry, fake_sync_handler)
        tool = registry.get("Delegate")
        assert tool is not None
        assert tool.risk_level == RiskLevel.WRITE

    def test_only_delegate_registered(self):
        """Only one Delegate tool — no DelegateAsync or DelegateCheck."""
        registry = ToolRegistry()

        async def fake_sync_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        async def fake_async_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="spawned")

        register_delegate_tool(registry, fake_sync_handler, fake_async_handler)
        names = registry.list_tool_names()
        assert "DelegateAsync" not in names
        assert "DelegateCheck" not in names
        assert "Delegate" in names

    def test_delegate_has_action_and_handle_in_schema(self):
        """Verify 'action' and 'handle' are in the Delegate tool schema for check/wait."""
        registry = ToolRegistry()

        async def fake_sync_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        register_delegate_tool(registry, fake_sync_handler)
        tool = registry.get("Delegate")
        schema = tool.schema.input_schema
        assert "action" in schema["properties"]
        assert "handle" in schema["properties"]


# ---------------------------------------------------------------------------
# Tool execution tests (mock handlers)
# ---------------------------------------------------------------------------


class TestDelegateExecution:
    def test_delegate_sync_explicit(self):
        """Delegate(task="...", background=False) blocks and returns result."""

        async def mock_sync_handler(**kwargs):
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="Completed the task",
                metadata={"agent_id": "child-001"},
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, mock_sync_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(tool.fn(task="do some work", background=False))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is True
        assert "Completed the task" in result.output
        assert "child-001" in ctx.children_ids

    def test_delegate_background_returns_file_path(self):
        """Delegate(task="...", background=true) returns output with file path."""

        async def mock_sync_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        async def mock_async_handler(**kwargs):
            return ToolResult(
                tool_name="Delegate",
                success=True,
                output="Async delegate spawned: delegate-001\nOutput file: /tmp/delegates/delegate-001.log",
                metadata={
                    "handle": "delegate-001",
                    "agent_id": "child-002",
                    "output_file": "/tmp/delegates/delegate-001.log",
                },
            )

        registry = ToolRegistry()
        register_delegate_tool(registry, mock_sync_handler, mock_async_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(tool.fn(task="do background work", background=True))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is True
        assert "delegate-001" in result.output
        assert "Output file:" in result.output
        assert result.metadata["handle"] == "delegate-001"
        assert "child-002" in ctx.children_ids

    def test_delegate_background_unavailable(self):
        """Delegate(background=true) without async_handler returns error."""

        async def mock_sync_handler(**kwargs):
            return ToolResult(tool_name="Delegate", success=True, output="done")

        registry = ToolRegistry()
        register_delegate_tool(registry, mock_sync_handler)
        tool = registry.get("Delegate")

        ctx = _make_ctx()
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            result = asyncio.run(tool.fn(task="do work", background=True))
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

        assert result.success is False
        assert "not available" in result.output.lower()


# ---------------------------------------------------------------------------
# AsyncDelegateManager unit tests
# ---------------------------------------------------------------------------


class TestAsyncDelegateManager:
    def test_start_assigns_handle(self):
        """Start a simple coroutine, verify handle format and status."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result()

            delegate = await mgr.start(_quick(), ctx, "test task")
            assert delegate.handle == "delegate-001"
            assert delegate.status == "running"
            assert delegate.task_description == "test task"
            return delegate

        asyncio.run(_run())

    def test_check_completed(self):
        """Start a coroutine that completes immediately, wait, check status."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result(output="All done")

            delegate = await mgr.start(_quick(), ctx, "quick task")
            # Wait for the task to finish
            await delegate.task_future
            status = mgr.check(delegate.handle)
            assert status["status"] == "completed"
            assert status["output"] == "All done"

        asyncio.run(_run())

    def test_check_running(self):
        """Start a long coroutine, check status is 'running'."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            event = asyncio.Event()

            async def _slow():
                await event.wait()
                return _make_agent_result()

            delegate = await mgr.start(_slow(), ctx, "slow task")
            status = mgr.check(delegate.handle)
            assert status["status"] == "running"
            # Unblock so cleanup works
            event.set()
            await delegate.task_future

        asyncio.run(_run())

    def test_check_failed(self):
        """Start a coroutine that raises, verify status is 'failed'."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _failing():
                raise RuntimeError("something broke")

            delegate = await mgr.start(_failing(), ctx, "failing task")
            # _run_and_track captures exceptions without re-raising,
            # so awaiting the task completes normally.
            await delegate.task_future
            status = mgr.check(delegate.handle)
            assert status["status"] == "failed"
            assert "something broke" in status["error"]

        asyncio.run(_run())

    def test_check_unknown_handle_raises(self):
        """Check with an unknown handle raises KeyError."""
        mgr = AsyncDelegateManager()
        try:
            mgr.check("delegate-999")
            assert False, "Expected KeyError"
        except KeyError as exc:
            assert "delegate-999" in str(exc)

    def test_cleanup_cancels_running(self):
        """Start multiple delegates, cleanup, verify all cancelled."""

        async def _run():
            mgr = AsyncDelegateManager()

            events = []
            for i in range(3):
                ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())
                event = asyncio.Event()
                events.append(event)

                async def _slow(e=event):
                    await e.wait()
                    return _make_agent_result()

                await mgr.start(_slow(), ctx, f"task-{i}")

            # All three should be running
            for handle in list(mgr._delegates):
                assert mgr._delegates[handle].status == "running"

            await mgr.cleanup(grace_period=0.1)

            # After cleanup, all should be failed/cancelled
            for handle in list(mgr._delegates):
                d = mgr._delegates[handle]
                assert d.status == "failed"
                assert "Cancelled on session cleanup" in d.error

        asyncio.run(_run())

    def test_counter_increments(self):
        """Each start call increments the counter for unique handles."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result()

            d1 = await mgr.start(_quick(), ctx, "task 1")
            d2 = await mgr.start(_quick(), ctx, "task 2")
            assert d1.handle == "delegate-001"
            assert d2.handle == "delegate-002"
            # Wait for completion
            await d1.task_future
            await d2.task_future

        asyncio.run(_run())

    def test_delegate_completion_notification(self):
        """Verify notification contains inline result, not file pointer."""

        async def _run():
            queue: asyncio.Queue = asyncio.Queue()
            mgr = AsyncDelegateManager(notification_queue=queue)
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result(output="All done", success=True)

            delegate = await mgr.start(_quick(), ctx, "notified task")
            await delegate.task_future

            assert not queue.empty()
            notification = await queue.get()
            assert isinstance(notification, str)
            assert "delegate-001" in notification
            assert "status=\"success\"" in notification
            assert "notified task" in notification
            # Inline result should be present.
            assert "All done" in notification

        asyncio.run(_run())

    def test_delegate_failure_notification(self):
        """Verify failure notification is pushed to queue on delegate error."""

        async def _run():
            queue: asyncio.Queue = asyncio.Queue()
            mgr = AsyncDelegateManager(notification_queue=queue)
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _failing():
                raise RuntimeError("boom")

            delegate = await mgr.start(_failing(), ctx, "failing task")
            try:
                await delegate.task_future
            except RuntimeError:
                pass

            assert not queue.empty()
            notification = await queue.get()
            assert isinstance(notification, str)
            assert "delegate-001" in notification
            assert "failed" in notification.lower()
            assert "boom" in notification

        asyncio.run(_run())

    def test_delegate_output_written_to_file(self, tmp_path):
        """Verify log file is written on delegate completion."""

        async def _run():
            mgr = AsyncDelegateManager(output_dir=tmp_path / "delegates")
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result(output="Task completed successfully", cost_usd=0.05, tokens_used=500)

            delegate = await mgr.start(_quick(), ctx, "logged task")
            await delegate.task_future

            assert delegate.output_file is not None
            assert delegate.output_file.exists()
            content = delegate.output_file.read_text()
            assert "handle: delegate-001" in content
            assert "status: completed" in content
            assert "logged task" in content
            assert "Task completed successfully" in content

        asyncio.run(_run())

    def test_delegate_error_written_to_file(self, tmp_path):
        """Verify log file is written on delegate failure."""

        async def _run():
            mgr = AsyncDelegateManager(output_dir=tmp_path / "delegates")
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _failing():
                raise RuntimeError("disk full")

            delegate = await mgr.start(_failing(), ctx, "error task")
            try:
                await delegate.task_future
            except RuntimeError:
                pass

            assert delegate.output_file is not None
            assert delegate.output_file.exists()
            content = delegate.output_file.read_text()
            assert "handle: delegate-001" in content
            assert "status: failed" in content
            assert "disk full" in content

        asyncio.run(_run())

    def test_no_output_file_without_output_dir(self):
        """Without output_dir, delegate.output_file is None."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result()

            delegate = await mgr.start(_quick(), ctx, "task")
            assert delegate.output_file is None
            await delegate.task_future

        asyncio.run(_run())


    def test_completion_event_signaled(self):
        """Verify completion event is set when a delegate finishes."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result()

            assert not mgr._completion_event.is_set()
            delegate = await mgr.start(_quick(), ctx, "event task")
            await delegate.task_future
            # Give the event loop a tick for _run_and_track to set the event.
            await asyncio.sleep(0)
            assert mgr._completion_event.is_set()

        asyncio.run(_run())

    def test_completion_event_on_failure(self):
        """Verify completion event is set even when a delegate fails."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _failing():
                raise RuntimeError("oops")

            delegate = await mgr.start(_failing(), ctx, "fail event")
            await delegate.task_future
            await asyncio.sleep(0)
            assert mgr._completion_event.is_set()

        asyncio.run(_run())

    def test_has_running_delegates(self):
        """Property reflects whether any delegates are running."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            assert not mgr.has_running_delegates

            event = asyncio.Event()

            async def _slow():
                await event.wait()
                return _make_agent_result()

            delegate = await mgr.start(_slow(), ctx, "slow task")
            assert mgr.has_running_delegates

            event.set()
            await delegate.task_future
            assert not mgr.has_running_delegates

        asyncio.run(_run())

    def test_wait_for_any_completion(self):
        """wait_for_any_completion returns True when a delegate finishes."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result()

            await mgr.start(_quick(), ctx, "wait task")
            signaled = await mgr.wait_for_any_completion(timeout=2.0)
            assert signaled is True
            # Event should be cleared after wait.
            assert not mgr._completion_event.is_set()

        asyncio.run(_run())

    def test_wait_for_any_completion_timeout(self):
        """wait_for_any_completion returns False on timeout."""

        async def _run():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            event = asyncio.Event()

            async def _slow():
                await event.wait()
                return _make_agent_result()

            await mgr.start(_slow(), ctx, "slow task")
            signaled = await mgr.wait_for_any_completion(timeout=0.05)
            assert signaled is False
            # Clean up
            event.set()

        asyncio.run(_run())

    def test_graceful_cleanup_preserves_fast_delegates(self):
        """Delegates that finish during grace period are preserved, not cancelled."""

        async def _run():
            mgr = AsyncDelegateManager()

            # One fast delegate, one blocked delegate.
            ctx1 = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())
            ctx2 = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _fast():
                await asyncio.sleep(0.05)
                return _make_agent_result(output="fast done")

            blocked = asyncio.Event()

            async def _blocked():
                await blocked.wait()
                return _make_agent_result()

            d1 = await mgr.start(_fast(), ctx1, "fast task")
            d2 = await mgr.start(_blocked(), ctx2, "blocked task")

            await mgr.cleanup(grace_period=0.5)

            # Fast delegate should have completed naturally.
            assert mgr._delegates[d1.handle].status == "completed"
            assert mgr._delegates[d1.handle].result is not None
            # Blocked delegate should be cancelled.
            assert mgr._delegates[d2.handle].status == "failed"
            assert "Cancelled" in mgr._delegates[d2.handle].error

        asyncio.run(_run())

    def test_notification_truncates_long_output(self):
        """Long delegate output is truncated in the notification."""

        async def _run():
            queue: asyncio.Queue = asyncio.Queue()
            mgr = AsyncDelegateManager(notification_queue=queue)
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            long_output = "x" * 15000  # exceeds _INLINE_OUTPUT_LIMIT (12K)

            async def _quick():
                return _make_agent_result(output=long_output)

            delegate = await mgr.start(_quick(), ctx, "long task")
            await delegate.task_future

            notification = await queue.get()
            assert "(truncated)" in notification
            # Notification should not contain the full 15000 chars.
            assert len(notification) < 14000

        asyncio.run(_run())


class TestMaxAsyncDelegatesConstant:
    def test_max_async_delegates_exists(self):
        assert MAX_ASYNC_DELEGATES == 5
