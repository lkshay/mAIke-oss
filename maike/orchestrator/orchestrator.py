"""Main orchestration flow."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from maike.utils import utcnow

from maike.agents.core import AgentCore
from maike.atoms.agent import AgentResult
from maike.atoms.context import AgentContext, TaskState
from maike.constants import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_REACT_MAX_ITERATIONS, DEFAULT_RUN_BUDGET_USD
from maike.cost.tracker import BudgetExceededError, CostTracker
from maike.gateway.llm_gateway import LLMGateway
from maike.gateway.providers import resolve_model_name, resolve_provider_name
from maike.memory.learning import SessionLearner
from maike.memory.longterm import create_long_term_memory
from maike.memory.session import SessionStore
from maike.memory.working import WorkingMemory
from maike.observability.tracer import TraceEventKind, Tracer
from maike.orchestrator.preflight import PreflightChecker
from maike.orchestrator.session import OrchestratorSession
from maike.runtime.local import LocalRuntime, RuntimeConfig
from maike.runtime.probe import EnvironmentProbe
from maike.runtime.protocol import ExecutionRuntime
from maike.safety.approval import ApprovalGate
from maike.safety.hooks import SafetyLayer
from maike.tools import register_default_tools
from maike.atoms.tool import ToolResult as TR
from maike.tools.context import CURRENT_CODE_INDEX, CURRENT_SKILL_LOADER, peek_current_agent_context
from maike.tools.registry import ToolRegistry


class OrchestratorError(RuntimeError):
    """Raised when a stage cannot complete successfully."""


class OrchestratorCancelled(RuntimeError):
    """Raised when a session is cancelled at a safe boundary."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session cancelled: {session_id}")
        self.session_id = session_id


@dataclass
class CancellationController:
    cancel_requested: bool = False

    def request_cancel(self) -> None:
        self.cancel_requested = True


@dataclass
class OrchestratorResult:
    session_id: str
    pipeline: str
    stage_results: dict[str, list[AgentResult]] = field(default_factory=dict)
    thread_id: str | None = None


def _parse_delegate_structured_output(agent_type: str, output: str) -> str:
    """Extract structured metadata from delegate outputs.

    - verify delegates: look for VERDICT: PASS|FAIL|PARTIAL
    - review delegates: count [CRITICAL], [WARNING], [NIT] tags
    Returns a summary line, or empty string for other types.
    """
    if not output:
        return ""
    if agent_type == "verify":
        import re
        m = re.search(r"VERDICT:\s*(PASS|FAIL|PARTIAL)", output, re.IGNORECASE)
        if m:
            return f"\n**Verdict: {m.group(1).upper()}**\n"
    elif agent_type == "review":
        critical = output.count("[CRITICAL]")
        warning = output.count("[WARNING]")
        nit = output.count("[NIT]")
        if critical or warning or nit:
            return f"\n**Review: {critical} critical, {warning} warnings, {nit} nits**\n"
    return ""


@dataclass
class DelegateQualityReport:
    """Assessment of a delegate's output quality."""
    tool_calls_made: int
    read_calls: int       # Read, Grep, SemanticSearch
    write_calls: int      # Write, Edit
    bash_calls: int
    quality_flag: str     # "good", "suspect", "hallucinated"
    warning_text: str     # empty if good

    @property
    def is_hallucinated(self) -> bool:
        return self.quality_flag == "hallucinated"


# Minimum tool call expectations by delegate type.
_DELEGATE_QUALITY_EXPECTATIONS: dict[str, set[str]] = {
    "explore": {"Read", "Grep", "SemanticSearch"},
    "plan": {"Read", "Grep", "SemanticSearch"},
    "implement": {"Write", "Edit"},
    "review": {"Read", "Grep"},
    "verify": {"Bash"},
    "debug": {"Read", "Grep"},
    "test": {"Bash"},
}


def assess_delegate_quality(
    result: AgentResult,
    agent_type: str,
) -> DelegateQualityReport:
    """Assess whether a delegate's output is grounded in actual tool use."""
    read_names = {"Read", "Grep", "SemanticSearch", "read_file", "grep_codebase", "search_files"}
    write_names = {"Write", "Edit", "write_file", "edit_file"}
    bash_names = {"Bash", "execute_bash", "run_tests"}

    tool_calls = 0
    read_calls = 0
    write_calls = 0
    bash_calls = 0
    only_ls = True  # Track if all Bash calls are just ls/find

    for msg in result.messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            tool_calls += 1
            if name in read_names:
                read_calls += 1
            elif name in write_names:
                write_calls += 1
            elif name in bash_names:
                bash_calls += 1
                cmd = block.get("input", {}).get("cmd", "")
                if not any(kw in cmd for kw in ("ls ", "ls\n", "find ", "tree ")):
                    only_ls = False

    # Zero tool calls + low tokens = hallucinated.
    if tool_calls == 0 and result.tokens_used < 5000:
        return DelegateQualityReport(
            tool_calls_made=tool_calls,
            read_calls=read_calls,
            write_calls=write_calls,
            bash_calls=bash_calls,
            quality_flag="hallucinated",
            warning_text=(
                "\n\n**WARNING: This delegate made 0 tool calls and "
                "used very few tokens. Its output is likely hallucinated "
                "rather than grounded in actual file reads. Consider "
                "re-delegating this task or exploring the directory yourself.**"
            ),
        )

    # Check type-specific expectations.
    expected_tools = _DELEGATE_QUALITY_EXPECTATIONS.get(agent_type, set())
    has_expected = (
        (expected_tools & read_names and read_calls > 0)
        or (expected_tools & write_names and write_calls > 0)
        or (expected_tools & bash_names and bash_calls > 0)
        or not expected_tools  # unknown type — no expectations
    )

    # Explore/plan delegate with only ls commands and no read/grep.
    if agent_type in ("explore", "plan") and read_calls == 0 and bash_calls > 0 and only_ls:
        return DelegateQualityReport(
            tool_calls_made=tool_calls,
            read_calls=read_calls,
            write_calls=write_calls,
            bash_calls=bash_calls,
            quality_flag="suspect",
            warning_text=(
                "\n\n**NOTE: This delegate only ran ls/find commands without "
                "reading any files. Its analysis may be based on file names "
                "alone rather than actual code inspection.**"
            ),
        )

    if not has_expected and tool_calls > 0:
        return DelegateQualityReport(
            tool_calls_made=tool_calls,
            read_calls=read_calls,
            write_calls=write_calls,
            bash_calls=bash_calls,
            quality_flag="suspect",
            warning_text=(
                f"\n\n**NOTE: This {agent_type} delegate did not use "
                f"the expected tools ({', '.join(sorted(expected_tools))}). "
                f"Its output may be incomplete.**"
            ),
        )

    return DelegateQualityReport(
        tool_calls_made=tool_calls,
        read_calls=read_calls,
        write_calls=write_calls,
        bash_calls=bash_calls,
        quality_flag="good",
        warning_text="",
    )


_HALLUCINATION_RETRY_NUDGE = {
    "role": "user",
    "content": (
        "Your previous response contained no tool calls. You MUST use tools "
        "to actually explore the codebase — do not make up or assume file contents. "
        "Start by using Read or Grep to examine the relevant files."
    ),
}


@dataclass
class AsyncDelegate:
    """Tracked handle for a sub-agent running in the background."""
    handle: str                        # "delegate-001"
    task_future: asyncio.Task          # asyncio.Task wrapping agent.run()
    ctx: AgentContext                   # for progress tracking
    task_description: str              # what the delegate is doing
    started_at: datetime
    status: TaskState = TaskState.RUNNING
    result: AgentResult | None = None
    error: str | None = None
    output_file: Path | None = None    # path to the log file
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)


class AsyncDelegateManager:
    """Track and manage async (non-blocking) delegate sub-agents."""

    def __init__(
        self,
        output_dir: Path | None = None,
        notification_queue: asyncio.Queue | None = None,
        session_store: "SessionStore | None" = None,
        session_id: str | None = None,
        tracer: "Tracer | None" = None,
    ) -> None:
        self._delegates: dict[str, AsyncDelegate] = {}
        self._counter: int = 0
        self._output_dir = output_dir
        self._notification_queue = notification_queue
        self._session_store = session_store
        self._session_id = session_id
        self._tracer = tracer
        self._completion_event: asyncio.Event = asyncio.Event()
        self._persisted_handles: set[str] = set()  # guards against double-write

    async def start(self, coro, ctx: AgentContext, task_desc: str) -> AsyncDelegate:
        """Wrap a coroutine in an asyncio.Task and track it."""
        self._counter += 1
        handle = f"delegate-{self._counter:03d}"

        # Compute output file path if output_dir is configured.
        output_file: Path | None = None
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self._output_dir / f"delegate-{self._counter:03d}.log"

        async def _run_and_track():
            try:
                result = await coro
                delegate = self._delegates[handle]
                delegate.status = TaskState.COMPLETED
                delegate.result = result
                # Write full result to log file (no truncation — disk is cheap).
                if delegate.output_file is not None:
                    full_output = result.output or "(no output)"
                    delegate.output_file.write_text(
                        f"handle: {handle}\n"
                        f"status: completed\n"
                        f"task: {delegate.task_description}\n"
                        f"cost_usd: {result.cost_usd}\n"
                        f"tokens_used: {result.tokens_used}\n"
                        f"---\n{full_output}\n"
                    )
                # Assess delegate output quality.
                _agent_type_for_quality = getattr(delegate.ctx, "metadata", {}).get("agent_type", "unknown")
                _quality = assess_delegate_quality(result, _agent_type_for_quality)

                # Push completion notification with inline result.
                if self._notification_queue is not None:
                    _raw = result.output or "(no output)"
                    _limit = self._INLINE_OUTPUT_LIMIT
                    if len(_raw) <= _limit:
                        _preview = _raw
                    else:
                        _preview = _raw[:_limit] + "\n...(truncated)"
                        if delegate.output_file is not None:
                            _preview += (
                                f"\n\nFull output ({len(_raw)} chars) saved to: "
                                f"{delegate.output_file}\n"
                                f"Use Read(file_path=\"{delegate.output_file}\") "
                                f"to see the complete result."
                            )
                    _agent_type = _agent_type_for_quality
                    # Parse structured outputs for verify/review delegates.
                    _structured = _parse_delegate_structured_output(_agent_type, _raw)

                    self._notification_queue.put_nowait(
                        f"<delegate-result handle=\"{handle}\" agent_type=\"{_agent_type}\" "
                        f"status=\"success\" quality=\"{_quality.quality_flag}\" "
                        f"tool_calls=\"{_quality.tool_calls_made}\" "
                        f"tokens=\"{delegate.ctx.tokens_used}\">\n"
                        f"Task: {delegate.task_description}\n"
                        f"Cost: ${delegate.ctx.cost_used_usd:.3f}\n"
                        f"{_structured}"
                        f"\n### Full Output\n{_preview}{_quality.warning_text}\n"
                        f"</delegate-result>"
                    )
                # Persist delegate result to session store.
                if self._session_store is not None and self._session_id is not None:
                    try:
                        await self._session_store.store_agent_run(self._session_id, result)
                        self._persisted_handles.add(handle)
                    except Exception:
                        pass  # best-effort — don't crash the delegate

                # Extract per-agent persistent memory if the delegate had
                # an agent-scoped TypedLongTermMemory in its context.
                _agent_mem = getattr(delegate.ctx, "metadata", {}).get("agent_typed_memory")
                if _agent_mem is not None:
                    try:
                        from maike.memory.automemory import extract_session_memories
                        _session_mem = getattr(result, "session_memory", None)
                        extract_session_memories(
                            session_memory=_session_mem,
                            agent_output=result.output,
                            typed_memory=_agent_mem,
                            task=delegate.task_description,
                        )
                    except Exception:
                        pass  # best-effort

                if self._tracer is not None:
                    self._tracer.log_event(
                        TraceEventKind.ASYNC_DELEGATE_COMPLETE,
                        handle=handle,
                        task=delegate.task_description,
                        success=True,
                        cost_usd=delegate.ctx.cost_used_usd,
                        tokens_used=delegate.ctx.tokens_used,
                    )
                self._completion_event.set()
                return result
            except Exception as exc:
                delegate = self._delegates[handle]
                delegate.status = TaskState.FAILED
                delegate.error = str(exc)
                # Write error to log file.
                if delegate.output_file is not None:
                    delegate.output_file.write_text(
                        f"handle: {handle}\n"
                        f"status: failed\n"
                        f"task: {delegate.task_description}\n"
                        f"error: {exc}\n"
                    )
                # Push failure notification with inline error.
                if self._notification_queue is not None:
                    self._notification_queue.put_nowait(
                        f"<delegate-result handle=\"{handle}\" "
                        f"agent_type=\"{getattr(delegate.ctx, 'metadata', {}).get('agent_type', 'unknown')}\" "
                        f"status=\"failed\">\n"
                        f"Task: {delegate.task_description}\n"
                        f"Error: {exc}\n"
                        f"</delegate-result>"
                    )
                if self._tracer is not None:
                    self._tracer.log_event(
                        TraceEventKind.ASYNC_DELEGATE_COMPLETE,
                        handle=handle,
                        task=delegate.task_description,
                        success=False,
                        error=str(exc),
                    )
                self._completion_event.set()
                # Don't re-raise — status and error are captured above,
                # and the notification is queued.  Re-raising would leave
                # an unhandled exception on the asyncio.Task.

        task = asyncio.create_task(_run_and_track())

        def _on_done(t: asyncio.Task, _handle: str = handle) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                import logging
                logging.getLogger("maike.orchestrator").debug(
                    "Background delegate %s failed: %s", _handle, exc,
                )

        task.add_done_callback(_on_done)
        delegate = AsyncDelegate(
            handle=handle,
            task_future=task,
            ctx=ctx,
            task_description=task_desc,
            started_at=datetime.now(timezone.utc),
            output_file=output_file,
        )
        self._delegates[handle] = delegate
        return delegate

    @property
    def running_count(self) -> int:
        """Number of currently running async delegates."""
        return sum(1 for d in self._delegates.values() if d.status == TaskState.RUNNING)

    @property
    def has_running_delegates(self) -> bool:
        """True if any delegates are still running."""
        return any(d.status == TaskState.RUNNING for d in self._delegates.values())

    async def wait_for_any_completion(self, timeout: float | None = None) -> bool:
        """Wait until any delegate completes. Returns True if signaled, False on timeout."""
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
            self._completion_event.clear()
            return True
        except asyncio.TimeoutError:
            return False

    # Full output is included inline up to this limit.  Above this,
    # a file path is returned instead so the parent can Read it.
    _INLINE_OUTPUT_LIMIT = 12_000  # chars (~3K tokens)

    def check(self, handle: str) -> dict:
        """Check status and progress of an async delegate."""
        if handle not in self._delegates:
            raise KeyError(f"Unknown delegate handle: {handle}")
        d = self._delegates[handle]
        progress = d.ctx.progress
        result: dict[str, Any] = {
            "handle": handle,
            "status": d.status.value,
            "task": d.task_description,
            "iterations": progress.iteration_count or getattr(d.ctx, "iterations_used", 0),
            "cost_usd": d.ctx.cost_used_usd,
            "tokens_used": d.ctx.tokens_used,
            "tool_use_count": progress.tool_use_count,
            "last_activity": progress.last_activity,
        }
        if d.status == TaskState.COMPLETED and d.result:
            raw = d.result.output or ""
            if len(raw) <= self._INLINE_OUTPUT_LIMIT:
                result["output"] = raw
            else:
                # Full output is in the log file — give the parent a path.
                result["output"] = raw[:self._INLINE_OUTPUT_LIMIT] + "\n...(truncated)"
                if d.output_file is not None:
                    result["output_file"] = str(d.output_file)
                    result["output"] += (
                        f"\n\nFull output ({len(raw)} chars) saved to: {d.output_file}\n"
                        f"Use Read(file_path=\"{d.output_file}\") to see the complete result."
                    )
        if d.status == TaskState.FAILED and d.error:
            result["error"] = d.error
        return result

    async def cleanup(self, grace_period: float = 15.0) -> None:
        """Gracefully shut down running delegates.

        1. Signal all running delegates to wrap up via their inbox.
        2. Wait up to *grace_period* seconds for natural completion.
        3. Hard-cancel anything still running.
        4. Persist any completed results.
        """
        running = [
            d for d in self._delegates.values()
            if d.status == TaskState.RUNNING and not d.task_future.done()
        ]
        if not running:
            return

        # Ask delegates to wrap up.
        for d in running:
            try:
                d.inbox.put_nowait(
                    "## System: Session ending\n\n"
                    "The session is shutting down. Produce a final summary "
                    "and stop immediately."
                )
            except Exception:
                pass

        # Wait for grace period — delegates that finish naturally are preserved.
        tasks = [d.task_future for d in running]
        try:
            await asyncio.wait(tasks, timeout=grace_period)
        except Exception:
            pass

        # Hard-cancel anything still running.
        for d in self._delegates.values():
            if d.status == TaskState.RUNNING and not d.task_future.done():
                d.task_future.cancel()
                try:
                    await asyncio.wait_for(d.task_future, timeout=3.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
                d.status = TaskState.FAILED
                d.error = "Cancelled on session cleanup (after grace period)"

        # Persist results from delegates that completed during grace period
        # (those not already persisted in _run_and_track).
        if self._session_store is not None and self._session_id is not None:
            for handle, d in self._delegates.items():
                if d.result is not None and handle not in self._persisted_handles:
                    try:
                        await self._session_store.store_agent_run(self._session_id, d.result)
                        self._persisted_handles.add(handle)
                    except Exception:
                        pass


async def _run_team_member(
    gateway,
    kwargs: dict,
    session,
    runtime,
    workspace,
    approval_gate,
    notification_queue,
    provider: str,
    agent_resolver,
):
    """Run a single team member delegate with its own LLMGateway.

    Extracted from ``delegate_handler`` so each team member can use an
    isolated gateway, enabling true parallel execution without corrupting
    the Gemini adapter's native history.
    """
    from maike.agents.core import AgentCore
    from maike.agents.delegate import build_delegate_context
    from maike.atoms.tool import ToolResult as TR
    from maike.constants import (
        DELEGATION_MAX_DEPTH,
        DELEGATION_MAX_ITERATIONS,
        model_for_tier,
    )
    from maike.memory.working import WorkingMemory
    from maike.safety.hooks import SafetyLayer
    from maike.tools.bash import register_bash_tools
    from maike.tools.edit import register_edit_tools
    from maike.tools.filesystem import register_filesystem_tools
    from maike.tools.registry import ToolRegistry
    from maike.tools.search import register_search_tools

    task = kwargs.get("task", "")
    context = kwargs.get("context", "")
    model_tier = kwargs.get("model_tier", "default")
    plugin_agent = kwargs.get("plugin_agent")
    agent_type = kwargs.get("agent_type", "implement")
    cost_budget = kwargs.get("remaining_cost_budget_usd", 1.0)
    token_budget = kwargs.get("remaining_token_budget", 500_000)

    # Resolve custom agent definition.
    _agent_def = None
    if plugin_agent and agent_resolver is not None:
        _agent_def = agent_resolver.resolve(plugin_agent)
        if _agent_def is None:
            available = agent_resolver.list_names()
            return TR(
                tool_name="Team", success=False,
                output=f"Agent '{plugin_agent}' not found. Available: {available}",
            )
        if model_tier == "default" and _agent_def.model_tier != "default":
            model_tier = _agent_def.model_tier

    delegate_model = model_for_tier(provider, model_tier)

    # Build tool registry.
    delegate_registry = ToolRegistry()
    register_filesystem_tools(delegate_registry, runtime)
    register_edit_tools(delegate_registry, runtime)
    register_bash_tools(delegate_registry, runtime)
    register_search_tools(delegate_registry, runtime)

    if _agent_def and _agent_def.allowed_tools:
        allowed = set(_agent_def.allowed_tools)
        delegate_registry._tools = {
            k: v for k, v in delegate_registry._tools.items()
            if k in allowed
        }
    if _agent_def and _agent_def.disallowed_tools:
        for tool_name in _agent_def.disallowed_tools:
            delegate_registry._tools.pop(tool_name, None)

    # Build context.
    if _agent_def:
        delegate_ctx, delegate_messages = await build_delegate_context(
            task=task, context=context, session=session,
            model=delegate_model,
            system_prompt_override=_agent_def.system_prompt,
            agent_type=agent_type,
            agent_def=_agent_def,
        )
    else:
        delegate_ctx, delegate_messages = await build_delegate_context(
            task=task, context=context, session=session,
            model=delegate_model,
            agent_type=agent_type,
        )

    delegate_ctx = delegate_ctx.model_copy(update={
        "token_budget": int(token_budget),
        "cost_budget_usd": cost_budget,
        "max_iterations": DELEGATION_MAX_ITERATIONS,
        "spawn_depth": DELEGATION_MAX_DEPTH,
    })

    # Run with per-member gateway.
    delegate_core = AgentCore(
        llm_gateway=gateway,
        tool_registry=delegate_registry,
        runtime=runtime,
        safety_layer=SafetyLayer(workspace),
        working_memory=WorkingMemory(),
        tracer=Tracer(),  # silent tracer to avoid cross-member noise
        approval_gate=approval_gate,
    )
    result = await delegate_core.run(delegate_ctx, delegate_messages)

    # Auto-retry on hallucination.
    _quality = assess_delegate_quality(result, agent_type)
    if _quality.is_hallucinated:
        try:
            retry_messages = delegate_messages + [{"role": "user", "content": (
                "Your previous response did not use any tools. "
                "You MUST use tools (Read, Grep, Bash, etc.) to complete "
                "the task. Do not guess or fabricate output."
            )}]
            result = await delegate_core.run(delegate_ctx, retry_messages)
            _quality = assess_delegate_quality(result, agent_type)
        except Exception:
            pass

    summary = result.output or "(no output)"
    return TR(
        tool_name="Team",
        success=result.success,
        output=summary + _quality.warning_text,
        metadata={
            "agent_id": result.agent_id,
            "cost_usd": result.cost_usd,
            "tokens_used": result.tokens_used,
            "quality": _quality.quality_flag,
        },
    )


class Orchestrator:
    def __init__(
        self,
        *,
        base_path: Path,
        llm_gateway: LLMGateway | None = None,
        cost_tracker: CostTracker | None = None,
        tracer: Tracer | None = None,
        parallel_coding_enabled: bool = False,
        dynamic_agents_enabled: bool = False,
        stream_sink: Callable | None = None,
    ) -> None:
        self.base_path = base_path
        if cost_tracker is not None:
            self.cost_tracker = cost_tracker
        elif llm_gateway is not None and hasattr(llm_gateway, "cost_tracker"):
            self.cost_tracker = llm_gateway.cost_tracker
        else:
            self.cost_tracker = CostTracker()
        if tracer is not None:
            self.tracer = tracer
        elif llm_gateway is not None and hasattr(llm_gateway, "tracer"):
            self.tracer = llm_gateway.tracer
        else:
            self.tracer = Tracer()
        self.llm_gateway = llm_gateway
        self.parallel_coding_enabled = parallel_coding_enabled
        self.dynamic_agents_enabled = dynamic_agents_enabled
        self._stream_sink = stream_sink
        self._async_delegate_manager = AsyncDelegateManager()
        self.session_store = SessionStore(base_path)

    def _build_advisor_session(
        self,
        *,
        enabled: bool,
        executor_provider: str,
        advisor_provider: str | None,
        advisor_model: str | None,
        budget: float | None,
        budget_pct: float | None,
    ):
        """Construct an AdvisorSession with its own silent LLMGateway.

        Returns None if advisor is disabled.  Kept private because callers
        shouldn't need to know about the gateway details.
        """
        from maike.agents.advisor import AdvisorSession, resolve_advisor_config
        from maike.constants import ADVISOR_BUDGET_FRACTION_DEFAULT

        fraction = (
            budget_pct
            if budget_pct is not None
            else ADVISOR_BUDGET_FRACTION_DEFAULT
        )
        config = resolve_advisor_config(
            enabled=enabled,
            executor_provider=executor_provider,
            advisor_provider=advisor_provider,
            advisor_model=advisor_model,
            session_budget_usd=budget,
            budget_fraction=fraction,
        )
        if not config.enabled:
            return AdvisorSession(gateway=None, config=config)

        advisor_gateway = LLMGateway(
            self.cost_tracker,
            self.tracer,
            provider_name=config.provider,
            silent=True,
        )
        return AdvisorSession(gateway=advisor_gateway, config=config)

    def _build_trajectory_auditor(
        self,
        *,
        bg_gateway: Any,
        provider: str,
        session_budget: float | None,
    ):
        """Construct a ``TrajectoryAuditor`` for the current session.

        Returns ``None`` when disabled — skipped entirely at the AgentCore
        hook point (strictly additive).  **Default OFF** pending wider eval
        validation — one targeted Pro-solo trial on sympy-19254 regressed
        with the auditor infrastructure active (v3 80-line diff → v4 0
        diff), even though no auditor events actually fired.  Likely Gemini
        Pro run variance, but keeping default OFF until we have statistical
        confidence.  Opt in via ``MAIKE_TRAJECTORY_AUDITOR_ENABLED=1``.
        """
        import os as _os
        enabled_env = _os.getenv("MAIKE_TRAJECTORY_AUDITOR_ENABLED", "0").strip().lower()
        if enabled_env not in {"1", "true", "yes", "on"}:
            return None
        # Budget — 5% of session budget (floor 0.01 so a single cheap call
        # can always land even on tiny budgets).
        try:
            fraction_env = float(
                _os.getenv("MAIKE_TRAJECTORY_AUDITOR_BUDGET_FRACTION", "0.05")
            )
        except ValueError:
            fraction_env = 0.05
        auditor_budget = 0.0
        if session_budget and session_budget > 0:
            auditor_budget = max(session_budget * fraction_env, 0.01)

        try:
            from maike.agents.trajectory_auditor import (
                AuditorConfig, TrajectoryAuditor,
            )
        except Exception:
            return None

        return TrajectoryAuditor(
            gateway=bg_gateway,
            provider=provider,
            config=AuditorConfig(
                enabled=True,
                budget_usd=auditor_budget,
                max_calls=6,
                cooldown_iterations=5,
                timeout_s=10.0,
            ),
            tracer=self.tracer,
        )

    def _make_advisor_handler(self, advisor_session):
        """Build the Advisor tool handler closure bound to *advisor_session*.

        The handler reads the live conversation + iteration count from the
        ``CURRENT_AGENT_LOOP_STATE`` ContextVar published by ``AgentCore.run``.
        Returns a simple ToolResult with the advice text (or a throttle note).
        """
        from maike.agents.advisor import AdvisorTrigger, AdvisorUrgency
        from maike.atoms.tool import ToolResult
        from maike.tools.context import peek_current_loop_state

        async def advisor_handler(*, question: str, urgency: str) -> ToolResult:
            loop_state = peek_current_loop_state()
            if loop_state is None or loop_state.agent_context is None:
                return ToolResult(
                    tool_name="Advisor",
                    success=False,
                    output=(
                        "Advisor invoked outside an active agent turn "
                        "— no conversation state to snapshot."
                    ),
                )
            urgency_enum = (
                AdvisorUrgency.STUCK if urgency == "stuck" else AdvisorUrgency.NORMAL
            )
            verdict = await advisor_session.advise(
                question=question,
                urgency=urgency_enum,
                trigger=AdvisorTrigger.TOOL,
                conversation=loop_state.conversation,
                ctx=loop_state.agent_context,
                iteration_count=loop_state.iteration_count,
            )
            if verdict.throttled:
                reason = verdict.throttle_reason
                # Give the executor a non-scary, actionable fallback message.
                msg_map = {
                    "disabled": "Advisor is not enabled in this session.",
                    "budget_exhausted": (
                        "Advisor budget exhausted — proceed with your own judgement."
                    ),
                    "max_calls_reached": (
                        "Advisor call limit reached for this session — proceed on your own."
                    ),
                }
                output = msg_map.get(
                    reason,
                    f"Advisor skipped: {reason}. Continue with your current plan.",
                )
                # cooldown: include remaining iterations count
                if reason.startswith("cooldown"):
                    output = (
                        "Advisor is on cooldown — wait a few iterations before asking again. "
                        "Continue with your current plan."
                    )
                return ToolResult(
                    tool_name="Advisor",
                    success=True,  # not an error — just throttled
                    output=output,
                    metadata={
                        "throttled": True,
                        "throttle_reason": reason,
                    },
                )
            advisor_session.record_verdict(verdict, loop_state.iteration_count)
            return ToolResult(
                tool_name="Advisor",
                success=True,
                output=verdict.advice,
                metadata={
                    "advisor_cost_usd": verdict.cost_usd,
                    "advisor_tokens": verdict.tokens_used,
                    "advisor_trigger": verdict.trigger.value,
                    "advisor_urgency": verdict.urgency.value,
                },
            )

        return advisor_handler

    async def run(
        self,
        task: str,
        workspace: Path,
        *,
        provider_name: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        runtime_config: RuntimeConfig | None = None,
        language_override: str | None = None,
        budget: float | None = DEFAULT_RUN_BUDGET_USD,
        agent_token_budget: int | None = None,
        auto_approve: bool = False,
        verbose: bool = False,
        session_id: str | None = None,
        cancellation: CancellationController | None = None,
        thread_id: str | None = None,
        new_thread: bool = False,
        adaptive_model: bool = True,
        advisor_enabled: bool = False,
        advisor_provider: str | None = None,
        advisor_model: str | None = None,
        advisor_budget_pct: float | None = None,
    ) -> OrchestratorResult:
        async with self.session_store.use_shared_connection():
            self.cost_tracker.reset(session_budget_usd=budget)
            resolved_provider = resolve_provider_name(provider_name=provider_name, model=model).value
            resolved_model = resolve_model_name(resolved_provider, model)
            manifest = EnvironmentProbe().resolve(
                workspace,
                language_override=language_override,
                default_config=runtime_config,
            )
            runtime: ExecutionRuntime = LocalRuntime(workspace, manifest=manifest)
            on_prompt = getattr(self.tracer.sink, "on_prompt", None)
            on_text_prompt = getattr(self.tracer.sink, "on_text_prompt", None)
            approval_gate = ApprovalGate(
                auto_approve=auto_approve,
                on_prompt=on_prompt,
                on_text_prompt=on_text_prompt,
            )

            # Bootstrap: ensure venv exists and deps are installed before
            # agents run.  Re-creates the runtime with the updated manifest
            # if anything changed (e.g., venv was created).
            from maike.runtime.bootstrap import (
                BootstrapPolicy,
                DependencyBootstrapper,
            )
            bootstrapper = DependencyBootstrapper(runtime, approval_gate)
            boot_policy = BootstrapPolicy(require_approval=not auto_approve)
            boot_result = await bootstrapper.bootstrap(
                workspace, manifest, boot_policy
            )
            if boot_result.success and boot_result.action_taken != "none":
                manifest = boot_result.manifest
                runtime = LocalRuntime(workspace, manifest=manifest)

            await PreflightChecker(runtime, approval_gate).ensure_ready(resolved_provider)

            session_metadata = self._build_session_metadata(
                provider=resolved_provider,
                model=resolved_model,
                budget=budget,
                agent_token_budget=agent_token_budget,
                language_override=language_override,
                auto_approve=auto_approve,
            )

            # Thread resolution — determine which thread this session belongs to.
            resolved_thread = await self._resolve_thread(
                workspace, task, thread_id=thread_id, new_thread=new_thread,
            )
            if resolved_thread is not None:
                session_metadata["thread_id"] = resolved_thread["id"]
                session_metadata["thread_name"] = resolved_thread["name"]

            resume_checkpoint = None
            if session_id is None:
                session_id = await self.session_store.create_session(
                    task,
                    workspace,
                    metadata=session_metadata,
                )
            else:
                existing_session = await self.session_store.get_session(session_id)
                if existing_session is None:
                    raise OrchestratorError(f"Session not found: {session_id}")
                existing_status = str(existing_session.get("status") or "")
                if existing_status == "completed":
                    raise OrchestratorError(f"Session already completed: {session_id}")
                if existing_status == "failed":
                    raise OrchestratorError(f"Failed sessions cannot be resumed: {session_id}")
                existing_workspace = Path(str(existing_session["workspace"])).resolve()
                if existing_workspace != workspace.resolve():
                    raise OrchestratorError(
                        f"Session workspace mismatch: expected {existing_workspace}, got {workspace.resolve()}"
                    )
                task = str(existing_session["task"])
                resume_checkpoint = await self.session_store.get_latest_checkpoint(session_id)
                if resume_checkpoint is None:
                    raise OrchestratorError(f"Session has no checkpoint to resume from: {session_id}")
                await self.session_store.mark_session_status(session_id, "running")

            long_term_memory = create_long_term_memory(workspace)
            learner = SessionLearner(long_term_memory)
            from maike.memory.longterm import create_typed_memory
            typed_memory = create_typed_memory(workspace)

            session = OrchestratorSession(
                self.session_store,
                session_id,
                task,
                workspace,
                environment_manifest=manifest,
                learner=learner,
                typed_memory=typed_memory,
                thread=resolved_thread,
            )

            session.skill_dirs = self._resolve_skill_dirs(workspace)
            session.plugin_skills = self._discover_plugin_skills(workspace)
            session.plugin_bundle = self._discover_plugin_bundle(workspace)

            # Agent resolver — unifies plugin, user, and project agents.
            from maike.agents.agent_resolver import AgentResolver
            from maike.constants import AGENTS_PROJECT_SUBDIR, AGENTS_USER_DIR
            _plugin_agents = session.plugin_bundle.agents if session.plugin_bundle else []
            self._agent_resolver = AgentResolver(
                plugin_agents=_plugin_agents,
                user_dir=AGENTS_USER_DIR,
                project_dir=workspace / AGENTS_PROJECT_SUBDIR,
            )
            session.agent_resolver = self._agent_resolver

            # Team resolver — discovers team definitions.
            from maike.agents.team_resolver import TeamResolver
            from maike.constants import TEAMS_PROJECT_SUBDIR, TEAMS_USER_DIR
            self._team_resolver = TeamResolver(
                user_dir=TEAMS_USER_DIR,
                project_dir=workspace / TEAMS_PROJECT_SUBDIR,
            )
            session.team_resolver = self._team_resolver

            # Background task notification queue and output directory.
            notification_queue: asyncio.Queue = asyncio.Queue()
            bg_output_dir = workspace / ".maike" / "bg" / session.id
            bg_output_dir.mkdir(parents=True, exist_ok=True)
            # Fresh manager per session to avoid cross-session state leakage.
            self._async_delegate_manager = AsyncDelegateManager(
                output_dir=bg_output_dir,
                notification_queue=notification_queue,
                session_store=self.session_store,
                session_id=session.id,
                tracer=self.tracer,
            )

            # Build code index for existing codebases (best-effort).
            from maike.agents.react import _has_prior_work
            if _has_prior_work(workspace):
                try:
                    from maike.intelligence.code_index import CodeIndex
                    code_index = CodeIndex(workspace, session.id)
                    await code_index.build()
                    session.code_index = code_index
                except Exception:
                    pass  # graceful degradation

            # Inject output dir and notification queue into runtime's background manager.
            if hasattr(runtime, 'background_manager'):
                runtime.background_manager._output_dir = bg_output_dir
                runtime.background_manager._notification_queue = notification_queue

            await self._raise_if_cancelled(session=session, cancellation=cancellation)
            created_llm_gateway = self.llm_gateway is None
            llm_gateway = self.llm_gateway or LLMGateway(
                self.cost_tracker,
                self.tracer,
                provider_name=resolved_provider,
            )
            # Separate silent gateway for background cheap-model calls
            # (session memory, constraints).  Own adapter avoids corrupting
            # Gemini native history; silent=True suppresses TUI noise.
            _bg_gateway = LLMGateway(
                self.cost_tracker, self.tracer,
                provider_name=resolved_provider,
                silent=True,
            )

            # Advisor gateway — opt-in via --advisor.  Mirrors _bg_gateway
            # pattern (own adapter, silent=True).  See maike/agents/advisor.py.
            _advisor_session = self._build_advisor_session(
                enabled=advisor_enabled,
                executor_provider=resolved_provider,
                advisor_provider=advisor_provider,
                advisor_model=advisor_model,
                budget=budget,
                budget_pct=advisor_budget_pct,
            )
            session.advisor_session = _advisor_session

            # Live session memory — maintained by background LLM calls.
            from maike.memory.session_memory import SessionMemoryService
            _session_memory = SessionMemoryService(
                workspace=workspace,
                gateway=_bg_gateway,
            ) if workspace else None

            stage_results: dict[str, list[AgentResult]] = {}
            _run_outcome = "unknown"

            try:
                tool_registry = ToolRegistry()

                async def user_input_handler(prompt: str, context_summary: str = ""):
                    """Pause execution and ask the user a question.

                    AskUser is the agent's channel for substantive questions
                    that need a real answer (e.g. "should this use Postgres
                    or SQLite?"). It is intentionally exempt from --yes /
                    auto-approve — bypassing the question with a canned
                    response defeats its purpose. ApprovalGate.prompt_text
                    handles the no-TTY / non-interactive fallback.
                    """
                    from maike.atoms.tool import ToolResult

                    full_prompt = prompt
                    if context_summary:
                        full_prompt = f"[Context: {context_summary}]\n{prompt}"
                    response = await approval_gate.prompt_text(full_prompt)
                    return ToolResult(
                        tool_name="AskUser",
                        success=True,
                        output=response,
                        metadata={"user_response": response},
                    )

                async def _build_delegate_agent(
                    *,
                    task: str,
                    context: str,
                    model_tier: str,
                    parent_agent_id: str,
                    remaining_token_budget: int,
                    remaining_cost_budget_usd: float,
                    agent_type: str = "implement",
                ):
                    """Shared setup: build context, registry, and AgentCore for a delegate."""
                    from maike.agents.delegate import build_delegate_context
                    from maike.constants import (
                        DELEGATION_BUDGET_CAP_USD,
                        DELEGATION_BUDGET_FRACTION,
                        DELEGATION_MAX_DEPTH,
                        DELEGATION_MAX_ITERATIONS,
                        DELEGATION_TOKEN_BUDGET,
                        model_for_tier,
                    )
                    from maike.tools.bash import register_bash_tools
                    from maike.tools.edit import register_edit_tools
                    from maike.tools.filesystem import register_filesystem_tools
                    from maike.tools.search import register_search_tools

                    delegate_model = model_for_tier(resolved_provider, model_tier)
                    cost_budget = min(
                        remaining_cost_budget_usd * DELEGATION_BUDGET_FRACTION,
                        DELEGATION_BUDGET_CAP_USD,
                    )
                    token_budget = min(remaining_token_budget, DELEGATION_TOKEN_BUDGET)

                    delegate_ctx, delegate_messages = await build_delegate_context(
                        task=task,
                        context=context,
                        session=session,
                        parent_id=parent_agent_id,
                        model=delegate_model,
                        agent_type=agent_type,
                    )
                    delegate_ctx = delegate_ctx.model_copy(
                        update={
                            "token_budget": token_budget,
                            "cost_budget_usd": cost_budget,
                            "max_iterations": DELEGATION_MAX_ITERATIONS,
                            "spawn_depth": DELEGATION_MAX_DEPTH,
                        }
                    )

                    delegate_registry = ToolRegistry()
                    register_filesystem_tools(delegate_registry, runtime, session=self.session_store)
                    register_edit_tools(delegate_registry, runtime, session=self.session_store)
                    register_bash_tools(delegate_registry, runtime)
                    register_search_tools(delegate_registry, runtime)

                    delegate_core = AgentCore(
                        llm_gateway=llm_gateway,
                        tool_registry=delegate_registry,
                        runtime=runtime,
                        safety_layer=SafetyLayer(workspace),
                        working_memory=WorkingMemory(),
                        tracer=self.tracer,
                        approval_gate=approval_gate,
                        notification_queue=notification_queue,
                    )

                    return delegate_ctx, delegate_messages, delegate_core, cost_budget, token_budget

                async def delegate_handler(
                    *,
                    task: str,
                    context: str = "",
                    model_tier: str = "default",
                    parent_agent_id: str,
                    parent_stage_name: str,
                    remaining_token_budget: int,
                    remaining_cost_budget_usd: float,
                ):
                    """Spawn a sub-agent with a fresh context to complete a task."""

                    delegate_ctx, delegate_messages, delegate_core, cost_budget, token_budget = (
                        await _build_delegate_agent(
                            task=task,
                            context=context,
                            model_tier=model_tier,
                            parent_agent_id=parent_agent_id,
                            remaining_token_budget=remaining_token_budget,
                            remaining_cost_budget_usd=remaining_cost_budget_usd,
                        )
                    )

                    self.tracer.log_event(
                        TraceEventKind.DELEGATE_SPAWN,
                        parent_agent_id=parent_agent_id,
                        delegate_agent_id=delegate_ctx.agent_id,
                        task=task[:200],
                        cost_budget=cost_budget,
                        token_budget=token_budget,
                    )

                    try:
                        result = await delegate_core.run(delegate_ctx, delegate_messages)
                    except Exception as exc:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=f"Delegation failed: {exc}",
                            error=str(exc),
                            metadata={"agent_id": delegate_ctx.agent_id},
                        )

                    # Auto-retry once on hallucinated output.
                    _quality = assess_delegate_quality(result, "implement")
                    if _quality.is_hallucinated:
                        try:
                            retry_messages = delegate_messages + [_HALLUCINATION_RETRY_NUDGE]
                            result = await delegate_core.run(delegate_ctx, retry_messages)
                            _quality = assess_delegate_quality(result, "implement")
                        except Exception:
                            pass  # keep original result

                    self.tracer.log_event(
                        TraceEventKind.DELEGATE_COMPLETE,
                        delegate_agent_id=delegate_ctx.agent_id,
                        success=result.success,
                        cost_usd=result.cost_usd,
                        tokens_used=result.tokens_used,
                    )

                    summary = result.output or "(no output from delegate)"
                    return TR(
                        tool_name="Delegate",
                        success=result.success,
                        output=summary + _quality.warning_text,
                        metadata={
                            "agent_id": delegate_ctx.agent_id,
                            "cost_usd": result.cost_usd,
                            "tokens_used": result.tokens_used,
                            "iterations": result.metadata.get("iteration_count", 0),
                            "quality": _quality.quality_flag,
                        },
                    )

                async def async_delegate_handler(
                    *,
                    task: str,
                    context: str = "",
                    model_tier: str = "default",
                    parent_agent_id: str,
                    parent_stage_name: str,
                    remaining_token_budget: int,
                    remaining_cost_budget_usd: float,
                ):
                    """Spawn a sub-agent in the background (non-blocking)."""
                    from maike.atoms.tool import ToolResult as TR
                    from maike.constants import MAX_ASYNC_DELEGATES

                    # Enforce concurrency limit.
                    running = self._async_delegate_manager.running_count
                    if running >= MAX_ASYNC_DELEGATES:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=(
                                f"Cannot spawn: {running}/{MAX_ASYNC_DELEGATES} async delegates running. "
                                f"Wait for a running delegate to finish before spawning more. "
                                f"Use Delegate(action='wait', handle='delegate-NNN') to block "
                                f"until one completes, then retry."
                            ),
                            error="max_async_delegates_exceeded",
                        )

                    delegate_ctx, delegate_messages, delegate_core, cost_budget, token_budget = (
                        await _build_delegate_agent(
                            task=task,
                            context=context,
                            model_tier=model_tier,
                            parent_agent_id=parent_agent_id,
                            remaining_token_budget=remaining_token_budget,
                            remaining_cost_budget_usd=remaining_cost_budget_usd,
                        )
                    )

                    self.tracer.log_event(
                        TraceEventKind.ASYNC_DELEGATE_SPAWN,
                        parent_agent_id=parent_agent_id,
                        delegate_agent_id=delegate_ctx.agent_id,
                        task=task[:200],
                        cost_budget=cost_budget,
                        token_budget=token_budget,
                    )

                    # Per-delegate inbox for parent→delegate messaging.
                    _inbox = asyncio.Queue()
                    delegate_core._inbox_queue = _inbox

                    # Wrap with auto-retry on hallucination.
                    async def _run_with_retry_run(
                        _core=delegate_core, _ctx=delegate_ctx,
                        _msgs=delegate_messages,
                    ):
                        result = await _core.run(_ctx, _msgs)
                        _q = assess_delegate_quality(result, "implement")
                        if _q.is_hallucinated:
                            try:
                                retry_msgs = _msgs + [_HALLUCINATION_RETRY_NUDGE]
                                result = await _core.run(_ctx, retry_msgs)
                            except Exception:
                                pass
                        return result

                    coro = _run_with_retry_run()
                    delegate = await self._async_delegate_manager.start(
                        coro, delegate_ctx, task[:200],
                    )
                    delegate.inbox = _inbox

                    output_parts = [f"Async delegate spawned: {delegate.handle} — {task[:200]}"]
                    if delegate.output_file is not None:
                        output_parts.append(f"Output file: {delegate.output_file}")
                    return TR(
                        tool_name="Delegate",
                        success=True,
                        output="\n".join(output_parts),
                        metadata={
                            "handle": delegate.handle,
                            "agent_id": delegate_ctx.agent_id,
                            "output_file": str(delegate.output_file) if delegate.output_file else None,
                        },
                    )

                from maike.agents.skill import SkillLoader
                _skill_dirs = getattr(session, "skill_dirs", None)
                _plugin_skills = getattr(session, "plugin_skills", None) or []
                _skill_loader: SkillLoader | None = None
                if _skill_dirs:
                    _skill_loader = SkillLoader(
                        builtin_dir=_skill_dirs.builtin,
                        user_dir=_skill_dirs.user,
                        project_dir=_skill_dirs.project,
                        extra_skills=_plugin_skills,
                    )
                else:
                    _skill_loader = SkillLoader(extra_skills=_plugin_skills)

                async def delegate_check_handler(*, handle: str) -> TR:
                    try:
                        status = self._async_delegate_manager.check(handle)
                        return TR(
                            tool_name="Delegate",
                            success=True,
                            output=json.dumps(status, indent=2, default=str),
                            metadata=status,
                        )
                    except KeyError:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=f"Unknown delegate handle: {handle}",
                        )

                async def delegate_wait_handler(*, handle: str) -> TR:
                    delegates = self._async_delegate_manager._delegates
                    if handle not in delegates:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=f"Unknown delegate handle: {handle}",
                        )
                    delegate = delegates[handle]
                    try:
                        await delegate.task_future
                    except Exception:
                        pass  # Status captured by _run_and_track
                    status = self._async_delegate_manager.check(handle)
                    return TR(
                        tool_name="Delegate",
                        success=True,
                        output=json.dumps(status, indent=2, default=str),
                        metadata=status,
                    )

                async def delegate_send_handler(*, handle: str, message: str):
                    from maike.atoms.tool import ToolResult as TR
                    delegates = self._async_delegate_manager._delegates
                    if handle not in delegates:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=f"Unknown delegate handle: {handle}",
                        )
                    d = delegates[handle]
                    if d.status != TaskState.RUNNING:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=f"Cannot send to {handle}: delegate is {d.status.value}.",
                        )
                    d.inbox.put_nowait(
                        f"## Message from parent agent\n\n{message}"
                    )
                    return TR(
                        tool_name="Delegate",
                        success=True,
                        output=f"Message sent to {handle}.",
                        metadata={"handle": handle},
                    )

                # Advisor tool handler — only wired when --advisor is enabled.
                advisor_handler = None
                if _advisor_session.enabled:
                    advisor_handler = self._make_advisor_handler(_advisor_session)

                register_default_tools(
                    tool_registry,
                    runtime,
                    session=self.session_store,
                    delegate_handler=delegate_handler,
                    user_input_handler=user_input_handler,
                    async_delegate_handler=async_delegate_handler,
                    delegate_check_handler=delegate_check_handler,
                    delegate_wait_handler=delegate_wait_handler,
                    delegate_send_handler=delegate_send_handler,
                    advisor_handler=advisor_handler,
                    skill_loader=_skill_loader,
                )

                # Start MCP servers and register their tools.
                bundle = session.plugin_bundle
                if bundle and bundle.mcp_configs:
                    from maike.plugins.mcp_registry import MCPToolRegistry
                    mcp_reg = MCPToolRegistry()
                    await mcp_reg.start_servers(bundle.mcp_configs)
                    await mcp_reg.register_tools(tool_registry)
                    session.mcp_registry = mcp_reg

                # Initialize hook executor and fire SessionStart.
                if bundle and bundle.hooks and bundle.hooks.hooks:
                    from maike.plugins.hook_executor import HookExecutor
                    session.hook_executor = HookExecutor(
                        bundle.hooks,
                        env={"MAIKE_WORKSPACE": str(workspace), "MAIKE_SESSION_ID": session.id},
                    )
                    from maike.plugins.hooks import HookEvent
                    await session.hook_executor.fire(
                        HookEvent.SESSION_START,
                        context={"session_id": session.id, "workspace": str(workspace), "task": task},
                    )

                # Start LSP servers.
                if bundle and bundle.lsp_configs:
                    from maike.plugins.lsp_manager import LSPManager
                    lsp_mgr = LSPManager(workspace_root=str(workspace))
                    await lsp_mgr.start_servers(bundle.lsp_configs)
                    session.lsp_manager = lsp_mgr

                # React checkpoint callback so AgentCore can persist
                # incremental progress after each file write.
                async def _react_checkpoint(_ctx):
                    try:
                        ckpt = await runtime.checkpoint(
                            label="react-progress", step="react",
                        )
                        await session.store_checkpoint(ckpt)
                    except Exception:
                        pass  # best-effort

                _sm_path = _session_memory.memory_path if _session_memory else None
                from maike.context.recovery import PostCompactionRecovery
                _recovery = PostCompactionRecovery(workspace=workspace)
                agent_core = AgentCore(
                    llm_gateway=llm_gateway,
                    tool_registry=tool_registry,
                    runtime=runtime,
                    safety_layer=SafetyLayer(workspace),
                    working_memory=WorkingMemory(session_memory_path=_sm_path),
                    tracer=self.tracer,
                    approval_gate=approval_gate,
                    checkpoint_callback=_react_checkpoint,
                    # Don't drain delegate notifications mid-turn — results
                    # are handled at turn boundary by auto-resume.  Mid-turn
                    # draining causes the agent to re-spawn delegates for
                    # directories already covered.
                    notification_queue=None,
                    stream_sink=self._stream_sink,
                    session_memory=_session_memory,
                    compaction_recovery=_recovery,
                )
                agent_core._bg_gateway = _bg_gateway
                agent_core._advisor_session = _advisor_session
                agent_core._trajectory_auditor = self._build_trajectory_auditor(
                    bg_gateway=_bg_gateway, provider=resolved_provider,
                    session_budget=budget,
                )

                # Initialize per-turn memory surfacer for typed memories.
                _memory_dir = workspace / ".maike" / "memories"
                if _memory_dir.exists():
                    from maike.memory.surfacer import MemorySurfacer
                    agent_core._memory_surfacer = MemorySurfacer(_memory_dir)

                # Resume: restore workspace from checkpoint so the agent
                # sees its prior work, then re-run.
                if resume_checkpoint is not None:
                    await runtime.restore(resume_checkpoint)

                from maike.agents.react import build_react_context

                # Set context variables BEFORE build_react_context so
                # skill injection (peek_current_skill_loader) works.
                code_index_token = CURRENT_CODE_INDEX.set(session.code_index)
                skill_loader_token = CURRENT_SKILL_LOADER.set(_skill_loader)

                # Extract task constraints: MAIKE.md protected files (static)
                # + markdown task-rules block (LLM, prompt-only, no hard
                # enforcement).  The markdown is injected verbatim into the
                # react agent's prompt; the read-only patterns feed a
                # post-hoc "you mutated a protected file" check in core.py.
                from maike.agents.constraints import build_task_constraints
                _constraints_md, _read_only_patterns = await build_task_constraints(
                    task=task, workspace=workspace,
                    session_bg_gateway=_bg_gateway,
                    session_provider=resolved_provider,
                )
                # Visible to build_react_context for <maike-constraints>.
                session.task_constraints_markdown = _constraints_md

                effective_token_budget = agent_token_budget or 0
                ctx, messages = await build_react_context(
                    task, session, model=resolved_model,
                    thread=resolved_thread,
                    adaptive_model=adaptive_model,
                )
                updates: dict = {"max_iterations": DEFAULT_REACT_MAX_ITERATIONS}
                if effective_token_budget:
                    updates["token_budget"] = effective_token_budget
                _md = dict(ctx.metadata)
                if _read_only_patterns:
                    # Feeds the post-hoc read-only check in agents/core.py.
                    _md["read_only_patterns"] = list(_read_only_patterns)
                if verbose:
                    _md["verbose_trace"] = True
                updates["metadata"] = _md
                ctx = ctx.model_copy(update=updates)
                from maike.tools.context import CURRENT_HOOK_EXECUTOR, CURRENT_LSP_MANAGER
                hook_executor_token = CURRENT_HOOK_EXECUTOR.set(session.hook_executor)
                lsp_manager_token = CURRENT_LSP_MANAGER.set(session.lsp_manager)
                try:
                    # Partition mode: if --parallel-coding is set, try to
                    # decompose the task into independent file-scoped
                    # partitions and run them concurrently.
                    partition_plan = None
                    if self.parallel_coding_enabled and resume_checkpoint is None:
                        partition_plan = await self._plan_partitions(
                            task, workspace, llm_gateway, resolved_model,
                        )
                    if partition_plan is not None:
                        partition_results = await self._run_partitioned(
                            task=task,
                            partition_plan=partition_plan,
                            session=session,
                            workspace=workspace,
                            runtime=runtime,
                            llm_gateway=llm_gateway,
                            approval_gate=approval_gate,
                            notification_queue=notification_queue,
                            model=resolved_model,
                            budget=budget or 5.0,
                        )
                        stage_results["react"] = partition_results
                        result = partition_results[-1] if partition_results else None
                    else:
                        result = await agent_core.run(ctx, messages)
                        stage_results["react"] = [result]
                finally:
                    CURRENT_SKILL_LOADER.reset(skill_loader_token)
                    CURRENT_CODE_INDEX.reset(code_index_token)
                    CURRENT_HOOK_EXECUTOR.reset(hook_executor_token)
                    CURRENT_LSP_MANAGER.reset(lsp_manager_token)

                if result is not None:
                    await self.session_store.store_agent_run(session.id, result)

                _run_outcome = "success"
                await self.session_store.mark_session_status(session.id, "completed")

                pipeline_name = self._pipeline_name()
                await self._record_session_learnings(
                    session=session,
                    pipeline_name=pipeline_name,
                    stage_results=stage_results,
                    outcome="success",
                )
                _thread_id = resolved_thread["id"] if resolved_thread else None
                return OrchestratorResult(session_id=session.id, pipeline=pipeline_name, stage_results=stage_results, thread_id=_thread_id)
            except OrchestratorCancelled:
                _run_outcome = "cancelled — session limit reached"
                await self._record_session_learnings(
                    session=session,
                    pipeline_name=self._pipeline_name(),
                    stage_results=stage_results,
                    outcome="cancelled",
                )
                raise
            except (KeyboardInterrupt, asyncio.CancelledError):
                _run_outcome = "cancelled — interrupted by user"
                await self.session_store.mark_session_status(session.id, "cancelled")
                raise
            except Exception as exc:
                _err_msg = str(exc).strip()[:200] or type(exc).__name__
                _run_outcome = f"failure — {_err_msg}"
                await self.session_store.mark_session_status(session.id, "failed")
                await self._record_session_learnings(
                    session=session,
                    pipeline_name=self._pipeline_name(),
                    stage_results=stage_results,
                    outcome="failure",
                )
                raise OrchestratorError(f"react failed: {exc}") from exc
            finally:
                # Wait for any in-flight session memory update before summary.
                if _session_memory is not None:
                    await _session_memory.wait_for_pending()

                # Generate session summary — runs on ANY exit path.
                react_results = stage_results.get("react", [])
                _last_run = react_results[-1] if react_results else None
                _msgs = _last_run.messages if _last_run and hasattr(_last_run, "messages") else []
                _agent_output_run = getattr(_last_run, "output", None)
                # Iteration-cap detection: AgentCore sets this in result.metadata
                # when max_iterations triggers termination.
                _iter_cap_run = bool(
                    _last_run
                    and getattr(_last_run, "metadata", {}).get("termination_reason") == "max_iterations"
                )
                _thread_summary_run = await self._write_session_summary(
                    thread=resolved_thread,
                    messages=_msgs,
                    task=task,
                    outcome=_run_outcome,
                    session_id=session.id,
                    workspace=workspace,
                    agent_output=_agent_output_run,
                    budget_usd=budget,
                    iteration_cap_hit=_iter_cap_run,
                    bg_gateway=_bg_gateway,
                    provider=resolved_provider,
                )

                # Extract durable learnings into persistent auto-memory.
                # Synchronous — no LLM call, immune to async cancellation.
                self._extract_auto_memories(
                    session=session,
                    session_memory_service=_session_memory,
                    agent_output=_agent_output_run,
                    thread_summary=_thread_summary_run,
                    task=task,
                )

                # Clean up async delegates and background processes.
                await self._async_delegate_manager.cleanup()
                if hasattr(runtime, 'background_manager'):
                    await runtime.background_manager.cleanup()
                # Shutdown plugin servers.
                if session.mcp_registry is not None:
                    await session.mcp_registry.shutdown()
                if session.lsp_manager is not None:
                    await session.lsp_manager.shutdown()
                if created_llm_gateway:
                    await llm_gateway.aclose()
                await _bg_gateway.aclose()

    async def run_interactive(
        self,
        initial_task: str,
        workspace: Path,
        *,
        provider_name: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        runtime_config: RuntimeConfig | None = None,
        language_override: str | None = None,
        budget: float | None = DEFAULT_RUN_BUDGET_USD,
        agent_token_budget: int | None = None,
        auto_approve: bool = False,
        verbose: bool = False,
        cancellation: CancellationController | None = None,
        thread_id: str | None = None,
        new_thread: bool = False,
        turn_callback: Callable | None = None,
        adaptive_model: bool = True,
        tool_profile_override: str | None = None,
        advisor_enabled: bool = False,
        advisor_provider: str | None = None,
        advisor_model: str | None = None,
        advisor_budget_pct: float | None = None,
    ) -> OrchestratorResult:
        """Run in interactive mode — multi-turn conversation within one session.

        After each agent turn, ``turn_callback(result)`` is called.  It should
        return a follow-up task string, or ``None`` to end the session.
        If ``turn_callback`` is None, behaves like single-turn (returns after
        one agent run).
        """
        async with self.session_store.use_shared_connection():
            self.cost_tracker.reset(session_budget_usd=budget)
            resolved_provider = resolve_provider_name(provider_name=provider_name, model=model).value
            resolved_model = resolve_model_name(resolved_provider, model)
            manifest = EnvironmentProbe().resolve(
                workspace,
                language_override=language_override,
                default_config=runtime_config,
            )
            runtime: ExecutionRuntime = LocalRuntime(workspace, manifest=manifest)
            on_prompt = getattr(self.tracer.sink, "on_prompt", None)
            on_text_prompt = getattr(self.tracer.sink, "on_text_prompt", None)
            approval_gate = ApprovalGate(
                auto_approve=auto_approve,
                on_prompt=on_prompt,
                on_text_prompt=on_text_prompt,
            )

            from maike.runtime.bootstrap import BootstrapPolicy, DependencyBootstrapper
            bootstrapper = DependencyBootstrapper(runtime, approval_gate)
            boot_policy = BootstrapPolicy(require_approval=not auto_approve)
            boot_result = await bootstrapper.bootstrap(workspace, manifest, boot_policy)
            if boot_result.success and boot_result.action_taken != "none":
                manifest = boot_result.manifest
                runtime = LocalRuntime(workspace, manifest=manifest)

            await PreflightChecker(runtime, approval_gate).ensure_ready(resolved_provider)

            # Thread resolution
            resolved_thread = await self._resolve_thread(
                workspace, initial_task, thread_id=thread_id, new_thread=new_thread,
            )

            session_metadata = self._build_session_metadata(
                provider=resolved_provider,
                model=resolved_model,
                budget=budget,
                agent_token_budget=agent_token_budget,
                language_override=language_override,
                auto_approve=auto_approve,
            )
            if resolved_thread is not None:
                session_metadata["thread_id"] = resolved_thread["id"]
                session_metadata["thread_name"] = resolved_thread["name"]

            session_id = await self.session_store.create_session(
                initial_task, workspace, metadata=session_metadata,
            )

            long_term_memory = create_long_term_memory(workspace)
            learner = SessionLearner(long_term_memory)
            from maike.memory.longterm import create_typed_memory
            typed_memory = create_typed_memory(workspace)
            session = OrchestratorSession(
                self.session_store, session_id, initial_task, workspace,
                environment_manifest=manifest, learner=learner,
                typed_memory=typed_memory,
                thread=resolved_thread,
            )
            session.skill_dirs = self._resolve_skill_dirs(workspace)
            session.plugin_skills = self._discover_plugin_skills(workspace)
            session.plugin_bundle = self._discover_plugin_bundle(workspace)

            # Agent resolver — unifies plugin, user, and project agents.
            from maike.agents.agent_resolver import AgentResolver
            from maike.constants import AGENTS_PROJECT_SUBDIR, AGENTS_USER_DIR
            _plugin_agents = session.plugin_bundle.agents if session.plugin_bundle else []
            self._agent_resolver = AgentResolver(
                plugin_agents=_plugin_agents,
                user_dir=AGENTS_USER_DIR,
                project_dir=workspace / AGENTS_PROJECT_SUBDIR,
            )
            session.agent_resolver = self._agent_resolver

            # Team resolver — discovers team definitions.
            from maike.agents.team_resolver import TeamResolver
            from maike.constants import TEAMS_PROJECT_SUBDIR, TEAMS_USER_DIR
            self._team_resolver = TeamResolver(
                user_dir=TEAMS_USER_DIR,
                project_dir=workspace / TEAMS_PROJECT_SUBDIR,
            )
            session.team_resolver = self._team_resolver

            # Background task notification queue and output directory.
            notification_queue: asyncio.Queue = asyncio.Queue()
            bg_output_dir = workspace / ".maike" / "bg" / session.id
            bg_output_dir.mkdir(parents=True, exist_ok=True)
            # Fresh manager per session to avoid cross-session state leakage.
            self._async_delegate_manager = AsyncDelegateManager(
                output_dir=bg_output_dir,
                notification_queue=notification_queue,
                session_store=self.session_store,
                session_id=session.id,
                tracer=self.tracer,
            )

            # Build code index for existing codebases (best-effort).
            from maike.agents.react import _has_prior_work
            if _has_prior_work(workspace):
                try:
                    from maike.intelligence.code_index import CodeIndex
                    code_index = CodeIndex(workspace, session.id)
                    await code_index.build()
                    session.code_index = code_index
                except Exception:
                    pass  # graceful degradation

            # Inject output dir into runtime's background manager.
            if hasattr(runtime, 'background_manager'):
                runtime.background_manager._output_dir = bg_output_dir
                runtime.background_manager._notification_queue = notification_queue

            created_llm_gateway = self.llm_gateway is None
            llm_gateway = self.llm_gateway or LLMGateway(
                self.cost_tracker, self.tracer, provider_name=resolved_provider,
            )
            # Separate silent gateway for background cheap-model calls.
            _bg_gateway = LLMGateway(
                self.cost_tracker, self.tracer,
                provider_name=resolved_provider,
                silent=True,
            )

            # Advisor gateway — opt-in via --advisor.  Mirrors _bg_gateway.
            _advisor_session = self._build_advisor_session(
                enabled=advisor_enabled,
                executor_provider=resolved_provider,
                advisor_provider=advisor_provider,
                advisor_model=advisor_model,
                budget=budget,
                budget_pct=advisor_budget_pct,
            )
            session.advisor_session = _advisor_session

            # Live session memory for interactive sessions.
            from maike.memory.session_memory import SessionMemoryService
            _session_memory_interactive = SessionMemoryService(
                workspace=workspace,
                gateway=_bg_gateway,
            ) if workspace else None

            try:
                # Build tool registry and agent core (same as run())
                tool_registry = ToolRegistry()

                async def user_input_handler(prompt: str, context_summary: str = ""):
                    """AskUser handler for run_interactive. See run() for the full doc.

                    AskUser is exempt from --yes — substantive questions need
                    real answers; ApprovalGate.prompt_text handles the no-TTY
                    fallback for non-interactive sessions.
                    """
                    from maike.atoms.tool import ToolResult
                    full_prompt = f"{prompt}\n\n{context_summary}" if context_summary else prompt
                    response = await approval_gate.prompt_text(full_prompt)
                    return ToolResult(
                        tool_name="AskUser", success=True,
                        output=response,
                        metadata={"user_response": response},
                    )

                async def delegate_handler(*, task: str, context: str = "", model_tier: str = "default", plugin_agent: str | None = None, agent_type: str = "implement", fork: bool = False, **_kwargs):
                    """Handle Delegate tool calls — spawn a sub-agent."""
                    from maike.agents.delegate import build_delegate_context
                    from maike.constants import (
                        DELEGATION_BUDGET_CAP_USD,
                        DELEGATION_BUDGET_FRACTION,
                        DELEGATION_MAX_DEPTH,
                        DELEGATION_MAX_ITERATIONS,
                        DELEGATION_TOKEN_BUDGET,
                        model_for_tier,
                    )
                    from maike.tools.bash import register_bash_tools
                    from maike.tools.edit import register_edit_tools
                    from maike.tools.filesystem import register_filesystem_tools
                    from maike.tools.search import register_search_tools

                    parent_ctx = peek_current_agent_context()

                    # Enforce max delegation depth.
                    if parent_ctx and parent_ctx.spawn_depth >= DELEGATION_MAX_DEPTH:
                        from maike.atoms.tool import ToolResult
                        return ToolResult(
                            tool_name="Delegate", success=False,
                            output="Cannot delegate: maximum delegation depth reached.",
                            error="Max delegation depth exceeded",
                        )

                    # ── Fork path: inherit parent's conversation context ──
                    if fork and parent_ctx:
                        from maike.agents.delegate import build_fork_context
                        parent_conversation = parent_ctx.metadata.get("_conversation_ref", [])
                        if not parent_conversation:
                            from maike.atoms.tool import ToolResult
                            return ToolResult(
                                tool_name="Delegate", success=False,
                                output="Fork failed: no parent conversation available. "
                                       "Use regular delegation instead.",
                            )
                        delegate_model = model_for_tier(resolved_provider, model_tier)
                        delegate_ctx, delegate_messages = build_fork_context(
                            parent_ctx, parent_conversation, task,
                            model=delegate_model,
                        )
                        # Fork uses parent's registry — skip tool profile filtering.
                        delegate_registry = ToolRegistry()
                        register_filesystem_tools(delegate_registry, runtime)
                        register_edit_tools(delegate_registry, runtime)
                        register_bash_tools(delegate_registry, runtime)
                        register_search_tools(delegate_registry, runtime)

                    else:
                        # ── Normal path: resolve custom agent + build fresh context ──
                        _agent_def = None
                        if plugin_agent and hasattr(self, "_agent_resolver"):
                            _agent_def = self._agent_resolver.resolve(plugin_agent)
                            if _agent_def is None:
                                from maike.atoms.tool import ToolResult
                                available = self._agent_resolver.list_names()
                                return ToolResult(
                                    tool_name="Delegate", success=False,
                                    output=f"Agent '{plugin_agent}' not found. Available: {available}",
                                )
                            # Use agent's model tier if not overridden
                            if model_tier == "default" and _agent_def.model_tier != "default":
                                model_tier = _agent_def.model_tier

                        delegate_model = model_for_tier(resolved_provider, model_tier)

                        delegate_registry = ToolRegistry()
                        register_filesystem_tools(delegate_registry, runtime)
                        register_edit_tools(delegate_registry, runtime)
                        register_bash_tools(delegate_registry, runtime)
                        register_search_tools(delegate_registry, runtime)

                        # Apply custom agent tool restrictions.
                        if _agent_def and _agent_def.allowed_tools:
                            allowed = set(_agent_def.allowed_tools)
                            delegate_registry._tools = {
                                k: v for k, v in delegate_registry._tools.items()
                                if k in allowed
                            }
                        if _agent_def and _agent_def.disallowed_tools:
                            for tool_name in _agent_def.disallowed_tools:
                                delegate_registry._tools.pop(tool_name, None)

                        if _agent_def:
                            # Build context with custom agent's system prompt.
                            delegate_ctx, delegate_messages = await build_delegate_context(
                                task=task, context=context, session=session,
                                model=delegate_model,
                                system_prompt_override=_agent_def.system_prompt,
                                agent_type=agent_type,
                                agent_def=_agent_def,
                            )
                        else:
                            delegate_ctx, delegate_messages = await build_delegate_context(
                                task=task, context=context, session=session,
                                model=delegate_model,
                                agent_type=agent_type,
                            )
                    remaining_cost = max(parent_ctx.cost_budget_usd - parent_ctx.cost_used_usd, 0.0) if parent_ctx else 1.0
                    remaining_tokens = max(parent_ctx.token_budget - parent_ctx.tokens_used, 0) if parent_ctx else 500_000
                    delegate_ctx = delegate_ctx.model_copy(update={
                        "token_budget": min(remaining_tokens, DELEGATION_TOKEN_BUDGET),
                        "cost_budget_usd": min(remaining_cost * DELEGATION_BUDGET_FRACTION, DELEGATION_BUDGET_CAP_USD),
                        "max_iterations": DELEGATION_MAX_ITERATIONS,
                        "spawn_depth": DELEGATION_MAX_DEPTH,
                    })

                    delegate_core = AgentCore(
                        llm_gateway=llm_gateway,
                        tool_registry=delegate_registry,
                        runtime=runtime,
                        safety_layer=SafetyLayer(workspace),
                        working_memory=WorkingMemory(),
                        tracer=self.tracer,
                        approval_gate=approval_gate,
                        notification_queue=notification_queue,
                    )
                    result = await delegate_core.run(delegate_ctx, delegate_messages)

                    # Auto-retry once on hallucinated output.
                    _quality = assess_delegate_quality(result, agent_type)
                    if _quality.is_hallucinated:
                        try:
                            retry_messages = delegate_messages + [_HALLUCINATION_RETRY_NUDGE]
                            result = await delegate_core.run(delegate_ctx, retry_messages)
                            _quality = assess_delegate_quality(result, agent_type)
                        except Exception:
                            pass  # keep original result

                    # Extract per-agent persistent memory for custom agents.
                    _agent_mem = delegate_ctx.metadata.get("agent_typed_memory")
                    if _agent_mem is not None:
                        try:
                            from maike.memory.automemory import extract_session_memories
                            extract_session_memories(
                                session_memory=None,
                                agent_output=result.output,
                                typed_memory=_agent_mem,
                                task=task,
                            )
                        except Exception:
                            pass  # best-effort

                    summary = result.output or "(no output from delegate)"
                    from maike.atoms.tool import ToolResult
                    return ToolResult(
                        tool_name="Delegate", success=result.success,
                        output=f"Delegate completed.\n\nSummary:\n{summary}{_quality.warning_text}",
                        metadata={
                            "delegate_agent_id": result.agent_id,
                            "delegate_tokens": result.tokens_used,
                            "quality": _quality.quality_flag,
                        },
                    )

                async def async_delegate_handler(
                    *,
                    task: str,
                    context: str = "",
                    model_tier: str = "default",
                    parent_agent_id: str,
                    parent_stage_name: str,
                    remaining_token_budget: int,
                    remaining_cost_budget_usd: float,
                    agent_type: str = "implement",
                    plugin_agent: str | None = None,
                    fork: bool = False,
                    **_kwargs,
                ):
                    """Spawn a sub-agent in the background (non-blocking)."""
                    from maike.agents.delegate import build_delegate_context
                    from maike.atoms.tool import ToolResult as TR
                    from maike.constants import (
                        DELEGATION_BUDGET_CAP_USD,
                        DELEGATION_BUDGET_FRACTION,
                        DELEGATION_MAX_DEPTH,
                        DELEGATION_MAX_ITERATIONS,
                        DELEGATION_TOKEN_BUDGET,
                        MAX_ASYNC_DELEGATES,
                        model_for_tier,
                    )
                    from maike.tools.bash import register_bash_tools
                    from maike.tools.edit import register_edit_tools
                    from maike.tools.filesystem import register_filesystem_tools
                    from maike.tools.search import register_search_tools

                    # Enforce concurrency limit.
                    running = self._async_delegate_manager.running_count
                    if running >= MAX_ASYNC_DELEGATES:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=(
                                f"Cannot spawn: {running}/{MAX_ASYNC_DELEGATES} async delegates running. "
                                f"Wait for a running delegate to finish before spawning more. "
                                f"Use Delegate(action='wait', handle='delegate-NNN') to block "
                                f"until one completes, then retry."
                            ),
                            error="max_async_delegates_exceeded",
                        )

                    delegate_model = model_for_tier(resolved_provider, model_tier)
                    cost_budget = min(
                        remaining_cost_budget_usd * DELEGATION_BUDGET_FRACTION,
                        DELEGATION_BUDGET_CAP_USD,
                    )
                    token_budget = min(remaining_token_budget, DELEGATION_TOKEN_BUDGET)

                    # ── Fork path ──
                    if fork:
                        from maike.agents.delegate import build_fork_context
                        parent_ctx = peek_current_agent_context()
                        parent_conversation = parent_ctx.metadata.get("_conversation_ref", []) if parent_ctx else []
                        if not parent_conversation:
                            return TR(
                                tool_name="Delegate", success=False,
                                output="Fork failed: no parent conversation available.",
                            )
                        delegate_ctx, delegate_messages = build_fork_context(
                            parent_ctx, parent_conversation, task,
                            model=delegate_model,
                        )
                    else:
                        # ── Normal path: resolve custom agent ──
                        _agent_def = None
                        if plugin_agent and hasattr(self, "_agent_resolver"):
                            _agent_def = self._agent_resolver.resolve(plugin_agent)
                            if _agent_def is None:
                                available = self._agent_resolver.list_names()
                                return TR(
                                    tool_name="Delegate", success=False,
                                    output=f"Agent '{plugin_agent}' not found. Available: {available}",
                                )
                            if model_tier == "default" and _agent_def.model_tier != "default":
                                model_tier = _agent_def.model_tier
                                delegate_model = model_for_tier(resolved_provider, model_tier)

                        if _agent_def:
                            delegate_ctx, delegate_messages = await build_delegate_context(
                                task=task, context=context, session=session,
                                parent_id=parent_agent_id, model=delegate_model,
                                system_prompt_override=_agent_def.system_prompt,
                                agent_type=agent_type,
                                agent_def=_agent_def,
                            )
                        else:
                            delegate_ctx, delegate_messages = await build_delegate_context(
                                task=task, context=context, session=session,
                                parent_id=parent_agent_id, model=delegate_model,
                                agent_type=agent_type,
                            )
                    delegate_ctx = delegate_ctx.model_copy(
                        update={
                            "token_budget": token_budget,
                            "cost_budget_usd": cost_budget,
                            "max_iterations": DELEGATION_MAX_ITERATIONS,
                            "spawn_depth": DELEGATION_MAX_DEPTH,
                        }
                    )

                    delegate_registry = ToolRegistry()
                    register_filesystem_tools(delegate_registry, runtime, session=self.session_store)
                    register_edit_tools(delegate_registry, runtime, session=self.session_store)
                    register_bash_tools(delegate_registry, runtime)
                    register_search_tools(delegate_registry, runtime)

                    # Apply custom agent tool restrictions.
                    if _agent_def and _agent_def.allowed_tools:
                        allowed = set(_agent_def.allowed_tools)
                        delegate_registry._tools = {
                            k: v for k, v in delegate_registry._tools.items()
                            if k in allowed
                        }
                    if _agent_def and _agent_def.disallowed_tools:
                        for tool_name in _agent_def.disallowed_tools:
                            delegate_registry._tools.pop(tool_name, None)

                    delegate_core = AgentCore(
                        llm_gateway=llm_gateway,
                        tool_registry=delegate_registry,
                        runtime=runtime,
                        safety_layer=SafetyLayer(workspace),
                        working_memory=WorkingMemory(),
                        tracer=self.tracer,
                        # Delegates auto-approve — they should not trigger
                        # interactive approval prompts on the parent's TUI.
                        approval_gate=ApprovalGate(auto_approve=True),
                        notification_queue=notification_queue,
                    )

                    self.tracer.log_event(
                        TraceEventKind.ASYNC_DELEGATE_SPAWN,
                        parent_agent_id=parent_agent_id,
                        delegate_agent_id=delegate_ctx.agent_id,
                        task=task[:200],
                        cost_budget=cost_budget,
                        token_budget=token_budget,
                    )

                    # Per-delegate inbox for parent→delegate messaging.
                    _inbox = asyncio.Queue()
                    delegate_core._inbox_queue = _inbox

                    # Wrap with auto-retry on hallucination.
                    async def _run_with_retry_interactive(
                        _core=delegate_core, _ctx=delegate_ctx,
                        _msgs=delegate_messages, _atype=agent_type,
                    ):
                        result = await _core.run(_ctx, _msgs)
                        _q = assess_delegate_quality(result, _atype)
                        if _q.is_hallucinated:
                            try:
                                retry_msgs = _msgs + [_HALLUCINATION_RETRY_NUDGE]
                                result = await _core.run(_ctx, retry_msgs)
                            except Exception:
                                pass
                        return result

                    coro = _run_with_retry_interactive()
                    delegate = await self._async_delegate_manager.start(
                        coro, delegate_ctx, task[:200],
                    )
                    delegate.inbox = _inbox

                    output_parts = [f"Async delegate spawned: {delegate.handle} — {task[:200]}"]
                    if delegate.output_file is not None:
                        output_parts.append(f"Output file: {delegate.output_file}")
                    return TR(
                        tool_name="Delegate",
                        success=True,
                        output="\n".join(output_parts),
                        metadata={
                            "handle": delegate.handle,
                            "agent_id": delegate_ctx.agent_id,
                            "output_file": str(delegate.output_file) if delegate.output_file else None,
                        },
                    )

                from maike.agents.skill import SkillLoader as _SL2
                _skill_dirs2 = getattr(session, "skill_dirs", None)
                _plugin_skills2 = getattr(session, "plugin_skills", None) or []
                _skill_loader2: _SL2 | None = None
                if _skill_dirs2:
                    _skill_loader2 = _SL2(
                        builtin_dir=_skill_dirs2.builtin,
                        user_dir=_skill_dirs2.user,
                        project_dir=_skill_dirs2.project,
                        extra_skills=_plugin_skills2,
                    )
                else:
                    _skill_loader2 = _SL2(extra_skills=_plugin_skills2)

                async def delegate_check_handler2(*, handle: str) -> TR:
                    try:
                        status = self._async_delegate_manager.check(handle)
                        return TR(
                            tool_name="Delegate",
                            success=True,
                            output=json.dumps(status, indent=2, default=str),
                            metadata=status,
                        )
                    except KeyError:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=f"Unknown delegate handle: {handle}",
                        )

                async def delegate_wait_handler2(*, handle: str) -> TR:
                    delegates = self._async_delegate_manager._delegates
                    if handle not in delegates:
                        return TR(
                            tool_name="Delegate",
                            success=False,
                            output=f"Unknown delegate handle: {handle}",
                        )
                    delegate = delegates[handle]
                    try:
                        await delegate.task_future
                    except Exception:
                        pass
                    status = self._async_delegate_manager.check(handle)
                    return TR(
                        tool_name="Delegate",
                        success=True,
                        output=json.dumps(status, indent=2, default=str),
                        metadata=status,
                    )

                async def delegate_send_handler2(*, handle: str, message: str):
                    from maike.atoms.tool import ToolResult as TR
                    delegates = self._async_delegate_manager._delegates
                    if handle not in delegates:
                        return TR(tool_name="Delegate", success=False,
                                  output=f"Unknown delegate handle: {handle}")
                    d = delegates[handle]
                    if d.status != TaskState.RUNNING:
                        return TR(tool_name="Delegate", success=False,
                                  output=f"Cannot send to {handle}: delegate is {d.status.value}.")
                    d.inbox.put_nowait(f"## Message from parent agent\n\n{message}")
                    return TR(tool_name="Delegate", success=True,
                              output=f"Message sent to {handle}.",
                              metadata={"handle": handle})

                # Team handler — routes Delegate(team="...") to TeamExecutor.
                async def team_handler2(*, team: str, task: str, context: str = ""):
                    from maike.agents.team_executor import execute_team as _exec_team
                    from maike.atoms.tool import ToolResult as TR
                    from maike.constants import (
                        DELEGATION_BUDGET_FRACTION,
                        DELEGATION_TOKEN_BUDGET,
                        TEAM_BUDGET_CAP_USD,
                    )

                    _tr = self._team_resolver
                    if _tr is None:
                        return TR(tool_name="Team", success=False,
                                  output="Team resolution is not available.")
                    team_def = _tr.resolve(team)
                    if team_def is None:
                        available = _tr.list_names()
                        return TR(tool_name="Team", success=False,
                                  output=f"Team '{team}' not found. Available: {available}")

                    parent_ctx = peek_current_agent_context()
                    remaining_cost = max(parent_ctx.cost_budget_usd - parent_ctx.cost_used_usd, 0.0) if parent_ctx else 1.0
                    remaining_tokens = max(parent_ctx.token_budget - parent_ctx.tokens_used, 0) if parent_ctx else 500_000
                    total_budget = min(remaining_cost * DELEGATION_BUDGET_FRACTION, TEAM_BUDGET_CAP_USD)
                    total_tokens = min(remaining_tokens, DELEGATION_TOKEN_BUDGET)

                    # Pre-create a pool of LLMGateways — one per member + one
                    # for synthesis.  Each gateway has its own Gemini adapter,
                    # enabling true parallel execution without corrupting the
                    # adapter's native history.
                    n_gateways = len(team_def.members) + 1  # +1 for synthesis
                    _gateway_pool = [
                        LLMGateway(self.cost_tracker, self.tracer, provider_name=resolved_provider)
                        for _ in range(n_gateways)
                    ]
                    _gateway_idx = 0

                    async def _spawn_with_pooled_gateway(**kwargs):
                        nonlocal _gateway_idx
                        gw_idx = _gateway_idx
                        _gateway_idx += 1
                        gw = _gateway_pool[gw_idx % len(_gateway_pool)]
                        return await _run_team_member(
                            gw, kwargs, session, runtime,
                            workspace, approval_gate, notification_queue,
                            resolved_provider, self._agent_resolver,
                        )

                    try:
                        return await _exec_team(
                            team=team_def,
                            task=task,
                            context=context,
                            total_budget_usd=total_budget,
                            total_token_budget=total_tokens,
                            spawn_member=_spawn_with_pooled_gateway,
                            agent_resolver=self._agent_resolver,
                        )
                    finally:
                        for gw in _gateway_pool:
                            await gw.aclose()

                # Advisor tool handler — only wired when --advisor is enabled.
                advisor_handler2 = None
                if _advisor_session.enabled:
                    advisor_handler2 = self._make_advisor_handler(_advisor_session)

                from maike.tools import register_default_tools
                register_default_tools(
                    tool_registry, runtime,
                    delegate_handler=delegate_handler,
                    user_input_handler=user_input_handler,
                    async_delegate_handler=async_delegate_handler,
                    delegate_check_handler=delegate_check_handler2,
                    delegate_wait_handler=delegate_wait_handler2,
                    delegate_send_handler=delegate_send_handler2,
                    team_handler=team_handler2,
                    advisor_handler=advisor_handler2,
                    skill_loader=_skill_loader2,
                )

                # Start MCP servers and register their tools.
                bundle2 = session.plugin_bundle
                if bundle2 and bundle2.mcp_configs:
                    from maike.plugins.mcp_registry import MCPToolRegistry as _MCPR
                    mcp_reg2 = _MCPR()
                    await mcp_reg2.start_servers(bundle2.mcp_configs)
                    await mcp_reg2.register_tools(tool_registry)
                    session.mcp_registry = mcp_reg2

                # Initialize hook executor and fire SessionStart.
                if bundle2 and bundle2.hooks and bundle2.hooks.hooks:
                    from maike.plugins.hook_executor import HookExecutor as _HE
                    session.hook_executor = _HE(
                        bundle2.hooks,
                        env={"MAIKE_WORKSPACE": str(workspace), "MAIKE_SESSION_ID": session.id},
                    )
                    from maike.plugins.hooks import HookEvent as _HEvt
                    await session.hook_executor.fire(
                        _HEvt.SESSION_START,
                        context={"session_id": session.id, "workspace": str(workspace), "task": initial_task},
                    )

                # Start LSP servers.
                if bundle2 and bundle2.lsp_configs:
                    from maike.plugins.lsp_manager import LSPManager as _LSPM
                    lsp_mgr2 = _LSPM(workspace_root=str(workspace))
                    await lsp_mgr2.start_servers(bundle2.lsp_configs)
                    session.lsp_manager = lsp_mgr2

                effective_token_budget = agent_token_budget or 0
                from maike.agents.react import build_react_context

                # Task constraints: MAIKE.md Protected Files (static) plus
                # an LLM-extracted markdown task-rules block (prompt-only, no
                # hard enforcement — the react agent reads and honors).
                from maike.agents.constraints import build_task_constraints
                _constraints_md, read_only_patterns = await build_task_constraints(
                    task=initial_task, workspace=workspace,
                    session_bg_gateway=_bg_gateway,
                    session_provider=resolved_provider,
                )
                session.task_constraints_markdown = _constraints_md

                task = initial_task
                accumulated_messages: list[dict] = []
                all_results: list[AgentResult] = []
                _session_outcome = "unknown"

                while True:
                    await self._raise_if_cancelled(session=session, cancellation=cancellation)

                    if not accumulated_messages:
                        ctx, messages = await build_react_context(
                            task, session, model=resolved_model,
                            thread=resolved_thread,
                            adaptive_model=adaptive_model,
                        )
                        # Apply iteration cap and optional token budget.
                        updates: dict = {"max_iterations": DEFAULT_REACT_MAX_ITERATIONS}
                        if effective_token_budget:
                            updates["token_budget"] = effective_token_budget
                        # read_only_patterns feeds the post-hoc "you mutated
                        # a protected file" check in agents/core.py.  The
                        # LLM-extracted constraints are prompt-injected via
                        # build_react_context (reads session.task_constraints_markdown).
                        _md = dict(ctx.metadata)
                        if read_only_patterns:
                            _md["read_only_patterns"] = read_only_patterns
                        updates["metadata"] = _md
                        if tool_profile_override:
                            updates["tool_profile"] = tool_profile_override
                        ctx = ctx.model_copy(update=updates)
                    else:
                        # Strip thought signatures before cross-turn reuse —
                        # they're tied to the previous turn's generation
                        # context and cause Gemini 400 errors on the next call.
                        from maike.agents.react import _strip_thought_signatures
                        accumulated_messages = _strip_thought_signatures(accumulated_messages)
                        accumulated_messages.append({
                            "role": "user",
                            "content": f"--- New instruction from the user ---\n\n{task}",
                        })
                        messages = accumulated_messages
                        ctx.task = task
                        ctx.agent_id = str(uuid4())
                        ctx.started_at = utcnow()

                    _sm_path_i = _session_memory_interactive.memory_path if _session_memory_interactive else None
                    from maike.context.recovery import PostCompactionRecovery
                    _recovery_i = PostCompactionRecovery(workspace=workspace)
                    agent_core = AgentCore(
                        llm_gateway=llm_gateway,
                        tool_registry=tool_registry,
                        runtime=runtime,
                        safety_layer=SafetyLayer(workspace),
                        working_memory=WorkingMemory(session_memory_path=_sm_path_i),
                        tracer=self.tracer,
                        approval_gate=approval_gate,
                        # Don't drain delegate notifications mid-turn —
                        # auto-resume handles results at turn boundaries.
                        notification_queue=None,
                        stream_sink=self._stream_sink,
                        session_memory=_session_memory_interactive,
                        compaction_recovery=_recovery_i,
                    )
                    agent_core._bg_gateway = _bg_gateway
                    agent_core._advisor_session = _advisor_session
                    agent_core._trajectory_auditor = self._build_trajectory_auditor(
                        bg_gateway=_bg_gateway, provider=resolved_provider,
                        session_budget=budget,
                    )

                    code_index_token = CURRENT_CODE_INDEX.set(session.code_index)
                    from maike.tools.context import CURRENT_HOOK_EXECUTOR as _CHE, CURRENT_LSP_MANAGER as _CLM
                    hook_token = _CHE.set(session.hook_executor)
                    lsp_token = _CLM.set(session.lsp_manager)
                    try:
                        result = await agent_core.run(ctx, messages)
                    finally:
                        CURRENT_CODE_INDEX.reset(code_index_token)
                        _CHE.reset(hook_token)
                        _CLM.reset(lsp_token)
                    accumulated_messages = result.messages
                    all_results.append(result)

                    await self.session_store.store_agent_run(session.id, result)

                    _tid = resolved_thread["id"] if resolved_thread else None
                    turn_result = OrchestratorResult(
                        session_id=session.id,
                        pipeline="react",
                        stage_results={"react": [result]},
                        thread_id=_tid,
                    )

                    if turn_callback is None:
                        break

                    # Auto-resume: if delegates completed during this turn,
                    # inject a system turn to process their results.
                    #
                    # Incremental delivery: if some delegates completed but
                    # others are still running, deliver the completed results
                    # immediately. The parent can start synthesizing while
                    # slow delegates finish. If all are done, deliver everything.
                    _has_completed = not notification_queue.empty()
                    _has_running = self._async_delegate_manager.has_running_delegates
                    if _has_completed or _has_running:
                        if _has_running and not _has_completed:
                            # Nothing completed yet — wait for the first one.
                            await self._async_delegate_manager.wait_for_any_completion(
                                timeout=300.0,
                            )
                        # Drain completed notifications (don't wait for stragglers).
                        _notifications: list[str] = []
                        while not notification_queue.empty():
                            try:
                                _notifications.append(str(notification_queue.get_nowait()))
                            except asyncio.QueueEmpty:
                                break
                        if _notifications:
                            _still_running = self._async_delegate_manager.has_running_delegates
                            if _still_running:
                                _suffix = (
                                    "\n\nSome delegates are still running. "
                                    "You can start synthesizing these results "
                                    "while waiting for the rest."
                                )
                            else:
                                _suffix = (
                                    "\n\nAll background delegates have completed. "
                                    "Synthesize the findings."
                                )
                            task = (
                                "[system] Background delegate results:\n\n"
                                + "\n\n---\n\n".join(_notifications)
                                + _suffix
                            )
                            continue

                    try:
                        follow_up = await turn_callback(turn_result)
                    except (EOFError, KeyboardInterrupt):
                        follow_up = None

                    if follow_up is None:
                        break
                    task = follow_up

                    try:
                        self.cost_tracker.check_session_budget()
                    except BudgetExceededError:
                        break

                # Append conversation to thread (strip thought signatures —
                _session_outcome = "success"
                _last_result = all_results[-1] if all_results else None
                if _last_result and not _last_result.success:
                    _session_outcome = "failure"

                await self.session_store.mark_session_status(session.id, "completed")
                _interactive_stages = {"react": all_results}
                await self._record_session_learnings(
                    session=session,
                    pipeline_name=self._pipeline_name(),
                    stage_results=_interactive_stages,
                    outcome="success",
                )
                _final_tid = resolved_thread["id"] if resolved_thread else None
                return OrchestratorResult(
                    session_id=session.id,
                    pipeline="react",
                    stage_results=_interactive_stages,
                    thread_id=_final_tid,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                _session_outcome = "cancelled — interrupted by user"
                await self.session_store.mark_session_status(session.id, "cancelled")
                raise
            except Exception as _exc:
                _err_msg = str(_exc).strip()[:200] or type(_exc).__name__
                _session_outcome = f"failure — {_err_msg}"
                await self.session_store.mark_session_status(session.id, "failed")
                raise
            finally:
                # Wait for any in-flight session memory update.
                if _session_memory_interactive is not None:
                    await _session_memory_interactive.wait_for_pending()

                # Generate and persist session summary — runs on ANY exit
                # (success, failure, cancellation, Ctrl+C).
                _last_run2 = all_results[-1] if all_results else None
                _agent_output = getattr(_last_run2, "output", None)
                _iter_cap_int = bool(
                    _last_run2
                    and getattr(_last_run2, "metadata", {}).get("termination_reason") == "max_iterations"
                )
                _thread_summary = await self._write_session_summary(
                    thread=resolved_thread,
                    messages=accumulated_messages,
                    task=initial_task,
                    outcome=_session_outcome,
                    session_id=session.id,
                    workspace=workspace,
                    agent_output=_agent_output,
                    budget_usd=budget,
                    iteration_cap_hit=_iter_cap_int,
                    bg_gateway=_bg_gateway,
                    provider=resolved_provider,
                )

                # Extract durable learnings into persistent auto-memory.
                # Synchronous — no LLM call, immune to async cancellation.
                self._extract_auto_memories(
                    session=session,
                    session_memory_service=_session_memory_interactive,
                    agent_output=_agent_output,
                    thread_summary=_thread_summary,
                    task=initial_task,
                )

                # Clean up async delegates and background processes.
                await self._async_delegate_manager.cleanup()
                if hasattr(runtime, 'background_manager'):
                    await runtime.background_manager.cleanup()
                # Shutdown plugin servers.
                if session.mcp_registry is not None:
                    await session.mcp_registry.shutdown()
                if session.lsp_manager is not None:
                    await session.lsp_manager.shutdown()
                if created_llm_gateway:
                    await llm_gateway.aclose()
                await _bg_gateway.aclose()

    # ------------------------------------------------------------------
    # Session summary + learning recording
    # ------------------------------------------------------------------

    async def _write_session_summary(
        self,
        *,
        thread: dict[str, Any] | None,
        messages: list[dict],
        task: str,
        outcome: str,
        session_id: str,
        workspace: Path,
        agent_output: str | None,
        budget_usd: float | None = None,
        iteration_cap_hit: bool = False,
        bg_gateway: Any = None,
        provider: str | None = None,
    ) -> str | None:
        """Write a session summary to the thread.  Returns the summary text
        (for downstream use by auto-memory), or None.  Never raises.

        The optional ``budget_usd`` / ``iteration_cap_hit`` / ``bg_gateway`` /
        ``provider`` arguments drive the session-verdict classification (see
        ``maike.memory.verdict``).  All are back-compat defaults — callers
        that don't pass them get the original behavior minus the verdict line.
        """
        if thread is None:
            return None

        # Classify the session.  Deterministic short-circuits are free;
        # LLM call only fires for ambiguous (satisfied vs partial) cases.
        verdict: Any = None
        try:
            from maike.memory.verdict import (
                classify_session, count_successful_edits,
            )
            edits_count = count_successful_edits(messages)
            # Budget threshold: 95% of cap matches the projected-cost precheck
            # used elsewhere in AgentCore.
            budget_hit = False
            # Two independent signals for budget exhaustion:
            #  (a) accumulated cost near the cap (>= 85% — projected-cost
            #      check typically bails at ~90% of budget, before the
            #      hard cap).
            #  (b) outcome string contains a budget-exceeded marker (the
            #      exception message from ``BudgetExceededError``).
            if budget_usd is not None and budget_usd > 0:
                try:
                    used = float(getattr(self.cost_tracker, "session_total", 0.0))
                except Exception:
                    used = 0.0
                if used >= budget_usd * 0.85:
                    budget_hit = True
            if outcome and ("budget" in outcome.lower() or "BudgetExceededError" in (outcome or "")):
                budget_hit = True
            # classify_session is fully deterministic now (no LLM call).
            # The bg_gateway / provider kwargs are retained only for legacy
            # callers that still pass them; the function ignores them.
            verdict = classify_session(
                outcome=outcome,
                edits_count=edits_count,
                budget_hit=budget_hit,
                iteration_cap_hit=iteration_cap_hit,
                task=task,
                agent_output=(agent_output or ""),
                messages=messages,
            )
            # Persist verdict into sessions.metadata (additive — does not
            # overwrite existing fields).
            if verdict is not None:
                await self._persist_verdict(session_id, verdict, edits_count)
        except Exception:  # noqa: BLE001 — verdict must never break summary
            import logging
            logging.getLogger(__name__).debug(
                "Verdict classification failed for %s (non-fatal)", session_id, exc_info=True,
            )
            verdict = None

        # Fallback: when messages is empty (e.g. Ctrl+C during agent turn),
        # try using the session memory file as a summary source.
        if not messages:
            try:
                from maike.constants import SESSION_MEMORY_FILENAME
                sm_path = workspace / ".maike" / SESSION_MEMORY_FILENAME
                if sm_path.exists():
                    sm_text = sm_path.read_text()
                    if "## Current State" in sm_text and len(sm_text) > 100:
                        verdict_frag = ""
                        if verdict is not None and hasattr(verdict, "render_line"):
                            try:
                                vl = verdict.render_line()
                                if vl:
                                    verdict_frag = f"**{vl}**\n"
                            except Exception:
                                pass
                        summary = (
                            f"## Session {session_id} (interrupted)\n\n"
                            f"**Task:** {task}\n"
                            f"**Outcome:** {outcome}\n"
                            f"{verdict_frag}\n"
                            f"### Session Memory at Interruption\n\n{sm_text}"
                        )
                        await self.session_store.append_thread_summary(thread["id"], summary)
                        return summary
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "Failed to write fallback session summary for %s", session_id, exc_info=True,
                )
            return None

        try:
            from maike.memory.summary import SessionSummaryBuilder
            sb = SessionSummaryBuilder()
            summary = sb.build_summary(
                messages=messages,
                task=task,
                outcome=outcome,
                session_id=session_id,
                timestamp=utcnow().isoformat(),
                workspace=workspace,
                agent_output=agent_output,
                verdict=verdict,
            )
            await self.session_store.append_thread_summary(thread["id"], summary)
            return summary
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "Failed to write session summary for %s", session_id, exc_info=True,
            )
            return None

    def _collect_modified_files(self, messages: list[dict]) -> list[str]:
        """Return a deduplicated list of file paths the session modified."""
        from maike.memory.working import WorkingMemory
        try:
            wm = WorkingMemory()
            ledger = wm._build_mutation_ledger(messages) or ""
        except Exception:
            return []
        # Ledger is a free-text block — pull any path-shaped tokens out.
        import re as _re
        paths: list[str] = []
        seen: set[str] = set()
        for m in _re.finditer(r"[\w./\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|c|cc|cpp|h|hpp|md|yml|yaml|toml|json|txt)", ledger):
            p = m.group(0)
            if p not in seen:
                seen.add(p)
                paths.append(p)
        return paths[:20]

    async def _persist_verdict(
        self, session_id: str, verdict: Any, edits_count: int,
    ) -> None:
        """Merge the verdict into the session's metadata JSON blob.

        Read-modify-write because ``update_session_metadata`` takes the full
        dict.  Additive — preserves all existing metadata keys.
        """
        try:
            session = await self.session_store.get_session(session_id)
            if session is None:
                return
            meta = dict(session.get("metadata") or {})
            meta["verdict"] = verdict.to_metadata() if hasattr(verdict, "to_metadata") else None
            meta["edits_count"] = edits_count
            await self.session_store.update_session_metadata(session_id, meta)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).debug(
                "Failed to persist verdict for %s (non-fatal)", session_id, exc_info=True,
            )

    def _extract_auto_memories(
        self,
        *,
        session: "OrchestratorSession",
        session_memory_service,
        agent_output: str | None,
        thread_summary: str | None = None,
        task: str = "",
    ) -> None:
        """Extract durable learnings into persistent auto-memory.

        Purely synchronous — no LLM call, no async, immune to TUI
        cancellation.  Parses session memory deterministically.
        Never raises.
        """
        typed_memory = getattr(session, "typed_memory", None)
        if typed_memory is None:
            return

        # Read session memory content (still on disk at this point — only
        # deleted when the *next* session's SessionMemoryService.__init__ runs).
        session_memory_content: str | None = None
        if session_memory_service is not None:
            session_memory_content = session_memory_service.read_memory()
        if not session_memory_content:
            # Fallback: try reading the file directly.
            from maike.constants import SESSION_MEMORY_FILENAME
            sm_path = session.workspace / ".maike" / SESSION_MEMORY_FILENAME
            if sm_path.exists():
                try:
                    session_memory_content = sm_path.read_text()
                except OSError:
                    pass

        try:
            from maike.memory.automemory import extract_session_memories
            count = extract_session_memories(
                session_memory=session_memory_content,
                agent_output=agent_output,
                thread_summary=thread_summary,
                typed_memory=typed_memory,
                task=task,
            )
            if count:
                import logging
                logging.getLogger(__name__).info(
                    "Auto-memory: extracted %d durable memories for session %s",
                    count, session.id,
                )
        except Exception as _exc:
            import logging
            logging.getLogger(__name__).warning(
                "Auto-memory extraction failed for session %s: %s: %s",
                session.id, type(_exc).__name__, str(_exc)[:300],
            )

    def _pipeline_name(self) -> str:
        return "react"

    async def _record_session_learnings(
        self,
        *,
        session: OrchestratorSession,
        pipeline_name: str,
        stage_results: dict[str, list[AgentResult]],
        outcome: str,
    ) -> None:
        """Extract and persist session learnings. Never raises."""
        learner = getattr(session, "learner", None)
        if learner is None:
            return
        language = getattr(
            getattr(session, "environment_manifest", None),
            "language",
            "unknown",
        )
        total_cost = sum(
            result.cost_usd
            for results in stage_results.values()
            for result in results
        )
        try:
            learner.record_session(
                session_id=session.id,
                task=session.task,
                outcome=outcome,
                pipeline=pipeline_name,
                language=str(language) if language else "unknown",
                cost_usd=total_cost,
                failure_reasons=self._extract_failure_reasons(stage_results),
                successful_strategies=self._extract_successful_strategies(stage_results),
                role_specific_learnings=self._extract_role_learnings(stage_results),
            )
        except Exception:
            self.tracer.log_event("learning_record_failed", session_id=session.id)

    def _extract_failure_reasons(
        self, stage_results: dict[str, list[AgentResult]],
    ) -> list[str]:
        reasons: list[str] = []
        for stage_name, results in stage_results.items():
            for result in results:
                if not result.success:
                    term = result.metadata.get("termination_reason", "unknown")
                    reasons.append(f"{stage_name}/{result.role}: {term}")
        return reasons[:10]

    def _extract_successful_strategies(
        self, stage_results: dict[str, list[AgentResult]],
    ) -> list[str]:
        strategies: list[str] = []
        for stage_name, results in stage_results.items():
            if results and all(r.success for r in results):
                strategies.append(f"{stage_name} passed on first attempt")
            for result in results:
                ct = result.metadata.get("context_telemetry", {})
                if ct.get("on_demand_fetches", 0) > 0:
                    strategies.append(
                        f"{result.role}: progressive loading ({ct['on_demand_fetches']} fetches)"
                    )
                if ct.get("prune_events", 0) > 0:
                    strategies.append(
                        f"{result.role}: pruning ({ct['prune_events']} events, {ct.get('tokens_pruned', 0)} tokens saved)"
                    )
        return strategies[:10]

    def _extract_role_learnings(
        self, stage_results: dict[str, list[AgentResult]],
    ) -> dict[str, list[str]]:
        role_learnings: dict[str, list[str]] = {}
        for results in stage_results.values():
            for result in results:
                role = result.role
                learnings = role_learnings.setdefault(role, [])
                ct = result.metadata.get("context_telemetry", {})
                iters = result.metadata.get("iteration_count", 0)
                max_iter = result.metadata.get("max_iterations", 0)
                if iters > 0 and max_iter > 0:
                    ratio = iters / max_iter
                    if ratio > 0.8:
                        learnings.append(f"near iteration limit ({iters}/{max_iter})")
                    elif ratio < 0.3:
                        learnings.append(f"converged quickly ({iters} iterations)")
                if ct.get("compression_applied"):
                    learnings.append(f"compression applied: {ct.get('compression_levels_used', [])}")
                if ct.get("convergence_nudge_injected"):
                    learnings.append("convergence nudge was triggered")
        return {role: items[:5] for role, items in role_learnings.items() if items}

    # ------------------------------------------------------------------
    # Partition fan-out/fan-in (--parallel-coding)
    # ------------------------------------------------------------------

    _PARTITION_PLAN_PROMPT = """\
You are a task planner. Given a coding task and a workspace file tree, determine if the task can be split into independent sub-tasks that can be worked on in parallel by separate agents.

Rules:
- Each sub-task must operate on a non-overlapping set of files.
- If files share tight coupling (circular imports, shared mutable state), they CANNOT be split.
- If the task cannot be partitioned (too small, too coupled, or single-file), return an empty list in ``partitions``.
- Maximum 5 partitions.

Return a JSON object shaped like:
  {{"partitions": [{{"subtask": "description", "files": ["path1.py", "path2.py"]}}, ...]}}

Workspace file tree:
{file_tree}

Task:
{task}
"""

    async def _plan_partitions(
        self,
        task: str,
        workspace: Path,
        llm_gateway: "LLMGateway",
        model: str,
    ) -> list[dict[str, Any]] | None:
        """Ask the LLM to decompose *task* into independent file-scoped partitions.

        Uses provider-native structured output via
        ``LLMRequest.response_schema`` — no ``json.loads`` on LLM text in
        the live execution path.  See maike/orchestrator/partition_schema.py
        for the Pydantic model.

        Returns a list of ``{"subtask": str, "files": [str]}`` dicts,
        or ``None`` if the task is not partitionable.
        """
        from maike.orchestrator.partition_schema import PartitionPlan

        # Build a compact file tree.
        skip = {".git", ".venv", "__pycache__", "node_modules", ".maike"}
        tree_lines: list[str] = []
        for root, dirs, files in sorted(workspace.walk()):
            dirs[:] = [d for d in dirs if d not in skip]
            rel = root.relative_to(workspace)
            for f in sorted(files):
                tree_lines.append(str(rel / f))

        prompt = self._PARTITION_PLAN_PROMPT.format(
            file_tree="\n".join(tree_lines[:500]),  # cap for context
            task=task,
        )

        try:
            result = await llm_gateway.call(
                system="You are a task decomposition planner.",
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=2000,
                response_schema=PartitionPlan,
            )
        except Exception:
            return None

        # Happy path: provider honored the schema, LLMResult.parsed is a dict.
        if result.parsed is None:
            return None

        try:
            plan = PartitionPlan.model_validate(result.parsed)
        except Exception:
            return None

        if len(plan.partitions) <= 1:
            return None  # Not partitionable or single partition.

        # Validate: no overlapping file scopes.
        seen_files: set[str] = set()
        for entry in plan.partitions:
            for f in entry.files:
                if f in seen_files:
                    return None  # Overlap detected.
                seen_files.add(f)

        # Return shape unchanged for callers: list[dict[str, Any]].
        return [
            {"subtask": e.subtask, "files": list(e.files)}
            for e in plan.partitions[:5]
        ]

    async def _run_partitioned(
        self,
        task: str,
        partition_plan: list[dict[str, Any]],
        session: "OrchestratorSession",
        workspace: Path,
        runtime: "ExecutionRuntime",
        llm_gateway: "LLMGateway",
        approval_gate: Any,
        notification_queue: asyncio.Queue,
        *,
        model: str,
        budget: float,
    ) -> list["AgentResult"]:
        """Execute partitioned sub-tasks in parallel, then fan-in."""
        from maike.agents.delegate import build_delegate_context
        from maike.agents.react import build_react_context
        from maike.tools import register_default_tools
        from maike.tools.blackboard import Blackboard, register_blackboard_tools

        blackboard = Blackboard()
        partition_budget = budget / (len(partition_plan) + 1)  # +1 for fan-in

        # --- Fan-out: run partition agents concurrently ---
        async def run_partition(entry: dict) -> "AgentResult":
            subtask = entry["subtask"]
            files = entry.get("files", [])

            from maike.agents.helpers import build_context, build_environment_metadata
            ctx = build_context(
                role="partition_agent",
                task=subtask,
                stage_name="partition",
                tool_profile="coding",
                model=model,
                input_artifacts=[],
                session_id=session.id,
                metadata={
                    **build_environment_metadata(session),
                    "pipeline": "react",
                    "coordination_mode": "partition",
                    "files_in_scope": files,
                    "adaptive_model": False,
                },
            )
            ctx = ctx.model_copy(update={
                "max_iterations": 30,
                "cost_budget_usd": partition_budget,
            })

            messages = [{"role": "user", "content": (
                f"## Partition Task\n\n{subtask}\n\n"
                f"## Your File Scope\n\nYou may ONLY modify these files: {', '.join(files)}\n\n"
                "Use BlackboardPost to share interface contracts or discoveries "
                "that other partition agents need to know about.\n"
                "Use BlackboardRead to check what other agents have posted."
            )}]

            partition_registry = ToolRegistry()
            from maike.tools.filesystem import register_filesystem_tools
            from maike.tools.edit import register_edit_tools
            from maike.tools.search import register_search_tools, register_semantic_search_tool
            from maike.tools.web import register_web_tools
            register_filesystem_tools(partition_registry, runtime, session=self.session_store)
            register_edit_tools(partition_registry, runtime, session=self.session_store)
            register_search_tools(partition_registry, runtime)
            register_semantic_search_tool(partition_registry, runtime)
            register_web_tools(partition_registry, runtime)
            register_blackboard_tools(partition_registry, blackboard)
            # No Bash — partition agents shouldn't run arbitrary commands.

            partition_core = AgentCore(
                llm_gateway=llm_gateway,
                tool_registry=partition_registry,
                runtime=runtime,
                safety_layer=SafetyLayer(workspace),
                working_memory=WorkingMemory(),
                tracer=self.tracer,
                approval_gate=approval_gate,
                notification_queue=notification_queue,
            )
            return await partition_core.run(ctx, messages)

        self.tracer.log_event(
            TraceEventKind.PARTITION_FANOUT_START,
            partition_count=len(partition_plan),
            session_id=session.id,
        )

        partition_results = await asyncio.gather(
            *(run_partition(entry) for entry in partition_plan),
            return_exceptions=True,
        )

        # Collect outputs for fan-in context.
        partition_summaries: list[str] = []
        agent_results: list[AgentResult] = []
        for i, (entry, result) in enumerate(zip(partition_plan, partition_results)):
            if isinstance(result, Exception):
                partition_summaries.append(
                    f"### Partition {i+1}: {entry['subtask']}\n**FAILED**: {result}\n"
                )
            else:
                agent_results.append(result)
                partition_summaries.append(
                    f"### Partition {i+1}: {entry['subtask']}\n"
                    f"**Status**: {'completed' if result.success else 'failed'}\n"
                    f"**Output**: {(result.output or '(no output)')[:500]}\n"
                )

        bb_contents = blackboard.read()
        bb_summary = json.dumps(bb_contents, indent=2) if bb_contents else "(empty)"

        # --- Fan-in: integration agent ---
        self.tracer.log_event(TraceEventKind.PARTITION_FANIN_START, session_id=session.id)

        fanin_ctx, fanin_messages = await build_react_context(
            task=(
                f"## Fan-In Integration\n\n"
                f"Multiple partition agents worked on independent parts of this task:\n\n"
                f"**Original task**: {task}\n\n"
                f"{''.join(partition_summaries)}\n\n"
                f"## Blackboard (shared state from partition agents)\n```json\n{bb_summary}\n```\n\n"
                f"## Your Job\n"
                f"1. Review what each partition agent did\n"
                f"2. Resolve any cross-cutting concerns or integration issues\n"
                f"3. Run the test suite to verify everything works together\n"
                f"4. Fix any issues found\n"
            ),
            session=session,
            model=model,
            adaptive_model=False,
        )
        fanin_ctx = fanin_ctx.model_copy(update={
            "max_iterations": 30,
            "cost_budget_usd": partition_budget,
        })

        tool_registry = ToolRegistry()
        register_default_tools(tool_registry, runtime, session=self.session_store)

        fanin_core = AgentCore(
            llm_gateway=llm_gateway,
            tool_registry=tool_registry,
            runtime=runtime,
            safety_layer=SafetyLayer(workspace),
            working_memory=WorkingMemory(),
            tracer=self.tracer,
            approval_gate=approval_gate,
            notification_queue=notification_queue,
        )
        fanin_result = await fanin_core.run(fanin_ctx, fanin_messages)
        agent_results.append(fanin_result)

        self.tracer.log_event(
            TraceEventKind.PARTITION_COMPLETE,
            partition_count=len(partition_plan),
            fanin_success=fanin_result.success,
            session_id=session.id,
        )

        return agent_results

    def _build_session_metadata(
        self,
        *,
        provider: str,
        model: str,
        budget: float | None,
        agent_token_budget: int | None,
        language_override: str | None,
        auto_approve: bool,
    ) -> dict[str, object]:
        return {
            "run_config": {
                "provider": provider,
                "model": model,
                "budget": budget,
                "agent_token_budget": agent_token_budget,
                "language_override": language_override,
                "dynamic_agents_enabled": self.dynamic_agents_enabled,
                "parallel_coding_enabled": self.parallel_coding_enabled,
                "auto_approve": auto_approve,
            }
        }

    async def _resolve_thread(
        self,
        workspace: Path,
        task: str,
        *,
        thread_id: str | None = None,
        new_thread: bool = False,
    ) -> dict[str, Any] | None:
        """Resolve the thread for this session.

        Returns the thread dict, or None if threading is not applicable
        (e.g., no thread_id given and new_thread not requested and no
        active thread exists in the workspace).
        """
        from maike.memory.session import generate_thread_name

        if thread_id is not None:
            thread = await self.session_store.get_thread(thread_id)
            if thread is None:
                raise OrchestratorError(f"Thread not found: {thread_id}")
            return thread

        if new_thread:
            name = generate_thread_name(task)
            tid = await self.session_store.create_thread(workspace, name)
            return await self.session_store.get_thread(tid)

        # Default: resume last active thread, or create a new one.
        thread = await self.session_store.get_active_thread(workspace)
        if thread is not None:
            return thread

        # No active thread — create one from the task description.
        name = generate_thread_name(task)
        tid = await self.session_store.create_thread(workspace, name)
        return await self.session_store.get_thread(tid)

    async def _raise_if_cancelled(
        self,
        *,
        session,
        cancellation: CancellationController | None,
    ) -> None:
        if cancellation is None or not cancellation.cancel_requested:
            return
        await self.session_store.mark_session_status(session.id, "cancelled")
        self.tracer.log_event(TraceEventKind.SESSION_CANCELLED, session_id=session.id)
        raise OrchestratorCancelled(session.id)

    @staticmethod
    def _resolve_skill_dirs(workspace: Path):
        """Resolve skill/knowledge directories for the session."""
        from maike.agents.knowledge import _KNOWLEDGE_DIR
        from maike.constants import SKILL_PROJECT_SUBDIR, SKILL_USER_DIR
        from maike.orchestrator.session import SkillDirectories

        user_dir = SKILL_USER_DIR if SKILL_USER_DIR.is_dir() else None
        project_dir = workspace / SKILL_PROJECT_SUBDIR
        project_dir = project_dir if project_dir.is_dir() else None
        return SkillDirectories(
            builtin=_KNOWLEDGE_DIR,
            user=user_dir,
            project=project_dir,
        )

    @staticmethod
    def _discover_plugin_skills(workspace: Path) -> list:
        """Discover plugins and load their skills.

        Scans user and project plugin directories, discovers plugin
        manifests, and loads all skills from them.  Returns an empty
        list when no plugins are found or on any error.
        """
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.loader import PluginLoader

        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_plugin_dir = workspace / PLUGIN_PROJECT_SUBDIR
        if project_plugin_dir.is_dir():
            search_dirs.append(project_plugin_dir)

        if not search_dirs:
            return []

        try:
            manifests = PluginDiscovery.discover_enabled(search_dirs)
            if not manifests:
                return []
            return PluginLoader.load_all_plugin_skills(manifests)
        except Exception:
            return []  # graceful degradation

    @staticmethod
    def _discover_plugin_bundle(workspace: Path):
        """Discover all plugin components: skills, agents, hooks, MCP, LSP.

        Returns a PluginBundle with all discovered components.
        Graceful degradation — returns empty bundle on any error.
        """
        from maike.constants import (
            LSP_PROJECT_CONFIG_NAME,
            MCP_PROJECT_CONFIG_NAME,
            PLUGIN_PROJECT_SUBDIR,
            PLUGIN_USER_DIR,
        )
        from maike.plugins.agent_loader import load_all_plugin_agents
        from maike.plugins.bundle import PluginBundle
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.hooks import load_hook_configs
        from maike.plugins.loader import PluginLoader
        from maike.plugins.lsp_config import load_lsp_configs
        from maike.plugins.mcp_config import load_mcp_configs

        bundle = PluginBundle()

        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_plugin_dir = workspace / PLUGIN_PROJECT_SUBDIR
        if project_plugin_dir.is_dir():
            search_dirs.append(project_plugin_dir)

        try:
            manifests = PluginDiscovery.discover_enabled(search_dirs) if search_dirs else []
            bundle.manifests = manifests

            # Skills (existing)
            bundle.skills = PluginLoader.load_all_plugin_skills(manifests) if manifests else []

            # Agents
            bundle.agents = load_all_plugin_agents(manifests)

            # Hooks (project + plugins)
            project_hooks = workspace / ".maike" / "hooks" / "hooks.json"
            bundle.hooks = load_hook_configs(
                plugin_manifests=manifests,
                project_hooks_path=project_hooks if project_hooks.is_file() else None,
            )

            # MCP servers
            bundle.mcp_configs = load_mcp_configs(workspace, manifests)

            # LSP servers
            project_lsp = workspace / LSP_PROJECT_CONFIG_NAME
            bundle.lsp_configs = load_lsp_configs(
                plugin_manifests=manifests,
                project_lsp_path=project_lsp if project_lsp.is_file() else None,
            )

        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Plugin discovery failed, continuing without plugins",
                exc_info=True,
            )

        return bundle


def result_to_tool_result(result: AgentResult):
    from maike.atoms.tool import ToolResult

    metadata = dict(result.metadata)
    metadata.update(
        {
            "agent_id": result.agent_id,
            "role": result.role,
            "produced_artifact_ids": list(result.produced_artifact_ids),
        }
    )
    return ToolResult(
        tool_name="Delegate",
        success=result.success,
        output=result.output or "",
        raw_output=result.output or "",
        metadata=metadata,
    )