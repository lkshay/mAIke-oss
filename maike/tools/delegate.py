"""Delegate tool — spawn sub-agents synchronously or in the background."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.tools.context import current_agent_context
from maike.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Shared input schema fragments
# ---------------------------------------------------------------------------

_TASK_SCHEMA = {
    "type": "string",
    "description": (
        "Clear, specific task for the sub-agent. Include what "
        "files to look at, what to change, and how to verify."
    ),
}

_CONTEXT_SCHEMA = {
    "type": "string",
    "description": (
        "Relevant context the sub-agent needs: error messages, "
        "file paths, design decisions, or anything you've learned "
        "that the sub-agent won't have access to."
    ),
}

_MODEL_TIER_SCHEMA = {
    "type": "string",
    "enum": ["default", "cheap", "strong"],
    "description": (
        "Model tier for the sub-agent. 'default' uses the session model, "
        "'cheap' uses a lightweight model (good for simple/mechanical tasks), "
        "'strong' uses the most capable model (for complex reasoning, "
        "architecture, or hard debugging). Defaults to 'default'."
    ),
}


# ---------------------------------------------------------------------------
# Unified Delegate tool
# ---------------------------------------------------------------------------

def register_delegate_tool(
    registry: ToolRegistry,
    delegate_handler: Callable[..., Awaitable[ToolResult]],
    async_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    check_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    wait_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    send_handler: Callable[..., Awaitable[ToolResult]] | None = None,
) -> None:
    """Register the unified ``Delegate`` tool.

    Routes based on parameters:
    - ``action="check"`` + ``handle``: query background delegate status
    - ``action="wait"`` + ``handle``: block until background delegate completes
    - ``action="send"`` + ``handle`` + ``task``: send message to running delegate
    - ``background=true``: spawn sub-agent in background (non-blocking)
    - default: spawn sub-agent synchronously (blocking)
    """

    async def unified_delegate(
        task: str | None = None,
        context: str = "",
        model_tier: str = "default",
        background: bool = True,
        action: str | None = None,
        handle: str | None = None,
        agent: str | None = None,
        agent_type: str = "explore",
        fork: bool = False,
    ) -> ToolResult:
        # ── Check / Wait actions ────────────────────────────────
        if action == "check":
            if not handle:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="handle is required for action='check'. "
                           "Use the handle returned when you spawned the background delegate.",
                )
            if check_handler is None:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="Background delegation is not available in this context.",
                )
            return await check_handler(handle=handle)

        if action == "wait":
            if not handle:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="handle is required for action='wait'. "
                           "Use the handle returned when you spawned the background delegate.",
                )
            if wait_handler is None:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="Background delegation is not available in this context.",
                )
            return await wait_handler(handle=handle)

        if action == "send":
            if not handle:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="handle is required for action='send'. "
                           "Use the handle returned when you spawned the background delegate.",
                )
            if not task:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="task (message content) is required for action='send'.",
                )
            if send_handler is None:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="Agent communication is not available in this context.",
                )
            return await send_handler(handle=handle, message=task)

        # ── Fork validation ──────────────────────────────────────
        if fork and agent:
            return ToolResult(
                tool_name="Delegate",
                success=False,
                output="Cannot combine fork=true with agent parameter. "
                       "Fork inherits the parent's prompt; custom agents have their own.",
            )
        if fork:
            # Fork guards: check if already inside a fork.
            ctx = current_agent_context()
            if ctx.metadata.get("is_fork"):
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="Cannot fork from inside a fork. "
                           "You are already a forked agent — complete your task and report.",
                )

        # ── Normalize agent_type ─────────────────────────────────
        _VALID_AGENT_TYPES = ("explore", "plan", "implement", "review", "verify", "debug", "test")
        if agent_type not in _VALID_AGENT_TYPES:
            agent_type = "explore"

        # Default explore agents to cheap model tier when not explicitly set.
        if agent_type == "explore" and model_tier == "default":
            model_tier = "cheap"

        # Default debug agents to strong model when not explicitly set.
        if agent_type == "debug" and model_tier == "default":
            model_tier = "strong"

        # ── Background delegation ────────────────────────────────
        if background:
            if async_handler is None:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="Background delegation is not available in this context.",
                )
            if not task:
                return ToolResult(
                    tool_name="Delegate",
                    success=False,
                    output="task is required when spawning a background delegate.",
                )
            if model_tier not in ("default", "cheap", "strong"):
                model_tier = "default"
            ctx = current_agent_context()
            result = await async_handler(
                task=task,
                context=context,
                model_tier=model_tier,
                parent_agent_id=ctx.agent_id,
                parent_stage_name=ctx.stage_name,
                remaining_token_budget=max(ctx.token_budget - ctx.tokens_used, 0),
                remaining_cost_budget_usd=max(ctx.cost_budget_usd - ctx.cost_used_usd, 0.0),
                plugin_agent=agent,
                agent_type=agent_type,
                fork=fork,
            )
            child_agent_id = result.metadata.get("agent_id")
            if child_agent_id and child_agent_id not in ctx.children_ids:
                ctx.children_ids.append(child_agent_id)
            return result

        # ── Synchronous delegation (default) ─────────────────────
        if not task:
            return ToolResult(
                tool_name="Delegate",
                success=False,
                output="task is required for delegation.",
            )
        if model_tier not in ("default", "cheap", "strong"):
            model_tier = "default"
        ctx = current_agent_context()
        result = await delegate_handler(
            task=task,
            context=context,
            model_tier=model_tier,
            parent_agent_id=ctx.agent_id,
            parent_stage_name=ctx.stage_name,
            remaining_token_budget=max(ctx.token_budget - ctx.tokens_used, 0),
            remaining_cost_budget_usd=max(ctx.cost_budget_usd - ctx.cost_used_usd, 0.0),
            plugin_agent=agent,
            agent_type=agent_type,
            fork=fork,
        )
        child_agent_id = result.metadata.get("agent_id")
        if child_agent_id and child_agent_id not in ctx.children_ids:
            ctx.children_ids.append(child_agent_id)
        return result

    registry.register(
        ToolSchema(
            name="Delegate",
            description=(
                "Spawn a sub-agent for a specific task. Runs in the background by default — "
                "returns a handle for tracking. Use action='check' + handle to query status. "
                "Use action='wait' + handle to block until completion. "
                "Set blocking=true to wait for the result synchronously."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task": _TASK_SCHEMA,
                    "context": _CONTEXT_SCHEMA,
                    "model_tier": _MODEL_TIER_SCHEMA,
                    "background": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Runs in background by default. Set to false (blocking=true) "
                            "only when you need the result immediately in your next step."
                        ),
                    },
                    "action": {
                        "type": "string",
                        "enum": ["check", "wait", "send"],
                        "description": (
                            "Action to perform on a background delegate: "
                            "'check' returns current status, 'wait' blocks until completion, "
                            "'send' sends a new message/instruction to a running delegate."
                        ),
                    },
                    "handle": {
                        "type": "string",
                        "description": "Handle of a background delegate (returned when spawned with background=true).",
                    },
                    "agent": {
                        "type": "string",
                        "description": (
                            "Name of a custom agent to use. Agents are loaded from: "
                            "project (.maike/agents/*.md), user (~/.config/maike/agents/*.md), "
                            "or plugins (namespace:name). Custom agents have specialized "
                            "system prompts and tool restrictions. "
                            "Omit to use the default delegate agent."
                        ),
                    },
                    "agent_type": {
                        "type": "string",
                        "enum": ["explore", "plan", "implement", "review", "verify", "debug", "test"],
                        "default": "explore",
                        "description": (
                            "Type of delegate agent: "
                            "'explore' (read-only research, cheap model — default), "
                            "'plan' (architecture planning, read-only), "
                            "'implement' (full coding — Write/Edit/Bash), "
                            "'review' (code review, read-only + read-only bash), "
                            "'verify' (adversarial testing, VERDICT output), "
                            "'debug' (debugging specialist, strong model), or "
                            "'test' (test suite writer, full tools). "
                            "Use 'implement' only when the delegate needs to write files."
                        ),
                    },
                    "fork": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Fork: inherit parent's full conversation context. "
                            "The forked child sees everything the parent has seen — "
                            "file contents, reasoning, discoveries. Use for research "
                            "tasks where the child needs the parent's accumulated knowledge. "
                            "Cannot be combined with agent parameter. "
                            "Cannot fork from inside a fork."
                        ),
                    },
                },
                "required": [],
            },
        ),
        fn=unified_delegate,
        # WRITE level: auto-approved in react mode.  Delegates have their
        # own SafetyLayer internally — the parent should not double-gate them.
        # Previously EXECUTE, which triggered approval prompts for every
        # delegate spawn including read-only explore agents.
        risk_level=RiskLevel.WRITE,
    )
