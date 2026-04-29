"""Built-in slash commands — migrated from repl.py's if/elif chain."""

from __future__ import annotations

import sys
from typing import Any

from maike.commands.registry import CommandRegistry


def register_builtins(registry: CommandRegistry) -> None:
    """Register all built-in slash commands."""
    registry.register("help", "Show available commands", _handle_help)
    registry.register("cost", "Show session cost breakdown", _handle_cost)
    registry.register("budget", "Show remaining budget", _handle_budget)
    registry.register("history", "Show conversation history for current thread", _handle_history)
    registry.register("new", "Start a new thread (fresh conversation)", _handle_new)
    registry.register("status", "Show session status", _handle_status)
    registry.register("tasks", "Show background tasks and delegates", _handle_tasks)
    registry.register("plugin", "Manage plugins (list|enable|disable|install)", _handle_plugin)
    registry.register("skill", "Manage skills (list|load <name>)", _handle_skill)
    registry.register("agent", "List available custom agents (user/project/plugin)", _handle_agent)
    registry.register("create-agent", "Create a new custom agent", _handle_create_agent)
    registry.register("team", "List available agent teams", _handle_team)
    registry.register("create-team", "Create a new agent team", _handle_create_team)
    registry.register("advisor", "Show advisor status / ask the advisor a question", _handle_advisor)
    registry.register("worktree", "Manage git worktrees (list|add <name>|remove <name>)", _handle_worktree)
    registry.register("hook", "List registered hooks", _handle_hook)
    registry.register("mcp", "List MCP servers and tools", _handle_mcp)
    registry.register("context", "Show context usage and token breakdown", _handle_context)
    registry.register("plan", "Plan a task (read-only mode, no file modifications)", _handle_plan)
    registry.register("quit", "Exit mAIke", _handle_quit)
    registry.register("exit", "Exit mAIke", _handle_quit)


def _write(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def _handle_help(session: Any, args: list[str]) -> None:
    _write("\nAvailable commands:\n")
    _write(session._command_registry.help_text())
    _write("\n\n")


def _handle_cost(session: Any, args: list[str]) -> None:
    budget_str = f"${session.budget:.2f}" if session.budget else "unlimited"
    _write(
        f"\nSession cost: ${session.total_cost:.4f}"
        f"  /  budget {budget_str}\n\n"
    )


def _handle_budget(session: Any, args: list[str]) -> None:
    if not session.budget:
        _write(f"\nBudget: unlimited  (used ${session.total_cost:.4f})\n\n")
        return
    remaining = session.budget - session.total_cost
    _write(
        f"\nBudget remaining: ${remaining:.4f}"
        f"  (used ${session.total_cost:.4f} of ${session.budget:.2f})\n\n"
    )


def _handle_history(session: Any, args: list[str]) -> None:
    if session.thread_id is None:
        _write("\nNo active thread yet. Start a task to begin.\n\n")
        return
    short = session.thread_id[:8]
    _write(
        f"\nThread: {short}\n"
        f"  Full ID: {session.thread_id}\n"
        f"  Run /new to start a fresh conversation\n\n"
    )


def _handle_new(session: Any, args: list[str]) -> None:
    session.thread_id = None
    session._force_new_thread = True
    _write("\nStarted a new conversation thread.\n\n")


def _handle_status(session: Any, args: list[str]) -> None:
    _write(f"\nProvider: {session.provider}  Model: {session.model}\n")
    _write(f"Budget: ${session.budget:.2f}  Used: ${session.total_cost:.4f}\n")
    _write(f"Workspace: {session.workspace}\n")
    if session.thread_id:
        _write(f"Thread: {session.thread_id[:8]}\n")
    _write("\n")


def _handle_tasks(session: Any, args: list[str]) -> None:
    task_registry = getattr(session, "_task_registry", None)
    if task_registry is None:
        _write("\nNo task registry available (run a task first).\n\n")
        return
    _write(f"\n{task_registry.format_table()}\n\n")


def _handle_plugin(session: Any, args: list[str]) -> None:
    session._handle_plugin(args)


def _handle_skill(session: Any, args: list[str]) -> None:
    session._handle_skill(args)


def _handle_agent(session: Any, args: list[str]) -> None:
    session._handle_agent(args)


async def _handle_create_agent(session: Any, args: list[str]) -> None:
    await session._handle_create_agent(args)


def _handle_team(session: Any, args: list[str]) -> None:
    session._handle_team(args)


def _handle_create_team(session: Any, args: list[str]) -> None:
    session._handle_create_team(args)


def _handle_advisor(session: Any, args: list[str]) -> None:
    """Show advisor status. Sub-command: `/advisor ask <question>` (REPL only)."""
    handler = getattr(session, "_handle_advisor", None)
    if handler is not None:
        handler(args)
        return
    # Fallback: render basic status from session attrs.
    enabled = getattr(session, "advisor_enabled", False)
    if not enabled:
        _write(
            "\nAdvisor is not enabled. Restart with --advisor (and optionally "
            "--advisor-provider/--advisor-model/--advisor-budget-pct) to enable.\n\n"
        )
        return
    provider = getattr(session, "advisor_provider", None) or "(default — same as executor)"
    model = getattr(session, "advisor_model", None) or "(strong-tier)"
    budget_pct = getattr(session, "advisor_budget_pct", 0.20)
    _write(
        f"\nAdvisor enabled\n"
        f"  Provider: {provider}\n"
        f"  Model: {model}\n"
        f"  Budget: {int(budget_pct * 100)}% of session budget\n\n"
    )


def _handle_worktree(session: Any, args: list[str]) -> None:
    session._handle_worktree(args)


def _handle_hook(session: Any, args: list[str]) -> None:
    session._handle_hook(args)


def _handle_mcp(session: Any, args: list[str]) -> None:
    session._handle_mcp(args)


def _handle_context(session: Any, args: list[str]) -> None:
    """Show context usage: token breakdown, compression, telemetry."""
    from maike.constants import context_limit_for_model

    _write("\nContext Usage:\n")

    # Model context limit.
    model = getattr(session, "model", "unknown")
    try:
        limit = context_limit_for_model(model)
        _write(f"  Model:             {model}\n")
        _write(f"  Context limit:     {limit:,} tokens\n")
    except Exception:
        _write(f"  Model:             {model} (limit unknown)\n")
        limit = 0

    # Last run telemetry.
    meta = getattr(session, "_last_run_metadata", {})
    telemetry = meta.get("context_telemetry", {})
    llm_usage = meta.get("llm_usage", {})

    if telemetry:
        initial = telemetry.get("initial_context_tokens", 0)
        peak = telemetry.get("peak_conversation_tokens", 0)
        prune_events = telemetry.get("prune_events", 0)
        tokens_pruned = telemetry.get("tokens_pruned", 0)
        compression = telemetry.get("compression_applied", False)
        levels = telemetry.get("compression_levels_used", [])

        _write(f"  Initial context:   {initial:,} tokens\n")
        _write(f"  Peak conversation: {peak:,} tokens\n")
        if limit:
            pct = (peak / limit) * 100
            _write(f"  Usage:             {pct:.1f}%\n")
        _write(f"  Prune events:      {prune_events}\n")
        if tokens_pruned:
            _write(f"  Tokens pruned:     {tokens_pruned:,}\n")
        _write(f"  Compression:       {'active (' + ', '.join(levels) + ')' if compression else 'none'}\n")

        # Tool call summary.
        tool_summary = telemetry.get("tool_call_summary", {})
        total_calls = tool_summary.get("total_calls", 0)
        if total_calls:
            _write(f"\n  Tool calls:        {total_calls}\n")
            by_tool = tool_summary.get("by_tool", {})
            for tname, tdata in sorted(by_tool.items()):
                count = tdata.get("count", 0)
                _write(f"    {tname:16s} {count} calls\n")
    else:
        _write("  No telemetry available — run a task first.\n")

    # LLM usage from last run.
    if llm_usage:
        calls = llm_usage.get("calls", 0)
        inp = llm_usage.get("input_tokens", 0)
        out = llm_usage.get("output_tokens", 0)
        cost = llm_usage.get("cost_usd", 0)
        _write(f"\n  LLM calls:         {calls}\n")
        _write(f"  Input tokens:      {inp:,}\n")
        _write(f"  Output tokens:     {out:,}\n")
        _write(f"  LLM cost:          ${cost:.4f}\n")

    _write("\n")


async def _handle_plan(session: Any, args: list[str]) -> None:
    """Run a task in plan mode — read-only, structured output."""
    if not args:
        _write("\nUsage: /plan <task description>\n")
        _write("Example: /plan add authentication to the REST API\n\n")
        return

    task = " ".join(args)
    await session._run_plan_task(task)


def _handle_quit(session: Any, args: list[str]) -> None:
    raise SystemExit(0)
