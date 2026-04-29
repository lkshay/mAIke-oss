"""Team executor — parallel and sequential execution for agent teams.

Spawns team members as delegates, coordinates their execution based on
the team's ``process_type`` (parallel or sequential), and synthesizes
results.  Uses a per-team Blackboard for inter-member communication.

Does NOT create its own ``AgentCore`` or ``LLMGateway`` — receives
callbacks from the orchestrator that reuse the session's existing
infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from maike.agents.team_resolver import TeamDefinition, TeamMember
from maike.atoms.tool import ToolResult
from maike.constants import MAX_TEAM_MEMBERS, TEAM_SYNTHESIS_BUDGET_FRACTION

logger = logging.getLogger(__name__)

# Cap for sequential context injection — keeps member prompts manageable.
_SEQUENTIAL_CONTEXT_CAP = 6000
_SEQUENTIAL_SUMMARY_CAP = 500


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MemberResult:
    """Outcome of a single team member's execution."""

    member_index: int
    display_name: str
    role: str
    status: str           # "completed" or "failed"
    output: str
    cost_usd: float
    tokens_used: int
    error: str | None = None


@dataclass
class TeamExecutionResult:
    """Outcome of a full team execution."""

    team_name: str
    member_results: list[MemberResult]
    synthesis_output: str
    total_cost_usd: float
    total_tokens: int
    failed_members: list[str]


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------

def compute_member_budgets(
    team: TeamDefinition,
    total_budget_usd: float,
) -> tuple[list[float], float]:
    """Compute per-member budgets and synthesis budget.

    Returns ``(member_budgets, synthesis_budget)``.
    """
    synthesis = total_budget_usd * TEAM_SYNTHESIS_BUDGET_FRACTION
    pool = total_budget_usd - synthesis

    total_weight = sum(m.budget_weight for m in team.members)
    if total_weight <= 0:
        total_weight = len(team.members)

    budgets = [
        pool * (m.budget_weight / total_weight)
        for m in team.members
    ]
    return budgets, synthesis


# ---------------------------------------------------------------------------
# Sequential context builder
# ---------------------------------------------------------------------------

def _build_sequential_context(
    prior_results: list[MemberResult],
    cap: int = _SEQUENTIAL_CONTEXT_CAP,
) -> str:
    """Build context string from prior member outputs for sequential mode.

    If total output exceeds *cap*, only the last 2 members are included
    in full; earlier ones are summarized.
    """
    if not prior_results:
        return ""

    parts: list[str] = []
    total_len = sum(len(mr.output) for mr in prior_results)

    for i, mr in enumerate(prior_results):
        status = "COMPLETED" if mr.status == "completed" else "FAILED"
        header = f"### {mr.display_name} [{status}]"

        if total_len > cap and i < len(prior_results) - 2:
            # Summarize older results.
            summary = mr.output[:_SEQUENTIAL_SUMMARY_CAP]
            if len(mr.output) > _SEQUENTIAL_SUMMARY_CAP:
                summary += "... (truncated)"
            parts.append(f"{header}\n{summary}")
        else:
            parts.append(f"{header}\n{mr.output}")

    return "## Previous Results\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

# Type for the spawn callback provided by the orchestrator.
SpawnMemberFn = Callable[..., Awaitable[ToolResult]]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

# Type for the spawn callback provided by the orchestrator.
SpawnMemberFn = Callable[..., Awaitable[ToolResult]]

async def _execute_single_member(
    idx: int,
    member: TeamMember,
    budget_usd: float,
    token_budget: int,
    task: str,
    team_context: str,
    spawn_member: SpawnMemberFn,
    team_board: 'Blackboard',
    extra_context: str = "",
) -> MemberResult:
    """Spawn and await a single team member."""
    role_desc = member.role or f"Execute your speciality as the '{member.agent}' agent"

    task_parts = []
    if extra_context:
        task_parts.append(extra_context)
    task_parts.extend([f"## Team Role\n{role_desc}", f"\n## Task\n{task}"])
    member_task = "\n\n".join(task_parts)

    try:
        result = await spawn_member(
            task=member_task,
            context=team_context,
            model_tier=member.model_tier,
            plugin_agent=member.agent,
            agent_type=member.agent_type,
            remaining_cost_budget_usd=budget_usd,
            remaining_token_budget=token_budget,
        )
        mr = MemberResult(
            member_index=idx,
            display_name=member.display_name,
            role=role_desc,
            status="completed" if result.success else "failed",
            output=result.output or "(no output)",
            cost_usd=result.metadata.get("cost_usd", 0) if result.metadata else 0,
            tokens_used=result.metadata.get("tokens_used", 0) if result.metadata else 0,
            error=result.error if not result.success else None,
        )
    except Exception as exc:
        mr = MemberResult(
            member_index=idx,
            display_name=member.display_name,
            role=role_desc,
            status="failed",
            output=f"Member failed with exception: {exc}",
            cost_usd=0,
            tokens_used=0,
            error=str(exc),
        )

    team_board.post(f"member:{mr.display_name}", mr.output)
    return mr

async def _synthesize_results(
    team: TeamDefinition,
    task: str,
    member_results: list[MemberResult],
    team_board: 'Blackboard',
    spawn_member: SpawnMemberFn,
    synthesis_budget: float,
    total_token_budget: int,
) -> tuple[str, float, int]:
    """Synthesize member results into a final report."""
    synthesis_parts = [
        f"## Team Execution Results: {team.name}",
        f"Task: {task}",
        f"Process: {team.process_type}",
        f"Members: {len(member_results)} ({len([r for r in member_results if r.status == 'failed'])} failed)",
    ]
    for mr in member_results:
        status_icon = "COMPLETED" if mr.status == "completed" else "FAILED"
        synthesis_parts.append(
            f"\n### Member: {mr.display_name} [{status_icon}]\nRole: {mr.role}\n\n{mr.output}\n"
        )

    if board_state := team_board.read(None):
        synthesis_parts.append(f"\n## Blackboard\n```json\n{json.dumps(board_state, indent=2)}\n```\n")

    synthesis_context = "\n".join(synthesis_parts)
    synth_task = team.synthesis_prompt or (
        "Synthesize the results from all team members into a single coherent report. "
        "Deduplicate overlapping findings. Note any gaps from failed members."
    )

    total_cost = 0
    total_tokens = 0
    try:
        synth_result = await spawn_member(
            task=synth_task,
            context=synthesis_context,
            model_tier="cheap",
            plugin_agent=None,
            agent_type="plan",
            remaining_cost_budget_usd=synthesis_budget,
            remaining_token_budget=total_token_budget // 4,
        )
        synthesis_output = synth_result.output or "(synthesis produced no output)"
        if synth_result.metadata:
            total_cost += synth_result.metadata.get("cost_usd", 0)
            total_tokens += synth_result.metadata.get("tokens_used", 0)
    except Exception as exc:
        logger.warning("Team synthesis failed: %s", exc)
        synthesis_output = synthesis_context

    return synthesis_output, total_cost, total_tokens

def _create_team_tool_result(team, exec_result: TeamExecutionResult) -> ToolResult:
    """Create the final ToolResult for the team execution."""
    failed_note = f"\n\nNote: {len(exec_result.failed_members)} member(s) failed: {', '.join(exec_result.failed_members)}" if exec_result.failed_members else ""
    output = (
        f"Team '{team.name}' completed ({len(exec_result.member_results)} members, "
        f"{len(exec_result.failed_members)} failed, ${exec_result.total_cost_usd:.3f}).\n\n"
        f"{exec_result.synthesis_output}{failed_note}"
    )
    return ToolResult(
        tool_name="Team",
        success=True,
        output=output,
        metadata={
            "team_name": team.name,
            "member_count": len(exec_result.member_results),
            "failed_count": len(exec_result.failed_members),
            "cost_usd": exec_result.total_cost_usd,
            "tokens_used": exec_result.total_tokens,
        },
    )

async def execute_team(
    team: TeamDefinition,
    task: str,
    context: str,
    total_budget_usd: float,
    total_token_budget: int,
    spawn_member: SpawnMemberFn,
    agent_resolver: Any | None = None,
) -> ToolResult:
    """Execute a team definition."""
    if len(team.members) > MAX_TEAM_MEMBERS:
        return ToolResult(tool_name="Team", success=False, output=f"Team '{team.name}' has {len(team.members)} members but the maximum is {MAX_TEAM_MEMBERS}.", error="too_many_members")

    if agent_resolver:
        for member in team.members:
            if member.agent and not agent_resolver.resolve(member.agent):
                return ToolResult(tool_name="Team", success=False, output=f"Team member references agent '{member.agent}' which does not exist. Available agents: {agent_resolver.list_names()}", error="agent_not_found")

    member_budgets, synthesis_budget = compute_member_budgets(team, total_budget_usd)
    member_token_budgets = [int(total_token_budget * (b / total_budget_usd)) if total_budget_usd > 0 else total_token_budget // (len(team.members) + 1) for b in member_budgets]

    from maike.tools.blackboard import Blackboard
    team_board = Blackboard()

    team_context_parts = [f"## Team: {team.name}", f"Description: {team.description}", f"Process: {team.process_type}", "Your role in the team: see task below."]
    if context:
        team_context_parts.append(f"\n## Parent Context\n{context}")
    team_context = "\n".join(team_context_parts)

    if team.process_type == "sequential":
        member_results = await _execute_sequential(team, member_budgets, member_token_budgets, task, team_context, spawn_member, team_board)
    else:
        member_results = await _execute_parallel(team, member_budgets, member_token_budgets, task, team_context, spawn_member, team_board)

    synthesis_output, synth_cost, synth_tokens = await _synthesize_results(team, task, member_results, team_board, spawn_member, synthesis_budget, total_token_budget)

    total_cost = sum(mr.cost_usd for mr in member_results) + synth_cost
    total_tokens = sum(mr.tokens_used for mr in member_results) + synth_tokens
    failed = [mr.display_name for mr in member_results if mr.status == "failed"]

    exec_result = TeamExecutionResult(team_name=team.name, member_results=member_results, synthesis_output=synthesis_output, total_cost_usd=total_cost, total_tokens=total_tokens, failed_members=failed)

    return _create_team_tool_result(team, exec_result)


# ---------------------------------------------------------------------------
# Execution strategies
# ---------------------------------------------------------------------------

async def _execute_parallel(
    team: TeamDefinition,
    member_budgets: list[float],
    member_token_budgets: list[int],
    task: str,
    team_context: str,
    spawn_member: SpawnMemberFn,
    team_board: 'Blackboard',
) -> list[MemberResult]:
    """Fan-out: spawn all members concurrently via asyncio.gather."""
    member_coros = [_execute_single_member(i, member, member_budgets[i], member_token_budgets[i], task, team_context, spawn_member, team_board) for i, member in enumerate(team.members)]

    if team.on_failure == "abort":
        try:
            return list(await asyncio.gather(*member_coros))
        except Exception:
            return []
    else:
        raw_results = await asyncio.gather(*member_coros, return_exceptions=True)
        member_results = [r if not isinstance(r, Exception) else MemberResult(member_index=i, display_name=team.members[i].display_name, role=team.members[i].role or "", status="failed", output=f"Exception: {r}", cost_usd=0, tokens_used=0, error=str(r)) for i, r in enumerate(raw_results)]

        if team.on_failure == "retry":
            for i, mr in enumerate(member_results):
                if mr.status == "failed":
                    logger.info("Retrying failed team member: %s", mr.display_name)
                    member_results[i] = await _execute_single_member(i, team.members[i], member_budgets[i] * 0.5, member_token_budgets[i], task, team_context, spawn_member, team_board)
        return member_results

async def _execute_sequential(
    team: TeamDefinition,
    member_budgets: list[float],
    member_token_budgets: list[int],
    task: str,
    team_context: str,
    spawn_member: SpawnMemberFn,
    team_board: 'Blackboard',
) -> list[MemberResult]:
    """Pipeline: run members in order, passing prior outputs as context."""
    member_results: list[MemberResult] = []
    for i, member in enumerate(team.members):
        extra_context = _build_sequential_context(member_results)
        mr = await _execute_single_member(i, member, member_budgets[i], member_token_budgets[i], task, team_context, spawn_member, team_board, extra_context=extra_context)
        member_results.append(mr)

        if mr.status == "failed":
            if team.on_failure == "abort":
                break
            if team.on_failure == "retry":
                logger.info("Retrying failed team member: %s", mr.display_name)
                retry = await _execute_single_member(i, member, member_budgets[i] * 0.5, member_token_budgets[i], task, team_context, spawn_member, team_board, extra_context=extra_context)
                member_results[i] = retry
    return member_results
