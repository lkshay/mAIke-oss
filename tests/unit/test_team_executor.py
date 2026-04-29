"""Tests for team execution — parallel, sequential, and blackboard."""

from __future__ import annotations

import asyncio

from maike.agents.team_executor import (
    MemberResult,
    _build_sequential_context,
    compute_member_budgets,
    execute_team,
)
from maike.agents.team_resolver import TeamDefinition, TeamMember
from maike.atoms.tool import ToolResult


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------


def _make_team(
    n_members: int = 3,
    weights: list[float] | None = None,
    **kwargs,
) -> TeamDefinition:
    members = []
    for i in range(n_members):
        w = weights[i] if weights else 1.0
        members.append(TeamMember(
            role=f"Member {i}",
            agent_type="explore",
            budget_weight=w,
        ))
    return TeamDefinition(
        name="test-team",
        description="Test team",
        members=members,
        source="project",
        **kwargs,
    )


def test_budget_equal_split():
    team = _make_team(3)
    member_budgets, synthesis = compute_member_budgets(team, 1.00)
    assert len(member_budgets) == 3
    assert abs(synthesis - 0.15) < 0.01
    assert abs(sum(member_budgets) - 0.85) < 0.01


def test_budget_weighted_split():
    team = _make_team(2, weights=[1.0, 3.0])
    member_budgets, synthesis = compute_member_budgets(team, 1.00)
    assert len(member_budgets) == 2
    assert member_budgets[1] > member_budgets[0] * 2.5


def test_budget_single_member():
    team = _make_team(1)
    member_budgets, synthesis = compute_member_budgets(team, 1.00)
    assert len(member_budgets) == 1
    assert abs(member_budgets[0] - 0.85) < 0.01


# ---------------------------------------------------------------------------
# Fake spawn helpers
# ---------------------------------------------------------------------------


async def _fake_spawn_member(**kwargs) -> ToolResult:
    """Fake spawn that returns the task as output."""
    task = kwargs.get("task", "no task")
    return ToolResult(
        tool_name="Team",
        success=True,
        output=f"Completed: {task[:80]}",
        metadata={"cost_usd": 0.01, "tokens_used": 100},
    )


async def _fake_spawn_fail(**kwargs) -> ToolResult:
    """Fake spawn that always fails."""
    return ToolResult(
        tool_name="Team",
        success=False,
        output="Something went wrong",
        error="test_error",
        metadata={"cost_usd": 0.005, "tokens_used": 50},
    )


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def test_parallel_execution_success():
    team = _make_team(3)
    result = asyncio.run(execute_team(
        team=team,
        task="Analyze the auth module",
        context="",
        total_budget_usd=1.00,
        total_token_budget=100_000,
        spawn_member=_fake_spawn_member,
    ))
    assert result.success
    assert "test-team" in result.output
    assert "3 members" in result.output
    assert "0 failed" in result.output


def test_parallel_execution_with_failure_continue():
    """on_failure=continue: failed members noted, others proceed."""
    members = [
        TeamMember(role="Good member", agent_type="explore"),
        TeamMember(role="Bad member", agent_type="explore"),
    ]
    team = TeamDefinition(
        name="mixed-team",
        description="Team with one failing member",
        members=members,
        source="project",
        on_failure="continue",
    )

    call_count = 0

    async def _mixed_spawn(**kwargs):
        nonlocal call_count
        call_count += 1
        task = kwargs.get("task", "")
        if "Bad member" in task:
            return ToolResult(
                tool_name="Team", success=False,
                output="I failed", error="bad",
                metadata={"cost_usd": 0, "tokens_used": 0},
            )
        return ToolResult(
            tool_name="Team", success=True,
            output="I succeeded",
            metadata={"cost_usd": 0.01, "tokens_used": 100},
        )

    result = asyncio.run(execute_team(
        team=team,
        task="Do work",
        context="",
        total_budget_usd=1.00,
        total_token_budget=100_000,
        spawn_member=_mixed_spawn,
    ))

    assert result.success
    assert "1 failed" in result.output
    assert call_count >= 2


def test_too_many_members():
    team = _make_team(10)
    result = asyncio.run(execute_team(
        team=team,
        task="Task",
        context="",
        total_budget_usd=1.00,
        total_token_budget=100_000,
        spawn_member=_fake_spawn_member,
    ))
    assert not result.success
    assert "maximum" in result.output.lower()


def test_missing_agent_reference():
    class FakeResolver:
        def resolve(self, name):
            return None
        def list_names(self):
            return ["existing-agent"]

    members = [TeamMember(agent="nonexistent", role="Does not exist")]
    team = TeamDefinition(
        name="bad-ref",
        description="Team with bad agent reference",
        members=members,
        source="project",
    )

    result = asyncio.run(execute_team(
        team=team,
        task="Task",
        context="",
        total_budget_usd=1.00,
        total_token_budget=100_000,
        spawn_member=_fake_spawn_member,
        agent_resolver=FakeResolver(),
    ))
    assert not result.success
    assert "nonexistent" in result.output


# ---------------------------------------------------------------------------
# Sequential execution
# ---------------------------------------------------------------------------


def test_sequential_execution_success():
    """Sequential team runs members in order."""
    team = _make_team(3, process_type="sequential")

    received_tasks: list[str] = []

    async def _tracking_spawn(**kwargs):
        task = kwargs.get("task", "")
        received_tasks.append(task)
        return ToolResult(
            tool_name="Team", success=True,
            output=f"Result from step {len(received_tasks)}",
            metadata={"cost_usd": 0.01, "tokens_used": 100},
        )

    result = asyncio.run(execute_team(
        team=team,
        task="Multi-step analysis",
        context="",
        total_budget_usd=1.00,
        total_token_budget=100_000,
        spawn_member=_tracking_spawn,
    ))

    assert result.success
    # 3 members + 1 synthesis = 4 total calls.
    assert len(received_tasks) == 4
    # Second member should see first member's output.
    assert "Previous Results" in received_tasks[1]
    assert "Result from step 1" in received_tasks[1]
    # Third member should see both previous outputs.
    assert "Result from step 2" in received_tasks[2]


def test_sequential_abort_on_failure():
    """Sequential with on_failure=abort stops after first failure."""
    team = _make_team(3, process_type="sequential", on_failure="abort")

    call_count = 0

    async def _fail_on_second(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return ToolResult(
                tool_name="Team", success=False,
                output="Failed at step 2", error="step2_fail",
                metadata={"cost_usd": 0, "tokens_used": 0},
            )
        return ToolResult(
            tool_name="Team", success=True,
            output=f"OK step {call_count}",
            metadata={"cost_usd": 0.01, "tokens_used": 100},
        )

    result = asyncio.run(execute_team(
        team=team,
        task="Pipeline",
        context="",
        total_budget_usd=1.00,
        total_token_budget=100_000,
        spawn_member=_fail_on_second,
    ))

    assert result.success  # Team always returns success with synthesis
    # Should have stopped after 2nd member (abort) + synthesis = 3 calls.
    assert call_count == 3  # member1 + member2(fail) + synthesis


# ---------------------------------------------------------------------------
# Sequential context builder
# ---------------------------------------------------------------------------


def test_build_sequential_context_empty():
    assert _build_sequential_context([]) == ""


def test_build_sequential_context_with_results():
    results = [
        MemberResult(0, "agent-a", "Role A", "completed", "Output A", 0.01, 100),
        MemberResult(1, "agent-b", "Role B", "completed", "Output B", 0.01, 100),
    ]
    ctx = _build_sequential_context(results)
    assert "Previous Results" in ctx
    assert "agent-a" in ctx
    assert "Output A" in ctx
    assert "agent-b" in ctx
    assert "Output B" in ctx


def test_build_sequential_context_truncates_old():
    """When total exceeds cap, older results are summarized."""
    results = [
        MemberResult(0, "old", "Old", "completed", "x" * 3000, 0, 0),
        MemberResult(1, "mid", "Mid", "completed", "y" * 3000, 0, 0),
        MemberResult(2, "new", "New", "completed", "z" * 3000, 0, 0),
    ]
    ctx = _build_sequential_context(results, cap=6000)
    # Oldest result should be truncated.
    assert "truncated" in ctx
    # Newest results should be full.
    assert "z" * 100 in ctx


# ---------------------------------------------------------------------------
# Team creation commands
# ---------------------------------------------------------------------------


def test_preview_team_markdown():
    from maike.agents.team_commands import TeamWizardData, TeamWizardMemberData, preview_team_markdown

    data = TeamWizardData(
        name="review-team",
        description="Code review pipeline",
        process_type="parallel",
        on_failure="continue",
        members=[
            TeamWizardMemberData(agent="doc-auditor", role="Check docs"),
            TeamWizardMemberData(role="Style reviewer"),
        ],
        synthesis_prompt="Combine all findings.",
    )
    md = preview_team_markdown(data)
    assert "name: review-team" in md
    assert "process: parallel" in md
    assert "agent: doc-auditor" in md
    assert "role: Style reviewer" in md
    assert "Combine all findings" in md


def test_create_team_file(tmp_path):
    from maike.agents.team_commands import TeamWizardData, TeamWizardMemberData, create_team_file

    data = TeamWizardData(
        name="test-team",
        description="A test team",
        members=[
            TeamWizardMemberData(agent="agent-a", role="Do A"),
        ],
    )
    path = create_team_file(data, workspace=tmp_path)
    assert path.exists()
    assert "test-team" in path.name
    content = path.read_text()
    assert "agent: agent-a" in content

    # Should be parseable by TeamResolver.
    from maike.agents.team_resolver import TeamResolver
    resolver = TeamResolver(project_dir=path.parent)
    defn = resolver.resolve("test-team")
    assert defn is not None
    assert len(defn.members) == 1
