"""Standalone Team tool — invokes a team of agents on a task.

Separate from the Delegate tool to keep the schema minimal (2 required
params: ``name`` and ``task``).  Gemini's function calling is more
reliable with simple schemas than with the 10-parameter Delegate tool.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.tools.registry import ToolRegistry


def register_team_tool(
    registry: ToolRegistry,
    team_handler: Callable[..., Awaitable[ToolResult]],
) -> None:
    """Register the ``Team`` tool.

    The *team_handler* callback is built by the orchestrator and routes
    to ``TeamExecutor.execute_team()``.
    """

    async def invoke_team(
        name: str,
        task: str,
        context: str = "",
    ) -> ToolResult:
        """Invoke an agent team by name."""
        if not name:
            return ToolResult(
                tool_name="Team",
                success=False,
                output="name is required. Use /team list to see available teams.",
            )
        if not task:
            return ToolResult(
                tool_name="Team",
                success=False,
                output="task is required — describe what the team should do.",
            )
        return await team_handler(team=name, task=task, context=context)

    registry.register(
        schema=ToolSchema(
            name="Team",
            description=(
                "Invoke a team of agents that work in parallel on a task. "
                "Each team member runs independently, then results are "
                "synthesized into a single report. "
                "Use /team list or check the Available Teams section in "
                "your context to see team names."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Name of the team to invoke (e.g. 'code-health-team'). "
                            "See Available Teams in your context."
                        ),
                    },
                    "task": {
                        "type": "string",
                        "description": (
                            "The task for the team to perform. All team members "
                            "receive this task along with their role-specific context."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional additional context to pass to all team members."
                        ),
                    },
                },
                "required": ["name", "task"],
            },
        ),
        fn=invoke_team,
        risk_level=RiskLevel.WRITE,
    )
