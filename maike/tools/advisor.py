"""Standalone Advisor tool — consult a frontier model for strategic advice.

The Advisor tool is a read-only call: it never executes code, never writes
files, never runs commands. The frontier model reviews a compressed
transcript of the current session and returns short strategic guidance.

Schema is intentionally minimal (2 params) so small-model function calling
stays reliable — same approach as the Team tool.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.tools.registry import ToolRegistry


def register_advisor_tool(
    registry: ToolRegistry,
    advisor_handler: Callable[..., Awaitable[ToolResult]],
) -> None:
    """Register the ``Advisor`` tool.

    The *advisor_handler* callback is built by the orchestrator and routes
    to ``AdvisorSession.advise()``.  It must accept keyword args:
    ``question`` (str) and ``urgency`` ("normal" | "stuck").
    """

    async def invoke_advisor(
        question: str,
        urgency: str = "normal",
    ) -> ToolResult:
        """Ask a frontier advisor for strategic advice."""
        q = (question or "").strip()
        if not q:
            return ToolResult(
                tool_name="Advisor",
                success=False,
                output=(
                    "question is required. Phrase it as a specific decision point, "
                    "e.g. 'is my plan to refactor X into Y sound?' or "
                    "'why does this test keep failing after my edits?'"
                ),
            )
        normalized_urgency = (urgency or "normal").lower().strip()
        if normalized_urgency not in {"normal", "stuck"}:
            normalized_urgency = "normal"
        return await advisor_handler(
            question=q,
            urgency=normalized_urgency,
        )

    registry.register(
        schema=ToolSchema(
            name="Advisor",
            description=(
                "Ask a frontier model for strategic advice when stuck, "
                "uncertain about approach, or before a major direction change. "
                "Does NOT execute code or touch files — returns advice only. "
                "Use sparingly: good moments are after exploration (before "
                "implementing), after repeated failures, or when choosing "
                "between two approaches. The advisor sees a compressed view "
                "of your session."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "Specific question or decision point. Be concrete "
                            "— 'should I use strategy A or B here?' works "
                            "better than 'any advice?'"
                        ),
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["normal", "stuck"],
                        "description": (
                            "'stuck' if repeated attempts have failed; "
                            "'normal' otherwise. Defaults to 'normal'."
                        ),
                    },
                },
                "required": ["question"],
            },
        ),
        fn=invoke_advisor,
        risk_level=RiskLevel.READ,
    )
