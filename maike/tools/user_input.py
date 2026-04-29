"""User input tool — allows the agent to pause and ask the user a question."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.tools.registry import ToolRegistry


def register_user_input_tool(
    registry: ToolRegistry,
    user_input_handler: Callable[[str, str], Awaitable[ToolResult]],
) -> None:
    """Register the ``request_user_input`` tool.

    *user_input_handler* is an async callback (injected by the orchestrator)
    that pauses execution, displays the prompt to the user, and returns their
    response wrapped in a ``ToolResult``.
    """

    async def request_user_input(
        prompt: str,
        context_summary: str = "",
    ) -> ToolResult:
        """Pause execution and ask the user a question.

        Use this before making significant decisions (e.g., CLI flag names,
        architecture choices) so the user can confirm or redirect.
        """
        return await user_input_handler(prompt, context_summary)

    registry.register(
        ToolSchema(
            name="AskUser",
            description=(
                "Pause and ask the user a question.  Use before making key "
                "decisions like CLI interface design, file structure, or "
                "architecture choices.  Returns the user's text response. "
                'Example: {"prompt": "I plan to use --title and --tag flags '
                'for search. OK?", "context_summary": "Building bookmark CLI"}'
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The question to ask the user.  Be specific about "
                            "what decision you need input on."
                        ),
                    },
                    "context_summary": {
                        "type": "string",
                        "description": "Brief summary of what you've done so far.",
                    },
                },
                "required": ["prompt"],
            },
        ),
        fn=request_user_input,
        risk_level=RiskLevel.READ,
    )
