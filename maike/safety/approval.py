"""Interactive approval and user-input gate."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from maike.atoms.context import AgentContext


@dataclass(frozen=True)
class ApprovalResult:
    """Result of an approval request — approved/denied plus optional user feedback."""

    approved: bool
    feedback: str = ""

    @property
    def has_feedback(self) -> bool:
        return bool(self.feedback.strip())


class ApprovalGate:
    def __init__(
        self,
        auto_approve: bool = False,
        input_fn: Callable[[str], str] = input,
        on_prompt: Callable[[str], bool] | None = None,
        on_text_prompt: Callable[[str], str] | None = None,
    ) -> None:
        self.auto_approve = auto_approve
        self.input_fn = input_fn
        self.on_prompt = on_prompt
        self.on_text_prompt = on_text_prompt

    async def request(
        self,
        tool_call: dict,
        ctx: AgentContext,
        prompt: str | None = None,
    ) -> ApprovalResult:
        """Request approval for a tool call.

        Returns an ApprovalResult with approved=True/False and optional
        user feedback.  The user can type 'y'/'yes' to approve, 'n'/'no'
        to deny, or any other text which is treated as approval + feedback
        (e.g. "yes but use pytest instead of unittest").
        """
        if self.auto_approve:
            return ApprovalResult(approved=True)
        rendered_prompt = prompt or (
            f"Approve tool '{tool_call['name']}' for agent {ctx.role} "
            f"during stage '{ctx.stage_name}'? [y/N/feedback]: "
        )
        return await self.confirm_with_feedback(rendered_prompt)

    async def confirm(self, prompt: str) -> bool:
        """Simple yes/no confirmation (backward compat)."""
        result = await self.confirm_with_feedback(prompt)
        return result.approved

    async def confirm_with_feedback(self, prompt: str) -> ApprovalResult:
        """Prompt user and return ApprovalResult with optional feedback.

        - 'y', 'yes', '' → approved, no feedback
        - 'n', 'no' → denied, no feedback
        - anything else → approved, with the text as feedback
          (lets user say "yes but use X instead of Y")
        """
        if self.auto_approve:
            return ApprovalResult(approved=True)
        if self.on_prompt:
            result = await asyncio.to_thread(self.on_prompt, prompt)
            return ApprovalResult(approved=bool(result))
        try:
            response = await asyncio.to_thread(self.input_fn, prompt)
        except (EOFError, KeyboardInterrupt):
            return ApprovalResult(approved=False)
        stripped = response.strip()
        lowered = stripped.lower()
        if lowered in {"n", "no"}:
            return ApprovalResult(approved=False)
        if lowered in {"y", "yes", ""}:
            return ApprovalResult(approved=True)
        # Any other text = approval + feedback
        return ApprovalResult(approved=True, feedback=stripped)

    async def prompt_text(self, prompt: str) -> str:
        """Prompt the user and return their free-text response.

        Used by AskUser to get substantive guidance from the user. Unlike
        tool-approval prompts, AskUser is the agent asking the user a real
        question — ``--yes`` deliberately does NOT short-circuit it, since
        bypassing the question with a canned response defeats the point.
        Tool calls (Read/Write/Edit/Bash/...) still honor ``auto_approve``
        via ``request`` / ``confirm`` — only AskUser is exempt.

        When stdin is unavailable (eval/CI with no TTY), falls back to a
        message that signals the agent to proceed on its own judgment.
        """
        if self.on_text_prompt:
            return await asyncio.to_thread(self.on_text_prompt, prompt)
        try:
            return await asyncio.to_thread(self.input_fn, prompt)
        except (EOFError, KeyboardInterrupt):
            return (
                "No user is available to answer (non-interactive session). "
                "Apply your best judgment from the existing task and context."
            )
