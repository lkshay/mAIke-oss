"""Custom Textual messages for cross-widget communication."""

from __future__ import annotations

import asyncio
from typing import Any

from textual.message import Message


class ToolStarted(Message):
    """A tool has started executing."""

    def __init__(self, tool_name: str, hint: str = "", activity: str = "") -> None:
        super().__init__()
        self.tool_name = tool_name
        self.hint = hint
        self.activity = activity


class ToolCompleted(Message):
    """A tool has finished executing."""

    def __init__(
        self, tool_name: str, success: bool, output: str = "", error: str | None = None,
    ) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.error = error


class LLMCallStarted(Message):
    """An LLM call has started."""

    def __init__(self, provider: str = "", model: str = "") -> None:
        super().__init__()
        self.provider = provider
        self.model = model


class LLMCallCompleted(Message):
    """An LLM call has completed."""

    def __init__(
        self, model: str = "", tokens: int = 0, cost: float = 0.0, thinking: str = "",
    ) -> None:
        super().__init__()
        self.model = model
        self.tokens = tokens
        self.cost = cost
        self.thinking = thinking


class StreamToken(Message):
    """A streaming text delta from the LLM."""

    def __init__(self, text_delta: str) -> None:
        super().__init__()
        self.text_delta = text_delta


class StreamFinished(Message):
    """Streaming has ended for the current LLM turn."""

    def __init__(self, full_text: str = "") -> None:
        super().__init__()
        self.full_text = full_text


class ApprovalRequested(Message):
    """A tool requires user approval before executing."""

    def __init__(self, prompt: str, future: asyncio.Future) -> None:
        super().__init__()
        self.prompt = prompt
        self.future = future


class TextInputRequested(Message):
    """A tool (e.g. AskUser) needs free-text input from the user."""

    def __init__(self, prompt: str, future: asyncio.Future) -> None:
        super().__init__()
        self.prompt = prompt
        self.future = future


class TurnComplete(Message):
    """An agent turn has completed. Re-enable input."""

    def __init__(self, result: Any = None) -> None:
        super().__init__()
        self.result = result


class UserSubmitted(Message):
    """User submitted input from the prompt."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StatusUpdate(Message):
    """Update the status bar with new metrics."""

    def __init__(
        self,
        model: str = "",
        provider: str = "",
        cost: float = 0.0,
        tokens: int = 0,
        iteration: int = 0,
        max_iterations: int = 0,
        budget: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.provider = provider
        self.cost = cost
        self.tokens = tokens
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.budget = budget


class AgentOutputReady(Message):
    """Final agent text output is ready for display."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
