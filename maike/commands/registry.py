"""Command registry for slash commands."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SlashCommand:
    """A registered slash command."""

    name: str
    description: str
    handler: Callable[[Any, list[str]], None]  # (session, args) -> None

    def __hash__(self) -> int:
        return hash(self.name)


class CommandRegistry:
    """Registry of slash commands with dispatch and help generation."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, name: str, description: str, handler: Callable) -> None:
        """Register a command. Name should not include the leading ``/``."""
        self._commands[name.lower()] = SlashCommand(
            name=name.lower(),
            description=description,
            handler=handler,
        )

    def dispatch(self, text: str, session: Any) -> bool:
        """Parse and dispatch a slash command (sync handlers only).

        *text* should include the leading ``/``.
        Returns ``True`` if the command was found and executed.
        """
        parts = text.lstrip("/").split(None, 1)
        name = parts[0].lower() if parts else ""
        args = parts[1].split() if len(parts) > 1 else []

        cmd = self._commands.get(name)
        if cmd is None:
            return False
        cmd.handler(session, args)
        return True

    async def async_dispatch(self, text: str, session: Any) -> bool:
        """Parse and dispatch a slash command, supporting async handlers.

        Returns ``True`` if the command was found and executed.
        """
        parts = text.lstrip("/").split(None, 1)
        name = parts[0].lower() if parts else ""
        args = parts[1].split() if len(parts) > 1 else []

        cmd = self._commands.get(name)
        if cmd is None:
            return False
        result = cmd.handler(session, args)
        if inspect.isawaitable(result):
            await result
        return True

    def help_text(self) -> str:
        """Generate formatted help text for all registered commands."""
        lines = []
        for cmd in sorted(self._commands.values(), key=lambda c: c.name):
            lines.append(f"  /{cmd.name:16s} {cmd.description}")
        return "\n".join(lines)

    @property
    def command_names(self) -> list[str]:
        """Sorted list of command names (without /)."""
        return sorted(self._commands.keys())
