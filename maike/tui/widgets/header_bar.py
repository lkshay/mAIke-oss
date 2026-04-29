"""Header bar widget — session info, workspace, thread, agent state."""

from __future__ import annotations

from rich.markup import escape as _rich_escape
from textual.widgets import Static

from maike.tui.theme import (
    MAIKE_ACCENT,
    STATE_COLORS,
    STATE_LABELS,
    STATE_SYMBOLS,
    STATE_WAITING,
)


class HeaderBar(Static):
    """Top-of-screen bar showing session metadata and agent state.

    Layout: left=brand, center=state, right=model+workspace
    """

    def __init__(
        self,
        workspace: str = "",
        session_id: str = "",
        thread: str = "",
        model: str = "",
        provider: str = "",
        **kwargs,
    ) -> None:
        self._workspace = workspace
        self._session_id = session_id[:8] if session_id else ""
        self._thread = thread or "new"
        self._model = model
        self._provider = provider
        self._agent_state = STATE_WAITING
        super().__init__(self._render_text(), **kwargs)

    def _render_text(self) -> str:
        # Left: brand
        left = f"[bold {MAIKE_ACCENT}]mAIke[/bold {MAIKE_ACCENT}]"

        # Center: agent state
        symbol = STATE_SYMBOLS.get(self._agent_state, "\u25c7")
        color = STATE_COLORS.get(self._agent_state, "#666666")
        label = STATE_LABELS.get(self._agent_state, self._agent_state.capitalize())
        center = f"[{color}]{symbol} {label}[/{color}]"

        # Right: model + workspace
        right_parts = []
        if self._provider and self._model:
            right_parts.append(f"[dim]{_rich_escape(self._provider)}:{_rich_escape(self._model)}[/dim]")
        elif self._model:
            right_parts.append(f"[dim]{_rich_escape(self._model)}[/dim]")
        if self._workspace:
            short = self._workspace.rstrip("/").rsplit("/", 1)[-1]
            right_parts.append(f"[dim]ws:{_rich_escape(short)}[/dim]")
        if self._session_id:
            right_parts.append(f"[dim]{_rich_escape(self._session_id)}[/dim]")
        right = "  ".join(right_parts)

        return f"{left}  {center}  {right}"

    def update_session(self, session_id: str = "", thread: str = "") -> None:
        if session_id:
            self._session_id = session_id[:8]
        if thread:
            self._thread = thread
        self.update(self._render_text())

    def set_agent_state(self, state: str) -> None:
        """Update the agent state indicator."""
        if state != self._agent_state:
            self._agent_state = state
            self.update(self._render_text())
