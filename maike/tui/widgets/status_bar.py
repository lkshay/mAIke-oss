"""Status bar widget — model, cost, tokens, iteration, elapsed, budget bar."""

from __future__ import annotations

import time

from textual.timer import Timer
from textual.widgets import Static

from maike.tui.events import StatusUpdate
from maike.tui.theme import MAIKE_ACCENT, MAIKE_ERROR, MAIKE_SUCCESS, MAIKE_WARNING


class StatusBar(Static):
    """Bottom-of-screen bar showing live session metrics with color-coded cost."""

    def __init__(
        self,
        model: str = "",
        provider: str = "",
        budget: float = 0.0,
        **kwargs,
    ) -> None:
        self._model = model
        self._provider = provider
        self._cost = 0.0
        self._tokens = 0
        self._iteration = 0
        self._max_iterations = 0
        self._budget = budget
        self._start_time: float | None = None
        self._elapsed_seconds = 0.0
        self._timer: Timer | None = None
        super().__init__(self._render_text(), **kwargs)

    def on_mount(self) -> None:
        self._timer = self.set_interval(1.0, self._tick_elapsed)

    def set_start_time(self, t: float) -> None:
        """Set the session start time (monotonic). Called by bridge on first LLM_START."""
        if self._start_time is None:
            self._start_time = t

    def _tick_elapsed(self) -> None:
        if self._start_time is not None:
            self._elapsed_seconds = time.monotonic() - self._start_time
            self.update(self._render_text())

    def _cost_color(self) -> str:
        """Color-coded cost: green < 50%, yellow 50-80%, red 80%+."""
        if self._budget <= 0:
            return MAIKE_SUCCESS
        ratio = self._cost / self._budget
        if ratio >= 0.8:
            return MAIKE_ERROR
        if ratio >= 0.5:
            return MAIKE_WARNING
        return MAIKE_SUCCESS

    def _render_text(self) -> str:
        parts = []
        if self._model:
            parts.append(f"[bold]{self._model}[/bold]")

        cost_color = self._cost_color()
        parts.append(f"[{cost_color}]${self._cost:.4f}[/{cost_color}]")

        parts.append(f"[dim]{self._tokens:,} tok[/dim]")

        if self._iteration:
            parts.append(f"[{MAIKE_ACCENT}]iter {self._iteration}[/{MAIKE_ACCENT}]")

        if self._elapsed_seconds > 0:
            m, s = divmod(int(self._elapsed_seconds), 60)
            parts.append(f"[dim]{m:02d}:{s:02d}[/dim]")

        if self._budget > 0:
            ratio = min(self._cost / self._budget, 1.0) if self._budget else 0
            filled = int(ratio * 10)
            bar_color = self._cost_color()
            filled_chars = "\u2588" * filled
            empty_chars = "\u2591" * (10 - filled)
            bar = f"[{bar_color}]{filled_chars}[/{bar_color}][dim]{empty_chars}[/dim]"
            parts.append(f"{bar} ${self._cost:.2f}/${self._budget:.2f}")

        return " \u00b7 ".join(parts)  # middle dot separator

    def handle_status_update(self, event: StatusUpdate) -> None:
        if event.model:
            self._model = event.model
        if event.provider:
            self._provider = event.provider
        self._cost = event.cost
        self._tokens = event.tokens
        self._iteration = event.iteration
        self._max_iterations = event.max_iterations
        if event.budget:
            self._budget = event.budget
        self.update(self._render_text())
