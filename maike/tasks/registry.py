"""Unified task registry — wraps AsyncDelegateManager + BackgroundProcessManager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from maike.atoms.context import TaskState


@dataclass
class TaskInfo:
    """Unified view of a background task (delegate or process)."""

    handle: str
    kind: Literal["delegate", "process"]
    status: TaskState
    description: str
    started_at: datetime
    cost_usd: float = 0.0
    tokens_used: int = 0
    last_activity: str = ""


class TaskRegistry:
    """Unified interface for listing all background work.

    Wraps ``AsyncDelegateManager`` and ``BackgroundProcessManager``
    behind a single ``list_all()`` method.
    """

    def __init__(self, delegate_mgr: Any, process_mgr: Any) -> None:
        self._delegates = delegate_mgr
        self._processes = process_mgr

    def list_all(self) -> list[TaskInfo]:
        """Return all background tasks sorted by start time."""
        tasks: list[TaskInfo] = []

        # Async delegates.
        for d in self._delegates._delegates.values():
            progress = getattr(d.ctx, "progress", None)
            tasks.append(TaskInfo(
                handle=d.handle,
                kind="delegate",
                status=d.status,
                description=d.task_description[:80],
                started_at=d.started_at,
                cost_usd=d.ctx.cost_used_usd,
                tokens_used=d.ctx.tokens_used,
                last_activity=progress.last_activity if progress else "",
            ))

        # Background processes.
        for bp in self._processes._processes.values():
            self._processes._sync_status(bp)
            tasks.append(TaskInfo(
                handle=bp.handle,
                kind="process",
                status=bp.status,
                description=bp.cmd[:80],
                started_at=bp.started_at,
            ))

        return sorted(tasks, key=lambda t: t.started_at)

    def count_running(self) -> int:
        """Count of currently running tasks."""
        return sum(1 for t in self.list_all() if t.status == TaskState.RUNNING)

    def format_table(self) -> str:
        """Format task list as a human-readable table."""
        tasks = self.list_all()
        if not tasks:
            return "No background tasks."

        lines = ["Handle          Kind       Status      Description"]
        lines.append("-" * 70)
        for t in tasks:
            status_str = t.status.value
            lines.append(
                f"{t.handle:15s} {t.kind:10s} {status_str:11s} {t.description[:40]}"
            )
        return "\n".join(lines)
