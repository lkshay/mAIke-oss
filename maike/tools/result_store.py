"""Persist large tool results to disk, keeping only a preview in context."""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class PersistedResult:
    """A tool result that was saved to disk."""

    path: Path
    preview: str
    total_chars: int


class ResultStore:
    """Save large tool results to disk and return a preview for context."""

    _MAX_SESSION_BYTES: int = 50 * 1024 * 1024  # 50 MB per session

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._dir = Path.home() / ".maike" / "tool-results" / session_id
        self._bytes_written: int = 0

    def persist_if_large(
        self,
        tool_name: str,
        output: str,
        *,
        threshold: int = 10_000,
        preview_chars: int = 2_000,
    ) -> PersistedResult | None:
        """Save output to disk if it exceeds threshold. Returns None if small enough."""
        if len(output) <= threshold:
            return None
        if self._bytes_written >= self._MAX_SESSION_BYTES:
            return None  # Session cap reached

        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            filename = f"{uuid4().hex[:12]}_{tool_name}.txt"
            path = self._dir / filename
            encoded = output.encode("utf-8", errors="replace")
            path.write_bytes(encoded)
            self._bytes_written += len(encoded)
        except OSError:
            return None  # Don't break the agent loop

        # Preview = last N chars (most recent output is most relevant)
        preview = output[-preview_chars:] if len(output) > preview_chars else output

        return PersistedResult(
            path=path,
            preview=preview,
            total_chars=len(output),
        )

    @classmethod
    def cleanup_old_results(cls, max_age_hours: int = 24) -> int:
        """Delete session directories older than max_age_hours. Returns count deleted."""
        base = Path.home() / ".maike" / "tool-results"
        if not base.exists():
            return 0
        cutoff = time.time() - (max_age_hours * 3600)
        deleted = 0
        for session_dir in base.iterdir():
            if session_dir.is_dir():
                try:
                    mtime = session_dir.stat().st_mtime
                    if mtime < cutoff:
                        shutil.rmtree(session_dir)
                        deleted += 1
                except OSError:
                    pass
        return deleted
