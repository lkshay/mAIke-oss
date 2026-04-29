"""Tracked background process management for long-running commands."""

from __future__ import annotations

import asyncio
import collections
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from maike.atoms.context import TaskState
from maike.utils import utcnow


_MAX_OUTPUT_LINES = 200  # bounded buffer to prevent memory leaks
_STOP_GRACE_SECONDS = 5  # time between SIGTERM and SIGKILL


@dataclass
class BackgroundProcess:
    handle: str  # "bg-001", "bg-002"
    cmd: str
    process: asyncio.subprocess.Process
    started_at: datetime
    output_buffer: collections.deque  # bounded deque of output lines
    status: TaskState = TaskState.RUNNING
    exit_code: int | None = None
    output_file: Path | None = field(default=None, repr=False)
    _collector_task: asyncio.Task | None = field(default=None, repr=False)
    _log_fh: object | None = field(default=None, repr=False)  # open file handle


class BackgroundProcessManager:
    """Manage tracked background processes with check/stop capabilities."""

    def __init__(
        self,
        output_dir: Path | None = None,
        notification_queue: asyncio.Queue | None = None,
    ) -> None:
        self._processes: dict[str, BackgroundProcess] = {}
        self._counter: int = 0
        self._output_dir = output_dir
        self._notification_queue = notification_queue

    async def start(
        self,
        cmd: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> BackgroundProcess:
        """Start a background process and begin collecting output."""
        self._counter += 1
        handle = f"bg-{self._counter:03d}"

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=env,
            # Start in a new process group so we can signal the whole tree.
            preexec_fn=os.setsid,
        )

        buf: collections.deque[str] = collections.deque(maxlen=_MAX_OUTPUT_LINES)

        # Create log file for output tee-ing.
        output_file: Path | None = None
        log_fh = None
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self._output_dir / f"{handle}.log"
            log_fh = open(output_file, "w")  # noqa: SIM115

        bp = BackgroundProcess(
            handle=handle,
            cmd=cmd,
            process=process,
            started_at=utcnow(),
            output_buffer=buf,
            output_file=output_file,
            _log_fh=log_fh,
        )
        bp._collector_task = asyncio.create_task(self._collect_output(bp))
        self._processes[handle] = bp
        return bp

    async def check(self, handle: str) -> dict:
        """Check status and recent output of a background process.

        Returns dict with status, recent_output, exit_code, uptime_seconds,
        and handle.
        """
        bp = self._processes.get(handle)
        if bp is None:
            return {"error": f"No process with handle {handle!r}", "handle": handle}

        self._sync_status(bp)

        recent = list(bp.output_buffer)[-50:]
        uptime = (utcnow() - bp.started_at).total_seconds()
        return {
            "handle": bp.handle,
            "status": bp.status.value,
            "recent_output": "\n".join(recent),
            "exit_code": bp.exit_code,
            "uptime_seconds": uptime,
        }

    async def stop(self, handle: str) -> dict:
        """Stop a background process. SIGTERM -> grace period -> SIGKILL.

        Returns dict with status, final_output, and exit_code.
        """
        bp = self._processes.get(handle)
        if bp is None:
            return {"error": f"No process with handle {handle!r}", "handle": handle}

        self._sync_status(bp)

        if bp.status == TaskState.RUNNING:
            # Try graceful termination first.
            try:
                os.killpg(os.getpgid(bp.process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

            try:
                await asyncio.wait_for(bp.process.wait(), timeout=_STOP_GRACE_SECONDS)
            except asyncio.TimeoutError:
                # Force kill.
                try:
                    os.killpg(os.getpgid(bp.process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await asyncio.wait_for(bp.process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass

            self._sync_status(bp)

        # Cancel the collector task.
        if bp._collector_task is not None and not bp._collector_task.done():
            bp._collector_task.cancel()
            try:
                await bp._collector_task
            except asyncio.CancelledError:
                pass

        final = list(bp.output_buffer)[-100:]
        return {
            "handle": bp.handle,
            "status": bp.status.value,
            "final_output": "\n".join(final),
            "exit_code": bp.exit_code,
        }

    async def cleanup(self) -> None:
        """Kill all tracked processes. Called on session end."""
        await asyncio.gather(*(self.stop(h) for h in list(self._processes)))

    def list_running(self) -> list[str]:
        """Return handles of all running processes."""
        result = []
        for bp in self._processes.values():
            self._sync_status(bp)
            if bp.status == TaskState.RUNNING:
                result.append(bp.handle)
        return result

    def _sync_status(self, bp: BackgroundProcess) -> None:
        """Synchronize the status field with the actual process state."""
        if bp.status == TaskState.RUNNING and bp.process.returncode is not None:
            bp.exit_code = bp.process.returncode
            bp.status = TaskState.COMPLETED if bp.exit_code == 0 else TaskState.FAILED

    async def _collect_output(self, bp: BackgroundProcess) -> None:
        """Read stdout and stderr into the bounded buffer, tee to log file."""

        async def _read_stream(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode(errors="replace").rstrip("\n")
                    bp.output_buffer.append(decoded)
                    # Tee to log file.
                    if bp._log_fh is not None:
                        try:
                            bp._log_fh.write(decoded + "\n")
                            bp._log_fh.flush()
                        except (OSError, ValueError):
                            pass
            except asyncio.CancelledError:
                return

        try:
            await asyncio.gather(
                _read_stream(bp.process.stdout),
                _read_stream(bp.process.stderr),
            )
            # Wait for process exit to capture the return code.
            await bp.process.wait()
        except asyncio.CancelledError:
            return
        finally:
            self._sync_status(bp)
            # Close log file.
            if bp._log_fh is not None:
                try:
                    bp._log_fh.close()
                except (OSError, ValueError):
                    pass
            # Push completion notification.
            if self._notification_queue is not None and bp.status in (TaskState.COMPLETED, TaskState.FAILED):
                self._notification_queue.put_nowait(
                    f"## Background Task Completed\n\n"
                    f"**{bp.handle}** exited (code {bp.exit_code}).\n"
                    f"Output: {bp.output_file}\n"
                    f"Use Read to review."
                )
