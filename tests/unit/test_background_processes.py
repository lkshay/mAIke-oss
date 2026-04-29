"""Tests for maike.runtime.background — BackgroundProcessManager."""

from __future__ import annotations

import asyncio
import sys

from maike.runtime.background import BackgroundProcessManager, _MAX_OUTPUT_LINES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine to completion."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_start_process(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        try:
            bp = await mgr.start("echo hello && sleep 60", cwd=tmp_path)
            assert bp.handle == "bg-001"
            assert bp.status == "running"
            assert bp.cmd == "echo hello && sleep 60"
        finally:
            await mgr.cleanup()

    _run(_test())


def test_check_running_process(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        try:
            script = (
                "import time, sys\n"
                "for i in range(5):\n"
                "    print(f'line {i}', flush=True)\n"
                "    time.sleep(0.1)\n"
            )
            bp = await mgr.start(
                f"{sys.executable} -c \"{script}\"",
                cwd=tmp_path,
            )
            # Wait for the process to finish producing output.
            await asyncio.sleep(1.5)
            info = await mgr.check(bp.handle)
            assert info["handle"] == bp.handle
            # Process may still be running or may have exited by now.
            assert info["status"] in ("running", "completed", "failed")
            # Verify output was captured.
            output = info["recent_output"]
            assert "line 0" in output
            assert "line 4" in output
        finally:
            await mgr.cleanup()

    _run(_test())


def test_check_nonexistent_handle():
    async def _test():
        mgr = BackgroundProcessManager()
        info = await mgr.check("bg-999")
        assert "error" in info
        assert "bg-999" in info["error"]

    _run(_test())


def test_stop_process(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        try:
            bp = await mgr.start("sleep 60", cwd=tmp_path)
            result = await asyncio.wait_for(mgr.stop(bp.handle), timeout=15)
            assert result["status"] in ("completed", "failed")
            assert result["exit_code"] is not None
        finally:
            await mgr.cleanup()

    _run(_test())


def test_stop_already_exited(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        try:
            bp = await mgr.start("echo done", cwd=tmp_path)
            # Wait for the short-lived process to exit.
            await asyncio.sleep(0.5)
            result = await asyncio.wait_for(mgr.stop(bp.handle), timeout=10)
            assert result["status"] in ("completed", "failed")
            assert result["exit_code"] == 0
            assert "done" in result["final_output"]
        finally:
            await mgr.cleanup()

    _run(_test())


def test_cleanup_kills_all(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        handles = []
        for _ in range(3):
            bp = await mgr.start("sleep 60", cwd=tmp_path)
            handles.append(bp.handle)

        await asyncio.wait_for(mgr.cleanup(), timeout=30)

        for handle in handles:
            info = await mgr.check(handle)
            assert info["status"] in ("completed", "failed")

    _run(_test())


def test_list_running(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        try:
            bp_long = await mgr.start("sleep 60", cwd=tmp_path)
            bp_short = await mgr.start("echo done", cwd=tmp_path)
            # Wait for the short one to exit.
            await asyncio.sleep(0.5)

            running = mgr.list_running()
            assert bp_long.handle in running
            assert bp_short.handle not in running
        finally:
            await mgr.cleanup()

    _run(_test())


def test_output_buffer_bounded(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        try:
            # Generate 500 lines of output.
            script = (
                "import sys\n"
                "for i in range(500):\n"
                "    print(f'line {i}', flush=True)\n"
            )
            bp = await mgr.start(
                f'{sys.executable} -c "{script}"',
                cwd=tmp_path,
            )
            # Wait for the process to finish writing.
            await asyncio.sleep(2)

            assert len(bp.output_buffer) <= _MAX_OUTPUT_LINES
        finally:
            await mgr.cleanup()

    _run(_test())


def test_handle_counter_increments(tmp_path):
    async def _test():
        mgr = BackgroundProcessManager()
        try:
            bp1 = await mgr.start("echo 1", cwd=tmp_path)
            bp2 = await mgr.start("echo 2", cwd=tmp_path)
            bp3 = await mgr.start("echo 3", cwd=tmp_path)
            assert bp1.handle == "bg-001"
            assert bp2.handle == "bg-002"
            assert bp3.handle == "bg-003"
        finally:
            await mgr.cleanup()

    _run(_test())
