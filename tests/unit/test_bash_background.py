"""Tests for unified Bash tool with background process support."""

import asyncio
import collections
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from maike.atoms.tool import RiskLevel, ToolResult
from maike.runtime.background import BackgroundProcess
from maike.tools.bash import register_bash_tools
from maike.tools.registry import ToolRegistry


def _make_bg_process(handle: str = "bg-001", cmd: str = "sleep 100") -> BackgroundProcess:
    """Create a BackgroundProcess with a mock subprocess."""
    proc = MagicMock()
    proc.returncode = None
    return BackgroundProcess(
        handle=handle,
        cmd=cmd,
        process=proc,
        started_at=datetime.now(timezone.utc),
        output_buffer=collections.deque(maxlen=200),
    )


class FakeRuntime:
    """Minimal runtime mock with background process support."""

    def __init__(self) -> None:
        self.start_background = AsyncMock()
        self.check_background = AsyncMock()
        self.stop_background = AsyncMock()
        self.execute_bash = AsyncMock()


def _build_registry() -> tuple[ToolRegistry, FakeRuntime]:
    registry = ToolRegistry()
    runtime = FakeRuntime()
    register_bash_tools(registry, runtime)
    return registry, runtime


# ── Registration tests ────────────────────────────────────────────


def test_bash_is_registered():
    registry, _ = _build_registry()
    tool = registry.get("Bash")
    assert tool is not None
    assert "Bash" in registry.list_tool_names()


def test_only_bash_registered():
    """Only one Bash tool is registered — no BashCheck or BashStop."""
    registry, _ = _build_registry()
    names = registry.list_tool_names()
    assert "BashCheck" not in names
    assert "BashStop" not in names


def test_bash_risk_level_is_execute():
    registry, _ = _build_registry()
    tool = registry.get("Bash")
    assert tool is not None
    assert tool.risk_level == RiskLevel.EXECUTE


# ── Synchronous Bash tests ───────────────────────────────────────


def test_bash_sync_default():
    """Bash(cmd="echo hi") blocks and returns output."""
    registry, runtime = _build_registry()
    runtime.execute_bash.return_value = ToolResult(
        tool_name="Bash",
        success=True,
        output="hi",
        metadata={},
    )

    tool = registry.get("Bash")
    assert tool is not None
    result: ToolResult = asyncio.run(tool.fn(cmd="echo hi"))

    assert result.success is True
    assert result.output == "hi"
    runtime.execute_bash.assert_called_once()


# ── Background Bash tests ────────────────────────────────────────


def test_bash_background_mode():
    """Bash(cmd="...", background=true) returns a handle."""
    registry, runtime = _build_registry()

    bg = _make_bg_process("bg-001", "python server.py")
    runtime.start_background.return_value = bg
    runtime.check_background.return_value = {
        "status": "running",
        "recent_output": "Server started on port 8000",
        "uptime_seconds": 3.0,
    }

    tool = registry.get("Bash")
    assert tool is not None
    result: ToolResult = asyncio.run(tool.fn(cmd="python server.py", background=True))

    assert result.success is True
    assert "bg-001" in result.output
    assert result.metadata["handle"] == "bg-001"
    assert result.metadata["still_running"] is True
    assert result.metadata["background"] is True


def test_background_bash_process_exits_immediately():
    registry, runtime = _build_registry()

    bg = _make_bg_process("bg-002", "false")
    runtime.start_background.return_value = bg
    runtime.check_background.return_value = {
        "status": "exited",
        "exit_code": 1,
        "recent_output": "Error: port in use",
        "uptime_seconds": 0.1,
    }

    tool = registry.get("Bash")
    assert tool is not None
    result: ToolResult = asyncio.run(tool.fn(cmd="false", background=True))

    assert result.success is False
    assert "exited immediately" in result.output.lower()
    assert result.metadata["still_running"] is False


def test_background_bash_process_exits_success():
    registry, runtime = _build_registry()

    bg = _make_bg_process("bg-003", "echo done")
    runtime.start_background.return_value = bg
    runtime.check_background.return_value = {
        "status": "exited",
        "exit_code": 0,
        "recent_output": "done",
        "uptime_seconds": 0.05,
    }

    tool = registry.get("Bash")
    assert tool is not None
    result: ToolResult = asyncio.run(tool.fn(cmd="echo done", background=True))

    assert result.success is True


# ── Stop via stop param tests ─────────────────────────────────────


def test_bash_stop_param():
    """Bash(stop="bg-001") stops the process."""
    registry, runtime = _build_registry()
    runtime.stop_background.return_value = {
        "exit_code": -15,
        "final_output": "Server shutting down\nGoodbye",
    }

    tool = registry.get("Bash")
    assert tool is not None
    result: ToolResult = asyncio.run(tool.fn(stop="bg-001"))

    assert result.success is True
    assert "bg-001" in result.output
    assert "stopped" in result.output.lower()
    assert "exit code: -15" in result.output.lower()
    assert "Server shutting down" in result.output
    assert result.metadata["handle"] == "bg-001"
    assert result.metadata["exit_code"] == -15


def test_bash_stop_unknown_handle():
    registry, runtime = _build_registry()
    runtime.stop_background.side_effect = KeyError("Unknown handle")

    tool = registry.get("Bash")
    assert tool is not None
    result: ToolResult = asyncio.run(tool.fn(stop="bg-999"))

    assert result.success is False
    assert "unknown handle" in result.output.lower()


def test_bash_no_handle_param_in_schema():
    """The unified Bash tool should not have a 'handle' parameter."""
    registry, _ = _build_registry()
    tool = registry.get("Bash")
    assert tool is not None
    schema = tool.schema.input_schema
    assert "handle" not in schema["properties"]
    assert "signal" not in schema["properties"]
    assert "stop" in schema["properties"]
