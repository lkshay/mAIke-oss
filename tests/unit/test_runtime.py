import asyncio
from pathlib import Path

from maike.atoms.context import AgentContext
from maike.atoms.tool import RiskLevel
from maike.runtime.local import LocalRuntime, RuntimeConfig
from maike.safety.hooks import SafetyLayer
from maike.safety.rules import Decision
from maike.tools.context import CURRENT_AGENT_CONTEXT
from maike.tools.filesystem import register_filesystem_tools
from maike.tools.registry import ToolRegistry


def test_local_runtime_rejects_path_escape(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        write_result = await runtime.write_file("../outside.txt", "x")
        read_result = await runtime.read_file("../outside.txt")
        list_result = await runtime.list_dir("../")
        delete_result = await runtime.delete_file("../outside.txt")
        return [write_result, read_result, list_result, delete_result]

    results = asyncio.run(scenario())
    for result in results:
        assert result.success is False
        assert result.error == "Path escapes workspace: ../outside.txt" or result.error == "Path escapes workspace: ../"


def test_local_runtime_restore_removes_untracked_files(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        init_result = await runtime.init_git_repo()
        assert init_result.success is True
        await runtime.write_file("tracked.txt", "one\n")
        checkpoint = await runtime.checkpoint("pre-coding", "coding")
        await runtime.write_file("tracked.txt", "two\n")
        await runtime.write_file("scratch/new.txt", "temp\n")
        await runtime.restore(checkpoint)
        return runtime

    runtime = asyncio.run(scenario())
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "one\n"
    assert not (tmp_path / "scratch").exists()
    git_status = asyncio.run(runtime.execute_bash("git status --short"))
    assert git_status.success is True
    assert git_status.raw_output.strip() == ""


def test_local_runtime_restore_preserves_maike_state(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        init_result = await runtime.init_git_repo()
        assert init_result.success is True
        session_db = tmp_path / ".maike" / "session.db"
        session_db.parent.mkdir(parents=True, exist_ok=True)
        session_db.write_text("session-state-v1\n", encoding="utf-8")
        checkpoint = await runtime.checkpoint("pre-coding", "coding")
        session_db.write_text("session-state-v2\n", encoding="utf-8")
        await runtime.write_file("tracked.txt", "changed\n")
        await runtime.restore(checkpoint)
        return session_db.read_text(encoding="utf-8")

    persisted = asyncio.run(scenario())

    assert persisted == "session-state-v2\n"


def test_local_runtime_read_file_truncates_large_utf8_content(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        await runtime.write_file("large.txt", "a" * 32_100)
        return await runtime.read_file("large.txt")

    result = asyncio.run(scenario())

    assert result.success is True
    assert result.metadata["truncated"] is True
    assert result.metadata["binary"] is False
    assert "file truncated" in result.output
    assert "start_line/end_line" in result.output
    assert len(result.output) > 32_000


def test_local_runtime_read_file_marks_binary_payloads(tmp_path):
    binary_path = Path(tmp_path / "image.bin")
    binary_path.write_bytes(b"\xff\xd8\xff\x00raw")

    async def scenario():
        runtime = LocalRuntime(tmp_path)
        return await runtime.read_file("image.bin")

    result = asyncio.run(scenario())

    assert result.success is True
    assert result.metadata["binary"] is True
    assert result.metadata["truncated"] is False
    assert result.output == "[BINARY FILE] image.bin (7 bytes)"


def test_local_runtime_read_file_line_range(tmp_path):
    """read_file with start_line/end_line returns only the requested range."""
    content = "\n".join(f"line {i}" for i in range(1, 21))  # 20 lines

    async def scenario():
        runtime = LocalRuntime(tmp_path)
        await runtime.write_file("code.py", content)
        return await runtime.read_file("code.py", start_line=5, end_line=10)

    result = asyncio.run(scenario())

    assert result.success is True
    lines = result.output.split("\n")
    assert lines[0] == "line 5"
    assert lines[-1] == "line 10"
    assert len(lines) == 6
    assert result.metadata.get("start_line") == 5
    assert result.metadata.get("end_line") == 10
    assert result.metadata.get("line_range") is True
    assert result.metadata.get("total_lines") == 20


def test_local_runtime_read_file_line_range_start_only(tmp_path):
    """read_file with only start_line reads from that line to end."""
    content = "\n".join(f"line {i}" for i in range(1, 6))  # 5 lines

    async def scenario():
        runtime = LocalRuntime(tmp_path)
        await runtime.write_file("code.py", content)
        return await runtime.read_file("code.py", start_line=3)

    result = asyncio.run(scenario())

    assert result.success is True
    lines = result.output.split("\n")
    assert lines[0] == "line 3"
    assert lines[-1] == "line 5"


def test_same_file_rewrite_does_not_require_multi_file_checkpoint(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        registry = ToolRegistry()
        register_filesystem_tools(registry, runtime)
        safety = SafetyLayer(tmp_path)
        ctx = AgentContext(
            role="coder",
            task="x",
            stage_name="coding",
            tool_profile="coding",
            metadata={
                "session_id": "session-1",
                "mutated_paths": [],
            },
        )
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        try:
            first = safety.assess(
                "Write",
                {"path": "app.py", "content": "one"},
                RiskLevel.WRITE,
                ctx=ctx,
            )
            assert first.decision == Decision.ALLOW
            tool = registry.get("Write")
            assert tool is not None
            result = await tool.fn(path="app.py", content="one")
            assert result.success is True
            second = safety.assess(
                "Write",
                {"path": "app.py", "content": "two"},
                RiskLevel.WRITE,
                ctx=ctx,
            )
            return second
        finally:
            CURRENT_AGENT_CONTEXT.reset(token)

    assessment = asyncio.run(scenario())
    assert assessment.decision == Decision.ALLOW
    assert assessment.requires_checkpoint is False


def test_execute_bash_surfaces_stderr_before_stdout(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        return await runtime.execute_bash(
            "python3 -c \"import sys; print('out'); print('err', file=sys.stderr)\"",
            timeout_class="normal",
        )

    result = asyncio.run(scenario())

    assert result.success is True
    assert result.metadata["stderr_present"] is True
    assert result.metadata["timeout_seconds"] == 30
    assert result.metadata["idle_timeout_seconds"] is None
    assert result.metadata["timeout_class"] == "normal"
    assert result.output.startswith("[stderr]\nerr")
    assert "[stdout]\nout" in result.output


def test_execute_bash_timeout_failure_reports_effective_timeout(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        return await runtime.execute_bash(
            "python3 -c \"import time; time.sleep(0.2)\"",
            timeout=0.01,
        )

    result = asyncio.run(scenario())

    assert result.success is False
    assert result.error == "Command timed out after 0.01s"
    assert result.metadata["timeout_seconds"] == 0.01
    assert result.metadata["idle_timeout_seconds"] is None
    assert result.metadata["timeout_kind"] == "max_runtime"


def test_execute_bash_idle_timeout_returns_partial_output(tmp_path):
    async def scenario():
        runtime = LocalRuntime(tmp_path)
        return await runtime.execute_bash(
            "python3 -c \"import sys,time; print('start'); sys.stdout.flush(); time.sleep(0.2)\"",
            timeout=1,
            idle_timeout=0.05,
            timeout_class="long",
        )

    result = asyncio.run(scenario())

    assert result.success is False
    assert result.error == "Command idle timed out after 0.05s"
    assert result.metadata["timeout_kind"] == "idle"
    assert result.metadata["timeout_class"] == "long"
    assert "start" in result.output


def test_execute_bash_uses_resolved_shell_environment(tmp_path):
    async def scenario():
        runtime = LocalRuntime(
            tmp_path,
            config=RuntimeConfig(
                shell_env={"MAIKE_RESOLVED_RUNTIME": "1"},
                package_manager="pip",
                interpreter_command="python3",
            ),
        )
        return await runtime.execute_bash(
            "python3 -c \"import os; print(os.environ['MAIKE_RESOLVED_RUNTIME'])\"",
        )

    result = asyncio.run(scenario())

    assert result.success is True
    assert result.raw_output.strip() == "1"
    assert result.metadata["package_manager"] == "pip"
    assert result.metadata["interpreter_command"] == "python3"


def test_collect_process_output_drains_streams_after_process_exit_without_false_timeout(tmp_path):
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = 0
            self.killed = False

        async def wait(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    async def scenario():
        runtime = LocalRuntime(tmp_path)
        process = FakeProcess()

        async def close_streams() -> None:
            await asyncio.sleep(0.02)
            process.stdout.feed_eof()
            process.stderr.feed_eof()

        closer = asyncio.create_task(close_streams())
        try:
            return await runtime._collect_process_output(
                process,
                timeout=0.2,
                idle_timeout=None,
            )
        finally:
            await closer

    stdout, stderr, timeout_kind, returncode = asyncio.run(scenario())

    assert stdout == ""
    assert stderr == ""
    assert timeout_kind is None
    assert returncode == 0
