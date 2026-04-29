import io
from unittest.mock import patch

from maike.observability.live import RichLiveSink
from maike.observability.tracer import TraceEvent


def test_rich_live_sink_prints_tool_failure_with_error():
    sink = RichLiveSink(enabled=True)

    with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
        sink.emit(
            TraceEvent(
                kind="tool_result",
                agent_id="1234567890abcdef",
                tool_name="Bash",
                success=False,
                payload={
                    "error": "Command timed out after 90s",
                    "input": {"cmd": "pytest tests/"},
                },
            )
        )
        output = mock_stderr.getvalue()

    assert "✗" in output
    assert "Bash" in output
    assert "pytest tests/" in output
    assert "timed out" in output


def test_rich_live_sink_prints_tool_success():
    sink = RichLiveSink(enabled=True)

    with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
        sink.emit(
            TraceEvent(
                kind="tool_result",
                agent_id="abc",
                tool_name="Write",
                success=True,
                payload={"input": {"path": "/workspace/hello.py"}},
            )
        )
        output = mock_stderr.getvalue()

    assert "✓" in output
    assert "Write" in output
    assert "hello.py" in output


def test_rich_live_sink_prints_llm_call():
    sink = RichLiveSink(enabled=True)

    with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
        sink.emit(
            TraceEvent(
                kind="llm_call",
                agent_id="abc",
                model="gemini-2.5-flash",
                total_tokens=5000,
            )
        )
        output = mock_stderr.getvalue()

    assert "gemini-2.5-flash" in output
    assert "5000" in output
