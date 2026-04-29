"""Tests for maike.tui.bridge — TUITraceSink event processing."""

from __future__ import annotations

from maike.observability.tracer import TraceEvent, TraceEventKind
from maike.tui.theme import TOOL_OUTPUT_MAX_CHARS, TOOL_OUTPUT_MAX_LINES


class FakeMessageList:
    """Fake MessageList for testing bridge processors."""

    def __init__(self):
        self.infos = []

    def add_info(self, text):
        self.infos.append(text)


class FakePromptInput:
    """Fake PromptInput for testing."""

    def __init__(self):
        self.disabled = False
        self._mode = "task"

    def set_mode(self, mode):
        self._mode = mode

    def focus(self):
        pass


class FakeApp:
    """Minimal fake Textual app for bridge tests."""

    def __init__(self):
        self._intervals = []
        self._from_thread_calls = []
        self._screens_pushed = []
        self._widgets = {
            "MessageList": FakeMessageList(),
            "PromptInput": FakePromptInput(),
        }
        self._approval_callback = None
        self._waiting_for_approval = False

    def set_interval(self, interval, callback):
        self._intervals.append((interval, callback))
        return None

    def query_one(self, selector):
        widget = self._widgets.get(selector)
        if widget is not None:
            return widget
        raise LookupError(f"no widget: {selector}")

    def call_from_thread(self, fn):
        self._from_thread_calls.append(fn)
        # Execute immediately for testing
        fn()

    def push_screen(self, screen, callback=None):
        self._screens_pushed.append((screen, callback))
        # Auto-approve for testing
        if callback:
            callback(True)


def _make_event(kind, tool_name=None, payload=None, **kwargs):
    return TraceEvent(
        kind=kind,
        tool_name=tool_name,
        payload=payload or {},
        **kwargs,
    )


def test_tool_output_extraction():
    """_emit_tool_result extracts output and execution_ms from payload."""
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    event = _make_event(
        TraceEventKind.TOOL_RESULT,
        tool_name="execute_bash",
        success=True,
        execution_ms=1234,
        payload={"output": "test passed", "raw_output": "raw", "error": None},
    )
    sink.emit(event)
    kind, data = sink._queue.get_nowait()

    assert kind == "tool_result"
    assert data["output"] == "test passed"
    assert data["execution_ms"] == 1234
    assert data["success"] is True
    assert data["tool_name"] == "execute_bash"
    assert data["error"] is None


def test_tool_output_falls_back_to_raw_output():
    """When output is empty, raw_output is used."""
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    event = _make_event(
        TraceEventKind.TOOL_RESULT,
        tool_name="Read",
        success=True,
        payload={"output": "", "raw_output": "file content here"},
    )
    sink.emit(event)
    _, data = sink._queue.get_nowait()
    assert data["output"] == "file content here"


def test_tool_output_passthrough_preserves_full_text():
    """Bridge forwards the full tool output; ToolCallWidget handles inline
    truncation for display while retaining the complete text for the
    Ctrl+O full-output modal.  Regression: earlier versions truncated
    in the bridge, which destroyed the full output before the widget
    ever saw it — the expand action then had nothing to reveal.
    """
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    long_output = "x" * (TOOL_OUTPUT_MAX_CHARS + 500)
    event = _make_event(
        TraceEventKind.TOOL_RESULT,
        tool_name="Read",
        success=True,
        payload={"output": long_output},
    )
    sink.emit(event)
    _, data = sink._queue.get_nowait()
    assert data["output"] == long_output
    assert len(data["output"]) == len(long_output)


def test_tool_output_passthrough_preserves_all_lines():
    """Bridge forwards every line of the output — truncation lives in the
    widget, not here.  The widget caps the inline preview but keeps the
    full text for the expand-to-full modal.
    """
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    many_lines = "\n".join(f"line {i}" for i in range(TOOL_OUTPUT_MAX_LINES + 50))
    event = _make_event(
        TraceEventKind.TOOL_RESULT,
        tool_name="Read",
        success=True,
        payload={"output": many_lines},
    )
    sink.emit(event)
    _, data = sink._queue.get_nowait()
    assert data["output"] == many_lines
    assert data["output"].count("\n") == TOOL_OUTPUT_MAX_LINES + 50 - 1


def test_delegate_events_filtered():
    """Delegate sub-agent internal tool calls are hidden from the UI."""
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    event = _make_event(
        TraceEventKind.TOOL_START,
        tool_name="Read",
        agent_role="delegate",
        payload={"activity": "reading file"},
    )
    sink.emit(event)
    assert sink._queue.empty()


def test_delegate_tool_suppressed():
    """TOOL_START/TOOL_RESULT for the Delegate tool itself are suppressed.

    The delegate lifecycle is shown via ASYNC_DELEGATE_SPAWN/COMPLETE events
    instead.  This also hides max_async_delegates_exceeded errors.
    """
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    # Delegate tool start — should be suppressed
    sink.emit(_make_event(
        TraceEventKind.TOOL_START,
        tool_name="Delegate",
        payload={"input": {"task": "explore codebase"}},
    ))
    assert sink._queue.empty()

    # Delegate tool result with max_async_delegates_exceeded — suppressed
    sink.emit(_make_event(
        TraceEventKind.TOOL_RESULT,
        tool_name="Delegate",
        success=False,
        payload={"error": "max_async_delegates_exceeded"},
    ))
    assert sink._queue.empty()


def test_per_tool_name_tracking():
    """Tool results are matched to starts by tool_name, not FIFO."""
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    # Simulate two tool starts: Read then Grep
    sink.emit(_make_event(
        TraceEventKind.TOOL_START, tool_name="Read",
        payload={"input": {"path": "a.py"}},
    ))
    sink.emit(_make_event(
        TraceEventKind.TOOL_START, tool_name="Grep",
        payload={"input": {"pattern": "foo"}},
    ))

    # Drain the starts (they enqueue tool_start items)
    items = []
    while not sink._queue.empty():
        items.append(sink._queue.get_nowait())
    assert len(items) == 2
    assert items[0][1]["tool_name"] == "Read"
    assert items[1][1]["tool_name"] == "Grep"


def test_iteration_tracking():
    """Iteration counter increments on LLM_START after TOOL_RESULT."""
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    # First LLM start — no prior tool result, no separator
    sink.emit(_make_event(TraceEventKind.LLM_START, model="test-model"))
    assert sink._iteration == 0

    # Tool result
    sink.emit(_make_event(
        TraceEventKind.TOOL_RESULT, tool_name="Read", success=True,
        payload={"output": "content"},
    ))
    # Next LLM start — should emit turn separator
    sink.emit(_make_event(TraceEventKind.LLM_START, model="test-model"))
    assert sink._iteration == 1

    # Drain and check for turn separator
    items = []
    while not sink._queue.empty():
        items.append(sink._queue.get_nowait())
    kinds = [item[0] for item in items]
    assert "turn_separator" in kinds


def test_stream_sink_reset_clears_accumulator():
    """TUIStreamSink.reset() wipes the buffer so a new turn starts fresh.

    Regression: without this, incremental deltas from turn N+1 appended
    to the buffered text of turn N, and the UI re-rendered turn N's
    content with each new chunk — the "previous-turn output bleeds into
    latest response" bug.
    """
    from types import SimpleNamespace

    from maike.tui.bridge import TUIStreamSink, TUITraceSink

    app = FakeApp()
    trace = TUITraceSink(app)
    sink = TUIStreamSink(app, trace)

    # Turn 1: stream two deltas
    sink(SimpleNamespace(text_delta="Hello ", is_final=False))
    sink(SimpleNamespace(text_delta="world\n", is_final=False))
    assert sink._buffer == "Hello world\n"

    # Between turns: reset (bridge calls this on LLM_START)
    sink.reset()
    assert sink._buffer == ""

    # Turn 2: new delta should NOT carry turn-1 content
    sink(SimpleNamespace(text_delta="Goodbye", is_final=False))
    assert sink._buffer == "Goodbye"


def test_stream_sink_final_chunk_emits_done_and_resets():
    """A chunk with is_final=True emits STREAM_DONE and resets the buffer.

    The final chunk carries the fully-accumulated text in its text_delta
    field (not an incremental delta), so the sink must skip appending it
    — otherwise the buffer would double the text.
    """
    from types import SimpleNamespace

    from maike.tui.bridge import TUIStreamSink, TUITraceSink

    app = FakeApp()
    trace = TUITraceSink(app)
    sink = TUIStreamSink(app, trace)

    # Stream a couple of deltas then the final chunk with accumulated text
    sink(SimpleNamespace(text_delta="foo", is_final=False))
    sink(SimpleNamespace(text_delta="bar", is_final=False))
    sink(SimpleNamespace(text_delta="foobar", is_final=True))

    # Buffer must be reset
    assert sink._buffer == ""

    # STREAM_DONE must be on the queue exactly once
    items = []
    while not trace._queue.empty():
        items.append(trace._queue.get_nowait())
    done_items = [i for i in items if i[0] == "stream_done"]
    assert len(done_items) == 1


def test_emit_llm_start_finalizes_stream_and_resets_sink():
    """LLM_START triggers STREAM_DONE and stream_sink.reset().

    This is the turn-boundary cleanup that prevents the previous turn's
    streaming widget from collecting new deltas.
    """
    from types import SimpleNamespace

    from maike.tui.bridge import TUIStreamSink, TUITraceSink

    app = FakeApp()
    trace = TUITraceSink(app)
    sink = TUIStreamSink(app, trace)
    app._stream_sink = sink

    # Put some stale text in the accumulator
    sink(SimpleNamespace(text_delta="stale", is_final=False))
    assert sink._buffer == "stale"

    # Drain the delta we just pushed (not what we're testing)
    while not trace._queue.empty():
        trace._queue.get_nowait()

    # LLM_START must emit STREAM_DONE and reset the sink
    trace.emit(_make_event(TraceEventKind.LLM_START, model="test-model"))
    assert sink._buffer == ""

    items = []
    while not trace._queue.empty():
        items.append(trace._queue.get_nowait())
    kinds = [i[0] for i in items]
    assert "stream_done" in kinds
    assert "llm_start" in kinds


def test_on_prompt_blocks_and_returns():
    """on_prompt posts to queue and blocks until approval callback fires."""
    import threading as _threading
    from maike.tui.bridge import TUITraceSink

    class ApprovalApp(FakeApp):
        def _show_approval_list(self, prompt, callback):
            self._approval_callback = callback
            self._waiting_for_approval = True
            callback(True)  # auto-approve

    app = ApprovalApp()
    sink = TUITraceSink(app)
    result_holder = [None]

    # on_prompt blocks, so run it in a thread
    def _call():
        result_holder[0] = sink.on_prompt("Approve this?")

    t = _threading.Thread(target=_call)
    t.start()
    # Give it a moment to post to queue and start waiting
    t.join(timeout=0.05)
    # Drain the queue — this processes the approval_request on "main thread"
    sink._drain_queue()
    t.join(timeout=1.0)
    assert not t.is_alive(), "on_prompt still blocked"
    assert result_holder[0] is True


def test_on_text_prompt_returns_text():
    """on_text_prompt posts to queue and blocks until dismiss callback fires."""
    import threading as _threading
    from maike.tui.bridge import TUITraceSink

    class TextApp(FakeApp):
        def push_screen(self, screen, callback=None):
            self._screens_pushed.append((screen, callback))
            if callback:
                callback("use pytest instead")

    app = TextApp()
    sink = TUITraceSink(app)
    result_holder = [None]

    def _call():
        result_holder[0] = sink.on_text_prompt("How should we proceed?")

    t = _threading.Thread(target=_call)
    t.start()
    t.join(timeout=0.05)
    sink._drain_queue()
    t.join(timeout=1.0)
    assert not t.is_alive(), "on_text_prompt still blocked"
    assert result_holder[0] == "use pytest instead"


def test_elapsed_time_tracking():
    """Start time is recorded on first LLM_START."""
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    assert sink._start_time is None
    sink.emit(_make_event(TraceEventKind.LLM_START, model="test"))
    assert sink._start_time is not None


def test_cumulative_cost_tracking():
    """Cost and tokens accumulate across LLM calls."""
    from maike.tui.bridge import TUITraceSink

    app = FakeApp()
    sink = TUITraceSink(app)

    sink.emit(_make_event(
        TraceEventKind.LLM_CALL, model="test",
        total_tokens=100, cost_usd=0.01,
    ))
    sink.emit(_make_event(
        TraceEventKind.LLM_CALL, model="test",
        total_tokens=200, cost_usd=0.02,
    ))

    assert sink._cumulative_tokens == 300
    assert abs(sink._cumulative_cost - 0.03) < 0.001
