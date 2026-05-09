"""Bridge adapters between mAIke's orchestrator and the Textual TUI.

Events are buffered in a thread-safe queue and drained by a periodic
timer on the Textual main thread. This avoids deadlocks from trying
to mount widgets or post messages from within worker coroutines.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from rich.markup import escape as _rich_escape

from maike.observability.tracer import TraceEvent, TraceEventKind, TraceSink


# Internal queue labels — used only inside TUITraceSink to pair
# emit() extractions with _process() handlers.  Not the same as
# TraceEventKind; these are the *processed* UI actions, not raw events.
_Q_TOOL_START = "tool_start"
_Q_TOOL_RESULT = "tool_result"
_Q_LLM_START = "llm_start"
_Q_LLM_CALL = "llm_call"
_Q_STREAM_DELTA = "stream_delta"
_Q_STREAM_DONE = "stream_done"
_Q_DELEGATE_SPAWN = "delegate_spawn"
_Q_DELEGATE_COMPLETE = "delegate_complete"
_Q_TURN_SEPARATOR = "turn_separator"
_Q_APPROVAL_REQUEST = "approval_request"
_Q_TEXT_PROMPT_REQUEST = "text_prompt_request"

# Delegate tool name — TOOL_START/TOOL_RESULT for this tool are
# suppressed from the main UI.  The delegate lifecycle is shown via
# ASYNC_DELEGATE_SPAWN/COMPLETE events instead, grouped in
# DelegateWidget collapsibles.
_DELEGATE_TOOL_NAME = "Delegate"


class TUITraceSink(TraceSink):
    """Buffers trace events for deferred processing on the main thread.

    Two registries keep the logic extensible:

    * ``_emitters`` maps a ``TraceEventKind`` to a function that extracts
      the relevant data from a :class:`TraceEvent` and enqueues a
      ``(queue_label, data)`` tuple.
    * ``_processors`` maps a queue label to a function that applies the
      data to the TUI widgets.

    Adding a new event kind only requires adding entries to these dicts
    — no if/elif chains to modify.
    """

    def __init__(self, app: Any) -> None:
        self._app = app
        self._queue: queue.Queue = queue.Queue()
        self._cumulative_cost: float = 0.0
        self._cumulative_tokens: int = 0
        self._timer = app.set_interval(0.1, self._drain_queue)

        # -- Tracking state ------------------------------------------------
        self._pending_tools: dict[str, deque[str]] = defaultdict(deque)
        self._pending_delegates: dict[str, str] = {}  # task_key → widget key
        self._delegate_counter: int = 0
        self._iteration: int = 0
        self._start_time: float | None = None
        self._last_event_was_tool_result: bool = False
        self._agent_state: str = "waiting"
        self._llm_start_time: float = 0.0

        # -- Emit handlers: TraceEventKind → (event) → enqueue ----------
        self._emitters: dict[str, Callable[[TraceEvent], None]] = {
            TraceEventKind.TOOL_START: self._emit_tool_start,
            TraceEventKind.TOOL_RESULT: self._emit_tool_result,
            TraceEventKind.LLM_START: self._emit_llm_start,
            TraceEventKind.LLM_CALL: self._emit_llm_call,
            TraceEventKind.ASYNC_DELEGATE_SPAWN: self._emit_delegate_spawn,
            TraceEventKind.ASYNC_DELEGATE_COMPLETE: self._emit_delegate_complete,
        }

        # -- Process handlers: queue label → (ml, data) → update UI -----
        self._processors: dict[str, Callable[[Any, dict], None]] = {
            _Q_TOOL_START: self._process_tool_start,
            _Q_TOOL_RESULT: self._process_tool_result,
            _Q_LLM_START: self._process_llm_start,
            _Q_LLM_CALL: self._process_llm_call,
            _Q_STREAM_DELTA: self._process_stream_delta,
            _Q_STREAM_DONE: self._process_stream_done,
            _Q_DELEGATE_SPAWN: self._process_delegate_spawn,
            _Q_DELEGATE_COMPLETE: self._process_delegate_complete,
            _Q_TURN_SEPARATOR: self._process_turn_separator,
            _Q_APPROVAL_REQUEST: self._process_approval_request,
            _Q_TEXT_PROMPT_REQUEST: self._process_text_prompt_request,
        }

    # ------------------------------------------------------------------
    # Queue drain (main Textual thread)
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        """Called every 100ms on the main Textual thread."""
        while True:
            try:
                kind, data = self._queue.get_nowait()
            except queue.Empty:
                break
            self._process(kind, data)

    def _process(self, kind: str, data: dict) -> None:
        """Dispatch a single buffered event to its processor."""
        try:
            ml = self._app.query_one("MessageList")
        except Exception:
            return
        handler = self._processors.get(kind)
        if handler is not None:
            handler(ml, data)

    # ------------------------------------------------------------------
    # Emit (worker thread → queue)
    # ------------------------------------------------------------------

    def emit(self, event: TraceEvent) -> None:
        """Buffer the event for processing on the main thread."""
        handler = self._emitters.get(event.kind)
        if handler is not None:
            handler(event)

    # -- Per-kind emit extractors --------------------------------------

    @staticmethod
    def _is_delegate_event(event: TraceEvent) -> bool:
        """True if this event comes from a delegate sub-agent."""
        return event.agent_role == "delegate"

    def _emit_tool_start(self, event: TraceEvent) -> None:
        if self._is_delegate_event(event):
            return  # Delegate sub-agent internal tool calls hidden.
        # Suppress the Delegate tool itself — lifecycle shown via
        # ASYNC_DELEGATE_SPAWN/COMPLETE events in DelegateWidget.
        if (event.tool_name or "") in (_DELEGATE_TOOL_NAME, "delegate"):
            return
        self._last_event_was_tool_result = False
        activity = (event.payload or {}).get("activity", "")
        hint = self._tool_hint(event) or activity
        self._queue.put((_Q_TOOL_START, {
            "tool_name": event.tool_name or "",
            "hint": hint,
        }))

    def _emit_tool_result(self, event: TraceEvent) -> None:
        if self._is_delegate_event(event):
            return
        # Suppress *successful* Delegate tool results and the specific
        # max_async_delegates_exceeded error (agent retries internally).
        # Surface other Delegate failures (budget exhaustion, spawn crashes)
        # so the user knows something went wrong.
        if (event.tool_name or "") in (_DELEGATE_TOOL_NAME, "delegate"):
            payload = event.payload or {}
            error = str(payload.get("error", ""))
            is_success = event.success if event.success is not None else True
            if is_success or error == "max_async_delegates_exceeded":
                return
            # Fall through — show genuine delegate failures
        self._last_event_was_tool_result = True
        payload = event.payload or {}

        # Extract full output (no truncation here — the widget retains
        # the complete text so the user can expand it via Ctrl+O).  The
        # widget applies its own inline preview cap for the collapsible.
        output = str(payload.get("output", ""))
        if not output:
            output = str(payload.get("raw_output", ""))

        self._queue.put((_Q_TOOL_RESULT, {
            "tool_name": event.tool_name or "",
            "success": event.success if event.success is not None else True,
            "error": str(payload.get("error")) if payload.get("error") else None,
            "output": output,
            "execution_ms": event.execution_ms or 0,
        }))

    def _emit_llm_start(self, event: TraceEvent) -> None:
        if self._is_delegate_event(event):
            return
        # Track iteration — LLM_START after TOOL_RESULT = new iteration
        if self._last_event_was_tool_result:
            self._iteration += 1
            self._queue.put((_Q_TURN_SEPARATOR, {
                "iteration": self._iteration,
            }))
        self._last_event_was_tool_result = False

        # Finalize any in-progress streaming widget BEFORE starting a new
        # LLM call.  Without this, the #streaming-block from the previous
        # iteration persists, and new stream deltas append to old text
        # (the "previous-turn output bleeds into latest response" bug).
        # Pair this with a stream_sink reset so the accumulator starts empty.
        self._queue.put((_Q_STREAM_DONE, {}))
        stream_sink = getattr(self._app, "_stream_sink", None)
        if stream_sink is not None and hasattr(stream_sink, "reset"):
            stream_sink.reset()

        # Track session start time
        if self._start_time is None:
            self._start_time = time.monotonic()

        self._queue.put((_Q_LLM_START, {
            "model": event.model or event.provider or "LLM",
        }))

    def _emit_llm_call(self, event: TraceEvent) -> None:
        if self._is_delegate_event(event):
            return
        tokens = event.total_tokens or 0
        cost = event.cost_usd or 0.0
        self._cumulative_cost += cost
        self._cumulative_tokens += tokens
        self._queue.put((_Q_LLM_CALL, {
            "model": event.model or "LLM",
            "tokens": tokens,
            "cost": cost,
            "thinking": (event.payload or {}).get("thinking", ""),
        }))

    def _emit_delegate_spawn(self, event: TraceEvent) -> None:
        payload = event.payload or {}
        self._queue.put((_Q_DELEGATE_SPAWN, {
            "task": payload.get("task", ""),
        }))

    def _emit_delegate_complete(self, event: TraceEvent) -> None:
        payload = event.payload or {}
        self._queue.put((_Q_DELEGATE_COMPLETE, {
            "task": payload.get("task", ""),
            "success": payload.get("success", False),
            "cost": payload.get("cost_usd", 0),
            "output": payload.get("output", ""),
        }))

    # ------------------------------------------------------------------
    # Per-kind process handlers (main thread, receives MessageList)
    # ------------------------------------------------------------------

    def _process_tool_start(self, ml: Any, data: dict) -> None:
        ml.stop_spinner()
        # Finalize any active streaming block so tools appear AFTER text
        ml.finalize_streaming()
        tool_name = data["tool_name"]
        key = ml.add_tool_start(tool_name, data["hint"])
        self._pending_tools[tool_name].append(key)
        ml.start_spinner(tool_name)
        self._set_agent_state("executing")

    def _process_tool_result(self, ml: Any, data: dict) -> None:
        ml.stop_spinner()
        tool_name = data.get("tool_name", "")

        # Match result to the correct tool widget by name
        key = ""
        if tool_name and self._pending_tools[tool_name]:
            key = self._pending_tools[tool_name].popleft()
        else:
            # Fallback: pop from any non-empty deque (FIFO across all tools)
            for name, dq in self._pending_tools.items():
                if dq:
                    key = dq.popleft()
                    break

        if key:
            ml.complete_tool(
                key,
                data["success"],
                data.get("output", ""),
                data.get("error"),
                data.get("execution_ms", 0),
            )

    def _process_llm_start(self, ml: Any, data: dict) -> None:
        self._llm_start_time = time.monotonic()
        # Clear the streaming-finalized latch on the message list so this
        # turn's first STREAM_DELTA is accepted.  Without this, any late
        # DELTA that drained after the PREVIOUS turn's finalize would
        # have locked the latch on — but a new turn needs a fresh widget.
        begin = getattr(ml, "begin_streaming_turn", None)
        if callable(begin):
            begin()
        ml.start_spinner(data["model"])
        self._set_agent_state("thinking")

    def _process_llm_call(self, ml: Any, data: dict) -> None:
        ml.stop_spinner()
        # Show duration for calls that completed too fast for the spinner to render
        elapsed = time.monotonic() - self._llm_start_time if self._llm_start_time else 0
        dur_str = f" {elapsed:.1f}s" if elapsed > 0 else ""
        from maike.tui.theme import DOT_ACTIVE
        ml.add_info(
            f"{DOT_ACTIVE} [dim]{_rich_escape(str(data['model']))} ({data['tokens']:,} tok, "
            f"${data['cost']:.4f},{dur_str})[/dim]"
        )
        if data.get("thinking"):
            ml.add_thinking(data["thinking"])

        # Update status bar
        try:
            sb = self._app.query_one("StatusBar")
            sb._cost = self._cumulative_cost
            sb._tokens = self._cumulative_tokens
            sb._model = data["model"]
            sb._iteration = self._iteration
            if self._start_time is not None:
                sb.set_start_time(self._start_time)
            sb.update(sb._render_text())
        except Exception:
            pass

    def _process_stream_delta(self, ml: Any, data: dict) -> None:
        # Stop the LLM-start spinner once the model starts producing text —
        # the spinner below the stream competes for the user's eye and
        # contributes to perceived jitter.  The LLM_CALL processor is
        # already a no-op on the spinner after this.
        ml.stop_spinner()
        ml.update_streaming(data["text"], render_markdown=data.get("render", True))
        # Mark that streaming delivered content — prevents duplicate output.
        try:
            self._app._streaming_shown = True
        except Exception:
            pass

    def _process_stream_done(self, ml: Any, data: dict) -> None:
        ml.finalize_streaming()

    def _process_delegate_spawn(self, ml: Any, data: dict) -> None:
        """Create a DelegateWidget — groups spawn + result lifecycle."""
        ml.finalize_streaming()  # ensure delegates appear after text
        self._delegate_counter += 1
        key = f"delegate-{self._delegate_counter}"
        task = data.get("task", "")
        task_key = task[:80]  # normalized key for matching complete events

        widget_key = ml.add_delegate(task, key)
        self._pending_delegates[task_key] = widget_key

    def _process_delegate_complete(self, ml: Any, data: dict) -> None:
        """Update existing DelegateWidget with completion status and output."""
        task = data.get("task", "")
        task_key = task[:80]

        widget_key = self._pending_delegates.pop(task_key, "")
        if widget_key:
            ml.complete_delegate(
                widget_key,
                data.get("success", False),
                data.get("cost", 0),
                data.get("output", ""),
            )
        else:
            # Fallback: no matching spawn found, show inline
            if data.get("success"):
                ml.add_info(
                    f"[bold green]Delegate completed:[/bold green] {_rich_escape(task[:60])} "
                    f"[dim](${data.get('cost', 0):.3f})[/dim]"
                )
            else:
                ml.add_info(
                    f"[bold red]Delegate failed:[/bold red] {_rich_escape(task[:60])}"
                )

    def _process_turn_separator(self, ml: Any, data: dict) -> None:
        ml.add_turn_separator(data.get("iteration", 0))

    def _process_approval_request(self, ml: Any, data: dict) -> None:
        """Show approval list — runs on main thread, safe to mount widgets."""
        prompt = data.get("prompt", "")
        safe_prompt = prompt.replace("[", "\\[")
        ml.add_info(f"[bold yellow]\u26a0 Approval:[/bold yellow] {safe_prompt}")
        cb = getattr(self, "_approval_on_response", None)
        if cb:
            self._app._show_approval_list(prompt, cb)

    def _process_text_prompt_request(self, ml: Any, data: dict) -> None:
        """Show text prompt modal — runs on main thread."""
        prompt = data.get("prompt", "")
        cb = getattr(self, "_text_prompt_on_dismiss", None)
        if cb:
            try:
                from maike.tui.screens.text_prompt_screen import TextPromptScreen
                self._app.push_screen(TextPromptScreen(prompt), callback=cb)
            except ImportError:
                # Fallback for environments without textual (e.g. tests with FakeApp)
                self._app.push_screen(prompt, callback=cb)

    # ------------------------------------------------------------------
    # Agent state tracking
    # ------------------------------------------------------------------

    def _set_agent_state(self, state: str) -> None:
        self._agent_state = state
        try:
            hb = self._app.query_one("HeaderBar")
            hb.set_agent_state(state)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ApprovalGate interface — inline approval via the input area
    # ------------------------------------------------------------------

    def on_prompt(self, prompt: str) -> bool:
        """Block worker thread until user selects from approval list.

        Posts an approval request to the queue (processed on main thread
        by _process_approval_request).  Blocks on threading.Event until
        the user responds.  No call_from_thread — no deadlock risk.
        """
        gate = threading.Event()
        result = [False]

        def _on_response(approved: bool) -> None:
            result[0] = approved
            gate.set()

        self._approval_gate = gate
        self._approval_result = result
        self._approval_on_response = _on_response
        self._queue.put((_Q_APPROVAL_REQUEST, {"prompt": prompt}))
        gate.wait()
        return result[0]

    def on_text_prompt(self, prompt: str) -> str:
        """Block worker thread until user types free-text response.

        Posts a text prompt request to the queue.  Blocks on
        threading.Event until the user responds.
        """
        gate = threading.Event()
        result = [""]

        def _on_dismiss(text: str) -> None:
            result[0] = text
            gate.set()

        self._text_prompt_gate = gate
        self._text_prompt_result = result
        self._text_prompt_on_dismiss = _on_dismiss
        self._queue.put((_Q_TEXT_PROMPT_REQUEST, {"prompt": prompt}))
        gate.wait()
        return result[0]

    @staticmethod
    def _tool_hint(event: TraceEvent) -> str:
        payload = event.payload or {}
        inp = payload.get("input", {})
        if isinstance(inp, dict):
            for key in ("path", "cmd", "pattern", "query", "url", "task"):
                val = inp.get(key)
                if val:
                    s = str(val)
                    return s[:60] if len(s) > 60 else s
        return ""


class TUIStreamSink:
    """Buffers stream chunks for processing on the main thread.

    Accumulates incremental deltas into a single buffer per LLM turn, then
    emits the full buffered text via ``_Q_STREAM_DELTA`` so the message
    list can update in place.

    We throttle aggressively — rendering the full accumulated buffer
    through Rich's text pipeline on every delta was burning ~1.5 ms per
    call, so a 3 KB response cost 400 ms+ of UI work before the user
    even saw the final markdown.  Instead we enqueue at most one render
    per :data:`_RENDER_INTERVAL_S` seconds (or when the stream finishes),
    and the queue drain batches any extras.

    :meth:`reset` clears the accumulator between turns so new-turn deltas
    do not bleed stale content into the current streaming widget.
    """

    # Render at most ~12 Hz during streaming — fast enough to feel live,
    # slow enough to avoid burning CPU on a buffer we're about to
    # overwrite with more deltas milliseconds later.
    _RENDER_INTERVAL_S = 0.08

    def __init__(self, app: Any, trace_sink: TUITraceSink) -> None:
        self._trace_sink = trace_sink
        self._buffer = ""
        self._last_render_time = 0.0

    def reset(self) -> None:
        """Drop any buffered text; the next delta starts a fresh stream."""
        self._buffer = ""
        self._last_render_time = 0.0

    def __call__(self, chunk: Any) -> None:
        delta = getattr(chunk, "text_delta", "") or ""
        is_final = getattr(chunk, "is_final", False)

        # The final chunk carries the full accumulated text as text_delta
        # (not an incremental delta).  Skip appending it — our buffer
        # already has the full content from incremental deltas, and
        # appending would double the text.
        #
        # BUT: throttling (_RENDER_INTERVAL_S) can swallow the trailing
        # deltas — if the last few tokens arrive within 80ms of the
        # previous render, no DELTA gets enqueued for them, and going
        # straight to STREAM_DONE means the user sees a truncated reply
        # ("We are in the year" instead of "We are in the year 2026.").
        # So before the final STREAM_DONE, we always emit one last
        # STREAM_DELTA with the complete buffer.  The order is preserved
        # by the queue, so STREAM_DONE still finalizes the widget that
        # now has the full text.
        if is_final:
            if self._buffer:
                self._trace_sink._queue.put(
                    (_Q_STREAM_DELTA, {"text": self._buffer, "render": True})
                )
            self._trace_sink._queue.put((_Q_STREAM_DONE, {}))
            self._buffer = ""
            self._last_render_time = 0.0
            return

        if not delta:
            return

        self._buffer += delta
        now = time.monotonic()
        if now - self._last_render_time >= self._RENDER_INTERVAL_S:
            self._last_render_time = now
            self._trace_sink._queue.put(
                (_Q_STREAM_DELTA, {"text": self._buffer, "render": True})
            )
