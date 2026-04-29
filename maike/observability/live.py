"""Trace sinks for operator-facing CLI output."""

from __future__ import annotations

import sys
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from maike.observability.tracer import TraceEvent, TraceEventKind, TraceSink

try:  # pragma: no cover - optional dependency
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Syntax = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]

# Re-export unused names so existing imports don't break.
Group = None  # type: ignore[assignment]
Live = None  # type: ignore[assignment]
Panel = None  # type: ignore[assignment]
Table = None  # type: ignore[assignment]


@dataclass
class CompositeTraceSink:
    sinks: list[TraceSink] = field(default_factory=list)
    _stack: ExitStack | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "CompositeTraceSink":
        self._stack = ExitStack()
        for sink in self.sinks:
            enter = getattr(sink, "__enter__", None)
            if callable(enter):
                self._stack.enter_context(sink)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def emit(self, event: TraceEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)

    def on_prompt(self, prompt: str) -> bool | None:
        for sink in self.sinks:
            if hasattr(sink, "pause_for_input"):
                return sink.pause_for_input(prompt)
        return None

    def on_text_prompt(self, prompt: str) -> str | None:
        """Route a free-text prompt to the first sink that supports it."""
        for sink in self.sinks:
            if hasattr(sink, "pause_for_text_input"):
                return sink.pause_for_text_input(prompt)
        return None


# ── Simple print-based sink (replaces RichLiveSink) ─────────────────

def _tool_cmd_hint(event: TraceEvent) -> str:
    """Extract a short command/path hint from a tool event payload."""
    payload = event.payload or {}
    inp = payload.get("input", {})
    if not isinstance(inp, dict):
        return ""
    cmd = inp.get("cmd")
    if cmd:
        cmd = str(cmd).strip()
        return cmd if len(cmd) <= 60 else cmd[:57] + "..."
    path = inp.get("path") or inp.get("file_path")
    if path:
        parts = str(path).replace("\\", "/").split("/")
        short = parts[-1] if parts else str(path)
        return short if len(short) <= 40 else short[:37] + "..."
    pattern = inp.get("pattern")
    if pattern:
        pattern = str(pattern).strip()
        return f"/{pattern}/" if len(pattern) <= 30 else f"/{pattern[:27]}.../"
    return ""


def _short_error(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text if len(text) <= 96 else text[:93] + "..."


@dataclass
class RichLiveSink:
    """Simple line-by-line trace sink.

    Prints tool calls and LLM calls to stderr as they happen.
    No Rich Live display — just plain print lines that don't
    interfere with interactive input prompts.
    """

    console: Any = None  # ignored — kept for backward compat
    max_log_entries: int = 12  # ignored
    enabled: bool = True
    react_mode: bool = False  # ignored — always uses simple output

    def __enter__(self) -> "RichLiveSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def pause_for_input(self, prompt: str) -> bool:
        """Show prompt and wait for yes/no."""
        sys.stderr.flush()
        sys.stdout.flush()
        response = input(f"\n{prompt}")
        return response.strip().lower() in {"y", "yes"}

    def pause_for_text_input(self, prompt: str) -> str:
        """Show prompt and return free-text response."""
        sys.stderr.flush()
        sys.stdout.flush()
        print(f"\n{'─' * 60}", flush=True)
        print(prompt, flush=True)
        print(f"{'─' * 60}", flush=True)
        return input("You: ")

    def emit(self, event: TraceEvent) -> None:
        if not self.enabled:
            return
        line = self._format(event)
        if line:
            print(line, file=sys.stderr, flush=True)

    def _format(self, event: TraceEvent) -> str:
        # Skip AskUser/Delegate trace lines — they have their own UI.
        tool = event.tool_name or ""
        if tool in ("AskUser", "Delegate"):
            return ""

        if event.kind == TraceEventKind.LLM_CALL:
            tok = event.total_tokens or 0
            model = event.model or event.provider or "default"
            header = f"  LLM → {model} ({tok} tok)"
            # Show model thinking (dimmed) in non-verbose mode too.
            thinking = (event.payload or {}).get("thinking", "")
            if thinking:
                lines = thinking.strip().splitlines()
                dimmed = "\n".join(f"  \033[90m{line}\033[0m" for line in lines)
                return f"{header}\n{dimmed}"
            return header

        if event.kind == TraceEventKind.TOOL_START:
            hint = _tool_cmd_hint(event)
            display = f"{tool}: {hint}" if hint else tool
            return f"  ▶ {display}"

        if event.kind == TraceEventKind.TOOL_RESULT:
            hint = _tool_cmd_hint(event)
            display = f"{tool}: {hint}" if hint else tool
            error = _short_error(event.payload.get("error"))
            suffix = f" ({error})" if error else ""
            icon = "✓" if event.success else "✗"
            return f"  {icon} {display}{suffix}"

        return ""


class StreamingRenderer:
    """Renders LLM streaming text to the terminal in real-time.

    Receives ``StreamChunk`` objects via :meth:`on_chunk` and writes
    ``text_delta`` directly to stderr for instant feedback.  Call
    :meth:`finish` when the LLM turn ends to reset state and retrieve
    the full accumulated text.
    """

    def __init__(self, file=None):
        self._file = file or sys.stderr
        self._buffer = ""
        self._started = False

    def on_chunk(self, chunk) -> None:
        """Write a streaming text delta to the terminal."""
        # The final chunk carries the full accumulated text in text_delta
        # (not an incremental delta).  Skip it — our buffer already holds
        # the full text from incremental deltas; writing it again would
        # double the output.
        if getattr(chunk, "is_final", False):
            return
        delta = getattr(chunk, "text_delta", "") or ""
        if not delta:
            return
        if not self._started:
            self._file.write("\n")
            self._started = True
        self._file.write(delta)
        self._file.flush()
        self._buffer += delta

    def finish(self) -> str:
        """End the current stream. Returns the full accumulated text."""
        if self._started:
            self._file.write("\n")
            self._file.flush()
        text = self._buffer
        self._buffer = ""
        self._started = False
        return text

    @property
    def has_output(self) -> bool:
        """True if any text was streamed in the current turn."""
        return bool(self._buffer) or self._started


@dataclass
class VerboseConsoleSink:
    console: Console | None = None

    def __post_init__(self) -> None:
        if self.console is None and Console is not None:  # pragma: no branch - trivial selection
            self.console = Console()

    def emit(self, event: TraceEvent) -> None:
        if self.console is None or Text is None:
            return
        if event.kind == "spawn_request":
            self._print_json(
                f"spawn_request {self._prefix(event)}",
                event.payload.get("request"),
            )
            return
        if event.kind == TraceEventKind.LLM_CALL:
            thinking = event.payload.get("thinking")
            assistant_output = event.payload.get("assistant_output")
            tool_calls = event.payload.get("tool_calls")
            label = f"llm_call {self._prefix(event)} {event.model or event.provider or ''}".strip()
            self.console.rule(Text(label, style="cyan"))
            if thinking:
                self.console.print(f"[dim italic]{thinking}[/dim italic]")
            if assistant_output:
                self.console.print(assistant_output)
            if tool_calls:
                self._print_json("tool_calls", tool_calls)
            return
        if event.kind == TraceEventKind.TOOL_RESULT:
            label = f"tool_result {self._prefix(event)} {event.tool_name or ''}".strip()
            self.console.rule(Text(label, style="yellow"))
            raw_output = event.payload.get("raw_output") or event.payload.get("output")
            if raw_output:
                self.console.print(raw_output)
            error = event.payload.get("error")
            if error:
                self.console.print(f"[red]{error}[/red]")
            return
        if event.kind == TraceEventKind.AGENT_COMPLETE:
            output = event.payload.get("output")
            error = event.payload.get("error")
            if output or error:
                label = f"agent_complete {self._prefix(event)}"
                self.console.rule(Text(label, style="green" if event.success else "red"))
                if output:
                    self.console.print(output)
                if error:
                    self.console.print(f"[red]{error}[/red]")

    def _prefix(self, event: TraceEvent) -> str:
        agent = event.agent_id or "-"
        stage = event.stage_name or "-"
        role = event.agent_role or "-"
        return f"[{agent[:8]} {role} {stage}]"

    def _print_json(self, title: str, payload: Any) -> None:
        if self.console is None or payload is None:
            return
        formatted = json.dumps(payload, indent=2, sort_keys=True)
        if Syntax is not None:
            self.console.rule(Text(title, style="blue"))
            self.console.print(Syntax(formatted, "json", word_wrap=True))
            return
        self.console.print(f"{title}\n{formatted}")

@dataclass
class FileTraceSink:
    log_path: Path
    _file: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "FileTraceSink":
        self._file = open(self.log_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def emit(self, event: TraceEvent) -> None:
        if self._file is not None:
            self._file.write(event.model_dump_json(exclude_none=True) + "\n")
            self._file.flush()
