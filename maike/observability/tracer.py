"""Structured tracing hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from maike.atoms.llm import LLMCallRecord
from maike.atoms.tool import ToolResult
from maike.tools.context import peek_current_agent_context
from maike.utils import utcnow

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace as otel_trace
except ImportError:  # pragma: no cover
    otel_trace = None


class TraceEventKind:
    """Known trace event kinds — single source of truth.

    Use these constants wherever you emit or match on event kinds.
    The generic ``log_event`` still accepts arbitrary strings for
    ad-hoc/one-off events, but well-known kinds live here.
    """

    # LLM lifecycle
    LLM_START = "llm_start"
    LLM_CALL = "llm_call"

    # Tool lifecycle
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"

    # Pipeline stages
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"

    # Agent lifecycle
    AGENT_SPAWN = "agent_spawn"
    AGENT_COMPLETE = "agent_complete"

    # Delegation
    DELEGATE_SPAWN = "delegate_spawn"
    DELEGATE_COMPLETE = "delegate_complete"
    ASYNC_DELEGATE_SPAWN = "async_delegate_spawn"
    ASYNC_DELEGATE_COMPLETE = "async_delegate_complete"

    # Session
    SESSION_CANCELLED = "session_cancelled"

    # Partitioning
    PARTITION_FANOUT_START = "partition_fanout_start"
    PARTITION_FANIN_START = "partition_fanin_start"
    PARTITION_COMPLETE = "partition_complete"

    # Advisor lifecycle
    ADVISOR_CALL = "advisor_call"
    ADVISOR_THROTTLED = "advisor_throttled"


class TraceEvent(BaseModel):
    kind: str
    timestamp: datetime = Field(default_factory=utcnow)
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    stop_reason: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    agent_role: str | None = None
    stage_name: str | None = None
    tool_profile: str | None = None
    tool_name: str | None = None
    success: bool | None = None
    execution_ms: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class OpenTelemetrySink:
    def __init__(self, tracer_name: str = "maike") -> None:
        self._tracer_name = tracer_name

    def emit(self, event: TraceEvent) -> None:
        if otel_trace is None:  # pragma: no cover - optional dependency
            return
        tracer = otel_trace.get_tracer(self._tracer_name)
        with tracer.start_as_current_span(f"maike.{event.kind}") as span:
            for key, value in event.model_dump(exclude_none=True).items():
                if key == "payload":
                    for payload_key, payload_value in value.items():
                        span.set_attribute(f"maike.payload.{payload_key}", str(payload_value))
                    continue
                if key == "timestamp":
                    span.set_attribute("maike.timestamp", value.isoformat())
                    continue
                span.set_attribute(f"maike.{key}", value)


@dataclass
class Tracer:
    sink: TraceSink | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def log_llm_call(self, record: LLMCallRecord, *, payload: dict[str, Any] | None = None) -> None:
        ctx = peek_current_agent_context()
        event = TraceEvent(
            kind=TraceEventKind.LLM_CALL,
            provider=record.provider,
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            cost_usd=record.cost_usd,
            latency_ms=record.latency_ms,
            stop_reason=record.stop_reason,
            session_id=record.session_id,
            agent_id=record.agent_id,
            stage_name=record.stage_name,
            tool_profile=record.tool_profile,
            payload=payload or {},
        )
        self._emit(self._apply_context(event, ctx))

    def log_llm_start(
        self,
        provider: str,
        model: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            kind=TraceEventKind.LLM_START,
            provider=provider,
            model=model,
            payload=payload or {},
        )
        self._emit(self._apply_context(event, peek_current_agent_context()))

    def log_tool_start(
        self,
        tool_name: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            kind=TraceEventKind.TOOL_START,
            tool_name=tool_name,
            payload=payload or {},
        )
        self._emit(self._apply_context(event, peek_current_agent_context()))

    def log_tool_result(self, result: ToolResult) -> None:
        event = TraceEvent(
            kind=TraceEventKind.TOOL_RESULT,
            tool_name=result.tool_name,
            success=result.success,
            execution_ms=result.execution_ms,
            payload={
                "error": result.error,
                "output": result.output,
                "raw_output": result.raw_output,
                **result.metadata,
            },
        )
        self._emit(self._apply_context(event, peek_current_agent_context()))

    def log_stage_start(self, stage_name: str, *, payload: dict[str, Any] | None = None) -> None:
        event = TraceEvent(
            kind=TraceEventKind.STAGE_START,
            stage_name=stage_name,
            payload=payload or {},
        )
        self._emit(self._apply_context(event, peek_current_agent_context()))

    def log_stage_complete(
        self,
        stage_name: str,
        *,
        success: bool,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            kind=TraceEventKind.STAGE_COMPLETE,
            stage_name=stage_name,
            success=success,
            payload=payload or {},
        )
        self._emit(self._apply_context(event, peek_current_agent_context()))

    def log_agent_spawn(
        self,
        *,
        session_id: str,
        agent_id: str,
        agent_role: str,
        stage_name: str,
        tool_profile: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            kind=TraceEventKind.AGENT_SPAWN,
            session_id=session_id,
            agent_id=agent_id,
            agent_role=agent_role,
            stage_name=stage_name,
            tool_profile=tool_profile,
            payload=payload or {},
        )
        self._emit(event)

    def log_agent_complete(
        self,
        *,
        session_id: str,
        agent_id: str,
        agent_role: str,
        stage_name: str,
        success: bool,
        cost_usd: float | None = None,
        total_tokens: int | None = None,
        tool_profile: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            kind=TraceEventKind.AGENT_COMPLETE,
            session_id=session_id,
            agent_id=agent_id,
            agent_role=agent_role,
            stage_name=stage_name,
            tool_profile=tool_profile,
            success=success,
            cost_usd=cost_usd,
            total_tokens=total_tokens,
            payload=payload or {},
        )
        self._emit(event)

    def log_event(self, kind: str, **payload: Any) -> None:
        event = TraceEvent(kind=kind, payload=payload)
        self._emit(self._apply_context(event, peek_current_agent_context()))

    def log_context_event(
        self,
        *,
        event_type: str,
        agent_id: str | None = None,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
        levels: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log a structured context management event (prune, compress, fetch)."""
        event = TraceEvent(
            kind=f"context_{event_type}",
            payload={
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "levels": levels,
                **(payload or {}),
            },
        )
        ctx = peek_current_agent_context()
        if ctx is not None:
            event = self._apply_context(event, ctx)
        if agent_id:
            event = event.model_copy(update={"agent_id": agent_id})
        self._emit(event)

    def _emit(self, event: TraceEvent) -> None:
        dumped = event.model_dump(mode="json")
        self.events.append(dumped)
        if self.sink is not None:
            self.sink.emit(event)

    def _apply_context(self, event: TraceEvent, ctx) -> TraceEvent:
        if ctx is None:
            return event
        payload = event.model_dump()
        payload["session_id"] = payload["session_id"] or ctx.metadata.get("session_id")
        payload["agent_id"] = payload["agent_id"] or ctx.agent_id
        payload["agent_role"] = payload["agent_role"] or ctx.role
        payload["stage_name"] = payload["stage_name"] or ctx.stage_name
        payload["tool_profile"] = payload["tool_profile"] or ctx.tool_profile
        return TraceEvent(**payload)
