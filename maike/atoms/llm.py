"""LLM-facing atom models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StopReason(str, Enum):
    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMContentBlock(BaseModel):
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    thought_signature: bytes | None = None


class LLMResult(BaseModel):
    provider: str | None = None
    content: str | None = None
    content_blocks: list[LLMContentBlock] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Populated when the caller requested structured output via
    # ``LLMRequest.response_schema`` AND the provider's native
    # structured-output path returned valid JSON.  Consumers should call
    # ``MyModel.model_validate(result.parsed)`` for full type safety.
    # ``None`` means: schema wasn't requested, or provider didn't honor it,
    # or the response wasn't valid JSON — caller must handle all three.
    parsed: dict[str, Any] | None = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""

    text_delta: str = ""
    tool_use_delta: dict[str, Any] | None = None
    usage_update: dict[str, Any] | None = None
    is_final: bool = False
    stop_reason: str | None = None
    accumulated_result: LLMResult | None = None


class LLMCallRecord(BaseModel):
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    stop_reason: str
    session_id: str | None = None
    agent_id: str | None = None
    stage_name: str | None = None
    tool_profile: str | None = None

    @model_validator(mode="after")
    def set_total_tokens(self) -> "LLMCallRecord":
        self.total_tokens = self.input_tokens + self.output_tokens
        return self
