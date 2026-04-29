"""Agent result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    agent_id: str
    role: str
    stage_name: str
    output: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    cost_usd: float = 0.0
    tokens_used: int = 0
    success: bool = True
    input_artifact_ids: list[str] = Field(default_factory=list)
    produced_artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
