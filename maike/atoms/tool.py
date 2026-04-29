"""Tool-related atom models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DESTRUCTIVE = "destructive"


class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: str = ""
    raw_output: str = ""
    error: str | None = None
    execution_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def blocked(
        cls,
        tool_name: str,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        payload = {"blocked": True, "block_reason": reason}
        if metadata:
            payload.update(metadata)
        return cls(
            tool_name=tool_name,
            success=False,
            output=f"[BLOCKED] {reason}",
            raw_output="",
            error=reason,
            metadata=payload,
        )
