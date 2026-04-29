"""Execution context atoms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from maike.constants import (
    DEFAULT_AGENT_COST_BUDGET_USD,
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_AGENT_TOKEN_BUDGET,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MODEL,
)
from maike.utils import utcnow


class TaskState(str, Enum):
    """Lifecycle state for background tasks (delegates and processes)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentProgress:
    """Tracks tool usage and activity for progress reporting."""

    tool_use_count: int = 0
    token_count: int = 0
    iteration_count: int = 0
    last_activity: str = ""
    recent_activities: list[str] = field(default_factory=list)
    cost_usd: float = 0.0

    def record_activity(self, description: str) -> None:
        self.last_activity = description
        self.recent_activities.append(description)
        if len(self.recent_activities) > 5:
            self.recent_activities = self.recent_activities[-5:]
        self.tool_use_count += 1


class Checkpoint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    sha: str
    label: str
    step: str
    created_at: datetime = Field(default_factory=utcnow)


class AgentContext(BaseModel):
    """Mutable per-agent state.

    The tool layer mutates several fields in place while an agent run is active. This is
    safe because `AgentCore.run` executes tool calls sequentially. If tool execution ever
    becomes parallel, these mutations must be synchronized or replaced with immutable
    updates.
    """

    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    role: str
    task: str
    stage_name: str
    tool_profile: str
    started_at: datetime = Field(default_factory=utcnow)

    token_budget: int = DEFAULT_AGENT_TOKEN_BUDGET
    cost_budget_usd: float = DEFAULT_AGENT_COST_BUDGET_USD
    max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS
    tokens_used: int = 0
    cost_used_usd: float = 0.0

    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    spawn_depth: int = 0

    input_artifact_ids: list[str] = Field(default_factory=list)
    input_artifact_names: list[str] = Field(default_factory=list)

    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_LLM_TEMPERATURE
    metadata: dict[str, Any] = Field(default_factory=dict)

    critical_reminder: str | None = None
    progress: AgentProgress = Field(default_factory=AgentProgress)
