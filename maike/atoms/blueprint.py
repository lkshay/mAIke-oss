"""Agent blueprint and spawn-request models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from maike.constants import (
    DEFAULT_AGENT_COST_BUDGET_USD,
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_AGENT_TOKEN_BUDGET,
    DEFAULT_FACTORY_COST_BUDGET_USD,
    DEFAULT_FACTORY_TOKEN_BUDGET,
)


class SpawnReason(str, Enum):
    PARALLEL_PARTITION = "parallel_partition"
    SPECIALIST_NEEDED = "specialist_needed"
    REFLECTION = "reflection"
    DELEGATION = "delegation"


class AgentBlueprint(BaseModel):
    blueprint_id: str = Field(default_factory=lambda: str(uuid4()))
    role: str
    task: str
    stage_name: str
    tool_profile: str
    spawn_reason: SpawnReason
    input_artifact_names: list[str] = Field(default_factory=list)
    inline_context: str | None = None
    constraints: list[str] = Field(default_factory=list)
    files_in_scope: list[str] = Field(default_factory=list)
    token_budget: int = DEFAULT_AGENT_TOKEN_BUDGET
    cost_budget_usd: float = DEFAULT_AGENT_COST_BUDGET_USD
    max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS
    parent_id: str | None = None
    spawn_depth: int = 0
    terminate_condition: str = ""
    report_to: str = "orchestrator"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpawnRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    requesting_agent_id: str
    stage_name: str
    tool_profile: str
    reason: str
    suggested_role: str
    context_summary: str
    remaining_token_budget: int
    remaining_cost_budget_usd: float
    urgency: str = "normal"
