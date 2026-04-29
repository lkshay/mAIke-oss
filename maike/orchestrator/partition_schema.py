"""Pydantic schema for the partition planner's structured LLM output.

The partition planner (``Orchestrator._plan_partitions``) asks a model to
decompose a coding task into non-overlapping file-scoped sub-tasks that
can run in parallel.  Previously the model was prompted to emit a JSON
array which was then ``json.loads``'d — unreliable at scale.

This schema is passed via ``LLMRequest.response_schema`` so each adapter
routes the call through its provider-native structured-output API.  The
resulting dict is surfaced on ``LLMResult.parsed`` and validated here
before the orchestrator operates on it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PartitionEntry(BaseModel):
    """One sub-task in a partition plan.

    ``files`` is the exclusive file scope for this partition — the agent
    is only permitted to modify files in this list.  Scopes must not
    overlap across partitions; the orchestrator checks this after parsing.
    """

    subtask: str = Field(
        ..., min_length=1,
        description="Short description of what this partition does.",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Paths (relative to workspace) this partition may modify.",
    )


class PartitionPlan(BaseModel):
    """Root schema.  Wraps a list so adapters that need a JSON *object*
    root (e.g. Anthropic tool-use input_schema) accept it cleanly.

    ``max_length=5`` enforces the same cap the free-text parser used —
    more than five parallel partitions is never the right call.
    """

    partitions: list[PartitionEntry] = Field(
        default_factory=list, max_length=5,
    )
