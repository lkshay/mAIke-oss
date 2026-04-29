"""Live agent registry and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from maike.agents import get_static_builder
from maike.agents.helpers import build_workspace_snapshot
from maike.atoms.agent import AgentResult
from maike.atoms.blueprint import AgentBlueprint
from maike.atoms.context import AgentContext
from maike.constants import MAX_CONCURRENT_AGENTS, MAX_SPAWN_DEPTH, SESSION_AGENT_CAP
from maike.context.budget import ContextBudgetManager
from maike.context.summarizer import ArtifactSummarizer


class SpawnLimitError(RuntimeError):
    """Raised when dynamic agent limits are exceeded."""


class AgentStatus(str, Enum):
    BORN = "born"
    WORKING = "working"
    REPORTING = "reporting"
    DEAD = "dead"


@dataclass
class LiveAgent:
    blueprint: AgentBlueprint
    context: AgentContext
    messages: list[dict]
    status: AgentStatus = AgentStatus.BORN
    result: AgentResult | None = None


class AgentRegistry:
    def __init__(self, session, agent_core, *, verbose: bool = False) -> None:
        self.session = session
        self.agent_core = agent_core
        self.verbose = verbose
        self._live: dict[str, LiveAgent] = {}
        self._all_ever_spawned: list[str] = []

    async def spawn(self, blueprint: AgentBlueprint) -> LiveAgent:
        self._enforce_limits(blueprint)
        latest_checkpoint = await self.session.get_latest_checkpoint(blueprint.stage_name)
        base_metadata = {
            "session_id": self.session.id,
            "stage_checkpoint_sha": latest_checkpoint.sha if latest_checkpoint else None,
            "mutated_paths": [],
            "blueprint_id": blueprint.blueprint_id,
            "spawn_reason": blueprint.spawn_reason.value,
            "parent_agent_id": blueprint.parent_id,
            "files_in_scope": list(blueprint.files_in_scope),
        }
        if self.verbose:
            base_metadata["verbose_trace"] = True
            base_metadata["spawn_details"] = self._build_spawn_details(blueprint, latest_checkpoint)
        base_metadata.update(self.session.environment_metadata())
        base_metadata.update(blueprint.metadata)
        if self._should_use_static_builder(blueprint):
            builder = get_static_builder(blueprint.stage_name)
            assert builder is not None
            context, messages = await builder(self.session.task, self.session)
            metadata = dict(context.metadata)
            metadata.update(base_metadata)
            context = context.model_copy(
                update={
                    "parent_id": blueprint.parent_id,
                    "spawn_depth": blueprint.spawn_depth,
                    "token_budget": blueprint.token_budget,
                    "cost_budget_usd": blueprint.cost_budget_usd,
                    "max_iterations": blueprint.max_iterations,
                    "metadata": metadata,
                }
            )
        else:
            artifacts = []
            for name in blueprint.input_artifact_names:
                artifact = await self.session.get_artifact_by_name(name)
                if artifact:
                    artifacts.append(artifact)
            context = AgentContext(
                role=blueprint.role,
                task=blueprint.task,
                stage_name=blueprint.stage_name,
                tool_profile=blueprint.tool_profile,
                parent_id=blueprint.parent_id,
                spawn_depth=blueprint.spawn_depth,
                token_budget=blueprint.token_budget,
                cost_budget_usd=blueprint.cost_budget_usd,
                max_iterations=blueprint.max_iterations,
                input_artifact_ids=[artifact.id for artifact in artifacts],
                input_artifact_names=[artifact.logical_name for artifact in artifacts],
                metadata=base_metadata,
            )
            messages = self._build_initial_messages(blueprint, artifacts)
        live = LiveAgent(blueprint=blueprint, context=context, messages=messages)
        self._live[context.agent_id] = live
        self._all_ever_spawned.append(context.agent_id)
        tracer = getattr(self.agent_core, "tracer", None)
        if tracer is not None:
            tracer.log_agent_spawn(
                session_id=self.session.id,
                agent_id=context.agent_id,
                agent_role=context.role,
                stage_name=context.stage_name,
                tool_profile=context.tool_profile,
                payload={
                    "spawn_reason": blueprint.spawn_reason.value,
                    "parent_agent_id": blueprint.parent_id,
                    "blueprint_id": blueprint.blueprint_id,
                },
            )
        return live

    def _build_spawn_details(self, blueprint: AgentBlueprint, latest_checkpoint) -> dict:
        checkpoint = None
        if latest_checkpoint is not None:
            checkpoint = {
                "id": latest_checkpoint.id,
                "sha": latest_checkpoint.sha,
                "label": latest_checkpoint.label,
                "step": latest_checkpoint.step,
            }
        return {
            "blueprint_id": blueprint.blueprint_id,
            "role": blueprint.role,
            "task": blueprint.task,
            "stage_name": blueprint.stage_name,
            "tool_profile": blueprint.tool_profile,
            "spawn_reason": blueprint.spawn_reason.value,
            "parent_id": blueprint.parent_id,
            "spawn_depth": blueprint.spawn_depth,
            "token_budget": blueprint.token_budget,
            "cost_budget_usd": blueprint.cost_budget_usd,
            "max_iterations": blueprint.max_iterations,
            "input_artifact_names": list(blueprint.input_artifact_names),
            "inline_context": blueprint.inline_context,
            "constraints": list(blueprint.constraints),
            "files_in_scope": list(blueprint.files_in_scope),
            "terminate_condition": blueprint.terminate_condition,
            "report_to": blueprint.report_to,
            "checkpoint": checkpoint,
            "metadata": blueprint.metadata,
        }

    async def run(self, live: LiveAgent) -> AgentResult:
        live.status = AgentStatus.WORKING
        tracer = getattr(self.agent_core, "tracer", None)
        try:
            result = await self.agent_core.run(live.context, live.messages)
            live.status = AgentStatus.REPORTING
            live.result = result
            if tracer is not None:
                tracer.log_agent_complete(
                    session_id=self.session.id,
                    agent_id=result.agent_id,
                    agent_role=result.role,
                    stage_name=result.stage_name,
                    tool_profile=live.context.tool_profile,
                    success=result.success,
                    cost_usd=result.cost_usd,
                    total_tokens=result.tokens_used,
                    payload={
                        "output": result.output,
                        "spawn_reason": live.blueprint.spawn_reason.value,
                    },
                )
            return result
        except Exception as exc:
            if tracer is not None:
                tracer.log_agent_complete(
                    session_id=self.session.id,
                    agent_id=live.context.agent_id,
                    agent_role=live.context.role,
                    stage_name=live.context.stage_name,
                    tool_profile=live.context.tool_profile,
                    success=False,
                    payload={
                        "error": str(exc),
                        "spawn_reason": live.blueprint.spawn_reason.value,
                    },
                )
            raise
        finally:
            live.status = AgentStatus.DEAD
            self._live.pop(live.context.agent_id, None)

    @property
    def live_count(self) -> int:
        return len(self._live)

    @property
    def total_spawned(self) -> int:
        return len(self._all_ever_spawned)

    def _enforce_limits(self, blueprint: AgentBlueprint) -> None:
        if blueprint.spawn_depth > MAX_SPAWN_DEPTH:
            raise SpawnLimitError(f"Max spawn depth {MAX_SPAWN_DEPTH} exceeded")
        if len(self._live) >= MAX_CONCURRENT_AGENTS:
            raise SpawnLimitError(f"Max concurrent agents {MAX_CONCURRENT_AGENTS} reached")
        if len(self._all_ever_spawned) >= SESSION_AGENT_CAP:
            raise SpawnLimitError(f"Session agent cap {SESSION_AGENT_CAP} reached")

    def _should_use_static_builder(self, blueprint: AgentBlueprint) -> bool:
        # PIPELINE_STAGE spawn reason was removed; static builders are no longer
        # selected via spawn reason.  Keep the method for API compatibility.
        return False

    def _build_initial_messages(self, blueprint: AgentBlueprint, artifacts) -> list[dict]:
        parts: list[str] = []
        # Proactively summarize large artifacts for dynamic agents to reduce
        # context waste even when the budget allows full content.
        summarizer = ArtifactSummarizer()
        for artifact in artifacts:
            content = artifact.content
            if summarizer.should_summarize(artifact, blueprint.role):
                content = f"[SUMMARIZED]\n{summarizer.summarize(artifact, blueprint.role)}"
            parts.append(f"## {artifact.logical_name}\n\n{content}")
        if blueprint.inline_context:
            parts.append(f"## Context\n\n{blueprint.inline_context}")
        if blueprint.constraints:
            parts.append("## Constraints\n" + "\n".join(f"- {item}" for item in blueprint.constraints))
        if blueprint.files_in_scope:
            parts.append("## File Scope\n" + "\n".join(f"- {item}" for item in blueprint.files_in_scope))
        if blueprint.metadata.get("coordination_mode") == "partition":
            parts.append(
                "## Partition Rules\n"
                "- Modify only the files listed in File Scope.\n"
                "- Do not run shell commands, install packages, or create commits in partition mode.\n"
                "- If you discover a needed change outside File Scope, report it in your result for fan-in."
            )
            parts.append(
                "## Partition Collaboration via Blackboard\n"
                "You have access to `blackboard_post` and `blackboard_read` for cross-partition communication.\n"
                "Post to the blackboard when you:\n"
                "- Define or change a shared interface, type, or API contract\n"
                "- Discover a dependency that affects other partitions\n"
                "- Change an import path or module structure that others reference\n"
                "- Find an issue that the fan-in agent needs to resolve\n\n"
                "Read the blackboard periodically to check for notes from sibling partitions. "
                "Adapt your implementation if another partition has posted relevant contract changes."
            )
        if blueprint.metadata.get("coordination_mode") == "specialist" and artifacts:
            parent_parts: list[str] = []
            for artifact in artifacts:
                content = artifact.content
                if len(content) > 500:
                    content = content[:500] + "\n[...truncated]"
                parent_parts.append(f"**{artifact.logical_name}**: {content}")
            if parent_parts:
                parts.append("## Parent Context\n\n" + "\n\n".join(parent_parts))
        environment_block = self.session.environment_context_block()
        if environment_block:
            parts.append(environment_block)

        # Workspace snapshot — scoped to files_in_scope for partition agents.
        files_in_scope = list(blueprint.files_in_scope) if blueprint.files_in_scope else None
        snapshot = build_workspace_snapshot(self.session, blueprint.role, files_in_scope=files_in_scope)
        if snapshot:
            parts.append(snapshot)

        parts.append(f"## Task\n\n{blueprint.task}")
        messages = [{"role": "user", "content": "\n\n---\n\n".join(parts)}]

        # Budget check — compress if the initial messages exceed the limit.
        if not ContextBudgetManager.fits_budget(messages, model=blueprint.metadata.get("model", "")):
            messages, _, _ = ContextBudgetManager.compress_to_fit(
                messages, model=blueprint.metadata.get("model", ""),
            )

        return messages
