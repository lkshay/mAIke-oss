"""Task-scoped session wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from maike.atoms.agent import AgentResult
from maike.atoms.artifact import Artifact, ArtifactKind, ArtifactStatus, ArtifactType
from maike.atoms.blueprint import SpawnRequest
from maike.atoms.context import Checkpoint
from maike.intelligence.code_index import CodeIndex
from maike.memory.learning import SessionLearner
from maike.memory.session import SessionStore

if TYPE_CHECKING:
    from maike.agents.advisor import AdvisorSession
    from maike.agents.agent_resolver import AgentResolver
    from maike.agents.skill import Skill
    from maike.agents.team_resolver import TeamResolver
    from maike.memory.longterm import TypedLongTermMemory
    from maike.plugins.bundle import PluginBundle
    from maike.plugins.hook_executor import HookExecutor
    from maike.plugins.lsp_manager import LSPManager
    from maike.plugins.mcp_registry import MCPToolRegistry


@dataclass(frozen=True)
class SkillDirectories:
    """Resolved paths for skill/knowledge module discovery."""
    builtin: Path
    user: Path | None = None
    project: Path | None = None


@dataclass
class OrchestratorSession:
    store: SessionStore
    id: str
    task: str
    workspace: Path
    environment_manifest: Any | None = None
    learner: SessionLearner | None = None
    typed_memory: "TypedLongTermMemory | None" = None
    code_index: CodeIndex | None = None
    thread: dict | None = None
    skill_dirs: SkillDirectories | None = None
    plugin_skills: list[Skill] | None = None
    # Agent system fields
    agent_resolver: "AgentResolver | None" = None
    team_resolver: "TeamResolver | None" = None
    advisor_session: "AdvisorSession | None" = None
    # Plugin system fields
    plugin_bundle: PluginBundle | None = None
    mcp_registry: MCPToolRegistry | None = None
    lsp_manager: LSPManager | None = None
    hook_executor: HookExecutor | None = None

    async def get_artifact_by_name(self, name: str):
        return await self.store.get_artifact_by_name(self.id, name)

    async def get_file_artifact(self, path: str):
        return await self.store.get_file_artifact(self.id, path)

    async def list_artifacts(
        self,
        stage_name: str | None = None,
        active_only: bool = False,
        include_observed: bool = False,
    ):
        return await self.store.list_artifacts(
            self.id,
            stage_name=stage_name,
            active_only=active_only,
            include_observed=include_observed,
        )

    async def artifacts_valid(self, stage_name: str, required_outputs: list[str] | None = None) -> bool:
        return await self.store.artifacts_valid(self.id, stage_name, required_outputs)

    async def store_artifact(self, artifact: Artifact) -> Artifact:
        return await self.store.store_artifact(self.id, artifact)

    async def store_text_artifact(
        self,
        *,
        logical_name: str,
        artifact_type: ArtifactType,
        content: str,
        produced_by: str,
        stage_name: str,
        depends_on: list[str] | None = None,
        kind: ArtifactKind = ArtifactKind.STAGE,
        path: str | None = None,
        status: ArtifactStatus = ArtifactStatus.DRAFT,
    ) -> Artifact:
        artifact = Artifact(
            kind=kind,
            type=artifact_type,
            logical_name=logical_name,
            path=path,
            content=content,
            produced_by=produced_by,
            stage_name=stage_name,
            depends_on=depends_on or [],
            status=status,
        )
        return await self.store_artifact(artifact)

    async def store_checkpoint(self, checkpoint: Checkpoint) -> None:
        await self.store.store_checkpoint(self.id, checkpoint)

    async def get_latest_checkpoint(self, stage_name: str | None = None) -> Checkpoint | None:
        return await self.store.get_latest_checkpoint(self.id, stage_name)

    async def store_result(self, result: AgentResult) -> None:
        await self.store.store_agent_run(self.id, result)

    async def queue_spawn_request(self, request: SpawnRequest) -> None:
        await self.store.queue_spawn_request(self.id, request)

    async def pop_spawn_requests(self) -> list[SpawnRequest]:
        return await self.store.pop_spawn_requests(self.id)

    async def get_agent_runs(self) -> list[dict]:
        return await self.store.get_agent_runs(self.id)

    async def list_spawn_requests(self, processed: bool | None = None) -> list[dict]:
        return await self.store.list_spawn_requests(self.id, processed=processed)

    async def get_session(self) -> dict | None:
        return await self.store.get_session(self.id)

    async def update_session_metadata(self, metadata: dict[str, Any]) -> None:
        await self.store.update_session_metadata(self.id, metadata)

    def environment_context_block(self) -> str | None:
        manifest = self.environment_manifest
        if manifest is None or not hasattr(manifest, "to_context_block"):
            return None
        return manifest.to_context_block()

    def environment_metadata(self) -> dict[str, Any]:
        manifest = self.environment_manifest
        if manifest is None:
            return {}
        return {
            "environment_language": getattr(manifest, "language", None),
            "environment_source": getattr(manifest, "source", None),
            "environment_package_manager": getattr(manifest, "package_manager", None),
            "environment_interpreter": getattr(manifest, "interpreter_command", None),
            "environment_confidence": getattr(manifest, "confidence", None),
            "environment_lint_command": getattr(manifest, "lint_command", None),
            "environment_typecheck_command": getattr(manifest, "typecheck_command", None),
        }
