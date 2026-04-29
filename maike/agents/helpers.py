"""Shared helpers for agent context builders."""

from __future__ import annotations

from typing import Any, Sequence

from pathlib import Path

from maike.atoms.artifact import Artifact, ArtifactKind
from maike.atoms.context import AgentContext
from maike.constants import DEFAULT_LLM_TEMPERATURE, DEFAULT_MODEL
from maike.context.budget import ContextBudgetManager
from maike.context.policies import StageContextPolicy
from maike.context.summarizer import summarize_artifacts
from maike.context.workspace import WorkspaceSnapshotBuilder


async def load_ordered_artifacts(session, logical_names: Sequence[str]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for logical_name in logical_names:
        artifact = await session.get_artifact_by_name(logical_name)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def build_context(
    *,
    role: str,
    task: str,
    stage_name: str,
    tool_profile: str,
    parent_id: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    input_artifacts: Sequence[Artifact] | None = None,
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> AgentContext:
    artifacts = list(input_artifacts or [])
    merged_metadata = {
        "session_id": session_id,
        "mutated_paths": [],
    }
    if metadata:
        merged_metadata.update(metadata)
    return AgentContext(
        role=role,
        task=task,
        stage_name=stage_name,
        tool_profile=tool_profile,
        parent_id=parent_id,
        model=model,
        temperature=temperature,
        input_artifact_ids=[artifact.id for artifact in artifacts],
        input_artifact_names=[artifact.logical_name for artifact in artifacts],
        metadata=merged_metadata,
    )


def build_environment_context_blocks(session) -> list[str]:
    block = getattr(session, "environment_context_block", lambda: None)()
    return [block] if block else []


def build_workspace_snapshot(
    session,
    role: str,
    *,
    files_in_scope: list[str] | None = None,
    task: str = "",
) -> str:
    """Build a role-specific workspace snapshot for the session's workspace.

    *task* is forwarded to the snapshot builder so the empty-workspace branch
    can surface a clone-first hint when the task references a remote git URL.
    """
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        return ""
    builder = WorkspaceSnapshotBuilder()
    return builder.build_snapshot_for_role(
        Path(workspace), role, files_in_scope=files_in_scope, task=task,
    )


def build_hot_context_block(session, task: str, role: str) -> str:
    """Pre-compute relevant code context using the symbol index (hot context)."""
    code_index = getattr(session, "code_index", None)
    if code_index is None or not code_index.is_built:
        return ""
    from maike.intelligence.hot_context import HotContextAssembler

    assembler = HotContextAssembler(code_index)
    hot = assembler.assemble(task, role)
    return assembler.format_hot_context(hot)


def build_environment_metadata(session) -> dict[str, Any]:
    getter = getattr(session, "environment_metadata", None)
    if getter is None:
        return {}
    return getter()


def build_messages(
    task: str,
    artifacts: Sequence[Artifact],
    *,
    raw_task_only: bool = False,
    context_blocks: Sequence[Any] | None = None,
    policy: StageContextPolicy | None = None,
    role: str = "",
    system_reminder: str | None = None,
) -> list[dict[str, str]]:
    """Build the initial user message with XML-tagged context sections.

    Accepts both ``ContextBlock`` instances (wrapped in XML tags) and bare
    strings (wrapped in ``<maike-guidance>`` for backward compatibility).
    The task is always wrapped in ``<maike-task priority="critical">``.
    """
    from maike.context.tags import ContextBlock, wrap_block, wrap_tag

    # Apply summarization policy if provided.
    if policy is not None:
        artifacts = summarize_artifacts(
            artifacts,
            full_names=policy.full_artifacts,
            summarized_names=policy.summarized_artifacts,
            omitted_names=policy.omitted_artifacts,
            role=role,
        )

    blocks = [block for block in (context_blocks or ()) if block]

    # Detect summarized artifacts for the progressive loading hint.
    summarized_names = [
        artifact.logical_name
        for artifact in artifacts
        if artifact.content.startswith("[SUMMARIZED]")
    ]

    if raw_task_only and not artifacts and not blocks and not system_reminder:
        return [{"role": "user", "content": task}]

    parts: list[str] = []

    # Inject MAIKE.md as a <system-reminder> at the very start of the first
    # user message — keeps prompt-cache-stable project context outside the
    # system prompt.
    if system_reminder:
        parts.append(
            "<system-reminder>\n"
            "As you answer the user's questions, you can use the following context:\n"
            "# Project Context (MAIKE.md)\n"
            f"{system_reminder}\n\n"
            "IMPORTANT: this context may or may not be relevant to your tasks. "
            "You should not respond to this context unless it is highly relevant "
            "to your task.\n"
            "</system-reminder>"
        )
    if artifacts:
        header = (
            "## Provided Artifacts\n\n"
            "The following artifacts are already included inline. Workflow stage artifacts such as "
            "`spec.md`, `plan.md`, `code-summary.md`, `test-results.md`, and `review.md` are internal "
            "artifacts, not guaranteed workspace files. Only use `read_file` for actual workspace paths "
            "you discovered from the workspace or when a `Workspace path:` is explicitly listed below."
        )
        if summarized_names:
            header += (
                "\n\n**Note**: Some artifacts below are marked `[SUMMARIZED]`. Use "
                "`fetch_artifact_detail(artifact_name)` to load the full content or a "
                "specific section when needed. "
                "Summarized: " + ", ".join(f"`{n}`" for n in summarized_names)
            )
        parts.append(header)
    for artifact in artifacts:
        kind_label = "stage artifact" if artifact.kind is ArtifactKind.STAGE else "workspace file artifact"
        workspace_path = artifact.path or "none"
        if artifact.kind is ArtifactKind.STAGE and artifact.path is None:
            workspace_path = "none (already inlined; not a workspace file)"
        parts.append(
            "\n".join(
                [
                    f"## Artifact: {artifact.logical_name}",
                    f"Artifact kind: {kind_label}",
                    f"Source stage: {artifact.stage_name}",
                    f"Workspace path: {workspace_path}",
                    "",
                    artifact.content,
                ]
            )
        )

    # Render context blocks — ContextBlock instances get XML tags,
    # bare strings get wrapped in <maike-guidance>.
    for block in blocks:
        if isinstance(block, ContextBlock):
            parts.append(wrap_block(block))
        elif isinstance(block, str) and block.strip():
            parts.append(wrap_tag("maike-guidance", block, priority="low"))

    # Task always last, always critical priority.
    parts.append(wrap_tag("maike-task", task, priority="critical"))

    return [{"role": "user", "content": "\n\n".join(parts)}]


def build_messages_checked(
    task: str,
    artifacts: Sequence[Artifact],
    *,
    raw_task_only: bool = False,
    context_blocks: Sequence[str] | None = None,
    model: str = "",
    tool_schemas: list[dict] | None = None,
    system_prompt: str = "",
) -> tuple[list[dict[str, str]], list[dict] | None, list[str]]:
    """Build messages and compress if they exceed the model's context budget.

    Returns ``(messages, compressed_tool_schemas_or_None, compression_levels)``.
    When no compression was needed, ``compressed_tool_schemas`` is *None* and
    the caller should use the original schemas.
    """
    messages = build_messages(
        task,
        artifacts,
        raw_task_only=raw_task_only,
        context_blocks=context_blocks,
    )
    if ContextBudgetManager.fits_budget(
        messages,
        tool_schemas=tool_schemas,
        system_prompt=system_prompt,
        model=model,
    ):
        return messages, None, []

    return ContextBudgetManager.compress_to_fit(
        messages,
        tool_schemas=tool_schemas,
        system_prompt=system_prompt,
        model=model,
    )
