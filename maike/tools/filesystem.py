"""Filesystem tools."""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from maike.atoms.artifact import ArtifactKind, ArtifactStatus, ArtifactType
from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.memory.session import SessionStore
from maike.runtime.protocol import ExecutionRuntime
from maike.tools.context import current_agent_context, peek_current_agent_context
from maike.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# File read-state tracking — shared between Read and Edit tools.
# ---------------------------------------------------------------------------

class FileReadState:
    """Tracks which files have been read and when.

    Used by Edit to verify that the agent has read a file before editing it,
    and to detect if the file was modified between the read and the edit.

    Paths are normalized to an absolute, resolved form before use so that
    `foo.py`, `./foo.py`, and `/abs/workspace/foo.py` all key to the same
    entry.  Without normalization, agents would hit spurious `file_not_read`
    errors whenever they varied the path spelling between Read and Edit.
    """

    # After this many consecutive "file not read" errors on the same path,
    # the gate self-bypasses (with a warning) to prevent approval-loop
    # spirals on cheap models that forget what they already read.  See
    # SWE-bench smoke 24 Apr 2026 — Flash Lite hit the gate, retried with
    # the same args, and burned budget cycling without progress.
    _UNREAD_BYPASS_THRESHOLD = 2

    def __init__(self) -> None:
        self._state: dict[str, float] = {}
        self._unread_errors: dict[str, int] = {}
        self._workspace: Path | None = None

    def configure(self, workspace: Path) -> None:
        """Set the workspace root used to resolve relative paths."""
        if isinstance(workspace, Path):
            self._workspace = workspace.resolve()

    def _normalize(self, path: str) -> str:
        try:
            p = Path(path)
            if p.is_absolute():
                return str(p.resolve())
            if self._workspace is not None:
                return str((self._workspace / p).resolve())
            return str(p)
        except (ValueError, OSError):
            return path

    def record_read(self, path: str) -> None:
        """Record that a file was read at the current time."""
        norm = self._normalize(path)
        self._state[norm] = time.monotonic()
        # A successful read clears any prior "not read" error count.
        self._unread_errors.pop(norm, None)

    def was_read(self, path: str) -> bool:
        """Check if a file has been read in this session."""
        return self._normalize(path) in self._state

    def note_unread_attempt(self, path: str) -> int:
        """Increment the per-path consecutive-failure counter, return new count.

        Used by Edit to decide whether to bypass the read-gate after
        ``_UNREAD_BYPASS_THRESHOLD`` repeated attempts.
        """
        norm = self._normalize(path)
        self._unread_errors[norm] = self._unread_errors.get(norm, 0) + 1
        return self._unread_errors[norm]

    def should_bypass_read_gate(self, path: str) -> bool:
        """Has the agent hit the read-gate enough times to warrant bypass?

        After bypass, the path is marked as read so subsequent Edits
        proceed normally.  Caller should log a warning.
        """
        return (
            self._unread_errors.get(self._normalize(path), 0)
            >= self._UNREAD_BYPASS_THRESHOLD
        )

    def clear(self, path: str) -> None:
        """Clear read state for a file (e.g. after write/edit changes it)."""
        norm = self._normalize(path)
        self._state.pop(norm, None)
        self._unread_errors.pop(norm, None)

    def reset(self) -> None:
        """Clear all tracked read state.

        Called when an external operation (e.g. a mutating Bash command)
        may have modified files without going through Read/Write/Edit.
        """
        self._state.clear()
        self._unread_errors.clear()


# Module-level singleton — shared across Read and Edit tool registrations
# within the same session.
_file_read_state = FileReadState()




def infer_artifact_type(path: str) -> ArtifactType:
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs"}:
        return ArtifactType.CODE
    if suffix in {".md", ".rst"}:
        return ArtifactType.DOCS
    if "test" in Path(path).name:
        return ArtifactType.TEST
    return ArtifactType.CODE


def normalize_artifact_path(path: str) -> str:
    normalized = Path(path).as_posix()
    return "." if normalized == "" else normalized


def append_context_dependency(ctx, artifact_id: str) -> None:
    if artifact_id and artifact_id not in ctx.input_artifact_ids:
        ctx.input_artifact_ids.append(artifact_id)


def mark_mutated_path(ctx, path: str) -> None:
    mutated_paths = ctx.metadata.setdefault("mutated_paths", [])
    normalized = normalize_artifact_path(path)
    if normalized not in mutated_paths:
        mutated_paths.append(normalized)


def register_filesystem_tools(
    registry: ToolRegistry,
    runtime: ExecutionRuntime,
    session: SessionStore | None = None,
) -> None:
    # Configure read-state normalization with the runtime's workspace so
    # `foo.py` and `./foo.py` resolve to the same key.
    workspace = getattr(runtime, "workspace", None)
    if isinstance(workspace, Path):
        _file_read_state.configure(workspace)

    async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        result = await runtime.read_file(path, start_line=start_line, end_line=end_line)
        if result.success:
            _file_read_state.record_read(path)
        if result.success and session is not None:
            ctx = current_agent_context()
            artifact_path = normalize_artifact_path(path)
            artifact = await session.snapshot_file_artifact(
                session_id=ctx.metadata["session_id"],
                logical_name=artifact_path,
                path=artifact_path,
                content=result.raw_output,
                produced_by=ctx.agent_id,
                stage_name=ctx.stage_name,
                artifact_type=infer_artifact_type(artifact_path),
                status=ArtifactStatus.OBSERVED,
            )
            append_context_dependency(ctx, artifact.id)
        return result

    async def write_file(path: str, content: str) -> ToolResult:
        result = await runtime.write_file(path, content)
        if result.success:
            # Mark as read — the agent knows the exact content (it just
            # provided it), so a subsequent Edit should work without a
            # redundant Read.  This avoids the "file_not_read" error when
            # the agent writes a file and immediately edits it.
            _file_read_state.record_read(path)
        ctx = peek_current_agent_context()
        if result.success and ctx is not None:
            mark_mutated_path(ctx, path)
        if result.success and session is not None:
            ctx = current_agent_context()
            artifact_path = normalize_artifact_path(path)
            artifact = await session.snapshot_file_artifact(
                session_id=ctx.metadata["session_id"],
                logical_name=artifact_path,
                path=artifact_path,
                content=content,
                produced_by=ctx.agent_id,
                stage_name=ctx.stage_name,
                artifact_type=infer_artifact_type(artifact_path),
                depends_on=list(ctx.input_artifact_ids),
            )
            append_context_dependency(ctx, artifact.id)
        # Update code index (best-effort).
        if result.success:
            from maike.tools.context import peek_current_code_index
            code_index = peek_current_code_index()
            if code_index is not None:
                try:
                    await code_index.update_file(path, content)
                except Exception:
                    pass
        return result

    async def list_dir(path: str = ".") -> ToolResult:
        return await runtime.list_dir(path)

    async def delete_file(path: str) -> ToolResult:
        result = await runtime.delete_file(path)
        ctx = peek_current_agent_context()
        if result.success and ctx is not None:
            mark_mutated_path(ctx, path)
        if result.success and session is not None:
            ctx = current_agent_context()
            await session.invalidate_artifact_by_name(
                ctx.metadata["session_id"],
                normalize_artifact_path(path),
                kind=ArtifactKind.FILE,
            )
        # Remove from code index (best-effort).
        if result.success:
            from maike.tools.context import peek_current_code_index
            code_index = peek_current_code_index()
            if code_index is not None:
                try:
                    await code_index.remove_file(path)
                except Exception:
                    pass
        return result

    registry.register(
        schema=ToolSchema(
            name="Read",
            description=(
                'Read a file from the workspace. Use start_line/end_line for targeted reads '
                'instead of reading entire files. Example: {"path": "src/main.py", "start_line": 10, "end_line": 30}'
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the workspace. Example: 'src/main.py'",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, inclusive). Omit to start from beginning.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, inclusive). Omit to read to end.",
                    },
                },
                "required": ["path"],
            },
        ),
        fn=read_file,
        risk_level=RiskLevel.READ,
    )
    registry.register(
        schema=ToolSchema(
            name="Write",
            description=(
                "Write UTF-8 content to a file in the workspace. Creates parent directories if needed. "
                'Example: {"path": "hello.py", "content": "print(\'hello\')"}'
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path for the file. Example: 'src/app.py'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.  Include all code — this replaces the entire file.",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        fn=write_file,
        risk_level=RiskLevel.WRITE,
    )
