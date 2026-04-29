"""SQLite-backed session and artifact store."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from maike.atoms.agent import AgentResult
from maike.atoms.artifact import Artifact, ArtifactKind, ArtifactStatus, ArtifactType
from maike.atoms.blueprint import SpawnRequest
from maike.atoms.context import Checkpoint
from maike.constants import DEFAULT_DB_RELATIVE_PATH
from maike.utils import dedupe_preserve_order, utcnow


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles bytes and other non-serializable types."""

    def default(self, o):
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        if isinstance(o, set):
            return list(o)
        return super().default(o)


def _safe_json_dumps(obj) -> str:
    """JSON-serialize with fallback for bytes and other non-standard types."""
    return json.dumps(obj, cls=_SafeEncoder)


def _estimate_message_tokens(messages: list[dict]) -> int:
    """Estimate token count for a list of messages.

    Uses the project's tokenizer when available, falls back to a
    conservative chars / 3.5 heuristic.
    """
    try:
        from maike.utils import estimate_message_tokens
        return estimate_message_tokens(messages)
    except Exception:
        total_chars = sum(len(json.dumps(m)) for m in messages)
        return int(total_chars / 3.5)


def generate_thread_name(task: str) -> str:
    """Auto-generate a short kebab-case thread name from a task description.

    >>> generate_thread_name("Create a Python CLI calculator with argparse")
    'create-python-cli-calculator-argparse'
    """
    _STOP_WORDS = frozenset({
        "a", "an", "the", "to", "for", "in", "with", "and", "or",
        "that", "this", "can", "you", "me", "my", "it", "is", "of",
        "on", "be", "do", "so", "how", "what", "please",
    })
    words = [
        w.lower().strip("?.!,;:'\"")
        for w in task.split()
        if w.lower().strip("?.!,;:'\"") not in _STOP_WORDS
        and w.lower().strip("?.!,;:'\"")
    ]
    return "-".join(words[:5]) or "default-thread"


class SessionStore:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.db_path = base_path / DEFAULT_DB_RELATIVE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._shared_db: aiosqlite.Connection | None = None
        self._shared_lock: asyncio.Lock | None = None
        self._shared_loop: asyncio.AbstractEventLoop | None = None

    async def initialize(self) -> None:
        async with self._db() as db:
            await self._initialize_schema_db(db)
            await db.commit()

    @asynccontextmanager
    async def use_shared_connection(self):
        loop = asyncio.get_running_loop()
        if self._shared_db is not None:
            if self._shared_loop is not loop:
                raise RuntimeError("SessionStore shared connection cannot span event loops.")
            yield self
            return

        db = await self._connect()
        await self._initialize_schema_db(db)
        await db.commit()
        self._shared_db = db
        self._shared_lock = asyncio.Lock()
        self._shared_loop = loop
        try:
            yield self
        finally:
            await db.close()
            self._shared_db = None
            self._shared_lock = None
            self._shared_loop = None

    async def create_session(
        self,
        task: str,
        workspace: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id = str(uuid4())
        now = utcnow().isoformat()
        serialized_metadata = json.dumps(metadata or {})
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO sessions (id, task, workspace, status, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, task, str(workspace), "running", serialized_metadata, now, now),
            )
            await db.commit()
        return session_id

    async def mark_session_status(self, session_id: str, status: str) -> None:
        async with self._db() as db:
            await db.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, utcnow().isoformat(), session_id),
            )
            await db.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self._db() as db:
            cursor = await db.execute(
                """
                SELECT id, task, workspace, status, metadata, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "task": row[1],
            "workspace": row[2],
            "status": row[3],
            "metadata": self._load_metadata(row[4]),
            "created_at": row[5],
            "updated_at": row[6],
        }

    async def update_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        async with self._db() as db:
            await db.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata), utcnow().isoformat(), session_id),
            )
            await db.commit()

    async def store_artifact(self, session_id: str, artifact: Artifact) -> Artifact:
        artifact.session_id = session_id
        async with self._db() as db:
            artifact = await self._store_artifact_transactionally(
                db=db,
                session_id=session_id,
                artifact=artifact,
                active_lookup=lambda: self._get_artifact_by_name_db(
                    db,
                    session_id=session_id,
                    logical_name=artifact.logical_name,
                    active_only=True,
                    kind=artifact.kind,
                ),
                latest_lookup=lambda: self._get_artifact_by_name_db(
                    db,
                    session_id=session_id,
                    logical_name=artifact.logical_name,
                    active_only=False,
                    kind=artifact.kind,
                ),
            )
        return artifact

    async def snapshot_file_artifact(
        self,
        session_id: str,
        *,
        logical_name: str,
        path: str,
        content: str,
        produced_by: str,
        stage_name: str,
        artifact_type: ArtifactType,
        depends_on: list[str] | None = None,
        status: ArtifactStatus = ArtifactStatus.DRAFT,
    ) -> Artifact:
        artifact = Artifact(
            kind=ArtifactKind.FILE,
            type=artifact_type,
            logical_name=logical_name,
            path=path,
            content=content,
            produced_by=produced_by,
            stage_name=stage_name,
            depends_on=dedupe_preserve_order(depends_on or []),
            status=status,
        )
        artifact.session_id = session_id
        async with self._db() as db:
            artifact = await self._store_file_artifact_transactionally(
                db=db,
                session_id=session_id,
                artifact=artifact,
                path=path,
                logical_name=logical_name,
            )
        return artifact

    async def list_artifacts(
        self,
        session_id: str,
        stage_name: str | None = None,
        active_only: bool = False,
        kind: ArtifactKind | None = None,
        include_observed: bool = False,
    ) -> list[Artifact]:
        async with self._db() as db:
            rows = await self._list_artifact_rows_db(
                db,
                session_id=session_id,
                stage_name=stage_name,
                active_only=active_only,
                kind=kind,
                include_observed=include_observed,
            )
        return [self._artifact_from_row(row) for row in rows]

    async def get_artifact_by_name(
        self,
        session_id: str,
        logical_name: str,
        active_only: bool = True,
        kind: ArtifactKind | None = None,
    ) -> Artifact | None:
        async with self._db() as db:
            return await self._get_artifact_by_name_db(
                db,
                session_id=session_id,
                logical_name=logical_name,
                active_only=active_only,
                kind=kind,
            )

    async def get_file_artifact(
        self,
        session_id: str,
        path: str,
        active_only: bool = True,
    ) -> Artifact | None:
        async with self._db() as db:
            return await self._get_file_artifact_db(
                db,
                session_id=session_id,
                path=path,
                active_only=active_only,
            )

    async def invalidate_artifact_by_name(
        self,
        session_id: str,
        logical_name: str,
        *,
        kind: ArtifactKind | None = None,
    ) -> Artifact | None:
        artifact = await self.get_artifact_by_name(
            session_id,
            logical_name,
            active_only=True,
            kind=kind,
        )
        if artifact is None:
            return None
        await self.invalidate_artifact_tree(session_id, artifact.id)
        return artifact

    async def invalidate_artifact_tree(self, session_id: str, artifact_id: str) -> None:
        async with self._db() as db:
            await self._invalidate_artifact_tree_db(db, session_id, artifact_id)
            await db.commit()

    async def artifacts_valid(
        self,
        session_id: str,
        stage_name: str,
        required_outputs: list[str] | None = None,
    ) -> bool:
        async with self._db() as db:
            if required_outputs:
                required_names = dedupe_preserve_order(required_outputs)
                placeholders = ", ".join("?" for _ in required_names)
                cursor = await db.execute(
                    f"""
                    SELECT logical_name FROM artifacts
                    WHERE session_id = ? AND stage_name = ? AND kind = ? AND invalidated = 0
                      AND logical_name IN ({placeholders})
                    GROUP BY logical_name
                    """,
                    [session_id, stage_name, ArtifactKind.STAGE.value, *required_names],
                )
                rows = await cursor.fetchall()
                return {row[0] for row in rows} == set(required_names)
            cursor = await db.execute(
                """
                SELECT 1 FROM artifacts
                WHERE session_id = ? AND stage_name = ? AND kind = ? AND invalidated = 0
                LIMIT 1
                """,
                (session_id, stage_name, ArtifactKind.STAGE.value),
            )
            return await cursor.fetchone() is not None

    async def store_checkpoint(self, session_id: str, checkpoint: Checkpoint) -> None:
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO checkpoints (id, session_id, sha, label, step, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    session_id,
                    checkpoint.sha,
                    checkpoint.label,
                    checkpoint.step,
                    checkpoint.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_latest_checkpoint(
        self,
        session_id: str,
        stage_name: str | None = None,
    ) -> Checkpoint | None:
        async with self._db() as db:
            if stage_name is None:
                cursor = await db.execute(
                    """
                    SELECT id, sha, label, step, created_at
                    FROM checkpoints
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (session_id,),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT id, sha, label, step, created_at
                    FROM checkpoints
                    WHERE session_id = ? AND step = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (session_id, stage_name),
                )
            row = await cursor.fetchone()
        if row is None:
            return None
        return Checkpoint(
            id=row[0],
            sha=row[1],
            label=row[2],
            step=row[3],
            created_at=datetime.fromisoformat(row[4]),
        )

    async def store_agent_run(self, session_id: str, result: AgentResult) -> None:
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO agent_runs (
                    id, session_id, stage_name, agent_id, role, success, output,
                    cost_usd, tokens_used, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    session_id,
                    result.stage_name,
                    result.agent_id,
                    result.role,
                    int(result.success),
                    result.output,
                    result.cost_usd,
                    result.tokens_used,
                    json.dumps(result.metadata),
                    utcnow().isoformat(),
                ),
            )
            await db.execute(
                """
                INSERT INTO step_results (
                    id, session_id, stage_name, agent_id, success, cost_usd,
                    tokens_used, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    session_id,
                    result.stage_name,
                    result.agent_id,
                    int(result.success),
                    result.cost_usd,
                    result.tokens_used,
                    utcnow().isoformat(),
                ),
            )
            await db.commit()

    async def get_agent_runs(self, session_id: str) -> list[dict[str, Any]]:
        async with self._db() as db:
            cursor = await db.execute(
                """
                SELECT stage_name, agent_id, role, success, output, cost_usd, tokens_used, metadata, created_at
                FROM agent_runs
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "stage_name": row[0],
                "agent_id": row[1],
                "role": row[2],
                "success": bool(row[3]),
                "output": row[4],
                "cost_usd": row[5],
                "tokens_used": row[6],
                "metadata": json.loads(row[7]),
                "created_at": row[8],
            }
            for row in rows
        ]

    async def queue_spawn_request(self, session_id: str, request: SpawnRequest) -> None:
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO spawn_requests (
                    id, session_id, requesting_agent_id, stage_name, tool_profile, reason,
                    suggested_role, context_summary, remaining_token_budget,
                    remaining_cost_budget_usd, urgency, processed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    request.request_id,
                    session_id,
                    request.requesting_agent_id,
                    request.stage_name,
                    request.tool_profile,
                    request.reason,
                    request.suggested_role,
                    request.context_summary,
                    request.remaining_token_budget,
                    request.remaining_cost_budget_usd,
                    request.urgency,
                    utcnow().isoformat(),
                ),
            )
            await db.commit()

    async def list_spawn_requests(
        self,
        session_id: str,
        *,
        processed: bool | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if processed is not None:
            clauses.append("processed = ?")
            params.append(int(processed))
        query = (
            "SELECT id, requesting_agent_id, stage_name, tool_profile, reason, suggested_role, "
            "context_summary, remaining_token_budget, remaining_cost_budget_usd, urgency, processed, created_at "
            f"FROM spawn_requests WHERE {' AND '.join(clauses)} ORDER BY created_at ASC"
        )
        async with self._db() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        return [
            {
                "request_id": row[0],
                "requesting_agent_id": row[1],
                "stage_name": row[2],
                "tool_profile": row[3],
                "reason": row[4],
                "suggested_role": row[5],
                "context_summary": row[6],
                "remaining_token_budget": row[7],
                "remaining_cost_budget_usd": row[8],
                "urgency": row[9],
                "processed": bool(row[10]),
                "created_at": row[11],
            }
            for row in rows
        ]

    async def pop_spawn_requests(self, session_id: str) -> list[SpawnRequest]:
        async with self._db() as db:
            cursor = await db.execute(
                """
                SELECT id, requesting_agent_id, stage_name, tool_profile, reason, suggested_role,
                       context_summary, remaining_token_budget, remaining_cost_budget_usd, urgency
                FROM spawn_requests
                WHERE session_id = ? AND processed = 0
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
            ids = [row[0] for row in rows]
            if ids:
                placeholders = ", ".join("?" for _ in ids)
                await db.execute(
                    f"UPDATE spawn_requests SET processed = 1 WHERE id IN ({placeholders})",
                    ids,
                )
                await db.commit()
        return [
            SpawnRequest(
                request_id=row[0],
                requesting_agent_id=row[1],
                stage_name=row[2],
                tool_profile=row[3],
                reason=row[4],
                suggested_role=row[5],
                context_summary=row[6],
                remaining_token_budget=row[7],
                remaining_cost_budget_usd=row[8],
                urgency=row[9],
            )
            for row in rows
        ]

    async def get_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        return await self.list_sessions(limit)

    async def list_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        async with self._db() as db:
            cursor = await db.execute(
                """
                SELECT id, task, workspace, status, created_at, updated_at, metadata
                FROM sessions ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            meta: dict[str, Any] = {}
            raw_meta = row[6]
            if raw_meta:
                try:
                    meta = json.loads(raw_meta)
                    if not isinstance(meta, dict):
                        meta = {}
                except Exception:
                    meta = {}
            out.append({
                "id": row[0],
                "task": row[1],
                "workspace": row[2],
                "status": row[3],
                "created_at": row[4],
                "updated_at": row[5],
                "metadata": meta,
            })
        return out

    async def get_session_cost(self, session_id: str) -> dict[str, Any] | None:
        session = await self.get_session(session_id)
        if session is None:
            return None

        agent_runs = await self.get_agent_runs(session_id)
        per_stage: dict[str, dict[str, Any]] = {}
        total_cost = 0.0
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        llm_calls = 0
        has_token_breakdown = False

        for run in agent_runs:
            metadata = run.get("metadata") or {}
            usage = metadata.get("llm_usage") or {}
            run_input_tokens = int(usage.get("input_tokens", 0) or 0)
            run_output_tokens = int(usage.get("output_tokens", 0) or 0)
            run_llm_calls = int(usage.get("calls", 0) or 0)

            total_cost += float(run.get("cost_usd", 0.0) or 0.0)
            total_tokens += int(run.get("tokens_used", 0) or 0)
            input_tokens += run_input_tokens
            output_tokens += run_output_tokens
            llm_calls += run_llm_calls
            has_token_breakdown = has_token_breakdown or bool(run_input_tokens or run_output_tokens or run_llm_calls)

            stage_name = run["stage_name"]
            stage = per_stage.setdefault(
                stage_name,
                {
                    "stage_name": stage_name,
                    "cost_usd": 0.0,
                    "tokens_used": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "llm_calls": 0,
                    "agent_runs": 0,
                    "failed_runs": 0,
                    "has_token_breakdown": False,
                },
            )
            stage["cost_usd"] += float(run.get("cost_usd", 0.0) or 0.0)
            stage["tokens_used"] += int(run.get("tokens_used", 0) or 0)
            stage["input_tokens"] += run_input_tokens
            stage["output_tokens"] += run_output_tokens
            stage["llm_calls"] += run_llm_calls
            stage["agent_runs"] += 1
            stage["failed_runs"] += int(not run.get("success", False))
            stage["has_token_breakdown"] = stage["has_token_breakdown"] or bool(
                run_input_tokens or run_output_tokens or run_llm_calls
            )

        return {
            **session,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "llm_calls": llm_calls,
            "agent_runs": len(agent_runs),
            "has_token_breakdown": has_token_breakdown,
            "per_stage": list(per_stage.values()),
        }

    async def _connect(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA foreign_keys=ON")
        return db

    @asynccontextmanager
    async def _db(self):
        shared = self._current_shared_connection()
        if shared is not None:
            db, lock = shared
            async with lock:
                yield db
            return

        db = await self._connect()
        try:
            yield db
        finally:
            await db.close()

    def _current_shared_connection(self) -> tuple[aiosqlite.Connection, asyncio.Lock] | None:
        if self._shared_db is None or self._shared_lock is None:
            return None
        loop = asyncio.get_running_loop()
        if self._shared_loop is not loop:
            return None
        return self._shared_db, self._shared_lock

    async def _initialize_schema_db(self, db: aiosqlite.Connection) -> None:
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                workspace TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                type TEXT NOT NULL,
                logical_name TEXT NOT NULL,
                path TEXT,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                produced_by TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                invalidated INTEGER NOT NULL,
                depends_on TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifact_dependencies (
                artifact_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                depends_on_id TEXT NOT NULL,
                PRIMARY KEY (artifact_id, depends_on_id)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sha TEXT NOT NULL,
                label TEXT NOT NULL,
                step TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                role TEXT NOT NULL,
                success INTEGER NOT NULL,
                output TEXT,
                cost_usd REAL NOT NULL,
                tokens_used INTEGER NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS step_results (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                tokens_used INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS spawn_requests (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                requesting_agent_id TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                tool_profile TEXT NOT NULL,
                reason TEXT NOT NULL,
                suggested_role TEXT NOT NULL,
                context_summary TEXT NOT NULL,
                remaining_token_budget INTEGER NOT NULL,
                remaining_cost_budget_usd REAL NOT NULL,
                urgency TEXT NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                workspace TEXT NOT NULL,
                name TEXT NOT NULL,
                conversation_history TEXT NOT NULL DEFAULT '[]',
                history_token_estimate INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_threads_workspace_status
                ON threads(workspace, status, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_artifacts_session_stage_active
                ON artifacts(session_id, stage_name, kind, invalidated, logical_name);
            CREATE INDEX IF NOT EXISTS idx_artifacts_session_name_active
                ON artifacts(session_id, logical_name, kind, invalidated, version DESC);
            CREATE INDEX IF NOT EXISTS idx_artifacts_session_path_active
                ON artifacts(session_id, path, kind, invalidated, version DESC);
            CREATE INDEX IF NOT EXISTS idx_artifact_dependencies_session_parent
                ON artifact_dependencies(session_id, depends_on_id);
            CREATE INDEX IF NOT EXISTS idx_artifact_dependencies_session_child
                ON artifact_dependencies(session_id, artifact_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_session_name_version_unique
                ON artifacts(session_id, kind, logical_name, version);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_session_path_version_unique
                ON artifacts(session_id, kind, path, version)
                WHERE path IS NOT NULL;
            """
        )
        await self._ensure_sessions_metadata_column(db)
        await self._ensure_sessions_thread_id_column(db)
        await self._ensure_threads_summaries_column(db)

    async def _ensure_sessions_metadata_column(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(sessions)")
        rows = await cursor.fetchall()
        column_names = {str(row[1]) for row in rows}
        if "metadata" not in column_names:
            await db.execute("ALTER TABLE sessions ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
        await db.execute("UPDATE sessions SET metadata = '{}' WHERE metadata IS NULL OR TRIM(metadata) = ''")

    async def _ensure_sessions_thread_id_column(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(sessions)")
        rows = await cursor.fetchall()
        column_names = {str(row[1]) for row in rows}
        if "thread_id" not in column_names:
            await db.execute("ALTER TABLE sessions ADD COLUMN thread_id TEXT")

    async def _ensure_threads_summaries_column(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(threads)")
        rows = await cursor.fetchall()
        column_names = {str(row[1]) for row in rows}
        if "thread_summaries" not in column_names:
            await db.execute("ALTER TABLE threads ADD COLUMN thread_summaries TEXT NOT NULL DEFAULT '[]'")

    # ------------------------------------------------------------------
    # Thread CRUD
    # ------------------------------------------------------------------

    async def create_thread(self, workspace: Path, name: str) -> str:
        thread_id = str(uuid4())
        now = utcnow().isoformat()
        async with self._db() as db:
            await db.execute(
                """
                INSERT INTO threads (id, workspace, name, conversation_history, history_token_estimate, status, created_at, updated_at)
                VALUES (?, ?, ?, '[]', 0, 'active', ?, ?)
                """,
                (thread_id, str(workspace), name, now, now),
            )
            await db.commit()
        return thread_id

    async def get_active_thread(self, workspace: Path) -> dict[str, Any] | None:
        async with self._db() as db:
            cursor = await db.execute(
                """
                SELECT id, workspace, name, conversation_history, history_token_estimate, status, created_at, updated_at, thread_summaries
                FROM threads
                WHERE workspace = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (str(workspace),),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._thread_from_row(row)

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        async with self._db() as db:
            cursor = await db.execute(
                """
                SELECT id, workspace, name, conversation_history, history_token_estimate, status, created_at, updated_at, thread_summaries
                FROM threads
                WHERE id = ?
                """,
                (thread_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._thread_from_row(row)

    async def append_thread_history(
        self, thread_id: str, messages: list[dict[str, Any]]
    ) -> None:
        thread = await self.get_thread(thread_id)
        if thread is None:
            return
        existing = thread.get("conversation_history", [])
        if not isinstance(existing, list):
            existing = []
        combined = existing + messages
        token_estimate = _estimate_message_tokens(combined)
        async with self._db() as db:
            await db.execute(
                """
                UPDATE threads
                SET conversation_history = ?, history_token_estimate = ?, updated_at = ?
                WHERE id = ?
                """,
                (_safe_json_dumps(combined), token_estimate, utcnow().isoformat(), thread_id),
            )
            await db.commit()

    async def append_thread_summary(self, thread_id: str, summary: str) -> None:
        """Append a session summary to the thread's summaries list."""
        thread = await self.get_thread(thread_id)
        if thread is None:
            return
        existing = thread.get("summaries", [])
        if not isinstance(existing, list):
            existing = []
        combined = existing + [summary]
        async with self._db() as db:
            await db.execute(
                """
                UPDATE threads
                SET thread_summaries = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(combined, ensure_ascii=False), utcnow().isoformat(), thread_id),
            )
            await db.commit()

    async def replace_thread_history(
        self, thread_id: str, messages: list[dict[str, Any]], token_estimate: int
    ) -> None:
        async with self._db() as db:
            await db.execute(
                """
                UPDATE threads
                SET conversation_history = ?, history_token_estimate = ?, updated_at = ?
                WHERE id = ?
                """,
                (_safe_json_dumps(messages), token_estimate, utcnow().isoformat(), thread_id),
            )
            await db.commit()

    async def list_threads(self, workspace: Path) -> list[dict[str, Any]]:
        async with self._db() as db:
            cursor = await db.execute(
                """
                SELECT id, workspace, name, conversation_history, history_token_estimate, status, created_at, updated_at, thread_summaries
                FROM threads
                WHERE workspace = ?
                ORDER BY updated_at DESC
                """,
                (str(workspace),),
            )
            rows = await cursor.fetchall()
        return [self._thread_from_row(row) for row in rows]

    @staticmethod
    def _thread_from_row(row: Any) -> dict[str, Any]:
        history_raw = row[3] or "[]"
        try:
            history = json.loads(history_raw) if isinstance(history_raw, str) else history_raw
        except (json.JSONDecodeError, TypeError):
            history = []
        # Parse thread_summaries (column 8, may not exist in old schemas)
        summaries: list[str] = []
        if len(row) > 8:
            summaries_raw = row[8] or "[]"
            try:
                summaries = json.loads(summaries_raw) if isinstance(summaries_raw, str) else summaries_raw
            except (json.JSONDecodeError, TypeError):
                summaries = []
        return {
            "id": row[0],
            "workspace": row[1],
            "name": row[2],
            "conversation_history": history,
            "history_token_estimate": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "summaries": summaries if isinstance(summaries, list) else [],
        }

    def _load_metadata(self, raw_value: Any) -> dict[str, Any]:
        if raw_value in (None, ""):
            return {}
        if isinstance(raw_value, dict):
            return raw_value
        try:
            parsed = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _store_artifact_transactionally(
        self,
        *,
        db: aiosqlite.Connection,
        session_id: str,
        artifact: Artifact,
        active_lookup,
        latest_lookup,
    ) -> Artifact:
        artifact.depends_on = dedupe_preserve_order(artifact.depends_on)
        for attempt in range(2):
            try:
                await db.execute("BEGIN IMMEDIATE")
                latest = await latest_lookup()
                active = await active_lookup()
                artifact.version = 1 if latest is None else latest.version + 1
                if active:
                    await self._invalidate_artifact_tree_db(db, session_id, active.id)
                await self._insert_artifact_db(db, session_id, artifact)
                await db.commit()
                return artifact
            except aiosqlite.IntegrityError:
                await db.rollback()
                if attempt == 1:
                    raise
            except Exception:
                await db.rollback()
                raise
        raise RuntimeError("Failed to store artifact transactionally")

    async def _store_file_artifact_transactionally(
        self,
        *,
        db: aiosqlite.Connection,
        session_id: str,
        artifact: Artifact,
        path: str,
        logical_name: str,
    ) -> Artifact:
        artifact.depends_on = dedupe_preserve_order(artifact.depends_on)
        for attempt in range(2):
            try:
                await db.execute("BEGIN IMMEDIATE")
                latest = await self._get_file_artifact_db(db, session_id=session_id, path=path, active_only=False)
                active = await self._get_file_artifact_db(db, session_id=session_id, path=path, active_only=True)
                if (
                    active
                    and active.logical_name == logical_name
                    and active.path == path
                    and active.content_hash == artifact.content_hash
                    and (
                        active.status is not ArtifactStatus.OBSERVED
                        or artifact.status is ArtifactStatus.OBSERVED
                    )
                ):
                    await db.commit()
                    return active
                artifact.version = 1 if latest is None else latest.version + 1
                if active:
                    await self._invalidate_artifact_tree_db(db, session_id, active.id)
                await self._insert_artifact_db(db, session_id, artifact)
                await db.commit()
                return artifact
            except aiosqlite.IntegrityError:
                await db.rollback()
                if attempt == 1:
                    raise
            except Exception:
                await db.rollback()
                raise
        raise RuntimeError("Failed to store file artifact transactionally")

    async def _insert_artifact_db(self, db: aiosqlite.Connection, session_id: str, artifact: Artifact) -> None:
        dependencies = dedupe_preserve_order(artifact.depends_on)
        await db.execute(
            """
            INSERT INTO artifacts (
                id, session_id, kind, type, logical_name, path, content, content_hash,
                produced_by, stage_name, version, status, invalidated, depends_on, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                session_id,
                artifact.kind.value,
                artifact.type.value,
                artifact.logical_name,
                artifact.path,
                artifact.content,
                artifact.content_hash,
                artifact.produced_by,
                artifact.stage_name,
                artifact.version,
                artifact.status.value,
                int(artifact.invalidated),
                json.dumps(dependencies),
                artifact.created_at.isoformat(),
            ),
        )
        if dependencies:
            await db.executemany(
                """
                INSERT OR IGNORE INTO artifact_dependencies (artifact_id, session_id, depends_on_id)
                VALUES (?, ?, ?)
                """,
                [(artifact.id, session_id, dependency_id) for dependency_id in dependencies],
            )

    async def _list_artifact_rows_db(
        self,
        db: aiosqlite.Connection,
        *,
        session_id: str,
        stage_name: str | None = None,
        active_only: bool = False,
        kind: ArtifactKind | None = None,
        logical_name: str | None = None,
        path: str | None = None,
        limit: int | None = None,
        include_observed: bool = True,
    ) -> list[Any]:
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if stage_name:
            clauses.append("stage_name = ?")
            params.append(stage_name)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        if logical_name is not None:
            clauses.append("logical_name = ?")
            params.append(logical_name)
        if path is not None:
            clauses.append("path = ?")
            params.append(path)
        if active_only:
            clauses.append("invalidated = 0")
        if not include_observed:
            clauses.append("status != ?")
            params.append(ArtifactStatus.OBSERVED.value)
        query = (
            "SELECT id, session_id, kind, type, logical_name, path, content, content_hash, "
            "produced_by, stage_name, version, status, invalidated, depends_on, created_at "
            f"FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY version DESC, created_at DESC"
        )
        if limit is not None:
            query = f"{query} LIMIT ?"
            params.append(limit)
        cursor = await db.execute(query, params)
        return await cursor.fetchall()

    async def _get_artifact_by_name_db(
        self,
        db: aiosqlite.Connection,
        *,
        session_id: str,
        logical_name: str,
        active_only: bool = True,
        kind: ArtifactKind | None = None,
    ) -> Artifact | None:
        rows = await self._list_artifact_rows_db(
            db,
            session_id=session_id,
            logical_name=logical_name,
            active_only=active_only,
            kind=kind,
            limit=1,
        )
        return self._artifact_from_row(rows[0]) if rows else None

    async def _get_file_artifact_db(
        self,
        db: aiosqlite.Connection,
        *,
        session_id: str,
        path: str,
        active_only: bool = True,
    ) -> Artifact | None:
        rows = await self._list_artifact_rows_db(
            db,
            session_id=session_id,
            path=path,
            active_only=active_only,
            kind=ArtifactKind.FILE,
            limit=1,
        )
        return self._artifact_from_row(rows[0]) if rows else None

    async def _invalidate_artifact_tree_db(
        self,
        db: aiosqlite.Connection,
        session_id: str,
        artifact_id: str,
    ) -> None:
        to_visit = [artifact_id]
        seen: set[str] = set()
        while to_visit:
            current = to_visit.pop()
            if current in seen:
                continue
            seen.add(current)
            await db.execute(
                """
                UPDATE artifacts
                SET invalidated = 1, status = ?
                WHERE id = ? AND invalidated = 0
                """,
                (ArtifactStatus.INVALIDATED.value, current),
            )
            cursor = await db.execute(
                """
                SELECT artifact_id FROM artifact_dependencies
                WHERE session_id = ? AND depends_on_id = ?
                """,
                (session_id, current),
            )
            rows = await cursor.fetchall()
            to_visit.extend(row[0] for row in rows)

    def _artifact_from_row(self, row: Any) -> Artifact:
        return Artifact(
            id=row[0],
            session_id=row[1],
            kind=ArtifactKind(row[2]),
            type=ArtifactType(row[3]),
            logical_name=row[4],
            path=row[5],
            content=row[6],
            content_hash=row[7],
            produced_by=row[8],
            stage_name=row[9],
            version=row[10],
            status=ArtifactStatus(row[11]),
            invalidated=bool(row[12]),
            depends_on=json.loads(row[13]),
            created_at=datetime.fromisoformat(row[14]),
        )
