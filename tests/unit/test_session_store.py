import asyncio
import sqlite3

import pytest

from maike.atoms.agent import AgentResult
from maike.atoms.artifact import Artifact, ArtifactKind, ArtifactStatus, ArtifactType
from maike.memory.session import SessionStore


def test_session_store_reuses_matching_file_snapshots(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)

        first = await store.snapshot_file_artifact(
            session_id,
            logical_name="app.py",
            path="app.py",
            content="print('hello')\n",
            produced_by="agent-a",
            stage_name="coding",
            artifact_type=ArtifactType.CODE,
            depends_on=["dep-1", "dep-1"],
        )
        second = await store.snapshot_file_artifact(
            session_id,
            logical_name="app.py",
            path="app.py",
            content="print('hello')\n",
            produced_by="agent-b",
            stage_name="review",
            artifact_type=ArtifactType.CODE,
            depends_on=["dep-2"],
        )

        all_files = await store.list_artifacts(session_id, active_only=False, kind=ArtifactKind.FILE)

        assert second.id == first.id
        assert second.version == 1
        assert len(all_files) == 1
        assert all_files[0].depends_on == ["dep-1"]

    asyncio.run(scenario())


def test_session_store_versions_changed_files_and_invalidates_dependents(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)

        spec = await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.SPEC,
                logical_name="spec.md",
                content="spec v1",
                produced_by="agent-a",
                stage_name="requirements",
            ),
        )
        file_v1 = await store.snapshot_file_artifact(
            session_id,
            logical_name="app.py",
            path="app.py",
            content="print('v1')\n",
            produced_by="agent-b",
            stage_name="coding",
            artifact_type=ArtifactType.CODE,
            depends_on=[spec.id],
        )
        summary = await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.CODE,
                logical_name="code-summary.md",
                content="summary",
                produced_by="agent-b",
                stage_name="coding",
                depends_on=[spec.id, file_v1.id],
            ),
        )

        file_v2 = await store.snapshot_file_artifact(
            session_id,
            logical_name="app.py",
            path="app.py",
            content="print('v2')\n",
            produced_by="agent-b",
            stage_name="coding",
            artifact_type=ArtifactType.CODE,
            depends_on=[spec.id],
        )

        all_artifacts = await store.list_artifacts(session_id, active_only=False)
        old_file = next(artifact for artifact in all_artifacts if artifact.id == file_v1.id)
        old_summary = next(artifact for artifact in all_artifacts if artifact.id == summary.id)

        assert file_v2.version == 2
        assert (await store.get_file_artifact(session_id, "app.py")).id == file_v2.id
        assert await store.get_artifact_by_name(session_id, "code-summary.md", kind=ArtifactKind.STAGE) is None
        assert old_file.invalidated is True
        assert old_summary.invalidated is True

    asyncio.run(scenario())


def test_session_store_delete_invalidation_cascades_from_active_file(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)

        file_artifact = await store.snapshot_file_artifact(
            session_id,
            logical_name="app.py",
            path="app.py",
            content="print('hello')\n",
            produced_by="agent-a",
            stage_name="coding",
            artifact_type=ArtifactType.CODE,
        )
        review = await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.REVIEW,
                logical_name="review.md",
                content="review",
                produced_by="agent-b",
                stage_name="review",
                depends_on=[file_artifact.id],
            ),
        )

        invalidated = await store.invalidate_artifact_by_name(
            session_id,
            "app.py",
            kind=ArtifactKind.FILE,
        )
        all_artifacts = await store.list_artifacts(session_id, active_only=False)
        old_review = next(artifact for artifact in all_artifacts if artifact.id == review.id)

        assert invalidated is not None
        assert invalidated.id == file_artifact.id
        assert await store.get_file_artifact(session_id, "app.py") is None
        assert await store.get_artifact_by_name(session_id, "review.md", kind=ArtifactKind.STAGE) is None
        assert old_review.invalidated is True

    asyncio.run(scenario())


def test_session_store_requires_declared_stage_outputs_and_scopes_active_lookups(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)

        file_plan = await store.snapshot_file_artifact(
            session_id,
            logical_name="plan.md",
            path="plan.md",
            content="# generated plan file\n",
            produced_by="agent-a",
            stage_name="coding",
            artifact_type=ArtifactType.DOCS,
        )
        await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.PLAN,
                logical_name="notes.md",
                content="notes",
                produced_by="agent-b",
                stage_name="planning",
            ),
        )

        assert await store.artifacts_valid(session_id, "planning", ["plan.md"]) is False

        stage_plan = await store.store_artifact(
            session_id,
            Artifact(
                type=ArtifactType.PLAN,
                logical_name="plan.md",
                content="actual plan",
                produced_by="agent-b",
                stage_name="planning",
            ),
        )

        assert await store.artifacts_valid(session_id, "planning", ["plan.md"]) is True
        assert (await store.get_artifact_by_name(session_id, "plan.md", kind=ArtifactKind.STAGE)).id == stage_plan.id
        assert (await store.get_file_artifact(session_id, "plan.md")).id == file_plan.id

    asyncio.run(scenario())


def test_session_store_hides_observed_file_reads_from_default_artifact_list(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)

        observed = await store.snapshot_file_artifact(
            session_id,
            logical_name="README.md",
            path="README.md",
            content="# read only\n",
            produced_by="agent-a",
            stage_name="review",
            artifact_type=ArtifactType.DOCS,
            status=ArtifactStatus.OBSERVED,
        )

        default_files = await store.list_artifacts(session_id, active_only=True, kind=ArtifactKind.FILE)
        all_files = await store.list_artifacts(
            session_id,
            active_only=True,
            kind=ArtifactKind.FILE,
            include_observed=True,
        )

        assert default_files == []
        assert [artifact.id for artifact in all_files] == [observed.id]

    asyncio.run(scenario())


def test_session_store_promotes_observed_read_snapshot_when_file_is_written(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)

        observed = await store.snapshot_file_artifact(
            session_id,
            logical_name="app.py",
            path="app.py",
            content="print('same')\n",
            produced_by="reader",
            stage_name="analysis",
            artifact_type=ArtifactType.CODE,
            status=ArtifactStatus.OBSERVED,
        )
        materialized = await store.snapshot_file_artifact(
            session_id,
            logical_name="app.py",
            path="app.py",
            content="print('same')\n",
            produced_by="writer",
            stage_name="coding",
            artifact_type=ArtifactType.CODE,
            depends_on=["spec-1"],
            status=ArtifactStatus.DRAFT,
        )

        all_files = await store.list_artifacts(
            session_id,
            active_only=False,
            kind=ArtifactKind.FILE,
            include_observed=True,
        )
        old_observed = next(artifact for artifact in all_files if artifact.id == observed.id)

        assert materialized.id != observed.id
        assert materialized.version == 2
        assert materialized.status is ArtifactStatus.DRAFT
        assert materialized.depends_on == ["spec-1"]
        assert old_observed.invalidated is True

    asyncio.run(scenario())


def test_session_store_shared_connection_enables_wal_and_busy_timeout(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        async with store.use_shared_connection():
            assert store._shared_db is not None
            journal_cursor = await store._shared_db.execute("PRAGMA journal_mode")
            timeout_cursor = await store._shared_db.execute("PRAGMA busy_timeout")
            journal_mode = (await journal_cursor.fetchone())[0]
            busy_timeout = (await timeout_cursor.fetchone())[0]
            await store.create_session("task", tmp_path)
        return store, journal_mode, busy_timeout

    store, journal_mode, busy_timeout = asyncio.run(scenario())

    assert store._shared_db is None
    assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 5000


def test_session_store_serializes_concurrent_artifact_versions_across_connections(tmp_path):
    async def scenario():
        primary = SessionStore(tmp_path)
        secondary = SessionStore(tmp_path)
        await primary.initialize()
        await secondary.initialize()
        session_id = await primary.create_session("task", tmp_path)

        async def write(store: SessionStore, produced_by: str, content: str):
            return await store.store_artifact(
                session_id,
                Artifact(
                    type=ArtifactType.CODE,
                    logical_name="code-summary.md",
                    content=content,
                    produced_by=produced_by,
                    stage_name="coding",
                ),
            )

        first, second = await asyncio.gather(
            write(primary, "agent-a", "summary v1"),
            write(secondary, "agent-b", "summary v2"),
        )
        artifacts = await primary.list_artifacts(session_id, active_only=False, kind=ArtifactKind.STAGE)
        versions = sorted(
            artifact.version
            for artifact in artifacts
            if artifact.logical_name == "code-summary.md"
        )
        return first, second, versions

    first, second, versions = asyncio.run(scenario())

    assert first.version != second.version
    assert versions == [1, 2]


def test_session_store_lists_sessions_and_aggregates_cost_breakdown(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()

        older = await store.create_session("older task", tmp_path)
        await store.mark_session_status(older, "completed")
        newer = await store.create_session("newer task", tmp_path)
        await store.mark_session_status(newer, "failed")

        await store.store_agent_run(
            newer,
            AgentResult(
                agent_id="agent-1",
                role="coder",
                stage_name="coding",
                cost_usd=0.12,
                tokens_used=180,
                metadata={
                    "llm_usage": {
                        "calls": 2,
                        "input_tokens": 100,
                        "output_tokens": 80,
                        "total_tokens": 180,
                        "cost_usd": 0.12,
                    }
                },
            ),
        )
        await store.store_agent_run(
            newer,
            AgentResult(
                agent_id="agent-2",
                role="tester",
                stage_name="testing",
                success=False,
                cost_usd=0.08,
                tokens_used=120,
                metadata={
                    "llm_usage": {
                        "calls": 1,
                        "input_tokens": 70,
                        "output_tokens": 50,
                        "total_tokens": 120,
                        "cost_usd": 0.08,
                    }
                },
            ),
        )

        sessions = await store.list_sessions(limit=5)
        summary = await store.get_session_cost(newer)
        return sessions, summary, older, newer

    sessions, summary, older, newer = asyncio.run(scenario())

    assert [session["id"] for session in sessions] == [newer, older]
    assert summary is not None
    assert summary["id"] == newer
    assert summary["status"] == "failed"
    assert summary["total_cost_usd"] == pytest.approx(0.2)
    assert summary["total_tokens"] == 300
    assert summary["input_tokens"] == 170
    assert summary["output_tokens"] == 130
    assert summary["llm_calls"] == 3
    assert summary["agent_runs"] == 2
    assert [stage["stage_name"] for stage in summary["per_stage"]] == ["coding", "testing"]
    assert summary["per_stage"][1]["failed_runs"] == 1


def test_session_store_migrates_existing_sessions_table_to_add_metadata(tmp_path):
    db_path = tmp_path / ".maike" / "session.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            workspace TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO sessions (id, task, workspace, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("session-1", "task", str(tmp_path), "running", "2026-03-20T00:00:00+00:00", "2026-03-20T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session = await store.get_session("session-1")
        return session

    session = asyncio.run(scenario())
    column_names = {
        row[1]
        for row in sqlite3.connect(db_path).execute("PRAGMA table_info(sessions)").fetchall()
    }

    assert "metadata" in column_names
    assert session is not None
    assert session["metadata"] == {}


def test_session_store_persists_session_metadata(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session(
            "task",
            tmp_path,
            metadata={
                "run_config": {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "budget": 2.5,
                    "agent_token_budget": 8000,
                    "language_override": "python",
                    "dynamic_agents_enabled": True,
                    "parallel_coding_enabled": False,
                    "auto_approve": True,
                }
            },
        )
        return await store.get_session(session_id)

    session = asyncio.run(scenario())

    assert session is not None
    assert session["metadata"]["run_config"]["provider"] == "gemini"
    assert session["metadata"]["run_config"]["model"] == "gemini-2.5-flash"
    assert session["metadata"]["run_config"]["agent_token_budget"] == 8000
