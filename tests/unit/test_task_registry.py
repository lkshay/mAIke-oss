"""Tests for unified TaskRegistry."""

from datetime import datetime, timezone
from types import SimpleNamespace

from maike.atoms.context import AgentProgress, TaskState
from maike.tasks.registry import TaskRegistry


def _make_delegate(handle, status=TaskState.RUNNING, desc="test task"):
    ctx = SimpleNamespace(
        cost_used_usd=0.01,
        tokens_used=100,
        progress=AgentProgress(last_activity="Reading foo.py"),
    )
    return SimpleNamespace(
        handle=handle,
        status=status,
        task_description=desc,
        started_at=datetime.now(timezone.utc),
        ctx=ctx,
    )


def _make_process(handle, status=TaskState.RUNNING, cmd="sleep 60"):
    return SimpleNamespace(
        handle=handle,
        status=status,
        cmd=cmd,
        started_at=datetime.now(timezone.utc),
    )


class FakeDelegateMgr:
    def __init__(self, delegates):
        self._delegates = {d.handle: d for d in delegates}


class FakeProcessMgr:
    def __init__(self, processes):
        self._processes = {p.handle: p for p in processes}

    def _sync_status(self, bp):
        pass


class TestTaskRegistry:
    def test_list_empty(self):
        reg = TaskRegistry(FakeDelegateMgr([]), FakeProcessMgr([]))
        assert reg.list_all() == []
        assert reg.count_running() == 0

    def test_list_delegates(self):
        d1 = _make_delegate("delegate-001")
        d2 = _make_delegate("delegate-002", status=TaskState.COMPLETED)
        reg = TaskRegistry(FakeDelegateMgr([d1, d2]), FakeProcessMgr([]))
        tasks = reg.list_all()
        assert len(tasks) == 2
        assert tasks[0].kind == "delegate"
        assert tasks[0].handle == "delegate-001"

    def test_list_processes(self):
        p1 = _make_process("bg-001")
        reg = TaskRegistry(FakeDelegateMgr([]), FakeProcessMgr([p1]))
        tasks = reg.list_all()
        assert len(tasks) == 1
        assert tasks[0].kind == "process"

    def test_list_mixed(self):
        d1 = _make_delegate("delegate-001")
        p1 = _make_process("bg-001")
        reg = TaskRegistry(FakeDelegateMgr([d1]), FakeProcessMgr([p1]))
        tasks = reg.list_all()
        assert len(tasks) == 2

    def test_count_running(self):
        d1 = _make_delegate("delegate-001")
        d2 = _make_delegate("delegate-002", status=TaskState.COMPLETED)
        p1 = _make_process("bg-001")
        reg = TaskRegistry(FakeDelegateMgr([d1, d2]), FakeProcessMgr([p1]))
        assert reg.count_running() == 2  # d1 + p1

    def test_format_table_empty(self):
        reg = TaskRegistry(FakeDelegateMgr([]), FakeProcessMgr([]))
        assert "No background tasks" in reg.format_table()

    def test_format_table_with_tasks(self):
        d1 = _make_delegate("delegate-001", desc="fix the bug")
        reg = TaskRegistry(FakeDelegateMgr([d1]), FakeProcessMgr([]))
        table = reg.format_table()
        assert "delegate-001" in table
        assert "fix the bug" in table
        assert "running" in table
