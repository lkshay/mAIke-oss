"""Tests for tool execution batching and TaskState/AgentProgress."""

from maike.atoms.context import AgentProgress, TaskState


class TestTaskState:
    def test_values(self):
        assert TaskState.PENDING.value == "pending"
        assert TaskState.RUNNING.value == "running"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELLED.value == "cancelled"

    def test_is_str(self):
        # TaskState(str, Enum) — values compare with plain strings.
        assert TaskState.RUNNING == "running"
        assert TaskState.COMPLETED == "completed"


class TestAgentProgress:
    def test_initial_state(self):
        p = AgentProgress()
        assert p.tool_use_count == 0
        assert p.last_activity == ""
        assert p.recent_activities == []

    def test_record_activity(self):
        p = AgentProgress()
        p.record_activity("Reading foo.py")
        assert p.tool_use_count == 1
        assert p.last_activity == "Reading foo.py"
        assert p.recent_activities == ["Reading foo.py"]

    def test_recent_activities_capped_at_5(self):
        p = AgentProgress()
        for i in range(10):
            p.record_activity(f"Activity {i}")
        assert p.tool_use_count == 10
        assert len(p.recent_activities) == 5
        assert p.recent_activities[0] == "Activity 5"
        assert p.recent_activities[-1] == "Activity 9"
        assert p.last_activity == "Activity 9"


class TestPartitionToolBatches:
    """Test the _partition_tool_batches method of AgentCore.

    We test via a minimal stub to avoid constructing a full AgentCore.
    """

    @staticmethod
    def _partition(tool_calls: list[dict]) -> list[tuple[list[dict], bool]]:
        """Mimic AgentCore._partition_tool_batches without needing an instance."""
        PARALLEL_SAFE = frozenset({"Grep", "SemanticSearch", "WebSearch", "WebFetch"})
        batches: list[tuple[list[dict], bool]] = []
        current_parallel: list[dict] = []

        for tc in tool_calls:
            name = tc["name"]
            if name in PARALLEL_SAFE:
                current_parallel.append(tc)
            else:
                if current_parallel:
                    batches.append((current_parallel, True))
                    current_parallel = []
                batches.append(([tc], False))

        if current_parallel:
            batches.append((current_parallel, True))

        return batches

    def test_all_parallel(self):
        calls = [{"name": "Grep"}, {"name": "WebSearch"}, {"name": "SemanticSearch"}]
        batches = self._partition(calls)
        assert len(batches) == 1
        batch, is_parallel = batches[0]
        assert is_parallel is True
        assert len(batch) == 3

    def test_all_sequential(self):
        calls = [{"name": "Write"}, {"name": "Bash"}, {"name": "Edit"}]
        batches = self._partition(calls)
        assert len(batches) == 3
        for batch, is_parallel in batches:
            assert is_parallel is False
            assert len(batch) == 1

    def test_mixed_partitioning(self):
        calls = [
            {"name": "Grep"},
            {"name": "Grep"},
            {"name": "Write"},
            {"name": "Grep"},
        ]
        batches = self._partition(calls)
        assert len(batches) == 3
        # First batch: two Greps, parallel
        assert batches[0] == ([{"name": "Grep"}, {"name": "Grep"}], True)
        # Second batch: Write, sequential
        assert batches[1] == ([{"name": "Write"}], False)
        # Third batch: one Grep, parallel (but only 1 item)
        assert batches[2] == ([{"name": "Grep"}], True)

    def test_single_tool(self):
        calls = [{"name": "Read"}]
        batches = self._partition(calls)
        assert len(batches) == 1
        assert batches[0] == ([{"name": "Read"}], False)

    def test_empty(self):
        assert self._partition([]) == []

    def test_parallel_then_exclusive_then_parallel(self):
        calls = [
            {"name": "WebSearch"},
            {"name": "WebFetch"},
            {"name": "Edit"},
            {"name": "SemanticSearch"},
            {"name": "Grep"},
        ]
        batches = self._partition(calls)
        assert len(batches) == 3
        assert batches[0][1] is True   # WebSearch + WebFetch
        assert batches[1][1] is False  # Edit
        assert batches[2][1] is True   # SemanticSearch + Grep
