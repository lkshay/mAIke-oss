"""Tests for tool activity descriptions."""

from maike.tools.activity import get_activity_description


class TestGetActivityDescription:
    def test_read_simple(self):
        desc = get_activity_description("Read", {"path": "src/main.py"})
        assert desc == "Reading src/main.py"

    def test_read_with_line_range(self):
        desc = get_activity_description("Read", {"path": "src/main.py", "start_line": 10, "end_line": 50})
        assert desc == "Reading src/main.py:10-50"

    def test_write(self):
        desc = get_activity_description("Write", {"path": "new_file.py"})
        assert desc == "Writing new_file.py"

    def test_edit(self):
        desc = get_activity_description("Edit", {"path": "foo.py"})
        assert desc == "Editing foo.py"

    def test_grep(self):
        desc = get_activity_description("Grep", {"pattern": "TaskState", "path": "maike/"})
        assert desc == "Searching for 'TaskState' in maike/"

    def test_grep_long_pattern_truncated(self):
        long = "a" * 100
        desc = get_activity_description("Grep", {"pattern": long, "path": "."})
        assert len(desc) < 60 + len("Searching for '' in .")

    def test_bash(self):
        desc = get_activity_description("Bash", {"cmd": "pytest tests/ -v"})
        assert desc == "Running: pytest tests/ -v"

    def test_bash_long_cmd_truncated(self):
        long_cmd = "echo " + "x" * 200
        desc = get_activity_description("Bash", {"cmd": long_cmd})
        assert len(desc) <= len("Running: ") + 60

    def test_bash_empty(self):
        desc = get_activity_description("Bash", {})
        assert desc == "Running Bash"

    def test_delegate(self):
        desc = get_activity_description("Delegate", {"task": "find all usages of TaskState", "agent_type": "explore"})
        assert desc.startswith("Delegating (explore):")

    def test_web_search(self):
        desc = get_activity_description("WebSearch", {"query": "python asyncio patterns"})
        assert "python asyncio patterns" in desc

    def test_web_fetch(self):
        desc = get_activity_description("WebFetch", {"url": "https://docs.python.org/3/library/asyncio.html"})
        assert "https://docs.python.org" in desc

    def test_semantic_search(self):
        desc = get_activity_description("SemanticSearch", {"query": "error handling"})
        assert "error handling" in desc

    def test_unknown_tool(self):
        desc = get_activity_description("UnknownTool", {"foo": "bar"})
        assert desc == "Running UnknownTool"
