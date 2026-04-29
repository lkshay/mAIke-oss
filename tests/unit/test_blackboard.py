"""Tests for Blackboard and blackboard tools."""

import asyncio
import threading

from maike.atoms.tool import RiskLevel
from maike.tools.blackboard import Blackboard, register_blackboard_tools
from maike.tools.registry import ToolRegistry


class TestBlackboard:
    def test_post_and_read(self):
        bb = Blackboard()
        idx = bb.post("topic", "message one")
        assert idx == 0
        idx2 = bb.post("topic", "message two")
        assert idx2 == 1
        entries = bb.read("topic")
        assert entries == ["message one", "message two"]

    def test_read_missing_key(self):
        bb = Blackboard()
        assert bb.read("missing") == []

    def test_read_all(self):
        bb = Blackboard()
        bb.post("a", "1")
        bb.post("b", "2")
        all_data = bb.read()
        assert isinstance(all_data, dict)
        assert "a" in all_data
        assert "b" in all_data

    def test_keys(self):
        bb = Blackboard()
        bb.post("x", "val")
        bb.post("y", "val")
        assert sorted(bb.keys()) == ["x", "y"]

    def test_clear(self):
        bb = Blackboard()
        bb.post("x", "val")
        bb.clear()
        assert bb.read() == {}

    def test_thread_safety(self):
        bb = Blackboard()
        errors = []

        def writer(prefix: str, n: int):
            try:
                for i in range(n):
                    bb.post("shared", f"{prefix}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"t{i}", 100)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        entries = bb.read("shared")
        assert len(entries) == 500


class TestBlackboardTools:
    def test_registration(self):
        registry = ToolRegistry()
        bb = Blackboard()
        register_blackboard_tools(registry, bb)
        assert registry.get("BlackboardPost") is not None
        assert registry.get("BlackboardRead") is not None
        assert registry.get("BlackboardPost").risk_level == RiskLevel.WRITE
        assert registry.get("BlackboardRead").risk_level == RiskLevel.READ

    def test_post_tool(self):
        registry = ToolRegistry()
        bb = Blackboard()
        register_blackboard_tools(registry, bb)
        tool = registry.get("BlackboardPost")
        result = asyncio.run(tool.fn(key="api", value="GET /users"))
        assert result.success is True
        assert "api" in result.output
        assert bb.read("api") == ["GET /users"]

    def test_read_tool(self):
        registry = ToolRegistry()
        bb = Blackboard()
        bb.post("contracts", "interface Foo { bar(): string }")
        register_blackboard_tools(registry, bb)
        tool = registry.get("BlackboardRead")
        result = asyncio.run(tool.fn(key="contracts"))
        assert result.success is True
        assert "Foo" in result.output

    def test_read_empty(self):
        registry = ToolRegistry()
        bb = Blackboard()
        register_blackboard_tools(registry, bb)
        tool = registry.get("BlackboardRead")
        result = asyncio.run(tool.fn())
        assert result.success is True
        assert "empty" in result.output.lower()
