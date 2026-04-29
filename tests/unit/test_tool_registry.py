import asyncio
from datetime import timedelta

from maike.atoms.context import AgentContext
from maike.atoms.tool import ToolResult
from maike.constants import DEFAULT_BASH_TOOL_TIMEOUT_SECONDS, MAX_BASH_TOOL_TIMEOUT_SECONDS
from maike.memory.session import SessionStore
from maike.runtime.local import LocalRuntime
from maike.tools import register_default_tools
from maike.tools.bash import register_bash_tools
from maike.tools.context import CURRENT_AGENT_CONTEXT
from maike.tools.registry import ToolRegistry
from maike.utils import utcnow


def test_flat_registry_returns_all_tools(tmp_path):
    """After removing profiles, all tools are available to every agent."""
    registry = ToolRegistry()
    register_default_tools(
        registry,
        LocalRuntime(tmp_path),
        session=SessionStore(tmp_path),
    )
    tool_names = set(registry.list_tool_names())
    # Core tools should always be present
    assert "Read" in tool_names
    assert "Write" in tool_names
    assert "Edit" in tool_names
    assert "Grep" in tool_names
    assert "Bash" in tool_names


def test_get_all_schemas_returns_every_tool(tmp_path):
    registry = ToolRegistry()
    register_default_tools(
        registry,
        LocalRuntime(tmp_path),
        session=SessionStore(tmp_path),
    )
    schemas = registry.get_all_schemas()
    schema_names = {s["name"] for s in schemas}
    assert "Read" in schema_names
    assert "Write" in schema_names
    assert "Edit" in schema_names
    assert "Grep" in schema_names
    assert "Bash" in schema_names
    # Each schema has required fields
    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema


def test_run_tests_treats_no_test_collection_as_non_fatal(tmp_path):
    runtime = LocalRuntime(tmp_path)
    result = __import__("asyncio").run(runtime.run_tests("."))
    assert result.tool_name == "run_tests"
    assert result.success is True


def test_bash_tool_defaults_and_clamps_timeout():
    class FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int | None, str | None]] = []

        async def execute_bash(
            self,
            cmd: str,
            timeout: float = 30,
            *,
            idle_timeout: float | None = None,
            timeout_class: str | None = None,
        ) -> ToolResult:
            self.calls.append((cmd, int(timeout), int(idle_timeout) if idle_timeout is not None else None, timeout_class))
            return ToolResult(
                tool_name="Bash",
                success=True,
                output=f"{cmd} ({timeout})",
                metadata={
                    "timeout_seconds": timeout,
                    "idle_timeout_seconds": idle_timeout,
                    "timeout_class": timeout_class,
                },
            )

    runtime = FakeRuntime()
    registry = ToolRegistry()
    register_bash_tools(registry, runtime)

    bash_tool = registry.get("Bash")
    assert bash_tool is not None

    omitted = asyncio.run(bash_tool.fn(cmd="pwd"))
    oversized = asyncio.run(bash_tool.fn(cmd="pwd", timeout=999))

    assert omitted.success is True
    assert oversized.success is True
    assert runtime.calls == [
        ("pwd", DEFAULT_BASH_TOOL_TIMEOUT_SECONDS, 20, "normal"),
        ("pwd", MAX_BASH_TOOL_TIMEOUT_SECONDS, 90, "long"),
    ]


def test_bash_tool_schema_advertises_timeout_class_and_hint_bounds(tmp_path):
    registry = ToolRegistry()
    register_default_tools(
        registry,
        LocalRuntime(tmp_path),
        session=None,
    )

    bash_tool = registry.get("Bash")
    assert bash_tool is not None
    properties = bash_tool.schema.input_schema["properties"]
    timeout_schema = bash_tool.schema.input_schema["properties"]["timeout"]

    # Unified schema: no fields are globally required since handle-based
    # operations don't need cmd.
    assert bash_tool.schema.input_schema["required"] == []
    assert timeout_schema["default"] == DEFAULT_BASH_TOOL_TIMEOUT_SECONDS
    assert timeout_schema["maximum"] == MAX_BASH_TOOL_TIMEOUT_SECONDS
    assert properties["timeout_class"]["default"] == "normal"
    assert properties["timeout_class"]["enum"] == ["short", "normal", "long"]


def test_bash_tool_respects_elapsed_agent_wall_time_budget():
    class FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int | None, str | None]] = []

        async def execute_bash(
            self,
            cmd: str,
            timeout: float = 30,
            *,
            idle_timeout: float | None = None,
            timeout_class: str | None = None,
        ) -> ToolResult:
            del cmd
            self.calls.append((int(timeout), int(idle_timeout) if idle_timeout is not None else None, timeout_class))
            return ToolResult(
                tool_name="Bash",
                success=True,
                output="ok",
                metadata={},
            )

    runtime = FakeRuntime()
    registry = ToolRegistry()
    register_bash_tools(registry, runtime)
    bash_tool = registry.get("Bash")
    assert bash_tool is not None

    ctx = AgentContext(
        role="coder",
        task="build",
        stage_name="coding",
        tool_profile="coding",
        started_at=utcnow() - timedelta(seconds=1_150),
        metadata={"session_id": "session-1"},
    )
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(bash_tool.fn(cmd="pytest -q", timeout=999, timeout_class="long"))
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is True
    assert runtime.calls == [(50, 49, "long")]
    assert result.metadata["agent_wall_time_remaining_seconds"] == 50


    # test_list_packages removed — package tools eliminated in tool refactoring.
