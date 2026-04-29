"""Tests for maike.tools.edit — edit_file search-and-replace tool."""

import asyncio

import pytest

from maike.atoms.context import AgentContext
from maike.atoms.tool import ToolResult
from maike.tools.context import CURRENT_AGENT_CONTEXT
from maike.tools.edit import register_edit_tools
from maike.tools.filesystem import _file_read_state
from maike.tools.registry import ToolRegistry
from maike.utils import utcnow


@pytest.fixture(autouse=True)
def _reset_file_read_state():
    """Prevent state leakage between tests — the singleton is module-level."""
    _file_read_state._state.clear()
    _file_read_state._workspace = None
    yield
    _file_read_state._state.clear()
    _file_read_state._workspace = None


class FakeRuntime:
    """Minimal runtime that stores files in memory."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def read_file(self, path: str) -> ToolResult:
        if path in self.files:
            return ToolResult(
                tool_name="read_file",
                success=True,
                output=self.files[path],
                raw_output=self.files[path],
            )
        return ToolResult(
            tool_name="read_file",
            success=False,
            output=f"File not found: {path}",
            error="File not found",
        )

    async def read_file_full(self, path: str) -> str | None:
        return self.files.get(path)

    async def write_file(self, path: str, content: str) -> ToolResult:
        self.files[path] = content
        return ToolResult(
            tool_name="write_file",
            success=True,
            output=f"Wrote {len(content)} bytes to {path}",
        )


def _mark_read(*paths: str) -> None:
    """Simulate Read tool having been called on these paths."""
    for p in paths:
        _file_read_state.record_read(p)


def _make_ctx() -> AgentContext:
    return AgentContext(
        role="coder",
        task="test",
        stage_name="coding",
        tool_profile="coding",
        started_at=utcnow(),
        metadata={"session_id": "test-session"},
    )


def test_edit_tool_is_registered():
    registry = ToolRegistry()
    register_edit_tools(registry, FakeRuntime())

    assert registry.get("Edit") is not None
    assert "Edit" in registry.list_tool_names()


def test_edit_file_single_match_succeeds():
    runtime = FakeRuntime()
    runtime.files["app.py"] = "def hello():\n    return 'world'\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="return 'world'",
                new_text="return 'hello world'",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is True
    assert result.tool_name == "edit_file"
    assert "return 'hello world'" in runtime.files["app.py"]
    assert result.metadata["replacements"] == 1


def test_edit_file_returns_diff_output():
    runtime = FakeRuntime()
    runtime.files["app.py"] = "x = 1\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="x = 1",
                new_text="x = 42",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is True
    assert "-x = 1" in result.output
    assert "+x = 42" in result.output


def test_edit_file_zero_matches_returns_error():
    runtime = FakeRuntime()
    runtime.files["app.py"] = "def hello():\n    pass\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="def goodbye():",
                new_text="def farewell():",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is False
    assert "old_text not found" in result.output


def test_edit_file_multiple_matches_returns_error():
    runtime = FakeRuntime()
    runtime.files["app.py"] = "x = 1\nx = 1\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="x = 1",
                new_text="x = 2",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is False
    assert "matches 2 locations" in result.output


def test_edit_file_nonexistent_file_returns_error():
    runtime = FakeRuntime()
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("missing.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="missing.py",
                old_text="x",
                new_text="y",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is False
    assert "Cannot read file" in result.output


def test_edit_file_tracks_mutated_paths():
    runtime = FakeRuntime()
    runtime.files["app.py"] = "x = 1\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="x = 1",
                new_text="x = 2",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert "app.py" in ctx.metadata.get("mutated_paths", [])


# ---------------------------------------------------------------------------
# Read-state gate tests
# ---------------------------------------------------------------------------


def test_edit_file_not_read_returns_error():
    """Edit fails if the file hasn't been Read first."""
    runtime = FakeRuntime()
    runtime.files["app.py"] = "x = 1\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    # Deliberately do NOT call _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="x = 1",
                new_text="x = 2",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is False
    assert "Read it first" in result.output


def test_edit_preserves_read_state_after_success():
    """After a successful Edit, read state is preserved.

    The tool wrote `new_content` and verified it against disk, so the
    agent's view matches the file — forcing a redundant Read before the
    next Edit on the same file just creates spurious `file_not_read`
    errors when the agent makes several edits in a row.
    """
    runtime = FakeRuntime()
    runtime.files["app.py"] = "x = 1\ny = 2\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result1 = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="x = 1",
                new_text="x = 10",
            )
        )
        assert result1.success is True

        # Second edit without re-reading should still succeed — the file
        # contents written by edit 1 are what Edit 2 matches against.
        result2 = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="y = 2",
                new_text="y = 20",
            )
        )
        assert result2.success is True
        assert "y = 20" in runtime.files["app.py"]
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)


def test_edit_unifies_relative_and_dotslash_paths():
    """Reading `foo.py` and editing `./foo.py` should succeed — they're the same file."""
    runtime = FakeRuntime()
    # FakeRuntime's read_file_full is keyed by raw path, so stash under both
    # spellings so the Edit's internal read resolves regardless of which
    # form the agent used.  The bug we're guarding against is purely in
    # FileReadState normalization.
    runtime.files["app.py"] = "x = 1\n"
    runtime.files["./app.py"] = "x = 1\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")  # canonical form

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        # Agent varies the path spelling on the Edit call.
        result = asyncio.run(
            registry.get("Edit").fn(
                path="./app.py",
                old_text="x = 1",
                new_text="x = 2",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is True, result.output


def test_batch_edit_not_registered():
    """BatchEdit was removed — verify it's not in the registry."""
    registry = ToolRegistry()
    register_edit_tools(registry, FakeRuntime())
    assert registry.get("BatchEdit") is None


# ---------------------------------------------------------------------------
# Read-state invalidation via Bash mutation
# ---------------------------------------------------------------------------


def test_bash_mutation_regex_catches_common_patterns():
    """The mutation-detector regex should flag the typical file-writing commands."""
    from maike.tools.bash import _command_could_mutate_files

    # Redirection
    assert _command_could_mutate_files("echo hi > foo.txt")
    assert _command_could_mutate_files("python build.py >> log.txt")
    # Heredoc
    assert _command_could_mutate_files("cat > file.py <<EOF\nprint(1)\nEOF")
    # In-place sed
    assert _command_could_mutate_files("sed -i 's/a/b/' foo.py")
    assert _command_could_mutate_files("sed -Ei 's/a/b/' foo.py")
    # Formatters
    assert _command_could_mutate_files("black src/")
    assert _command_could_mutate_files("ruff check src/ --fix")
    assert _command_could_mutate_files("ruff format src/")
    assert _command_could_mutate_files("prettier --write app.ts")
    # File moves / deletes
    assert _command_could_mutate_files("mv foo.py bar.py")
    assert _command_could_mutate_files("rm -rf build/")
    assert _command_could_mutate_files("cp a.txt b.txt")
    # Git working-tree rewrites
    assert _command_could_mutate_files("git apply patch.diff")
    assert _command_could_mutate_files("git checkout -- foo.py")
    assert _command_could_mutate_files("git restore foo.py")

    # Non-mutating commands should NOT trip the regex.
    assert not _command_could_mutate_files("pytest tests/ -v")
    assert not _command_could_mutate_files("ls -la")
    assert not _command_could_mutate_files("grep -r 'foo' src/")
    assert not _command_could_mutate_files("git status")
    assert not _command_could_mutate_files("git diff HEAD")
    assert not _command_could_mutate_files("python -c 'print(1)'")


def test_bash_mutating_command_resets_read_state():
    """A mutating Bash command should clear read state so next Edit re-reads."""
    from maike.tools.bash import register_bash_tools

    class _Runtime:
        async def execute_bash(self, cmd, **_kwargs):
            return ToolResult(tool_name="Bash", success=True, output="", raw_output="")

    _mark_read("app.py", "helper.py")
    assert _file_read_state.was_read("app.py")

    registry = ToolRegistry()
    register_bash_tools(registry, _Runtime())
    asyncio.run(registry.get("Bash").fn(cmd="black app.py"))

    # All read state should be gone after a mutating command.
    assert not _file_read_state.was_read("app.py")
    assert not _file_read_state.was_read("helper.py")


def test_bash_nonmutating_command_preserves_read_state():
    """A pure-read Bash command should leave read state alone."""
    from maike.tools.bash import register_bash_tools

    class _Runtime:
        async def execute_bash(self, cmd, **_kwargs):
            return ToolResult(tool_name="Bash", success=True, output="", raw_output="")

    _mark_read("app.py")
    registry = ToolRegistry()
    register_bash_tools(registry, _Runtime())
    asyncio.run(registry.get("Bash").fn(cmd="pytest tests/ -q"))

    assert _file_read_state.was_read("app.py")


# ---------------------------------------------------------------------------
# Read-state invalidation via stale-read pruning
# ---------------------------------------------------------------------------


def test_pruning_clears_read_state_for_stubbed_reads():
    """When pruning replaces a Read result with a stub, the read-state
    entry for that path must also be cleared — otherwise a later Edit
    passes the file_not_read gate and matches against content the agent
    no longer actually has in view."""
    from maike.memory.working import WorkingMemory

    _mark_read("old.py")
    assert _file_read_state.was_read("old.py")

    # Build a conversation with an old Read for old.py and enough recent
    # messages that the Read falls outside the recent window.
    old_read_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_name": "Read",
                "input": {"path": "old.py"},
                "content": "1\tprint('old')\n",
                "is_error": False,
            }
        ],
    }
    # Recent_window=8 means we need >10 messages for the old one to be
    # considered stale.  Fill the tail with text messages that don't
    # reference old.py.
    filler = [
        {"role": "assistant", "content": [{"type": "text", "text": "working on something else"}]}
        for _ in range(12)
    ]
    conversation = [old_read_msg, *filler]

    wm = WorkingMemory()
    pruned = wm.clear_stale_tool_results(conversation, recent_window=8)

    # The Read block's content should now be a stub.
    first_block = pruned[0]["content"][0]
    assert "content pruned" in first_block["content"], first_block["content"]

    # And the read-state entry for old.py must be gone.
    assert not _file_read_state.was_read("old.py")


# ---------------------------------------------------------------------------
# CRLF / line ending normalisation
# ---------------------------------------------------------------------------


def test_edit_crlf_file_with_lf_old_text():
    """File has CRLF line endings but agent provides LF old_text — should succeed."""
    runtime = FakeRuntime()
    # File on disk has CRLF endings.
    runtime.files["app.py"] = "def hello():\r\n    return 'world'\r\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                # Agent provides LF (normal for LLM output).
                old_text="return 'world'",
                new_text="return 'hello world'",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is True
    # The file content should be updated (normalised to LF).
    assert "return 'hello world'" in runtime.files["app.py"]


def test_edit_lf_file_with_crlf_old_text():
    """File has LF endings but agent provides CRLF old_text — should succeed."""
    runtime = FakeRuntime()
    runtime.files["app.py"] = "x = 1\ny = 2\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                # Agent somehow provides CRLF.
                old_text="x = 1\r\ny = 2",
                new_text="x = 10\ny = 20",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is True
    assert "x = 10" in runtime.files["app.py"]
    assert "y = 20" in runtime.files["app.py"]


def test_edit_mixed_endings_file():
    """File with mixed CRLF/LF endings should still match correctly."""
    runtime = FakeRuntime()
    # Mixed: first line CRLF, second line LF.
    runtime.files["app.py"] = "line1\r\nline2\nline3\r\n"
    registry = ToolRegistry()
    register_edit_tools(registry, runtime)
    _mark_read("app.py")

    ctx = _make_ctx()
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        result = asyncio.run(
            registry.get("Edit").fn(
                path="app.py",
                old_text="line2",
                new_text="LINE2",
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    assert result.success is True
    assert "LINE2" in runtime.files["app.py"]


# ---------------------------------------------------------------------------
# Fuzzy match diagnostics
# ---------------------------------------------------------------------------


def test_fuzzy_match_detects_crlf_cause():
    """_find_fuzzy_match should identify CRLF vs LF as the specific cause."""
    from maike.tools.edit import _find_fuzzy_match

    # File content with CRLF, search with LF.
    content = "def hello():\r\n    return 'world'\r\n"
    old_text = "def hello():\n    return 'world'\n"

    hint = _find_fuzzy_match(content, old_text)
    assert hint is not None
    assert "line ending" in hint.lower() or "CRLF" in hint
