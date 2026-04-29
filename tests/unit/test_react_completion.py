"""Tests for react agent completion validation and tool name aliases."""

from types import SimpleNamespace

from maike.agents.core import AgentCore, ReactCompletionValidation


def _mock_ctx(task: str = "build something") -> SimpleNamespace:
    """Create a minimal mock context with a task field."""
    return SimpleNamespace(task=task)


class TestExtractToolCalls:
    """Test extracting tool call names from conversation history."""

    def test_extracts_tool_calls_from_assistant_messages(self):
        conversation = [
            {"role": "user", "content": "build something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Grep", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file.py", "is_error": False},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "Write", "input": {"path": "app.py"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "OK", "is_error": False},
                ],
            },
        ]
        tool_calls = AgentCore._extract_tool_calls_from_conversation(conversation)
        names = {tc["name"] for tc in tool_calls}
        assert "Grep" in names
        assert "Write" in names

    def test_tracks_success_status(self):
        conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "error", "is_error": True},
                ],
            },
        ]
        tool_calls = AgentCore._extract_tool_calls_from_conversation(conversation)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "Bash"
        assert tool_calls[0]["success"] is False

    def test_empty_conversation(self):
        tool_calls = AgentCore._extract_tool_calls_from_conversation([])
        assert tool_calls == []

    def test_plain_text_messages_ignored(self):
        conversation = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        tool_calls = AgentCore._extract_tool_calls_from_conversation(conversation)
        assert tool_calls == []


class TestValidateReactCompletion:
    """Test that the react completion validator catches premature exits."""

    def _make_core(self):
        """Create a minimal AgentCore for testing the static methods."""
        return AgentCore.__new__(AgentCore)

    def _conversation_with_tools(self, tool_names: list[str]) -> list:
        """Build a conversation with the given tool calls (all successful)."""
        conversation = []
        for i, name in enumerate(tool_names):
            conversation.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": {}}],
            })
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{i}", "content": "OK", "is_error": False},
                ],
            })
        return conversation

    def test_no_files_written_but_explored_question_task_is_satisfied(self):
        """Agent that explored (read files) on a question task is satisfied."""
        core = self._make_core()
        ctx = _mock_ctx("explain how the database config works")
        conversation = self._conversation_with_tools(["Grep", "Read"])
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_no_files_written_exploration_only_is_satisfied(self):
        """Agent that explored but didn't write trusts the LLM — satisfied."""
        core = self._make_core()
        ctx = _mock_ctx("create a CLI tool")
        conversation = self._conversation_with_tools(["Grep", "Read"])
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_files_written_but_not_verified_returns_not_satisfied(self):
        core = self._make_core()
        ctx = _mock_ctx("build a web server")
        conversation = self._conversation_with_tools(["Grep", "Write"])
        result = core._validate_react_completion(ctx, conversation)
        assert not result.satisfied
        assert "verified" in result.nudge_message.lower() or "verify" in result.nudge_message.lower()

    def test_files_written_and_verified_returns_satisfied(self):
        core = self._make_core()
        ctx = _mock_ctx("create a bookmark manager")
        conversation = self._conversation_with_tools(["Grep", "Write", "Bash"])
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied
        assert result.nudge_message == ""

    def test_edit_counts_as_write(self):
        core = self._make_core()
        ctx = _mock_ctx("fix the bug in auth.py")
        conversation = self._conversation_with_tools(["Edit", "Bash"])
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_empty_conversation_satisfied_trusts_llm(self):
        """Empty conversation with no writes — LLM chose to stop, trust it."""
        core = self._make_core()
        ctx = _mock_ctx("build a REST API")
        result = core._validate_react_completion(ctx, [])
        assert result.satisfied


class TestBehaviorBasedQuestionDetection:
    """Test that tasks are classified by agent behavior, not task text."""

    def test_exploration_only_is_satisfied(self):
        """Agent that only reads files (answering a question) can finish."""
        core = AgentCore.__new__(AgentCore)
        ctx = _mock_ctx("Can you explain the code we have generated so far?")
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
        ]
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_grep_counts_as_exploration(self):
        """Grep is an exploration tool."""
        core = AgentCore.__new__(AgentCore)
        ctx = _mock_ctx("Where is the database config?")
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Grep", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
        ]
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_no_tools_at_all_satisfied(self):
        """Agent that used no tools — LLM chose to stop, trust it."""
        core = AgentCore.__new__(AgentCore)
        ctx = _mock_ctx("Build a CLI tool")
        result = core._validate_react_completion(ctx, [])
        assert result.satisfied

    def test_no_writes_exploration_only_satisfied(self):
        """No writes means no verification needed — satisfied."""
        core = AgentCore.__new__(AgentCore)
        ctx = _mock_ctx("Build something")
        result = core._validate_react_completion(ctx, [])
        assert result.satisfied

    def test_coding_task_with_exploration_only_satisfied(self):
        """Exploration only (no writes) is satisfied — trust LLM decision."""
        core = AgentCore.__new__(AgentCore)
        ctx = _mock_ctx("Build a Python CLI for bookmarks")
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
        ]
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_wrote_files_without_verification_nudges(self):
        """Agent that wrote files but didn't verify should be nudged."""
        core = AgentCore.__new__(AgentCore)
        ctx = _mock_ctx("Create a hello world script")
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Write", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
        ]
        result = core._validate_react_completion(ctx, conversation)
        assert not result.satisfied
        assert "verif" in result.nudge_message.lower()


class TestAskUserCompletionValidation:
    """Test that the agent is nudged when it presents a plan via AskUser but doesn't execute."""

    def _make_core(self):
        return AgentCore.__new__(AgentCore)

    def test_askuser_then_no_writes_satisfied(self):
        """Agent called AskUser but made no writes — no writes = satisfied (trust LLM)."""
        core = self._make_core()
        ctx = _mock_ctx("add a dark mode toggle")
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "AskUser", "input": {"prompt": "I plan to..."}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "yes go ahead", "is_error": False}]},
        ]
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_askuser_then_write_is_satisfied(self):
        """Agent called AskUser then actually wrote files and verified → satisfied."""
        core = self._make_core()
        ctx = _mock_ctx("add a dark mode toggle")
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "AskUser", "input": {"prompt": "I plan to..."}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "yes", "is_error": False}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Edit", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "OK", "is_error": False}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t3", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t3", "content": "OK", "is_error": False}]},
        ]
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_askuser_on_question_task_is_satisfied(self):
        """AskUser on a question task (no action keywords) → satisfied, no nudge."""
        core = self._make_core()
        ctx = _mock_ctx("explain how the auth system works")
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "AskUser", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "yes", "is_error": False}]},
        ]
        result = core._validate_react_completion(ctx, conversation)
        assert result.satisfied

    def test_extract_preserves_order(self):
        """Tool calls should be returned in chronological order."""
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "OK", "is_error": False}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "AskUser", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "go ahead", "is_error": False}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Write", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "OK", "is_error": False}]},
        ]
        tool_calls = AgentCore._extract_tool_calls_from_conversation(conversation)
        names = [tc["name"] for tc in tool_calls]
        assert names == ["Read", "AskUser", "Write"]

    def test_extract_includes_askuser_response(self):
        """AskUser tool calls should include the user's response text."""
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "AskUser", "input": {"prompt": "confirm?"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "yes go ahead", "is_error": False}]},
        ]
        tool_calls = AgentCore._extract_tool_calls_from_conversation(conversation)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "AskUser"
        assert tool_calls[0]["response"] == "yes go ahead"

    def test_non_askuser_calls_have_no_response_key(self):
        """Non-AskUser tool calls should not have a 'response' key."""
        conversation = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "file content", "is_error": False}]},
        ]
        tool_calls = AgentCore._extract_tool_calls_from_conversation(conversation)
        assert "response" not in tool_calls[0]


class TestToolNameAliases:
    """Test that tool name aliases resolve to new canonical names."""

    def test_execute_cmd_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("execute_cmd") == "Bash"

    def test_run_bash_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("run_bash") == "Bash"

    def test_exec_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("exec") == "Bash"

    def test_save_file_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("save_file") == "Write"

    def test_old_read_file_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("read_file") == "Read"

    def test_old_write_file_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("write_file") == "Write"

    def test_old_edit_file_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("edit_file") == "Edit"

    def test_old_execute_bash_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("execute_bash") == "Bash"

    def test_old_grep_codebase_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("grep_codebase") == "Grep"

    def test_old_run_tests_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("run_tests") == "Bash"

    def test_test_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("test") == "Bash"

    def test_old_request_user_input_alias(self):
        assert AgentCore._TOOL_NAME_ALIASES.get("request_user_input") == "AskUser"
