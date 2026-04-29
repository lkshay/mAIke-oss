"""Tests for delegate agent tool profile filtering and prompt loading."""

from pathlib import Path

from maike.agents.delegate import _AGENT_TYPE_CONFIG, _load_delegate_prompt, tool_profile_for_agent_type


class TestAgentTypeConfig:
    def test_all_seven_types_registered(self):
        assert set(_AGENT_TYPE_CONFIG.keys()) == {
            "explore", "plan", "implement", "review", "verify", "debug", "test",
        }

    def test_each_config_has_prompt_and_profile(self):
        for agent_type, (prompt_file, profile) in _AGENT_TYPE_CONFIG.items():
            assert prompt_file.endswith(".md"), f"{agent_type}: prompt file should be .md"
            assert isinstance(profile, str), f"{agent_type}: profile should be a string"

    def test_prompt_files_exist(self):
        prompts_dir = Path(__file__).parent.parent.parent / "maike" / "agents" / "prompts"
        for agent_type, (prompt_file, _) in _AGENT_TYPE_CONFIG.items():
            path = prompts_dir / prompt_file
            assert path.exists(), f"Prompt file missing for {agent_type}: {path}"

    def test_implement_and_test_share_delegate_profile(self):
        """implement and test both use unrestricted 'delegate' profile."""
        assert _AGENT_TYPE_CONFIG["implement"][1] == "delegate"
        assert _AGENT_TYPE_CONFIG["test"][1] == "delegate"


class TestToolProfileForAgentType:
    def test_explore_profile(self):
        assert tool_profile_for_agent_type("explore") == "delegate_explore"

    def test_plan_profile(self):
        assert tool_profile_for_agent_type("plan") == "delegate_plan"

    def test_implement_profile(self):
        assert tool_profile_for_agent_type("implement") == "delegate"

    def test_review_profile(self):
        assert tool_profile_for_agent_type("review") == "delegate_review"

    def test_verify_profile(self):
        assert tool_profile_for_agent_type("verify") == "delegate_verify"

    def test_debug_profile(self):
        assert tool_profile_for_agent_type("debug") == "delegate_debug"

    def test_test_profile(self):
        assert tool_profile_for_agent_type("test") == "delegate"

    def test_unknown_falls_back_to_implement(self):
        assert tool_profile_for_agent_type("nonexistent") == "delegate"


class TestLoadDelegatePrompt:
    def test_loads_each_agent_type(self):
        for agent_type in _AGENT_TYPE_CONFIG:
            prompt = _load_delegate_prompt(agent_type)
            assert len(prompt) > 50, f"{agent_type} prompt is too short"

    def test_unknown_type_falls_back_to_implement(self):
        prompt = _load_delegate_prompt("nonexistent")
        implement_prompt = _load_delegate_prompt("implement")
        assert prompt == implement_prompt


class TestToolProfileFiltering:
    """Test that profile names map to correct tool restrictions.

    Uses a minimal stub to test the filtering logic without instantiating AgentCore.
    """

    # Mirror the profiles defined in AgentCore._DELEGATE_PROFILE_TOOLS.
    _PROFILES = {
        "delegate_explore": frozenset({"Read", "Grep", "SemanticSearch"}),
        "delegate_plan":    frozenset({"Read", "Grep", "SemanticSearch"}),
        "delegate_verify":  frozenset({"Read", "Grep", "Bash"}),
        "delegate_review":  frozenset({"Read", "Grep", "Bash"}),
        "delegate_debug":   frozenset({"Read", "Grep", "Bash", "Edit"}),
    }

    _ALL_TOOLS = [
        {"name": "Read"}, {"name": "Write"}, {"name": "Edit"},
        {"name": "Grep"}, {"name": "Bash"}, {"name": "Delegate"},
        {"name": "SemanticSearch"}, {"name": "WebSearch"}, {"name": "AskUser"},
    ]

    def _filter(self, profile: str) -> set[str]:
        if profile in self._PROFILES:
            allowed = self._PROFILES[profile]
            return {s["name"] for s in self._ALL_TOOLS if s["name"] in allowed}
        return {s["name"] for s in self._ALL_TOOLS}

    def test_explore_gets_read_only(self):
        tools = self._filter("delegate_explore")
        assert tools == {"Read", "Grep", "SemanticSearch"}
        assert "Write" not in tools
        assert "Bash" not in tools

    def test_plan_gets_same_as_explore(self):
        assert self._filter("delegate_plan") == self._filter("delegate_explore")

    def test_verify_gets_read_plus_bash(self):
        tools = self._filter("delegate_verify")
        assert tools == {"Read", "Grep", "Bash"}
        assert "Write" not in tools
        assert "Edit" not in tools

    def test_review_gets_same_as_verify(self):
        assert self._filter("delegate_review") == self._filter("delegate_verify")

    def test_debug_gets_read_grep_bash_edit(self):
        tools = self._filter("delegate_debug")
        assert tools == {"Read", "Grep", "Bash", "Edit"}
        assert "Write" not in tools
        assert "Delegate" not in tools

    def test_implement_gets_all_tools(self):
        """'delegate' profile (implement/test) returns all tools unfiltered."""
        tools = self._filter("delegate")
        assert tools == {s["name"] for s in self._ALL_TOOLS}

    def test_test_gets_all_tools(self):
        """test uses 'delegate' profile — same as implement."""
        tools = self._filter("delegate")
        assert "Write" in tools
        assert "Edit" in tools
        assert "Bash" in tools
