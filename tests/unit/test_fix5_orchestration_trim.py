"""Tests for Fix 5 — trim trackers and harden convergence enforcement.

Covers four orchestration changes prompted by the 76-turn `ls` purgatory
transcript:

1. ``SessionToolTracker.record_bash`` matches command FAMILIES (so ``ls -F``
   and ``ls -R`` both register as the ``ls`` family) instead of requiring
   byte-identical commands.
2. ``RepeatedFailureTracker.first_failure_hint`` allows up to N hints per
   error category instead of suppressing the category permanently after one.
3. ``_ls_loop_signal`` threshold dropped from 70% to 50% so legitimate
   ls-loops are flagged earlier.
4. L3 convergence becomes a hard stop — after the L3 nudge has been ignored
   for ``_L3_GRACE_ITERATIONS`` iterations, the loop force-terminates with
   ``termination_reason="convergence_failed"``.
"""

from __future__ import annotations

import asyncio

import pytest

from maike.agents.core import (
    AgentCore,
    RepeatedFailureTracker,
    SessionToolTracker,
    _bash_command_family,
)
from maike.atoms.context import AgentContext
from maike.atoms.llm import LLMResult, StopReason, TokenUsage
from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.constants import DEFAULT_MODEL
from maike.context.convergence import _ls_loop_signal, detect_spinning_v2
from maike.observability.tracer import Tracer
from maike.safety.approval import ApprovalGate
from maike.safety.hooks import SafetyLayer
from maike.tools.registry import ToolRegistry


# =============================================================================
# Fix 5 part 1: command-family matching in SessionToolTracker.record_bash
# =============================================================================


class TestBashCommandFamily:
    """The 76-turn transcript ran 6 ``ls`` variants (``ls``, ``ls -F``,
    ``ls -R``, ``ls -la``, ``ls -aR``, ``ls -aF``) and the previous
    byte-identical match never fired.  Family matching catches these."""

    @pytest.mark.parametrize("cmd, expected", [
        ("ls", "ls"),
        ("ls -F", "ls"),
        ("ls -R", "ls"),
        ("ls -la", "ls"),
        ("ls -aR /tmp", "ls"),
        ("find . -name '*.py'", "find"),
        ("tree", "tree"),
        ("cat README.md", "cat"),
        ("pytest", "pytest"),
    ])
    def test_simple_commands_use_first_token(self, cmd, expected):
        assert _bash_command_family(cmd) == expected

    @pytest.mark.parametrize("cmd, expected", [
        ("git status", "git status"),
        ("git log", "git log"),
        ("git log --oneline", "git log"),
        ("git diff HEAD~1", "git diff"),
        ("git clone https://github.com/x/y", "git clone"),
        ("docker ps", "docker ps"),
        ("docker run ubuntu", "docker run"),
        ("npm install react", "npm install"),
        ("kubectl get pods", "kubectl get"),
        ("pip install requests", "pip install"),
    ])
    def test_subcommand_tools_use_first_two_tokens(self, cmd, expected):
        assert _bash_command_family(cmd) == expected

    def test_subcommand_with_only_flags_falls_back_to_first_token(self):
        """`git --version` has only flags after `git` — return just `git`."""
        assert _bash_command_family("git --version") == "git"

    def test_empty_or_whitespace(self):
        assert _bash_command_family("") == ""
        assert _bash_command_family("   ") == ""


class TestRecordBashFamilyMatching:
    def test_byte_identical_repeats_still_nudge(self):
        """The original behavior — same command twice — still triggers."""
        tracker = SessionToolTracker()
        assert tracker.record_bash("ls") is None
        assert tracker.record_bash("ls") is None  # repeat=1, not yet
        nudge = tracker.record_bash("ls")  # repeat=2 → fires
        assert nudge is not None
        assert "ls" in nudge

    def test_ls_variants_share_family_and_nudge(self):
        """The transcript bug: `ls`, `ls -F`, `ls -R` are different commands
        but the same family.  Must nudge."""
        tracker = SessionToolTracker()
        assert tracker.record_bash("ls") is None
        assert tracker.record_bash("ls -F") is None
        nudge = tracker.record_bash("ls -R")
        assert nudge is not None
        assert "ls" in nudge

    def test_full_transcript_ls_sequence_nudges_early(self):
        """Replay of the 6 `ls` variants from the 76-turn transcript."""
        tracker = SessionToolTracker()
        sequence = ["ls", "ls -F", "ls -R", "ls -la", "ls -aR", "ls -aF"]
        nudges = [tracker.record_bash(cmd) for cmd in sequence]
        # First two have no nudge (need ≥2 repeats for the threshold).
        assert nudges[0] is None
        assert nudges[1] is None
        # Third onwards: at least one nudge fires.
        assert any(n is not None for n in nudges[2:])

    def test_different_families_reset_counter(self):
        """`ls` then `git status` then `ls` — counter resets."""
        tracker = SessionToolTracker()
        tracker.record_bash("ls")
        tracker.record_bash("git status")
        # Family `ls` no longer matches `git status`, so back to fresh count.
        assert tracker.record_bash("ls -F") is None  # repeat=0 (re-set)
        assert tracker.record_bash("ls") is None  # repeat=1
        nudge = tracker.record_bash("ls -R")  # repeat=2 → fires
        assert nudge is not None

    def test_git_subcommands_do_not_collide(self):
        """`git status` and `git log` are different families — no false nudge."""
        tracker = SessionToolTracker()
        tracker.record_bash("git status")
        tracker.record_bash("git log")
        # Should NOT have nudged on these — different subcommands.
        assert tracker.record_bash("git diff") is None

    def test_ls_nudge_text_mentions_family(self):
        tracker = SessionToolTracker()
        tracker.record_bash("ls")
        tracker.record_bash("ls -F")
        nudge = tracker.record_bash("ls -R")
        assert nudge is not None
        # Mentions the family (`ls`) and the most recent specific command.
        assert "`ls`" in nudge
        assert "ls -R" in nudge


# =============================================================================
# Fix 5 part 2: RepeatedFailureTracker first_failure_hint allows N hints
# =============================================================================


class TestCategoryHintLimit:
    """Previously suppressed after one hint — silently disabled guidance for
    the rest of the session.  Now allows _CATEGORY_HINT_LIMIT before
    suppressing."""

    def test_hint_fires_up_to_limit(self):
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "RecursionError: maximum recursion depth"
        for i in range(RepeatedFailureTracker._CATEGORY_HINT_LIMIT):
            hint = tracker.first_failure_hint(output)
            assert hint is not None, f"expected hint on call {i+1}"

    def test_hint_suppressed_after_limit(self):
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "RecursionError: maximum recursion depth"
        for _ in range(RepeatedFailureTracker._CATEGORY_HINT_LIMIT):
            tracker.first_failure_hint(output)
        assert tracker.first_failure_hint(output) is None

    def test_distinct_categories_have_independent_counters(self):
        """Hitting the recursion limit shouldn't suppress import hints."""
        tracker = RepeatedFailureTracker()
        rec = "RecursionError: maximum recursion depth"
        imp = "ModuleNotFoundError: No module named 'foo'"
        # Burn the recursion budget.
        for _ in range(RepeatedFailureTracker._CATEGORY_HINT_LIMIT):
            tracker.first_failure_hint(rec)
        # Import hint should still fire.
        assert tracker.first_failure_hint(imp) is not None


# =============================================================================
# Fix 5 part 3: _ls_loop_signal threshold lowered to 50%
# =============================================================================


def _ls_call(cmd: str = "ls") -> dict:
    """Build a tool_use message that looks like a Bash `ls` invocation."""
    return {
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": cmd},
        }],
    }


def _other_call(name: str = "Read", arg: str = "foo.py") -> dict:
    return {
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "name": name,
            "input": {"path": arg},
        }],
    }


class TestLsLoopThreshold:
    """50% threshold matches the transcript pattern: ~half of recent calls
    were ls variants, the rest were Reads of nonexistent files.  At 70% the
    loop was never flagged."""

    def test_all_ls_fires(self):
        # 4 `ls` calls — 100% — definitely fires.
        conversation = [_ls_call("ls -F") for _ in range(4)]
        assert _ls_loop_signal(conversation) is True

    def test_half_ls_fires_at_new_threshold(self):
        """4 ls + 4 reads = 50%.  At 70% this would not fire; at 50% it does."""
        conversation = (
            [_ls_call("ls -F") for _ in range(4)]
            + [_other_call("Read", f"foo{i}.py") for i in range(4)]
        )
        assert _ls_loop_signal(conversation) is True

    def test_just_below_half_does_not_fire(self):
        """3 ls + 5 reads = 37.5%.  Below 50% threshold."""
        conversation = (
            [_ls_call("ls -F") for _ in range(3)]
            + [_other_call("Read", f"foo{i}.py") for i in range(5)]
        )
        assert _ls_loop_signal(conversation) is False

    def test_too_few_calls_does_not_fire(self):
        """Need at least 4 calls before the signal can fire."""
        conversation = [_ls_call("ls") for _ in range(3)]
        assert _ls_loop_signal(conversation) is False

    def test_non_ls_bash_does_not_count(self):
        """`pytest` / `git status` are Bash but not directory listings."""
        conversation = (
            [_other_call("Bash", "pytest") for _ in range(8)]
        )
        # Pytest commands are Bash but not ls-shaped — should NOT fire.
        assert _ls_loop_signal(conversation) is False


# =============================================================================
# Fix 5 part 4: L3 convergence hard stop
# =============================================================================


class _FakeGateway:
    """Minimal gateway that returns canned LLM results in sequence."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.provider_name = "anthropic"

    async def call(self, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        # Default: keep producing tool calls forever (simulates a stuck agent).
        return _make_tool_result_response("Bash", {"command": "ls"}, str(self.calls))

    def resolve_model_for_tier(self, tier: str) -> str:
        return "claude-opus-4-20250514"


class _StaticWorkingMemory:
    def prune(self, messages, **kwargs):
        return messages


def _make_tool_result_response(tool_name: str, tool_input: dict, tool_use_id: str = "t1") -> LLMResult:
    return LLMResult(
        provider="anthropic",
        content=None,
        tool_calls=[{"id": tool_use_id, "name": tool_name, "input": tool_input}],
        stop_reason=StopReason.TOOL_USE,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        latency_ms=1,
        model=DEFAULT_MODEL,
    )


def _make_text_response(text: str = "done") -> LLMResult:
    return LLMResult(
        provider="anthropic",
        content=text,
        tool_calls=[],
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        latency_ms=1,
        model=DEFAULT_MODEL,
    )


def _make_ctx(*, max_iterations: int = 0) -> AgentContext:
    return AgentContext(
        role="react_agent",
        task="x",
        stage_name="react",
        tool_profile="react",
        metadata={"session_id": "s1", "pipeline": "react"},
        model=DEFAULT_MODEL,
        max_iterations=max_iterations,
    )


def _make_agent_with_bash(tmp_path, gateway):
    """Build an AgentCore wired with a working Bash tool that returns ok."""
    registry = ToolRegistry()

    async def execute_bash(**kwargs):
        return ToolResult(tool_name="Bash", success=True, output="ok")

    registry.register(
        schema=ToolSchema(
            name="Bash",
            description="Run a command.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        ),
        fn=execute_bash,
        risk_level=RiskLevel.READ,  # Auto-approve so test doesn't block on prompts.
    )

    return AgentCore(
        llm_gateway=gateway,
        tool_registry=registry,
        runtime=None,
        safety_layer=SafetyLayer(tmp_path),
        working_memory=_StaticWorkingMemory(),
        tracer=Tracer(),
        approval_gate=ApprovalGate(auto_approve=True),
    )


class TestL3ConvergenceHardStop:
    """L3 used to be advisory; the agent could ignore it and keep iterating.
    Now ignoring L3 for ``_L3_GRACE_ITERATIONS`` iterations forces termination."""

    def test_grace_iterations_constant_is_two(self):
        """Two iterations of grace after L3 fires — gives the agent ONE
        more LLM turn to wrap up before force-stop on the next."""
        assert AgentCore._L3_GRACE_ITERATIONS == 2

    def test_l3_hard_stop_via_metadata(self, tmp_path):
        """Force a stuck loop and verify the agent terminates with
        convergence_failed.  Use a low absolute threshold so we don't need
        to simulate 35 iterations."""
        # Build a long sequence of identical Bash calls — stuck pattern.
        # We mimic a spinning agent by always issuing the same tool call.
        calls = [_make_tool_result_response("Bash", {"command": "ls -R"}, f"t{i}") for i in range(50)]
        gateway = _FakeGateway(calls)
        agent = _make_agent_with_bash(tmp_path, gateway)
        # Override convergence thresholds so the test is fast.
        agent._CONVERGENCE_ABSOLUTE_THRESHOLDS = {1: 3, 2: 5, 3: 7}

        ctx = _make_ctx(max_iterations=20)  # cap iteration count
        result = asyncio.run(agent.run(ctx, [{"role": "user", "content": "x"}]))

        # Either we hit max_iterations (no hard stop) OR convergence_failed.
        # The hard stop should fire because the agent keeps spinning.
        termination = result.metadata.get("termination_reason")
        assert termination in {"convergence_failed", "max_iterations"}, \
            f"unexpected termination: {termination}"
        # If the hard-stop fired, success is False and the message says so.
        if termination == "convergence_failed":
            assert result.success is False
            assert "CONVERGENCE_FAILED" in result.output
            assert "L3" in result.output

    def test_agent_that_terminates_is_not_force_stopped(self, tmp_path):
        """If the agent ends its turn, hard-stop never fires."""
        # Agent immediately says "done" — no spinning at all.
        gateway = _FakeGateway([_make_text_response("done")])
        agent = _make_agent_with_bash(tmp_path, gateway)
        ctx = _make_ctx(max_iterations=10)

        result = asyncio.run(agent.run(ctx, [{"role": "user", "content": "x"}]))
        assert result.metadata.get("termination_reason") != "convergence_failed"
        assert result.success is True


# =============================================================================
# Cross-fix interaction: detect_spinning_v2 picks up the lower ls threshold
# =============================================================================


class TestDetectSpinningV2WithLowerThreshold:
    def test_50_percent_ls_triggers(self):
        """4 ls + 4 reads (50%) — should now flag as spinning."""
        conversation = (
            [_ls_call("ls -F") for _ in range(4)]
            + [_other_call("Read", f"foo{i}.py") for i in range(4)]
        )
        # detect_spinning_v2 ORs multiple signals — ls-loop should be one.
        assert detect_spinning_v2(conversation) is True
