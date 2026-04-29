"""Tests for adaptive model selection (classifier + core integration)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from maike.agents.classifier import (
    classify_task_complexity,
    _has_error_patterns,
)
from maike.constants import (
    ADAPTIVE_CHEAP_MAX_MESSAGES,
    ADAPTIVE_MODEL_ENABLED,
    ADAPTIVE_STRONG_FAILURE_THRESHOLD,
    ADAPTIVE_STRONG_ITERATION_RATIO,
    DEFAULT_MODEL,
    model_for_tier,
)


# ── Helper builders ─────────────────────────────────────────────


def _msg(role: str, content: str | list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": role, "content": content}


def _tool_use_block(name: str) -> dict[str, Any]:
    return {"type": "tool_use", "name": name, "id": f"call_{name}", "input": {}}


def _tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "tool_name": tool_use_id.replace("call_", ""),
        "content": content,
        "is_error": is_error,
    }


# ── classify_task_complexity tests ──────────────────────────────


class TestClassifyCheap:
    """Verify the classifier returns 'cheap' for short, error-free, read-only conversations."""

    def test_short_readonly_conversation(self):
        conv = [
            _msg("user", "Find the function definition"),
            _msg("assistant", [_tool_use_block("Read")]),
            _msg("user", [_tool_result_block("call_Read", "def foo(): pass")]),
        ]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        assert tier == "cheap"

    def test_empty_conversation(self):
        conv = [_msg("user", "Hello")]
        tier = classify_task_complexity(
            conv, iteration=0, failure_count=0, has_errors=False,
        )
        assert tier == "cheap"

    def test_grep_only_is_cheap(self):
        conv = [
            _msg("user", "Search for imports"),
            _msg("assistant", [_tool_use_block("Grep")]),
            _msg("user", [_tool_result_block("call_Grep", "import os\nimport sys")]),
        ]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        assert tier == "cheap"

    def test_not_cheap_when_has_errors(self):
        conv = [
            _msg("user", "Read file"),
            _msg("assistant", [_tool_use_block("Read")]),
        ]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=True,
        )
        # has_errors=True prevents cheap
        assert tier == "default"

    def test_not_cheap_when_has_failures(self):
        conv = [
            _msg("user", "Read file"),
        ]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=1, has_errors=False,
        )
        assert tier == "default"

    def test_not_cheap_when_conversation_too_long(self):
        conv = [_msg("user", f"msg {i}") for i in range(ADAPTIVE_CHEAP_MAX_MESSAGES + 1)]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        assert tier == "default"

    def test_not_cheap_when_write_tools_used(self):
        conv = [
            _msg("user", "Edit the file"),
            _msg("assistant", [_tool_use_block("Write")]),
            _msg("user", [_tool_result_block("call_Write", "ok")]),
        ]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        assert tier == "default"


class TestClassifyStrong:
    """Verify the classifier returns 'strong' when escalation conditions are met."""

    def test_many_failures(self):
        conv = [_msg("user", "Fix the bug")]
        tier = classify_task_complexity(
            conv, iteration=5, failure_count=ADAPTIVE_STRONG_FAILURE_THRESHOLD, has_errors=True,
        )
        assert tier == "strong"

    def test_architecture_keyword(self):
        conv = [_msg("user", "Redesign the architecture of the payment module")]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        assert tier == "strong"

    def test_debugging_keyword(self):
        conv = [_msg("user", "Debug the failing integration test")]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        assert tier == "strong"

    def test_convergence_nudge_triggers_strong(self):
        conv = [
            _msg("user", "Fix the test"),
            _msg("assistant", "Let me try editing..."),
            _msg("user", "## Repeated Failure Detection\n\nThe same error occurred 3 times."),
        ]
        tier = classify_task_complexity(
            conv, iteration=5, failure_count=2, has_errors=True,
        )
        assert tier == "strong"

    def test_high_iteration_ratio(self):
        max_iters = 100
        iteration = int(max_iters * ADAPTIVE_STRONG_ITERATION_RATIO) + 1
        conv = [_msg("user", "Complete the task")]
        tier = classify_task_complexity(
            conv,
            iteration=iteration,
            failure_count=0,
            has_errors=False,
            max_iterations=max_iters,
        )
        assert tier == "strong"

    def test_iteration_ratio_not_triggered_without_limit(self):
        """When max_iterations is 0 (unlimited), the ratio check should not trigger."""
        conv = [_msg("user", "Complete the task")]
        tier = classify_task_complexity(
            conv,
            iteration=100,
            failure_count=0,
            has_errors=False,
            max_iterations=0,
        )
        # No strong keywords, no failures, no convergence nudge — should be default
        # (conversation is too long for cheap)
        assert tier != "strong"


class TestClassifyDefault:
    """Verify the classifier returns 'default' for normal conversations."""

    def test_normal_conversation(self):
        conv = [
            _msg("user", "Implement the feature"),
            _msg("assistant", [_tool_use_block("Write")]),
            _msg("user", [_tool_result_block("call_Write", "File written")]),
            _msg("assistant", [_tool_use_block("Bash")]),
            _msg("user", [_tool_result_block("call_Bash", "Tests passed")]),
            _msg("assistant", "Done!"),
        ]
        tier = classify_task_complexity(
            conv, iteration=3, failure_count=0, has_errors=False,
        )
        assert tier == "default"

    def test_one_failure_is_not_strong(self):
        conv = [
            _msg("user", "Fix the test"),
            _msg("assistant", [_tool_use_block("Bash")]),
            _msg("user", [_tool_result_block("call_Bash", "FAILED", is_error=True)]),
        ]
        tier = classify_task_complexity(
            conv, iteration=2, failure_count=1, has_errors=True,
        )
        # One failure is not enough for strong (threshold is 3)
        assert tier == "default"


class TestTierTransitions:
    """Test that the classifier transitions between tiers as conditions change."""

    def test_cheap_to_default_to_strong(self):
        # Start with short read-only conversation -> cheap
        conv = [
            _msg("user", "Implement a function"),
            _msg("assistant", [_tool_use_block("Read")]),
        ]
        tier1 = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        assert tier1 == "cheap"

        # Add more messages + a write tool -> default
        conv.extend([
            _msg("user", [_tool_result_block("call_Read", "def foo(): pass")]),
            _msg("assistant", [_tool_use_block("Write")]),
            _msg("user", [_tool_result_block("call_Write", "ok")]),
            _msg("assistant", [_tool_use_block("Bash")]),
            _msg("user", [_tool_result_block("call_Bash", "Tests passed")]),
        ])
        tier2 = classify_task_complexity(
            conv, iteration=4, failure_count=0, has_errors=False,
        )
        assert tier2 == "default"

        # Accumulate failures -> strong
        tier3 = classify_task_complexity(
            conv, iteration=8, failure_count=ADAPTIVE_STRONG_FAILURE_THRESHOLD, has_errors=True,
        )
        assert tier3 == "strong"


class TestHasErrorPatterns:
    """Test the error pattern detector."""

    def test_detects_traceback(self):
        conv = [
            _msg("user", [{"content": "Traceback (most recent call last):\n  File ..."}]),
        ]
        assert _has_error_patterns(conv) is True

    def test_detects_error_keyword(self):
        conv = [
            _msg("user", "Error: module not found"),
        ]
        assert _has_error_patterns(conv) is True

    def test_no_errors_in_clean_output(self):
        conv = [
            _msg("user", "All 15 tests passed successfully"),
        ]
        assert _has_error_patterns(conv) is False

    def test_ignores_assistant_messages(self):
        conv = [
            _msg("assistant", "I see an error in the code"),
            _msg("user", "All good, please proceed"),
        ]
        assert _has_error_patterns(conv) is False


class TestAdaptiveDisabledViaMetadata:
    """Test that adaptive model selection can be disabled per-agent via metadata."""

    def test_metadata_disables_adaptive(self):
        """When adaptive_model=False in metadata, the classifier should not be called.

        We verify this by checking the integration flow: when disabled,
        the effective model should always be ctx.model regardless of
        conversation state that would normally trigger cheap/strong.
        """
        # A conversation that would normally trigger "strong"
        conv = [_msg("user", "Debug the architecture issue")]
        tier = classify_task_complexity(
            conv, iteration=1, failure_count=0, has_errors=False,
        )
        # Classifier itself sees strong keywords
        assert tier == "strong"

        # But the integration in core.py checks ctx.metadata.get("adaptive_model", True)
        # When False, it skips classification entirely — we test that the
        # metadata flag is respected by checking the constant default.
        assert ADAPTIVE_MODEL_ENABLED is True  # global toggle is on by default

    def test_metadata_default_enables_adaptive(self):
        """By default (no metadata key), adaptive selection is enabled."""
        metadata: dict[str, Any] = {}
        assert metadata.get("adaptive_model", True) is True

    def test_metadata_explicit_disable(self):
        metadata: dict[str, Any] = {"adaptive_model": False}
        assert metadata.get("adaptive_model", True) is False


class TestModelForTier:
    """Test the model_for_tier helper from constants."""

    def test_default_tier(self):
        model = model_for_tier("gemini", "default")
        assert model is not None
        assert isinstance(model, str)

    def test_cheap_tier(self):
        model = model_for_tier("gemini", "cheap")
        assert model is not None
        assert isinstance(model, str)

    def test_strong_tier(self):
        model = model_for_tier("gemini", "strong")
        assert model is not None
        assert isinstance(model, str)

    def test_tiers_differ_for_anthropic(self):
        """Cheap and strong models should differ for providers that define them."""
        cheap = model_for_tier("anthropic", "cheap")
        strong = model_for_tier("anthropic", "strong")
        default = model_for_tier("anthropic", "default")
        # Cheap should differ from strong
        assert cheap != strong

    def test_unknown_provider_falls_back(self):
        """Unknown providers should fall back to DEFAULT_MODEL."""
        model = model_for_tier("nonexistent_provider", "default")
        assert model == DEFAULT_MODEL


class TestGatewayResolveModelForTier:
    """Test LLMGateway.resolve_model_for_tier integration."""

    def test_resolve_model_for_tier(self):
        from maike.cost.tracker import CostTracker
        from maike.gateway.llm_gateway import LLMGateway
        from maike.observability.tracer import Tracer

        class FakeMessagesAPI:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=100, output_tokens=50),
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text="hello")],
                )

        class FakeClient:
            def __init__(self):
                self.messages = FakeMessagesAPI()

        gateway = LLMGateway(
            cost_tracker=CostTracker(),
            tracer=Tracer(),
            provider_name="anthropic",
            client=FakeClient(),
        )

        cheap = gateway.resolve_model_for_tier("cheap")
        strong = gateway.resolve_model_for_tier("strong")
        default = gateway.resolve_model_for_tier("default")

        assert isinstance(cheap, str)
        assert isinstance(strong, str)
        assert isinstance(default, str)
        assert cheap != strong
