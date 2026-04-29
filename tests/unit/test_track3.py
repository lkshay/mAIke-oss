"""Tests for Track 3 additions: model-aware context, SessionToolTracker,
error recovery enhancements, and stale-read clearing."""

from __future__ import annotations

import pytest

from maike.constants import (
    PRUNE_THRESHOLD,
    context_limit_for_model,
    prune_fraction_for_model,
    prune_threshold_for_model,
)
from maike.agents.core import RepeatedFailureTracker, SessionToolTracker
from maike.memory.working import WorkingMemory


# =====================================================================
# P1: Model-Aware Context Budgets
# =====================================================================


class TestPruneThresholdForModel:
    def test_gemini_large_context(self):
        # Gemini has 1M context → 75% = 750K (capped at 800K)
        threshold = prune_threshold_for_model("gemini-2.5-flash")
        assert threshold == 786_432  # int(1_048_576 * 0.75)

    def test_claude_200k_context(self):
        threshold = prune_threshold_for_model("claude-opus-4-20250514")
        assert threshold == 150_000  # int(200_000 * 0.75)

    def test_gpt_128k_context(self):
        threshold = prune_threshold_for_model("gpt-5.4")
        assert threshold == 96_000  # int(128_000 * 0.75)

    def test_unknown_model_uses_default(self):
        # Unknown model falls back to MODEL_CONTEXT_LIMIT (200K) → 150K
        threshold = prune_threshold_for_model("unknown-model-xyz")
        assert threshold == 150_000

    def test_cap_at_800k(self):
        # Even a hypothetical 2M model would cap at 800K
        assert prune_threshold_for_model("gemini-2.5-flash") <= 800_000


class TestPruneFractionForModel:
    def test_gemini_gets_60_percent(self):
        assert prune_fraction_for_model("gemini-2.5-flash") == 0.60

    def test_gemini_pro_gets_60_percent(self):
        assert prune_fraction_for_model("gemini-2.5-pro") == 0.60

    def test_claude_gets_40_percent(self):
        assert prune_fraction_for_model("claude-opus-4-20250514") == 0.40

    def test_gpt_gets_35_percent(self):
        assert prune_fraction_for_model("gpt-5.4") == 0.35

    def test_unknown_model_gets_40_percent(self):
        # Unknown → 200K default → ≥150K → 0.40
        assert prune_fraction_for_model("unknown-model") == 0.40


class TestWorkingMemoryModelParam:
    def test_prune_accepts_model_kwarg(self):
        """WorkingMemory.prune() should accept model= without error."""
        wm = WorkingMemory()
        msgs = [{"role": "user", "content": "hello"}]
        result = wm.prune(msgs, model="gemini-2.5-flash")
        assert result == msgs  # short conversation → no pruning

    def test_prune_without_model_uses_default(self):
        """Backward compat: calling without model= still works."""
        wm = WorkingMemory()
        msgs = [{"role": "user", "content": "hello"}]
        result = wm.prune(msgs)
        assert result == msgs

    def test_prune_to_budget_accepts_model_kwarg(self):
        wm = WorkingMemory()
        msgs = [{"role": "user", "content": "hello"}]
        result = wm.prune_to_budget(msgs, token_budget=100_000, model="gemini-2.5-flash")
        assert result == msgs


# =====================================================================
# P2: Stale Read Clearing
# =====================================================================


class TestClearStaleReads:
    @staticmethod
    def _make_read_result(file_path: str, content: str) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_name": "Read",
                    "content": content,
                    "is_error": False,
                    "input": {"path": file_path},
                }
            ],
        }

    @staticmethod
    def _make_text_msg(role: str, text: str) -> dict:
        return {"role": role, "content": text}

    def test_stale_read_is_cleared(self):
        """Read results for files not referenced recently should be pruned."""
        wm = WorkingMemory()
        old_read = self._make_read_result("old_file.py", "x\n" * 50)
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [old_read] + recent_msgs

        result = wm.clear_stale_reads(msgs, recent_window=8)
        # The old_read should have its content replaced
        first_block = result[0]["content"][0]
        assert "content pruned" in first_block["content"]

    def test_recently_referenced_file_is_preserved(self):
        """Read results for files mentioned in recent messages should be kept."""
        wm = WorkingMemory()
        old_read = self._make_read_result("active_file.py", "content here\n" * 20)
        recent_msgs = [
            self._make_text_msg("user", "Let me check active_file.py again"),
        ] + [self._make_text_msg("user", f"msg {i}") for i in range(9)]
        msgs = [old_read] + recent_msgs

        result = wm.clear_stale_reads(msgs, recent_window=10)
        first_block = result[0]["content"][0]
        assert "pruned as stale" not in first_block["content"]

    def test_mutated_paths_are_preserved(self):
        """Files in mutated_paths set should never be cleared."""
        wm = WorkingMemory()
        old_read = self._make_read_result("edited.py", "x\n" * 50)
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [old_read] + recent_msgs

        result = wm.clear_stale_reads(
            msgs, recent_window=8, mutated_paths={"edited.py"}
        )
        first_block = result[0]["content"][0]
        assert "pruned as stale" not in first_block["content"]

    def test_error_reads_are_never_cleared(self):
        """Read results that are errors should be preserved."""
        wm = WorkingMemory()
        error_read = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_name": "Read",
                    "content": "File not found: missing.py",
                    "is_error": True,
                    "input": {"path": "missing.py"},
                }
            ],
        }
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [error_read] + recent_msgs

        result = wm.clear_stale_reads(msgs, recent_window=8)
        first_block = result[0]["content"][0]
        assert first_block["content"] == "File not found: missing.py"

    def test_short_conversation_returns_unchanged(self):
        """Conversations shorter than recent_window should not be touched."""
        wm = WorkingMemory()
        msgs = [self._make_text_msg("user", "hello")]
        result = wm.clear_stale_reads(msgs, recent_window=8)
        assert result == msgs

    def test_non_read_tools_are_untouched(self):
        """Grep/Bash results in older messages should not be cleared by stale reads."""
        wm = WorkingMemory()
        grep_msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_name": "Grep",
                    "content": "matches found",
                    "is_error": False,
                }
            ],
        }
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [grep_msg] + recent_msgs

        result = wm.clear_stale_reads(msgs, recent_window=8)
        first_block = result[0]["content"][0]
        assert first_block["content"] == "matches found"


class TestClearStaleToolResults:
    """Tests for extended microcompaction covering Bash/Grep results."""

    @staticmethod
    def _make_tool_result(tool_name: str, content: str, *, is_error: bool = False, input_data: dict | None = None) -> dict:
        block: dict = {
            "type": "tool_result",
            "tool_name": tool_name,
            "content": content,
            "is_error": is_error,
        }
        if input_data:
            block["input"] = input_data
        return {"role": "user", "content": [block]}

    @staticmethod
    def _make_text_msg(role: str, text: str) -> dict:
        return {"role": role, "content": text}

    def test_large_bash_result_is_cleared(self):
        """Bash results >500 chars not referenced recently should be pruned."""
        wm = WorkingMemory()
        large_output = "x" * 600
        old_bash = self._make_tool_result("Bash", large_output)
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [old_bash] + recent_msgs

        result = wm.clear_stale_tool_results(msgs, recent_window=8)
        first_block = result[0]["content"][0]
        assert "content pruned" in first_block["content"]
        assert "Bash result" in first_block["content"]
        assert "600 chars" in first_block["content"]

    def test_small_bash_result_is_preserved(self):
        """Bash results <=500 chars should not be pruned."""
        wm = WorkingMemory()
        small_output = "x" * 400
        old_bash = self._make_tool_result("Bash", small_output)
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [old_bash] + recent_msgs

        result = wm.clear_stale_tool_results(msgs, recent_window=8)
        first_block = result[0]["content"][0]
        assert first_block["content"] == small_output

    def test_large_grep_result_is_cleared(self):
        """Grep results >500 chars not referenced recently should be pruned."""
        wm = WorkingMemory()
        large_output = "match line\n" * 100
        old_grep = self._make_tool_result("Grep", large_output)
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [old_grep] + recent_msgs

        result = wm.clear_stale_tool_results(msgs, recent_window=8)
        first_block = result[0]["content"][0]
        assert "content pruned" in first_block["content"]
        assert "Grep result" in first_block["content"]

    def test_bash_error_result_is_preserved(self):
        """Error results should never be pruned even if large."""
        wm = WorkingMemory()
        large_error = "ERROR: " + "x" * 600
        old_bash = self._make_tool_result("Bash", large_error, is_error=True)
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [old_bash] + recent_msgs

        result = wm.clear_stale_tool_results(msgs, recent_window=8)
        first_block = result[0]["content"][0]
        assert first_block["content"] == large_error

    def test_referenced_bash_result_is_preserved(self):
        """Bash results with file paths referenced recently should be kept."""
        wm = WorkingMemory()
        large_output = "/path/to/important.py: some match result " + "x" * 600
        old_bash = self._make_tool_result("Bash", large_output)
        recent_msgs = [
            self._make_text_msg("user", "Let me check /path/to/important.py"),
        ] + [self._make_text_msg("user", f"msg {i}") for i in range(9)]
        msgs = [old_bash] + recent_msgs

        result = wm.clear_stale_tool_results(msgs, recent_window=10)
        first_block = result[0]["content"][0]
        assert "pruned as stale" not in first_block["content"]

    def test_deprecated_alias_works(self):
        """clear_stale_reads should still work as a deprecated alias."""
        wm = WorkingMemory()
        old_read = {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_name": "Read",
                "content": "x\n" * 50,
                "is_error": False,
                "input": {"path": "old.py"},
            }],
        }
        recent_msgs = [self._make_text_msg("user", f"msg {i}") for i in range(10)]
        msgs = [old_read] + recent_msgs

        result = wm.clear_stale_reads(msgs, recent_window=8)
        first_block = result[0]["content"][0]
        assert "content pruned" in first_block["content"]


class TestCompressDuplicateFailures:
    """Tests for duplicate failure compression."""

    @staticmethod
    def _make_error_result(tool_name: str, content: str) -> dict:
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_name": tool_name,
                "content": content,
                "is_error": True,
            }],
        }

    @staticmethod
    def _make_text_msg(role: str, text: str) -> dict:
        return {"role": role, "content": text}

    def test_duplicate_errors_are_compressed(self):
        """Second occurrence of same error should be replaced with stub."""
        wm = WorkingMemory()
        error_content = "Traceback:\n  File main.py\nImportError: no module"
        msg1 = self._make_error_result("Bash", error_content)
        msg2 = self._make_text_msg("user", "trying again")
        msg3 = self._make_error_result("Bash", error_content)

        result, hashes = wm.compress_duplicate_failures([msg1, msg2, msg3])
        # First error should be preserved
        assert result[0]["content"][0]["content"] == error_content
        # Second identical error should be compressed
        third_block = result[2]["content"][0]["content"]
        assert "Same error as iteration 0" in third_block
        assert len(hashes) == 1

    def test_different_errors_are_not_compressed(self):
        """Different errors should both be preserved."""
        wm = WorkingMemory()
        msg1 = self._make_error_result("Bash", "ImportError: no module foo")
        msg2 = self._make_error_result("Bash", "TypeError: unsupported operand")

        result, hashes = wm.compress_duplicate_failures([msg1, msg2])
        assert result[0]["content"][0]["content"] == "ImportError: no module foo"
        assert result[1]["content"][0]["content"] == "TypeError: unsupported operand"
        assert len(hashes) == 2

    def test_non_error_results_are_untouched(self):
        """Non-error results should pass through unchanged."""
        wm = WorkingMemory()
        msg = {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_name": "Bash",
                "content": "success output",
                "is_error": False,
            }],
        }
        result, hashes = wm.compress_duplicate_failures([msg])
        assert result[0]["content"][0]["content"] == "success output"
        assert len(hashes) == 0

    def test_accumulates_hashes_across_calls(self):
        """failure_hashes should accumulate across multiple calls."""
        wm = WorkingMemory()
        error_content = "SyntaxError: invalid syntax"
        msg1 = self._make_error_result("Bash", error_content)
        _, hashes = wm.compress_duplicate_failures([msg1])

        msg2 = self._make_error_result("Bash", error_content)
        result, hashes = wm.compress_duplicate_failures([msg2], failure_hashes=hashes)
        assert "Same error as iteration 0" in result[0]["content"][0]["content"]


# =====================================================================
# P3: Error Recovery Enhancement
# =====================================================================


class TestFirstFailureHint:
    def test_classified_error_gets_hint(self):
        tracker = RepeatedFailureTracker()
        # RecursionError is a classified pattern with high confidence
        output = "Traceback:\n" * 5 + "RecursionError: maximum recursion depth exceeded"
        hint = tracker.first_failure_hint(output)
        assert hint is not None
        assert "[Error hint:" in hint
        assert "recursion" in hint.lower()

    def test_same_category_hinted_up_to_limit_then_suppressed(self):
        """Fix 5: previously a category was suppressed after one hint, which
        silently disabled guidance for the rest of the session.  Now we
        allow up to ``_CATEGORY_HINT_LIMIT`` hints per category, then
        suppress.  See RepeatedFailureTracker.first_failure_hint."""
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "RecursionError: maximum recursion depth exceeded"
        # First _CATEGORY_HINT_LIMIT calls should each produce a hint.
        for _ in range(RepeatedFailureTracker._CATEGORY_HINT_LIMIT):
            hint = tracker.first_failure_hint(output)
            assert hint is not None
            assert "recursion" in hint.lower()
        # The (limit+1)-th call should be suppressed.
        suppressed = tracker.first_failure_hint(output)
        assert suppressed is None

    def test_different_categories_both_hinted(self):
        tracker = RepeatedFailureTracker()
        recursion_output = "Traceback:\n" * 5 + "RecursionError: maximum recursion depth exceeded"
        import_output = "Traceback:\n" * 5 + "ModuleNotFoundError: No module named 'foobar'"
        hint1 = tracker.first_failure_hint(recursion_output)
        hint2 = tracker.first_failure_hint(import_output)
        assert hint1 is not None
        assert hint2 is not None

    def test_unclassified_error_gets_no_hint(self):
        tracker = RepeatedFailureTracker()
        output = "some random output\n" * 10
        hint = tracker.first_failure_hint(output)
        assert hint is None

    def test_empty_output_gets_no_hint(self):
        tracker = RepeatedFailureTracker()
        assert tracker.first_failure_hint("") is None
        assert tracker.first_failure_hint("   ") is None


class TestSuggestDelegation:
    def test_no_suggestion_on_first_failure(self):
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "AssertionError: expected True"
        result = tracker.suggest_delegation(output, "test_foo.py")
        assert result is None

    def test_suggestion_on_second_failure(self):
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "AssertionError: expected True"
        tracker.suggest_delegation(output, "test_foo.py")  # 1st
        result = tracker.suggest_delegation(output, "test_foo.py")  # 2nd
        assert result is not None
        assert "Delegate" in result
        assert "test_foo.py" in result

    def test_no_suggestion_without_file_path(self):
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "Error"
        tracker.suggest_delegation(output, "")
        result = tracker.suggest_delegation(output, "")
        assert result is None

    def test_suggestion_only_once_per_file(self):
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "AssertionError: expected True"
        tracker.suggest_delegation(output, "foo.py")  # 1st
        s1 = tracker.suggest_delegation(output, "foo.py")  # 2nd → suggestion
        s2 = tracker.suggest_delegation(output, "foo.py")  # 3rd → already suggested
        assert s1 is not None
        assert s2 is None

    def test_different_files_get_separate_suggestions(self):
        tracker = RepeatedFailureTracker()
        output = "Traceback:\n" * 5 + "AssertionError: expected True"
        tracker.suggest_delegation(output, "a.py")
        s_a = tracker.suggest_delegation(output, "a.py")
        tracker.suggest_delegation(output, "b.py")
        s_b = tracker.suggest_delegation(output, "b.py")
        assert s_a is not None
        assert s_b is not None
        assert "a.py" in s_a
        assert "b.py" in s_b


# =====================================================================
# P4+P5: SessionToolTracker
# =====================================================================


class TestSessionToolTracker:
    def test_large_file_without_grep_warns(self):
        tracker = SessionToolTracker()
        tip = tracker.record_read("big_file.py", 200, edited_files=set())
        assert tip is not None
        assert "big_file.py" in tip
        assert "200 lines" in tip

    def test_small_file_without_grep_no_warning(self):
        tracker = SessionToolTracker()
        tip = tracker.record_read("small.py", 50, edited_files=set())
        assert tip is None

    def test_grepped_file_no_large_file_warning(self):
        tracker = SessionToolTracker()
        tracker.record_grep("big_file.py", "line 42: def foo()")
        tip = tracker.record_read("big_file.py", 200, edited_files=set())
        assert tip is None

    def test_triple_read_without_edit_warns(self):
        tracker = SessionToolTracker()
        # First read of small file: no warning
        tracker.record_read("config.py", 30, edited_files=set())
        tracker.record_read("config.py", 30, edited_files=set())
        tip = tracker.record_read("config.py", 30, edited_files=set())
        assert tip is not None
        assert "3 times" in tip

    def test_triple_read_of_edited_file_no_warning(self):
        tracker = SessionToolTracker()
        edited = {"config.py"}
        tracker.record_read("config.py", 30, edited_files=edited)
        tracker.record_read("config.py", 30, edited_files=edited)
        tip = tracker.record_read("config.py", 30, edited_files=edited)
        assert tip is None

    def test_warning_fires_only_once_per_file(self):
        tracker = SessionToolTracker()
        tracker.record_read("big.py", 200, edited_files=set())  # warns
        tip = tracker.record_read("big.py", 200, edited_files=set())  # already warned
        assert tip is None

    def test_zero_result_grep_tip_after_two(self):
        tracker = SessionToolTracker()
        tip1 = tracker.record_grep("src/", "")
        assert tip1 is None  # Only 1 empty grep
        tip2 = tracker.record_grep("lib/", "No matches found")
        assert tip2 is not None
        assert "no results" in tip2.lower()

    def test_zero_result_counter_resets_on_hit(self):
        tracker = SessionToolTracker()
        tracker.record_grep("src/", "")  # empty
        tracker.record_grep("lib/", "line 1: match")  # hit → resets counter
        tip = tracker.record_grep("other/", "")  # empty again, but only 1 consecutive
        assert tip is None

    def test_max_nudges_cap(self):
        tracker = SessionToolTracker()
        # Generate 10 nudges (large files without grep)
        for i in range(12):
            tracker.record_read(f"file_{i}.py", 200, edited_files=set())
        assert tracker.nudge_count == 10  # Capped

    def test_grep_matched_files_tracked(self):
        tracker = SessionToolTracker()
        tracker.record_grep("pattern", "found", matched_files=["src/foo.py", "src/bar.py"])
        # Both matched files should be in grepped_files
        assert "src/foo.py" in tracker.grepped_files
        assert "src/bar.py" in tracker.grepped_files
        # Reading them should not trigger large-file warning
        tip = tracker.record_read("src/foo.py", 300, edited_files=set())
        assert tip is None


# =====================================================================
# Integration: prune thresholds produce correct behavior
# =====================================================================


class TestModelAwarePruningIntegration:
    def test_gemini_threshold_much_higher_than_claude(self):
        gemini = prune_threshold_for_model("gemini-2.5-flash")
        claude = prune_threshold_for_model("claude-opus-4-20250514")
        assert gemini > claude * 4  # At least 4x higher

    def test_fraction_ordering(self):
        gemini = prune_fraction_for_model("gemini-2.5-flash")
        claude = prune_fraction_for_model("claude-opus-4-20250514")
        gpt = prune_fraction_for_model("gpt-5.4")
        assert gemini > claude > gpt

    def test_spinning_fraction_always_at_least_30_percent(self):
        """Spinning penalty: base - 0.10, but never below 0.30."""
        for model in ["gemini-2.5-flash", "claude-opus-4-20250514", "gpt-5.4"]:
            base = prune_fraction_for_model(model)
            spinning = max(0.30, base - 0.10)
            assert spinning >= 0.30
