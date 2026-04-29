"""Tests for edit spiral detection and empty response handling."""

from maike.agents.core import RepeatedFailureTracker


class TestEditFileExtraction:
    """Test _extract_edit_target_file regex."""

    def test_extracts_path_from_not_found(self):
        output = "old_text not found in src/parser.py.\n\nRecovery strategies:\n1. Use Read..."
        result = RepeatedFailureTracker._extract_edit_target_file(output)
        assert result == "src/parser.py"

    def test_extracts_path_from_matches_n_locations(self):
        output = "old_text matches 4 locations in test_lru_cache.py. Provide more surrounding context."
        result = RepeatedFailureTracker._extract_edit_target_file(output)
        assert result == "test_lru_cache.py"

    def test_extracts_deeply_nested_path(self):
        output = "old_text not found in src/core/utils/helpers.py.\n\nSome hint."
        result = RepeatedFailureTracker._extract_edit_target_file(output)
        assert result == "src/core/utils/helpers.py"

    def test_returns_none_for_unrelated_output(self):
        output = "Some random error that does not mention edit"
        result = RepeatedFailureTracker._extract_edit_target_file(output)
        assert result is None


class TestEditSpiralDetection:
    """Test per-file edit failure tracking (varied failures on same file)."""

    def test_varied_edit_failures_on_same_file_triggers_nudge(self):
        """Three different edit failures on the same file should fire a nudge."""
        tracker = RepeatedFailureTracker()

        # Each failure uses a different old_text → different hash → hash-based
        # tracker won't fire.  Per-file tracker should catch it.
        outputs = [
            "old_text not found in app.py.\n\nRecovery strategies:\n1. Use Read...",
            "old_text matches 3 locations in app.py. Provide more surrounding context.",
            "old_text not found in app.py.\n\nDid you mean line 42?\n\nRecovery strategies:\n1. Use Read...",
        ]
        nudge = None
        for output in outputs:
            result = tracker.record("Edit", output, success=False)
            if result is not None:
                nudge = result

        assert nudge is not None, "Expected a nudge after 3 varied edit failures on same file"
        assert "app.py" in nudge
        assert "Edit Spiral" in nudge

    def test_edit_success_resets_per_file_counter(self):
        """A successful edit on a file should reset its failure count."""
        tracker = RepeatedFailureTracker()

        # Two failures
        tracker.record("Edit", "old_text not found in app.py.\nHint A", success=False)
        tracker.record("Edit", "old_text matches 2 locations in app.py. More context.", success=False)

        # Success on same file — the edit tool output includes the file path
        # in the diff.  We extract the file from the success output too.
        tracker.record("Edit", "--- app.py\n+++ app.py\n@@ ...", success=True)

        # Two more failures — should NOT trigger (counter was reset)
        result = None
        result = tracker.record("Edit", "old_text not found in app.py.\nDifferent hint", success=False)
        assert result is None, "Counter should have been reset by success"
        result = tracker.record("Edit", "old_text matches 5 locations in app.py. More context.", success=False)
        assert result is None, "Only 2 failures after reset, threshold is 3"

    def test_different_files_tracked_independently(self):
        """Failures on different files should not accumulate together."""
        tracker = RepeatedFailureTracker()

        tracker.record("Edit", "old_text not found in foo.py.\nHint", success=False)
        tracker.record("Edit", "old_text not found in bar.py.\nHint", success=False)
        result = tracker.record("Edit", "old_text not found in baz.py.\nHint", success=False)

        assert result is None, "Failures on 3 different files should not trigger"

    def test_hash_based_still_fires_at_two_identical(self):
        """Existing behavior: 2 identical edit failures still fire hash-based nudge."""
        tracker = RepeatedFailureTracker()
        output = "old_text not found in app.py.\nExact same hint"

        nudge = None
        for _ in range(2):
            result = tracker.record("Edit", output, success=False)
            if result is not None:
                nudge = result

        assert nudge is not None, "Hash-based tracker should fire at 2 identical failures"
        assert "Edit" in nudge

    def test_nudge_contains_recovery_strategies(self):
        """Edit spiral nudge should include actionable recovery strategies."""
        tracker = RepeatedFailureTracker()
        outputs = [
            "old_text not found in config.py.\nHint 1",
            "old_text matches 2 locations in config.py. More context.",
            "old_text not found in config.py.\nHint 2",
        ]
        nudge = None
        for output in outputs:
            result = tracker.record("Edit", output, success=False)
            if result is not None:
                nudge = result

        assert nudge is not None
        lower = nudge.lower()
        assert "read" in lower, "Should suggest reading the file"
        assert "write" in lower, "Should suggest using Write for small files"
        assert "grep" in lower, "Should suggest using Grep"
