"""Tests for the agentic metrics analysis engine."""

from maike.eval.agentic_metrics import (
    _count_fix_test_fix_cycles,
    compute_error_recovery,
    detect_wasted_calls,
    verify_file_changes,
)
from maike.eval.contracts import AgenticMetrics
from maike.eval.metrics import agentic_eval_score


def _tc(name: str, inp: dict | None = None, success: bool = True, output: str = "") -> dict:
    """Build a minimal tool call dict for testing."""
    return {
        "resolved_tool_name": name,
        "name": name,
        "input": inp or {},
        "success": success,
        "is_error": not success,
        "output": output,
        "content": output,
    }


class TestFixTestFixCycles:
    def test_no_cycles(self):
        calls = [_tc("Read", {"path": "a.py"}), _tc("Write", {"path": "a.py"})]
        assert _count_fix_test_fix_cycles(calls) == 0

    def test_single_cycle(self):
        calls = [
            _tc("Bash", {"cmd": "pytest"}, success=False, output="FAILED"),
            _tc("Edit", {"path": "a.py"}),
            _tc("Bash", {"cmd": "pytest"}, success=True, output="passed"),
        ]
        assert _count_fix_test_fix_cycles(calls) == 1

    def test_two_successful_cycles(self):
        calls = [
            _tc("Bash", {"cmd": "pytest"}, success=False, output="FAILED"),
            _tc("Edit", {"path": "a.py"}),
            _tc("Bash", {"cmd": "pytest"}, success=True, output="passed"),
            _tc("Bash", {"cmd": "pytest test_b"}, success=False, output="FAILED"),
            _tc("Write", {"path": "b.py"}),
            _tc("Bash", {"cmd": "pytest test_b"}, success=True, output="passed"),
        ]
        assert _count_fix_test_fix_cycles(calls) == 2

    def test_empty(self):
        assert _count_fix_test_fix_cycles([]) == 0


class TestWastedCalls:
    def test_no_waste(self):
        calls = [
            _tc("Read", {"path": "a.py"}),
            _tc("Edit", {"path": "a.py"}),
        ]
        total, reads, greps = detect_wasted_calls(calls)
        assert total == 2
        assert reads == 0

    def test_wasted_read(self):
        calls = [
            _tc("Read", {"path": "a.py"}),
            _tc("Read", {"path": "b.py"}),
            _tc("Edit", {"path": "a.py"}),
        ]
        total, reads, greps = detect_wasted_calls(calls)
        assert reads == 1  # b.py was read but never edited

    def test_empty(self):
        assert detect_wasted_calls([]) == (0, 0, 0)


class TestErrorRecovery:
    def test_successful_recovery(self):
        calls = [
            _tc("Bash", {"cmd": "pytest"}, success=False, output="FAILED"),
            _tc("Edit", {"path": "fix.py"}),
            _tc("Bash", {"cmd": "pytest"}, success=True, output="passed"),
        ]
        corrections, unrecovered, rate = compute_error_recovery(calls)
        assert corrections == 1
        assert unrecovered == 0
        assert rate == 1.0

    def test_unrecovered(self):
        calls = [
            _tc("Bash", {"cmd": "pytest"}, success=False, output="FAILED"),
            _tc("Edit", {"path": "fix.py"}),
            _tc("Bash", {"cmd": "pytest"}, success=False, output="FAILED"),
        ]
        corrections, unrecovered, rate = compute_error_recovery(calls)
        assert corrections == 0
        assert unrecovered == 1
        assert rate == 0.0

    def test_no_errors_is_perfect(self):
        """No errors at all = perfect run, rate should be 1.0."""
        calls = [
            _tc("Read", {"path": "a.py"}),
            _tc("Edit", {"path": "a.py"}),
        ]
        corrections, unrecovered, rate = compute_error_recovery(calls)
        assert corrections == 0
        assert unrecovered == 0
        assert rate == 1.0  # perfect: no errors to recover from


class TestFileChanges:
    def test_correct_changes(self):
        pre = {"a.py", "b.py"}
        post = {"a.py", "b.py", "c.py"}
        modified, correct, unnecessary, minimality = verify_file_changes(pre, post, ("c.py",))
        assert "c.py" in modified
        assert correct is True
        assert unnecessary == ()
        assert minimality == 1.0

    def test_unnecessary_changes(self):
        pre = {"a.py"}
        post = {"a.py", "b.py", "c.py"}
        modified, correct, unnecessary, minimality = verify_file_changes(pre, post, ("b.py",))
        assert correct is True
        assert "c.py" in unnecessary
        assert minimality < 1.0

    def test_empty_expected(self):
        pre = set()
        post = {"new.py"}
        modified, correct, unnecessary, minimality = verify_file_changes(pre, post, ())
        assert correct is True


class TestAgenticEvalScore:
    def test_perfect_score(self):
        score = agentic_eval_score(
            session_completed=True,
            workspace_verified=True,
            tests_pass=True,
            error_recovery_rate=1.0,
            wasted_call_ratio=0.0,
            change_minimality_score=1.0,
        )
        assert score == 1.0

    def test_zero_score(self):
        score = agentic_eval_score(
            session_completed=False,
            workspace_verified=False,
            tests_pass=False,
            error_recovery_rate=0.0,
            wasted_call_ratio=1.0,
            change_minimality_score=0.0,
        )
        assert score == 0.0

    def test_partial(self):
        score = agentic_eval_score(
            session_completed=True,
            workspace_verified=True,
        )
        assert 0.5 < score < 1.0


class TestAgenticEvalScoreVerdictCap:
    """Verdict-label caps on the final score.

    The cap is never allowed to RAISE the score — only tighten it.  Sessions
    without a verdict label see identical behavior to the original signature.
    """

    def _full_score_inputs(self) -> dict:
        return dict(
            session_completed=True, workspace_verified=True, tests_pass=True,
            error_recovery_rate=1.0, wasted_call_ratio=0.0, change_minimality_score=1.0,
        )

    def test_no_verdict_is_passthrough(self):
        # Baseline: existing callers work unchanged.
        a = agentic_eval_score(**self._full_score_inputs())
        b = agentic_eval_score(verdict_label=None, **self._full_score_inputs())
        assert a == b == 1.0

    def test_cancelled_is_passthrough(self):
        score = agentic_eval_score(verdict_label="cancelled", **self._full_score_inputs())
        assert score == 1.0

    def test_unknown_is_passthrough(self):
        score = agentic_eval_score(verdict_label="unknown", **self._full_score_inputs())
        assert score == 1.0

    def test_satisfied_does_not_lower(self):
        score = agentic_eval_score(verdict_label="satisfied", **self._full_score_inputs())
        assert score == 1.0  # cap is 1.0 — already equal

    def test_partial_caps_at_0_7(self):
        score = agentic_eval_score(verdict_label="partial", **self._full_score_inputs())
        assert score == 0.7

    def test_unproductive_budget_caps_at_0_15(self):
        score = agentic_eval_score(
            verdict_label="unproductive_budget_exhaustion",
            **self._full_score_inputs(),
        )
        assert score == 0.15

    def test_unproductive_loop_caps_at_0_15(self):
        score = agentic_eval_score(
            verdict_label="unproductive_loop",
            **self._full_score_inputs(),
        )
        assert score == 0.15

    def test_cap_does_not_raise_low_scores(self):
        # A genuinely low-signal session with a 'satisfied' verdict should
        # keep its low score — the cap only lowers, never raises.
        # (Default minimality=1.0 and wasted=0.0 give a floor of 0.15,
        # which is below the satisfied cap of 1.0 — still must not rise.)
        baseline = agentic_eval_score(
            session_completed=False, workspace_verified=False,
        )
        with_verdict = agentic_eval_score(
            session_completed=False, workspace_verified=False,
            verdict_label="satisfied",
        )
        assert with_verdict == baseline

    def test_unknown_verdict_label_treated_as_passthrough(self):
        # Defensive: a label outside the known cap set should not crash.
        score = agentic_eval_score(
            verdict_label="bogus_label",
            **self._full_score_inputs(),
        )
        assert score == 1.0
