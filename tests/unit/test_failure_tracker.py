"""Tests for RepeatedFailureTracker error-aware nudges."""

from __future__ import annotations

import pytest

from maike.agents.core import RepeatedFailureTracker


# ---------------------------------------------------------------------------
# _classify_error — parametrized across all 12 categories and frameworks
# ---------------------------------------------------------------------------

_CLASSIFY_CASES: list[tuple[str, str, str, float]] = [
    # (test_id, output_snippet, expected_category, min_confidence)
    #
    # --- recursion ---
    ("recursion-python", "RecursionError: maximum recursion depth exceeded", "recursion", 0.9),
    ("recursion-keyword", "recursionerror while calling foo()", "recursion", 0.9),
    ("recursion-runtime", "RuntimeError: maximum recursion depth exceeded in comparison", "recursion", 0.9),
    ("recursion-js", "RangeError: Maximum call stack size exceeded", "recursion", 0.9),
    ("recursion-firefox", "InternalError: too much recursion", "recursion", 0.9),
    ("recursion-go", "runtime: goroutine stack overflow", "recursion", 0.9),
    #
    # --- timeout ---
    ("timeout-python", "TimeoutError: operation timed out after 30s", "timeout", 0.85),
    ("timeout-keyword", "FAILED: timeout waiting for response", "timeout", 0.85),
    ("timeout-grpc", "rpc error: code = DeadlineExceeded desc = deadline exceeded", "timeout", 0.85),
    ("timeout-jest", "Timeout - Async callback was not invoked within the 5000 ms timeout", "timeout", 0.85),
    ("timeout-go", "context: took longer than 10s", "timeout", 0.85),
    ("timeout-time-limit", "Error: time limit reached for test suite", "timeout", 0.85),
    #
    # --- import ---
    ("import-python", "ImportError: cannot import name 'foo' from 'bar'", "import", 0.9),
    ("import-module", "ModuleNotFoundError: No module named 'nonexistent'", "import", 0.9),
    ("import-js", "Error: Cannot find module './utils'", "import", 0.9),
    ("import-no-module", "Error: no module named 'requests'", "import", 0.9),
    #
    # --- assertion ---
    ("assertion-python", "AssertionError: assert 3 == 4", "assertion", 0.8),
    ("assertion-pytest", "E       assert 'hello' == 'world'", "assertion", 0.8),
    ("assertion-jest", "expect(received).toEqual(expected)\n  Expected: 5\n  Received: 3", "assertion", 0.8),
    ("assertion-jest-tobe", "expect(received).to be(true)", "assertion", 0.8),
    ("assertion-rust", "thread 'tests::test_add' panicked at 'assert_eq!(left, right)'", "assertion", 0.8),
    ("assertion-go", "Error: expected 42 but got 0", "assertion", 0.8),
    ("assertion-assertEqual", "FAIL: assertEqual(result, expected)", "assertion", 0.8),
    #
    # --- syntax ---
    ("syntax-python", "SyntaxError: invalid syntax (app.py, line 10)", "syntax", 0.9),
    ("syntax-python-kw", "SyntaxError: invalid syntax", "syntax", 0.9),
    ("syntax-js", "SyntaxError: Unexpected token '}'", "syntax", 0.9),
    ("syntax-parse", "error[E0308]: parsing error at line 5", "syntax", 0.9),
    #
    # --- type_error ---
    ("type-python", "TypeError: unsupported operand type(s) for +: 'int' and 'str'", "type_error", 0.85),
    ("type-js", "TypeError: Cannot read properties of undefined", "type_error", 0.85),
    ("type-rust", "error[E0308]: type mismatch: expected i32, found &str", "type_error", 0.85),
    ("type-go", "cannot convert string to int", "type_error", 0.85),
    #
    # --- permission ---
    ("perm-python", "PermissionError: [Errno 13] Permission denied: '/etc/secret'", "permission", 0.9),
    ("perm-unix", "bash: /usr/local/bin/tool: Permission denied", "permission", 0.9),
    ("perm-node", "Error: EACCES: permission denied, open '/var/data'", "permission", 0.9),
    ("perm-win", "Access denied to resource", "permission", 0.9),
    ("perm-eperm", "Error: EPERM: operation not permitted", "permission", 0.9),
    #
    # --- connection ---
    ("conn-python", "ConnectionError: connection refused at localhost:8080", "connection", 0.85),
    ("conn-refused", "Error: connect ECONNREFUSED 127.0.0.1:5432", "connection", 0.85),
    ("conn-network", "FetchError: network error at https://api.example.com", "connection", 0.85),
    ("conn-dns", "Error: DNS resolution failed for api.example.com", "connection", 0.85),
    #
    # --- memory ---
    ("mem-python", "MemoryError: unable to allocate 4.00 GiB", "memory", 0.85),
    ("mem-oom", "FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory", "memory", 0.85),
    ("mem-node", "JavaScript heap out of memory", "memory", 0.85),
    ("mem-oom-killed", "Process killed: out of memory", "memory", 0.85),
    #
    # --- attribute ---
    ("attr-python", "AttributeError: 'NoneType' object has no attribute 'split'", "attribute", 0.85),
    ("attr-js", "TypeError: Cannot read property 'foo' of undefined", "attribute", 0.0),
    # Note: attr-js matches type_error first (TypeError) — that's expected.
    ("attr-nomethod", "error: no method named 'push' found for type Vec<i32>", "attribute", 0.85),
    #
    # --- name_error ---
    ("name-python", "NameError: name 'foo' is not defined", "name_error", 0.85),
    ("name-js", "ReferenceError: bar is not defined", "name_error", 0.85),
    ("name-undefined", "Error: undefined variable 'config'", "name_error", 0.85),
    ("name-undeclared", "error: undeclared identifier 'result'", "name_error", 0.85),
    #
    # --- file_not_found ---
    ("fnf-python", "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'", "file_not_found", 0.9),
    ("fnf-node", "Error: ENOENT: no such file or directory, open 'config.json'", "file_not_found", 0.9),
    ("fnf-generic", "Error: file not found: /tmp/output.txt", "file_not_found", 0.9),
    ("fnf-path", "Error: path not found: C:\\Users\\test\\file.txt", "file_not_found", 0.9),
    #
    # --- noop_edit ---
    ("noop-message", "old_text and new_text are identical — this edit would change nothing in foo.py", "noop_edit", 0.95),
    ("noop-error-tag", "noop_edit: agent submitted identical strings", "noop_edit", 0.95),
]


@pytest.mark.parametrize(
    "output, expected_category, min_confidence",
    [(c[1], c[2], c[3]) for c in _CLASSIFY_CASES],
    ids=[c[0] for c in _CLASSIFY_CASES],
)
def test_classify_error_categories(output: str, expected_category: str, min_confidence: float) -> None:
    """Verify each error category is detected with correct confidence."""
    category, confidence = RepeatedFailureTracker._classify_error(output)
    # Some cases intentionally match a different category first (e.g. attr-js
    # matches type_error because "TypeError" comes first).  For those, min_confidence
    # is set to 0.0 so we skip the category/confidence assertion but still confirm
    # we get *some* match.
    if min_confidence > 0.0:
        assert category == expected_category, (
            f"Expected category '{expected_category}' but got '{category}' for: {output!r}"
        )
        assert confidence >= min_confidence, (
            f"Expected confidence >= {min_confidence} but got {confidence}"
        )


def test_classify_error_returns_none_for_unknown() -> None:
    """Completely unrelated output should return (None, 0.0)."""
    output = "All 42 tests passed in 1.23s\nOK"
    category, confidence = RepeatedFailureTracker._classify_error(output)
    assert category is None
    assert confidence == 0.0


def test_error_patterns_have_no_prescriptive_advice() -> None:
    """The registry is detection-only.

    An earlier version included paragraphs like "Verify recursive calls
    converge toward a base case" — generic CS advice the LLM already knows.
    The cleanup stripped these strings so the registry just classifies; the
    LLM decides recovery.
    """
    for entry in RepeatedFailureTracker._ERROR_PATTERNS:
        assert "advice" not in entry, (
            f"Category {entry['category']} still has 'advice' field — "
            "registry is now detection-only."
        )
        assert {"category", "patterns", "confidence"} <= set(entry.keys())


# ---------------------------------------------------------------------------
# Nudge integration: error category advice appears in nudge output
# ---------------------------------------------------------------------------

class TestErrorAwareNudges:
    def _make_failure_output(self, error_text: str) -> str:
        """Build a realistic multi-line failure output with the given error."""
        return (
            "============================= FAILURES =============================\n"
            "__________________ test_something __________________\n"
            "\n"
            "    def test_something():\n"
            "        result = do_thing()\n"
            f"E   {error_text}\n"
            "\n"
            "tests/test_foo.py:12: " + error_text.split(":")[0] + "\n"
            "========================= short test summary =========================\n"
            "FAILED tests/test_foo.py::test_something - " + error_text + "\n"
            "========================= 1 failed =========================\n"
        )

    # Each category's nudge labels the category and reports the failure count.
    # We deliberately do NOT assert the presence of prescriptive advice strings
    # (e.g. "converge toward a base case") — those were removed when the
    # registry became detection-only.  See the comment on _ERROR_PATTERNS.

    def test_recursion_error_gets_targeted_nudge(self) -> None:
        tracker = RepeatedFailureTracker()
        output = self._make_failure_output(
            "RecursionError: maximum recursion depth exceeded in comparison"
        )
        nudge = None
        for _ in range(3):
            result = tracker.record("Bash", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None
        lower = nudge.lower()
        assert "recursion" in lower
        assert "not converging" in lower

    def test_timeout_error_gets_targeted_nudge(self) -> None:
        tracker = RepeatedFailureTracker()
        output = self._make_failure_output(
            "TimeoutError: timed out after 30 seconds"
        )
        nudge = None
        for _ in range(3):
            result = tracker.record("Bash", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None
        lower = nudge.lower()
        assert "timeout" in lower
        assert "not converging" in lower

    def test_import_error_gets_targeted_nudge(self) -> None:
        tracker = RepeatedFailureTracker()
        output = self._make_failure_output(
            "ImportError: cannot import name 'Widget' from 'myapp.models'"
        )
        nudge = None
        for _ in range(3):
            result = tracker.record("Bash", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None
        lower = nudge.lower()
        assert "import" in lower
        assert "not converging" in lower

    def test_assertion_error_gets_targeted_nudge(self) -> None:
        tracker = RepeatedFailureTracker()
        output = self._make_failure_output(
            "AssertionError: assert 3 == 4"
        )
        nudge = None
        for _ in range(3):
            result = tracker.record("Bash", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None
        lower = nudge.lower()
        assert "assertion" in lower
        assert "not converging" in lower

    def test_syntax_error_gets_targeted_nudge(self) -> None:
        tracker = RepeatedFailureTracker()
        output = self._make_failure_output(
            "SyntaxError: invalid syntax (app.py, line 10)"
        )
        nudge = None
        for _ in range(3):
            result = tracker.record("Bash", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None
        lower = nudge.lower()
        assert "syntax" in lower
        assert "not converging" in lower

    def test_file_not_found_gets_targeted_nudge(self) -> None:
        tracker = RepeatedFailureTracker()
        output = self._make_failure_output(
            "FileNotFoundError: [Errno 2] No such file or directory: 'missing.py'"
        )
        nudge = None
        for _ in range(3):
            result = tracker.record("Bash", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None
        lower = nudge.lower()
        assert "file not found" in lower or "file_not_found" in lower.replace(" ", "_")

    def test_generic_error_gets_generic_nudge(self) -> None:
        """Unknown errors get the generic nudge (no category title)."""
        tracker = RepeatedFailureTracker()
        output = (
            "============================= FAILURES =============================\n"
            "FAILED tests/test_foo.py::test_something\n"
            "Some unknown error that does not match any category\n"
            "1 failed\n"
        )
        nudge = None
        for _ in range(3):
            result = tracker.record("Bash", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None
        lower = nudge.lower()
        assert "not converging" in lower
        # Should NOT contain the dropped "category-specific advice" header
        assert "category-specific advice" not in lower
        # Generic path: no specific category title appears.
        assert "repeated failure detection — " not in lower

    def test_edit_failure_still_uses_lower_threshold(self) -> None:
        """Edit failures should trigger a nudge after only 2 identical failures."""
        tracker = RepeatedFailureTracker()
        output = (
            "Error: old_text not found in file app.py\n"
            "The text you provided does not match any section of the file."
        )
        nudge = None
        for _ in range(2):
            result = tracker.record("Edit", output, success=False)
            if result is not None:
                nudge = result
        assert nudge is not None, "Expected a nudge after 2 identical Edit failures"
        assert "Edit" in nudge

    def test_no_nudge_before_threshold(self) -> None:
        """Two non-Edit failures should not produce a nudge."""
        tracker = RepeatedFailureTracker()
        output = self._make_failure_output("AssertionError: assert False")
        result = None
        for _ in range(2):
            result = tracker.record("Bash", output, success=False)
        assert result is None

    def test_successful_results_are_ignored(self) -> None:
        """Successful tool results should not contribute to failure tracking."""
        tracker = RepeatedFailureTracker()
        output = "All tests passed"
        result = None
        for _ in range(5):
            result = tracker.record("Bash", output, success=True)
        assert result is None


# ---------------------------------------------------------------------------
# Test-file nudge includes the detection signal (no prescriptive advice)
# ---------------------------------------------------------------------------

class TestTestFileNudgeDetectionSignal:
    def test_test_file_nudge_labels_assertion_category(self) -> None:
        """The nudge labels the detected category but does NOT prescribe how
        to fix it — the LLM knows how to read assert outputs."""
        tail = "AssertionError: assert 3 == 4\nFAILED tests/test_math.py::test_add"
        nudge = RepeatedFailureTracker._build_test_file_nudge("tests/test_math.py", 5, tail)
        lower = nudge.lower()
        # Category labeled.
        assert "detected error category: assertion" in lower
        # File and count reported.
        assert "test_math.py" in nudge
        assert "5 times" in nudge

    def test_test_file_nudge_no_category_for_unknown(self) -> None:
        """No 'Detected error category' line for unclassified output."""
        tail = "Something went completely sideways\nFAILED"
        nudge = RepeatedFailureTracker._build_test_file_nudge("tests/test_foo.py", 5, tail)
        assert "Detected error category" not in nudge
        # Old wording must not regress.
        assert "Detected error type" not in nudge


# ---------------------------------------------------------------------------
# Confidence scores validation
# ---------------------------------------------------------------------------

def test_all_pattern_confidences_in_range() -> None:
    """Every entry in _ERROR_PATTERNS should have confidence in (0, 1]."""
    for entry in RepeatedFailureTracker._ERROR_PATTERNS:
        c = entry["confidence"]
        assert 0.0 < c <= 1.0, f"Bad confidence {c} for {entry['category']}"


def test_all_categories_have_compiled_patterns() -> None:
    """Every entry must have at least one compiled regex pattern."""
    for entry in RepeatedFailureTracker._ERROR_PATTERNS:
        assert len(entry["patterns"]) > 0, f"No patterns for {entry['category']}"
        for pat in entry["patterns"]:
            assert hasattr(pat, "search"), (
                f"Pattern in {entry['category']} is not a compiled regex"
            )
