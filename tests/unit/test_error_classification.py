"""Tests for eval error classification."""

from maike.eval.contracts import ErrorCategory, classify_error


def test_timeout_classification():
    assert classify_error("Agent timed out after 120s") == ErrorCategory.TIMEOUT
    assert classify_error("iteration limit reached") == ErrorCategory.TIMEOUT
    assert classify_error(None, ["max_iterations exceeded"]) == ErrorCategory.TIMEOUT


def test_budget_classification():
    assert classify_error("Budget exceeded: $5.00 limit") == ErrorCategory.BUDGET_EXHAUSTED
    assert classify_error("token limit reached") == ErrorCategory.BUDGET_EXHAUSTED


def test_provider_error_classification():
    assert classify_error("401 Unauthorized") == ErrorCategory.PROVIDER_ERROR
    assert classify_error("Rate limit hit (429)") == ErrorCategory.PROVIDER_ERROR
    assert classify_error("API error from provider") == ErrorCategory.PROVIDER_ERROR


def test_tool_error_classification():
    assert classify_error("Bash command failed with exit code 1") == ErrorCategory.TOOL_ERROR
    assert classify_error("File not found: src/main.py") == ErrorCategory.TOOL_ERROR


def test_test_failure_classification():
    assert classify_error("Tests failed: 3 failures") == ErrorCategory.TEST_FAILURE
    assert classify_error(None, ["pytest returned exit code 1"]) == ErrorCategory.TEST_FAILURE


def test_spec_mismatch_classification():
    assert classify_error("workspace_verified=False") == ErrorCategory.SPEC_MISMATCH
    assert classify_error("Acceptance criteria not met") == ErrorCategory.SPEC_MISMATCH


def test_empty_output_classification():
    assert classify_error("No artifacts produced") == ErrorCategory.EMPTY_OUTPUT
    assert classify_error("Empty output from agent") == ErrorCategory.EMPTY_OUTPUT


def test_crash_classification():
    assert classify_error("Unhandled exception in agent loop") == ErrorCategory.CRASH
    assert classify_error("Traceback (most recent call last)") == ErrorCategory.CRASH


def test_unknown_classification():
    assert classify_error("Something went wrong") == ErrorCategory.UNKNOWN
    assert classify_error(None) == ErrorCategory.UNKNOWN
    assert classify_error("") == ErrorCategory.UNKNOWN


def test_combined_error_and_reasons():
    """Multiple signals should still work."""
    assert classify_error("Error occurred", ["test failures detected"]) == ErrorCategory.TEST_FAILURE
