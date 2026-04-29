"""Tests for maike.context.convergence — spinning detection and nudge generation."""

from maike.context.convergence import (
    build_convergence_nudge,
    build_escalated_nudge,
    detect_spinning,
    _extract_contract_items,
    _extract_file_mutations,
    _extract_validation_results,
)


class TestDetectSpinning:
    def _tool_use_msg(self, name: str, path: str = "app.py"):
        return {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": name, "input": {"path": path}}],
        }

    def test_spinning_when_repeated(self):
        conversation = [
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
        ]
        assert detect_spinning(conversation, window=6) is True

    def test_not_spinning_when_varied(self):
        conversation = [
            self._tool_use_msg("write_file", "app.py"),
            self._tool_use_msg("read_file", "test.py"),
            self._tool_use_msg("edit_file", "utils.py"),
            self._tool_use_msg("syntax_check", "app.py"),
            self._tool_use_msg("run_tests", "tests/"),
            self._tool_use_msg("edit_file", "main.py"),
        ]
        assert detect_spinning(conversation, window=6) is False

    def test_not_spinning_with_too_few_calls(self):
        conversation = [
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
        ]
        assert detect_spinning(conversation, window=6) is False

    def test_empty_conversation(self):
        assert detect_spinning([], window=6) is False

    def test_spinning_threshold_boundary(self):
        # 3 of 6 = 50%, which is NOT > 50% — should not detect spinning.
        conversation = [
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("read_file", "b.py"),
            self._tool_use_msg("write_file", "c.py"),
            self._tool_use_msg("syntax_check", "d.py"),
        ]
        assert detect_spinning(conversation, window=6) is False

    def test_spinning_just_over_threshold(self):
        # 4 of 6 > 50% — should detect spinning.
        conversation = [
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("edit_file", "app.py"),
            self._tool_use_msg("read_file", "b.py"),
            self._tool_use_msg("write_file", "c.py"),
        ]
        assert detect_spinning(conversation, window=6) is True


class TestExtractFileMutations:
    def test_tracks_mutations(self):
        conversation = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "write_file", "input": {"path": "app.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "write_file", "content": "ok"},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "edit_file", "input": {"path": "app.py"}},
            ]},
        ]
        result = _extract_file_mutations(conversation)
        assert "app.py" in result
        assert len(result["app.py"]) == 2

    def test_ignores_reads(self):
        conversation = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "read_file", "input": {"path": "app.py"}},
            ]},
        ]
        result = _extract_file_mutations(conversation)
        assert result == {}


class TestExtractValidationResults:
    def test_extracts_pass(self):
        conversation = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "syntax_check", "content": "ok", "is_error": False},
            ]},
        ]
        results = _extract_validation_results(conversation)
        assert len(results) == 1
        assert results[0][0] == "syntax_check"
        assert results[0][1] is True

    def test_extracts_fail(self):
        conversation = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "run_tests", "content": "2 failed", "is_error": True},
            ]},
        ]
        results = _extract_validation_results(conversation)
        assert len(results) == 1
        assert results[0][1] is False

    def test_ignores_non_validation_tools(self):
        conversation = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "read_file", "content": "data"},
            ]},
        ]
        assert _extract_validation_results(conversation) == []


class TestExtractContractItems:
    def test_extracts_required_files(self):
        conversation = [
            {"role": "user", "content": (
                "## Artifact: acceptance-contract.md\n"
                "### Required Files and Docs\n"
                "- `app.py` — main application\n"
                "- `test_app.py` — unit tests\n"
                "- `README.md` — documentation\n"
                "---\n"
                "## Task\n"
                "Build the app."
            )},
        ]
        items = _extract_contract_items(conversation)
        assert len(items) == 3
        assert "`app.py` — main application" in items[0]

    def test_empty_contract(self):
        conversation = [
            {"role": "user", "content": "## Task\nBuild something."},
        ]
        assert _extract_contract_items(conversation) == []

    def test_empty_conversation(self):
        assert _extract_contract_items([]) == []


class TestBuildConvergenceNudge:
    def test_nudge_includes_all_sections(self):
        conversation = [
            {"role": "user", "content": (
                "## Artifact: acceptance-contract.md\n"
                "### Required Files and Docs\n"
                "- `app.py`\n"
                "---\n"
                "## Task\nBuild app."
            )},
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "write_file", "input": {"path": "app.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "write_file", "content": "ok"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "syntax_check", "content": "passed", "is_error": False},
            ]},
        ]
        nudge = build_convergence_nudge(conversation, 18, 30)
        assert "Convergence Check (iteration 18/30)" in nudge
        assert "Files Modified" in nudge
        assert "app.py" in nudge
        assert "Validation Results" in nudge
        assert "Suggested Focus" in nudge

    def test_nudge_with_no_activity(self):
        conversation = [
            {"role": "user", "content": "## Task\nBuild app."},
        ]
        nudge = build_convergence_nudge(conversation, 10, 20)
        assert "Convergence Check" in nudge
        assert "None yet" in nudge

    def test_nudge_highlights_failing_validation(self):
        conversation = [
            {"role": "user", "content": "## Task\nBuild app."},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "run_tests", "content": "FAILED test_add", "is_error": True},
            ]},
        ]
        nudge = build_convergence_nudge(conversation, 20, 30)
        assert "FAIL" in nudge
        assert "Fix the failing" in nudge


class TestBuildEscalatedNudgeErrorPatterns:
    """Tests for error-aware convergence nudges in level-2 escalation."""

    def _make_conversation_with_validation_error(self, error_text: str) -> list:
        """Build a conversation containing a failing validation with the given error text."""
        return [
            {"role": "user", "content": "## Task\nBuild app."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Write", "input": {"path": "app.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_name": "Write", "content": "ok"},
            ]},
            {"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_name": "Bash",
                    "content": error_text,
                    "is_error": True,
                },
            ]},
        ]

    def test_level_2_nudge_detects_recursion_error(self):
        conversation = self._make_conversation_with_validation_error(
            "Traceback (most recent call last):\n"
            "  File \"app.py\", line 10, in solve\n"
            "    return solve(n)\n"
            "RecursionError: maximum recursion depth exceeded"
        )
        nudge = build_escalated_nudge(conversation, 20, 30, level=2)
        assert "Infinite Recursion" in nudge
        assert "recursive function" in nudge
        assert "Timeout" not in nudge

    def test_level_2_nudge_detects_timeout(self):
        conversation = self._make_conversation_with_validation_error(
            "ERROR: test_solve timed out after 30 seconds"
        )
        nudge = build_escalated_nudge(conversation, 20, 30, level=2)
        assert "Timeout" in nudge
        assert "Infinite Loop" in nudge
        assert "Infinite Recursion" not in nudge

    def test_level_2_nudge_generic_without_pattern(self):
        conversation = self._make_conversation_with_validation_error(
            "AssertionError: expected 4 but got 5"
        )
        nudge = build_escalated_nudge(conversation, 20, 30, level=2)
        assert "Infinite Recursion" not in nudge
        assert "Timeout" not in nudge
        # Should still contain the standard level 2 nudge content.
        assert "Strategy Reset" in nudge
        assert "Required Actions" in nudge
