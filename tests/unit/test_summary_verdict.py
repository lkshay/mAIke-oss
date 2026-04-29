"""Tests that SessionSummaryBuilder optionally renders a verdict line."""

from __future__ import annotations

from maike.memory.summary import SessionSummaryBuilder
from maike.memory.verdict import SessionVerdict


def _minimal_messages() -> list[dict]:
    """A minimal conversation — assistant emits a final output, no tool calls."""
    return [
        {"role": "user", "content": "Fix the bug"},
        {"role": "assistant", "content": "Done."},
    ]


class TestBuilderVerdictLine:
    def test_no_verdict_no_line(self):
        s = SessionSummaryBuilder().build_summary(
            messages=_minimal_messages(),
            task="Fix the bug",
            outcome="success",
            session_id="s1",
            timestamp="2026-04-16T17:00:00Z",
            agent_output="I fixed it.",
        )
        assert "Verdict:" not in s

    def test_verdict_present_renders_line(self):
        v = SessionVerdict(
            label="satisfied", confidence=0.9,
            rationale="patch applied", source="llm",
        )
        s = SessionSummaryBuilder().build_summary(
            messages=_minimal_messages(),
            task="Fix the bug",
            outcome="success",
            session_id="s1",
            timestamp="2026-04-16T17:00:00Z",
            agent_output="I fixed it.",
            verdict=v,
        )
        assert "Verdict: satisfied" in s
        assert "patch applied" in s
        # Verdict must appear right after Outcome line.
        lines = s.splitlines()
        outcome_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Outcome:"))
        assert lines[outcome_idx + 1].startswith("Verdict:")

    def test_verdict_with_broken_render_does_not_crash(self):
        class _BadVerdict:
            def render_line(self):
                raise RuntimeError("synthetic")
        # Should not raise and should not include any Verdict line.
        s = SessionSummaryBuilder().build_summary(
            messages=_minimal_messages(),
            task="Fix the bug",
            outcome="success",
            session_id="s1",
            timestamp="2026-04-16T17:00:00Z",
            agent_output="I fixed it.",
            verdict=_BadVerdict(),
        )
        assert "Outcome: success" in s
        assert "Verdict:" not in s

    def test_verdict_none_is_same_as_omitted(self):
        a = SessionSummaryBuilder().build_summary(
            messages=_minimal_messages(),
            task="t", outcome="success", session_id="s1",
            timestamp="2026-04-16T17:00:00Z", agent_output="done",
        )
        b = SessionSummaryBuilder().build_summary(
            messages=_minimal_messages(),
            task="t", outcome="success", session_id="s1",
            timestamp="2026-04-16T17:00:00Z", agent_output="done",
            verdict=None,
        )
        assert a == b
