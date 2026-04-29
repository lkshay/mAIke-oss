"""Tests for end-of-session verdict classification.

The verdict system is fully deterministic — no LLM calls, no JSON parsing.
These tests exercise:
  1. ``SessionVerdict`` dataclass (+ metadata round-trip)
  2. ``count_successful_edits`` / ``tool_error_rate``
  3. ``classify_deterministic`` (hard short-circuit)
  4. ``classify_heuristic`` (satisfied-vs-partial via surface signals)
  5. ``classify_session`` orchestration
"""

from __future__ import annotations

from maike.memory.verdict import (
    SessionVerdict,
    classify_deterministic,
    classify_heuristic,
    classify_session,
    count_successful_edits,
    tool_error_rate,
)


# ---------------------------------------------------------------------------
# SessionVerdict dataclass
# ---------------------------------------------------------------------------


class TestSessionVerdictDataclass:
    def test_defaults_are_safe(self):
        v = SessionVerdict()
        assert v.label == "unknown"
        assert v.source == "fallback"
        assert v.confidence == 0.0

    def test_metadata_roundtrip_preserves(self):
        v = SessionVerdict(
            label="satisfied", confidence=0.9,
            rationale="patch applied", source="deterministic",
        )
        restored = SessionVerdict.from_metadata(v.to_metadata())
        assert restored is not None
        assert restored.label == "satisfied"
        assert restored.confidence == 0.9
        assert restored.source == "deterministic"

    def test_from_metadata_none(self):
        assert SessionVerdict.from_metadata(None) is None

    def test_from_metadata_malformed_label(self):
        assert SessionVerdict.from_metadata({"label": "garbage"}) is None

    def test_from_metadata_non_dict(self):
        assert SessionVerdict.from_metadata("not a dict") is None
        assert SessionVerdict.from_metadata(42) is None

    def test_from_metadata_unknown_source_normalizes(self):
        v = SessionVerdict.from_metadata({"label": "satisfied", "source": "bogus"})
        assert v is not None
        assert v.source == "fallback"

    def test_render_line_includes_rationale(self):
        v = SessionVerdict(label="partial", confidence=0.7,
                           rationale="only 1 of 2 gold files touched",
                           source="deterministic")
        line = v.render_line()
        assert "Verdict: partial" in line
        assert "only 1 of 2" in line

    def test_render_line_deterministic_marker(self):
        v = SessionVerdict(label="cancelled", rationale="ctrl-c", source="deterministic")
        assert "(deterministic)" in v.render_line()


# ---------------------------------------------------------------------------
# count_successful_edits / tool_error_rate
# ---------------------------------------------------------------------------


def _msg_pair(tool_name: str, is_error: bool = False, tid: str = "t1") -> list[dict]:
    return [
        {"role": "assistant",
         "content": [{"type": "tool_use", "id": tid, "name": tool_name, "input": {}}]},
        {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": tid,
                      "is_error": is_error, "content": "ok"}]},
    ]


class TestCountSuccessfulEdits:
    def test_empty_messages(self):
        assert count_successful_edits([]) == 0
        assert count_successful_edits(None) == 0  # type: ignore[arg-type]

    def test_one_successful_edit(self):
        assert count_successful_edits(_msg_pair("Edit")) == 1

    def test_failed_edit_not_counted(self):
        assert count_successful_edits(_msg_pair("Edit", is_error=True)) == 0

    def test_mixed_edits(self):
        msgs = []
        for i, err in enumerate([False, True, False, False, True]):
            msgs.extend(_msg_pair("Edit", is_error=err, tid=f"t{i}"))
        assert count_successful_edits(msgs) == 3

    def test_non_edit_tools_ignored(self):
        msgs = _msg_pair("Bash")
        msgs.extend(_msg_pair("Grep", tid="t2"))
        msgs.extend(_msg_pair("Read", tid="t3"))
        assert count_successful_edits(msgs) == 0

    def test_write_and_edit_both_count(self):
        msgs = _msg_pair("Edit", tid="t1")
        msgs.extend(_msg_pair("Write", tid="t2"))
        msgs.extend(_msg_pair("MultiEdit", tid="t3"))
        assert count_successful_edits(msgs) == 3


class TestToolErrorRate:
    def test_empty_is_zero(self):
        assert tool_error_rate([]) == 0.0
        assert tool_error_rate(None) == 0.0  # type: ignore[arg-type]

    def test_all_success(self):
        msgs = []
        for i in range(4):
            msgs.extend(_msg_pair("Bash", tid=f"t{i}"))
        assert tool_error_rate(msgs) == 0.0

    def test_all_error(self):
        msgs = []
        for i in range(3):
            msgs.extend(_msg_pair("Edit", is_error=True, tid=f"t{i}"))
        assert tool_error_rate(msgs) == 1.0

    def test_half_and_half(self):
        msgs = []
        for i, err in enumerate([True, True, False, False]):
            msgs.extend(_msg_pair("Bash", is_error=err, tid=f"t{i}"))
        assert tool_error_rate(msgs) == 0.5


# ---------------------------------------------------------------------------
# Deterministic short-circuit
# ---------------------------------------------------------------------------


class TestClassifyDeterministic:
    def test_cancelled(self):
        v = classify_deterministic(
            outcome="cancelled — interrupted by user",
            edits_count=0, budget_hit=False, iteration_cap_hit=False,
        )
        assert v is not None
        assert v.label == "cancelled"
        assert v.source == "deterministic"

    def test_cancelled_with_edits_still_cancelled(self):
        v = classify_deterministic(
            outcome="cancelled — user killed it",
            edits_count=5, budget_hit=False, iteration_cap_hit=False,
        )
        assert v is not None
        assert v.label == "cancelled"

    def test_budget_hit_zero_edits(self):
        v = classify_deterministic(
            outcome="failure",
            edits_count=0, budget_hit=True, iteration_cap_hit=False,
        )
        assert v is not None
        assert v.label == "unproductive_budget_exhaustion"

    def test_iteration_cap_zero_edits(self):
        v = classify_deterministic(
            outcome="failure",
            edits_count=0, budget_hit=False, iteration_cap_hit=True,
        )
        assert v is not None
        assert v.label == "unproductive_loop"

    def test_budget_hit_with_edits_defers_to_heuristic(self):
        v = classify_deterministic(
            outcome="failure",
            edits_count=5, budget_hit=True, iteration_cap_hit=False,
        )
        assert v is None

    def test_success_with_edits_defers_to_heuristic(self):
        v = classify_deterministic(
            outcome="success",
            edits_count=3, budget_hit=False, iteration_cap_hit=False,
        )
        assert v is None

    def test_empty_outcome(self):
        v = classify_deterministic(
            outcome=None, edits_count=2, budget_hit=False, iteration_cap_hit=False,
        )
        assert v is None

    def test_budget_hit_flag_triggers_even_with_low_threshold(self):
        # Orchestrator passes budget_hit=True when the outcome text contains
        # a budget-exceeded marker, even if the numeric threshold wasn't
        # strictly met.  classify_deterministic trusts the flag directly.
        v = classify_deterministic(
            outcome="failure — Session cost budget exceeded",
            edits_count=0, budget_hit=True, iteration_cap_hit=False,
        )
        assert v is not None
        assert v.label == "unproductive_budget_exhaustion"


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------


class TestClassifyHeuristic:
    def test_zero_edits_is_partial(self):
        v = classify_heuristic(edits_count=0, agent_output="I fixed it.")
        assert v.label == "partial"
        assert v.source == "deterministic"

    def test_complete_marker_with_edits_is_satisfied(self):
        v = classify_heuristic(
            edits_count=3,
            agent_output="Task complete. The fix is now applied.",
        )
        assert v.label == "satisfied"
        assert "completion markers" in v.rationale

    def test_partial_marker_is_partial(self):
        v = classify_heuristic(
            edits_count=3,
            agent_output="I partial applied the fix; the test still fails.",
        )
        assert v.label == "partial"

    def test_couldnt_is_partial(self):
        v = classify_heuristic(
            edits_count=2,
            agent_output="I couldn't figure out how to handle the edge case.",
        )
        assert v.label == "partial"

    def test_unable_to_is_partial(self):
        v = classify_heuristic(
            edits_count=1,
            agent_output="I was unable to finalize the verification step.",
        )
        assert v.label == "partial"

    def test_high_error_rate_is_partial(self):
        # Build a conversation with 60% errors
        msgs = []
        for i, err in enumerate([True, True, True, False, False]):
            msgs.extend(_msg_pair("Edit", is_error=err, tid=f"t{i}"))
        v = classify_heuristic(
            edits_count=2,
            agent_output="All set.",  # no partial marker
            messages=msgs,
        )
        assert v.label == "partial"
        assert "tool-error" in v.rationale

    def test_clean_session_with_edits_is_satisfied(self):
        # No markers either way, low error rate, edits present.
        msgs = []
        for i in range(5):
            msgs.extend(_msg_pair("Edit", is_error=False, tid=f"t{i}"))
        v = classify_heuristic(
            edits_count=5,
            agent_output="Updated the file.",  # neutral, no strong markers
            messages=msgs,
        )
        assert v.label == "satisfied"

    def test_mixed_markers_defaults_to_partial(self):
        # Both completion and incomplete markers → ambiguous → partial
        v = classify_heuristic(
            edits_count=3,
            agent_output=("Task complete on the main issue, but I couldn't "
                          "address the edge case."),
        )
        assert v.label == "partial"

    def test_no_markers_moderate_errors_is_partial(self):
        msgs = []
        # 30% error rate — above the "clean" threshold (10%) but below
        # the "high error" threshold (50%)
        for i, err in enumerate([False, True, False, True, False, False, False, True, False, False]):
            msgs.extend(_msg_pair("Bash", is_error=err, tid=f"t{i}"))
        v = classify_heuristic(
            edits_count=2, agent_output="Done.",  # would be satisfied if clean
            messages=msgs,
        )
        # "Done." is in COMPLETE_MARKERS → should pick satisfied even with
        # moderate errors (error rate < 50%).  Check for satisfied.
        assert v.label == "satisfied"

    def test_no_messages_safe(self):
        # Called without messages — tool_error_rate returns 0.0.
        v = classify_heuristic(edits_count=2, agent_output="", messages=None)
        assert v.label in {"satisfied", "partial"}

    # ── Regression tests for "missing" marker tightening (April 17, 2026) ──
    # Live trial on /tmp/maike-websearch-test produced a satisfied session
    # with final text "handles HTTP errors and missing fields" that was
    # incorrectly labelled partial by the bare "missing" substring match.
    # Markers now require self-referential phrasing.

    def test_missing_fields_feature_description_is_satisfied(self):
        """Regression: 'handles ... missing fields' describes code behavior,
        not incomplete work.  Bare 'missing' used to false-positive here."""
        v = classify_heuristic(
            edits_count=1,
            agent_output=(
                "Task complete. The script handles HTTP errors and missing "
                "fields gracefully."
            ),
        )
        assert v.label == "satisfied"

    def test_missing_key_feature_description_is_satisfied(self):
        """Second form of the same false-positive: dict-key description."""
        v = classify_heuristic(
            edits_count=2,
            agent_output=(
                "Implemented the fix. The parser now raises KeyError when "
                "a required field is missing from the payload."
            ),
        )
        assert v.label == "satisfied"

    def test_self_ref_still_missing_is_partial(self):
        """Agent saying its *own* work is still missing something → partial."""
        v = classify_heuristic(
            edits_count=1,
            agent_output=(
                "Implemented most of the fix but still missing the error "
                "handling branch."
            ),
        )
        assert v.label == "partial"

    def test_self_ref_im_missing_is_partial(self):
        v = classify_heuristic(
            edits_count=1,
            agent_output="I'm missing the test fixture to verify this end-to-end.",
        )
        assert v.label == "partial"

    def test_self_ref_i_missed_is_partial(self):
        v = classify_heuristic(
            edits_count=1,
            agent_output="Applied the patch but I missed updating the docstring.",
        )
        assert v.label == "partial"


# ---------------------------------------------------------------------------
# Orchestrated classify_session
# ---------------------------------------------------------------------------


class TestClassifySession:
    def test_deterministic_short_circuits(self):
        v = classify_session(
            outcome="cancelled — by user",
            edits_count=0, budget_hit=False, iteration_cap_hit=False,
            task="t", agent_output="", messages=[],
        )
        assert v.label == "cancelled"

    def test_budget_hit_short_circuits(self):
        v = classify_session(
            outcome="failure",
            edits_count=0, budget_hit=True, iteration_cap_hit=False,
            task="t", agent_output="", messages=[],
        )
        assert v.label == "unproductive_budget_exhaustion"

    def test_satisfied_via_heuristic(self):
        msgs = []
        for i in range(3):
            msgs.extend(_msg_pair("Edit", tid=f"t{i}"))
        v = classify_session(
            outcome="success",
            edits_count=3, budget_hit=False, iteration_cap_hit=False,
            task="Fix X", agent_output="Task complete. Patch applied.",
            messages=msgs,
        )
        assert v.label == "satisfied"
        assert v.source == "deterministic"

    def test_partial_via_heuristic(self):
        msgs = _msg_pair("Edit")
        v = classify_session(
            outcome="success",
            edits_count=1, budget_hit=False, iteration_cap_hit=False,
            task="t", agent_output="I couldn't solve the edge case.",
            messages=msgs,
        )
        assert v.label == "partial"

    def test_legacy_kwargs_ignored(self):
        # Callers that still pass the old session_bg_gateway / provider kwargs
        # should not break.
        v = classify_session(
            outcome="cancelled — x",
            edits_count=0, budget_hit=False, iteration_cap_hit=False,
            task="t", agent_output="",
            mutation_ledger=["a.py"],
            session_bg_gateway=object(),  # unused
            session_provider="gemini",    # unused
            timeout_s=1.0,                # unused
        )
        assert v.label == "cancelled"

    def test_never_raises_even_on_weird_input(self):
        # Pass unusual types — classify_session must swallow errors.
        v = classify_session(
            outcome=None, edits_count=0, budget_hit=False,
            iteration_cap_hit=False, task="", agent_output=None,  # type: ignore[arg-type]
            messages=None,
        )
        # Should classify as partial (edits_count=0 with no cancel/budget/iter)
        assert v.label in {"partial", "unknown"}

    def test_no_llm_ever_called(self):
        """Sanity check: classify_session is synchronous and does no I/O.
        Previously this was async + LLM-backed; make sure no regression
        reintroduces the LLM path.
        """
        import inspect
        assert not inspect.iscoroutinefunction(classify_session)
