"""End-of-session task-alignment verdict.

Classifies a finished session into one of six labels so ``maike history`` can
show outcome quality and eval scoring can distinguish "produced a patch" from
"burned budget with zero output."

**Fully deterministic.**  An earlier version called a cheap LLM for the
satisfied-vs-partial split, but the targeted eval showed 50% JSON-parse
failures on real model output.  JSON-parsing LLM responses is an unreliable
foundation for session-critical metadata, so we dropped the LLM path entirely.

Classification is a two-phase pipeline, both phases deterministic:

1. **Hard short-circuit** — unambiguous cases:
     * cancelled (user Ctrl+C)
     * unproductive_budget_exhaustion (budget hit with 0 successful edits)
     * unproductive_loop (iteration cap hit with 0 successful edits)

2. **Heuristic classifier** — for the remaining satisfied/partial split.
   Uses surface signals already present in the session:
     * successful-edit count
     * tool-error rate
     * completion / failure markers in the agent's final natural-language
       output

Public API:
  - ``SessionVerdict`` dataclass (+ metadata round-trip)
  - ``count_successful_edits(messages)`` / ``tool_error_rate(messages)``
  - ``classify_deterministic(...)`` — the hard short-circuit
  - ``classify_heuristic(...)`` — the satisfied-vs-partial heuristic
  - ``classify_session(...)`` — orchestrates both; never raises
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


VerdictLabel = Literal[
    "satisfied",
    "partial",
    "unproductive_budget_exhaustion",
    "unproductive_loop",
    "cancelled",
    "unknown",
]


_VALID_LABELS: set[str] = {
    "satisfied",
    "partial",
    "unproductive_budget_exhaustion",
    "unproductive_loop",
    "cancelled",
    "unknown",
}


@dataclass
class SessionVerdict:
    """Structured classification of a finished session.

    ``source`` tracks which phase produced the label:
      - ``deterministic`` — from ``classify_deterministic`` (free, high confidence)
      - ``llm`` — from ``classify_llm`` (cheap model, moderate confidence)
      - ``fallback`` — from an error path (label is ``unknown``)
    """

    label: VerdictLabel = "unknown"
    confidence: float = 0.0
    rationale: str = ""
    source: Literal["deterministic", "llm", "fallback"] = "fallback"

    def to_metadata(self) -> dict[str, Any]:
        """Serialize for ``sessions.metadata`` JSON column."""
        return asdict(self)

    @classmethod
    def from_metadata(cls, data: Any) -> SessionVerdict | None:
        """Reconstruct; returns ``None`` for missing/malformed input rather
        than raising — keeps ``maike history`` robust against old sessions."""
        if not isinstance(data, dict):
            return None
        label = data.get("label")
        if label not in _VALID_LABELS:
            return None
        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        rationale = str(data.get("rationale", ""))[:500]
        source = data.get("source", "fallback")
        if source not in {"deterministic", "llm", "fallback"}:
            source = "fallback"
        return cls(label=label, confidence=confidence, rationale=rationale, source=source)

    def render_line(self) -> str:
        """Human-readable single-line summary embedded in session summary."""
        if self.confidence > 0 and self.source == "llm":
            conf_frag = f" (LLM {self.confidence:.2f})"
        elif self.source == "deterministic":
            conf_frag = " (deterministic)"
        else:
            conf_frag = ""
        tail = f" — {self.rationale}" if self.rationale else ""
        return f"Verdict: {self.label}{conf_frag}{tail}"


# ---------------------------------------------------------------------------
# Edit counting
# ---------------------------------------------------------------------------


_EDIT_TOOL_NAMES: frozenset[str] = frozenset({
    "Edit", "edit_file", "Write", "write_file", "MultiEdit", "multi_edit",
})


def _iter_tool_pairs(messages: list[dict[str, Any]]):
    """Yield (tool_use_name, tool_result_is_error) pairs from conversation.

    mAIke's conversation shape embeds tool_use and tool_result blocks inside
    message ``content`` lists.  This walks them in order and pairs them by
    ``tool_use_id`` when available.
    """
    pending: dict[str, str] = {}  # id → tool name
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                tid = block.get("id")
                name = block.get("name", "")
                if tid:
                    pending[tid] = name
            elif btype == "tool_result":
                tid = block.get("tool_use_id")
                is_error = bool(block.get("is_error"))
                if tid and tid in pending:
                    yield pending.pop(tid), is_error


def count_successful_edits(messages: list[dict[str, Any]]) -> int:
    """Count edit tool calls that returned ``is_error=False``.

    Failed edits (old_text mismatch, etc.) do NOT count — a session that ran
    10 broken Edits produced no net change and should not be called productive.
    """
    if not messages:
        return 0
    successful = 0
    for name, is_error in _iter_tool_pairs(messages):
        if name in _EDIT_TOOL_NAMES and not is_error:
            successful += 1
    return successful


# ---------------------------------------------------------------------------
# Deterministic classification
# ---------------------------------------------------------------------------


def classify_deterministic(
    *,
    outcome: str | None,
    edits_count: int,
    budget_hit: bool,
    iteration_cap_hit: bool,
) -> SessionVerdict | None:
    """Return a verdict for cases that don't need an LLM call, else ``None``.

    ``outcome`` is the free-text string from ``_session_outcome`` —
    "success", "failure", "cancelled — interrupted by user", "failure — <err>".
    We look for "cancelled" as a prefix.
    """
    if outcome and outcome.lower().startswith("cancelled"):
        return SessionVerdict(
            label="cancelled",
            confidence=1.0,
            rationale=f"session cancelled ({edits_count} successful edits before cancel)",
            source="deterministic",
        )
    if budget_hit and edits_count == 0:
        return SessionVerdict(
            label="unproductive_budget_exhaustion",
            confidence=1.0,
            rationale="session hit the cost budget without producing any successful edits",
            source="deterministic",
        )
    if iteration_cap_hit and edits_count == 0:
        return SessionVerdict(
            label="unproductive_loop",
            confidence=1.0,
            rationale="session hit the iteration cap without producing any successful edits",
            source="deterministic",
        )
    return None


# ---------------------------------------------------------------------------
# Heuristic classification (satisfied vs partial)
# ---------------------------------------------------------------------------


# Markers in agent's final text that suggest incomplete / unsure outcome.
# Lowercased; checked as substring.  Order does not matter.
#
# Markers must be *self-referential* — describing the agent's own work, not
# the code under edit.  Bare "missing" was originally in this list but it
# false-positives on feature descriptions like "handles HTTP errors and
# missing fields" (observed April 17, 2026 on a clean session).  Replaced
# with narrower forms that carry a self-referential subject.
_PARTIAL_MARKERS: tuple[str, ...] = (
    " partial", "incomplete", "could not ", "couldn't ",
    "unable to ", "gave up", "needs more", "still need",
    "still missing", "went missing", "we're missing", "we are missing",
    "i'm missing", "i am missing", "i missed ",
    "broke ", "broken", "failed to ",
    "issue remains", "still failing", "not entirely",
    "not fully", "wasn't able", "was unable",
)

# Markers that suggest the agent believes the task is done.
_COMPLETE_MARKERS: tuple[str, ...] = (
    "task complete", "successfully", "fix applied", "verified",
    "all tests pass", "done.", "patch applied", "implemented ",
    "completed the", "is now fixed", "resolved", "applied the fix",
)


def tool_error_rate(messages: list[dict[str, Any]]) -> float:
    """Fraction of tool_result entries that have ``is_error=True``.

    Returns 0.0 if there are no tool calls at all.
    """
    total = 0
    errors = 0
    for _name, is_error in _iter_tool_pairs(messages or []):
        total += 1
        if is_error:
            errors += 1
    if total == 0:
        return 0.0
    return errors / total


def classify_heuristic(
    *,
    edits_count: int,
    agent_output: str,
    messages: list[dict[str, Any]] | None = None,
) -> SessionVerdict:
    """Classify the satisfied-vs-partial split deterministically.

    Never raises, never calls an LLM, never does JSON parsing.  Uses surface
    signals that are always present at session close:

    - ``edits_count`` (successful Edit/Write calls)
    - ``tool_error_rate`` (fraction of tool calls that errored)
    - Markers in the agent's final natural-language output

    Conservative: when signals are ambiguous, defaults to ``partial``.
    The intent is that ``satisfied`` is only awarded when the evidence
    clearly supports it — so the label is useful as a pass filter.
    """
    output_lower = (agent_output or "").lower()
    err_rate = tool_error_rate(messages or [])

    if edits_count == 0:
        # Shouldn't happen given upstream deterministic short-circuits,
        # but safe default: edits=0 means the task is not "done" by any
        # reasonable measure.
        return SessionVerdict(
            label="partial",
            confidence=0.6,
            rationale="no successful edits despite no hard failure",
            source="deterministic",
        )

    # High tool-error rate → many tool calls failed; likely a messy session
    # that scraped by.
    if err_rate > 0.5:
        return SessionVerdict(
            label="partial",
            confidence=0.7,
            rationale=(
                f"{edits_count} successful edits but {err_rate:.0%} tool-error "
                "rate — many operations failed"
            ),
            source="deterministic",
        )

    has_partial_marker = any(m in output_lower for m in _PARTIAL_MARKERS)
    has_complete_marker = any(m in output_lower for m in _COMPLETE_MARKERS)

    # Partial marker (with or without a concurrent complete marker) → partial.
    # The conservative rule: if the agent itself hedges about incomplete work,
    # trust that hedge even when a completion phrase also appears.
    if has_partial_marker:
        return SessionVerdict(
            label="partial",
            confidence=0.75 if not has_complete_marker else 0.6,
            rationale=(
                f"{edits_count} edits made; agent's final message contains "
                "incomplete-work markers (e.g. 'couldn't', 'incomplete', 'partial')"
            ),
            source="deterministic",
        )

    if has_complete_marker:
        return SessionVerdict(
            label="satisfied",
            confidence=0.8,
            rationale=(
                f"{edits_count} edits made; agent's final message contains "
                "completion markers (e.g. 'done', 'verified', 'applied')"
            ),
            source="deterministic",
        )

    # No markers either way — quiet session.  With edits and low error rate,
    # call it satisfied; otherwise partial.
    if err_rate <= 0.1 and edits_count >= 1:
        return SessionVerdict(
            label="satisfied",
            confidence=0.6,
            rationale=(
                f"{edits_count} edits made with only {err_rate:.0%} tool-error "
                "rate; no explicit incomplete-work markers"
            ),
            source="deterministic",
        )

    return SessionVerdict(
        label="partial",
        confidence=0.5,
        rationale=(
            f"{edits_count} edits made; no clear completion signal "
            f"(error rate {err_rate:.0%})"
        ),
        source="deterministic",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def classify_session(
    *,
    outcome: str | None,
    edits_count: int,
    budget_hit: bool,
    iteration_cap_hit: bool,
    task: str,
    agent_output: str,
    messages: list[dict[str, Any]] | None = None,
    # The following kwargs are accepted for back-compat with earlier
    # callers that passed them; they are now ignored (no LLM path).
    mutation_ledger: list[str] | None = None,
    session_bg_gateway: Any = None,
    session_provider: str | None = None,
    timeout_s: float | None = None,
) -> SessionVerdict:
    """End-to-end classify a session.  Never raises, never calls an LLM.

    Tries the hard deterministic short-circuit first (cancelled / budget /
    iteration cap).  Ambiguous cases fall through to the heuristic
    classifier which reads surface signals (edit count, tool-error rate,
    completion markers).
    """
    try:
        deterministic = classify_deterministic(
            outcome=outcome,
            edits_count=edits_count,
            budget_hit=budget_hit,
            iteration_cap_hit=iteration_cap_hit,
        )
        if deterministic is not None:
            return deterministic

        return classify_heuristic(
            edits_count=edits_count,
            agent_output=agent_output,
            messages=messages,
        )
    except Exception as exc:  # noqa: BLE001 — safety net; never raise
        log.debug("classify_session failed (non-fatal): %s", exc)
        return SessionVerdict(rationale=f"unexpected error: {exc}")
