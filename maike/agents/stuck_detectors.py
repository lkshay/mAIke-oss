"""Deterministic pathology detectors producing candidate nudges.

Each detector walks the conversation BACKWARD from the tail and returns a
``CandidateNudge`` if a pathological pattern is found, else ``None``.  They
are pure (no side effects), stateless, and quick — safe to call every turn.

Detectors emit *candidates*.  A ``TrajectoryAuditor`` validates each candidate
against the task before injection — cheap LLM gate that vetoes patterns which
are legitimate for the specific task (e.g. "summarize this codebase" will
legitimately produce many Reads and no Edits).

Both detectors respect a "has-fired" set on ``ctx.metadata["stuck_fired"]`` so
they only nudge once per kind per session.

Design constraints:
- **Additive**.  They do not replace or modify any existing nudge source
  (``SessionToolTracker``, ``RepeatedFailureTracker``, convergence detection).
- **Conservative thresholds**.  The defaults (15 reads, 8 introspections) are
  well above what healthy exploratory sessions produce; a detector firing is
  strong evidence of pathology.
- **Contiguous only**.  Reset counters when an Edit/Write intervenes — past
  thrashing doesn't matter if the executor is editing now.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal


_STUCK_KINDS = Literal["reads_without_edit", "script_introspection"]

_READ_TOOL_NAMES: frozenset[str] = frozenset({
    "Read", "read_file", "Grep", "grep_codebase", "grep_files", "SemanticSearch",
})
_EDIT_TOOL_NAMES: frozenset[str] = frozenset({
    "Edit", "edit_file", "Write", "write_file", "MultiEdit", "multi_edit",
})

# Interpreter one-liner patterns across common languages.  Each entry:
#   (regex to detect the invocation, regex to extract a "target" token for
#    the repeated-introspection grouping key).
#
# Intent is to catch "agent keeps poking at the same thing via one-liner
# evals" — equally useful for Python (python -c "import x"), Node
# (node -e "require('./x')"), Ruby (ruby -e), Go (go run -), etc.
_INTERPRETER_ONELINER_PATTERNS: list[tuple[re.Pattern[str], re.Pattern[str]]] = [
    # Python: python3 -c "import/from X ..."
    (re.compile(r"^\s*python3?\s+-c\s+[\"']", re.MULTILINE),
     re.compile(r"\b(?:import|from)\s+([\w.]+)")),
    # Node: node -e "require('./x')"
    (re.compile(r"^\s*(?:node|deno)\s+(?:eval|-e)\s+[\"']", re.MULTILINE),
     re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)|from\s+['\"]([^'\"]+)['\"]")),
    # Ruby: ruby -e "require 'x'"
    (re.compile(r"^\s*ruby\s+-e\s+[\"']", re.MULTILINE),
     re.compile(r"require\s+['\"]([^'\"]+)['\"]")),
]


@dataclass
class CandidateNudge:
    """Proposed nudge from a deterministic detector.

    The ``text`` is the pre-audit nudge content.  After the auditor approves
    (optionally rephrasing), this text may or may not be injected verbatim.
    """

    kind: str  # _STUCK_KINDS
    text: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def task_hash(self, task: str) -> str:
        """Stable hash of (kind, task) for per-session caching."""
        h = hashlib.sha1(f"{self.kind}::{task.strip()}".encode("utf-8")).hexdigest()
        return h[:16]


# ---------------------------------------------------------------------------
# Trajectory extraction helpers
# ---------------------------------------------------------------------------


def _iter_tool_events_reversed(conversation: list[dict[str, Any]]):
    """Yield (kind, name, args, is_error) tuples walking backward.

    kind ∈ {"use", "result"}; args is the tool input (for use) or the text
    output (for result — we don't need it here but keep it for extensibility).
    """
    # Pair tool_use with tool_result.  Walk backward to reconstruct latest
    # events first.  We only need tool_use for these detectors.
    for msg in reversed(conversation):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                yield ("use",
                       block.get("name", ""),
                       block.get("input") or {},
                       False)
            elif btype == "tool_result":
                yield ("result",
                       "",
                       {},
                       bool(block.get("is_error")))


def _already_fired(ctx: Any, kind: str) -> bool:
    """Has this detector kind already fired for this session?"""
    try:
        fired = ctx.metadata.setdefault("stuck_fired", set())
    except Exception:
        return False
    # Support both set and list (JSON serialization might turn it into list).
    if isinstance(fired, list):
        return kind in fired
    return kind in fired


def _mark_fired(ctx: Any, kind: str) -> None:
    try:
        fired = ctx.metadata.setdefault("stuck_fired", set())
        if isinstance(fired, set):
            fired.add(kind)
        elif isinstance(fired, list) and kind not in fired:
            fired.append(kind)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Detector 1: N consecutive Read/Grep on same file(s) with 0 Edit between
# ---------------------------------------------------------------------------


def detect_reads_without_edit(
    conversation: list[dict[str, Any]],
    *,
    ctx: Any = None,
    threshold: int = 15,
) -> CandidateNudge | None:
    """Return a candidate nudge if the agent has read a file ≥ threshold times
    with zero successful Edit in between, else ``None``.

    "Reading the same file" = exact path match on Read, Grep, SemanticSearch.
    Counters reset as soon as we hit an Edit/Write (walking backward).
    """
    if ctx is not None and _already_fired(ctx, "reads_without_edit"):
        return None
    if not conversation:
        return None

    # Count contiguous read/grep calls per path from the tail until we hit
    # an Edit/Write.
    counts: dict[str, int] = {}
    total_reads = 0
    for kind, name, args, _is_error in _iter_tool_events_reversed(conversation):
        if kind != "use":
            continue
        if name in _EDIT_TOOL_NAMES:
            # Contiguous reset — if an edit happened recently, the agent is
            # making progress.
            break
        if name not in _READ_TOOL_NAMES:
            continue
        # Get a path-ish key for counting.  Different tools use different
        # arg names.
        path_key = (
            args.get("path")
            or args.get("file_path")
            or args.get("target")
            or args.get("pattern")
            or ""
        )
        path_key = str(path_key) if path_key else "<unknown>"
        counts[path_key] = counts.get(path_key, 0) + 1
        total_reads += 1
        if counts[path_key] >= threshold:
            if ctx is not None:
                _mark_fired(ctx, "reads_without_edit")
            return CandidateNudge(
                kind="reads_without_edit",
                text=(
                    f"You have read `{path_key}` {counts[path_key]} times in a row "
                    f"without making any edits.  If you already understand the "
                    f"relevant code, switch to Edit now — further Reads are "
                    f"unlikely to reveal new information.  If the task is "
                    f"genuinely read-only (e.g. a summary or review), you can "
                    f"safely ignore this nudge."
                ),
                evidence={
                    "path": path_key,
                    "count": counts[path_key],
                    "total_contiguous_reads": total_reads,
                },
            )

    # Also catch the case where many reads span several files but no Edit has
    # happened in the last ``2 * threshold`` reads.
    if total_reads >= 2 * threshold:
        if ctx is not None:
            _mark_fired(ctx, "reads_without_edit")
        top_path = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "<unknown>"
        return CandidateNudge(
            kind="reads_without_edit",
            text=(
                f"You have made {total_reads} Read/Grep calls in a row "
                f"without any Edit.  The most-read path was `{top_path}`.  "
                f"You likely have enough context — try an Edit now.  If the "
                f"task is read-only (summary/review), ignore this nudge."
            ),
            evidence={"total_contiguous_reads": total_reads, "top_path": top_path},
        )

    return None


# ---------------------------------------------------------------------------
# Detector 2: Repeated `python3 -c "..."` on same module without Edit
# ---------------------------------------------------------------------------


def _extract_oneliner_target(cmd: str) -> str | None:
    """Match ``cmd`` against known interpreter one-liner patterns and return
    a "target" string (module name, path, etc.) for grouping repeated calls.

    Returns ``None`` if this isn't an interpreter one-liner.  Language-agnostic
    — catches Python ``python -c``, Node ``node -e``, Ruby ``ruby -e``, etc.
    """
    if not isinstance(cmd, str) or not cmd:
        return None
    first_line = cmd.split("\n", 1)[0]
    for invoke_re, target_re in _INTERPRETER_ONELINER_PATTERNS:
        if not invoke_re.match(first_line):
            continue
        # Extract the first target from the whole command body (the -c/-e
        # payload often spans multiple lines for Python heredocs).
        matches = target_re.findall(cmd)
        if matches:
            target = matches[0]
            # findall with groups may return tuples; flatten to the first
            # non-empty group.
            if isinstance(target, tuple):
                target = next((t for t in target if t), "")
            target = str(target).split(".")[0].strip()
            return target or "<script>"
        return "<script>"
    return None


def detect_script_introspection_loop(
    conversation: list[dict[str, Any]],
    *,
    ctx: Any = None,
    threshold: int = 8,
) -> CandidateNudge | None:
    """Return a candidate nudge if the agent has run interpreter one-liner
    introspection calls targeting the same module ≥ threshold times with
    zero successful Edit in between.

    Language-agnostic: matches Python ``python3 -c``, Node ``node -e``,
    Ruby ``ruby -e``, etc.  Catches the django-16263 pattern (Pro poking at
    Django internals via ``python3 -c "from django.db ..."`` 18 times without
    editing) AND the equivalent patterns in other languages.
    """
    if ctx is not None and _already_fired(ctx, "script_introspection"):
        return None
    if not conversation:
        return None

    target_counts: dict[str, int] = {}
    total_calls = 0

    for kind, name, args, _is_error in _iter_tool_events_reversed(conversation):
        if kind != "use":
            continue
        if name in _EDIT_TOOL_NAMES:
            break  # contiguous reset
        if name not in {"Bash", "execute_bash"}:
            continue
        cmd = args.get("cmd") or ""
        target = _extract_oneliner_target(cmd)
        if target is None:
            continue
        total_calls += 1
        target_counts[target] = target_counts.get(target, 0) + 1
        if target_counts[target] >= threshold:
            if ctx is not None:
                _mark_fired(ctx, "script_introspection")
            return CandidateNudge(
                kind="script_introspection",
                text=(
                    f"You have run interpreter one-liners targeting "
                    f"`{target}` {target_counts[target]} times in a row "
                    f"without making any edits.  Introspection is useful, "
                    f"but at this point you likely have enough signal — "
                    f"switch to Edit on the relevant file, or ask for "
                    f"guidance if the bug location is still unclear."
                ),
                evidence={
                    "target": target,
                    "count": target_counts[target],
                    "total_introspection_calls": total_calls,
                },
            )

    return None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def detect_all_stuck_patterns(
    conversation: list[dict[str, Any]],
    *,
    ctx: Any = None,
    reads_threshold: int = 15,
    script_threshold: int = 8,
) -> list[CandidateNudge]:
    """Run all detectors in a fixed order.  Returns the list of firings
    (zero or more).  Both detectors self-limit to one fire per kind per
    session via ``ctx.metadata["stuck_fired"]``.
    """
    out: list[CandidateNudge] = []
    r1 = detect_reads_without_edit(conversation, ctx=ctx, threshold=reads_threshold)
    if r1 is not None:
        out.append(r1)
    r2 = detect_script_introspection_loop(conversation, ctx=ctx, threshold=script_threshold)
    if r2 is not None:
        out.append(r2)
    return out
