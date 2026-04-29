"""Post-run agentic metrics — analyze session DB to measure agent behavior.

All functions operate on data already stored in the session DB
(agent_runs.metadata). No schema changes required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from maike.eval.contracts import AgenticMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Iteration tracking
# ---------------------------------------------------------------------------


def compute_iteration_metrics(
    agent_runs: list[dict[str, Any]],
) -> tuple[int, int]:
    """Return (total_iterations, fix_test_fix_cycles) from agent run metadata."""
    total_iterations = 0
    total_cycles = 0
    for run in agent_runs:
        meta = run.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        total_iterations += meta.get("iteration_count", 0)
        tool_calls = meta.get("tool_calls", [])
        if tool_calls:
            total_cycles += _count_fix_test_fix_cycles(tool_calls)
    return total_iterations, total_cycles


def _count_fix_test_fix_cycles(tool_calls: list[dict[str, Any]]) -> int:
    """Count test-fail → edit → test-pass sequences in a tool call list."""
    cycles = 0
    i = 0
    while i < len(tool_calls):
        tc = tool_calls[i]
        if _is_test_run(tc) and _is_failure(tc):
            # Found a test failure. Look ahead for edit + re-test.
            j = i + 1
            found_edit = False
            while j < len(tool_calls):
                if _is_edit_call(tool_calls[j]):
                    found_edit = True
                elif _is_test_run(tool_calls[j]) and found_edit:
                    if not _is_failure(tool_calls[j]):
                        cycles += 1
                    i = j
                    break
                j += 1
            else:
                i = j
                continue
        i += 1
    return cycles


# ---------------------------------------------------------------------------
# Wasted call detection
# ---------------------------------------------------------------------------


def detect_wasted_calls(
    tool_calls: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Return (total_calls, wasted_reads, wasted_greps).

    Heuristic: a Read is wasted if the file was never edited.
    A Grep is wasted if none of the matched files were edited.
    """
    total = len(tool_calls)
    if total == 0:
        return 0, 0, 0

    # Collect all files that were edited/written during the session.
    edited_files: set[str] = set()
    for tc in tool_calls:
        name = _canonical_tool_name(tc)
        inp = tc.get("input", {})
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except (json.JSONDecodeError, TypeError):
                inp = {}
        if name in ("write", "edit"):
            path = inp.get("path") or inp.get("file_path") or ""
            if path:
                edited_files.add(_normalize(path))

    # Check reads.
    wasted_reads = 0
    for tc in tool_calls:
        name = _canonical_tool_name(tc)
        if name == "read":
            inp = tc.get("input", {})
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except (json.JSONDecodeError, TypeError):
                    inp = {}
            path = inp.get("path") or inp.get("file_path") or ""
            if path and _normalize(path) not in edited_files:
                wasted_reads += 1

    # Check greps.
    wasted_greps = 0
    for tc in tool_calls:
        name = _canonical_tool_name(tc)
        if name == "grep":
            output = tc.get("output", "")
            matched_files = _extract_files_from_output(output)
            if matched_files and not matched_files.intersection(edited_files):
                wasted_greps += 1

    return total, wasted_reads, wasted_greps


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------


def compute_error_recovery(
    tool_calls: list[dict[str, Any]],
) -> tuple[int, int, float]:
    """Return (self_corrections, unrecovered_errors, recovery_rate)."""
    corrections = 0
    unrecovered = 0
    i = 0
    while i < len(tool_calls):
        tc = tool_calls[i]
        if _is_test_run(tc) and _is_failure(tc):
            j = i + 1
            found_edit = False
            while j < len(tool_calls):
                if _is_edit_call(tool_calls[j]):
                    found_edit = True
                elif _is_test_run(tool_calls[j]) and found_edit:
                    if not _is_failure(tool_calls[j]):
                        corrections += 1
                    else:
                        unrecovered += 1
                    i = j
                    break
                j += 1
            else:
                if found_edit:
                    unrecovered += 1
                i = j
                continue
        i += 1
    total = corrections + unrecovered
    # No errors at all = perfect run → rate 1.0, not 0.0.
    rate = 1.0 if total == 0 else corrections / total
    return corrections, unrecovered, rate


# ---------------------------------------------------------------------------
# Delegation tracking
# ---------------------------------------------------------------------------


def detect_delegation_usage(
    tool_calls: list[dict[str, Any]],
) -> tuple[int, int]:
    """Return (sync_delegates, async_delegates) from tool call history."""
    sync = 0
    async_ = 0
    for tc in tool_calls:
        name = _canonical_tool_name(tc)
        if name != "delegate":
            continue
        inp = tc.get("input", {})
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except (json.JSONDecodeError, TypeError):
                inp = {}
        action = inp.get("action")
        if action in ("check", "wait"):
            continue  # Not a spawn, just a status query
        background = inp.get("background", True)  # default is now True
        if background:
            async_ += 1
        else:
            sync += 1
    return sync, async_


# ---------------------------------------------------------------------------
# Multi-file change verification
# ---------------------------------------------------------------------------


def verify_file_changes(
    pre_snapshot: set[str] | tuple[str, ...],
    post_snapshot: set[str] | tuple[str, ...],
    expected_files: tuple[str, ...],
) -> tuple[tuple[str, ...], bool, tuple[str, ...], float]:
    """Compare pre/post file snapshots against expected modifications.

    Returns (files_modified, correct_files_touched, unnecessary, minimality_score).
    """
    pre_set = set(pre_snapshot)
    post_set = set(post_snapshot)
    modified = post_set - pre_set
    # Also detect content changes: files in both but with different content
    # would need content comparison, but snapshots are just file paths.
    # For path-based analysis, "modified" = new files + any files that
    # changed content.  Since we only have paths here, we approximate:
    # all files present post that weren't pre = new/modified.
    # This is a simplification — the full version would hash contents.

    modified_names = tuple(sorted(modified))
    expected_set = set(expected_files)

    correct = expected_set.issubset(modified) if expected_set else True
    unnecessary = tuple(sorted(modified - expected_set))
    minimality = len(expected_set) / max(len(modified), 1) if modified else 1.0

    return modified_names, correct, unnecessary, min(minimality, 1.0)


def verify_file_changes_by_content(
    pre_hashes: dict[str, str],
    post_hashes: dict[str, str],
    expected_files: tuple[str, ...],
) -> tuple[tuple[str, ...], bool, tuple[str, ...], float]:
    """Compare pre/post content hashes to find actually-modified files.

    Unlike verify_file_changes (path-based), this detects files whose
    CONTENT changed, not just files that are new.  Files that exist in
    both snapshots with the same hash are not counted as modified.
    """
    modified: set[str] = set()
    # New files (in post but not in pre).
    for path in post_hashes:
        if path not in pre_hashes:
            modified.add(path)
    # Changed files (in both but different hash).
    for path in post_hashes:
        if path in pre_hashes and post_hashes[path] != pre_hashes[path]:
            modified.add(path)

    modified_names = tuple(sorted(modified))
    expected_set = set(expected_files)
    correct = expected_set.issubset(modified) if expected_set else True
    unnecessary = tuple(sorted(modified - expected_set))
    minimality = len(expected_set) / max(len(modified), 1) if modified else 1.0
    return modified_names, correct, unnecessary, min(minimality, 1.0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def collect_agentic_metrics(
    workspace: Path,
    session_id: str,
    cost_usd: float,
    passed: bool,
    difficulty_weight: float = 1.0,
    expected_modified_files: tuple[str, ...] = (),
    pre_files: set[str] | None = None,
    post_files: set[str] | None = None,
    pre_hashes: dict[str, str] | None = None,
    post_hashes: dict[str, str] | None = None,
) -> AgenticMetrics:
    """Collect all agentic metrics for a completed eval case."""
    import sqlite3 as _sqlite3

    db_path = workspace / ".maike" / "session.db"
    if not db_path.exists():
        return AgenticMetrics()

    conn = _sqlite3.connect(str(db_path))
    conn.row_factory = _sqlite3.Row
    rows = conn.execute(
        "SELECT metadata FROM agent_runs WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    conn.close()
    runs = [{"metadata": row["metadata"]} for row in rows]

    # Parse metadata from runs.
    all_tool_calls: list[dict[str, Any]] = []
    for run in runs:
        meta = run.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        if isinstance(meta, dict):
            all_tool_calls.extend(meta.get("tool_calls", []))

    iterations, cycles = compute_iteration_metrics(runs)
    total, wasted_reads, wasted_greps = detect_wasted_calls(all_tool_calls)
    corrections, unrecovered, recovery_rate = compute_error_recovery(all_tool_calls)
    sync_delegates, async_delegates = detect_delegation_usage(all_tool_calls)

    wasted_ratio = (wasted_reads + wasted_greps) / max(total, 1)
    cost_per_resolved = (cost_usd / difficulty_weight) if passed else None

    # File change verification.
    files_modified: tuple[str, ...] = ()
    correct_files = True
    unnecessary: tuple[str, ...] = ()
    minimality = 1.0
    if pre_hashes is not None and post_hashes is not None:
        files_modified, correct_files, unnecessary, minimality = verify_file_changes_by_content(
            pre_hashes, post_hashes, expected_modified_files,
        )
    elif pre_files is not None and post_files is not None:
        files_modified, correct_files, unnecessary, minimality = verify_file_changes(
            pre_files, post_files, expected_modified_files,
        )

    return AgenticMetrics(
        iteration_count=iterations,
        fix_test_fix_cycles=cycles,
        cost_per_resolved=cost_per_resolved,
        difficulty_weight=difficulty_weight,
        total_tool_calls=total,
        wasted_read_calls=wasted_reads,
        wasted_grep_calls=wasted_greps,
        wasted_call_ratio=wasted_ratio,
        self_corrections=corrections,
        unrecovered_errors=unrecovered,
        error_recovery_rate=recovery_rate,
        sync_delegates=sync_delegates,
        async_delegates=async_delegates,
        files_modified=files_modified,
        expected_files_modified=expected_modified_files,
        correct_files_touched=correct_files,
        unnecessary_files_touched=unnecessary,
        change_minimality_score=minimality,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_tool_name(tc: dict[str, Any]) -> str:
    """Normalize a tool call's name to lowercase canonical form."""
    name = tc.get("resolved_tool_name") or tc.get("tool_name") or tc.get("name") or ""
    return name.lower().strip()


def _is_test_run(tc: dict[str, Any]) -> bool:
    """Check if a tool call is a test execution (pytest, npm test, etc.)."""
    name = _canonical_tool_name(tc)
    if name != "bash":
        return False
    inp = tc.get("input", {})
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except (json.JSONDecodeError, TypeError):
            inp = {}
    cmd = inp.get("cmd") or inp.get("command") or ""
    return any(kw in cmd.lower() for kw in ("pytest", "test", "jest", "mocha", "cargo test"))


def _is_failure(tc: dict[str, Any]) -> bool:
    """Check if a tool call result indicates failure."""
    if tc.get("is_error") or tc.get("success") is False:
        return True
    output = tc.get("output") or tc.get("content") or ""
    return any(kw in output.upper() for kw in ("FAILED", "ERROR", "FAILURE"))


def _is_edit_call(tc: dict[str, Any]) -> bool:
    """Check if a tool call is a file edit/write."""
    name = _canonical_tool_name(tc)
    return name in ("write", "edit", "file_write", "str_replace_editor")


def _normalize(path: str) -> str:
    """Normalize a file path for comparison."""
    return path.strip().lstrip("./")


def _extract_files_from_output(output: str) -> set[str]:
    """Extract file paths from grep/search output."""
    files: set[str] = set()
    for line in output.split("\n"):
        # Grep output format: path:line:content or just path
        if ":" in line:
            path = line.split(":")[0].strip()
            if path and not path.startswith(" ") and "/" in path or "." in path:
                files.add(_normalize(path))
    return files
