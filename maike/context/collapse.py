"""Context collapse — read-time projection that groups old tool sequences.

Consecutive tool-call/result pairs that are older than a configurable window
are collapsed into single summary messages.  The full messages remain in
WorkingMemory's internal state; only the LLM sees the collapsed projection.

This layer sits between stale clearing (per-result) and full pruning
(conversation replacement).  It reduces token usage without losing any data.

Example collapse:
  [assistant: tool_use Read(app.py)]
  [user: tool_result Read → 200 lines of code]
  [assistant: tool_use Read(config.py)]
  [user: tool_result Read → 50 lines of code]
  [assistant: tool_use Grep(pattern="DATABASE")]
  [user: tool_result Grep → 3 matches]
  →
  [user: "[Collapsed 3 tool calls: Read app.py (200 lines), Read config.py (50 lines),
          Grep 'DATABASE' (3 matches). Re-read files if you need exact content.]"]
"""

from __future__ import annotations

import re
from typing import Any


# Tool sequences older than this many messages from the end are eligible.
COLLAPSE_RECENT_WINDOW = 12

# Minimum consecutive tool pairs to collapse (don't collapse single calls).
MIN_SEQUENCE_LENGTH = 2


def collapse_tool_sequences(
    messages: list[dict[str, Any]],
    *,
    recent_window: int = COLLAPSE_RECENT_WINDOW,
    min_sequence: int = MIN_SEQUENCE_LENGTH,
) -> list[dict[str, Any]]:
    """Collapse old tool-call sequences into summary messages.

    Returns a new list — the input is not mutated.  Messages in the
    recent window (last *recent_window* messages) are never collapsed.
    """
    if len(messages) <= recent_window + 2:
        return messages

    boundary = len(messages) - recent_window
    older = messages[:boundary]
    recent = messages[boundary:]

    collapsed = _collapse_region(older, min_sequence)
    return collapsed + recent


def _collapse_region(
    messages: list[dict[str, Any]],
    min_sequence: int,
) -> list[dict[str, Any]]:
    """Walk through messages, detect tool-call sequences, collapse them."""
    result: list[dict[str, Any]] = []
    i = 0

    while i < len(messages):
        # Try to detect a tool sequence starting at i.
        seq_end, tool_summaries = _detect_tool_sequence(messages, i)

        if len(tool_summaries) >= min_sequence:
            # Collapse the sequence into a single message.
            summary_text = _format_collapsed_sequence(tool_summaries)
            result.append({
                "role": "user",
                "content": summary_text,
            })
            i = seq_end
        else:
            result.append(messages[i])
            i += 1

    return result


def _detect_tool_sequence(
    messages: list[dict[str, Any]],
    start: int,
) -> tuple[int, list[dict[str, str]]]:
    """Detect consecutive tool_use → tool_result pairs starting at *start*.

    Returns (end_index, list_of_tool_summaries).  end_index is the index
    AFTER the last message in the sequence.
    """
    summaries: list[dict[str, str]] = []
    i = start

    while i < len(messages):
        msg = messages[i]

        # Check for assistant message with tool_use blocks.
        if msg.get("role") == "assistant" and _has_tool_use(msg):
            tool_calls = _extract_tool_calls(msg)
            # The next message should be user with tool_result blocks.
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.get("role") == "user" and _has_tool_result(next_msg):
                    results = _extract_tool_results(next_msg)
                    # Pair them up.
                    for tc in tool_calls:
                        matching = next(
                            (r for r in results if r.get("tool_use_id") == tc.get("id")),
                            None,
                        )
                        summary = _summarize_tool_pair(tc, matching)
                        if summary:
                            summaries.append(summary)
                    i += 2  # Skip both messages.
                    continue

        # Not a tool sequence — stop.
        break

    return i, summaries


def _has_tool_use(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_use"
        for b in content
    )


def _has_tool_result(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


def _extract_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    content = msg.get("content", [])
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _extract_tool_results(msg: dict[str, Any]) -> list[dict[str, Any]]:
    content = msg.get("content", [])
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def _summarize_tool_pair(
    tool_call: dict[str, Any],
    tool_result: dict[str, Any] | None,
) -> dict[str, str] | None:
    """Create a one-line summary of a tool call + result pair."""
    name = tool_call.get("name", "?")
    inp = tool_call.get("input", {})

    # Build a brief description based on tool type.
    if name in ("Read", "read_file", "open_file", "view_file"):
        path = inp.get("path", inp.get("file_path", "?"))
        if tool_result:
            content = tool_result.get("content", "")
            lines = content.count("\n") + 1 if isinstance(content, str) else 0
            return {"tool": "Read", "detail": f"{path} ({lines} lines)"}
        return {"tool": "Read", "detail": path}

    if name in ("Grep", "grep_codebase", "search_files"):
        pattern = inp.get("pattern", "?")
        if tool_result:
            content = tool_result.get("content", "")
            matches = content.count("\n") if isinstance(content, str) else 0
            return {"tool": "Grep", "detail": f"'{pattern}' ({matches} matches)"}
        return {"tool": "Grep", "detail": f"'{pattern}'"}

    if name in ("Bash", "execute_bash", "run_command"):
        cmd = inp.get("cmd", inp.get("command", "?"))
        cmd_short = cmd[:60] + "..." if len(str(cmd)) > 60 else cmd
        is_error = tool_result.get("is_error", False) if tool_result else False
        status = "failed" if is_error else "ok"
        return {"tool": "Bash", "detail": f"`{cmd_short}` ({status})"}

    if name in ("Write", "write_file", "create_file"):
        path = inp.get("path", inp.get("file_path", "?"))
        return {"tool": "Write", "detail": path}

    if name in ("Edit", "edit_file", "apply_edit"):
        path = inp.get("path", inp.get("file_path", "?"))
        return {"tool": "Edit", "detail": path}

    # Generic fallback.
    return {"tool": name, "detail": str(inp)[:80]}


def _format_collapsed_sequence(summaries: list[dict[str, str]]) -> str:
    """Format collapsed tool summaries into a single message."""
    lines = [f"[Collapsed {len(summaries)} tool calls:"]
    for s in summaries:
        lines.append(f"  - {s['tool']}: {s['detail']}")
    lines.append("Re-read files if you need exact content.]")
    return "\n".join(lines)
