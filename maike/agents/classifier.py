"""Task complexity classifier for adaptive model tier selection."""

from __future__ import annotations

import re
from typing import Any

from maike.constants import (
    ADAPTIVE_CHEAP_MAX_MESSAGES,
    ADAPTIVE_STRONG_FAILURE_THRESHOLD,
    ADAPTIVE_STRONG_ITERATION_RATIO,
)

# Read-only tool names that indicate an exploration phase.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({"Read", "Grep", "read_file", "grep_codebase", "search_files"})

# Keywords in task descriptions that suggest high complexity.
_STRONG_KEYWORDS: frozenset[str] = frozenset({
    "architecture", "architect", "debug", "debugging", "diagnose",
    "refactor", "redesign", "migrate", "migration", "concurrency",
    "deadlock", "race condition", "security", "vulnerability",
})

# Patterns indicating errors in recent tool output.
_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"error", re.IGNORECASE),
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"exception", re.IGNORECASE),
    re.compile(r"FAILED", re.IGNORECASE),
    re.compile(r"panic", re.IGNORECASE),
)


def _recent_tool_names(conversation: list[dict[str, Any]], window: int = 5) -> list[str]:
    """Extract tool names from recent tool-call assistant messages."""
    names: list[str] = []
    for msg in reversed(conversation):
        if not names and msg.get("role") != "assistant":
            continue
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        names.append(block.get("name", ""))
        if len(names) >= window:
            break
    return names


def _has_error_patterns(conversation: list[dict[str, Any]], window: int = 3) -> bool:
    """Check whether recent user messages (tool results) contain error patterns.

    Skips injected context blocks (thread summaries, memory, environment)
    which may contain error keywords from prior sessions — those should not
    trigger model escalation in the current session.
    """
    checked = 0
    for msg in reversed(conversation):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        text = ""
        if isinstance(content, str):
            # Skip injected context — thread summaries, memory, environment.
            if "<maike-" in content or "Prior work on thread" in content:
                continue
            text = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text += block.get("content", "") + " "
        for pat in _ERROR_PATTERNS:
            if pat.search(text):
                return True
        checked += 1
        if checked >= window:
            break
    return False


def _has_convergence_nudge(conversation: list[dict[str, Any]], window: int = 10) -> bool:
    """Check whether a convergence nudge has been injected recently."""
    checked = 0
    for msg in reversed(conversation):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and "Repeated Failure Detection" in content:
            return True
        if isinstance(content, str) and "convergence" in content.lower():
            return True
        checked += 1
        if checked >= window:
            break
    return False


def _task_has_strong_keywords(conversation: list[dict[str, Any]]) -> bool:
    """Check whether early user messages contain architecture/debugging keywords.

    Skips injected context blocks (thread summaries, memory) which may
    contain keywords like "architecture" from prior session descriptions.
    """
    for msg in conversation[:3]:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            # Skip injected context — only check the actual user task.
            if "<maike-" in content:
                continue
            lower = content.lower()
            if any(kw in lower for kw in _STRONG_KEYWORDS):
                return True
    return False


def classify_task_complexity(
    conversation: list[dict[str, Any]],
    iteration: int,
    failure_count: int,
    has_errors: bool,
    *,
    max_iterations: int = 0,
) -> str:
    """Classify conversation complexity and return a model tier.

    Returns one of ``"default"``, ``"cheap"``, or ``"strong"``.

    Heuristics
    ----------
    **cheap** — when ALL of:
      - Conversation is short (< ADAPTIVE_CHEAP_MAX_MESSAGES messages)
      - No failures recorded
      - Last tool calls were read-only (exploration phase)
      - No error patterns in recent output

    **strong** — when ANY of:
      - Agent has failed >= ADAPTIVE_STRONG_FAILURE_THRESHOLD times
      - Task involves architecture/debugging keywords
      - A convergence nudge has been injected (agent is struggling)
      - Iteration count > ADAPTIVE_STRONG_ITERATION_RATIO of max_iterations

    **default** — everything else.
    """
    # --- strong checks (any triggers escalation) ---
    if failure_count >= ADAPTIVE_STRONG_FAILURE_THRESHOLD:
        return "strong"

    if _task_has_strong_keywords(conversation):
        return "strong"

    if _has_convergence_nudge(conversation):
        return "strong"

    if max_iterations > 0 and iteration > max_iterations * ADAPTIVE_STRONG_ITERATION_RATIO:
        return "strong"

    # --- cheap checks (all must be satisfied) ---
    msg_count = len(conversation)
    if (
        msg_count < ADAPTIVE_CHEAP_MAX_MESSAGES
        and failure_count == 0
        and not has_errors
    ):
        recent_tools = _recent_tool_names(conversation)
        if not recent_tools or all(t in _READ_ONLY_TOOLS for t in recent_tools):
            return "cheap"

    return "default"
