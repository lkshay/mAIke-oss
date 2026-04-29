"""Post-compaction hot context recovery.

After context pruning removes messages from the conversation, critical context
is lost — recently-read files, the active plan, loaded skills.  This module
tracks a "hot context set" during the session and re-injects it after any
compaction event, within a token budget.

Recovery items (in priority order):
  1. MAIKE.md project context (existing, handled by core.py)
  2. Active plan the agent was following
  3. Recently-read files (up to POST_COMPACT_MAX_FILES)
  4. Loaded skills that were in context before compaction
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from maike.constants import (
    POST_COMPACT_MAX_FILES,
    POST_COMPACT_MAX_TOKENS_PER_FILE,
    POST_COMPACT_TOKEN_BUDGET,
)
from maike.utils import estimate_message_tokens

logger = logging.getLogger(__name__)


class PostCompactionRecovery:
    """Tracks hot context during a session and re-injects it after compaction.

    Call :meth:`record_file_read` whenever the agent reads a file, and
    :meth:`record_plan` when the agent outputs a numbered plan.
    After pruning, call :meth:`build_recovery_messages` to get a list of
    messages to insert into the conversation.
    """

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = workspace
        # OrderedDict preserves insertion order; most recent reads at the end.
        # Maps file_path → (line_count, first_line_summary).
        self._recent_reads: OrderedDict[str, _FileReadInfo] = OrderedDict()
        self._active_plan: str | None = None
        self._loaded_skills: dict[str, str] = {}  # skill_name → content
        self._compaction_count: int = 0

    # -- Recording API (called during agent loop) --------------------------

    def record_file_read(
        self,
        file_path: str,
        content: str,
        line_count: int = 0,
    ) -> None:
        """Record that a file was read.  Keeps a summary for re-injection."""
        if not file_path:
            return
        # Build a compact summary: first meaningful lines (imports, class defs).
        summary = _extract_file_summary(content, max_chars=POST_COMPACT_MAX_TOKENS_PER_FILE * 4)
        info = _FileReadInfo(
            path=file_path,
            line_count=line_count or (content.count("\n") + 1),
            summary=summary,
        )
        # Move to end (most recently read).
        self._recent_reads.pop(file_path, None)
        self._recent_reads[file_path] = info
        # Cap the tracking size.
        while len(self._recent_reads) > POST_COMPACT_MAX_FILES * 2:
            self._recent_reads.popitem(last=False)

    def record_plan(self, plan_text: str) -> None:
        """Record the agent's active plan (numbered steps)."""
        if plan_text and len(plan_text) > 20:
            self._active_plan = plan_text

    def record_skill(self, skill_name: str, content: str) -> None:
        """Record a loaded skill for re-injection after compaction."""
        self._loaded_skills[skill_name] = content

    # -- Recovery API (called after compaction) ----------------------------

    def build_recovery_messages(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build recovery messages to re-inject after compaction.

        Returns a list of user messages to insert into the conversation.
        Respects POST_COMPACT_TOKEN_BUDGET.
        """
        self._compaction_count += 1
        messages: list[dict[str, Any]] = []
        tokens_used = 0

        # 0. Warn about stale file content — Edit calls WILL fail if the
        #    agent uses old_text from memory rather than a fresh Read.
        stale_files = [info.path for info in self._recent_reads.values()]
        if stale_files:
            file_list = ", ".join(f"`{p}`" for p in stale_files[-POST_COMPACT_MAX_FILES:])
            warn_msg = {
                "role": "user",
                "content": (
                    "<system-reminder>\n"
                    "[IMPORTANT: Context was compressed. Your memory of file contents "
                    "is now stale. Before using Edit on any file, you MUST Read it first "
                    "to get the current content. Files you were working with: "
                    f"{file_list}]\n"
                    "</system-reminder>"
                ),
            }
            warn_tokens = estimate_message_tokens([warn_msg])
            if warn_tokens <= POST_COMPACT_TOKEN_BUDGET:
                messages.append(warn_msg)
                tokens_used += warn_tokens

        # 1. Re-inject active plan (highest priority after MAIKE.md).
        if self._active_plan:
            plan_msg = {
                "role": "user",
                "content": (
                    "<system-reminder>\n"
                    "[Re-injected after context compression]\n"
                    "## Your Active Plan\n\n"
                    f"{self._active_plan}\n"
                    "</system-reminder>"
                ),
            }
            plan_tokens = estimate_message_tokens([plan_msg])
            if tokens_used + plan_tokens <= POST_COMPACT_TOKEN_BUDGET:
                messages.append(plan_msg)
                tokens_used += plan_tokens

        # 2. Re-inject recently-read files (most recent first, budget-capped).
        files_injected = 0
        # Iterate from most recent to oldest.
        for info in reversed(list(self._recent_reads.values())):
            if files_injected >= POST_COMPACT_MAX_FILES:
                break
            if not info.summary:
                continue

            # Check if file is already mentioned in remaining conversation.
            conv_text = str(conversation[-10:]) if len(conversation) > 10 else str(conversation)
            if info.path in conv_text:
                continue  # Still in context, no need to re-inject.

            file_msg = {
                "role": "user",
                "content": (
                    "<system-reminder>\n"
                    f"[Re-injected after context compression — file you were working with]\n"
                    f"## {info.path} ({info.line_count} lines)\n\n"
                    f"{info.summary}\n"
                    "</system-reminder>"
                ),
            }
            file_tokens = estimate_message_tokens([file_msg])
            if tokens_used + file_tokens > POST_COMPACT_TOKEN_BUDGET:
                break
            messages.append(file_msg)
            tokens_used += file_tokens
            files_injected += 1

        # 3. Re-inject loaded skills.
        for skill_name, content in self._loaded_skills.items():
            skill_msg = {
                "role": "user",
                "content": (
                    "<system-reminder>\n"
                    f"[Re-injected skill after context compression]\n"
                    f"## Skill: {skill_name}\n\n"
                    f"{content[:3000]}\n"
                    "</system-reminder>"
                ),
            }
            skill_tokens = estimate_message_tokens([skill_msg])
            if tokens_used + skill_tokens > POST_COMPACT_TOKEN_BUDGET:
                break
            messages.append(skill_msg)
            tokens_used += skill_tokens

        if messages:
            logger.debug(
                "Post-compaction recovery: %d messages, ~%d tokens "
                "(%d files, plan=%s, %d skills)",
                len(messages), tokens_used, files_injected,
                "yes" if self._active_plan else "no",
                len(self._loaded_skills),
            )

        return messages


class _FileReadInfo:
    """Compact record of a file read for recovery purposes."""

    __slots__ = ("path", "line_count", "summary")

    def __init__(self, path: str, line_count: int, summary: str) -> None:
        self.path = path
        self.line_count = line_count
        self.summary = summary


def _extract_file_summary(content: str, max_chars: int = 5000) -> str:
    """Extract a compact summary of file content for re-injection.

    Keeps the first N chars, prioritising imports, class/function definitions,
    and docstrings.  This is intentionally simple — not AST-based — to work
    across all languages.
    """
    if not content:
        return ""
    if len(content) <= max_chars:
        return content
    # Take first max_chars, try to break at a line boundary.
    truncated = content[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    return truncated + "\n[...truncated for context budget]"
