"""Live session memory — a structured summary maintained during conversation.

The ``SessionMemoryService`` runs a lightweight LLM call in the background to
maintain a structured ``.md`` file that captures what the agent is doing, what
it has decided, which files it's working with, and what errors it has hit.

This live memory serves two purposes:
  1. **Compaction source** — when context must be pruned, the session memory
     file can replace the LLM summarisation step (no extra LLM call needed).
  2. **Self-awareness** — after pruning, the agent can read back its own notes
     to regain context about prior work.

The memory is written to ``{workspace}/.maike/session_memory.md`` and is
updated every N tool calls once token usage crosses an init threshold.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from maike.constants import (
    SESSION_MEMORY_FILENAME,
    SESSION_MEMORY_INIT_THRESHOLD,
    SESSION_MEMORY_MAX_TOKENS,
    SESSION_MEMORY_TOKEN_THRESHOLD,
    SESSION_MEMORY_UPDATE_INTERVAL,
)

if TYPE_CHECKING:
    from maike.gateway.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

SESSION_MEMORY_TEMPLATE = """\
# Session Memory

## Current State
*What the agent is actively working on right now. Updated every few tool calls.*

(not yet populated)

## Task
*The user's original request and any clarifications.*

(not yet populated)

## Files & Functions
*Key files read or modified, with one-line purposes.*

(not yet populated)

## Errors & Corrections
*Errors encountered and how they were resolved.*

(none yet)

## Key Decisions
*Design decisions, trade-offs chosen, approaches taken and why.*

(none yet)

## Worklog
*Chronological summary of significant actions taken.*

(not yet started)
"""

SESSION_MEMORY_UPDATE_PROMPT = """\
You are maintaining live session notes for a coding agent. Below is the \
current session memory file and recent conversation messages.

Update the session memory to reflect what happened in the recent messages. \
Rules:
- Keep the exact section headers (## Current State, ## Task, etc.) — do NOT \
rename or remove them.
- Replace "(not yet populated)" / "(none yet)" with actual content.
- Be concise — each section should be a few bullet points, not paragraphs.
- In "Current State", always reflect the MOST RECENT work, not old history.
- In "Files & Functions", list file paths with one-line descriptions.
- In "Key Decisions", capture WHY a decision was made, not just WHAT.
- In "Worklog", add new entries at the bottom. Keep entries brief (one line).
- Total output MUST be under 3000 characters. Compress older worklog entries \
if needed.
- Output ONLY the updated markdown. No preamble, no explanation.
- Do NOT call any tools. Respond with text only.
"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SessionMemoryService:
    """Maintains a live structured session memory file.

    The service is initialised with a workspace path and gateway reference.
    Call :meth:`maybe_update` after each tool-call batch to conditionally
    trigger a background update.
    """

    def __init__(
        self,
        workspace: Path,
        gateway: "LLMGateway",
        *,
        cheap_model: str | None = None,
    ) -> None:
        self._workspace = workspace
        self._gateway = gateway
        self._cheap_model = cheap_model  # resolved lazily if None
        self._memory_path = workspace / ".maike" / SESSION_MEMORY_FILENAME
        self._initialised = False
        self._tool_calls_since_update = 0
        self._tokens_at_last_update = 0

        # Clear stale session memory from prior sessions.  Each new session
        # should start with a clean slate — the old memory would pollute the
        # agent's context with outdated state (wrong task, wrong files, etc.).
        if self._memory_path.exists():
            try:
                self._memory_path.unlink()
            except OSError:
                pass  # best-effort
        self._update_lock = asyncio.Lock()
        self._update_task: asyncio.Task | None = None

    # -- Public API --------------------------------------------------------

    @property
    def memory_path(self) -> Path:
        return self._memory_path

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    def read_memory(self) -> str | None:
        """Read the current session memory file, or None if it doesn't exist."""
        if self._memory_path.exists():
            try:
                return self._memory_path.read_text()
            except OSError:
                return None
        return None

    def record_tool_calls(self, count: int = 1) -> None:
        """Record that *count* tool calls were executed."""
        self._tool_calls_since_update += count

    async def maybe_update(
        self,
        messages: list[dict[str, Any]],
        tokens_used: int,
    ) -> None:
        """Conditionally trigger a session memory update.

        An update fires when ALL of these are true:
          - Token usage has crossed the init threshold (first time) or
            sufficient tool calls / token growth since last update.
          - No update is currently running (non-blocking).
        """
        if not self._should_update(tokens_used):
            return

        # Don't block the agent loop — run in background.
        if self._update_lock.locked():
            return

        self._update_task = asyncio.create_task(
            self._do_update(messages, tokens_used),
        )

    async def wait_for_pending(self) -> None:
        """Wait for any in-flight update to complete (call at session end)."""
        if self._update_task is not None and not self._update_task.done():
            try:
                await asyncio.wait_for(self._update_task, timeout=15)
            except (asyncio.TimeoutError, Exception):
                pass

    # -- Internal ----------------------------------------------------------

    def _should_update(self, tokens_used: int) -> bool:
        """Decide whether an update is warranted."""
        if not self._initialised:
            # First update: wait until we've used enough tokens.
            return tokens_used >= SESSION_MEMORY_INIT_THRESHOLD

        # Subsequent updates: enough tool calls OR enough token growth.
        enough_calls = self._tool_calls_since_update >= SESSION_MEMORY_UPDATE_INTERVAL
        enough_tokens = (tokens_used - self._tokens_at_last_update) >= SESSION_MEMORY_TOKEN_THRESHOLD
        return enough_calls or enough_tokens

    async def _do_update(
        self,
        messages: list[dict[str, Any]],
        tokens_used: int,
    ) -> None:
        """Run the actual LLM-based memory update."""
        async with self._update_lock:
            try:
                current_memory = self.read_memory() or SESSION_MEMORY_TEMPLATE

                # Build a compact view of recent messages for the update prompt.
                recent = self._extract_recent_context(messages)

                update_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"## Current Session Memory\n\n{current_memory}\n\n"
                            f"## Recent Conversation\n\n{recent}"
                        ),
                    },
                ]

                model = self._resolve_cheap_model()
                result = await self._gateway.call(
                    system=SESSION_MEMORY_UPDATE_PROMPT,
                    messages=update_messages,
                    tools=None,
                    model=model,
                    temperature=0.0,
                    max_tokens=2048,
                )

                updated_text = (result.content or "").strip()
                if updated_text and "## Current State" in updated_text:
                    # Sanity check: the response looks like valid session memory.
                    self._memory_path.parent.mkdir(parents=True, exist_ok=True)
                    self._memory_path.write_text(updated_text)
                    self._initialised = True
                    self._tool_calls_since_update = 0
                    self._tokens_at_last_update = tokens_used
                    logger.debug("Session memory updated (%d chars)", len(updated_text))
                else:
                    logger.debug("Session memory update skipped — invalid response")

            except Exception:
                logger.debug("Session memory update failed", exc_info=True)

    def _extract_recent_context(self, messages: list[dict[str, Any]]) -> str:
        """Build a compact text summary of recent messages for the update LLM.

        Keeps the last ~20 messages, truncating tool results to 200 chars each.
        Total budget: ~SESSION_MEMORY_MAX_TOKENS tokens (~16K chars).
        """
        char_budget = SESSION_MEMORY_MAX_TOKENS * 4  # ~4 chars/token estimate
        recent = messages[-20:]
        parts: list[str] = []
        total_chars = 0

        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Multi-block message (tool results, etc.)
                block_parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            text = block.get("text", "")
                            block_parts.append(text[:300])
                        elif btype == "tool_use":
                            name = block.get("name", "?")
                            inp = str(block.get("input", {}))[:150]
                            block_parts.append(f"[tool_use: {name}({inp})]")
                        elif btype == "tool_result":
                            name = block.get("tool_name", block.get("name", "?"))
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                result_content = result_content[:200]
                            elif isinstance(result_content, list):
                                result_content = str(result_content)[:200]
                            is_error = block.get("is_error", False)
                            prefix = "ERROR" if is_error else "result"
                            block_parts.append(f"[{prefix}: {name} → {result_content}]")
                    elif isinstance(block, str):
                        block_parts.append(block[:300])
                line = f"[{role}] " + " | ".join(block_parts)
            elif isinstance(content, str):
                line = f"[{role}] {content[:500]}"
            else:
                continue

            if total_chars + len(line) > char_budget:
                break
            parts.append(line)
            total_chars += len(line)

        return "\n".join(parts)

    def _resolve_cheap_model(self) -> str:
        """Return the cheap model for this gateway's provider."""
        if self._cheap_model:
            return self._cheap_model
        self._cheap_model = self._gateway.resolve_model_for_tier("cheap")
        return self._cheap_model
