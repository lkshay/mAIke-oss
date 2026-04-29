"""LLM-assisted compaction prompt — structured conversation summarization.

Used as a fallback when live session memory is unavailable.  Sends the
conversation to a cheap model with a structured template and returns a
compact summary that replaces the pruned messages.

The prompt uses a two-phase approach:
  1. ``<analysis>`` scratchpad — model drafts its understanding (stripped)
  2. ``<summary>`` block — structured output that becomes the new context
"""

from __future__ import annotations

from typing import Any

COMPACTION_SYSTEM_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

You are compacting a coding agent's conversation history into a structured \
summary. The agent will continue working with only this summary as context \
for everything that happened before.

Your summary MUST preserve:
- The user's original request and any clarifications
- All file paths, function names, and line numbers mentioned
- Error messages and how they were resolved
- Key technical decisions and WHY they were made
- What code was written or modified (include small snippets for critical changes)
- What tests were run and their results
- What work remains incomplete

Write an <analysis> block first (this will be stripped), then a <summary> block.

The <summary> MUST have these sections:
1. **Primary Request**: What the user asked for (1-2 sentences)
2. **Key Technical Context**: Important concepts, patterns, frameworks
3. **Files & Code**: Files read/modified with paths and key content
4. **Errors & Fixes**: Problems encountered and resolutions
5. **Decisions Made**: Technical choices with rationale
6. **All User Messages**: Every user instruction (verbatim if short)
7. **Pending Tasks**: What still needs to be done
8. **Current State**: Most recent work and where things stand
9. **Next Step**: What the agent should do next (be specific)
"""


def build_compaction_messages(
    conversation: list[dict[str, Any]],
    *,
    max_chars: int = 100_000,
) -> list[dict[str, Any]]:
    """Build the messages for an LLM compaction call.

    Includes a compact view of the conversation to be summarized.
    """
    # Build a text representation of the conversation.
    parts: list[str] = []
    total = 0
    for msg in conversation:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = _flatten_blocks(content)
        elif isinstance(content, str):
            text = content
        else:
            continue
        # Truncate individual messages to keep total reasonable.
        if len(text) > 2000:
            text = text[:2000] + "\n[...truncated]"
        line = f"[{role}] {text}"
        if total + len(line) > max_chars:
            parts.append("[...older messages omitted for space]")
            break
        parts.append(line)
        total += len(line)

    return [{"role": "user", "content": "\n\n".join(parts)}]


def extract_summary(response_text: str) -> str | None:
    """Extract the <summary> block from the LLM response.

    Returns None if the response doesn't contain a valid summary.
    """
    import re
    match = re.search(r"<summary>(.*?)</summary>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if no tags, use the whole response if it has section markers.
    if "**Primary Request**" in response_text or "**Current State**" in response_text:
        return response_text.strip()
    return None


def _flatten_blocks(blocks: list[Any]) -> str:
    """Flatten a multi-block content list into text."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = str(block.get("input", {}))[:200]
                parts.append(f"[tool_use: {name}({inp})]")
            elif btype == "tool_result":
                name = block.get("tool_name", block.get("name", "?"))
                content = block.get("content", "")
                if isinstance(content, str):
                    content = content[:300]
                is_error = block.get("is_error", False)
                prefix = "ERROR" if is_error else "result"
                parts.append(f"[{prefix}: {name} → {content}]")
        elif isinstance(block, str):
            parts.append(block[:500])
    return " | ".join(parts)
