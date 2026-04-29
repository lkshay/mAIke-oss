"""Context budget enforcement with progressive compression."""

from __future__ import annotations

import logging
from typing import Any

from maike.constants import (
    CONTEXT_BUDGET_SAFETY_MARGIN,
    DEFAULT_LLM_MAX_TOKENS,
    MODEL_CONTEXT_LIMIT,
    context_limit_for_model,
)
from maike.context.tokenizer import (
    count_message_tokens,
    count_system_prompt_tokens,
    count_tool_schema_tokens,
)

logger = logging.getLogger(__name__)


class ContextBudgetError(RuntimeError):
    """Raised when context cannot be compressed to fit the model's window."""


class ContextBudgetManager:
    """Validates and compresses context to fit within model context windows.

    The progressive compression cascade tries increasingly aggressive strategies:
      Level 1 — strip tool schema descriptions (keep name + input_schema only)
      Level 2 — truncate large artifact content (head + tail)
      Level 3 — reduce environment context blocks
      Level 4 — hard-truncate oldest artifact content
    """

    # ------------------------------------------------------------------ #
    # Estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_payload_tokens(
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
    ) -> int:
        """Accurate token count for the full LLM payload using tiktoken BPE."""
        return (
            count_message_tokens(messages)
            + count_system_prompt_tokens(system_prompt)
            + count_tool_schema_tokens(tool_schemas)
        )

    # ------------------------------------------------------------------ #
    # Budget checks
    # ------------------------------------------------------------------ #

    @classmethod
    def effective_limit(cls, model: str) -> int:
        """Usable input budget = context_limit * safety_margin - output_reserve."""
        raw_limit = context_limit_for_model(model)
        return int(raw_limit * CONTEXT_BUDGET_SAFETY_MARGIN) - DEFAULT_LLM_MAX_TOKENS

    @classmethod
    def fits_budget(
        cls,
        messages: list[dict[str, Any]],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        model: str = "",
    ) -> bool:
        limit = cls.effective_limit(model) if model else int(MODEL_CONTEXT_LIMIT * CONTEXT_BUDGET_SAFETY_MARGIN) - DEFAULT_LLM_MAX_TOKENS
        estimated = cls.estimate_payload_tokens(messages, tool_schemas, system_prompt)
        return estimated <= limit

    # ------------------------------------------------------------------ #
    # Progressive compression
    # ------------------------------------------------------------------ #

    @classmethod
    def compress_to_fit(
        cls,
        messages: list[dict[str, Any]],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        model: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[str]]:
        """Progressively compress *messages* (and optionally *tool_schemas*) until
        they fit.

        Returns ``(compressed_messages, compressed_tool_schemas, levels_applied)``.
        ``compressed_tool_schemas`` is *None* when the schemas were not touched.
        """
        limit = cls.effective_limit(model) if model else int(MODEL_CONTEXT_LIMIT * CONTEXT_BUDGET_SAFETY_MARGIN) - DEFAULT_LLM_MAX_TOKENS
        levels_applied: list[str] = []
        current_schemas = tool_schemas

        def _over() -> bool:
            return cls.estimate_payload_tokens(messages, current_schemas, system_prompt) > limit

        if not _over():
            return messages, None, levels_applied

        # Level 1 — strip tool schema descriptions
        if current_schemas:
            current_schemas = cls._strip_tool_descriptions(current_schemas)
            levels_applied.append("strip_tool_descriptions")
            if not _over():
                return messages, current_schemas, levels_applied

        # Level 2 — strip low-priority guidance blocks (cheapest to discard)
        messages = cls._strip_tagged_blocks(messages, "maike-guidance")
        levels_applied.append("strip_guidance_blocks")
        if not _over():
            return messages, current_schemas, levels_applied

        # Level 3 — compact old thread summaries (keep recent 5 in full,
        # condense older ones to one-liners)
        messages = cls._compact_thread_summaries(messages, keep_recent=5)
        levels_applied.append("compact_thread_summaries")
        if not _over():
            return messages, current_schemas, levels_applied

        # Level 4 — truncate large artifact blocks inside the first user message
        messages = cls._truncate_artifact_blocks(messages, fraction=0.5)
        levels_applied.append("truncate_artifacts_50pct")
        if not _over():
            return messages, current_schemas, levels_applied

        # Level 4 — strip memory and intelligence blocks (reproducible context)
        messages = cls._strip_tagged_blocks(messages, "maike-memory")
        messages = cls._strip_tagged_blocks(messages, "maike-intelligence")
        levels_applied.append("strip_memory_intelligence")
        if not _over():
            return messages, current_schemas, levels_applied

        # Level 5 — strip environment context blocks
        messages = cls._strip_tagged_blocks(messages, "maike-environment")
        messages = cls._strip_environment_blocks(messages)  # legacy fallback
        levels_applied.append("strip_environment_blocks")
        if not _over():
            return messages, current_schemas, levels_applied

        # Level 6 — aggressive artifact truncation
        messages = cls._truncate_artifact_blocks(messages, fraction=0.25)
        levels_applied.append("truncate_artifacts_25pct")
        if not _over():
            return messages, current_schemas, levels_applied

        # If still over, log but return what we have — the gateway guard will catch it.
        logger.warning(
            "Context still exceeds budget after all compression levels "
            "(estimated=%d, limit=%d)",
            cls.estimate_payload_tokens(messages, current_schemas, system_prompt),
            limit,
        )
        return messages, current_schemas, levels_applied

    # ------------------------------------------------------------------ #
    # Compression helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compact_thread_summaries(
        messages: list[dict[str, Any]],
        keep_recent: int = 5,
    ) -> list[dict[str, Any]]:
        """Compact old thread summaries, keeping the most recent in full detail.

        Finds ``<maike-memory>`` blocks containing thread summaries (identified
        by the ``## Prior work on thread`` header).  Keeps the last
        *keep_recent* session entries in full and condenses older entries to
        one-line stubs.
        """
        import re
        result = []
        for msg in messages:
            if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
                result.append(msg)
                continue
            content = msg["content"]
            if "<maike-memory" not in content or "## Prior work on thread" not in content:
                result.append(msg)
                continue

            # Find the thread summary block and split into individual session entries.
            # Sessions are separated by "\n\n---\n\n" or start with "### Session"
            sessions = re.split(r"\n---\n", content)
            if len(sessions) <= keep_recent + 1:  # +1 for the header block
                result.append(msg)
                continue

            # Keep header + recent sessions, condense older ones.
            header = sessions[0]  # includes the <maike-memory> tag + "## Prior work..." header
            older = sessions[1:-keep_recent]
            recent = sessions[-keep_recent:]

            # Condense older sessions to one-liners.
            condensed_lines: list[str] = []
            for entry in older:
                # Extract task line from "Task: ..." and outcome from "Outcome: ..."
                task_match = re.search(r"Task:\s*(.+)", entry)
                outcome_match = re.search(r"Outcome:\s*(.+)", entry)
                task_text = task_match.group(1)[:60] if task_match else "..."
                outcome_text = outcome_match.group(1)[:40] if outcome_match else "..."
                condensed_lines.append(f"- {task_text} → {outcome_text}")

            if condensed_lines:
                compacted_section = f"\n[{len(older)} earlier sessions compacted]:\n" + "\n".join(condensed_lines)
            else:
                compacted_section = ""

            new_content = header + compacted_section + "\n---\n".join([""] + recent)
            result.append({**msg, "content": new_content})

        return result

    @staticmethod
    def _strip_tagged_blocks(
        messages: list[dict[str, Any]],
        tag: str,
    ) -> list[dict[str, Any]]:
        """Remove all ``<tag>...</tag>`` blocks from user message content."""
        from maike.context.tags import strip_tag
        result = []
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                content = msg["content"]
                if f"<{tag}" in content:
                    content = strip_tag(content, tag)
                    if content.strip():
                        result.append({**msg, "content": content})
                    continue
            result.append(msg)
        return result

    @staticmethod
    def _strip_tool_descriptions(
        schemas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove verbose ``description`` fields from tool schemas."""
        compressed: list[dict[str, Any]] = []
        for schema in schemas:
            slim = dict(schema)
            slim.pop("description", None)
            # Also strip parameter-level descriptions.
            input_schema = slim.get("input_schema")
            if isinstance(input_schema, dict):
                slim["input_schema"] = _strip_nested_descriptions(input_schema)
            compressed.append(slim)
        return compressed

    @staticmethod
    def _truncate_artifact_blocks(
        messages: list[dict[str, Any]],
        *,
        fraction: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Truncate artifact content in the first user message.

        Keeps the first ``fraction`` of lines and a small tail for each artifact
        block that exceeds 800 chars.
        """
        if not messages:
            return messages

        first = messages[0]
        content = first.get("content")
        if not isinstance(content, str):
            return messages

        separator = "\n\n---\n\n"
        parts = content.split(separator)
        compressed_parts: list[str] = []
        for part in parts:
            if part.startswith("## Artifact:") and len(part) > 800:
                lines = part.split("\n")
                # Preserve the artifact header (first 5 lines: title, kind, source, path, blank).
                header_lines = lines[:5]
                body_lines = lines[5:]
                if body_lines:
                    keep = max(5, int(len(body_lines) * fraction))
                    tail = min(3, len(body_lines) - keep) if keep < len(body_lines) else 0
                    truncated = body_lines[:keep]
                    if tail > 0:
                        truncated.append(f"\n[...{len(body_lines) - keep - tail} lines omitted...]")
                        truncated.extend(body_lines[-tail:])
                    compressed_parts.append("\n".join(header_lines + truncated))
                else:
                    compressed_parts.append(part)
            else:
                compressed_parts.append(part)
        new_content = separator.join(compressed_parts)
        return [{**first, "content": new_content}, *messages[1:]]

    @staticmethod
    def _strip_environment_blocks(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove environment context blocks from the first user message.

        Environment blocks start with ``## Environment`` or
        ``## Detected Environment``.
        """
        if not messages:
            return messages
        first = messages[0]
        content = first.get("content")
        if not isinstance(content, str):
            return messages

        separator = "\n\n---\n\n"
        parts = content.split(separator)
        filtered = [
            part
            for part in parts
            if not part.lstrip().startswith("## Environment")
            and not part.lstrip().startswith("## Detected Environment")
        ]
        if len(filtered) == len(parts):
            return messages
        return [{**first, "content": separator.join(filtered)}, *messages[1:]]


# ---------------------------------------------------------------------- #
# Internal helpers
# ---------------------------------------------------------------------- #


def _strip_nested_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove ``description`` keys from a JSON Schema dict."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "description":
            continue
        if isinstance(value, dict):
            out[key] = _strip_nested_descriptions(value)
        elif isinstance(value, list):
            out[key] = [
                _strip_nested_descriptions(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            out[key] = value
    return out
