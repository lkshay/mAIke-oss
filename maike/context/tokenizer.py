"""Production-grade token counting using tiktoken.

Provides accurate BPE-based token counts instead of the naive ``len(text) // 4``
heuristic.  Code tokens average ~2.5-3.5 chars/token (not 4), so the old
heuristic underestimates by 15-40% — causing context overflows and wasted budget.

Architecture
------------
* ``count_tokens(text)`` — count tokens for a single string (cached).
* ``count_message_tokens(messages)`` — count tokens for a full conversation.
* ``count_tool_schema_tokens(schemas)`` — count tokens for tool definitions.
* ``TokenCounter`` — stateful counter with per-content LRU cache.

The module uses ``cl100k_base`` encoding which covers Claude, GPT-4, and most
modern models.  A ``chars // 4`` fallback is used only if tiktoken fails to
import (should never happen in production).
"""

from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tiktoken setup with graceful fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover — tiktoken should always be available
    _ENCODING = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False
    logger.warning(
        "tiktoken not available — falling back to chars//4 heuristic. "
        "Install tiktoken for accurate token counting: pip install tiktoken"
    )

# Per-message overhead: role token + structural delimiters.
_MESSAGE_OVERHEAD_TOKENS = 4
# System prompt framing overhead.
_SYSTEM_PROMPT_OVERHEAD = 4


# ---------------------------------------------------------------------------
# Core counting
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2048)
def _cached_encode_len(text: str) -> int:
    """Encode *text* and return the token count.  Results are LRU-cached."""
    if _HAS_TIKTOKEN:
        return len(_ENCODING.encode(text, disallowed_special=()))
    # Fallback: conservative estimate (chars / 3.2 for code-heavy content).
    return max(1, int(len(text) / 3.2))


def count_tokens(text: str) -> int:
    """Count tokens in *text* using tiktoken (cached)."""
    if not text:
        return 0
    # For very long strings, skip the cache (they won't be repeated often and
    # would evict more useful cache entries).
    if len(text) > 100_000:
        if _HAS_TIKTOKEN:
            return len(_ENCODING.encode(text, disallowed_special=()))
        return max(1, int(len(text) / 3.2))
    return _cached_encode_len(text)


# ---------------------------------------------------------------------------
# Message-level counting
# ---------------------------------------------------------------------------


def _extract_text_from_value(value: Any) -> str:
    """Recursively extract text content from a message value."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts: list[str] = []
        for key, nested in value.items():
            if key in ("role", "type", "id", "tool_use_id", "is_error"):
                continue
            parts.append(_extract_text_from_value(nested))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_extract_text_from_value(item) for item in value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens for a full message list, including per-message overhead.

    This mirrors how the API counts tokens:
    - Each message costs ~4 tokens for role/delimiters
    - Content is tokenized with BPE
    - Tool use blocks include the JSON serialization overhead
    """
    total = 0
    for message in messages:
        total += _MESSAGE_OVERHEAD_TOKENS
        text = _extract_text_from_value(message)
        total += count_tokens(text)
    return total


def count_system_prompt_tokens(system_prompt: str) -> int:
    """Count tokens for a system prompt including framing overhead."""
    if not system_prompt:
        return 0
    return count_tokens(system_prompt) + _SYSTEM_PROMPT_OVERHEAD


def count_tool_schema_tokens(schemas: list[dict[str, Any]] | None) -> int:
    """Count tokens for tool schema definitions.

    Tool schemas are serialized as text in the API request. Each tool
    contributes its name, description, and JSON Schema for parameters.
    """
    if not schemas:
        return 0
    total = 0
    for schema in schemas:
        # Serialize the schema dict to approximate what the API sees.
        text = str(schema)
        total += count_tokens(text)
    return total


def estimate_payload_tokens(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
) -> int:
    """Estimate total tokens for a full LLM API payload.

    Combines message tokens, system prompt tokens, and tool schema tokens.
    """
    return (
        count_message_tokens(messages)
        + count_system_prompt_tokens(system_prompt)
        + count_tool_schema_tokens(tool_schemas)
    )


# ---------------------------------------------------------------------------
# TokenCounter — stateful wrapper for telemetry / budget tracking
# ---------------------------------------------------------------------------


class TokenCounter:
    """Stateful token counter that tracks cumulative counts.

    Useful for telemetry: tracks how many tokens have been counted and
    provides cache hit statistics.
    """

    def __init__(self) -> None:
        self._total_counted: int = 0
        self._call_count: int = 0

    @property
    def total_counted(self) -> int:
        return self._total_counted

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def has_tiktoken(self) -> bool:
        return _HAS_TIKTOKEN

    @property
    def cache_info(self) -> dict[str, int]:
        info = _cached_encode_len.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "currsize": info.currsize,
        }

    def count(self, text: str) -> int:
        tokens = count_tokens(text)
        self._total_counted += tokens
        self._call_count += 1
        return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        tokens = count_message_tokens(messages)
        self._total_counted += tokens
        self._call_count += 1
        return tokens

    def count_payload(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
    ) -> int:
        tokens = estimate_payload_tokens(messages, tool_schemas, system_prompt)
        self._total_counted += tokens
        self._call_count += 1
        return tokens
