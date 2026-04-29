"""Shared utility helpers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def iter_extracted_text(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = " ".join(value.strip().split())
        return [text] if text else []
    if isinstance(value, dict):
        parts: list[str] = []
        for key, nested_value in value.items():
            if key in {"role", "type", "id", "tool_use_id", "is_error"}:
                continue
            parts.extend(iter_extracted_text(nested_value))
        return parts
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(iter_extracted_text(item))
        return parts
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    return []


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in *messages* using tiktoken BPE encoding.

    Delegates to ``maike.context.tokenizer`` for accurate, cached counting.
    Falls back to a conservative ``chars / 3.2`` heuristic only if tiktoken
    is unavailable.
    """
    from maike.context.tokenizer import count_message_tokens

    return count_message_tokens(messages)
