"""Blackboard tools for cross-partition agent communication.

The blackboard is a simple append-only key-value store scoped to a
single session.  Partition agents use it to share interface contracts,
discovered APIs, or coordination signals without requiring direct
communication channels.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.tools.registry import ToolRegistry


class Blackboard:
    """Thread-safe in-memory append-only store for cross-agent communication."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def post(self, key: str, value: str) -> int:
        """Append *value* under *key*. Returns the entry index."""
        with self._lock:
            entries = self._data.setdefault(key, [])
            entries.append(value)
            return len(entries) - 1

    def read(self, key: str | None = None) -> dict[str, list[str]] | list[str]:
        """Read entries.  If *key* is given return its list, else return all."""
        with self._lock:
            if key is None:
                return {k: list(v) for k, v in self._data.items()}
            return list(self._data.get(key, []))

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def register_blackboard_tools(registry: ToolRegistry, blackboard: Blackboard) -> None:
    """Register BlackboardPost and BlackboardRead tools."""

    async def blackboard_post(key: str, value: str) -> ToolResult:
        idx = blackboard.post(key, value)
        return ToolResult(
            tool_name="BlackboardPost",
            success=True,
            output=f"Posted to '{key}' (entry #{idx}).",
            metadata={"key": key, "index": idx},
        )

    async def blackboard_read(key: str | None = None) -> ToolResult:
        data = blackboard.read(key)
        if not data:
            msg = f"No entries for key '{key}'." if key else "Blackboard is empty."
            return ToolResult(tool_name="BlackboardRead", success=True, output=msg)
        return ToolResult(
            tool_name="BlackboardRead",
            success=True,
            output=json.dumps(data, indent=2),
            raw_output=json.dumps(data),
        )

    registry.register(
        ToolSchema(
            name="BlackboardPost",
            description=(
                "Post a message to the shared blackboard for cross-partition communication. "
                "Other partition agents can read your posts. Use to share interface contracts, "
                "API signatures, or coordination signals."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Topic key (e.g., 'interfaces', 'api_contracts', 'shared_types').",
                    },
                    "value": {
                        "type": "string",
                        "description": "The message or data to post.",
                    },
                },
                "required": ["key", "value"],
            },
        ),
        fn=blackboard_post,
        risk_level=RiskLevel.WRITE,
    )

    registry.register(
        ToolSchema(
            name="BlackboardRead",
            description=(
                "Read messages from the shared blackboard. "
                "Omit key to see all topics. Use to check what other partition agents have posted."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Topic key to read. Omit to see all keys and entries.",
                    },
                },
                "required": [],
            },
        ),
        fn=blackboard_read,
        risk_level=RiskLevel.READ,
    )
