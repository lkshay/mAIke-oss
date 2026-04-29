"""Thin gateway over session and long-term memory."""

from __future__ import annotations

from maike.memory.longterm import LongTermMemory
from maike.memory.session import SessionStore


class MemoryGateway:
    def __init__(self, session_store: SessionStore, longterm: LongTermMemory) -> None:
        self.session_store = session_store
        self.longterm = longterm

    async def remember_decision(self, collection: str, key: str, content: str) -> None:
        self.longterm.add(collection=collection, key=key, content=content)

    async def recall(self, collection: str, query: str, limit: int = 5) -> list[dict[str, str]]:
        return self.longterm.query(collection=collection, query=query, limit=limit)

