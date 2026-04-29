"""Per-turn memory surfacing via cheap model side-query.

Scans the typed memory directory, presents frontmatter manifests to a cheap
model, and selects up to 5 relevant memories per turn.  Budget-capped at
60KB cumulative per session (resets on compaction).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

MAX_MEMORIES_PER_TURN = 5
MAX_LINES_PER_MEMORY = 200
MAX_BYTES_PER_MEMORY = 4_096
MAX_SESSION_BYTES = 60_000  # cumulative across session

# System prompt for the memory selection side-query.
_SELECT_MEMORIES_SYSTEM_PROMPT = (
    "You are selecting memories that will be useful to a coding agent. "
    "You will receive a list of memory summaries and the agent's current task. "
    "Return a JSON array of filenames for memories that will clearly be useful "
    "(up to 5). Only include memories you are certain will be helpful. "
    "If unsure, do not include. Do not select usage references for tools the "
    "agent is already actively using — but DO select gotchas, warnings, or "
    "corrections about those tools."
)


class MemorySurfacer:
    """Manages per-turn memory surfacing with budget enforcement.

    Usage::

        surfacer = MemorySurfacer(memory_dir)
        selected = await surfacer.find_relevant(
            query="implement auth middleware",
            gateway=llm_gateway,
            provider_name="anthropic",
        )
        for entry in selected:
            # inject into conversation as <system-reminder>
            ...
    """

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self._surfaced: set[str] = set()
        self._cumulative_bytes: int = 0

    @property
    def budget_remaining(self) -> int:
        """Bytes remaining in the session surfacing budget."""
        return max(0, MAX_SESSION_BYTES - self._cumulative_bytes)

    def reset_budget(self) -> None:
        """Reset the cumulative budget (call after compaction)."""
        self._cumulative_bytes = 0
        self._surfaced.clear()

    async def find_relevant(
        self,
        query: str,
        gateway: object,
        provider_name: str,
    ) -> list[dict[str, str]]:
        """Select and return relevant memories for the current turn.

        Args:
            query: The agent's current task or recent context.
            gateway: LLMGateway instance for the side-query.
            provider_name: Provider name for cheap model resolution.

        Returns:
            List of dicts with 'name', 'content', and 'age' keys.
        """
        if self._cumulative_bytes >= MAX_SESSION_BYTES:
            logger.debug("Memory surfacing budget exhausted (%d bytes)", self._cumulative_bytes)
            return []

        if not self.memory_dir.exists():
            return []

        # Scan memory files for manifests (frontmatter only).
        from maike.memory.taxonomy import MemoryEntry

        manifest_lines: list[str] = []
        available: dict[str, Path] = {}

        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            if str(path) in self._surfaced:
                continue

            entry = MemoryEntry.from_file(path)
            if not entry:
                continue

            manifest_lines.append(f"- {path.name}: [{entry.type.value}] {entry.description}")
            available[path.name] = path

        if not manifest_lines:
            return []

        # Call cheap model to select relevant memories.
        selected_filenames = await self._select_via_llm(
            manifest="\n".join(manifest_lines),
            query=query,
            gateway=gateway,
            provider_name=provider_name,
        )

        # Load selected memories with budget enforcement.
        results: list[dict[str, str]] = []
        for filename in selected_filenames[:MAX_MEMORIES_PER_TURN]:
            if filename not in available:
                continue

            path = available[filename]
            entry = MemoryEntry.from_file(path)
            if not entry:
                continue

            # Truncate content to per-file limits.
            content = entry.content
            lines = content.split("\n")
            if len(lines) > MAX_LINES_PER_MEMORY:
                content = "\n".join(lines[:MAX_LINES_PER_MEMORY])
                content += f"\n... ({len(lines)} lines total, truncated)"
            content_bytes = content.encode("utf-8")
            if len(content_bytes) > MAX_BYTES_PER_MEMORY:
                content = content_bytes[:MAX_BYTES_PER_MEMORY].decode("utf-8", errors="ignore")
                content += "\n... (truncated at 4KB)"

            # Budget check.
            content_size = len(content.encode("utf-8"))
            if self._cumulative_bytes + content_size > MAX_SESSION_BYTES:
                logger.debug("Skipping memory %s: would exceed session budget", filename)
                continue

            self._cumulative_bytes += content_size
            self._surfaced.add(str(path))

            # Compute age string.
            age = _compute_age(entry.created_at)

            results.append({
                "name": entry.name,
                "content": content,
                "age": age,
                "type": entry.type.value,
                "description": entry.description,
            })

        return results

    async def _select_via_llm(
        self,
        manifest: str,
        query: str,
        gateway: object,
        provider_name: str,
    ) -> list[str]:
        """Use a cheap model to select relevant memory filenames."""
        from maike.constants import cheap_model_for_provider

        try:
            cheap_model = cheap_model_for_provider(provider_name)
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Current task: {query}\n\n"
                        f"Available memories:\n{manifest}\n\n"
                        "Return a JSON array of filenames to surface (up to 5). "
                        "Example: [\"auth-patterns.md\", \"deploy-gotcha.md\"]\n"
                        "If no memories are relevant, return []."
                    ),
                },
            ]
            result = await gateway.call(
                system=_SELECT_MEMORIES_SYSTEM_PROMPT,
                messages=messages,
                model=cheap_model,
                temperature=0.0,
                max_tokens=256,
                tools=None,
            )
            text = (result.content or "").strip()
            # Strip markdown code fences if present.
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            import json
            filenames = json.loads(text)
            if isinstance(filenames, list):
                return [f for f in filenames if isinstance(f, str)]
        except Exception as exc:
            logger.debug("Memory selection side-query failed (non-fatal): %s", exc)

        return []


def _compute_age(created_at: str) -> str:
    """Compute a human-readable age string from an ISO timestamp."""
    from datetime import datetime, timezone

    if not created_at:
        return "unknown age"

    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - created
        days = delta.days

        if days == 0:
            return "today"
        elif days == 1:
            return "yesterday"
        else:
            return f"{days} days ago"
    except (ValueError, TypeError):
        return "unknown age"
