"""Memory type taxonomy for structured long-term memory.

Defines four memory types (user, feedback, project, reference) with explicit
save/don't-save rules to constrain what gets persisted to non-derivable
context only.

Memories are stored as individual Markdown files with YAML frontmatter,
indexed via a MEMORY.md file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory type definitions
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    """Four-category memory taxonomy.

    - user:      Role, goals, preferences, knowledge level.
    - feedback:  Corrections AND confirmations of approach.
    - project:   Ongoing work, decisions, incidents not derivable from code.
    - reference: Pointers to external systems (Jira, Slack, dashboards).
    """

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


@dataclass
class MemoryEntry:
    """A single structured memory with type metadata."""

    name: str
    description: str
    type: MemoryType
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    file_path: str | None = None

    def to_frontmatter_md(self) -> str:
        """Serialize to Markdown with YAML frontmatter."""
        return (
            f"---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"type: {self.type.value}\n"
            f"created_at: {self.created_at}\n"
            f"---\n\n"
            f"{self.content}\n"
        )

    @classmethod
    def from_file(cls, path: Path) -> MemoryEntry | None:
        """Parse a memory file with YAML frontmatter.

        Returns None if the file cannot be parsed.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.debug("Failed to read memory file: %s", path)
            return None

        frontmatter, content = _parse_frontmatter(text)
        if not frontmatter:
            return None

        name = frontmatter.get("name", path.stem)
        description = frontmatter.get("description", "")
        type_str = frontmatter.get("type", "reference")
        created_at = frontmatter.get("created_at", "")

        try:
            mem_type = MemoryType(type_str)
        except ValueError:
            mem_type = MemoryType.REFERENCE  # safe fallback

        return cls(
            name=name,
            description=description,
            type=mem_type,
            content=content.strip(),
            created_at=created_at,
            file_path=str(path),
        )


# ---------------------------------------------------------------------------
# MEMORY.md index management
# ---------------------------------------------------------------------------

# Caps from the reference: 200 lines or 25KB, whichever fires first.
MEMORY_INDEX_MAX_LINES = 200
MEMORY_INDEX_MAX_BYTES = 25_000


def build_memory_index(entries: list[MemoryEntry]) -> str:
    """Build a MEMORY.md index from a list of memory entries.

    Each entry becomes a single line: ``- [Name](filename) -- description``

    The index is capped at MEMORY_INDEX_MAX_LINES lines and
    MEMORY_INDEX_MAX_BYTES bytes.
    """
    lines: list[str] = ["# Memory Index", ""]
    for entry in entries:
        filename = Path(entry.file_path).name if entry.file_path else f"{entry.name}.md"
        line = f"- [{entry.name}]({filename}) -- {entry.description}"
        # Cap individual lines at 150 chars for readability
        if len(line) > 150:
            line = line[:147] + "..."
        lines.append(line)

    # Apply line cap
    if len(lines) > MEMORY_INDEX_MAX_LINES + 2:  # +2 for header
        lines = lines[: MEMORY_INDEX_MAX_LINES + 2]
        lines.append(f"\n... ({len(entries)} memories total, index truncated)")

    result = "\n".join(lines) + "\n"

    # Apply byte cap
    if len(result.encode("utf-8")) > MEMORY_INDEX_MAX_BYTES:
        while len(result.encode("utf-8")) > MEMORY_INDEX_MAX_BYTES and len(lines) > 3:
            lines.pop(-1)
            result = "\n".join(lines) + f"\n... (index truncated at {MEMORY_INDEX_MAX_BYTES // 1000}KB)\n"

    return result


def write_memory_index(memory_dir: Path, entries: list[MemoryEntry]) -> None:
    """Write the MEMORY.md index file in the given directory."""
    index_content = build_memory_index(entries)
    index_path = memory_dir / "MEMORY.md"
    index_path.write_text(index_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML-like frontmatter from a markdown file.

    Returns (frontmatter_dict, body_text). Returns ({}, full_text) if no
    frontmatter is found.  Uses simple key:value parsing (no nested YAML).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    fm_text = match.group(1)
    body = text[match.end():]

    fm: dict[str, str] = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        colon_idx = line.find(":")
        if colon_idx > 0:
            key = line[:colon_idx].strip()
            value = line[colon_idx + 1:].strip()
            fm[key] = value

    return fm, body


# ---------------------------------------------------------------------------
# Memory save/don't-save guidance (for prompt injection)
# ---------------------------------------------------------------------------

MEMORY_GUIDANCE = """\
## Memory Management

When saving memories, constrain to context NOT derivable from the current project state:

**What to save:**
- **user**: Your role, goals, preferences, knowledge level
- **feedback**: Corrections and confirmations of approach (include Why + How to apply)
- **project**: Ongoing work, decisions, incidents not derivable from code or git
- **reference**: Pointers to external systems (Jira boards, Slack channels, dashboards)

**What NOT to save:**
- Code patterns, architecture, project structure (derivable from reading the project)
- Git history, file paths, function signatures (derivable from grep/git)
- Information already in MAIKE.md or README

**Memory drift caveat:** A memory that names a specific function, file, or flag is a
claim that it existed *when the memory was written*. It may have been renamed, removed,
or never merged. Before recommending it: if the memory names a file path, check the
file exists. If the memory names a function or flag, grep for it.
"""
