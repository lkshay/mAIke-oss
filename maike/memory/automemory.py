"""Session-end memory extraction into persistent auto-memory.

Extracts durable project knowledge from the session memory and agent output
at session end.  Uses **deterministic parsing** (no LLM call) to avoid
async cancellation issues when the TUI shuts down.

Results are persisted as ``MemoryEntry`` objects via ``TypedLongTermMemory``,
surviving across sessions and loaded into the system prompt on next start.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maike.memory.longterm import TypedLongTermMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_session_memories(
    *,
    session_memory: str | None,
    agent_output: str | None,
    thread_summary: str | None = None,
    typed_memory: "TypedLongTermMemory",
    task: str = "",
) -> int:
    """Extract and persist durable memories from a completed session.

    Purely synchronous — no LLM call, no async, no cancellation risk.
    Parses structured content from three sources (session memory, agent
    output, and thread summary) to build MemoryEntry objects.

    Returns the number of memories written.  Never raises.
    """
    if not session_memory and not agent_output and not thread_summary:
        return 0

    try:
        return _do_extract(
            session_memory=session_memory or "",
            agent_output=agent_output or "",
            thread_summary=thread_summary or "",
            typed_memory=typed_memory,
            task=task,
        )
    except Exception as exc:
        logger.warning(
            "Auto-memory extraction failed: %s: %s",
            type(exc).__name__, str(exc)[:200],
        )
        return 0


def _do_extract(
    *,
    session_memory: str,
    agent_output: str,
    thread_summary: str,
    typed_memory: "TypedLongTermMemory",
    task: str,
) -> int:
    from maike.memory.taxonomy import MemoryEntry, MemoryType, write_memory_index

    existing = typed_memory.list_all()
    existing_by_name = {e.name: e for e in existing}

    # Parse structured sections from session memory.
    # Also parse agent_output as a fallback — short sessions may not
    # produce a session_memory.md but always have agent output.
    sections = _parse_session_memory_sections(session_memory)
    if not sections and agent_output:
        sections = _parse_session_memory_sections(agent_output)
    task_desc = sections.get("task", task) or task
    files = sections.get("files", "")
    decisions = sections.get("decisions", "")
    errors = sections.get("errors", "")

    # Build the project overview memory — the most important one.
    overview_lines: list[str] = []
    if task_desc and task_desc != "(not yet populated)":
        overview_lines.append(f"- Purpose: {task_desc.strip()}")

    # Extract file structure summary from files section.
    file_summary = _extract_file_summary(files)
    if file_summary:
        overview_lines.append(f"- Key modules: {file_summary}")

    # Extract architecture summary from agent output — the richest source.
    # The agent's final response typically has a structured description of
    # what was built, how components connect, and key design patterns.
    # This naturally captures the tech stack in context (e.g. "REST API
    # with FastAPI, SQLAlchemy ORM") without brittle keyword matching.
    arch_summary = _extract_architecture_summary(agent_output, thread_summary)
    if arch_summary:
        overview_lines.append(f"- Architecture: {arch_summary}")

    entries_to_write: list[MemoryEntry] = []

    if overview_lines:
        entries_to_write.append(MemoryEntry(
            name="project_overview",
            description="Project purpose, tech stack, and structure",
            type=MemoryType.PROJECT,
            content="\n".join(overview_lines),
        ))

    # Extract key decisions — from session memory section AND from
    # agent output / thread summary (which often contain richer
    # decision rationale in their architecture descriptions).
    decision_content = _extract_decisions(decisions, agent_output, thread_summary)
    if decision_content:
        entries_to_write.append(MemoryEntry(
            name="key_decisions",
            description="Architectural and design decisions made",
            type=MemoryType.PROJECT,
            content=decision_content,
        ))

    # Extract pitfalls/errors resolved.
    if errors and errors != "(none yet)":
        entries_to_write.append(MemoryEntry(
            name="pitfalls",
            description="Errors encountered and how they were resolved",
            type=MemoryType.FEEDBACK,
            content=errors.strip(),
        ))

    # Write entries to disk.
    written = 0
    for entry in entries_to_write:
        if entry.name in existing_by_name:
            # Update in-place.
            old = existing_by_name[entry.name]
            if old.file_path:
                try:
                    from pathlib import Path
                    Path(old.file_path).write_text(
                        entry.to_frontmatter_md(), encoding="utf-8",
                    )
                    written += 1
                except OSError:
                    pass
                continue
        typed_memory.add(entry)
        written += 1

    if written:
        all_entries = typed_memory.list_all()
        write_memory_index(typed_memory.memory_dir, all_entries)
        logger.info("Auto-memory: wrote %d memories", written)

    return written


# ---------------------------------------------------------------------------
# Section parsing helpers
# ---------------------------------------------------------------------------

_SECTION_HEADERS = {
    "task": re.compile(r"^##\s*Task\b", re.IGNORECASE | re.MULTILINE),
    "files": re.compile(r"^##\s*Files\b", re.IGNORECASE | re.MULTILINE),
    "decisions": re.compile(r"^##\s*Key Decisions\b", re.IGNORECASE | re.MULTILINE),
    "errors": re.compile(r"^##\s*Errors\b", re.IGNORECASE | re.MULTILINE),
}


def _parse_session_memory_sections(text: str) -> dict[str, str]:
    """Parse session memory into named sections."""
    # Find all ## headers and their positions.
    header_re = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    headers = [(m.start(), m.end(), m.group(1).strip()) for m in header_re.finditer(text)]

    sections: dict[str, str] = {}
    for i, (start, end, name) in enumerate(headers):
        next_start = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        content = text[end:next_start].strip()
        # Normalize section names to our keys.
        name_lower = name.lower()
        if "task" in name_lower:
            sections["task"] = content
        elif "file" in name_lower or "function" in name_lower:
            sections["files"] = content
        elif "decision" in name_lower:
            sections["decisions"] = content
        elif "error" in name_lower or "correction" in name_lower:
            sections["errors"] = content

    return sections


def _extract_file_summary(files: str) -> str:
    """Extract a brief summary of key files from the files section."""
    if not files or files == "(not yet populated)":
        return ""
    lines = []
    for line in files.split("\n"):
        line = line.strip()
        if line.startswith(("* ", "- ")) and ":" in line:
            parts = line.lstrip("*- ").split(":", 1)
            if len(parts) == 2:
                name = parts[0].strip().split("/")[-1]  # just filename
                purpose = parts[1].strip()[:60]
                lines.append(f"{name} ({purpose})")
    return ", ".join(lines[:6]) if lines else ""


def _extract_architecture_summary(agent_output: str, thread_summary: str) -> str:
    """Extract a concise architecture summary from agent output and thread summary.

    Looks for structured sections like "Architecture", "Features",
    "Key Implementations" that agents typically include in their final
    response.  Returns a condensed version (key bullet points).
    """
    # Try agent output first — it's the most structured.
    for source in [agent_output, thread_summary]:
        if not source:
            continue
        # Look for architecture/feature headers and extract bullet points.
        arch_re = re.compile(
            r"(?:###?\s*(?:Architect|Feature|Key Implement|How it works|Design).*?\n)"
            r"((?:[-*]\s+.+\n?)+)",
            re.IGNORECASE,
        )
        matches = arch_re.findall(source)
        if matches:
            # Collect unique bullet points, deduped.
            bullets: list[str] = []
            seen: set[str] = set()
            for block in matches:
                for line in block.strip().splitlines():
                    line = line.strip().lstrip("-* ").strip()
                    if line and len(line) > 15 and line.lower() not in seen:
                        seen.add(line.lower())
                        bullets.append(line[:120])
            if bullets:
                return "; ".join(bullets[:5])

    # Fallback: look for bold-labeled items (e.g. "**REST API**: ...")
    bold_re = re.compile(r"\*\*([^*]+)\*\*:\s*(.+?)(?:\n|$)")
    combined = f"{agent_output}\n{thread_summary}"
    bold_matches = bold_re.findall(combined)
    if bold_matches:
        items = [f"{name.strip()}: {desc.strip()[:80]}" for name, desc in bold_matches[:5]]
        return "; ".join(items)

    return ""


def _extract_decisions(
    session_decisions: str,
    agent_output: str,
    thread_summary: str,
) -> str:
    """Extract key decisions from all available sources.

    Combines the session memory's Key Decisions section (often thin) with
    richer decision content from the agent output and thread summary.
    The agent typically documents decisions in code comments and summarizes
    them in the final output — this function captures both.
    """
    decision_lines: list[str] = []
    seen: set[str] = set()

    def _add_line(line: str) -> None:
        normalized = line.strip().lower()[:80]
        if normalized and normalized not in seen:
            seen.add(normalized)
            decision_lines.append(line.strip())

    # Source 1: Session memory's Key Decisions section.
    if session_decisions and session_decisions != "(none yet)":
        for line in session_decisions.strip().splitlines():
            line = line.strip()
            if line.startswith(("* ", "- ")):
                _add_line(line)
            elif line and not line.startswith("#"):
                _add_line(f"- {line}")

    # Source 2: Agent output — look for decision-language patterns.
    # Agents document decisions with phrases like "chose X over Y",
    # "decided to", "using X because", etc.
    _DECISION_PATTERNS = [
        re.compile(r"[-*]\s+\*\*([^*]+)\*\*:\s*(.+)", re.IGNORECASE),  # **Bold label**: explanation
        re.compile(r"[-*]\s+((?:chose|decided|using|selected|opted for|picked)\b.+)", re.IGNORECASE),
    ]
    for source in [agent_output, thread_summary]:
        if not source:
            continue
        for pattern in _DECISION_PATTERNS:
            for match in pattern.finditer(source):
                full = match.group(0).strip().lstrip("-* ")
                if len(full) > 20:
                    _add_line(f"- {full[:150]}")

    return "\n".join(decision_lines[:10]) if decision_lines else ""
