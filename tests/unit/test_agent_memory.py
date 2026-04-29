"""Tests for per-agent persistent memory via TypedLongTermMemory."""

from __future__ import annotations

from pathlib import Path

from maike.memory.longterm import TypedLongTermMemory
from maike.memory.taxonomy import MemoryEntry, MemoryType


# ---------------------------------------------------------------------------
# TypedLongTermMemory with custom memory_dir
# ---------------------------------------------------------------------------


def test_custom_memory_dir(tmp_path: Path):
    """TypedLongTermMemory with explicit memory_dir skips .maike/memories."""
    mem_dir = tmp_path / "agent-memory" / "reviewer"
    mem = TypedLongTermMemory(tmp_path, memory_dir=mem_dir)
    assert mem.memory_dir == mem_dir
    assert mem_dir.is_dir()


def test_default_memory_dir(tmp_path: Path):
    """Without memory_dir, uses base_path/.maike/memories."""
    mem = TypedLongTermMemory(tmp_path)
    assert mem.memory_dir == tmp_path / ".maike" / "memories"


# ---------------------------------------------------------------------------
# load_topics_text
# ---------------------------------------------------------------------------


def test_load_topics_text_empty(tmp_path: Path):
    mem = TypedLongTermMemory(tmp_path)
    assert mem.load_topics_text() == ""


def test_load_topics_text_with_entries(tmp_path: Path):
    mem = TypedLongTermMemory(tmp_path)
    mem.add(MemoryEntry(
        name="conventions",
        description="Code review conventions",
        type=MemoryType.FEEDBACK,
        content="Always check error handling.\nPrefer explicit returns.",
    ))
    mem.add(MemoryEntry(
        name="patterns",
        description="Common patterns",
        type=MemoryType.PROJECT,
        content="Use factory functions for complex objects.",
    ))

    text = mem.load_topics_text()
    assert "conventions" in text
    assert "patterns" in text
    assert "feedback" in text.lower()  # type label
    assert "project" in text.lower()


def test_load_topics_text_cap(tmp_path: Path):
    mem = TypedLongTermMemory(tmp_path)
    mem.add(MemoryEntry(
        name="big",
        description="Large memory",
        type=MemoryType.PROJECT,
        content="x" * 5000,
    ))
    mem.add(MemoryEntry(
        name="small",
        description="Small memory",
        type=MemoryType.PROJECT,
        content="y" * 100,
    ))

    # With a tight cap, only the first entry (truncated) should appear.
    text = mem.load_topics_text(cap=2100, per_file_cap=2000)
    assert "big" in text
    # The second entry may or may not fit depending on the first's size.
    # With per_file_cap=2000, the first section is ~2020 chars, so "small" won't fit at cap=2100.


def test_load_topics_text_per_file_cap(tmp_path: Path):
    mem = TypedLongTermMemory(tmp_path)
    mem.add(MemoryEntry(
        name="huge",
        description="Very large content",
        type=MemoryType.PROJECT,
        content="a" * 10000,
    ))

    text = mem.load_topics_text(per_file_cap=500)
    assert len(text) < 600  # section header + 500 chars + truncation note
    assert "truncated" in text


# ---------------------------------------------------------------------------
# Agent-scoped memory round-trip
# ---------------------------------------------------------------------------


def test_agent_scoped_memory_round_trip(tmp_path: Path):
    """Simulate the full lifecycle: write to agent memory, read back."""
    agent_mem_dir = tmp_path / ".maike" / "agent-memory" / "reviewer"

    # Write phase (simulates post-delegate extraction).
    writer = TypedLongTermMemory(tmp_path, memory_dir=agent_mem_dir)
    writer.add(MemoryEntry(
        name="review-style",
        description="How this agent reviews code",
        type=MemoryType.FEEDBACK,
        content="Focus on error handling first. Check for security issues.",
    ))

    # Read phase (simulates next delegate context injection).
    reader = TypedLongTermMemory(tmp_path, memory_dir=agent_mem_dir)
    topics = reader.load_topics_text(cap=4000)
    assert "review-style" in topics
    assert "error handling" in topics

    entries = reader.list_all()
    assert len(entries) == 1
    assert entries[0].name == "review-style"
