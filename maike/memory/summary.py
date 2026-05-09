"""Session summary builder — generates importance-adaptive summaries.

At session end, produces a text summary of what happened: the agent's own
description of its work, files changed, significant commands run, decisions
made, errors resolved, milestones reached.  No LLM call — pure string
processing reusing the working memory pruner's event extraction.

Summaries have no hard cap.  Detail level adapts to session importance.
The agent's final output text is included untruncated — it's the highest-
value content for cross-session understanding.  The compression cascade
handles total context budget at the model level.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Importance thresholds for detail levels.
_LIGHT_THRESHOLD = 3
_RICH_THRESHOLD = 10

# Bash commands that are ephemeral exploration — not worth preserving.
_NOISE_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "grep", "find", "wc",
    "echo", "pwd", "cd", "tree", "which", "type", "file",
    "sed", "awk", "sort", "uniq", "diff", "less", "more",
    "true", "false", "test", "stat", "readlink", "basename",
    "dirname", "realpath",
})

# Files that are noise in mutation ledgers.
_NOISE_FILES = frozenset({
    "__init__.py", ".gitignore", ".gitkeep",
})
_NOISE_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules",
})


def _git_diff_stat(workspace: Path | None) -> tuple[int, str]:
    """Return (net_lines_added, formatted_stat) from git diff --stat.

    Returns (0, "") if not a git repo root or git is unavailable.
    """
    if workspace is None:
        return 0, ""
    # Only use git if .git is in THIS directory, not inherited from parent.
    if not (workspace / ".git").exists():
        return 0, ""
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--stat", "HEAD~1"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if result.returncode != 0:
            return 0, ""
        stat = result.stdout.strip()
        lines = stat.splitlines()
        if not lines:
            return 0, ""
        last = lines[-1]
        insertions = 0
        deletions = 0
        ins_match = re.search(r"(\d+) insertion", last)
        del_match = re.search(r"(\d+) deletion", last)
        if ins_match:
            insertions = int(ins_match.group(1))
        if del_match:
            deletions = int(del_match.group(1))
        return insertions - deletions, stat
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0, ""


def _is_significant_command(cmd: str) -> bool:
    """Return True if a Bash command is worth preserving in the summary."""
    stripped = cmd.strip()
    if not stripped:
        return False
    # Get the first word, stripping path prefixes
    first_word = stripped.split()[0].split("/")[-1]
    return first_word not in _NOISE_COMMANDS


def _extract_significant_commands(messages: list[dict[str, Any]]) -> list[str]:
    """Extract significant Bash commands from the conversation."""
    commands: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") in ("Bash", "execute_bash"):
                cmd = block.get("input", {}).get("cmd", "") or block.get("input", {}).get("command", "")
                if cmd and _is_significant_command(cmd):
                    # Deduplicate and truncate each command
                    key = cmd[:120]
                    if key not in seen:
                        seen.add(key)
                        commands.append(key)
    return commands


def _is_noise_file(path: str) -> bool:
    """Return True if a file path is noise (not worth listing in summary)."""
    basename = path.rstrip("/").split("/")[-1]
    if basename in _NOISE_FILES:
        return True
    parts = path.split("/")
    return any(p in _NOISE_DIRS for p in parts)


def _deduplicate_mutation_ledger(raw_ledger: str) -> str:
    """Deduplicate the mutation ledger: show each file once with the most significant operation.

    Write > Edit (Write means the file was created, Edit means modified).
    """
    file_ops: dict[str, str] = {}  # path -> "Write" or "Edit"
    for line in raw_ledger.strip().splitlines():
        line = line.strip().lstrip("- ")
        if ": " not in line:
            continue
        op, path = line.split(": ", 1)
        path = path.strip()
        if _is_noise_file(path):
            continue
        # Write beats Edit (creation is more significant than modification)
        if path not in file_ops or op == "Write":
            file_ops[path] = op

    if not file_ops:
        return ""

    # Group by operation
    created = sorted(p for p, op in file_ops.items() if op == "Write")
    modified = sorted(p for p, op in file_ops.items() if op == "Edit")

    parts: list[str] = []
    if created:
        parts.append(f"Files created: {', '.join(created)}")
    if modified:
        parts.append(f"Files modified: {', '.join(modified)}")
    return "\n".join(parts)


class SessionSummaryBuilder:
    """Generates importance-adaptive session summaries."""

    def build_summary(
        self,
        messages: list[dict[str, Any]],
        *,
        task: str,
        outcome: str,
        session_id: str,
        timestamp: str,
        workspace: Path | None = None,
        agent_output: str | None = None,
        verdict: Any = None,
    ) -> str:
        """Build a session summary from the raw conversation messages.

        The ``agent_output`` parameter is the agent's final text response —
        included untruncated as the primary description of what was done.

        The optional ``verdict`` parameter is a ``SessionVerdict`` (typed
        loosely as ``Any`` to avoid a memory→memory import cycle).  If
        present AND has a ``render_line()`` method, a single ``Verdict: ...``
        line is emitted directly after the ``Outcome:`` line.  Back-compat:
        callers that don't pass verdict see the original output.
        """
        from maike.memory.working import WorkingMemory

        wm = WorkingMemory()

        # Extract structured data from messages.
        raw_mutation_ledger = wm._build_mutation_ledger(messages)
        events = wm._extract_events(messages)
        env_state = wm._extract_environment_state(messages)

        # Categorize events.
        milestones = [e for e in events if e.category == "milestone_note"]
        decisions = [e for e in events if e.category == "design_decision"]
        error_fixes = [e for e in events if e.category == "error_fix_sequence"]

        # Deduplicated file list.
        files_section = _deduplicate_mutation_ledger(raw_mutation_ledger)
        new_files = files_section.count("Files created:")
        modified_count = files_section.count(",") + (1 if files_section else 0)

        # Significant commands.
        commands = _extract_significant_commands(messages)

        # Package count from environment state.
        packages = 0
        if "Installed:" in env_state:
            inst_match = re.search(r"Installed:\s*(.+)", env_state)
            if inst_match:
                packages = len([p for p in inst_match.group(1).split(",") if p.strip()])

        # Git stats.
        net_lines, git_stat = _git_diff_stat(workspace)

        # Compute importance score.
        importance = (
            new_files * 3
            + modified_count * 2
            + len(error_fixes) * 2
            + len(milestones) * 3
            + packages * 1
            + net_lines // 50
        )

        # Build summary — no hard cap on length.
        date_short = timestamp[:16].replace("T", " ") if "T" in timestamp else timestamp[:16]
        parts: list[str] = [f"### Session ({date_short})"]
        parts.append(f"Task: {task[:200]}")
        parts.append(f"Outcome: {outcome}")
        # Optional verdict line — purely additive.  Guarded via duck typing
        # to avoid depending on maike.memory.verdict at import time.
        if verdict is not None and hasattr(verdict, "render_line"):
            try:
                line = verdict.render_line()
                if line:
                    parts.append(line)
            except Exception:  # noqa: BLE001 — verdict rendering must not crash summary
                pass

        # Agent's own description of what was done — the highest-value content.
        if agent_output and agent_output.strip():
            parts.append(f"\n{agent_output.strip()}")

        # Deduplicated files — always included if any were changed.
        if files_section:
            parts.append(f"\n{files_section}")

        # Significant commands — medium+ detail.
        if commands and importance >= _LIGHT_THRESHOLD:
            parts.append(f"\nCommands: {', '.join(commands[:10])}")

        # Errors resolved — medium+ detail.
        if error_fixes and importance >= _LIGHT_THRESHOLD:
            parts.append("\nErrors resolved:")
            for ef in error_fixes[:5]:
                parts.append(f"- {ef.content[:150]}")

        # Key decisions — medium+ detail.
        if decisions and importance >= _LIGHT_THRESHOLD:
            parts.append("\nKey decisions:")
            for d in decisions[:5]:
                parts.append(f"- {d.content[:150]}")

        # Key learnings — extract durable project knowledge that would help
        # a future session avoid re-exploration.  Built from decisions, error
        # resolutions, and the file ledger.
        learnings = self._extract_key_learnings(
            decisions=decisions,
            error_fixes=error_fixes,
            files_section=files_section,
            env_state=env_state,
        )
        if learnings and importance >= _LIGHT_THRESHOLD:
            parts.append("\nKey learnings:")
            for learning in learnings[:5]:
                parts.append(f"- {learning}")

        # Environment and git — rich detail.
        if importance >= _RICH_THRESHOLD:
            if env_state.strip():
                parts.append(f"\n{env_state.strip()}")
            if git_stat:
                stat_lines = git_stat.strip().splitlines()
                if stat_lines:
                    parts.append(f"\nGit: {stat_lines[-1].strip()}")

        # Milestones — always included.
        if milestones:
            parts.append("\nMilestones:")
            for m in milestones[:3]:
                parts.append(f"- {m.content[:200]}")

        return "\n".join(parts)

    @staticmethod
    def _extract_key_learnings(
        *,
        decisions: list,
        error_fixes: list,
        files_section: str,
        env_state: str,
    ) -> list[str]:
        """Extract high-level learnings that help future sessions.

        Unlike decisions (which record choices), learnings capture
        knowledge about the project itself — tech stack, patterns
        discovered, pitfalls encountered.
        """
        learnings: list[str] = []

        # From error fixes: what patterns/pitfalls were discovered.
        for ef in error_fixes[:3]:
            content = ef.content
            # Extract the resolution part if it follows "→" or "fixed by" patterns.
            if "→" in content:
                fix = content.split("→", 1)[1].strip()
                learnings.append(f"Pitfall resolved: {fix[:120]}")
            elif len(content) < 150:
                learnings.append(f"Issue resolved: {content}")

        # From environment state: capture tech stack details.
        # Skip the bracketed header line (e.g. "[ENVIRONMENT STATE — ...]")
        # and the "- Approach: ..." line (which is just the last
        # design-decision payload re-included for env-reconstruction
        # purposes — it isn't a durable learning).  Keep concrete
        # tech-stack signals: Venv, Installed, Versions, Tools.
        if env_state.strip():
            kept = 0
            for line in env_state.strip().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    continue
                lower = stripped.lstrip("-* ").lower()
                if lower.startswith("approach:"):
                    continue
                learnings.append(f"Environment: {stripped[:120]}")
                kept += 1
                if kept >= 3:
                    break

        return learnings
