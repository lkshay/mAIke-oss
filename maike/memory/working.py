"""Working-memory pruning helpers with priority-based semantic summarization."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maike.constants import (
    DEFAULT_LLM_MAX_TOKENS,
    DESIGN_DECISION_KEYWORDS,
    MAX_PRUNED_EVENTS,
    PRUNE_THRESHOLD,
    TOOL_RESULT_TRUNCATE_LIMIT,
    prune_threshold_for_model,
)
from maike.utils import estimate_message_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event categories and priorities
# ---------------------------------------------------------------------------

_CATEGORY_PRIORITY = {
    "error": 0,                # Always kept
    "design_decision": 0,     # Decision-bearing reasoning — never discard
    "milestone_note": 0,      # Progress milestones — survive all pruning
    "error_fix_sequence": 0,  # Atomic error→fix chains — never split
    "file_write": 0,          # Never prune — mutation history is critical
    "decision": 2,            # Important reasoning context
    "file_read": 3,           # Useful but reproducible
    "command": 4,             # Least critical — can re-run
}

# Per-category truncation limits for pruning summaries.
# Errors need full tracebacks; file reads are reproducible (agent can re-read).
_TRUNCATE_LIMITS = {
    "error": 800,
    "file_write": 400,
    "file_read": 200,
    "command": 300,
    "decision": 200,
    "design_decision": 400,
    "milestone_note": 400,
    "error_fix_sequence": 800,
}

_WRITE_TOOL_NAMES = {"Write", "Edit", "write_file", "edit_file", "delete_file"}

_CLEARABLE_TOOL_NAMES = {
    "Read", "Grep", "Bash",
    "read_file", "grep_codebase", "search_files",
    "execute_bash", "list_dir", "repo_map",
    "SemanticSearch",
}

_READ_TOOL_NAMES = {"Read", "read_file", "open_file", "view_file"}

_BASH_GREP_TOOL_NAMES = _CLEARABLE_TOOL_NAMES - _READ_TOOL_NAMES

# Patterns for extracting semantic hints from pruned tool results.
_SYMBOL_RE = re.compile(
    r"^(?:class|def|function|export|import|from|interface|struct|type|enum)\s+(\w+)",
    re.MULTILINE,
)


def _semantic_hint(content: str, max_symbols: int = 5) -> str:
    """Extract a brief semantic hint from tool result content.

    Returns a string like ", contains: class Foo, def bar, def baz" or empty
    string if nothing interesting found.  This gives the agent breadcrumbs
    about what was in a pruned result without keeping the full content.
    """
    if not content or len(content) < 50:
        return ""
    symbols = _SYMBOL_RE.findall(content)
    if not symbols:
        return ""
    unique = list(dict.fromkeys(symbols))[:max_symbols]
    return ", contains: " + ", ".join(unique)


@dataclass
class PrunedEvent:
    """A structured event extracted from a message for priority-based pruning."""

    category: str  # error, design_decision, error_fix_sequence, file_write, decision, file_read, command
    content: str
    priority: int


class WorkingMemory:
    def __init__(self, *, session_memory_path: Path | None = None) -> None:
        self._session_memory_path = session_memory_path

    def prune(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
    ) -> list[dict[str, Any]]:
        threshold = prune_threshold_for_model(model) if model else PRUNE_THRESHOLD
        return self._prune(messages, target_tokens=threshold)

    def prune_to_target(
        self,
        messages: list[dict[str, Any]],
        *,
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        """Prune when conversation exceeds *target_tokens*."""
        return self._prune(messages, target_tokens=target_tokens)

    def prune_to_budget(
        self,
        messages: list[dict[str, Any]],
        *,
        token_budget: int,
        reserve_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        model: str = "",
    ) -> list[dict[str, Any]]:
        cap = prune_threshold_for_model(model) if model else PRUNE_THRESHOLD
        target_tokens = min(cap, self._budget_prune_target(token_budget, reserve_tokens))
        return self._prune(messages, target_tokens=target_tokens)

    # Content-aware recent window thresholds.  Expand backward from the end
    # of the conversation until BOTH minimums are met, up to the hard cap.
    _RECENT_MIN_TOKENS = 10_000   # At least this many tokens in the window
    _RECENT_MIN_TEXT_BLOCKS = 5   # At least this many messages with text
    _RECENT_MAX_TOKENS = 40_000   # Stop expanding beyond this

    def _effective_recent_window(self, messages: list[dict[str, Any]]) -> int:
        """Content-aware recent window: expand backward until both a token
        minimum AND a text-message minimum are met, up to a hard token cap.

        This adapts to actual content density — a session of large file reads
        keeps fewer messages than a session of short tool calls.  Same model,
        different behavior, correct in both cases.
        """
        if not messages:
            return 0

        total = len(messages)
        tokens_acc = 0
        text_blocks = 0
        window = 0

        for i in range(total - 1, -1, -1):
            msg = messages[i]
            msg_tokens = estimate_message_tokens([msg])
            tokens_acc += msg_tokens
            window += 1

            # Count text-bearing messages (assistant text or user text,
            # excluding bare tool_result / tool_use-only messages).
            if self._has_text_content(msg):
                text_blocks += 1

            # Stop if we've hit the hard cap.
            if tokens_acc >= self._RECENT_MAX_TOKENS:
                break

            # Stop if BOTH minimums are satisfied.
            if tokens_acc >= self._RECENT_MIN_TOKENS and text_blocks >= self._RECENT_MIN_TEXT_BLOCKS:
                break

        return min(window, total)

    @staticmethod
    def _has_text_content(msg: dict[str, Any]) -> bool:
        """Check if a message contains meaningful text (not just tool calls)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(
                isinstance(b, dict) and b.get("type") in ("text", "output_text")
                and b.get("text", "").strip()
                for b in content
            )
        return False

    def _clear_stale_tool_results(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace content of stale, non-error tool results with a stub.

        This reduces token count before summarization by clearing the bulky
        output of read-only / reproducible tools (Read, Grep, Bash, etc.)
        while preserving tool_use blocks, error results, and Write/Edit results.
        """
        result: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                result.append(message)
                continue
            new_blocks: list[Any] = []
            changed = False
            for block in content:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue
                if block.get("type") != "tool_result":
                    new_blocks.append(block)
                    continue
                tool_name = block.get("tool_name") or block.get("name") or ""
                is_error = block.get("is_error", False)
                if tool_name in _CLEARABLE_TOOL_NAMES and not is_error:
                    new_block = {**block, "content": "[cleared — re-fetch if needed]"}
                    new_blocks.append(new_block)
                    changed = True
                    if tool_name in _READ_TOOL_NAMES:
                        file_path = self._extract_read_file_path(block)
                        if file_path:
                            try:
                                from maike.tools.filesystem import _file_read_state
                                _file_read_state.clear(file_path)
                            except Exception:
                                pass
                else:
                    new_blocks.append(block)
            if changed:
                result.append({**message, "content": new_blocks})
            else:
                result.append(message)
        return result

    def clear_stale_tool_results(
        self,
        messages: list[dict[str, Any]],
        *,
        recent_window: int = 8,
        mutated_paths: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Replace content of stale tool results not referenced recently.

        Extends the original Read-only clearing to also compress old Bash
        and Grep results (>500 chars) that aren't referenced in recent
        messages.  Lighter than a full prune — runs between prune events
        to keep stale tool output from consuming context tokens.
        """
        if len(messages) <= recent_window + 2:
            return messages

        # Build recently-referenced file set from the last `recent_window` messages.
        recent_files: set[str] = set()
        for msg in messages[-recent_window:]:
            self._extract_file_paths(msg, recent_files)

        # Mutation ledger files are always considered active.
        if mutated_paths:
            recent_files.update(mutated_paths)

        # Collect text snippets from recent messages for reference checking.
        recent_text = self._collect_recent_text(messages[-recent_window:])

        result: list[dict[str, Any]] = []
        older = messages[:-recent_window]
        for message in older:
            content = message.get("content")
            if not isinstance(content, list):
                result.append(message)
                continue
            new_blocks: list[Any] = []
            changed = False
            for block in content:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue
                if block.get("type") != "tool_result":
                    new_blocks.append(block)
                    continue
                tool_name = block.get("tool_name") or block.get("name") or ""
                is_error = block.get("is_error", False)
                if is_error:
                    new_blocks.append(block)
                    continue

                # --- Pass 1: Read tool clearing (path-based) ---
                if tool_name in _READ_TOOL_NAMES:
                    file_path = self._extract_read_file_path(block)
                    if file_path and file_path not in recent_files:
                        old_content = block.get("content", "")
                        line_count = old_content.count("\n") + 1 if old_content else 0
                        hint = _semantic_hint(old_content)
                        stub = f"[Read {file_path} — {line_count} lines{hint}; content pruned, re-read if needed]"
                        new_blocks.append({**block, "content": stub})
                        changed = True
                        # Invalidate read state: the content this Read produced is
                        # no longer in the agent's view, so Edit must not accept an
                        # old_text derived from the agent's (now possibly fabricated)
                        # memory of it.  Next Edit on this path requires a fresh Read.
                        try:
                            from maike.tools.filesystem import _file_read_state
                            _file_read_state.clear(file_path)
                        except Exception:
                            pass
                    else:
                        new_blocks.append(block)
                    continue

                # --- Pass 2: Bash/Grep/SemanticSearch clearing (size-based) ---
                if tool_name in _BASH_GREP_TOOL_NAMES:
                    block_content = block.get("content", "")
                    char_count = len(block_content) if isinstance(block_content, str) else 0
                    if char_count > 500 and not self._content_referenced_recently(block_content, recent_text):
                        hint = _semantic_hint(block_content)
                        stub = f"[{tool_name} result — {char_count} chars{hint}; content pruned]"
                        new_blocks.append({**block, "content": stub})
                        changed = True
                    else:
                        new_blocks.append(block)
                    continue

                new_blocks.append(block)
            if changed:
                result.append({**message, "content": new_blocks})
            else:
                result.append(message)
        result.extend(messages[-recent_window:])
        return result

    def clear_stale_reads(
        self,
        messages: list[dict[str, Any]],
        *,
        recent_window: int = 8,
        mutated_paths: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Deprecated alias for :meth:`clear_stale_tool_results`."""
        return self.clear_stale_tool_results(
            messages, recent_window=recent_window, mutated_paths=mutated_paths,
        )

    def compress_duplicate_failures(
        self,
        messages: list[dict[str, Any]],
        failure_hashes: dict[str, int] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Replace duplicate error outputs with back-references.

        For each tool_result with ``is_error=True``, compute an MD5 hash of
        the last 30 lines.  If the same hash was already seen, replace the
        content with a short stub pointing to the first occurrence.

        Returns the (possibly modified) message list and the updated hash
        dict so the caller can accumulate across calls.
        """
        if failure_hashes is None:
            failure_hashes = {}

        result: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                result.append(message)
                continue
            new_blocks: list[Any] = []
            changed = False
            for block in content:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue
                if block.get("type") != "tool_result" or not block.get("is_error", False):
                    new_blocks.append(block)
                    continue
                block_content = block.get("content", "")
                if not isinstance(block_content, str) or not block_content:
                    new_blocks.append(block)
                    continue
                # Hash last 30 lines
                lines = block_content.splitlines()
                tail = "\n".join(lines[-30:])
                h = hashlib.md5(tail.encode()).hexdigest()
                if h in failure_hashes:
                    first_iter = failure_hashes[h]
                    stub = (
                        f"Same error as iteration {first_iter} "
                        f"(hash: {h[:8]}). See earlier output for full traceback."
                    )
                    new_blocks.append({**block, "content": stub})
                    changed = True
                else:
                    failure_hashes[h] = idx
                    new_blocks.append(block)
            if changed:
                result.append({**message, "content": new_blocks})
            else:
                result.append(message)
        return result, failure_hashes

    @staticmethod
    def _collect_recent_text(messages: list[dict[str, Any]]) -> str:
        """Concatenate text from recent messages for reference checking."""
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        bc = block.get("content", "")
                        if isinstance(bc, str):
                            parts.append(bc)
                        inp = block.get("input")
                        if isinstance(inp, dict):
                            for v in inp.values():
                                if isinstance(v, str):
                                    parts.append(v)
        return "\n".join(parts)

    @staticmethod
    def _content_referenced_recently(block_content: str, recent_text: str) -> bool:
        """Check if any file-path token from *block_content* appears in recent text.

        Samples up to 50 tokens from the block and checks whether any
        file-path-like token appears in the recent conversation text.
        This avoids clearing results the agent is still actively working
        with.
        """
        if not block_content or not recent_text:
            return False
        # Check if any file-like token from the block appears in recent text
        for token in block_content.split()[:50]:  # sample first 50 tokens
            if "/" in token and not token.startswith("http"):
                clean = token.strip("'\"`,;:()[]")
                if clean and clean in recent_text:
                    return True
        return False

    @staticmethod
    def _extract_file_paths(message: dict[str, Any], out: set[str]) -> None:
        """Collect file path strings mentioned in a message."""
        content = message.get("content")
        if isinstance(content, str):
            # Quick heuristic: lines that look like file paths
            for token in content.split():
                if "/" in token and not token.startswith("http"):
                    out.add(token.strip("'\"`,;:()[]"))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                # tool_use input may have path/file_path
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                for key in ("path", "file_path", "file"):
                    val = inp.get(key)
                    if isinstance(val, str) and val:
                        out.add(val)
                # tool_result content may mention files
                block_content = block.get("content", "")
                if isinstance(block_content, str):
                    for token in block_content.split():
                        if "/" in token and not token.startswith("http"):
                            out.add(token.strip("'\"`,;:()[]"))

    @staticmethod
    def _extract_read_file_path(block: dict[str, Any]) -> str | None:
        """Extract the file path from a Read tool_result block."""
        # The block may have input metadata from the tool_use
        inp = block.get("input")
        if isinstance(inp, dict):
            for key in ("path", "file_path", "file"):
                val = inp.get(key)
                if isinstance(val, str) and val:
                    return val
        # Fallback: parse from content (first line often has the path)
        content = block.get("content", "")
        if isinstance(content, str) and content:
            first_line = content.split("\n", 1)[0]
            # Common patterns: "1\tpath/to/file.py" or "=== path/to/file ==="
            for token in first_line.split():
                if "/" in token and not token.startswith("http"):
                    return token.strip("'\"`,;:()[]")
        return None

    def _prune(self, messages: list[dict[str, Any]], *, target_tokens: int) -> list[dict[str, Any]]:
        token_count = self.estimate_tokens(messages)
        effective_window = self._effective_recent_window(messages)
        if token_count < target_tokens or len(messages) <= effective_window + 2:
            return messages
        anchor_index = self._anchor_index(messages)
        prefix = messages[: anchor_index + 1]
        recent_start = max(anchor_index + 1, len(messages) - effective_window)
        middle = messages[anchor_index + 1 : recent_start]
        if not middle:
            return messages
        recent = messages[recent_start:]

        # Strategy A: Use live session memory as compaction source (no LLM cost).
        session_memory_summary = self._try_session_memory_compaction(middle)
        if session_memory_summary is not None:
            summary = session_memory_summary
            compression_note = (
                f"Context was compressed using live session memory "
                f"({len(middle)} messages summarized). "
                f"If you need file contents that were pruned, re-read them."
            )
        else:
            # Strategy B: Deterministic event-based summarisation.
            middle = self._clear_stale_tool_results(middle)
            summary, compression_note = self._summarize(middle)

        # Inject a <context-note> so the agent knows what was compressed.
        note_msg = {
            "role": "user",
            "content": f"<context-note>{compression_note}</context-note>",
        }
        return [*prefix, summary, note_msg, *recent]

    def _try_session_memory_compaction(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, str] | None:
        """Attempt to use the live session memory file as compaction source.

        Returns a summary message dict if session memory is available and
        looks valid, or ``None`` to fall back to deterministic summarisation.
        """
        if self._session_memory_path is None:
            return None
        if not self._session_memory_path.exists():
            return None
        try:
            memory_text = self._session_memory_path.read_text()
        except OSError:
            return None
        if "## Current State" not in memory_text or len(memory_text) < 100:
            return None

        # Build a hybrid: session memory + mutation ledger (always preserved).
        mutation_ledger = self._build_mutation_ledger(messages)
        parts = [
            f"[COMPACTED CONTEXT — {len(messages)} messages replaced by session memory]",
        ]
        if mutation_ledger:
            parts.append(mutation_ledger)
        parts.append(memory_text)
        logger.debug(
            "Using session memory for compaction (%d chars, %d messages replaced)",
            len(memory_text), len(messages),
        )
        return {"role": "user", "content": "\n".join(parts)}

    def _extract_environment_state(self, messages: list[dict[str, Any]]) -> str:
        """Scan messages for environment setup info (venv, packages, approach).

        Returns a formatted block summarizing detected environment state, or
        empty string if nothing was detected.
        """
        venv_paths: list[str] = []
        packages: list[str] = []
        last_approach = ""

        # Regex patterns for venv and pip detection
        source_activate_re = re.compile(r"source\s+(\S+)/bin/activate")
        venv_create_re = re.compile(r"python3?\s+-m\s+venv\s+(\S+)")
        pip_install_re = re.compile(r"pip\s+install\s+(.+)")

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                if name not in {"Bash", "execute_bash"}:
                    continue
                inp = block.get("input", {})
                cmd = inp.get("cmd") or inp.get("command") or ""
                if not cmd:
                    continue
                # Detect venv activation
                m = source_activate_re.search(cmd)
                if m:
                    venv_dir = m.group(1).rsplit("/", 1)[0] if "/" in m.group(1) else m.group(1)
                    # The full path captured is up to /bin/activate; parent is the venv
                    full_path = m.group(1)
                    # e.g. source .venv/bin/activate → venv_dir = .venv
                    venv_name = full_path  # keep the full activate path for display
                    if venv_name not in venv_paths:
                        venv_paths.append(venv_name)
                # Detect venv creation
                m = venv_create_re.search(cmd)
                if m:
                    venv_name = m.group(1)
                    if venv_name not in venv_paths:
                        venv_paths.append(venv_name)
                # Detect pip install
                m = pip_install_re.search(cmd)
                if m:
                    raw_pkgs = m.group(1)
                    for pkg in raw_pkgs.split():
                        # Skip flags
                        if pkg.startswith("-"):
                            continue
                        # Strip version specifiers
                        pkg_name = re.split(r"[=<>!~]", pkg)[0]
                        if pkg_name and pkg_name not in packages:
                            packages.append(pkg_name)

        # Extract last design_decision for approach
        events = self._extract_events(messages)
        for event in reversed(events):
            if event.category == "design_decision":
                last_approach = event.content[:200]
                break

        if not venv_paths and not packages and not last_approach:
            return ""

        lines = ["[ENVIRONMENT STATE — reconstructed from pruned context]"]
        for vp in venv_paths:
            lines.append(f"- Venv: {vp} (source {vp}/bin/activate)")
        if packages:
            lines.append(f"- Installed: {', '.join(packages)}")
        if last_approach:
            lines.append(f"- Approach: {last_approach}")
        return "\n".join(lines)

    def _summarize(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, str], str]:
        """Summarize pruned messages into a compact context message.

        Returns:
            Tuple of (summary_message, compression_note).  The compression note
            is a human-readable description of what was removed/preserved, to be
            injected as a ``<context-note>`` so the agent knows what happened.
        """
        events = self._extract_events(messages)
        events = self._detect_error_fix_sequences(messages, events)
        all_events = list(events)  # before prioritization
        events = self._prioritize_events(events)
        summary_text = self._format_event_summary(events)
        mutation_ledger = self._build_mutation_ledger(messages)
        environment_state = self._extract_environment_state(messages)
        parts = [f"[PRUNED CONTEXT - {len(messages)} messages summarized]"]
        if mutation_ledger:
            parts.append(mutation_ledger)
        if environment_state:
            parts.append(environment_state)
        parts.append(summary_text)

        # Build compression transparency note.
        compression_note = self._build_compression_note(
            messages, all_events, events, mutation_ledger,
        )

        return (
            {"role": "user", "content": "\n".join(parts)},
            compression_note,
        )

    def _build_compression_note(
        self,
        messages: list[dict[str, Any]],
        all_events: list,  # type: ignore[type-arg]
        kept_events: list,  # type: ignore[type-arg]
        mutation_ledger: str,
    ) -> str:
        """Build a human-readable note about what was removed/preserved."""
        from collections import Counter

        all_cats = Counter(e.category for e in all_events)
        kept_cats = Counter(e.category for e in kept_events)

        removed_parts: list[str] = []
        preserved_parts: list[str] = []

        # Count removed by category.
        for cat in ("file_read", "command", "decision"):
            total = all_cats.get(cat, 0)
            kept = kept_cats.get(cat, 0)
            removed = total - kept
            if removed > 0:
                label = {"file_read": "file reads", "command": "Bash outputs", "decision": "decisions"}.get(cat, cat)
                removed_parts.append(f"{removed} {label}")

        # Count preserved.
        for cat in ("error", "error_fix_sequence", "milestone_note", "file_write", "design_decision"):
            count = kept_cats.get(cat, 0)
            if count > 0:
                label = {
                    "error": "errors",
                    "error_fix_sequence": "error-fix sequences",
                    "milestone_note": "milestones",
                    "file_write": "file writes",
                    "design_decision": "design decisions",
                }.get(cat, cat)
                preserved_parts.append(f"{count} {label}")

        # Count mutation ledger files.
        ledger_count = mutation_ledger.count("\n- ") if mutation_ledger else 0
        if ledger_count > 0:
            preserved_parts.append(f"mutation ledger ({ledger_count} files)")

        lines = [f"Context was compressed ({len(messages)} messages summarized)."]
        if removed_parts:
            lines.append(f"Removed: {', '.join(removed_parts)}.")
        if preserved_parts:
            lines.append(f"Preserved: {', '.join(preserved_parts)}.")
        lines.append("If you need file contents that were pruned, re-read them.")
        return " ".join(lines)

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------

    def _extract_events(self, messages: list[dict[str, Any]]) -> list[PrunedEvent]:
        events: list[PrunedEvent] = []
        for message in messages:
            events.extend(self._extract_message_events(message))
        return events

    def _extract_message_events(self, message: dict[str, Any]) -> list[PrunedEvent]:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, str):
            return self._classify_text_event(role, content)
        if isinstance(content, list):
            events: list[PrunedEvent] = []
            for block in content:
                events.extend(self._extract_block_events(role, block))
            return events
        return []

    _MILESTONE_RE = re.compile(
        r"^\s*(?:##\s*Milestone:|##\s*Note:|\[MILESTONE\]|\[NOTE\])",
        re.IGNORECASE,
    )

    def _classify_text_event(self, role: str, text: str) -> list[PrunedEvent]:
        text = self._truncate_text(text, limit=TOOL_RESULT_TRUNCATE_LIMIT)
        if not text:
            return []

        # Tag-aware classification for XML-tagged messages.
        from maike.context.tags import extract_tag_priority
        tag_info = extract_tag_priority(text)
        if tag_info is not None:
            tag_name, priority = tag_info
            # Map tags to pruning categories
            if tag_name == "maike-skill":
                return [PrunedEvent(category="design_decision", content=f"{role}: [skill]", priority=0)]
            if tag_name == "maike-nudge" and priority == "critical":
                return [PrunedEvent(category="decision", content=f"{role}: [nudge]", priority=2)]
            if tag_name in ("maike-status", "maike-nudge"):
                return [PrunedEvent(category="command", content=f"{role}: [{tag_name}]", priority=4)]
            if tag_name == "maike-guidance":
                return [PrunedEvent(category="command", content=f"{role}: [guidance]", priority=4)]

        if role == "assistant":
            truncated = self._truncate_text(text, limit=80)
            # Check for milestone / note markers first.
            if self._MILESTONE_RE.match(text):
                return [PrunedEvent(
                    category="milestone_note",
                    content=f"{role}: {truncated}",
                    priority=_CATEGORY_PRIORITY["milestone_note"],
                )]
            # Check for design-decision bearing keywords.
            text_lower = text.lower()
            if any(kw in text_lower for kw in DESIGN_DECISION_KEYWORDS):
                return [PrunedEvent(
                    category="design_decision",
                    content=f"{role}: {truncated}",
                    priority=_CATEGORY_PRIORITY["design_decision"],
                )]
            return [PrunedEvent(
                category="decision",
                content=f"{role}: {truncated}",
                priority=_CATEGORY_PRIORITY["decision"],
            )]
        return [PrunedEvent(
            category="decision",
            content=f"{role}: {text}",
            priority=_CATEGORY_PRIORITY["decision"],
        )]

    def _extract_block_events(self, role: str, block: dict[str, Any]) -> list[PrunedEvent]:
        block_type = block.get("type", "unknown")

        if block_type == "text":
            return self._classify_text_event(role, block.get("text", ""))

        if block_type == "tool_use":
            name = block.get("name", "unknown")
            category = self._tool_category(name)
            return [PrunedEvent(
                category=category,
                content=f"{role} tool: {name}",
                priority=_CATEGORY_PRIORITY.get(category, 4),
            )]

        if block_type == "tool_result":
            name = block.get("tool_name") or block.get("name") or "unknown"
            is_error = block.get("is_error", False)
            category = "error" if is_error else self._tool_category(name)
            limit = _TRUNCATE_LIMITS.get(category, TOOL_RESULT_TRUNCATE_LIMIT)
            detail = self._truncate_text(block.get("content", ""), limit=limit)
            if is_error:
                content = f"tool {name} ERROR: {detail}" if detail else f"tool {name} ERROR"
                return [PrunedEvent(
                    category="error",
                    content=content,
                    priority=_CATEGORY_PRIORITY["error"],
                )]
            status_label = "ok"
            content = f"tool {name} {status_label}: {detail}" if detail else f"tool {name} {status_label}"
            return [PrunedEvent(
                category=category,
                content=content,
                priority=_CATEGORY_PRIORITY.get(category, 4),
            )]

        return [PrunedEvent(
            category="command",
            content=f"{role} block: {block_type}",
            priority=_CATEGORY_PRIORITY["command"],
        )]

    @staticmethod
    def _tool_category(tool_name: str) -> str:
        if tool_name in {"Write", "Edit", "write_file", "edit_file", "delete_file"}:
            return "file_write"
        if tool_name in {"Read", "Grep", "read_file", "search_files", "list_dir", "repo_map"}:
            return "file_read"
        if tool_name in {"Bash", "execute_bash", "run_tests", "syntax_check", "run_lint", "run_typecheck"}:
            return "command"
        return "decision"

    # ------------------------------------------------------------------
    # Mutation ledger — survives even the most aggressive pruning
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mutation_ledger(messages: list[dict[str, Any]]) -> str:
        """Build a ledger of all file mutations in the pruned messages.

        This is prepended to every pruning summary so the agent never
        loses track of what files it has already modified.
        """
        mutations: list[str] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                if name not in _WRITE_TOOL_NAMES:
                    continue
                inp = block.get("input", {})
                path = inp.get("file_path") or inp.get("path") or "(unknown)"
                mutations.append(f"- {name}: {path}")
        if not mutations:
            return ""
        return "[MUTATION LEDGER - files changed in pruned context]\n" + "\n".join(mutations)

    # ------------------------------------------------------------------
    # Error-fix sequence detection
    # ------------------------------------------------------------------

    def _detect_error_fix_sequences(
        self,
        messages: list[dict[str, Any]],
        events: list[PrunedEvent],
    ) -> list[PrunedEvent]:
        """Detect error→reasoning→fix triplets and merge into atomic units.

        Walks a 3-message sliding window looking for:
          1. User message containing a tool_result with ``is_error=True``
          2. Assistant text message (the reasoning / fix decision)
          3. Assistant message containing a tool_use for write_file / edit_file

        When found the individual events from those three messages are replaced
        with a single ``error_fix_sequence`` event at priority 0 so the entire
        chain is preserved as a unit during pruning.
        """
        if len(messages) < 3:
            return events

        # Build message-index → event-indices map.
        # We walk the events in extraction order (same order as messages) and
        # map each event back to its originating message index.
        msg_event_map: dict[int, list[int]] = {}
        event_cursor = 0
        for msg_idx, message in enumerate(messages):
            msg_events = self._extract_message_events(message)
            n = len(msg_events)
            msg_event_map[msg_idx] = list(range(event_cursor, event_cursor + n))
            event_cursor += n

        merged_event_indices: set[int] = set()
        new_events: list[tuple[int, PrunedEvent]] = []  # (insertion_position, event)

        for i in range(len(messages) - 2):
            msg_a = messages[i]
            msg_b = messages[i + 1]
            msg_c = messages[i + 2]

            # Step 1: msg_a must be a user message with an error tool_result.
            if msg_a.get("role") != "user":
                continue
            if not self._message_has_error_tool_result(msg_a):
                continue

            # Step 2: msg_b must be an assistant text message (reasoning).
            if msg_b.get("role") != "assistant":
                continue
            if not self._message_has_text(msg_b):
                continue

            # Step 3: msg_c must be an assistant message with a write/edit tool_use.
            if msg_c.get("role") != "assistant":
                continue
            fix_tool = self._message_has_write_tool_use(msg_c)
            if fix_tool is None:
                continue

            # Merge!
            error_snippet = self._truncate_text(self._error_snippet(msg_a), limit=80)
            reasoning_snippet = self._truncate_text(self._text_snippet(msg_b), limit=80)
            combined_content = (
                f"error→fix: {error_snippet} | reasoning: {reasoning_snippet} | fix: {fix_tool}"
            )
            insertion_pos = min(msg_event_map.get(i, [event_cursor])) if msg_event_map.get(i) else 0
            new_events.append((insertion_pos, PrunedEvent(
                category="error_fix_sequence",
                content=combined_content,
                priority=_CATEGORY_PRIORITY["error_fix_sequence"],
            )))
            for idx in msg_event_map.get(i, []):
                merged_event_indices.add(idx)
            for idx in msg_event_map.get(i + 1, []):
                merged_event_indices.add(idx)
            for idx in msg_event_map.get(i + 2, []):
                merged_event_indices.add(idx)

        if not new_events:
            return events

        # Build the result: keep non-merged events and insert merged ones.
        result: list[PrunedEvent] = []
        inserted_positions: set[int] = set()
        for idx, event in enumerate(events):
            # Insert any merged events that should appear at this position.
            for pos, merged_event in new_events:
                if pos == idx and pos not in inserted_positions:
                    result.append(merged_event)
                    inserted_positions.add(pos)
            if idx not in merged_event_indices:
                result.append(event)

        # Insert any remaining merged events.
        for pos, merged_event in new_events:
            if pos not in inserted_positions:
                result.append(merged_event)

        return result

    @staticmethod
    def _message_has_error_tool_result(message: dict[str, Any]) -> bool:
        content = message.get("content", "")
        if isinstance(content, list):
            return any(
                block.get("type") == "tool_result" and block.get("is_error", False)
                for block in content
            )
        return False

    @staticmethod
    def _message_has_text(message: dict[str, Any]) -> bool:
        content = message.get("content", "")
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(block.get("type") == "text" and block.get("text", "").strip() for block in content)
        return False

    @staticmethod
    def _message_has_write_tool_use(message: dict[str, Any]) -> str | None:
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use" and block.get("name") in _WRITE_TOOL_NAMES:
                    return str(block.get("name"))
        return None

    def _error_snippet(self, message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result" and block.get("is_error", False):
                    return str(block.get("content", ""))
        return ""

    def _text_snippet(self, message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    return str(block.get("text", ""))
        return ""

    # ------------------------------------------------------------------
    # Priority-based capping
    # ------------------------------------------------------------------

    def _prioritize_events(self, events: list[PrunedEvent]) -> list[PrunedEvent]:
        if len(events) <= MAX_PRUNED_EVENTS:
            return events
        # Sort by priority (errors first), then preserve original order via enumerate
        indexed = [(e.priority, i, e) for i, e in enumerate(events)]
        indexed.sort(key=lambda t: (t[0], t[1]))
        kept = indexed[:MAX_PRUNED_EVENTS]
        # Restore original order
        kept.sort(key=lambda t: t[1])
        return [e for _, _, e in kept]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_event_summary(self, events: list[PrunedEvent]) -> str:
        if not events:
            return "No significant events captured."

        grouped: dict[str, list[str]] = {}
        for event in events:
            grouped.setdefault(event.category, []).append(event.content)

        sections: list[str] = []
        # Highest priority first, then in priority order.
        for category in [
            "error",
            "error_fix_sequence",
            "design_decision",
            "milestone_note",
            "file_write",
            "decision",
            "file_read",
            "command",
        ]:
            items = grouped.get(category)
            if not items:
                continue
            label = category.replace("_", " ").title()
            sections.append(f"[{label}]")
            sections.extend(f"  {item}" for item in items)

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Helpers (kept for backward compat)
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        return self._estimate_tokens(messages)

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        return estimate_message_tokens(messages)

    def _anchor_index(self, messages: list[dict[str, Any]]) -> int:
        for index, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return index
            if isinstance(content, list) and any(block.get("type") != "tool_result" for block in content):
                return index
        return 0

    def _truncate_text(self, value: Any, limit: int = 80) -> str:
        text = " ".join(str(value).strip().split())
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    def _budget_prune_target(self, token_budget: int, reserve_tokens: int) -> int:
        headroom = max(token_budget - reserve_tokens - 2_048, reserve_tokens * 2)
        return max(headroom, reserve_tokens)

    def _extract_text(self, value: Any) -> Iterable[str]:
        return []
