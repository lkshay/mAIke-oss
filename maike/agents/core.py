"""Universal agent runner."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from collections.abc import Callable

if TYPE_CHECKING:
    from maike.context.recovery import PostCompactionRecovery
    from maike.memory.session_memory import SessionMemoryService

from maike.atoms.agent import AgentResult
from maike.atoms.context import AgentContext
from maike.atoms.llm import StreamChunk
from maike.atoms.tool import RiskLevel, ToolResult
from maike.agents.classifier import classify_task_complexity, _has_error_patterns
from maike.constants import (
    ADAPTIVE_MODEL_ENABLED,
    DEFAULT_LLM_MAX_TOKENS,
    context_limit_for_model,
    pricing_for_model,
    prune_fraction_for_model,
)
from maike.context.budget import ContextBudgetManager
from maike.tools.result_store import ResultStore
from maike.context.compressor import compress_tool_result
from maike.context.convergence import build_convergence_nudge, build_escalated_nudge, detect_spinning, detect_spinning_v2
from maike.context.telemetry import ContextTelemetry
from maike.cost.tracker import BudgetEnforcer
from maike.observability.tracer import Tracer
from maike.runtime.protocol import ExecutionRuntime
from maike.safety.approval import ApprovalGate
from maike.safety.hooks import SafetyLayer
from maike.safety.rules import Decision
from maike.tools.context import (
    CURRENT_ADVISOR_SESSION,
    CURRENT_AGENT_CONTEXT,
    CURRENT_AGENT_LOOP_STATE,
    CURRENT_CONTEXT_TELEMETRY,
    AgentLoopState,
    peek_current_skill_loader,
)
from maike.utils import dedupe_preserve_order, utcnow


def _serialize_content_block(block) -> dict:
    """Serialize an LLMContentBlock for conversation storage.

    Base64-encodes ``thought_signature`` bytes so they survive JSON
    round-trips (thread history storage and replay).
    """
    import base64

    d = block.model_dump(exclude_none=True)
    sig = d.get("thought_signature")
    if sig is not None and isinstance(sig, bytes):
        d["thought_signature"] = base64.b64encode(sig).decode("ascii")
    return d


@dataclass(frozen=True)
class ToolResolution:
    requested_name: str
    resolved_name: str
    resolution_kind: str


@dataclass(frozen=True)
class ReactCompletionValidation:
    """Result of checking whether a react agent has done enough to exit."""
    satisfied: bool
    nudge_message: str


@dataclass
class SessionToolTracker:
    """Lightweight per-session tracker for search/waste nudges.

    Provides inline feedback when the agent:
    - Reads large files without grepping first
    - Reads the same un-edited file 3+ times
    - Gets 2+ consecutive zero-result Greps
    - Runs the same Bash command FAMILY 2+ times in a row
      (e.g. `ls`, `ls -F`, `ls -R` all share family ``ls``)
    """

    grepped_files: set = None  # type: ignore[assignment]
    read_counts: dict = None  # type: ignore[assignment]
    warned_files: set = None  # type: ignore[assignment]
    zero_result_greps: int = 0
    nudge_count: int = 0
    _MAX_NUDGES: int = 10
    _last_bash_cmd: str | None = None
    _last_bash_family: str | None = None
    _bash_repeat_count: int = 0

    def __post_init__(self) -> None:
        if self.grepped_files is None:
            self.grepped_files = set()
        if self.read_counts is None:
            self.read_counts = {}
        if self.warned_files is None:
            self.warned_files = set()

    def record_grep(self, path_or_glob: str, output: str, matched_files: list[str] | None = None) -> str | None:
        """Record a Grep call. Returns a nudge if consecutive zero-results."""
        self.grepped_files.add(path_or_glob)
        if matched_files:
            self.grepped_files.update(matched_files)
        # Detect zero-result greps
        is_empty = not output.strip() or "0 matches" in output.lower() or output.strip() == "No matches found"
        if is_empty:
            self.zero_result_greps += 1
            if self.zero_result_greps >= 2 and self.nudge_count < self._MAX_NUDGES:
                self.nudge_count += 1
                self.zero_result_greps = 0
                return (
                    "[Tip: Last 2 Grep searches returned no results. "
                    "Check the file path/glob pattern, try a broader regex, "
                    "or use SemanticSearch for natural-language queries.]"
                )
        else:
            self.zero_result_greps = 0
        return None

    def record_read(
        self, file_path: str, line_count: int, edited_files: set[str],
        *, has_line_range: bool = False,
    ) -> str | None:
        """Record a Read call. Returns a nudge if wasteful.

        *has_line_range* should be True when the agent used start_line/end_line.
        Targeted reads are legitimate exploration and do NOT count toward the
        repeated-read warning.  Only full-file reads (which re-fetch the same
        truncated content) indicate a stuck loop.
        """
        if has_line_range:
            # Targeted read — no nudge, don't increment the counter.
            return None

        self.read_counts[file_path] = self.read_counts.get(file_path, 0) + 1
        count = self.read_counts[file_path]

        # Large-file-without-grep nudge
        if (
            line_count > 100
            and file_path not in self.grepped_files
            and file_path not in self.warned_files
            and self.nudge_count < self._MAX_NUDGES
        ):
            self.warned_files.add(file_path)
            self.nudge_count += 1
            return (
                f"[Tip: {file_path} is {line_count} lines. Consider Grep to find "
                f"specific sections, then Read with start_line/end_line.]"
            )

        # Repeated full-file read without editing — escalating nudge.
        if (
            count >= 3
            and file_path not in edited_files
            and self.nudge_count < self._MAX_NUDGES
        ):
            self.nudge_count += 1
            if count >= 5:
                # Escalate: agent is re-reading the same truncated view.
                return (
                    f"[Warning: You have read the full file {file_path} {count} "
                    f"times without editing it. You may be re-reading the same "
                    f"truncated content. To make progress:\n"
                    f"1. Use Grep(pattern=\"<what you need>\", path=\"{file_path}\") "
                    f"to find exact line numbers.\n"
                    f"2. Then Read(path=\"{file_path}\", start_line=N, end_line=N+50) "
                    f"for just that section.\n"
                    f"Targeted reads with start_line/end_line are fine — only "
                    f"full-file re-reads are flagged.]"
                )
            if file_path not in self.warned_files:
                self.warned_files.add(file_path)
                return (
                    f"[Warning: Read {file_path} {count} times without editing. "
                    f"If you need specific information, use Grep. "
                    f"If you've decided not to change this file, stop reading it.]"
                )

        return None

    def record_bash(self, command: str) -> str | None:
        """Record a Bash call. Returns a nudge on repeated commands.

        Repeats are detected by **command family**, not byte-identical match.
        ``ls``, ``ls -F``, ``ls -R``, ``ls -la`` all share family ``ls`` — so
        the agent that ran 6 ``ls`` variants in the 76-turn transcript would
        now have been nudged at the third variant instead of never (the
        previous byte-identical check never matched across flag changes).

        For subcommand-style tools (``git``, ``docker``, ``npm``, etc.) the
        family is the first two tokens, so ``git status`` and ``git log``
        are distinct families.
        """
        cmd_normalized = command.strip()
        family = _bash_command_family(cmd_normalized)
        if family and family == self._last_bash_family:
            self._bash_repeat_count += 1
            if self._bash_repeat_count >= 2 and self.nudge_count < self._MAX_NUDGES:
                self.nudge_count += 1
                count = self._bash_repeat_count + 1  # total including first
                # Check if it's a directory listing family.
                _LS_FAMILIES = ("ls", "find", "tree", "dir")
                is_ls = family in _LS_FAMILIES
                if is_ls:
                    return (
                        f"[Warning: You have run `{family}` {count} times in a row "
                        f"(most recent: `{cmd_normalized[:60]}`).  Variations of "
                        f"`{family}` haven't given new information.  If the "
                        f"workspace is empty, there is nothing to explore — start "
                        f"implementing or use Read/Grep on a specific path.  Do "
                        f"NOT retry another `{family}` variant.]"
                    )
                return (
                    f"[Warning: You have run `{family}` {count} times in a row "
                    f"(most recent: `{cmd_normalized[:60]}`).  Try a different "
                    f"approach instead of repeating the same command family.]"
                )
        else:
            self._last_bash_family = family
            self._bash_repeat_count = 0
        # Track latest command for nudge text even when family matches.
        self._last_bash_cmd = cmd_normalized
        return None


# Subcommand-style tools whose first arg meaningfully changes the action
# (so ``git status`` and ``git log`` are distinct families, not both ``git``).
_SUBCOMMAND_TOOLS: frozenset[str] = frozenset({
    "git", "docker", "kubectl", "helm",
    "npm", "pnpm", "yarn", "pip", "pip3", "uv", "cargo", "go",
    "brew", "apt", "apt-get", "dnf", "yum", "zypper", "pacman",
    "systemctl", "service", "rustup",
})


def _bash_command_family(cmd: str) -> str:
    """Return the 'command family' of a Bash command for repeat detection.

    For most commands the family is the first whitespace-separated token
    (``ls``, ``find``, ``cat``).  For subcommand-style tools the family is
    the first two tokens (``git status``, ``docker ps``).  Returns the
    empty string for an empty/whitespace-only command — callers should
    treat that as "no family" (no match).
    """
    tokens = cmd.split()
    if not tokens:
        return ""
    first = tokens[0]
    if first in _SUBCOMMAND_TOOLS and len(tokens) >= 2 and not tokens[1].startswith("-"):
        return f"{first} {tokens[1]}"
    return first


class RepeatedFailureTracker:
    """Tracks repeated test/bash/edit failures and generates nudges.

    When the same failing output appears N times, the agent is stuck.
    Instead of parsing assertion formats (pytest vs jest vs cargo),
    we hash the failing output and count repetitions.  Language-agnostic.

    A secondary tracker counts failures by test file name (regardless of
    exact output), catching the common case where the agent tweaks a function
    and reruns tests — producing different assertion values each time.
    """

    _REPEAT_THRESHOLD = 3
    _EDIT_REPEAT_THRESHOLD = 2  # Edit spinning is obvious faster
    _EDIT_FILE_REPEAT_THRESHOLD = 3  # Varied failures on same file
    _TEST_FILE_REPEAT_THRESHOLD = 5  # Higher — agent IS trying different things
    _TAIL_LINES = 30  # how many lines from the end to hash
    _TRACKED_TOOLS = frozenset({"Bash", "execute_bash", "run_tests", "Edit", "edit_file"})
    _EDIT_TOOLS = frozenset({"Edit", "edit_file"})

    # Structured error pattern registry: each entry maps regex patterns to a
    # category and a confidence score.  Patterns are cross-language
    # (Python / JS / Go / Rust where applicable).
    #
    # We deliberately do NOT include prescriptive recovery advice.  An
    # earlier version included paragraphs like "Verify that recursive calls
    # converge toward a base case" — which any modern LLM already knows.
    # The value of this registry is the *detection* signal: "you have hit
    # category=X N times in a row".  What to do about it is the LLM's job.
    _ERROR_PATTERNS: list[dict[str, Any]] = [
        {
            "category": "recursion",
            "patterns": [
                re.compile(r"recursionerror", re.IGNORECASE),
                re.compile(r"maximum recursion depth", re.IGNORECASE),
                re.compile(r"stack overflow", re.IGNORECASE),
                re.compile(r"call stack size exceeded", re.IGNORECASE),
                re.compile(r"too much recursion", re.IGNORECASE),
            ],
            "confidence": 0.95,
        },
        {
            "category": "timeout",
            "patterns": [
                re.compile(r"timeout", re.IGNORECASE),
                re.compile(r"timed?\s*out", re.IGNORECASE),
                re.compile(r"deadline exceeded", re.IGNORECASE),
                re.compile(r"took longer than", re.IGNORECASE),
                re.compile(r"time limit", re.IGNORECASE),
            ],
            "confidence": 0.90,
        },
        {
            "category": "import",
            "patterns": [
                re.compile(r"importerror", re.IGNORECASE),
                re.compile(r"modulenotfounderror", re.IGNORECASE),
                re.compile(r"cannot find module", re.IGNORECASE),
                re.compile(r"no module named", re.IGNORECASE),
            ],
            "confidence": 0.95,
        },
        {
            "category": "assertion",
            "patterns": [
                re.compile(r"assertionerror", re.IGNORECASE),
                re.compile(r"assert.*==", re.IGNORECASE),
                re.compile(r"expected\s+.*\s+but\s+got", re.IGNORECASE),
                re.compile(r"to\s?equal\b", re.IGNORECASE),
                re.compile(r"to\s?be\b", re.IGNORECASE),
                re.compile(r"assert_eq(?:ual)?!", re.IGNORECASE),
                re.compile(r"assertEqual", re.IGNORECASE),
            ],
            "confidence": 0.85,
        },
        {
            "category": "syntax",
            "patterns": [
                re.compile(r"syntaxerror", re.IGNORECASE),
                re.compile(r"unexpected token", re.IGNORECASE),
                re.compile(r"parsing error", re.IGNORECASE),
                re.compile(r"invalid syntax", re.IGNORECASE),
            ],
            "confidence": 0.95,
        },
        {
            "category": "type_error",
            "patterns": [
                re.compile(r"typeerror", re.IGNORECASE),
                re.compile(r"type mismatch", re.IGNORECASE),
                re.compile(r"expected type\b", re.IGNORECASE),
                re.compile(r"cannot convert\b", re.IGNORECASE),
            ],
            "confidence": 0.90,
        },
        {
            "category": "permission",
            "patterns": [
                re.compile(r"permissionerror", re.IGNORECASE),
                re.compile(r"permission denied", re.IGNORECASE),
                re.compile(r"access denied", re.IGNORECASE),
                re.compile(r"\beacces\b", re.IGNORECASE),
                re.compile(r"\beperm\b", re.IGNORECASE),
            ],
            "confidence": 0.95,
        },
        {
            "category": "connection",
            "patterns": [
                re.compile(r"connectionerror", re.IGNORECASE),
                re.compile(r"connection refused", re.IGNORECASE),
                re.compile(r"\beconnrefused\b", re.IGNORECASE),
                re.compile(r"network error", re.IGNORECASE),
                re.compile(r"dns resolution", re.IGNORECASE),
            ],
            "confidence": 0.90,
        },
        {
            "category": "memory",
            "patterns": [
                re.compile(r"memoryerror", re.IGNORECASE),
                re.compile(r"out of memory", re.IGNORECASE),
                re.compile(r"\boom\b", re.IGNORECASE),
                re.compile(r"heap out of memory", re.IGNORECASE),
                re.compile(r"allocation failed", re.IGNORECASE),
            ],
            "confidence": 0.90,
        },
        {
            "category": "attribute",
            "patterns": [
                re.compile(r"attributeerror", re.IGNORECASE),
                re.compile(r"has no attribute", re.IGNORECASE),
                re.compile(r"property\s+.*\s*undefined", re.IGNORECASE),
                re.compile(r"\bno method\b", re.IGNORECASE),
            ],
            "confidence": 0.90,
        },
        {
            "category": "name_error",
            "patterns": [
                re.compile(r"nameerror", re.IGNORECASE),
                re.compile(r"undefined variable", re.IGNORECASE),
                re.compile(r"\bis not defined\b", re.IGNORECASE),
                re.compile(r"\bundeclared\b", re.IGNORECASE),
            ],
            "confidence": 0.90,
        },
        {
            "category": "file_not_found",
            "patterns": [
                re.compile(r"filenotfounderror", re.IGNORECASE),
                re.compile(r"no such file", re.IGNORECASE),
                re.compile(r"\benoent\b", re.IGNORECASE),
                re.compile(r"file not found", re.IGNORECASE),
                re.compile(r"path not found", re.IGNORECASE),
            ],
            "confidence": 0.95,
        },
        {
            # Edit tool produced no change — old_text didn't match or matched
            # ambiguously.  Cheap models like Flash Lite can loop here for
            # minutes (see SWE-bench smoke on pytest-dev__pytest-10356,
            # 24 Apr 2026).  Fast detection lets the recovery hint fire on
            # the first repeat instead of after the agent has already burned
            # tokens.
            "category": "malformed_edit",
            "patterns": [
                re.compile(r"old_text not found", re.IGNORECASE),
                re.compile(r"matches \d+ locations", re.IGNORECASE),
            ],
            "confidence": 0.95,
        },
        {
            # Edit tool received old_text == new_text — the agent submitted
            # a non-edit (e.g. `</div>` → `</div>`).  Without this category,
            # gemma4 has been observed looping for 30+ iterations
            # "fixing" a file by re-submitting identical strings.  See
            # session 566ce359 (test workspace, 9 May 2026).  Fires on the
            # first occurrence so the agent gets the structured signal
            # before iterating again.
            "category": "noop_edit",
            "patterns": [
                re.compile(r"old_text and new_text are identical", re.IGNORECASE),
                re.compile(r"\bnoop_edit\b", re.IGNORECASE),
            ],
            "confidence": 0.98,
        },
    ]

    def __init__(self) -> None:
        self._failure_hashes: list[str] = []
        self._failure_outputs: dict[str, str] = {}  # hash → raw tail
        self._failure_tool_names: list[str] = []  # parallel to _failure_hashes
        self._test_file_failures: dict[str, int] = {}  # test_file → consecutive count
        self._edit_file_failures: dict[str, int] = {}  # file_path → consecutive count
        self._per_file_failures: dict[str, int] = {}  # any file → total failures
        # category → number of times we've already hinted it.  Replaces the old
        # one-strike-and-suppressed boolean set, which silently disabled
        # category guidance after a single match.  See first_failure_hint().
        self._category_hint_counts: dict[str, int] = {}
        self._delegation_suggested_files: set[str] = set()  # files already suggested

    def record(self, tool_name: str, output: str, success: bool) -> str | None:
        """Record a tool result.  Returns a nudge message if stuck, else None."""
        if tool_name not in self._TRACKED_TOOLS:
            return None
        if success:
            self._test_file_failures.clear()
            if tool_name in self._EDIT_TOOLS:
                # Clear per-file counter for the file that was edited
                edit_file = self._extract_edit_target_file(output)
                if edit_file:
                    self._edit_file_failures.pop(edit_file, None)
            return None

        # Hash the tail of the output — this is where assertion errors live
        tail = self._extract_failure_tail(output)
        if not tail:
            return None

        import hashlib
        h = hashlib.md5(tail.encode(), usedforsecurity=False).hexdigest()[:12]
        self._failure_hashes.append(h)
        self._failure_outputs[h] = tail
        self._failure_tool_names.append(tool_name)

        # Count recent consecutive occurrences of this hash
        consecutive = 0
        for past_hash in reversed(self._failure_hashes):
            if past_hash == h:
                consecutive += 1
            else:
                break

        is_edit = tool_name in self._EDIT_TOOLS
        threshold = self._EDIT_REPEAT_THRESHOLD if is_edit else self._REPEAT_THRESHOLD
        if consecutive >= threshold:
            # Reset to avoid firing every iteration after threshold
            self._failure_hashes.clear()
            self._failure_tool_names.clear()
            return self._build_nudge(tail, consecutive, is_edit=is_edit)

        # Secondary: track by test file name (catches varied-output spinning)
        if not is_edit:
            test_file = self._extract_failing_test_file(output)
            if test_file:
                self._test_file_failures[test_file] = self._test_file_failures.get(test_file, 0) + 1
                count = self._test_file_failures[test_file]
                if count >= self._TEST_FILE_REPEAT_THRESHOLD:
                    self._test_file_failures[test_file] = 0
                    return self._build_test_file_nudge(test_file, count, tail)

        # Secondary: track edit failures per target file (catches varied
        # edit attempts — different old_text each time — on the same file)
        if is_edit:
            edit_file = self._extract_edit_target_file(output)
            if edit_file:
                self._edit_file_failures[edit_file] = self._edit_file_failures.get(edit_file, 0) + 1
                ef_count = self._edit_file_failures[edit_file]
                if ef_count >= self._EDIT_FILE_REPEAT_THRESHOLD:
                    self._edit_file_failures[edit_file] = 0
                    return self._build_edit_file_nudge(edit_file, ef_count, tail)

        return None

    @staticmethod
    def _extract_failing_test_file(output: str) -> str | None:
        """Extract a failing test file path from test framework output."""
        import re

        # pytest: "FAILED tests/test_foo.py::test_bar - ..."
        m = re.search(r"FAILED\s+([\w/.\-]+\.py)", output)
        if m:
            return m.group(1)

        # jest: "FAIL src/tests/foo.test.js"
        m = re.search(r"FAIL\s+([\w/.\-]+\.(?:js|ts|jsx|tsx))", output)
        if m:
            return m.group(1)

        # Generic fallback: any path-like string containing "test" with a code extension
        m = re.search(r"([\w/.\-]*test[\w/.\-]*\.(?:py|js|ts|rs|go))", output, re.IGNORECASE)
        if m:
            return m.group(1)

        return None

    @staticmethod
    def _extract_edit_target_file(output: str) -> str | None:
        """Extract the target file path from an edit tool output.

        Handles both error messages (``old_text not found in {path}``) and
        successful diffs (``--- {path}``).
        """
        import re

        # Error: "old_text not found in {path}." or "old_text matches N locations in {path}."
        m = re.search(r"(?:old_text not found in|old_text matches \d+ locations in)\s+(.+?)\.(?:\s|$)", output)
        if m:
            return m.group(1).strip()
        # Success: unified diff header "--- {path}"
        m = re.search(r"^---\s+(.+)$", output, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _build_edit_file_nudge(file_path: str, count: int, failure_tail: str) -> str:
        """Build a nudge for repeated varied edit failures on the same file."""
        tail_lines = failure_tail.strip().splitlines()[-10:]
        output_block = "\n".join(tail_lines)
        return (
            f"## Edit Spiral on `{file_path}`\n\n"
            f"You have failed to edit `{file_path}` {count} times in a row "
            f"with different approaches. Stop trying Edit and switch strategy:\n\n"
            f"**Recent failure:**\n```\n{output_block}\n```\n\n"
            f"1. **Read the file fresh** — your mental model of its contents is wrong.\n"
            f"2. If the file is under 80 lines, **use Write** to replace the entire "
            f"file with the corrected version.\n"
            f"3. If the file is large, **Grep for the exact line** you need to change, "
            f"then use a longer, more unique old_text that includes 3+ surrounding lines.\n"
            f"4. Consider whether you are editing the right file at all.\n\n"
            f"**Do NOT attempt another Edit on this file without first using Read "
            f"to see its current contents.**\n"
        )

    @staticmethod
    def _build_test_file_nudge(test_file: str, count: int, failure_tail: str) -> str:
        """Build a nudge for repeated failures in the same test file.

        Reports the detection signals (which file, how many times, what
        category) and stops short of prescribing strategy.  The LLM knows
        how to debug — what it might not know is that THIS test has been
        failing the same way ``count`` times.
        """
        tail_lines = failure_tail.strip().splitlines()[-10:]
        output_block = "\n".join(tail_lines)

        category, _confidence = RepeatedFailureTracker._classify_error(failure_tail)
        category_line = (
            f"**Detected error category: {category.replace('_', ' ')}**\n\n"
            if category is not None else ""
        )

        return (
            f"## Repeated Test File Failure — `{test_file}`\n\n"
            f"`{test_file}` has failed {count} times in a row.  Different "
            f"attempts have not converged.\n\n"
            f"{category_line}"
            f"**Recent output:**\n```\n{output_block}\n```\n\n"
            f"Stop iterating on the same approach.  Pick a fundamentally "
            f"different strategy or use Delegate to hand this off."
        )

    @staticmethod
    def _extract_failure_tail(output: str) -> str:
        """Extract the meaningful tail of a test failure output."""
        lines = output.strip().splitlines()
        if not lines:
            return ""
        # Take the last N lines — assertion errors, tracebacks, summaries
        tail_lines = lines[-RepeatedFailureTracker._TAIL_LINES:]
        return "\n".join(tail_lines).strip()

    @classmethod
    def _classify_error(cls, failure_tail: str) -> tuple[str | None, float]:
        """Detect common error patterns in failure output.

        Returns ``(category, confidence)`` for the first matching pattern,
        or ``(None, 0.0)`` when nothing matches.  Recovery is the LLM's
        responsibility — the registry is detection-only.
        """
        for entry in cls._ERROR_PATTERNS:
            for pat in entry["patterns"]:
                if pat.search(failure_tail):
                    return entry["category"], entry["confidence"]
        return None, 0.0

    # Number of times a category-specific hint can fire per session.  The
    # original implementation suppressed a category permanently after one
    # hint; in the 76-turn transcript that meant a `file_not_found` hint
    # fired once on a stray match and then went silent for the rest of the
    # session, even as the agent kept producing the same error class.
    # Three is enough to be useful without spamming.
    _CATEGORY_HINT_LIMIT = 3

    def first_failure_hint(self, output: str, file_path: str = "") -> str | None:
        """Return a one-line error-category detection signal.

        Fires on each occurrence (up to ``_CATEGORY_HINT_LIMIT`` per
        category) so the agent sees a structured "I detected this kind of
        error" marker.  No prescriptive recovery advice — the LLM is
        capable of debugging once it knows the category.
        """
        tail = self._extract_failure_tail(output)
        if not tail:
            return None
        category, confidence = self._classify_error(tail)
        if category is None or confidence < 0.85:
            return None
        already = self._category_hint_counts.get(category, 0)
        if already >= self._CATEGORY_HINT_LIMIT:
            return None  # Suppressed after N hints
        self._category_hint_counts[category] = already + 1
        return f"[Error hint: detected {category} (occurrence {already + 1})]"

    def suggest_delegation(self, output: str, file_path: str) -> str | None:
        """Suggest delegation after 2 failures on the same file.

        Returns a copy-pasteable Delegate call, or None.
        """
        if not file_path:
            return None
        self._per_file_failures[file_path] = self._per_file_failures.get(file_path, 0) + 1
        count = self._per_file_failures[file_path]
        if count < 2 or file_path in self._delegation_suggested_files:
            return None
        self._delegation_suggested_files.add(file_path)
        tail = self._extract_failure_tail(output)
        category, _ = self._classify_error(tail) if tail else (None, 0.0)
        error_snippet = tail[:100].replace("\n", " ") if tail else "see error output"
        cat_label = category or "error"
        return (
            f"[Tip: You've failed on {file_path} twice. Try a fresh perspective:\n"
            f'Delegate(task="Debug and fix the {cat_label} in {file_path}. '
            f'Error: {error_snippet}", '
            f'agent_type="debug", '
            f'context="Prior attempts failed — reproduce the error, isolate the cause, and fix it.", '
            f"background=false)]"
        )

    @staticmethod
    def _build_nudge(failure_tail: str, count: int, *, is_edit: bool = False) -> str:
        tail_lines = failure_tail.strip().splitlines()[-15:]
        output_block = "\n".join(tail_lines)

        if is_edit:
            return (
                f"## Repeated Edit Failure Detection\n\n"
                f"The same Edit failure has occurred {count} times in a row. "
                f"Your old_text does not match the file content.\n\n"
                f"**Failing output:**\n"
                f"```\n{output_block}\n```\n\n"
                f"**Recovery strategies (try in order):**\n"
                f"1. Read the file fresh with Read — your cached version is stale.\n"
                f"2. If the file is short (<50 lines), use Write to replace the entire file.\n"
                f"3. Search for a unique nearby line to use as old_text instead.\n"
                f"4. If you edited this file earlier in this session, the content has changed "
                f"since you last read it.\n"
            )

        category, _confidence = RepeatedFailureTracker._classify_error(failure_tail)

        if category is not None:
            title = category.replace("_", " ").title()
            return (
                f"## Repeated Failure Detection — {title}\n\n"
                f"The same {title} failure has occurred {count} times in a "
                f"row.  The output is unchanged across attempts — your "
                f"current approach is not converging.\n\n"
                f"**Failing output (last {len(tail_lines)} lines):**\n"
                f"```\n{output_block}\n```\n"
            )

        return (
            f"## Repeated Failure Detection\n\n"
            f"The same failure has occurred {count} times in a row.  The "
            f"output is unchanged across attempts — your current approach "
            f"is not converging.\n\n"
            f"**Failing output (last {len(tail_lines)} lines):**\n"
            f"```\n{output_block}\n```\n"
        )


class AgentCore:
    _PROMPTS_DIR = Path(__file__).parent / "prompts"
    _GENERIC_SPECIALIST_PROMPT = "specialist"
    _BLOCKED_TOOL_REPEAT_LIMIT = 2
    _MAX_REACT_COMPLETION_NUDGES = 2
    _EXECUTION_TOOL_NAMES = {"Bash"}
    # Tools that have zero AgentContext side effects and can be run concurrently.
    # Only tools that never mutate ctx, metadata, or telemetry qualify.
    _PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset({"Grep", "SemanticSearch", "WebSearch", "WebFetch"})
    _MAX_PARALLEL_TOOLS: int = 10
    _TOOL_NAME_ALIASES = {
        # Old canonical names → new canonical names (backward compat)
        "execute_bash": "Bash",
        "run_tests": "Bash",
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
        "grep_codebase": "Grep",
        "search_files": "Grep",
        "request_user_input": "AskUser",
        "list_dir": "Bash",
        "delete_file": "Bash",
        "request_specialist": "Delegate",
        # Bash hallucination recovery
        "run_code": "Bash",
        "run_shell": "Bash",
        "run_command": "Bash",
        "execute_command": "Bash",
        "execute_cmd": "Bash",
        "run_bash": "Bash",
        "exec": "Bash",
        "shell": "Bash",
        "bash": "Bash",
        "run_pytest": "Bash",
        "pytest": "Bash",
        "test": "Bash",
        "tests": "Bash",
        "ls": "Bash",
        "list": "Bash",
        "list_files": "Bash",
        "remove_file": "Bash",
        "commit": "Bash",
        "diff": "Bash",
        "status": "Bash",
        # Read hallucination recovery
        "open_file": "Read",
        "read": "Read",
        "view_file": "Read",
        # Write hallucination recovery
        "create_file": "Write",
        "save_file": "Write",
        # Edit hallucination recovery
        "apply_edit": "Edit",
        "replace_in_file": "Edit",
        "search_replace": "Edit",
        # Grep hallucination recovery
        "search": "Grep",
        "grep": "Grep",
        "find": "Grep",
        # Background bash hallucination recovery (unified into Bash)
        "check_process": "Bash",
        "check_background": "Bash",
        "bash_check": "Bash",
        "BashCheck": "Bash",
        "process_status": "Bash",
        "stop_process": "Bash",
        "kill_process": "Bash",
        "bash_stop": "Bash",
        "BashStop": "Bash",
        "stop_background": "Bash",
        # Async delegate hallucination recovery (unified into Delegate)
        "delegate_async": "Delegate",
        "async_delegate": "Delegate",
        "spawn_agent": "Delegate",
        "DelegateAsync": "Delegate",
        "delegate_check": "Delegate",
        "check_delegate": "Delegate",
        "agent_status": "Delegate",
        "DelegateCheck": "Delegate",
        # Skill hallucination recovery
        "skill": "Skill",
        "get_skill": "Skill",
        "load_skill": "Skill",
        "knowledge": "Skill",
        "load_knowledge": "Skill",
        # Blackboard hallucination recovery
        "blackboard_post": "BlackboardPost",
        "blackboard_read": "BlackboardRead",
        "post_to_blackboard": "BlackboardPost",
        "read_blackboard": "BlackboardRead",
        # Web tool hallucination recovery
        "web_search": "WebSearch",
        "search_web": "WebSearch",
        "google": "WebSearch",
        "google_search": "WebSearch",
        "internet_search": "WebSearch",
        "fetch_url": "WebFetch",
        "browse": "WebFetch",
        "curl": "WebFetch",
        "web_fetch": "WebFetch",
        "http_get": "WebFetch",
        "fetch_page": "WebFetch",
        "read_url": "WebFetch",
    }

    def __init__(
        self,
        llm_gateway,
        tool_registry,
        runtime: ExecutionRuntime,
        safety_layer: SafetyLayer,
        working_memory,
        tracer: Tracer,
        approval_gate: ApprovalGate,
        budget_enforcer: BudgetEnforcer | None = None,
        checkpoint_callback: Any | None = None,
        notification_queue: asyncio.Queue | None = None,
        stream_sink: Callable[[StreamChunk], None] | None = None,
        inbox_queue: asyncio.Queue | None = None,
        session_memory: "SessionMemoryService | None" = None,
        compaction_recovery: "PostCompactionRecovery | None" = None,
    ) -> None:
        self.llm_gateway = llm_gateway
        self.tool_registry = tool_registry
        self.runtime = runtime
        self.safety_layer = safety_layer
        self.working_memory = working_memory
        self.tracer = tracer
        self.approval_gate = approval_gate
        self._checkpoint_callback = checkpoint_callback
        self.budget_enforcer = budget_enforcer or BudgetEnforcer()
        self._notification_queue = notification_queue
        self._inbox_queue = inbox_queue
        self._stream_sink = stream_sink
        self._session_memory = session_memory
        self._compaction_recovery = compaction_recovery
        self._memory_surfacer = None  # Optional[MemorySurfacer], set by orchestrator
        self._bg_gateway = None  # Silent gateway for cheap-model calls (compaction)
        # Optional AdvisorSession — set by orchestrator when --advisor is enabled.
        # Phase 4 wires auto-trigger detection into the main loop; for Phase 2
        # this attribute is only read by the Advisor tool handler.
        self._advisor_session = None
        # Optional TrajectoryAuditor — set by orchestrator.  When None, the
        # auditor code path is entirely skipped (strictly additive feature).
        self._trajectory_auditor = None

    async def _call_llm(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        ctx: AgentContext,
        model_override: str | None = None,
    ) -> Any:
        """Call the LLM, using streaming when a stream_sink is configured.

        Falls back transparently to non-streaming if streaming is unavailable
        or fails.  *model_override* replaces ``ctx.model`` for this call only
        (used by adaptive model selection).
        """
        call_kwargs = dict(
            system=system_prompt,
            messages=conversation,
            tools=tool_schemas,
            model=model_override or ctx.model,
            temperature=ctx.temperature,
        )
        # Debug: dump context to file if MAIKE_DUMP_CONTEXT is set.
        import os
        if os.environ.get("MAIKE_DUMP_CONTEXT"):
            self._dump_context(call_kwargs, ctx)

        try:
            return await self._call_llm_inner(call_kwargs)
        except Exception as exc:
            # Detect Gemini thought_signature errors (400 INVALID_ARGUMENT)
            # and retry after stripping signatures from the conversation.
            exc_str = str(exc).lower()
            if "thought_signature" in exc_str or "thought signature" in exc_str:
                import logging
                logging.getLogger(__name__).warning(
                    "Thought signature error — stripping signatures and retrying"
                )
                cleaned = self._strip_signatures_from_conversation(conversation)
                call_kwargs["messages"] = cleaned
                # Invalidate native history — signatures are stripped.
                self.llm_gateway.clear_native_history()
                return await self._call_llm_inner(call_kwargs)
            raise

    async def _call_llm_inner(self, call_kwargs: dict) -> Any:
        """Execute the actual LLM call (streaming or non-streaming)."""
        if self._stream_sink is None or not hasattr(self.llm_gateway, "stream_call"):
            return await self.llm_gateway.call(**call_kwargs)

        # Streaming path
        result = None
        try:
            async for chunk in self.llm_gateway.stream_call(**call_kwargs):
                # Forward ALL chunks (including is_final) to the sink so it
                # can emit STREAM_DONE and reset its accumulator between
                # LLM turns.  The sink itself guards against double-emitting
                # the accumulated text when is_final=True.
                self._stream_sink(chunk)
                if chunk.is_final:
                    result = chunk.accumulated_result
        except Exception:
            # Streaming failed entirely — fall back to non-streaming
            return await self.llm_gateway.call(**call_kwargs)

        if result is None:
            # No final chunk received — fall back
            return await self.llm_gateway.call(**call_kwargs)

        return result

    def _drain_notifications(self, conversation: list[dict]) -> int:
        """Drain background task notifications and inbox messages.

        Drains both the shared notification queue (background task completions)
        and the per-delegate inbox queue (messages from the parent agent via
        ``Delegate(action="send")``).

        Returns the number of messages injected.
        """
        notifications: list[str] = []
        for q in (self._notification_queue, self._inbox_queue):
            if q is None:
                continue
            while True:
                try:
                    msg = q.get_nowait()
                    notifications.append(str(msg))
                except asyncio.QueueEmpty:
                    break

        if notifications:
            combined = "\n\n---\n\n".join(notifications)
            conversation.append({"role": "user", "content": combined})
        return len(notifications)

    async def _maybe_fire_advisor(
        self,
        *,
        conversation: list[dict[str, Any]],
        ctx: AgentContext,
        iteration_count: int,
        failure_tracker,
    ) -> None:
        """Detect auto-trigger conditions and inject advisor advice as a nudge.

        Two triggers fire here:
        - ``on_stuck`` — repeated identical failures or detected spinning.
          Fires every time conditions are met, subject to AdvisorSession's
          cooldown / max_calls / budget caps.
        - ``after_exploration`` — fires once per session, after the agent has
          made N read/grep calls and is about to start writing.

        Successful advice is appended to the conversation as a high-priority
        ``<maike-advisor>`` block.  Throttled calls are silent.
        """
        from maike.agents.advisor import (
            AdvisorTrigger,
            AdvisorUrgency,
            before_first_edit_condition,
            exploration_threshold_met,
        )
        from maike.context.tags import wrap_tag

        session = self._advisor_session
        if session is None or not session.enabled:
            return

        verdict = None
        # on_stuck takes precedence — it implies the executor needs help now.
        # Use AdvisorSession.failures_seen (monotonic) instead of reading
        # failure_tracker._failure_hashes directly. The tracker clears its
        # own list after firing a nudge, which would otherwise create a
        # blind spot right when the advisor is most likely to help.
        is_stuck = (
            session.failures_seen >= 2
            or detect_spinning_v2(conversation)
        )
        if is_stuck:
            verdict = await session.advise(
                question=(
                    "The executor is showing signs of being stuck "
                    "(repeated failures or spinning). Diagnose and "
                    "tell it what to try next."
                ),
                urgency=AdvisorUrgency.STUCK,
                trigger=AdvisorTrigger.ON_STUCK,
                conversation=conversation,
                ctx=ctx,
                iteration_count=iteration_count,
            )
        elif (
            "after_exploration" not in session.triggers_fired
            and exploration_threshold_met(conversation)
        ):
            verdict = await session.advise(
                question=(
                    "The executor has finished exploring and is about to "
                    "implement. Sanity-check its plan and flag anything "
                    "important it missed."
                ),
                urgency=AdvisorUrgency.PLAN_CHECK,
                trigger=AdvisorTrigger.AFTER_EXPLORATION,
                conversation=conversation,
                ctx=ctx,
                iteration_count=iteration_count,
            )
        elif (
            "before_first_edit" not in session.triggers_fired
            and before_first_edit_condition(conversation, ctx)
        ):
            # Fires unconditionally before the first Edit/Write in the session.
            # Purely behavioral — catches both scaffold-first and minimal-
            # exploration patterns regardless of how the task is phrased.
            verdict = await session.advise(
                question=(
                    "The executor is about to make its first code change. "
                    "Sanity-check its current plan and flag anything important "
                    "it missed before it starts writing."
                ),
                urgency=AdvisorUrgency.PLAN_CHECK,
                trigger=AdvisorTrigger.AFTER_EXPLORATION,  # reuse enum; distinguished via triggers_fired marker below
                conversation=conversation,
                ctx=ctx,
                iteration_count=iteration_count,
            )
            if verdict and not verdict.throttled:
                # Mark this trigger separately so after_exploration can still
                # fire independently if the agent keeps reading later.
                session.triggers_fired.add("before_first_edit")

        if verdict is None or verdict.throttled:
            return

        # Inject as a high-priority nudge.  AgentCore's pruning treats
        # priority="high" content as never-prune.
        block = wrap_tag(
            "maike-advisor",
            verdict.advice,
            priority="high",
            source=verdict.trigger.value,
            urgency=verdict.urgency.value,
        )
        conversation.append({"role": "user", "content": block})

        session.record_verdict(verdict, iteration_count)
        try:
            from maike.observability.tracer import TraceEventKind
            self.tracer.log_context_event(
                event_type=TraceEventKind.ADVISOR_CALL,
                agent_id=ctx.agent_id,
                payload={
                    "trigger": verdict.trigger.value,
                    "urgency": verdict.urgency.value,
                    "iteration": iteration_count,
                    "cost_usd": verdict.cost_usd,
                    "tokens": verdict.tokens_used,
                    "advice": verdict.advice,
                },
            )
        except Exception:
            pass  # tracing is best-effort

    async def _maybe_fire_completion_advisor(
        self,
        *,
        conversation: list[dict[str, Any]],
        ctx: AgentContext,
        iteration_count: int,
    ) -> bool:
        """Fire the advisor when the agent is about to finish prematurely.

        Returns True if advice was injected (the caller should continue the
        loop instead of returning). Returns False if no advice was warranted
        or the session was throttled.

        Fires at most ONCE per agent session — recorded in
        ``session.triggers_fired`` as ``"before_completion"``.
        """
        from maike.agents.advisor import (
            AdvisorTrigger,
            AdvisorUrgency,
            before_completion_condition,
        )
        from maike.context.tags import wrap_tag

        session = self._advisor_session
        if session is None or not session.enabled or ctx.role == "delegate":
            return False
        if "before_completion" in session.triggers_fired:
            return False

        # The LLM just emitted a text-only response (no tool_calls) — we
        # reach this helper only from that path, so last_llm_had_tool_calls=False.
        should_fire, reason = before_completion_condition(
            conversation=conversation,
            last_llm_had_tool_calls=False,
        )
        if not should_fire:
            return False

        if reason == "zero_edits":
            question = (
                "The executor is ending the turn with a text-only response and "
                "has made ZERO code changes. Either it narrated a fix without "
                "emitting Edit/Write, or the task doesn't actually require code "
                "changes. Tell it exactly what to do next: emit the edit, or "
                "confirm the task is truly complete."
            )
        else:  # unverified_edit
            question = (
                "The executor made code changes but is ending the turn without "
                "running a verification command. Tell it to run tests (or the "
                "appropriate check) before declaring the task complete, OR "
                "confirm that verification genuinely isn't needed."
            )

        verdict = await session.advise(
            question=question,
            urgency=AdvisorUrgency.DONE_CHECK,
            trigger=AdvisorTrigger.ON_STUCK,  # closest existing enum value
            conversation=conversation,
            ctx=ctx,
            iteration_count=iteration_count,
        )
        if verdict.throttled:
            return False

        conversation.append({"role": "user", "content": wrap_tag(
            "maike-advisor", verdict.advice,
            priority="high", source="before_completion",
            urgency=verdict.urgency.value,
        )})
        session.record_verdict(verdict, iteration_count)
        session.triggers_fired.add("before_completion")
        try:
            from maike.observability.tracer import TraceEventKind
            self.tracer.log_context_event(
                event_type=TraceEventKind.ADVISOR_CALL,
                agent_id=ctx.agent_id,
                payload={
                    "trigger": "before_completion",
                    "urgency": verdict.urgency.value,
                    "iteration": iteration_count,
                    "cost_usd": verdict.cost_usd,
                    "tokens": verdict.tokens_used,
                    "advice": verdict.advice,
                    "reason": reason,
                },
            )
        except Exception:
            pass
        return True

    @staticmethod
    def _is_react_mode(ctx: AgentContext) -> bool:
        return ctx.stage_name == "react"

    async def _maybe_fire_stuck_detectors(
        self,
        *,
        conversation: list[dict[str, Any]],
        ctx: AgentContext,
        iteration: int,
    ) -> None:
        """Run deterministic stuck-detectors and inject any auditor-approved
        nudges into ``conversation``.

        Strictly additive — does NOT interact with ``SessionToolTracker``,
        ``RepeatedFailureTracker``, or convergence detection.  If the auditor
        vetoes a candidate, nothing is injected.  Any exception is logged and
        swallowed.
        """
        auditor = self._trajectory_auditor
        if auditor is None:
            return
        from maike.agents.stuck_detectors import detect_all_stuck_patterns
        from maike.context.tags import wrap_tag

        candidates = detect_all_stuck_patterns(conversation, ctx=ctx)
        if not candidates:
            return

        task_text = str(ctx.task or "")
        for candidate in candidates:
            verdict = await auditor.audit(
                candidate=candidate, task=task_text, iteration=iteration,
            )
            if verdict.decision != "approve" or not verdict.text:
                # Veto or empty text — skip.  Log for observability.
                self.tracer.log_context_event(
                    event_type="trajectory_auditor_veto",
                    agent_id=ctx.agent_id,
                    payload={
                        "kind": candidate.kind,
                        "reason": verdict.reason[:200],
                        "source": verdict.source,
                    },
                )
                continue
            # Approve — inject as a priority-high ``maike-nudge`` block,
            # same tag family used by convergence detection so existing
            # context-pruning logic treats it consistently.
            conversation.append({
                "role": "user",
                "content": wrap_tag(
                    "maike-nudge", verdict.text,
                    priority="high",
                    level="stuck-detector",
                    kind=candidate.kind,
                ),
            })
            self.tracer.log_context_event(
                event_type="trajectory_auditor_injected",
                agent_id=ctx.agent_id,
                payload={
                    "kind": candidate.kind,
                    "reason": verdict.reason[:200],
                    "source": verdict.source,
                    "iteration": iteration,
                },
            )

    async def run(self, ctx: AgentContext, messages: list[dict[str, Any]]) -> AgentResult:
        token = CURRENT_AGENT_CONTEXT.set(ctx)
        # Publish AdvisorSession so the Advisor tool handler can reach it.
        advisor_token = CURRENT_ADVISOR_SESSION.set(self._advisor_session)
        # Loop state holds live refs to conversation + iteration count so the
        # Advisor tool (invoked mid-turn) can see what the agent just did.
        loop_state = AgentLoopState(
            conversation=[],  # set to conversation ref below once created
            iteration_count=0,
            agent_context=ctx,
        )
        loop_state_token = CURRENT_AGENT_LOOP_STATE.set(loop_state)
        conversation = deepcopy(messages)
        loop_state.conversation = conversation
        # Store conversation reference for fork subagents to inherit.
        ctx.metadata["_conversation_ref"] = conversation
        # Opportunistic cleanup of old persisted tool results (once per process)
        if not getattr(AgentCore, "_result_cleanup_done", False):
            try:
                ResultStore.cleanup_old_results(max_age_hours=24)
            except Exception:
                pass
            AgentCore._result_cleanup_done = True
        session_id = str(ctx.metadata.get("session_id", ctx.agent_id))
        result_store = ResultStore(session_id)
        iteration_count = 0
        blocked_tool_counts: dict[tuple[str, str], int] = {}
        react_mode = self._is_react_mode(ctx)
        has_iteration_limit = ctx.max_iterations > 0
        failure_tracker = RepeatedFailureTracker()
        session_tracker = SessionToolTracker()

        from maike.context.cadence import CadenceTracker
        cadence_tracker = CadenceTracker()

        # When max_iterations is set, trigger compaction at 50-80% of the cap.
        # When unlimited (0), use reasonable absolute thresholds so spinning
        # detection and context pruning still fire.
        if has_iteration_limit:
            warning_threshold = max(int(ctx.max_iterations * (0.7 if react_mode else 0.8)), 1)
            convergence_threshold = max(int(ctx.max_iterations * (0.5 if react_mode else 0.6)), 1)
        else:
            # Absolute fallbacks: start checking for spinning at 40 iterations,
            # warn (aggressive prune) at 60.  Budget is the real constraint.
            convergence_threshold = 40 if react_mode else 25
            warning_threshold = 60 if react_mode else 35
        warned_about_iterations = False
        convergence_level = 0  # Escalates: 0 → 1 → 2 → 3
        # Iteration index at which L3 nudge fired.  None until L3 hits.  When
        # set, the loop force-terminates with `convergence_failed` after the
        # agent has had ``_L3_GRACE_ITERATIONS`` iterations to wrap up.  This
        # turns the previously-advisory L3 into a hard stop — the 76-turn
        # transcript showed the agent receiving L3 and continuing for ~30
        # more iterations regardless.
        l3_nudge_iteration: int | None = None
        react_completion_nudges = 0
        consecutive_empty_responses = 0
        _MAX_EMPTY_RESPONSES = 2
        _seen_files: set[str] = set()  # tracks first-read for compression exemption
        telemetry = ContextTelemetry()
        estimator = getattr(self.working_memory, "estimate_tokens", None)
        if callable(estimator):
            telemetry.initial_context_tokens = estimator(conversation)
        telemetry.update_peak(telemetry.initial_context_tokens)
        self._populate_initial_telemetry(telemetry, conversation)
        telemetry_token = CURRENT_CONTEXT_TELEMETRY.set(telemetry)
        _injected_skills: set[str] = set()
        # Task-based skills are now injected inline by build_react_context
        # (inside the initial user message as <maike-skill> blocks).
        # Seed the _injected_skills set from context metadata so mid-session
        # re-injection doesn't duplicate them.
        _pre_injected = ctx.metadata.get("injected_skills", [])
        for skill_name in _pre_injected:
            _injected_skills.add(skill_name)
            self.tracer.log_context_event(
                event_type="skill_auto_inject_task",
                agent_id=ctx.agent_id,
                payload={"skill": skill_name},
            )
        # Auto-populate critical_reminder for partition agents so they
        # receive a constraint nudge on every LLM turn.
        if str(ctx.metadata.get("coordination_mode")) == "partition" and not ctx.critical_reminder:
            scope = ctx.metadata.get("files_in_scope") or ctx.metadata.get("owned_deliverables") or []
            if scope:
                ctx.critical_reminder = (
                    "REMINDER: You are a partition agent. "
                    f"You may only modify files in: {', '.join(sorted(scope))}. "
                    "Bash, git, and package installs are blocked."
                )

        try:
            while True:
                # Publish live loop state for mid-turn tool handlers (Advisor).
                loop_state.iteration_count = iteration_count
                if has_iteration_limit and iteration_count >= ctx.max_iterations:
                    return AgentResult(
                        agent_id=ctx.agent_id,
                        role=ctx.role,
                        stage_name=ctx.stage_name,
                        output=f"[ITERATION_LIMIT] Agent reached maximum iterations ({ctx.max_iterations})",
                        messages=conversation,
                        cost_usd=ctx.cost_used_usd,
                        tokens_used=ctx.tokens_used,
                        success=False,
                        input_artifact_ids=dedupe_preserve_order(ctx.input_artifact_ids),
                        metadata=self._result_metadata(
                            ctx,
                            termination_reason="max_iterations",
                            iteration_count=iteration_count,
                            telemetry=telemetry,
                        ),
                    )
                # Convergence hard stop: if L3 fired and the agent has
                # ignored it for ``_L3_GRACE_ITERATIONS``, terminate the
                # turn rather than letting the loop run away.  The L3 nudge
                # explicitly tells the agent to wrap up; refusing to do so
                # is the strongest possible spinning signal.
                if (
                    l3_nudge_iteration is not None
                    and iteration_count - l3_nudge_iteration >= self._L3_GRACE_ITERATIONS
                ):
                    return AgentResult(
                        agent_id=ctx.agent_id,
                        role=ctx.role,
                        stage_name=ctx.stage_name,
                        output=(
                            "[CONVERGENCE_FAILED] L3 convergence nudge fired at "
                            f"iteration {l3_nudge_iteration} and was ignored for "
                            f"{self._L3_GRACE_ITERATIONS} subsequent iterations. "
                            "Terminating to prevent runaway iteration."
                        ),
                        messages=conversation,
                        cost_usd=ctx.cost_used_usd,
                        tokens_used=ctx.tokens_used,
                        success=False,
                        input_artifact_ids=dedupe_preserve_order(ctx.input_artifact_ids),
                        metadata=self._result_metadata(
                            ctx,
                            termination_reason="convergence_failed",
                            iteration_count=iteration_count,
                            telemetry=telemetry,
                        ),
                    )
                # Multi-tier convergence nudges: escalate from progress
                # summary (L1) through strategy reset (L2) to final
                # warning (L3).  L1-L2 require spinning detection;
                # L3 fires unconditionally to force wrap-up.
                for fraction, level in self._CONVERGENCE_LEVELS:
                    if convergence_level >= level:
                        continue
                    if has_iteration_limit:
                        threshold = max(int(ctx.max_iterations * fraction), 1)
                    else:
                        threshold = self._CONVERGENCE_ABSOLUTE_THRESHOLDS.get(level, 999)
                    if iteration_count < threshold:
                        continue
                    # L1-L2: only fire when spinning is detected.
                    # L3: fires unconditionally — agent must wrap up.
                    if level < 3 and not detect_spinning_v2(conversation):
                        continue
                    from maike.context.tags import wrap_tag as _wt
                    nudge = build_escalated_nudge(
                        conversation, iteration_count, ctx.max_iterations, level,
                    )
                    conversation.append({"role": "user", "content": _wt(
                        "maike-nudge", nudge, priority="critical", level=str(level),
                    )})
                    convergence_level = level
                    if level == 3:
                        # Start the grace-period timer.  See the hard-stop
                        # check at the top of the loop.
                        l3_nudge_iteration = iteration_count
                    telemetry.convergence_nudge_injected = True
                    self.tracer.log_context_event(
                        event_type="convergence_nudge",
                        agent_id=ctx.agent_id,
                        payload={"iteration": iteration_count, "level": level},
                    )
                    break  # one nudge per iteration

                # === Advisor auto-triggers ===
                # Fire when stuck or after exploration completes.  Only for
                # main agents (delegates skip — cost control and simplicity).
                if (
                    self._advisor_session is not None
                    and self._advisor_session.enabled
                    and ctx.role != "delegate"
                ):
                    await self._maybe_fire_advisor(
                        conversation=conversation,
                        ctx=ctx,
                        iteration_count=iteration_count,
                        failure_tracker=failure_tracker,
                    )

                # Drain background task notifications between turns.
                self._drain_notifications(conversation)

                # Cadence-based proactive reminders (plan, milestone, context).
                # Only fire when convergence nudges are NOT active (they take priority).
                if convergence_level == 0 and ctx.role != "delegate":
                    from maike.context.tags import wrap_tag as _cadence_wt
                    _estimator_c = getattr(self.working_memory, "estimate_tokens", None)
                    _ctx_tokens = _estimator_c(conversation) if callable(_estimator_c) else 0
                    _ctx_limit = context_limit_for_model(ctx.model) if _ctx_tokens else 1
                    _ctx_frac = _ctx_tokens / _ctx_limit if _ctx_limit > 0 else 0.0
                    _cadence_reminders = cadence_tracker.get_reminders(
                        convergence_level=convergence_level,
                        context_usage_fraction=_ctx_frac,
                    )
                    for _cr in _cadence_reminders:
                        conversation.append({"role": "user", "content": _cadence_wt(
                            "maike-nudge", _cr, priority="low", source="cadence",
                        )})

                # Per-turn memory surfacing: inject relevant memories from
                # the typed memory store.  Only for main agents (not delegates),
                # and only when the memory directory exists.
                if ctx.role != "delegate" and self._memory_surfacer is not None:
                    try:
                        _task_text = ctx.task or ""
                        _surfaced = await self._memory_surfacer.find_relevant(
                            query=_task_text,
                            gateway=self.llm_gateway,
                            provider_name=self.llm_gateway.provider_name,
                        )
                        for _mem in _surfaced:
                            _mem_block = (
                                f"<system-reminder>\n"
                                f"Memory ({_mem['type']}, saved {_mem['age']}): "
                                f"{_mem['description']}\n\n"
                                f"{_mem['content']}\n"
                                f"</system-reminder>"
                            )
                            conversation.append({"role": "user", "content": _mem_block})
                    except Exception:
                        pass  # non-fatal — memory surfacing is best-effort

                if not warned_about_iterations and iteration_count >= warning_threshold:
                    self.tracer.log_event(
                        "agent_iteration_warning",
                        current_iterations=iteration_count,
                        max_iterations=ctx.max_iterations,
                    )
                    warned_about_iterations = True
                _estimator = getattr(self.working_memory, "estimate_tokens", None)
                pre_prune_tokens = _estimator(conversation) if callable(_estimator) else 0

                # Mark spinning state for convergence-aware pruning.
                ctx.metadata["_spinning"] = len(failure_tracker._failure_hashes) >= 2

                # Snapshot conversation before pruning for LLM compaction.
                _pre_prune_conversation = list(conversation)

                # Aggressive compaction when nearing max_iterations: force
                # pruning at 50% of current size to give the model a fresh
                # perspective and avoid spinning.
                if (
                    warned_about_iterations
                    and pre_prune_tokens > 0
                    and hasattr(self.working_memory, "prune_to_target")
                ):
                    aggressive_target = max(pre_prune_tokens // 2, 2000)
                    conversation = self.working_memory.prune_to_target(
                        conversation, target_tokens=aggressive_target,
                    )
                else:
                    conversation = self._prune_conversation(ctx, conversation)

                post_prune_tokens = _estimator(conversation) if callable(_estimator) else 0
                pruning_occurred = post_prune_tokens < pre_prune_tokens
                if pruning_occurred:
                    telemetry.record_prune(pre_prune_tokens, post_prune_tokens)
                    self.tracer.log_context_event(
                        event_type="prune",
                        agent_id=ctx.agent_id,
                        tokens_before=pre_prune_tokens,
                        tokens_after=post_prune_tokens,
                    )
                    # LLM compaction: if deterministic pruning was used
                    # (no session memory), try replacing the summary with
                    # a richer LLM-generated structured summary.
                    if not self._has_session_memory_summary(conversation):
                        await self._try_llm_compaction(
                            conversation, _pre_prune_conversation,
                        )

                    # Post-compaction recovery: re-inject hot context (plan,
                    # recently-read files, skills) that pruning removed.
                    if self._compaction_recovery:
                        recovery_msgs = self._compaction_recovery.build_recovery_messages(
                            conversation,
                        )
                        if recovery_msgs:
                            # Insert after the pruned summary (index 2) but
                            # before recent messages.
                            insert_idx = min(3, len(conversation))
                            for rm in reversed(recovery_msgs):
                                conversation.insert(insert_idx, rm)

                # Layer 1: Stale-tool-result clearing every 3 iterations —
                # replaces old Read/Bash/Grep content with semantic stubs.
                _clear_stale = getattr(self.working_memory, "clear_stale_tool_results", None)
                if callable(_clear_stale) and iteration_count % 3 == 0:
                    conversation = _clear_stale(
                        conversation,
                        mutated_paths=set(ctx.metadata.get("mutated_paths", [])),
                    )

                # Layer 2: Context collapse every 5 iterations — groups
                # old tool-call sequences into single summary messages.
                if iteration_count % 5 == 0 and iteration_count > 5:
                    from maike.context.collapse import collapse_tool_sequences
                    conversation = collapse_tool_sequences(conversation)
                # Compress duplicate error outputs to save context tokens.
                _compress_dups = getattr(self.working_memory, "compress_duplicate_failures", None)
                if callable(_compress_dups) and iteration_count % 3 == 0:
                    conversation, self._failure_hashes = _compress_dups(
                        conversation,
                        failure_hashes=getattr(self, "_failure_hashes", None),
                    )

                # Re-inject critical context (MAIKE.md) if pruning removed it.
                self._check_and_reinject_context(conversation, ctx, iteration_count)

                telemetry.update_peak(pre_prune_tokens)
                self._check_budgets(ctx, conversation)
                tool_schemas = self.tool_registry.get_all_schemas()
                tool_schemas = self._filter_tools_for_profile(tool_schemas, ctx.tool_profile)
                system_prompt = self._render_system_prompt(ctx)

                # Context budget enforcement — compress if the payload
                # would exceed the model's context window.
                if not ContextBudgetManager.fits_budget(
                    conversation,
                    tool_schemas=tool_schemas,
                    system_prompt=system_prompt,
                    model=ctx.model,
                ):
                    conversation, compressed_schemas, levels = (
                        ContextBudgetManager.compress_to_fit(
                            conversation,
                            tool_schemas=tool_schemas,
                            system_prompt=system_prompt,
                            model=ctx.model,
                        )
                    )
                    if compressed_schemas is not None:
                        tool_schemas = compressed_schemas
                    if levels:
                        telemetry.record_compression(levels)
                        self.tracer.log_context_event(
                            event_type="compress",
                            agent_id=ctx.agent_id,
                            levels=levels,
                            payload={"iteration": iteration_count},
                        )

                iteration_count += 1
                ctx.progress.iteration_count = iteration_count
                ctx.progress.token_count = ctx.tokens_used
                ctx.progress.cost_usd = ctx.cost_used_usd

                # Adaptive model selection: pick a tier based on
                # conversation state, then resolve the model name.
                #
                # DISABLED for Gemini: switching models mid-conversation
                # corrupts thought_signature chains and stripping signatures
                # destroys tool call history, causing spinning loops.
                # Gemini models must stay on the same model for the entire
                # agent turn.
                _model_override: str | None = None
                _is_gemini = self.llm_gateway.provider_name == "gemini"
                _adaptive_enabled = (
                    ADAPTIVE_MODEL_ENABLED
                    and not _is_gemini
                    and ctx.metadata.get("adaptive_model", True)
                )
                if _adaptive_enabled:
                    _tier = classify_task_complexity(
                        conversation,
                        iteration=iteration_count,
                        failure_count=len(failure_tracker._failure_hashes),
                        has_errors=_has_error_patterns(conversation),
                        max_iterations=ctx.max_iterations,
                    )
                    if _tier != "default":
                        _model_override = self.llm_gateway.resolve_model_for_tier(_tier)
                        self.tracer.log_context_event(
                            event_type="adaptive_model_selection",
                            agent_id=ctx.agent_id,
                            payload={
                                "tier": _tier,
                                "model": _model_override,
                                "iteration": iteration_count,
                                "failure_count": len(failure_tracker._failure_hashes),
                            },
                        )

                # Inject critical reminder ephemerally — appended before
                # the LLM call and removed after so it never accumulates
                # in the stored conversation history.
                _reminder_injected = False
                if ctx.critical_reminder:
                    conversation.append({
                        "role": "user",
                        "content": f"<system-reminder>\n{ctx.critical_reminder}\n</system-reminder>",
                    })
                    _reminder_injected = True

                result = await self._call_llm(
                    system_prompt=system_prompt,
                    conversation=conversation,
                    tool_schemas=tool_schemas,
                    ctx=ctx,
                    model_override=_model_override,
                )

                if _reminder_injected:
                    conversation.pop()
                ctx.tokens_used += result.usage.total
                ctx.cost_used_usd += result.cost_usd

                assistant_content: Any
                if result.content_blocks:
                    # Keep ALL content blocks including thinking parts.
                    # Gemini requires thinking parts to be echoed back for
                    # the thought_signature chain to remain valid. Dropping
                    # them causes "missing thought_signature" errors on the
                    # next turn.  Thinking content is displayed to the user
                    # via the trace sink but excluded from output extraction.
                    assistant_content = [
                        _serialize_content_block(block)
                        for block in result.content_blocks
                    ]
                    if not assistant_content:
                        assistant_content = result.content or ""
                else:
                    assistant_content = result.content or ""
                conversation.append({"role": "assistant", "content": assistant_content})

                # Track numbered plans for post-compaction recovery.
                if self._compaction_recovery and isinstance(assistant_content, str):
                    self._try_record_plan(assistant_content)

                # Cadence tracker: detect milestone writes and extract plan.
                from maike.context.cadence import detect_milestone_in_text, extract_plan_from_conversation
                _asst_text = assistant_content if isinstance(assistant_content, str) else ""
                if isinstance(assistant_content, list):
                    _asst_text = " ".join(
                        b.get("text", "") for b in assistant_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                _wrote_milestone = detect_milestone_in_text(_asst_text)
                cadence_tracker.tick(wrote_milestone=_wrote_milestone)
                # Extract plan on early turns for periodic re-injection.
                if iteration_count <= 3 and not cadence_tracker._plan_text:
                    _plan = extract_plan_from_conversation(conversation)
                    if _plan:
                        cadence_tracker.set_plan(_plan)

                # Detect empty LLM responses (no tool calls, no text).
                _has_text = bool(result.content) or bool(
                    result.content_blocks
                    and any(b.text for b in result.content_blocks if b.type == "text")
                )
                if not result.tool_calls and not _has_text:
                    consecutive_empty_responses += 1
                    if consecutive_empty_responses >= _MAX_EMPTY_RESPONSES:
                        self.tracer.log_context_event(
                            event_type="empty_response_abort",
                            agent_id=ctx.agent_id,
                            payload={"count": consecutive_empty_responses},
                        )
                        return AgentResult(
                            agent_id=ctx.agent_id,
                            role=ctx.role,
                            stage_name=ctx.stage_name,
                            output="[EMPTY_RESPONSE] Agent returned empty responses repeatedly.",
                            messages=conversation,
                            cost_usd=ctx.cost_used_usd,
                            tokens_used=ctx.tokens_used,
                            success=False,
                            input_artifact_ids=dedupe_preserve_order(ctx.input_artifact_ids),
                            metadata=self._result_metadata(
                                ctx,
                                termination_reason="empty_responses",
                                iteration_count=iteration_count,
                                telemetry=telemetry,
                                react_completion_nudges=react_completion_nudges if react_mode else 0,
                            ),
                        )
                    # Include session memory excerpt if available to help
                    # the model recover context after an empty turn.
                    _empty_hint = ""
                    if self._session_memory:
                        _sm_text = self._session_memory.read_memory()
                        if _sm_text:
                            # Extract just the "Current State" section.
                            _cs_start = _sm_text.find("## Current State")
                            _cs_end = _sm_text.find("\n## ", _cs_start + 1) if _cs_start >= 0 else -1
                            if _cs_start >= 0:
                                _current = _sm_text[_cs_start:_cs_end if _cs_end > 0 else _cs_start + 500]
                                _empty_hint = f"\n\nFor reference, here's where you were:\n{_current[:500]}"
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Your last response was empty — no text and no tool calls. "
                            "You are not done yet. Re-read the task and the last tool "
                            f"results, then take your next action.{_empty_hint}"
                        ),
                    })
                    self.tracer.log_context_event(
                        event_type="empty_response_nudge",
                        agent_id=ctx.agent_id,
                        payload={"count": consecutive_empty_responses},
                    )
                    continue
                else:
                    consecutive_empty_responses = 0

                # Trigger session memory update in background (non-blocking).
                # Runs on EVERY iteration (not just tool-call iterations) so
                # the memory file is created even when the agent does text-only
                # turns or delegates that resolve via auto-resume.
                if self._session_memory is not None:
                    self._session_memory.record_tool_calls(
                        len(result.tool_calls) if result.tool_calls else 0,
                    )
                    await self._session_memory.maybe_update(
                        conversation, ctx.tokens_used,
                    )

                if result.tool_calls:
                    tool_results = []

                    # Partition tool calls into sub-batches for optimal
                    # execution.  Consecutive parallel-safe tools are
                    # grouped and run concurrently; non-parallel-safe
                    # tools get their own exclusive sequential batch.
                    sub_batches = self._partition_tool_batches(result.tool_calls)
                    executed_results: list[ToolResult] = []
                    for batch, is_parallel in sub_batches:
                        if is_parallel and len(batch) > 1:
                            sem = asyncio.Semaphore(self._MAX_PARALLEL_TOOLS)

                            async def _limited(tc: dict) -> ToolResult:
                                async with sem:
                                    return await self._execute_tool(tc, ctx)

                            batch_results = list(
                                await asyncio.gather(
                                    *(_limited(tc) for tc in batch),
                                )
                            )
                        else:
                            batch_results = []
                            for tc in batch:
                                batch_results.append(await self._execute_tool(tc, ctx))
                        executed_results.extend(batch_results)

                    # Post-execution bookkeeping runs sequentially in
                    # original order to preserve telemetry ordering and
                    # checkpoint semantics.
                    for tool_call, tool_result in zip(result.tool_calls, executed_results):
                        self._record_tool_call(
                            ctx,
                            tool_call,
                            tool_result,
                            iteration_count=iteration_count,
                        )
                        self.tracer.log_tool_result(tool_result)
                        if (
                            react_mode
                            and self._checkpoint_callback
                            and tool_result.success
                            and tool_call["name"] in self._REACT_WRITE_TOOLS
                        ):
                            try:
                                await self._checkpoint_callback(ctx)
                            except Exception:
                                pass  # best-effort; don't crash the loop
                        raw_output = tool_result.output or ""
                        # Extract file path and targeted-read info for
                        # smarter compression (first-read gets full content).
                        _tc_input = tool_call.get("input", {})
                        _tc_file = _tc_input.get("path", "") or _tc_input.get("file_path", "") if isinstance(_tc_input, dict) else ""
                        _tc_targeted = bool(
                            isinstance(_tc_input, dict)
                            and (_tc_input.get("start_line") or _tc_input.get("end_line"))
                        )
                        conversation_output = compress_tool_result(
                            tool_call["name"], raw_output,
                            file_path=_tc_file,
                            is_targeted_read=_tc_targeted,
                            seen_files=_seen_files,
                        )
                        # Persist large Bash/Delegate results to disk
                        if tool_call["name"] in ("Bash", "execute_bash", "Delegate", "delegate_async"):
                            try:
                                persisted = result_store.persist_if_large(
                                    tool_call["name"], conversation_output,
                                )
                                if persisted:
                                    conversation_output = (
                                        f"[Output truncated — {persisted.total_chars} chars total, "
                                        f"saved to {persisted.path}]\n"
                                        f"Use Read to access full output.\n\n"
                                        f"--- Last {min(2000, persisted.total_chars)} chars ---\n"
                                        f"{persisted.preview}"
                                    )
                            except Exception:
                                pass  # Don't break the agent loop

                        telemetry.record_tool_call(
                            tool_name=tool_call["name"],
                            input_chars=len(str(tool_call.get("input", {}))),
                            output_chars=len(raw_output),
                            compressed_chars=len(conversation_output),
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call["id"],
                                "tool_name": tool_call["name"],
                                "content": conversation_output,
                                "is_error": not tool_result.success,
                            }
                        )
                        loop_abort = self._maybe_abort_on_blocked_tool_loop(
                            ctx,
                            conversation,
                            tool_call,
                            tool_result,
                            blocked_tool_counts,
                            iteration_count=iteration_count,
                            telemetry=telemetry,
                        )
                        if loop_abort is not None:
                            conversation.append({"role": "user", "content": tool_results})
                            return loop_abort
                    conversation.append({"role": "user", "content": tool_results})

                    # Generate a semantic label for multi-tool batches.
                    # Labels survive compaction as breadcrumbs. Only for
                    # batches with 2+ tools — single calls are self-evident.
                    if react_mode and len(tool_results) >= 2:
                        from maike.context.labels import generate_batch_label
                        _label = generate_batch_label(tool_results)
                        if _label:
                            conversation.append({
                                "role": "user",
                                "content": f"<tool-label>{_label}</tool-label>",
                            })

                    # Conditional skill activation: activate path-matched
                    # skills when the agent touches files matching their patterns.
                    if react_mode and ctx.role != "delegate":
                        _sl = peek_current_skill_loader()
                        if _sl is not None:
                            try:
                                _est = getattr(self.working_memory, "estimate_tokens", None)
                                _cur_tokens = _est(conversation) if callable(_est) else 0
                                _ctx_limit = context_limit_for_model(ctx.model)
                                if _cur_tokens < _ctx_limit * 0.8:
                                    _touched: list[str] = []
                                    for _tr in tool_results:
                                        _tn = _tr.get("tool_name", "")
                                        if _tn in ("Read", "Write", "Edit"):
                                            # Extract path from the tool call input.
                                            for _tc in (assistant_content if isinstance(assistant_content, list) else []):
                                                if isinstance(_tc, dict) and _tc.get("id") == _tr.get("tool_use_id"):
                                                    _inp = _tc.get("input", {})
                                                    _p = _inp.get("file_path", "") or _inp.get("path", "")
                                                    if _p:
                                                        _touched.append(_p)
                                    if _touched:
                                        _all_sk = _sl.load_all()
                                        _cond = _sl.select_conditional(_touched, _all_sk, _injected_skills)
                                        if _cond:
                                            _cs = _cond[0]  # max 1 per batch
                                            _injected_skills.add(_cs.name)
                                            from maike.context.tags import wrap_tag as _wt3
                                            conversation.append({
                                                "role": "user",
                                                "content": _wt3(
                                                    "maike-skill", _cs.content,
                                                    priority="critical",
                                                    name=_cs.name,
                                                    injected="path-conditional",
                                                ),
                                            })
                            except Exception:
                                pass  # graceful degradation

                    # Surface mutated files list after any mutation.
                    # Check for read-only constraint violations.
                    mutated = ctx.metadata.get("mutated_paths", [])
                    if mutated:
                        _ro_patterns = ctx.metadata.get("read_only_patterns", [])
                        _violations: list[str] = []
                        if _ro_patterns:
                            from maike.agents.constraints import check_path_against_constraints
                            for _mp in mutated:
                                if check_path_against_constraints(_mp, _ro_patterns):
                                    _violations.append(_mp)
                        from maike.context.tags import wrap_tag as _wrap
                        if _violations:
                            _viol_list = ", ".join(_violations)
                            conversation.append({
                                "role": "user",
                                "content": _wrap(
                                    "maike-status",
                                    f"[CONSTRAINT VIOLATION: You modified {_viol_list} — "
                                    f"the task requires these files to remain unchanged. "
                                    f"Revert your changes to these files and find a way to "
                                    f"achieve the goal without modifying them.]\n"
                                    f"[Files modified this session: {', '.join(mutated)}]",
                                    priority="critical",
                                    type="constraint-violation",
                                ),
                            })
                        else:
                            conversation.append({
                                "role": "user",
                                "content": _wrap(
                                    "maike-status",
                                    f"[Files modified this session: {', '.join(mutated)}]",
                                    priority="low",
                                    type="mutations",
                                ),
                            })

                    # Session tool tracking — inline waste/search nudges.
                    edited_files = set(mutated)
                    for tr_entry in tool_results:
                        tn = tr_entry.get("tool_name", "")
                        content_str = tr_entry.get("content", "")
                        inp = tr_entry.get("input") if isinstance(tr_entry.get("input"), dict) else {}
                        resolved = self._resolve_tool_name(tn).resolved_name

                        if resolved == "Grep":
                            path_or_glob = inp.get("path", "") or inp.get("pattern", "")
                            # Extract matched file paths from grep output
                            matched = [
                                line.split(":")[0] for line in content_str.split("\n")
                                if ":" in line and "/" in line.split(":")[0]
                            ][:20]  # Cap to avoid overhead
                            tip = session_tracker.record_grep(path_or_glob, content_str, matched)
                            if tip:
                                conversation.append({"role": "user", "content": tip})

                        elif resolved == "Read":
                            file_path = inp.get("path", "") or inp.get("file_path", "")
                            line_count = content_str.count("\n") + 1 if content_str else 0
                            _has_range = inp.get("start_line") is not None or inp.get("end_line") is not None
                            tip = session_tracker.record_read(
                                file_path, line_count, edited_files,
                                has_line_range=_has_range,
                            )
                            if tip:
                                conversation.append({"role": "user", "content": tip})
                            # Track for post-compaction recovery.
                            if self._compaction_recovery and file_path and content_str:
                                self._compaction_recovery.record_file_read(
                                    file_path, content_str, line_count,
                                )

                        elif resolved in ("Bash", "execute_bash"):
                            bash_cmd = inp.get("command", "") or inp.get("cmd", "")
                            if bash_cmd:
                                tip = session_tracker.record_bash(bash_cmd)
                                if tip:
                                    conversation.append({"role": "user", "content": tip})

                    # Drain background task notifications (processes/delegates
                    # that completed while tools were executing).
                    self._drain_notifications(conversation)

                    # Check for repeated test failures — inject a nudge if
                    # the agent is stuck on the same error.
                    for tr_entry in tool_results:
                        tool_output = tr_entry.get("content", "")
                        is_error = tr_entry.get("is_error", False)
                        # Bump the advisor's monotonic failure counter. Done
                        # here (not inside failure_tracker) because the
                        # tracker clears its internal list after its own
                        # nudge fires — the advisor needs a counter that
                        # survives that reset.
                        if is_error and self._advisor_session is not None:
                            try:
                                self._advisor_session.record_failure()
                            except Exception:
                                pass  # best-effort
                        nudge = failure_tracker.record(
                            tr_entry.get("tool_name", ""),
                            tool_output,
                            not is_error,
                        )
                        if nudge:
                            conversation.append({"role": "user", "content": nudge})
                            self.tracer.log_context_event(
                                event_type="repeated_failure_nudge",
                                agent_id=ctx.agent_id,
                                payload={"message": nudge[:200]},
                            )
                            # Auto-inject a relevant skill when the agent is
                            # stuck.  At most 1 skill per nudge event, and
                            # skip if context is > 80% full.
                            _sl = peek_current_skill_loader()
                            if _sl is not None:
                                try:
                                    _est = getattr(self.working_memory, "estimate_tokens", None)
                                    _cur_tokens = _est(conversation) if callable(_est) else 0
                                    _ctx_limit = context_limit_for_model(ctx.model)
                                    if _cur_tokens < _ctx_limit * 0.8:
                                        _tool_output = tr_entry.get("content", "")
                                        _matched = _sl.match_tool_output(
                                            _tool_output, _injected_skills,
                                        )
                                        if _matched:
                                            _skill = _matched[0]
                                            _injected_skills.add(_skill.name)
                                            from maike.context.tags import wrap_tag as _wt2
                                            conversation.append({
                                                "role": "user",
                                                "content": _wt2(
                                                    "maike-skill", _skill.content,
                                                    priority="critical",
                                                    name=_skill.name,
                                                    injected="mid-session",
                                                ),
                                            })
                                            self.tracer.log_context_event(
                                                event_type="skill_auto_inject_failure",
                                                agent_id=ctx.agent_id,
                                                payload={"skill": _skill.name},
                                            )
                                except Exception:
                                    pass  # graceful degradation
                            break  # one nudge per iteration is enough
                        elif is_error:
                            # First-failure hint: immediate error-category guidance
                            _inp = tr_entry.get("input") if isinstance(tr_entry.get("input"), dict) else {}
                            _file = _inp.get("path", "") or _inp.get("file_path", "")
                            hint = failure_tracker.first_failure_hint(tool_output, _file)
                            if hint:
                                conversation.append({"role": "user", "content": hint})
                            # Second-failure delegation suggestion
                            if _file:
                                deleg_tip = failure_tracker.suggest_delegation(tool_output, _file)
                                if deleg_tip:
                                    conversation.append({"role": "user", "content": deleg_tip})

                    # TrajectoryAuditor: new detectors + validated injection.
                    # Strictly additive — existing nudge sources above are
                    # untouched.  Skipped entirely when auditor is None.
                    if self._trajectory_auditor is not None:
                        try:
                            await self._maybe_fire_stuck_detectors(
                                conversation=conversation, ctx=ctx,
                                iteration=iteration_count,
                            )
                        except Exception:  # noqa: BLE001 — auditor must never crash the loop
                            import logging as _log
                            _log.getLogger(__name__).debug(
                                "Trajectory auditor hook failed (non-fatal)",
                                exc_info=True,
                            )

                    continue

                # React mode: validate the agent has done enough work before
                # allowing exit.  Only nudges when there's clear evidence of
                # incomplete work (wrote files but didn't verify, or last
                # command failed).  Text-only and read-only responses are
                # always allowed to complete.
                react_validated = True
                if react_mode:
                    validation = self._validate_react_completion(ctx, conversation)
                    if not validation.satisfied:
                        if react_completion_nudges < self._MAX_REACT_COMPLETION_NUDGES:
                            react_completion_nudges += 1
                            conversation.append({"role": "user", "content": validation.nudge_message})
                            self.tracer.log_context_event(
                                event_type="react_completion_nudge",
                                agent_id=ctx.agent_id,
                                payload={
                                    "nudge_count": react_completion_nudges,
                                    "reason": validation.nudge_message[:100],
                                },
                            )
                            continue
                        react_validated = False

                # Advisor before_completion: text-only response + (zero edits
                # OR unverified edit) → consult the frontier model for one
                # last course correction before we terminate the session.
                # Fires at most once per session.
                if await self._maybe_fire_completion_advisor(
                    conversation=conversation,
                    ctx=ctx,
                    iteration_count=iteration_count,
                ):
                    continue

                # Extract text output — prefer result.content, fall back to
                # content_blocks (some providers put text there instead).
                final_output = result.content
                if not final_output and result.content_blocks:
                    text_parts = [
                        b.text for b in result.content_blocks
                        if b.type == "text" and b.text
                    ]
                    if text_parts:
                        final_output = "\n".join(text_parts)

                return AgentResult(
                    agent_id=ctx.agent_id,
                    role=ctx.role,
                    stage_name=ctx.stage_name,
                    output=final_output,
                    messages=conversation,
                    cost_usd=ctx.cost_used_usd,
                    tokens_used=ctx.tokens_used,
                    success=react_validated if react_mode else True,
                    input_artifact_ids=dedupe_preserve_order(ctx.input_artifact_ids),
                    metadata=self._result_metadata(
                        ctx,
                        termination_reason=result.stop_reason.value,
                        iteration_count=iteration_count,
                        telemetry=telemetry,
                        react_completion_nudges=react_completion_nudges if react_mode else 0,
                    ),
                )
        finally:
            # Best-effort final checkpoint before exit.  The per-tool
            # checkpoint callback at the Write/Edit success site only
            # fires after a successful file mutation.  Agents that exit
            # via max_iterations, convergence_failed, empty_responses,
            # blocked_execution, or any non-write terminal step would
            # otherwise leave their last edits uncommitted in the working
            # tree — and the SWE-bench capture_patch (which uses
            # ``git diff base_commit..HEAD``) misses uncommitted changes.
            # See django__django-11400 (24 Apr 2026): agent did real work
            # per session memory, but predictions.jsonl was empty because
            # the final exit happened via Bash, not Edit.  Wrapped in
            # try/except to keep this strictly best-effort: failure here
            # must not mask the original exit reason.
            if react_mode and self._checkpoint_callback:
                try:
                    await self._checkpoint_callback(ctx)
                except Exception:
                    pass
            CURRENT_CONTEXT_TELEMETRY.reset(telemetry_token)
            CURRENT_AGENT_CONTEXT.reset(token)
            try:
                CURRENT_ADVISOR_SESSION.reset(advisor_token)
            except (NameError, LookupError, ValueError):
                pass
            try:
                CURRENT_AGENT_LOOP_STATE.reset(loop_state_token)
            except (NameError, LookupError, ValueError):
                pass

    async def _execute_tool(self, tool_call: dict[str, Any], ctx: AgentContext) -> ToolResult:
        resolution = self._resolve_tool_name(tool_call["name"])
        resolved_call = {**tool_call, "name": resolution.resolved_name}

        # Generate activity description for progress tracking.
        from maike.tools.activity import get_activity_description
        activity = get_activity_description(
            resolution.resolved_name, tool_call.get("input", {}),
        )
        ctx.progress.record_activity(activity)

        self.tracer.log_tool_start(
            resolution.resolved_name,
            payload={
                "input": tool_call.get("input", {}),
                "requested_tool_name": resolution.requested_name,
                "resolved_tool_name": resolution.resolved_name,
                "tool_resolution": resolution.resolution_kind,
                "activity": activity,
            },
        )
        # Flat registry — all tools available to all agents.
        registered = self.tool_registry.get(resolution.resolved_name)
        if registered is None:
            available_tools = self._available_tool_names()
            suggestion = self._suggest_tool_name(resolution.resolved_name, available_tools)
            hint = f" Did you mean '{suggestion}'?" if suggestion else ""
            reason = (
                f"Tool '{resolution.requested_name}' does not exist.{hint} "
                f"Available tools: {', '.join(available_tools)}"
            )
            return ToolResult.blocked(
                resolution.requested_name,
                reason,
                metadata={
                    "blocked_reason_type": "tool_not_found",
                    "suggested_tool": suggestion,
                    "requested_tool_name": resolution.requested_name,
                    "resolved_tool_name": resolution.resolved_name,
                    "tool_resolution": resolution.resolution_kind,
                    "available_tools": available_tools,
                },
            )

        assessment = self.safety_layer.assess(
            resolution.resolved_name,
            resolved_call.get("input", {}),
            registered.risk_level,
            ctx=ctx,
        )
        if assessment.decision == Decision.BLOCK:
            return ToolResult.blocked(
                resolution.requested_name,
                assessment.reason or "Blocked by safety rules",
                metadata={
                    "blocked_reason_type": "safety_block",
                    "requested_tool_name": resolution.requested_name,
                    "resolved_tool_name": resolution.resolved_name,
                    "tool_resolution": resolution.resolution_kind,
                },
            )
        if assessment.decision == Decision.REQUIRE_APPROVAL:
            approval = await self.approval_gate.request(
                resolved_call,
                ctx,
                prompt=assessment.approval_prompt,
            )
            if not approval.approved:
                return ToolResult.blocked(
                    resolution.requested_name,
                    "Rejected by user. Adjust your approach based on this feedback and try a different strategy.",
                    metadata={
                        "blocked_reason_type": "approval_rejected",
                        "requested_tool_name": resolution.requested_name,
                        "resolved_tool_name": resolution.resolved_name,
                        "tool_resolution": resolution.resolution_kind,
                    },
                )
            if approval.has_feedback:
                # User approved but provided feedback — inject it into the
                # tool result so the LLM sees it alongside the output.
                self._pending_user_feedback = approval.feedback

        # Fire PreToolUse hooks.
        await self._fire_hooks(
            "PreToolUse",
            tool_name=resolution.resolved_name,
            tool_args=resolved_call.get("input", {}),
        )

        try:
            raw_result = await registered.fn(**resolved_call.get("input", {}))
        except Exception as exc:
            # Gracefully handle any tool execution error so the agent can
            # recover.  Common causes: hallucinated parameter names (TypeError),
            # bad values (ValueError), missing files (FileNotFoundError), etc.
            # Fire PostToolUseFailure hooks.
            await self._fire_hooks(
                "PostToolUseFailure",
                tool_name=resolution.resolved_name,
                tool_args=resolved_call.get("input", {}),
                error=str(exc),
            )
            return ToolResult(
                tool_name=resolution.resolved_name,
                success=False,
                output=(
                    f"Tool execution failed: {type(exc).__name__}: {exc}. "
                    "Check the tool schema for correct parameter names and types."
                ),
                error=str(exc),
                metadata={
                    "error_type": type(exc).__name__,
                    "requested_tool_name": resolution.requested_name,
                    "resolved_tool_name": resolution.resolved_name,
                    "tool_resolution": resolution.resolution_kind,
                },
            )
        formatted_output = registered.output_formatter(raw_result)
        metadata = dict(raw_result.metadata)
        if resolution.requested_name != resolution.resolved_name:
            metadata.update(
                {
                    "requested_tool_name": resolution.requested_name,
                    "resolved_tool_name": resolution.resolved_name,
                    "tool_resolution": resolution.resolution_kind,
                }
            )

        # Fire PostToolUse hooks and collect output.
        hook_output = await self._fire_hooks(
            "PostToolUse",
            tool_name=resolution.resolved_name,
            tool_args=resolved_call.get("input", {}),
            file_path=resolved_call.get("input", {}).get("path", ""),
        )

        # Collect LSP diagnostics after Write/Edit.
        lsp_output = await self._collect_lsp_diagnostics(
            resolution.resolved_name, resolved_call.get("input", {}),
        )

        # Append hook and LSP feedback to tool output.
        extras: list[str] = []
        if hook_output:
            extras.append(hook_output)
        if lsp_output:
            extras.append(lsp_output)

        # If user provided feedback during approval, append it to the output
        # so the LLM sees it alongside the tool result.
        pending_feedback = getattr(self, "_pending_user_feedback", None)
        if pending_feedback:
            extras.append(
                f"[USER FEEDBACK]: {pending_feedback}\n"
                f"Take this feedback into account for your next steps."
            )
            self._pending_user_feedback = None

        if extras:
            formatted_output = formatted_output + "\n\n" + "\n\n".join(extras)

        return raw_result.model_copy(update={"output": formatted_output, "metadata": metadata})

    async def _fire_hooks(
        self,
        event_name: str,
        tool_name: str = "",
        tool_args: dict | None = None,
        file_path: str = "",
        error: str = "",
    ) -> str:
        """Fire plugin hooks for an event. Returns combined hook stdout (if any)."""
        from maike.tools.context import peek_current_hook_executor
        executor = peek_current_hook_executor()
        if executor is None:
            return ""
        try:
            from maike.plugins.hooks import HookEvent
            event_map = {v.value: v for v in HookEvent}
            event = event_map.get(event_name)
            if event is None:
                return ""

            context = {
                "tool_name": tool_name,
                "file_path": file_path or (tool_args or {}).get("path", ""),
                "error": error,
            }
            results = await executor.fire(event, context=context, tool_name=tool_name)
            outputs = [r.stdout for r in results if r.stdout]
            return "\n".join(outputs) if outputs else ""
        except Exception:
            return ""  # hooks must never break the agent loop

    async def _collect_lsp_diagnostics(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> str:
        """After Write/Edit, notify LSP and collect diagnostics."""
        if tool_name not in ("Write", "Edit"):
            return ""
        from maike.tools.context import peek_current_lsp_manager
        lsp = peek_current_lsp_manager()
        if lsp is None:
            return ""
        file_path = tool_args.get("path", "")
        if not file_path:
            return ""
        try:
            import asyncio
            # Notify the LSP server about the change.
            # Read the file content to send didChange.
            from pathlib import Path
            p = Path(file_path)
            if p.is_file():
                content = p.read_text(encoding="utf-8", errors="replace")
                await lsp.notify_file_change(file_path, content)
                # Small delay for the server to process and publish diagnostics.
                await asyncio.sleep(0.5)
            diag_text = lsp.format_diagnostics_for_agent(file_path)
            return diag_text or ""
        except Exception:
            return ""  # LSP must never break the agent loop

    @staticmethod
    def _strip_signatures_from_conversation(messages: list[dict]) -> list[dict]:
        """Strip thought signatures and drop unsigned tool_use blocks.

        Used as a recovery mechanism when Gemini rejects a request due to
        corrupted or missing thought signatures.  Gemini requires
        ``thought_signature`` on every ``tool_use`` block when thinking mode
        is active, so simply removing the key isn't enough — unsigned
        tool_use blocks must be dropped entirely along with their orphaned
        tool_result responses.
        """
        # Pass 1: collect IDs of tool_use blocks that will lose their signature.
        dropped_tool_ids: set[str] = set()
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    if tool_id:
                        dropped_tool_ids.add(tool_id)

        # Pass 2: rebuild conversation without signatures, thinking blocks,
        # unsigned tool_use blocks, or their orphaned tool_result messages.
        cleaned = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if not isinstance(block, dict):
                        new_blocks.append(block)
                        continue
                    # Drop thinking blocks.
                    if block.get("type") == "thinking":
                        continue
                    # Drop tool_use blocks (they lose their signatures).
                    if block.get("type") == "tool_use":
                        continue
                    # Drop orphaned tool_result blocks.
                    if block.get("type") == "tool_result" and block.get("tool_use_id") in dropped_tool_ids:
                        continue
                    # Strip signature keys from remaining blocks (text).
                    if "thought_signature" in block or "thoughtSignature" in block:
                        block = {
                            k: v for k, v in block.items()
                            if k not in ("thought_signature", "thoughtSignature")
                        }
                    new_blocks.append(block)
                if new_blocks:
                    cleaned.append({**msg, "content": new_blocks})
                elif msg.get("role") == "assistant":
                    cleaned.append({**msg, "content": ""})
            else:
                cleaned.append(msg)
        return cleaned

    _dump_counter = 0

    def _dump_context(self, call_kwargs: dict, ctx: Any) -> None:
        """Write the full LLM payload to /tmp/maike-context-dumps/ for inspection."""
        import json, os
        AgentCore._dump_counter += 1
        dump_dir = Path("/tmp/maike-context-dumps")
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_file = dump_dir / f"call_{AgentCore._dump_counter:03d}_{ctx.agent_id[:8]}.json"
        # Serialize — handle bytes and non-serializable types
        def _default(o):
            if isinstance(o, bytes):
                return f"<bytes:{len(o)}>"
            if isinstance(o, Path):
                return str(o)
            return repr(o)
        payload = {
            "call_number": AgentCore._dump_counter,
            "agent_id": ctx.agent_id,
            "model": call_kwargs.get("model"),
            "system_prompt_length": len(call_kwargs.get("system", "")),
            "system_prompt_preview": call_kwargs.get("system", "")[:500],
            "num_messages": len(call_kwargs.get("messages", [])),
            "messages": [
                {
                    "role": m.get("role"),
                    "content_length": len(str(m.get("content", ""))),
                    "content_preview": str(m.get("content", ""))[:2000],
                    "has_maike_tags": "<maike-" in str(m.get("content", "")),
                }
                for m in call_kwargs.get("messages", [])
            ],
            "num_tools": len(call_kwargs.get("tools", [])),
            "tool_names": [t.get("name") for t in call_kwargs.get("tools", [])],
        }
        dump_file.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")

    def _render_system_prompt(self, ctx: AgentContext) -> str:
        # Delegate agents: use the agent-type-specific prompt as the system
        # prompt (e.g. delegate-explore.md for agent_type="explore").  This
        # prevents the generic specialist.md from overriding the real prompt.
        agent_type = ctx.metadata.get("agent_type")
        if ctx.role == "delegate" and agent_type:
            from maike.agents.delegate import _AGENT_TYPE_CONFIG, _PROMPTS_DIR as _DEL_PROMPTS
            config = _AGENT_TYPE_CONFIG.get(agent_type)
            if config:
                prompt_file = _DEL_PROMPTS / config[0]
                if prompt_file.exists():
                    prompt = prompt_file.read_text(encoding="utf-8")
                    return f"{prompt}\n\n{self._tool_use_contract(ctx)}"

        prompt_name, specialty_hint = self._resolve_prompt_name(ctx.role)
        core_path = self._PROMPTS_DIR / f"{prompt_name}-core.md"

        if core_path.exists():
            # Split format: core + boundary + contract + guidance
            core = core_path.read_text(encoding="utf-8")
            guidance_path = self._PROMPTS_DIR / f"{prompt_name}-guidance.md"
            guidance = guidance_path.read_text(encoding="utf-8") if guidance_path.exists() else ""

            if specialty_hint:
                core = f"{core}\n\nSpecialty Hint: {specialty_hint}"

            parts = [core]
            parts.append("\n<!-- CACHE_BOUNDARY -->\n")
            parts.append(self._tool_use_contract(ctx))
            if guidance:
                parts.append(f"\n<maike-guidance priority=\"low\">\n{guidance}\n</maike-guidance>")
            return "\n".join(parts)
        else:
            # Legacy single-file format
            prompt = self._load_system_prompt(ctx.role)
            return f"{prompt}\n\n{self._tool_use_contract(ctx)}"

    def _load_system_prompt(self, role: str) -> str:
        prompt_name, specialty_hint = self._resolve_prompt_name(role)
        prompt_path = self._PROMPTS_DIR / f"{prompt_name}.md"
        prompt = prompt_path.read_text(encoding="utf-8")
        if specialty_hint:
            prompt = f"{prompt}\n\nSpecialty Hint: {specialty_hint}"
        return prompt

    def _resolve_prompt_name(self, role: str) -> tuple[str, str | None]:
        normalized = self._normalize_prompt_role(role)
        # Check both single-file ({name}.md) and split format ({name}-core.md).
        prompt_path = self._PROMPTS_DIR / f"{normalized}.md"
        core_path = self._PROMPTS_DIR / f"{normalized}-core.md"
        if prompt_path.exists() or core_path.exists():
            return normalized, None
        alias = self._prompt_alias_for_role(normalized)
        if alias is not None:
            return alias, role.strip() or None
        return self._GENERIC_SPECIALIST_PROMPT, role.strip() or None

    def _prompt_alias_for_role(self, normalized: str) -> str | None:
        aliases = {
            "architect": "architect",
            "architecture": "architect",
            "coder": "coder",
            "coding": "coder",
            "debug": "debugger",
            "debugger": "debugger",
            "plan": "planner",
            "planner": "planner",
            "reflection": "reflection",
            "requirement": "requirements",
            "requirements": "requirements",
            "review": "reviewer",
            "reviewer": "reviewer",
            "test": "tester",
            "tester": "tester",
        }
        for token, prompt_name in aliases.items():
            if token in normalized:
                return prompt_name
        return None

    def _normalize_prompt_role(self, role: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", role.strip().lower()).strip("-")
        return normalized or self._GENERIC_SPECIALIST_PROMPT

    def _partition_tool_batches(
        self, tool_calls: list[dict],
    ) -> list[tuple[list[dict], bool]]:
        """Partition tool calls into ``(batch, is_parallel)`` sub-batches.

        Consecutive parallel-safe tools are grouped together for concurrent
        execution.  Non-parallel-safe tools get their own single-item batch
        and run exclusively (blocking parallel work).
        """
        batches: list[tuple[list[dict], bool]] = []
        current_parallel: list[dict] = []

        for tc in tool_calls:
            resolved = self._resolve_tool_name(tc["name"]).resolved_name
            if resolved in self._PARALLEL_SAFE_TOOLS:
                current_parallel.append(tc)
            else:
                if current_parallel:
                    batches.append((current_parallel, True))
                    current_parallel = []
                batches.append(([tc], False))

        if current_parallel:
            batches.append((current_parallel, True))

        return batches

    # Tool profiles that restrict which tools are sent to the LLM.
    # Profiles not listed here pass all tools through unfiltered.
    _READONLY_PROFILES = frozenset({"reflection_readonly"})
    _READONLY_ALLOWED_TOOLS = frozenset({"Read", "Grep", "AskUser", "Delegate", "Skill"})

    # Delegate sub-agent tool profiles.
    # explore/plan include WebSearch+WebFetch so research-style delegate
    # tasks ("research X in 2026", "look up Y") aren't forced to fabricate
    # answers from training data — they can fetch fresh sources.
    _DELEGATE_PROFILE_TOOLS: dict[str, frozenset[str]] = {
        "delegate_explore": frozenset({"Read", "Grep", "SemanticSearch", "Bash", "WebSearch", "WebFetch"}),
        "delegate_plan":    frozenset({"Read", "Grep", "SemanticSearch", "WebSearch", "WebFetch"}),
        "delegate_verify":  frozenset({"Read", "Grep", "Bash"}),
        "delegate_review":  frozenset({"Read", "Grep", "Bash"}),
        "delegate_debug":   frozenset({"Read", "Grep", "Bash", "Edit"}),
    }

    def _filter_tools_for_profile(
        self, schemas: list[dict], profile: str,
    ) -> list[dict]:
        """Filter tool schemas based on the agent's tool profile.

        Read-only profiles strip Write, Edit, and Bash so the LLM physically
        cannot propose mutations — this is more reliable than prompt instructions.
        """
        if profile in self._READONLY_PROFILES:
            return [s for s in schemas if s.get("name") in self._READONLY_ALLOWED_TOOLS]
        if profile in self._DELEGATE_PROFILE_TOOLS:
            allowed = self._DELEGATE_PROFILE_TOOLS[profile]
            return [s for s in schemas if s.get("name") in allowed]
        return schemas

    def _canonicalize_tool_name(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")

    def _resolve_tool_name(self, requested_name: str) -> ToolResolution:
        if self.tool_registry.get(requested_name) is not None:
            return ToolResolution(requested_name, requested_name, "exact")
        registered_by_canonical = {
            self._canonicalize_tool_name(name): name
            for name in self.tool_registry.list_tool_names()
        }
        canonical_name = self._canonicalize_tool_name(requested_name)
        if canonical_name in registered_by_canonical:
            return ToolResolution(
                requested_name=requested_name,
                resolved_name=registered_by_canonical[canonical_name],
                resolution_kind="normalized",
            )
        alias_target = self._TOOL_NAME_ALIASES.get(canonical_name)
        if alias_target is not None:
            resolved_name = registered_by_canonical.get(
                self._canonicalize_tool_name(alias_target),
                alias_target,
            )
            return ToolResolution(
                requested_name=requested_name,
                resolved_name=resolved_name,
                resolution_kind="alias",
            )
        return ToolResolution(requested_name, requested_name, "unresolved")

    # ------------------------------------------------------------------
    # React completion validation
    # ------------------------------------------------------------------

    _REACT_WRITE_TOOLS = {"Write", "Edit"}
    _REACT_VERIFY_TOOLS = {"Bash"}

    # Multi-tier convergence: (fraction_of_max_iterations, level)
    _CONVERGENCE_LEVELS = [(0.40, 1), (0.55, 2), (0.70, 3)]
    _CONVERGENCE_ABSOLUTE_THRESHOLDS = {1: 15, 2: 25, 3: 35}
    # After L3 convergence nudge fires, the agent has this many iterations
    # to wrap up before the loop force-terminates with `convergence_failed`.
    # 2 means: L3 fires at iteration N, agent gets to run iteration N+1,
    # and is force-stopped at the START of iteration N+2.
    _L3_GRACE_ITERATIONS = 2

    def _validate_react_completion(
        self,
        ctx: AgentContext,
        conversation: list[dict[str, Any]],
    ) -> ReactCompletionValidation:
        """Check if a react agent has done enough work to exit.

        Trust the LLM's tool choices (guided by the system prompt) and only
        nudge when there is clear evidence of incomplete work:
        - Wrote files but never ran Bash to verify → nudge
        - Last Bash exited non-zero → nudge
        - Everything else (no tools, read-only, write+passing Bash) → let it exit
        """
        tool_calls = self._extract_tool_calls_from_conversation(conversation)
        tool_names_used = {tc["name"] for tc in tool_calls if tc.get("success", True)}

        wrote_files = bool(tool_names_used & self._REACT_WRITE_TOOLS)
        ran_verification = bool(tool_names_used & self._REACT_VERIFY_TOOLS)

        # No files written → agent answered a question, brainstormed, or
        # explored.  The system prompt guides when to use tools; trust it.
        if not wrote_files:
            return ReactCompletionValidation(satisfied=True, nudge_message="")

        # Wrote files but never ran Bash to verify.
        if not ran_verification:
            return ReactCompletionValidation(
                satisfied=False,
                nudge_message=(
                    "You wrote files but haven't verified them.  Run the "
                    "code or tests with Bash to confirm everything works."
                ),
            )

        # Check that the LAST Bash call actually passed (exit code 0).
        verify_calls = [
            tc for tc in tool_calls
            if tc["name"] in self._REACT_VERIFY_TOOLS
        ]
        if verify_calls:
            last_verify = verify_calls[-1]
            if not last_verify.get("success", True):
                return ReactCompletionValidation(
                    satisfied=False,
                    nudge_message=(
                        "Your last command exited with a non-zero status.  "
                        "Review the output and decide if you need to fix "
                        "something before finishing."
                    ),
                )

        return ReactCompletionValidation(satisfied=True, nudge_message="")

    @staticmethod
    def _extract_tool_calls_from_conversation(
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract tool calls from the conversation history in order.

        Returns a list of dicts with at least ``name`` and ``success``.
        For AskUser calls, also includes ``response`` (the user's reply).
        Order is preserved so callers can reason about what happened
        after a particular tool call.
        """
        # First pass: collect metadata keyed by tool_use_id.
        id_to_name: dict[str, str] = {}
        id_to_input: dict[str, dict] = {}
        id_to_success: dict[str, bool] = {}
        id_to_response: dict[str, str] = {}
        # Track insertion order so the returned list is chronological.
        ordered_ids: list[str] = []

        for msg in conversation:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    name = block.get("name", "")
                    id_to_name[tool_id] = name
                    id_to_input[tool_id] = block.get("input", {})
                    ordered_ids.append(tool_id)
                elif block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    id_to_success[tool_id] = not block.get("is_error", False)
                    id_to_response[tool_id] = str(block.get("content", ""))

        tool_calls: list[dict[str, Any]] = []
        for tool_id in ordered_ids:
            entry: dict[str, Any] = {
                "name": id_to_name.get(tool_id, ""),
                "success": id_to_success.get(tool_id, True),
            }
            # Include response text for AskUser so completion validation
            # can detect plan-then-exit without text analysis of the output.
            if entry["name"] == "AskUser":
                entry["response"] = id_to_response.get(tool_id, "")
            # Include file path for write tools so callers can reason
            # about what was modified (e.g. MAIKE.md nudge suppression).
            tool_input = id_to_input.get(tool_id, {})
            path = tool_input.get("file_path") or tool_input.get("path") or ""
            if path and entry["name"] in ("Write", "Edit", "write_file", "edit_file"):
                entry["path"] = path
            tool_calls.append(entry)
        return tool_calls

    def _tool_use_contract(self, ctx: AgentContext) -> str:
        tools = self.tool_registry.get_all_tools()
        tool_names = ", ".join(sorted(tool.schema.name for tool in tools)) or "(none)"
        lines = [
            "Tool Use Contract:",
            f"- Call only these exact tool names: {tool_names}.",
            "- Never invent tool aliases or generic tool names.",
            "- If a tool call is blocked or unavailable, recover using the allowed tools instead of retrying the same blocked request.",
        ]
        return "\n".join(lines)

    def _available_tool_names(self) -> list[str]:
        return sorted(self.tool_registry.list_tool_names())

    def _suggest_tool_name(self, name: str, available: list[str]) -> str | None:
        """Suggest the closest available tool name via substring/prefix matching."""
        canonical = self._canonicalize_tool_name(name)
        if not canonical:
            return None
        # Exact substring: "execute_cmd" matches "execute_bash" via "execute"
        for tool_name in available:
            if canonical in self._canonicalize_tool_name(tool_name):
                return tool_name
        # Reverse substring: "bash" matches "execute_bash"
        for tool_name in available:
            if self._canonicalize_tool_name(tool_name) in canonical:
                return tool_name
        # Shared prefix (≥4 chars): "write_content" → "write_file"
        if len(canonical) >= 4:
            prefix = canonical[:4]
            for tool_name in available:
                if self._canonicalize_tool_name(tool_name).startswith(prefix):
                    return tool_name
        return None

    def _maybe_abort_on_blocked_tool_loop(
        self,
        ctx: AgentContext,
        conversation: list[dict[str, Any]],
        tool_call: dict[str, Any],
        tool_result: ToolResult,
        blocked_tool_counts: dict[tuple[str, str], int],
        *,
        iteration_count: int,
        telemetry: ContextTelemetry | None = None,
    ) -> AgentResult | None:
        reason_type = str(tool_result.metadata.get("blocked_reason_type", ""))
        if reason_type == "execution_unavailable":
            requested_tool_name = str(
                tool_result.metadata.get("requested_tool_name")
                or tool_call.get("name")
                or tool_result.tool_name
            )
            available_tools = self._available_tool_names()
            self.tracer.log_event(
                "agent_blocked_execution_request",
                requested_tool_name=requested_tool_name,
                resolved_tool_name=tool_result.metadata.get("resolved_tool_name"),
                blocked_reason=tool_result.error,
                available_tools=available_tools,
            )
            return AgentResult(
                agent_id=ctx.agent_id,
                role=ctx.role,
                stage_name=ctx.stage_name,
                output=(
                    f"[EXECUTION_BLOCKED] Agent requested command execution via '{requested_tool_name}' "
                    f"during a non-execution stage. {tool_result.error}"
                ),
                messages=conversation,
                cost_usd=ctx.cost_used_usd,
                tokens_used=ctx.tokens_used,
                success=False,
                input_artifact_ids=dedupe_preserve_order(ctx.input_artifact_ids),
                metadata={
                    **self._result_metadata(
                        ctx,
                        termination_reason="blocked_execution_request",
                        iteration_count=iteration_count,
                        telemetry=telemetry,
                    ),
                    "blocked_tool_name": requested_tool_name,
                    "blocked_tool_reason": tool_result.error,
                    "available_tools": available_tools,
                },
            )
        if reason_type not in {"tool_not_found", "tool_not_allowed"}:
            return None
        requested_tool_name = str(
            tool_result.metadata.get("requested_tool_name")
            or tool_call.get("name")
            or tool_result.tool_name
        )
        reason = tool_result.error or "Tool was blocked"
        key = (requested_tool_name, reason)
        blocked_tool_counts[key] = blocked_tool_counts.get(key, 0) + 1
        if blocked_tool_counts[key] < self._BLOCKED_TOOL_REPEAT_LIMIT:
            return None
        available_tools = self._available_tool_names()
        self.tracer.log_event(
            "agent_blocked_tool_loop",
            requested_tool_name=requested_tool_name,
            resolved_tool_name=tool_result.metadata.get("resolved_tool_name"),
            blocked_reason=reason,
            repeats=blocked_tool_counts[key],
            available_tools=available_tools,
        )
        return AgentResult(
            agent_id=ctx.agent_id,
            role=ctx.role,
            stage_name=ctx.stage_name,
            output=(
                f"[TOOL_LOOP] Agent repeatedly requested blocked tool '{requested_tool_name}'. "
                f"{reason}"
            ),
            messages=conversation,
            cost_usd=ctx.cost_used_usd,
            tokens_used=ctx.tokens_used,
            success=False,
            input_artifact_ids=dedupe_preserve_order(ctx.input_artifact_ids),
            metadata={
                **self._result_metadata(
                    ctx,
                    termination_reason="blocked_tool_loop",
                    iteration_count=iteration_count,
                    telemetry=telemetry,
                ),
                "blocked_tool_name": requested_tool_name,
                "blocked_tool_reason": reason,
                "available_tools": available_tools,
            },
        )

    def _check_budgets(self, ctx: AgentContext, conversation: list[dict[str, Any]]) -> None:
        reserved_tokens = self._estimated_next_call_tokens(conversation)
        reserved_cost_usd = self._estimated_next_call_cost(ctx, conversation)
        self.budget_enforcer.check(
            ctx,
            reserved_tokens=reserved_tokens,
            reserved_cost_usd=reserved_cost_usd,
        )

    def _prune_conversation(self, ctx: AgentContext, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Compute a model-aware prune target.  Use a tighter fraction when
        # the agent is spinning (repeated failures) to free up context for
        # fresh approaches.
        model_limit = context_limit_for_model(ctx.model)
        spinning = ctx.metadata.get("_spinning", False)
        base_fraction = prune_fraction_for_model(ctx.model)
        fraction = max(0.30, base_fraction - 0.10) if spinning else base_fraction
        model_aware_target = int(model_limit * fraction)

        budget_aware_prune = getattr(self.working_memory, "prune_to_budget", None)
        if callable(budget_aware_prune) and ctx.token_budget > 0:
            return budget_aware_prune(
                conversation,
                token_budget=min(ctx.token_budget, model_aware_target),
                reserve_tokens=DEFAULT_LLM_MAX_TOKENS,
                model=ctx.model,
            )
        target_prune = getattr(self.working_memory, "prune_to_target", None)
        if callable(target_prune):
            return target_prune(conversation, target_tokens=model_aware_target)
        return self.working_memory.prune(conversation, model=ctx.model)

    # ------------------------------------------------------------------
    # LLM compaction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_session_memory_summary(conversation: list[dict[str, Any]]) -> bool:
        """Check if pruning used session memory (Strategy A) vs deterministic."""
        for msg in conversation:
            content = msg.get("content", "")
            if isinstance(content, str) and "[COMPACTED CONTEXT" in content:
                return True
        return False

    async def _try_llm_compaction(
        self,
        conversation: list[dict[str, Any]],
        pre_prune_conversation: list[dict[str, Any]],
    ) -> None:
        """Replace deterministic prune summary with LLM-generated structured summary.

        Fires only when session memory was NOT available.  Uses a cheap model
        to produce a 9-section summary preserving file paths, decisions, errors,
        and next steps.  Mutates *conversation* in-place on success; on failure
        the deterministic summary remains untouched.
        """
        try:
            from maike.context.compaction_prompt import (
                COMPACTION_SYSTEM_PROMPT,
                build_compaction_messages,
                extract_summary,
            )

            compaction_messages = build_compaction_messages(pre_prune_conversation)
            # Use background gateway for cheap-model compaction to avoid
            # corrupting Gemini native history and suppress TUI noise.
            _gw = self._bg_gateway or self.llm_gateway
            model = _gw.resolve_model_for_tier("cheap")
            result = await _gw.call(
                system=COMPACTION_SYSTEM_PROMPT,
                messages=compaction_messages,
                tools=None,
                model=model,
                temperature=0.0,
                max_tokens=4096,
            )

            summary_text = extract_summary(result.content or "")
            if not summary_text:
                import logging
                logging.getLogger(__name__).debug(
                    "LLM compaction returned no valid summary — keeping deterministic",
                )
                return

            # Find and replace the deterministic summary in the pruned conversation.
            for i, msg in enumerate(conversation):
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue
                if "[PRUNED CONTEXT" not in content:
                    continue

                # Preserve the mutation ledger if present.
                mutation_ledger = ""
                if "[MUTATION LEDGER" in content:
                    ledger_start = content.index("[MUTATION LEDGER")
                    # Find the end of the ledger section (double newline or end).
                    ledger_end = content.find("\n\n[PRUNED CONTEXT", ledger_start)
                    if ledger_end < 0:
                        ledger_end = content.find("\n\n", ledger_start + 1)
                    if ledger_end < 0:
                        ledger_end = len(content)
                    mutation_ledger = content[ledger_start:ledger_end] + "\n\n"

                conversation[i] = {
                    "role": msg.get("role", "user"),
                    "content": (
                        f"{mutation_ledger}"
                        f"[LLM-COMPACTED CONTEXT]\n\n{summary_text}"
                    ),
                }
                import logging
                logging.getLogger(__name__).debug(
                    "LLM compaction replaced deterministic summary (%d chars)",
                    len(summary_text),
                )
                break

        except Exception:
            # LLM compaction is best-effort; deterministic summary remains.
            import logging
            logging.getLogger(__name__).debug(
                "LLM compaction failed — keeping deterministic summary",
                exc_info=True,
            )

    _PLAN_PATTERN = re.compile(r"^\s*[1-3]\.\s+\S", re.MULTILINE)

    def _try_record_plan(self, text: str) -> None:
        """Record a numbered plan from assistant output for recovery."""
        # Heuristic: at least 3 numbered steps (1. 2. 3.) in the text.
        matches = self._PLAN_PATTERN.findall(text)
        if len(matches) >= 3 and self._compaction_recovery:
            # Extract just the plan portion (from first "1." to end or next section).
            start = text.find("1.")
            if start >= 0:
                plan = text[start:start + 2000]  # Cap plan size
                self._compaction_recovery.record_plan(plan)

    def _check_and_reinject_context(
        self,
        conversation: list[dict[str, Any]],
        ctx: AgentContext,
        iteration_count: int,
    ) -> None:
        """Re-inject MAIKE.md project context if pruning removed it."""
        # Cooldown: don't re-inject every iteration.
        last_reinjection = ctx.metadata.get("_reinjected_turn", -10)
        if iteration_count - last_reinjection < 3:
            return

        # Check headroom: only re-inject if >20% context remains
        # (avoid fighting with pruner).
        from maike.context.budget import ContextBudgetManager
        from maike.context.tokenizer import count_message_tokens

        current_tokens = count_message_tokens(conversation)
        limit = ContextBudgetManager.effective_limit(ctx.model)
        if current_tokens > limit * 0.80:
            return

        # Scan conversation for missing MAIKE.md content.
        conv_text = str(conversation)
        needs_maike = (
            "<maike-project" not in conv_text
            and "MAIKE.md (re-injected" not in conv_text
        )

        if not needs_maike:
            return

        workspace = ctx.metadata.get("workspace")
        if not workspace:
            return

        from maike.agents.react import read_maike_md

        maike_content = read_maike_md(workspace)
        if not maike_content:
            return

        # Cap at 5K chars to avoid bloating context.
        if len(maike_content) > 5000:
            maike_content = maike_content[:5000] + "\n[...truncated for context budget]"

        reinject_msg = {
            "role": "user",
            "content": (
                "<system-reminder>\n"
                "[Re-injected project context after pruning]\n"
                f"## MAIKE.md (re-injected after pruning)\n\n{maike_content}\n"
                "</system-reminder>"
            ),
        }

        # Insert after first user message (not at end).
        insert_idx = 1
        for i, msg in enumerate(conversation):
            if msg.get("role") == "user":
                insert_idx = i + 1
                break
        conversation.insert(insert_idx, reinject_msg)
        ctx.metadata["_reinjected_turn"] = iteration_count

    def _estimated_next_call_tokens(self, conversation: list[dict[str, Any]]) -> int:
        estimator = getattr(self.working_memory, "estimate_tokens", None)
        input_tokens = int(estimator(conversation)) if callable(estimator) else 0
        return max(input_tokens + DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_MAX_TOKENS)

    def _estimated_next_call_cost(self, ctx: AgentContext, conversation: list[dict[str, Any]]) -> float:
        pricing = pricing_for_model(ctx.model)
        if pricing is None:
            return 0.0
        estimator = getattr(self.working_memory, "estimate_tokens", None)
        input_tokens = int(estimator(conversation)) if callable(estimator) else 0
        return (
            input_tokens / 1_000_000 * pricing.input_per_million_usd
            + DEFAULT_LLM_MAX_TOKENS / 1_000_000 * pricing.output_per_million_usd
        )

    def _record_tool_call(
        self,
        ctx: AgentContext,
        tool_call: dict[str, Any],
        tool_result: ToolResult,
        *,
        iteration_count: int,
    ) -> None:
        if not bool(ctx.metadata.get("verbose_trace")):
            return
        resolution = self._resolve_tool_name(str(tool_call.get("name") or tool_result.tool_name))
        tool_calls = ctx.metadata.setdefault("tool_calls", [])
        tool_calls.append(
            {
                "sequence": len(tool_calls) + 1,
                "timestamp": utcnow().isoformat(),
                "iteration": iteration_count,
                "tool_use_id": tool_call.get("id"),
                "requested_tool_name": tool_result.metadata.get("requested_tool_name")
                or resolution.requested_name,
                "resolved_tool_name": tool_result.metadata.get("resolved_tool_name")
                or resolution.resolved_name,
                "tool_resolution": tool_result.metadata.get("tool_resolution")
                or resolution.resolution_kind,
                "input": self._json_safe(tool_call.get("input", {})),
                "success": tool_result.success,
                "output": self._json_safe(tool_result.output),
                "raw_output": self._json_safe(tool_result.raw_output),
                "error": self._json_safe(tool_result.error),
                "execution_ms": tool_result.execution_ms,
                "result_metadata": self._json_safe(tool_result.metadata),
            }
        )

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return self._json_safe(model_dump(mode="json"))
        return str(value)

    def _result_metadata(
        self,
        ctx: AgentContext,
        *,
        termination_reason: str,
        iteration_count: int,
        telemetry: ContextTelemetry | None = None,
        react_completion_nudges: int = 0,
    ) -> dict[str, Any]:
        metadata = dict(ctx.metadata)
        metadata.update(
            {
                "tool_profile": ctx.tool_profile,
                "session_id": ctx.metadata.get("session_id"),
                "stage_checkpoint_sha": ctx.metadata.get("stage_checkpoint_sha"),
                "mutated_paths": list(ctx.metadata.get("mutated_paths", [])),
                "children_ids": list(ctx.children_ids),
                "input_artifact_ids": dedupe_preserve_order(ctx.input_artifact_ids),
                "input_artifact_names": list(ctx.input_artifact_names),
                "spawn_depth": ctx.spawn_depth,
                "termination_reason": termination_reason,
                "iteration_count": iteration_count,
                "react_completion_nudges": react_completion_nudges,
                "max_iterations": ctx.max_iterations,
            }
        )
        if telemetry is not None:
            metadata["context_telemetry"] = telemetry.report()
        return metadata

    def _populate_initial_telemetry(
        self,
        telemetry: ContextTelemetry,
        messages: list[dict[str, Any]],
    ) -> None:
        """Scan initial messages for artifact sections and populate telemetry."""
        if not messages:
            return
        first_msg = messages[0]
        content = first_msg.get("content", "")
        if not isinstance(content, str):
            return

        # Parse artifact sections: lines starting with "## Artifact: <name>"
        current_name: str | None = None
        current_lines: list[str] = []
        from maike.utils import estimate_message_tokens as _est

        for line in content.split("\n"):
            if line.startswith("## Artifact: "):
                if current_name is not None:
                    chunk = "\n".join(current_lines)
                    telemetry.artifact_tokens[current_name] = _est(
                        [{"role": "user", "content": chunk}]
                    )
                current_name = line[len("## Artifact: "):].strip()
                current_lines = [line]
            elif line.startswith("## ") and current_name is not None:
                # End of current artifact section
                chunk = "\n".join(current_lines)
                telemetry.artifact_tokens[current_name] = _est(
                    [{"role": "user", "content": chunk}]
                )
                current_name = None
                current_lines = []
                # Check for workspace snapshot
                if line.startswith("## Workspace Snapshot"):
                    current_name = "__workspace_snapshot__"
                    current_lines = [line]
            elif current_name is not None:
                current_lines.append(line)

        # Flush last artifact
        if current_name is not None and current_lines:
            chunk = "\n".join(current_lines)
            tokens = _est([{"role": "user", "content": chunk}])
            if current_name == "__workspace_snapshot__":
                telemetry.workspace_snapshot_tokens = tokens
            else:
                telemetry.artifact_tokens[current_name] = tokens

        # Detect summarized artifacts
        for name, _ in telemetry.artifact_tokens.items():
            marker = f"## Artifact: {name}"
            idx = content.find(marker)
            if idx >= 0:
                after = content[idx:idx + len(marker) + 500]
                if "[SUMMARIZED]" in after:
                    telemetry.summarized_artifacts.append(name)

