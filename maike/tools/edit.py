"""Diff-based search-and-replace editing tool."""

from __future__ import annotations

import difflib
import logging

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.memory.session import SessionStore
from maike.runtime.protocol import ExecutionRuntime
from maike.tools.context import current_agent_context, peek_current_agent_context
from maike.tools.filesystem import (
    _file_read_state,
    append_context_dependency,
    infer_artifact_type,
    mark_mutated_path,
    normalize_artifact_path,
)
from maike.tools.registry import ToolRegistry

_MAX_DIFF_LINES = 50
_FUZZY_CONTEXT_LINES = 5


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace for fuzzy comparison."""
    import re
    # Normalize line endings, collapse blank lines, strip trailing whitespace
    lines = text.splitlines()
    normalized = "\n".join(line.rstrip() for line in lines)
    return normalized


def _find_fuzzy_match(content: str, old_text: str) -> str | None:
    """When exact match fails, try to find where old_text *almost* matches.

    Returns a helpful diagnostic string, or None if nothing close was found.
    """
    # Strategy 0: Detect CRLF vs LF mismatch explicitly.
    # This is the #1 cause of silent failures on cross-platform repos.
    content_lf = content.replace("\r\n", "\n").replace("\r", "\n")
    old_lf = old_text.replace("\r\n", "\n").replace("\r", "\n")
    if content_lf != content or old_lf != old_text:
        if old_lf in content_lf:
            start = content_lf.index(old_lf)
            line_num = content_lf[:start].count("\n") + 1
            return (
                f"Line ending mismatch (CRLF vs LF) at line {line_num}. "
                f"The file has different line endings than your old_text. "
                f"This has been auto-fixed — retry the same Edit command."
            )

    # Strategy 1: Normalize whitespace on both sides and try again.
    norm_content = _normalize_whitespace(content)
    norm_old = _normalize_whitespace(old_text)
    if norm_old in norm_content:
        # The match would succeed with normalized whitespace — the issue
        # is trailing spaces, different line endings, or blank lines.
        start = norm_content.index(norm_old)
        # Find the corresponding lines in the original content
        line_num = norm_content[:start].count("\n") + 1
        return (
            f"A whitespace-normalized match was found at line {line_num}. "
            f"The old_text has trailing spaces, extra blank lines, or different "
            f"line endings compared to the file. Try re-reading the file and "
            f"copying the exact text."
        )

    # Strategy 2: Find the most similar block using SequenceMatcher.
    # Split into lines and find the best matching window.
    old_lines = old_text.splitlines()
    content_lines = content.splitlines()
    if not old_lines:
        return None

    best_ratio = 0.0
    best_start = 0
    window_size = len(old_lines)

    # Slide a window across content lines and score similarity
    for i in range(max(1, len(content_lines) - window_size + 1)):
        candidate = "\n".join(content_lines[i : i + window_size])
        ratio = difflib.SequenceMatcher(None, old_text, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio < 0.30:
        return None  # Nothing close enough — even a context block would mislead.

    # Show what the file actually contains at the closest match.  We emit a
    # context block down to 30% similarity (was 50%) — cheap models like
    # Flash Lite often produce 30–50% matches when they get whitespace,
    # indentation, or surrounding context wrong; without this block they
    # loop instead of recovering.  See SWE-bench smoke on
    # pytest-dev__pytest-10356 (24 Apr 2026): Flash burned 308s in a
    # 50%-match-blind loop until budget capped.
    context_start = max(0, best_start - 1)
    context_end = min(len(content_lines), best_start + window_size + 1)
    actual_lines = content_lines[context_start:context_end]
    actual_block = "\n".join(
        f"  {context_start + 1 + j:4d} | {line}"
        for j, line in enumerate(actual_lines)
    )
    if best_ratio >= 0.5:
        confidence_note = "high-confidence"
    else:
        confidence_note = (
            "low-confidence (your old_text differs significantly — verify "
            "before retrying)"
        )

    return (
        f"No exact match, but a {best_ratio:.0%} similar block "
        f"({confidence_note}) was found at lines "
        f"{best_start + 1}-{best_start + window_size}:\n"
        f"```\n{actual_block}\n```\n"
        f"Compare this with your old_text and fix the differences "
        f"(whitespace, indentation, exact characters). If unsure, "
        f"re-read the file with line range start_line={best_start + 1}."
    )


def _unified_diff(old: str, new: str, path: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path))
    if not diff:
        return "(no changes)"
    if len(diff) > _MAX_DIFF_LINES:
        kept = _MAX_DIFF_LINES // 2
        diff = diff[:kept] + [f"\n... ({len(diff) - _MAX_DIFF_LINES} diff lines omitted) ...\n"] + diff[-kept:]
    return "".join(diff)


def register_edit_tools(
    registry: ToolRegistry,
    runtime: ExecutionRuntime,
    session: SessionStore | None = None,
) -> None:

    async def edit_file(path: str, old_text: str, new_text: str, **kwargs) -> ToolResult:
        replace_all = kwargs.get("replace_all", False)

        # Gate: require the file to have been Read first.  This prevents
        # edits based on stale or hallucinated content.  Cheap models sometimes
        # forget they already read a file (context pruning, prior turns).
        # After ``_UNREAD_BYPASS_THRESHOLD`` consecutive "file not read"
        # errors on the same path, we self-bypass with a warning — failing
        # the same way for two iterations in a row buys nothing.  The
        # subsequent Edit will either succeed or surface a different,
        # more actionable error (e.g. "old_text not found").
        if not _file_read_state.was_read(path):
            attempts = _file_read_state.note_unread_attempt(path)
            if not _file_read_state.should_bypass_read_gate(path):
                return ToolResult(
                    tool_name="edit_file",
                    success=False,
                    output=(
                        f"File has not been read yet. Read it first before editing.\n"
                        f'  Read(path="{path}")\n'
                        f"Then retry the Edit with exact text from the file."
                    ),
                    error="file_not_read",
                )
            logger.warning(
                "Bypassing read-gate for %s after %d consecutive 'file not read' errors",
                path, attempts,
            )
            # Mark as read so the rest of this Edit and subsequent Edits
            # proceed normally.  The agent already had its chance to Read.
            _file_read_state.record_read(path)

        # Read the FULL file content — no truncation.  The Edit tool
        # must see every line to match old_text reliably.  The Read tool
        # truncates for LLM display, but Edit needs the complete file.
        _read_full = getattr(runtime, "read_file_full", None)
        if callable(_read_full):
            content = await _read_full(path)
            if content is None:
                return ToolResult(
                    tool_name="edit_file",
                    success=False,
                    output=f"Cannot read file: {path} (not found or binary)",
                    error="File not readable",
                )
        else:
            # Fallback for runtimes without read_file_full.
            read_result = await runtime.read_file(path)
            if not read_result.success:
                return ToolResult(
                    tool_name="edit_file",
                    success=False,
                    output=f"Cannot read file: {read_result.error or read_result.output}",
                    error=read_result.error or "File not readable",
                )
            content = read_result.raw_output

        # Normalize line endings — CRLF/CR → LF.  Files may have mixed
        # endings from cross-platform clones or .gitattributes; the LLM
        # always provides LF.  Without this, exact match fails silently
        # on repos like sympy cloned across OSes.
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        old_text = old_text.replace("\r\n", "\n").replace("\r", "\n")
        new_text = new_text.replace("\r\n", "\n").replace("\r", "\n")

        # No-op edit guard.  When old_text == new_text the agent has
        # asked to replace text with itself — nothing changes.  Silently
        # returning success=True (with output "(no changes)") let cheap
        # models loop indefinitely "fixing" a file by submitting
        # identical-string edits, e.g. </div> → </div>.  The repeated-
        # failure tracker can't see it (success=True), and the
        # convergence outcome-signal can't see it (is_error=False), so
        # the loop only breaks when iteration cap or wall-time runs out.
        # Surface this as a real error so both trackers catch it on the
        # first repeat.
        if old_text == new_text:
            return ToolResult(
                tool_name="edit_file",
                success=False,
                output=(
                    f"old_text and new_text are identical — this edit would "
                    f"change nothing in {path}.\n\n"
                    "If you want to make a change, the new_text must differ "
                    "from old_text.  If you wanted to confirm the file's "
                    "contents are correct, use Read instead — Edit is for "
                    "modifications only.  If you're trying to fix a syntax "
                    "error, re-read the file and identify the actual broken "
                    "characters before editing."
                ),
                error="noop_edit",
            )

        count = content.count(old_text)

        if count == 0:
            # Try to help the agent understand why the match failed.
            hint = _find_fuzzy_match(content, old_text)
            recovery = (
                "\n\nRecovery strategies:\n"
                "1. Use Read to view the current file content, then retry with exact text.\n"
                "2. If the file is short (<50 lines), use Write to replace the entire file.\n"
                "3. Your cached version may be stale — context compression may have pruned "
                "the file content you remember. Always re-read before editing if unsure."
            )
            if hint:
                msg = f"old_text not found in {path}.\n\n{hint}{recovery}"
            else:
                msg = f"old_text not found in {path}.{recovery}"
            return ToolResult(
                tool_name="edit_file",
                success=False,
                output=msg,
                error="old_text not found",
            )

        if count > 1 and not replace_all:
            return ToolResult(
                tool_name="edit_file",
                success=False,
                output=(
                    f"old_text matches {count} locations in {path}. "
                    "Provide more surrounding context to make the match unique, "
                    "or use replace_all=true to replace all occurrences."
                ),
                error=f"old_text matches {count} locations",
            )

        if replace_all:
            new_content = content.replace(old_text, new_text)
        else:
            new_content = content.replace(old_text, new_text, 1)
        diff_output = _unified_diff(content, new_content, path)

        # Write the file back
        write_result = await runtime.write_file(path, new_content)
        if not write_result.success:
            return ToolResult(
                tool_name="edit_file",
                success=False,
                output=f"Edit matched but write failed: {write_result.error or write_result.output}",
                error=write_result.error or "Write failed",
            )

        # Post-edit verification: re-read the file and confirm it matches
        # what we wrote.  Catches filesystem races, encoding round-trip
        # issues (e.g., write_text normalising line endings back to CRLF
        # on Windows), or other surprises where the edit doesn't stick.
        if callable(_read_full) and old_text != new_text:
            try:
                verify_content = await _read_full(path)
                if verify_content is not None:
                    verify_norm = verify_content.replace("\r\n", "\n").replace("\r", "\n")
                    if verify_norm != new_content:
                        _file_read_state.clear(path)
                        return ToolResult(
                            tool_name="edit_file",
                            success=False,
                            output=(
                                f"Edit did not apply cleanly to {path}. "
                                f"The file on disk differs from the expected result. "
                                f"This can happen with encoding or line-ending round-trip issues. "
                                f"Try using Write to replace the entire file content instead."
                            ),
                            error="edit_not_applied",
                        )
            except Exception:
                pass  # verification is best-effort

        # Track mutations and artifacts (same pattern as write_file in filesystem.py)
        ctx = peek_current_agent_context()
        if write_result.success and ctx is not None:
            mark_mutated_path(ctx, path)
        if write_result.success and session is not None:
            ctx = current_agent_context()
            artifact_path = normalize_artifact_path(path)
            artifact = await session.snapshot_file_artifact(
                session_id=ctx.metadata["session_id"],
                logical_name=artifact_path,
                path=artifact_path,
                content=new_content,
                produced_by=ctx.agent_id,
                stage_name=ctx.stage_name,
                artifact_type=infer_artifact_type(artifact_path),
                depends_on=list(ctx.input_artifact_ids),
            )
            append_context_dependency(ctx, artifact.id)

        # Update code index (best-effort).
        if write_result.success:
            from maike.tools.context import peek_current_code_index
            code_index = peek_current_code_index()
            if code_index is not None:
                try:
                    await code_index.update_file(path, new_content)
                except Exception:
                    pass

        # Mark the file as read with fresh timestamp.  We wrote `new_content`
        # and verified it against disk above, so the agent effectively knows
        # the current file contents — no need to force a redundant Read
        # before the next Edit on this file.  Consecutive Edits on the same
        # file are common (series of small changes) and used to hit
        # `file_not_read` spuriously.
        _file_read_state.record_read(path)

        replacements = count if replace_all else 1
        return ToolResult(
            tool_name="edit_file",
            success=True,
            output=diff_output if replacements == 1 else f"Replaced {replacements} occurrences.\n\n{diff_output}",
            raw_output=diff_output,
            metadata={"path": path, "replacements": replacements},
        )

    registry.register(
        schema=ToolSchema(
            name="Edit",
            description=(
                "Apply a search-and-replace edit to a file. "
                "You MUST Read the file first — Edit will fail if the file "
                "hasn't been read in this session. After each Edit, the file "
                "must be Read again before making another Edit to the same file. "
                "old_text must match exactly one location in the file (use "
                "replace_all=true for multiple). Returns a unified diff. "
                'Example: {"path": "main.py", "old_text": "def foo():", "new_text": "def bar():"}'
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit."},
                    "old_text": {
                        "type": "string",
                        "description": (
                            "Exact text to find and replace.  Must match file content exactly "
                            "(whitespace-sensitive).  IMPORTANT: parameter name is 'old_text', "
                            "not 'search', 'find', or 'original'."
                        ),
                    },
                    "new_text": {
                        "type": "string",
                        "description": (
                            "Replacement text.  The old_text will be replaced with this.  "
                            "IMPORTANT: parameter name is 'new_text', not 'replace', "
                            "'replacement', or 'updated'."
                        ),
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Replace ALL occurrences of old_text, not just the first. "
                            "Use for renaming variables or updating repeated patterns. "
                            "Default: false (single match required)."
                        ),
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        ),
        fn=edit_file,
        risk_level=RiskLevel.WRITE,
    )

    # BatchEdit removed — caused cascading failures when multiple edits
    # targeted the same file.  Use sequential Edit calls instead.
