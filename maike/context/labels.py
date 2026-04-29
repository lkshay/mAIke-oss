"""Tool use semantic labels — one-line summaries of tool-call batches.

After each tool-call batch, a brief label is generated describing what the
batch accomplished.  These labels are stored as metadata on the tool result
messages and survive compaction as breadcrumbs — giving the agent a trail
of what it explored even after the full results are pruned.

Labels are generated deterministically (no LLM call) by pattern-matching
the tool names and arguments.
"""

from __future__ import annotations

from typing import Any


def generate_batch_label(tool_results: list[dict[str, Any]]) -> str:
    """Generate a one-line semantic label for a batch of tool results.

    Examples:
      - "Read auth middleware and test file"
      - "Grep for DATABASE_URL across config files"
      - "Edit main.py and ran pytest"
      - "Wrote 3 new files"
    """
    if not tool_results:
        return ""

    parts: list[str] = []
    read_files: list[str] = []
    grep_patterns: list[str] = []
    edit_files: list[str] = []
    write_files: list[str] = []
    bash_cmds: list[str] = []
    other_tools: list[str] = []

    for tr in tool_results:
        tool_name = tr.get("tool_name") or tr.get("name") or ""
        inp = tr.get("input") if isinstance(tr.get("input"), dict) else {}
        canonical = _canonical_name(tool_name)

        if canonical == "Read":
            path = inp.get("path") or inp.get("file_path") or ""
            read_files.append(_short_path(path))
        elif canonical == "Grep":
            pattern = inp.get("pattern") or ""
            grep_patterns.append(pattern[:30])
        elif canonical == "Edit":
            path = inp.get("path") or inp.get("file_path") or ""
            edit_files.append(_short_path(path))
        elif canonical == "Write":
            path = inp.get("path") or inp.get("file_path") or ""
            write_files.append(_short_path(path))
        elif canonical == "Bash":
            cmd = inp.get("cmd") or inp.get("command") or ""
            bash_cmds.append(_short_cmd(cmd))
        else:
            other_tools.append(tool_name)

    if read_files:
        if len(read_files) == 1:
            parts.append(f"Read {read_files[0]}")
        else:
            parts.append(f"Read {len(read_files)} files")
    if grep_patterns:
        patterns = ", ".join(f"'{p}'" for p in grep_patterns[:2])
        parts.append(f"Grep {patterns}")
    if edit_files:
        if len(edit_files) == 1:
            parts.append(f"Edit {edit_files[0]}")
        else:
            parts.append(f"Edit {len(edit_files)} files")
    if write_files:
        if len(write_files) == 1:
            parts.append(f"Wrote {write_files[0]}")
        else:
            parts.append(f"Wrote {len(write_files)} files")
    if bash_cmds:
        parts.append(bash_cmds[0])
    if other_tools:
        parts.append(", ".join(other_tools[:2]))

    return "; ".join(parts) if parts else ""


def _canonical_name(name: str) -> str:
    """Map tool name to canonical form."""
    lower = name.lower()
    if lower in ("read", "read_file", "open_file", "view_file"):
        return "Read"
    if lower in ("grep", "grep_codebase", "search_files", "semanticsearch"):
        return "Grep"
    if lower in ("edit", "edit_file", "apply_edit", "search_replace"):
        return "Edit"
    if lower in ("write", "write_file", "create_file"):
        return "Write"
    if lower in ("bash", "execute_bash", "run_command", "run_code"):
        return "Bash"
    return name


def _short_path(path: str) -> str:
    """Shorten a file path to just the filename or last two components."""
    if not path:
        return "?"
    parts = path.replace("\\", "/").split("/")
    if len(parts) <= 2:
        return path
    return "/".join(parts[-2:])


def _short_cmd(cmd: str) -> str:
    """Shorten a bash command to a brief description."""
    cmd = cmd.strip()
    if cmd.startswith("pytest") or cmd.startswith("python -m pytest"):
        return "ran pytest"
    if cmd.startswith("npm test") or cmd.startswith("yarn test"):
        return "ran tests"
    if cmd.startswith("pip install") or cmd.startswith("npm install"):
        return "installed deps"
    if len(cmd) > 40:
        return cmd[:37] + "..."
    return cmd
