"""Human-readable activity descriptions for tool calls."""

from __future__ import annotations


def get_activity_description(tool_name: str, input_params: dict) -> str:
    """Return a short human-readable description for a tool invocation.

    Used by progress tracking, spinners, and task status displays.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is not None:
        return handler(input_params)
    return f"Running {tool_name}"


def _desc_read(p: dict) -> str:
    path = p.get("path", "")
    start = p.get("start_line")
    end = p.get("end_line")
    if start and end:
        return f"Reading {path}:{start}-{end}"
    return f"Reading {path}"


def _desc_write(p: dict) -> str:
    return f"Writing {p.get('path', '')}"


def _desc_edit(p: dict) -> str:
    return f"Editing {p.get('path', '')}"


def _desc_grep(p: dict) -> str:
    pattern = p.get("pattern", "")
    path = p.get("path", ".")
    return f"Searching for '{pattern[:40]}' in {path}"


def _desc_bash(p: dict) -> str:
    cmd = p.get("cmd", "")
    if not cmd:
        return "Running Bash"
    return f"Running: {cmd[:60]}"


def _desc_delegate(p: dict) -> str:
    agent_type = p.get("agent_type", "implement")
    task = p.get("task", "")
    return f"Delegating ({agent_type}): {task[:50]}"


def _desc_web_search(p: dict) -> str:
    return f"Searching web: {p.get('query', '')[:50]}"


def _desc_web_fetch(p: dict) -> str:
    return f"Fetching: {p.get('url', '')[:50]}"


def _desc_semantic_search(p: dict) -> str:
    return f"Semantic search: {p.get('query', '')[:50]}"


_HANDLERS: dict[str, callable] = {
    "Read": _desc_read,
    "Write": _desc_write,
    "Edit": _desc_edit,
    "Grep": _desc_grep,
    "Bash": _desc_bash,
    "Delegate": _desc_delegate,
    "WebSearch": _desc_web_search,
    "WebFetch": _desc_web_fetch,
    "SemanticSearch": _desc_semantic_search,
}
