"""XML tag utilities for structured context delivery.

Tags create semantic boundaries in the LLM context, enabling:
- Models to distinguish instructions from context from task
- Tag-aware pruning (drop low-priority sections first)
- Tag-aware compression (strip guidance before skills)
- Clean thread replay (strip skills on save, re-inject fresh)

Tag vocabulary::

    <maike-environment>   Project environment (language, toolchain)
    <maike-intelligence>  Hot context: symbols, related files
    <maike-memory>        Long-term memory / session learnings
    <maike-project>       MAIKE.md content
    <maike-skill>         Skill content (name, priority attrs)
    <maike-guidance>      Planning reminders, delegation hints
    <maike-task>          The user's task (highest priority)
    <maike-status>        Mid-loop: file mutations, violations
    <maike-nudge>         Mid-loop: convergence, failure hints
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Priority ordering (lower number = higher priority) ────────────────

PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
}


# ── ContextBlock ──────────────────────────────────────────────────────


@dataclass
class ContextBlock:
    """A tagged block of context for the LLM.

    Used by ``build_react_context`` to produce structured context that
    ``build_messages`` wraps in XML tags.
    """

    tag: str                # e.g. "maike-environment"
    content: str
    priority: str = "normal"  # "critical", "high", "normal", "low"
    attrs: dict[str, str] = field(default_factory=dict)


# ── Tag rendering ─────────────────────────────────────────────────────


def wrap_tag(
    tag: str,
    content: str,
    priority: str = "normal",
    **attrs: str,
) -> str:
    """Wrap *content* in an XML tag with optional attributes.

    >>> wrap_tag("maike-task", "Fix the bug", priority="critical")
    '<maike-task priority="critical">\\nFix the bug\\n</maike-task>'
    """
    parts = [f'priority="{priority}"']
    for key, value in attrs.items():
        parts.append(f'{key}="{value}"')
    attr_str = " ".join(parts)
    return f"<{tag} {attr_str}>\n{content}\n</{tag}>"


def wrap_block(block: ContextBlock) -> str:
    """Render a ``ContextBlock`` as an XML-tagged string."""
    return wrap_tag(
        block.tag,
        block.content,
        priority=block.priority,
        **block.attrs,
    )


# ── Tag stripping ─────────────────────────────────────────────────────

_TAG_PATTERN = re.compile(
    r"<(maike-\w+)[^>]*>.*?</\1>",
    re.DOTALL,
)

_SPECIFIC_TAG_CACHE: dict[str, re.Pattern] = {}


def _tag_re(tag: str) -> re.Pattern:
    """Get (cached) regex for a specific tag name."""
    if tag not in _SPECIFIC_TAG_CACHE:
        _SPECIFIC_TAG_CACHE[tag] = re.compile(
            rf"<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>",
            re.DOTALL,
        )
    return _SPECIFIC_TAG_CACHE[tag]


def strip_tag(content: str, tag: str) -> str:
    """Remove all instances of *tag* from *content*."""
    return _tag_re(tag).sub("", content).strip()


def strip_all_tags(content: str) -> str:
    """Remove all ``<maike-*>`` tags from *content*, keeping inner text."""
    # Remove closing tags
    result = re.sub(r"</maike-\w+>", "", content)
    # Remove opening tags (with attributes)
    result = re.sub(r"<maike-\w+[^>]*>", "", result)
    return result.strip()


# ── Tag extraction ────────────────────────────────────────────────────

_OPEN_TAG_RE = re.compile(r"<(maike-\w+)([^>]*)>")
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def extract_tag_info(content: str) -> tuple[str, str, dict[str, str]] | None:
    """Extract tag name, priority, and attributes from content starting with a maike tag.

    Returns ``(tag_name, priority, attrs)`` or ``None`` if no tag found.
    """
    match = _OPEN_TAG_RE.match(content.strip())
    if not match:
        return None
    tag_name = match.group(1)
    attr_str = match.group(2)
    attrs = dict(_ATTR_RE.findall(attr_str))
    priority = attrs.pop("priority", "normal")
    return tag_name, priority, attrs


def extract_tag_priority(content: str) -> tuple[str, str] | None:
    """Extract ``(tag_name, priority)`` from content starting with a maike tag.

    Lightweight version of ``extract_tag_info`` for pruner use.
    """
    info = extract_tag_info(content)
    if info is None:
        return None
    return info[0], info[1]
