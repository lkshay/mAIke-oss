"""Agent context builders."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from maike.agents.react import build_react_context
from maike.atoms.context import AgentContext

StaticContextBuilder = Callable[..., Awaitable[tuple[AgentContext, list[dict[str, Any]]]]]

STATIC_CONTEXT_BUILDERS: dict[str, StaticContextBuilder] = {
    "react": build_react_context,
}


def get_static_builder(stage_name: str) -> StaticContextBuilder | None:
    return STATIC_CONTEXT_BUILDERS.get(stage_name)


__all__ = [
    "STATIC_CONTEXT_BUILDERS",
    "StaticContextBuilder",
    "build_react_context",
    "get_static_builder",
]
