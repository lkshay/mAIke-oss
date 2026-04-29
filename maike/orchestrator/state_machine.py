"""Pipeline definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    name: str
    agent_role: str
    deps: list[str]
    tool_profile: str
    parallel_capable: bool = False
    produces: list[str] = field(default_factory=list)
    required_for_next: bool = True


REACT_PIPELINE = [
    Stage("react", "react_agent", [], "react", produces=["output.md"]),
]
