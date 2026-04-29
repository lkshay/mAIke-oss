"""Flat tool registry — every agent gets every tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema


@dataclass
class RegisteredTool:
    schema: ToolSchema
    fn: Callable[..., Awaitable[ToolResult]]
    risk_level: RiskLevel
    output_formatter: Callable[[ToolResult], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        schema: ToolSchema,
        fn: Callable[..., Awaitable[ToolResult]],
        risk_level: RiskLevel,
        output_formatter: Callable[[ToolResult], str] | None = None,
    ) -> None:
        self._tools[schema.name] = RegisteredTool(
            schema=schema,
            fn=fn,
            risk_level=risk_level,
            output_formatter=output_formatter or (lambda result: result.raw_output or result.output),
        )

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> list[RegisteredTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_all_schemas(self) -> list[dict]:
        """Return schemas for every registered tool."""
        return [
            {
                "name": tool.schema.name,
                "description": tool.schema.description,
                "input_schema": tool.schema.input_schema,
            }
            for tool in sorted(self._tools.values(), key=lambda t: t.schema.name)
        ]
