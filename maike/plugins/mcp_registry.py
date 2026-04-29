"""MCP tool registration — bridges MCP servers to mAIke's ToolRegistry."""
from __future__ import annotations

import json
import logging
from typing import Any

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.plugins.mcp_client import MCPClient, MCPClientError, MCPToolInfo
from maike.plugins.mcp_config import MCPServerConfig
from maike.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _extract_text_content(raw: dict[str, Any]) -> str:
    """Extract text from MCP tool call result content array."""
    content = raw.get("content", [])
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        if texts:
            return "\n".join(texts)
    # Fallback: stringify the whole result
    return json.dumps(raw, indent=2, default=str)


class MCPToolRegistry:
    """Manages MCP server lifecycles and registers their tools."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    async def start_servers(self, configs: list[MCPServerConfig]) -> None:
        """Start all configured MCP servers.

        Servers that fail to start are skipped with a warning.
        """
        for config in configs:
            client = MCPClient(config)
            try:
                await client.start()
                self._clients[config.name] = client
                logger.info("MCP server '%s' started", config.name)
            except MCPClientError as exc:
                logger.warning("MCP server '%s' failed to start: %s", config.name, exc)
            except Exception:
                logger.warning(
                    "MCP server '%s' failed to start",
                    config.name,
                    exc_info=True,
                )

    async def register_tools(self, registry: ToolRegistry) -> list[str]:
        """Discover tools from all started servers and register them.

        Tool names use the convention ``mcp__<server>__<tool>``.
        All MCP tools default to ``RiskLevel.EXECUTE`` (requires approval).

        Returns list of registered tool names.
        """
        registered: list[str] = []

        for server_name, client in self._clients.items():
            try:
                tools = await client.list_tools()
            except MCPClientError as exc:
                logger.warning(
                    "Failed to list tools from MCP server '%s': %s",
                    server_name,
                    exc,
                )
                continue

            for tool_info in tools:
                namespaced = f"mcp__{server_name}__{tool_info.name}"
                # Sanitize double underscores in component names
                namespaced = namespaced.replace("___", "__")

                handler = _make_tool_handler(client, tool_info, namespaced)

                schema = ToolSchema(
                    name=namespaced,
                    description=tool_info.description,
                    input_schema=tool_info.input_schema,
                )

                registry.register(
                    schema=schema,
                    fn=handler,
                    risk_level=RiskLevel.EXECUTE,
                )
                registered.append(namespaced)
                logger.debug("Registered MCP tool: %s", namespaced)

        if registered:
            logger.info("Registered %d MCP tools from %d servers", len(registered), len(self._clients))
        return registered

    async def shutdown(self) -> None:
        """Stop all MCP server subprocesses."""
        for name, client in self._clients.items():
            try:
                await client.stop()
                logger.debug("MCP server '%s' stopped", name)
            except Exception:
                logger.warning("Error stopping MCP server '%s'", name, exc_info=True)
        self._clients.clear()

    @property
    def server_names(self) -> list[str]:
        """Return names of started servers."""
        return list(self._clients.keys())


def _make_tool_handler(
    client: MCPClient,
    tool_info: MCPToolInfo,
    namespaced_name: str,
):
    """Create an async closure that calls an MCP tool and returns ToolResult."""

    async def _handler(**kwargs) -> ToolResult:
        try:
            raw_result = await client.call_tool(tool_info.name, kwargs)
            is_error = raw_result.get("isError", False)
            text = _extract_text_content(raw_result)

            return ToolResult(
                tool_name=namespaced_name,
                success=not is_error,
                output=text,
                raw_output=json.dumps(raw_result, default=str),
                error=text if is_error else None,
            )
        except MCPClientError as exc:
            return ToolResult(
                tool_name=namespaced_name,
                success=False,
                output="",
                error=f"MCP tool call failed: {exc}",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=namespaced_name,
                success=False,
                output="",
                error=f"Unexpected error calling MCP tool: {exc}",
            )

    return _handler
