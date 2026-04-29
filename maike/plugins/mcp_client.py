"""MCP STDIO client — JSON-RPC 2.0 over subprocess stdin/stdout.

Provides tool discovery (``tools/list``) and invocation (``tools/call``)
for MCP servers.  Uses only stdlib ``asyncio.subprocess`` and ``json``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from typing import Any

from maike.plugins.mcp_config import MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPToolInfo:
    """Tool discovered from an MCP server via ``tools/list``."""

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClientError(Exception):
    """Raised when the MCP client encounters an unrecoverable error."""


class MCPClient:
    """JSON-RPC 2.0 client communicating over STDIO with an MCP server process."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._started: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self, timeout: float | None = None) -> None:
        """Spawn the MCP server subprocess and perform the initialize handshake."""
        from maike.constants import MCP_SERVER_START_TIMEOUT_S

        if self._started:
            return

        timeout = timeout if timeout is not None else float(MCP_SERVER_START_TIMEOUT_S)

        # Build environment
        env = dict(os.environ)
        env.update(self.config.env)

        cwd = self.config.cwd

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise MCPClientError(
                f"Failed to start MCP server '{self.config.name}': {exc}"
            ) from exc

        # Start background reader
        self._reader_task = asyncio.create_task(
            self._read_responses(),
            name=f"mcp-reader-{self.config.name}",
        )

        # Send initialize handshake
        try:
            result = await asyncio.wait_for(
                self._send_request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "maike", "version": "0.1.0"},
                }),
                timeout=timeout,
            )
            logger.info(
                "MCP server '%s' initialized: %s",
                self.config.name,
                result.get("serverInfo", {}).get("name", "unknown"),
            )

            # Send initialized notification
            await self._send_notification("notifications/initialized", {})

            self._started = True
        except asyncio.TimeoutError:
            await self.stop()
            raise MCPClientError(
                f"MCP server '{self.config.name}' initialize timed out after {timeout}s"
            )
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Gracefully terminate the MCP server subprocess."""
        self._started = False

        # Cancel reader
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        # Fail all pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPClientError("MCP server shutting down"))
        self._pending.clear()

        # Terminate process
        if self._process is not None and self._process.returncode is None:
            try:
                self._process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None

    # ── Tool operations ───────────────────────────────────────────────

    async def list_tools(self) -> list[MCPToolInfo]:
        """Discover tools from the MCP server."""
        if not self._started:
            raise MCPClientError(f"MCP server '{self.config.name}' not started")

        result = await self._send_request("tools/list", {})
        tools_raw = result.get("tools", [])

        tools: list[MCPToolInfo] = []
        for t in tools_raw:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not name:
                continue
            tools.append(MCPToolInfo(
                name=str(name),
                description=str(t.get("description", "")),
                input_schema=t.get("inputSchema", {}),
            ))
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Call a tool on the MCP server.

        Returns the raw result dict with ``content`` and ``isError`` fields.
        """
        from maike.constants import MCP_TOOL_CALL_TIMEOUT_S

        if not self._started:
            raise MCPClientError(f"MCP server '{self.config.name}' not started")

        timeout = timeout if timeout is not None else float(MCP_TOOL_CALL_TIMEOUT_S)

        try:
            return await asyncio.wait_for(
                self._send_request("tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                }),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise MCPClientError(
                f"MCP tool call '{tool_name}' on server '{self.config.name}' "
                f"timed out after {timeout}s"
            )

    # ── JSON-RPC transport ────────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the result."""
        if self._process is None or self._process.stdin is None:
            raise MCPClientError("No process available")

        req_id = self._next_id()
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending[req_id] = future

        line = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(line.encode("utf-8"))
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            self._pending.pop(req_id, None)
            raise MCPClientError(f"Failed to write to MCP server: {exc}") from exc

        try:
            response = await future
        finally:
            self._pending.pop(req_id, None)

        if "error" in response:
            err = response["error"]
            raise MCPClientError(
                f"MCP error ({err.get('code', '?')}): {err.get('message', '?')}"
            )

        return response.get("result", {})

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self._process is None or self._process.stdin is None:
            return

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        line = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(line.encode("utf-8"))
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def _read_responses(self) -> None:
        """Background task: read JSON-RPC responses from stdout."""
        assert self._process is not None and self._process.stdout is not None

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    # EOF — server process has ended
                    logger.warning(
                        "MCP server '%s' stdout closed (process ended)",
                        self.config.name,
                    )
                    break

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending[msg_id]
                    if not fut.done():
                        fut.set_result(msg)
                # Notifications (no id) are logged and ignored
                elif "method" in msg:
                    logger.debug(
                        "MCP notification from '%s': %s",
                        self.config.name,
                        msg.get("method"),
                    )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning(
                "MCP reader for '%s' failed",
                self.config.name,
                exc_info=True,
            )
        finally:
            # Fail all pending requests
            self._started = False
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(
                        MCPClientError(f"MCP server '{self.config.name}' connection lost")
                    )
