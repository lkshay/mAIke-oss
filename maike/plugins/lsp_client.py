"""LSP STDIO client — JSON-RPC 2.0 over subprocess.

Lightweight Language Server Protocol client for diagnostics, go-to-definition,
and find-references.  Uses only stdlib asyncio.subprocess and json.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from typing import Any

from maike.plugins.lsp_config import LSPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class Diagnostic:
    """A single diagnostic (error/warning) from an LSP server."""

    file_path: str
    line: int         # 0-based
    character: int    # 0-based
    severity: int     # 1=Error, 2=Warning, 3=Info, 4=Hint
    message: str
    source: str = ""

    @property
    def severity_label(self) -> str:
        return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(self.severity, "unknown")

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line + 1}:{self.character + 1}: {self.severity_label}: {self.message}"


class LSPClientError(Exception):
    """Raised for LSP client errors."""


class LSPClient:
    """JSON-RPC 2.0 client for LSP servers over STDIO."""

    def __init__(self, config: LSPServerConfig, workspace_root: str = ".") -> None:
        self.config = config
        self.workspace_root = workspace_root
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._started: bool = False
        self._diagnostics: dict[str, list[Diagnostic]] = {}  # uri -> diagnostics
        self._restart_count: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start LSP server and perform initialize handshake."""
        env = dict(os.environ)
        env.update(self.config.env)

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise LSPClientError(f"Failed to start LSP server '{self.config.name}': {exc}") from exc

        self._reader_task = asyncio.create_task(
            self._read_messages(),
            name=f"lsp-reader-{self.config.name}",
        )

        try:
            result = await asyncio.wait_for(
                self._send_request("initialize", {
                    "processId": os.getpid(),
                    "rootUri": f"file://{os.path.abspath(self.workspace_root)}",
                    "capabilities": {
                        "textDocument": {
                            "publishDiagnostics": {"relatedInformation": True},
                        },
                    },
                    "initializationOptions": self.config.initialization_options,
                }),
                timeout=self.config.startup_timeout_ms / 1000.0,
            )
            logger.info("LSP server '%s' initialized", self.config.name)

            await self._send_notification("initialized", {})
            self._started = True
        except asyncio.TimeoutError:
            await self.stop()
            raise LSPClientError(f"LSP server '{self.config.name}' initialize timed out")
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Shutdown the LSP server."""
        self._started = False

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(LSPClientError("LSP server shutting down"))
        self._pending.clear()

        if self._process and self._process.returncode is None:
            try:
                # Try graceful shutdown
                await asyncio.wait_for(
                    self._send_request("shutdown", {}),
                    timeout=2.0,
                )
                await self._send_notification("exit", {})
            except Exception:
                pass
            try:
                self._process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self._process.wait(), timeout=self.config.shutdown_timeout_ms / 1000.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass
            self._process = None

    # ── Document notifications ────────────────────────────────────────

    async def did_open(self, file_path: str, content: str, language_id: str = "") -> None:
        """Notify server that a document was opened."""
        if not self._started:
            return
        if not language_id:
            ext = os.path.splitext(file_path)[1]
            language_id = self.config.extension_to_language.get(ext, "")
        uri = f"file://{os.path.abspath(file_path)}"
        await self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": 1,
                "text": content,
            },
        })

    async def did_change(self, file_path: str, content: str, version: int = 2) -> None:
        """Notify server of document changes (full content replacement)."""
        if not self._started:
            return
        uri = f"file://{os.path.abspath(file_path)}"
        await self._send_notification("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": content}],
        })

    async def did_save(self, file_path: str) -> None:
        """Notify server that a document was saved."""
        if not self._started:
            return
        uri = f"file://{os.path.abspath(file_path)}"
        await self._send_notification("textDocument/didSave", {
            "textDocument": {"uri": uri},
        })

    # ── Diagnostics ───────────────────────────────────────────────────

    def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Return cached diagnostics for a file (populated via publishDiagnostics)."""
        uri = f"file://{os.path.abspath(file_path)}"
        return self._diagnostics.get(uri, [])

    def get_all_diagnostics(self) -> dict[str, list[Diagnostic]]:
        """Return all cached diagnostics."""
        return dict(self._diagnostics)

    # ── JSON-RPC transport (LSP uses Content-Length headers) ──────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None:
            raise LSPClientError("No process available")

        req_id = self._next_id()
        message = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending[req_id] = future

        await self._write_message(message)

        try:
            response = await future
        finally:
            self._pending.pop(req_id, None)

        if "error" in response:
            err = response["error"]
            raise LSPClientError(f"LSP error ({err.get('code', '?')}): {err.get('message', '?')}")
        return response.get("result", {})

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            return
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            await self._write_message(message)
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def _write_message(self, message: dict) -> None:
        """Write an LSP message with Content-Length header."""
        assert self._process is not None and self._process.stdin is not None
        body = json.dumps(message)
        header = f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n"
        self._process.stdin.write(header.encode("utf-8"))
        self._process.stdin.write(body.encode("utf-8"))
        await self._process.stdin.drain()

    async def _read_messages(self) -> None:
        """Background task: read LSP messages from stdout."""
        assert self._process is not None and self._process.stdout is not None
        reader = self._process.stdout

        try:
            while True:
                # Read Content-Length header
                content_length = 0
                while True:
                    line = await reader.readline()
                    if not line:
                        logger.warning("LSP server '%s' stdout closed", self.config.name)
                        return
                    decoded = line.decode("utf-8").strip()
                    if not decoded:
                        break  # empty line = end of headers
                    if decoded.lower().startswith("content-length:"):
                        content_length = int(decoded.split(":", 1)[1].strip())

                if content_length == 0:
                    continue

                body = await reader.readexactly(content_length)
                try:
                    msg = json.loads(body)
                except json.JSONDecodeError:
                    continue

                # Dispatch
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending[msg_id]
                    if not fut.done():
                        fut.set_result(msg)
                elif msg.get("method") == "textDocument/publishDiagnostics":
                    self._handle_diagnostics(msg.get("params", {}))
                elif "method" in msg:
                    logger.debug("LSP notification: %s", msg.get("method"))
        except asyncio.CancelledError:
            return
        except asyncio.IncompleteReadError:
            logger.warning("LSP server '%s' disconnected", self.config.name)
        except Exception:
            logger.warning("LSP reader for '%s' failed", self.config.name, exc_info=True)
        finally:
            self._started = False
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(LSPClientError("LSP connection lost"))

    def _handle_diagnostics(self, params: dict[str, Any]) -> None:
        """Process publishDiagnostics notification."""
        uri = params.get("uri", "")
        diagnostics_raw = params.get("diagnostics", [])

        diags: list[Diagnostic] = []
        for d in diagnostics_raw:
            if not isinstance(d, dict):
                continue
            rng = d.get("range", {})
            start = rng.get("start", {})
            diags.append(Diagnostic(
                file_path=uri.replace("file://", ""),
                line=int(start.get("line", 0)),
                character=int(start.get("character", 0)),
                severity=int(d.get("severity", 1)),
                message=str(d.get("message", "")),
                source=str(d.get("source", "")),
            ))

        self._diagnostics[uri] = diags
        if diags:
            logger.debug(
                "LSP diagnostics for %s: %d items",
                uri,
                len(diags),
            )
