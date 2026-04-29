"""LSP server lifecycle manager.

Starts, manages, and shuts down LSP servers.  Provides a unified interface
for sending document notifications and collecting diagnostics.
"""
from __future__ import annotations

import logging
import os

from maike.plugins.lsp_client import Diagnostic, LSPClient, LSPClientError
from maike.plugins.lsp_config import LSPServerConfig

logger = logging.getLogger(__name__)


class LSPManager:
    """Manages LSP server lifecycles and provides a unified diagnostic interface."""

    def __init__(self, workspace_root: str = ".") -> None:
        self._clients: dict[str, LSPClient] = {}
        self._ext_to_server: dict[str, str] = {}  # ".py" -> "pyright"
        self._workspace_root = workspace_root

    async def start_servers(self, configs: list[LSPServerConfig]) -> None:
        """Start all configured LSP servers.

        Servers that fail to start are skipped with a warning.
        """
        for config in configs:
            client = LSPClient(config, workspace_root=self._workspace_root)
            try:
                await client.start()
                self._clients[config.name] = client
                # Register extension mappings
                for ext, lang in config.extension_to_language.items():
                    self._ext_to_server[ext] = config.name
                logger.info("LSP server '%s' started", config.name)
            except LSPClientError as exc:
                logger.warning("LSP server '%s' failed to start: %s", config.name, exc)
            except Exception:
                logger.warning(
                    "LSP server '%s' failed to start",
                    config.name,
                    exc_info=True,
                )

    def _client_for_file(self, file_path: str) -> LSPClient | None:
        """Find the LSP client responsible for a file based on extension."""
        ext = os.path.splitext(file_path)[1]
        server_name = self._ext_to_server.get(ext)
        if server_name is None:
            return None
        return self._clients.get(server_name)

    async def notify_file_open(self, file_path: str, content: str) -> None:
        """Notify relevant LSP server that a file was opened."""
        client = self._client_for_file(file_path)
        if client:
            await client.did_open(file_path, content)

    async def notify_file_change(self, file_path: str, content: str) -> None:
        """Notify relevant LSP server of file changes."""
        client = self._client_for_file(file_path)
        if client:
            await client.did_change(file_path, content)

    async def notify_file_save(self, file_path: str) -> None:
        """Notify relevant LSP server that a file was saved."""
        client = self._client_for_file(file_path)
        if client:
            await client.did_save(file_path)

    def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Get cached diagnostics for a file from the relevant LSP server."""
        client = self._client_for_file(file_path)
        if client:
            return client.get_diagnostics(file_path)
        return []

    def get_all_diagnostics(self) -> dict[str, list[Diagnostic]]:
        """Get all cached diagnostics from all LSP servers."""
        all_diags: dict[str, list[Diagnostic]] = {}
        for client in self._clients.values():
            all_diags.update(client.get_all_diagnostics())
        return all_diags

    def format_diagnostics_for_agent(self, file_path: str) -> str | None:
        """Format diagnostics for injection into agent context.

        Returns a formatted string, or None if no diagnostics.
        """
        diags = self.get_diagnostics(file_path)
        if not diags:
            return None

        errors = [d for d in diags if d.severity == 1]
        warnings = [d for d in diags if d.severity == 2]

        parts: list[str] = []
        if errors:
            parts.append(f"[LSP] {len(errors)} error(s) in {file_path}:")
            for d in errors[:5]:  # cap at 5
                parts.append(f"  line {d.line + 1}: {d.message}")
        if warnings:
            parts.append(f"[LSP] {len(warnings)} warning(s) in {file_path}:")
            for d in warnings[:3]:  # cap at 3
                parts.append(f"  line {d.line + 1}: {d.message}")

        return "\n".join(parts) if parts else None

    async def shutdown(self) -> None:
        """Stop all LSP servers."""
        for name, client in self._clients.items():
            try:
                await client.stop()
                logger.debug("LSP server '%s' stopped", name)
            except Exception:
                logger.warning("Error stopping LSP server '%s'", name, exc_info=True)
        self._clients.clear()
        self._ext_to_server.clear()

    @property
    def server_names(self) -> list[str]:
        """Return names of started LSP servers."""
        return list(self._clients.keys())

    @property
    def supported_extensions(self) -> list[str]:
        """Return file extensions handled by LSP servers."""
        return list(self._ext_to_server.keys())
