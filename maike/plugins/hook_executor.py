"""Hook executor — fires hooks in response to agent lifecycle events."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from maike.plugins.hooks import HookConfig, HookDefinition, HookEvent

logger = logging.getLogger(__name__)


@dataclass
class HookResult:
    """Result of executing a single hook."""

    hook: HookDefinition
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    error: str | None = None


class HookExecutor:
    """Executes hooks in response to events."""

    def __init__(
        self,
        config: HookConfig,
        env: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self._base_env = dict(os.environ)
        if env:
            self._base_env.update(env)

    async def fire(
        self,
        event: HookEvent,
        context: dict[str, Any] | None = None,
        tool_name: str | None = None,
    ) -> list[HookResult]:
        """Fire all matching hooks for an event.

        For PreToolUse/PostToolUse events, ``tool_name`` is used to filter
        hooks by their matcher pattern.

        Returns list of results (one per hook executed).
        """
        hooks = self.config.get_hooks(event, tool_name=tool_name)
        if not hooks:
            return []

        results: list[HookResult] = []
        for hook in hooks:
            result = await self._execute_hook(hook, context or {})
            results.append(result)
        return results

    async def _execute_hook(
        self,
        hook: HookDefinition,
        context: dict[str, Any],
    ) -> HookResult:
        """Execute a single hook."""
        if not hook.command:
            return HookResult(
                hook=hook,
                success=False,
                error="Empty command",
            )

        # Build environment with context variables
        env = dict(self._base_env)
        for key, value in context.items():
            if isinstance(value, str):
                env[f"MAIKE_HOOK_{key.upper()}"] = value

        try:
            proc = await asyncio.create_subprocess_shell(
                hook.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(hook.timeout_s),
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return HookResult(
                    hook=hook,
                    success=False,
                    error=f"Hook timed out after {hook.timeout_s}s",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode or 0

            return HookResult(
                hook=hook,
                success=(exit_code == 0),
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                error=stderr if exit_code != 0 else None,
            )

        except Exception as exc:
            return HookResult(
                hook=hook,
                success=False,
                error=f"Hook execution failed: {exc}",
            )
