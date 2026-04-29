"""Bash execution tools."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.constants import DEFAULT_BASH_TOOL_TIMEOUT_SECONDS, MAX_BASH_TOOL_TIMEOUT_SECONDS
from maike.runtime.protocol import ExecutionRuntime
from maike.tools.context import peek_current_agent_context
from maike.tools.filesystem import _file_read_state
from maike.tools.registry import ToolRegistry
from maike.utils import utcnow


# Patterns that indicate the command writes to the filesystem.  If any
# match, we invalidate the read-state singleton since we can't know which
# paths were affected.  Misses here cause stale-read Edit failures;
# false positives only force an (already-cheap) re-read.
_BASH_MUTATION_RE = re.compile(
    r"(?:^|[^\w])>{1,2}\s*[\w./~-]"          # shell redirection: `> file`, `>> file`
    r"|<<\s*['\"]?\w"                           # heredoc: `<<EOF`, `<<'END'`
    r"|\bsed\s+(?:-[a-zA-Z]*i\b|-i\b)"          # sed -i (in-place)
    r"|\b(?:mv|cp|rm|touch|mkdir|tee|patch)\b"  # file-moving / deleting / writing
    r"|\b(?:black|isort|autopep8|yapf)\b"       # Python formatters (mutate by default)
    r"|\bruff\s+(?:check\s+[^;|&]*--fix|format)\b"  # ruff --fix / format
    r"|\bprettier\s+[^;|&]*--write\b"           # prettier --write
    r"|\bgit\s+(?:apply|restore|reset\s+--hard|mv|rm)\b"
    r"|\bgit\s+checkout\s+--"                   # `git checkout -- <path>` (no trailing \b — `--` isn't a word char)
)


def _command_could_mutate_files(cmd: str) -> bool:
    """Return True if *cmd* plausibly modifies files on disk."""
    return bool(_BASH_MUTATION_RE.search(cmd))


# Default seconds to wait for a background process to emit startup output.
_BACKGROUND_STARTUP_WAIT = 5

_TIMEOUT_CLASS_DEFAULT = "normal"
_TIMEOUT_CLASS_LIMITS = {
    "short": (20, 10),
    "normal": (90, 20),
    "long": (180, 90),
}
_AGENT_WALL_TIME_BUDGETS = {
    "requirements": 300,
    "analysis": 300,
    "planning": 300,
    "review": 300,
    "reflection_readonly": 300,
    "architecture": 480,
    "coding": 1_200,
    "debugging": 900,
    "testing": 900,
    "acceptance": 900,
}
_LONG_COMMAND_HINTS = (
    "pytest",
    "unittest",
    "npm test",
    "pnpm test",
    "yarn test",
    "cargo test",
    "go test",
    "mvn test",
    "gradle test",
    "ruff",
    "mypy",
    "npm run build",
    "pnpm build",
    "yarn build",
    "cargo build",
)


@dataclass(frozen=True)
class BashExecutionPolicy:
    timeout_class: str
    max_runtime_seconds: int
    idle_timeout_seconds: int
    requested_timeout_seconds: int | None
    agent_wall_time_budget_seconds: int | None
    agent_wall_time_remaining_seconds: int | None
    agent_elapsed_seconds: int


def _parse_timeout_hint(timeout: int | None) -> int | None:
    if timeout is None:
        return None
    try:
        requested = int(timeout)
    except (TypeError, ValueError):
        return None
    if requested < 1:
        return None
    return min(requested, MAX_BASH_TOOL_TIMEOUT_SECONDS)


def _normalize_timeout_class(timeout_class: str | None) -> str:
    normalized = str(timeout_class or "").strip().lower()
    if normalized in _TIMEOUT_CLASS_LIMITS:
        return normalized
    return _TIMEOUT_CLASS_DEFAULT


def _infer_timeout_class(
    cmd: str,
    timeout_class: str | None,
    requested_timeout_seconds: int | None,
) -> str:
    if timeout_class:
        return _normalize_timeout_class(timeout_class)
    ctx = peek_current_agent_context()
    lowered = cmd.lower()
    if requested_timeout_seconds is not None and requested_timeout_seconds > _TIMEOUT_CLASS_LIMITS["normal"][0]:
        return "long"
    if any(hint in lowered for hint in _LONG_COMMAND_HINTS):
        return "long"
    if ctx is not None and ctx.tool_profile == "testing":
        return "long"
    if ctx is not None and ctx.tool_profile in {
        "requirements",
        "analysis",
        "planning",
        "review",
        "reflection_readonly",
    }:
        return "short"
    return _TIMEOUT_CLASS_DEFAULT


def resolve_bash_execution_policy(
    *,
    cmd: str,
    timeout: int | None = None,
    timeout_class: str | None = None,
) -> BashExecutionPolicy:
    requested_timeout_seconds = _parse_timeout_hint(timeout)
    if timeout is None and timeout_class is None:
        requested_timeout_seconds = DEFAULT_BASH_TOOL_TIMEOUT_SECONDS
    resolved_timeout_class = _infer_timeout_class(cmd, timeout_class, requested_timeout_seconds)
    max_runtime_seconds, idle_timeout_seconds = _TIMEOUT_CLASS_LIMITS[resolved_timeout_class]
    ctx = peek_current_agent_context()

    agent_wall_time_budget_seconds: int | None = None
    agent_wall_time_remaining_seconds: int | None = None
    agent_elapsed_seconds = 0
    if ctx is not None:
        agent_wall_time_budget_seconds = _AGENT_WALL_TIME_BUDGETS.get(ctx.tool_profile, 600)
        agent_elapsed_seconds = max(int((utcnow() - ctx.started_at).total_seconds()), 0)
        agent_wall_time_remaining_seconds = max(agent_wall_time_budget_seconds - agent_elapsed_seconds, 5)

    effective_timeout = max_runtime_seconds
    if requested_timeout_seconds is not None:
        effective_timeout = min(effective_timeout, requested_timeout_seconds)
    if agent_wall_time_remaining_seconds is not None:
        effective_timeout = min(effective_timeout, agent_wall_time_remaining_seconds)
    effective_timeout = max(effective_timeout, 5)

    # Idle timeout floor: never below 10s regardless of wall-time pressure.
    # A 5s idle timeout kills pip install, pytest collection, and any build.
    effective_idle_timeout = min(idle_timeout_seconds, max(effective_timeout - 1, 10))
    return BashExecutionPolicy(
        timeout_class=resolved_timeout_class,
        max_runtime_seconds=effective_timeout,
        idle_timeout_seconds=effective_idle_timeout,
        requested_timeout_seconds=requested_timeout_seconds,
        agent_wall_time_budget_seconds=agent_wall_time_budget_seconds,
        agent_wall_time_remaining_seconds=agent_wall_time_remaining_seconds,
        agent_elapsed_seconds=agent_elapsed_seconds,
    )


def register_bash_tools(registry: ToolRegistry, runtime: ExecutionRuntime) -> None:

    async def execute_bash(
        cmd: str | None = None,
        timeout: int | None = None,
        timeout_class: str | None = None,
        background: bool = False,
        stop: str | None = None,
    ) -> ToolResult:
        # ── Stop a background process ─────────────────────────────
        if stop is not None:
            return await _stop_background(stop)

        # ── Background mode ──────────────────────────────────────
        if background:
            if not cmd:
                return ToolResult(
                    tool_name="Bash",
                    success=False,
                    output="cmd is required when starting a background process.",
                )
            bg_process = await runtime.start_background(cmd)
            await asyncio.sleep(_BACKGROUND_STARTUP_WAIT)
            check_result = await runtime.check_background(bg_process.handle)
            status = check_result.get("status", "running")
            recent = check_result.get("recent_output", "")

            output_file = str(bg_process.output_file) if bg_process.output_file else "N/A"
            pid = bg_process.process.pid if bg_process.process else "unknown"

            if status == "running":
                return ToolResult(
                    tool_name="Bash",
                    success=True,
                    output=(
                        f"Background process started ({bg_process.handle}, pid {pid}).\n"
                        f"Output: {output_file}\n"
                        f"Use Read(file_path=\"{output_file}\") to check output.\n"
                        f"Use Bash(stop=\"{bg_process.handle}\") to terminate.\n"
                        f"You will be notified when it exits.\n\n"
                        f"Startup output ({_BACKGROUND_STARTUP_WAIT}s):\n{recent.strip()}"
                    ),
                    raw_output=recent,
                    metadata={"background": True, "handle": bg_process.handle, "still_running": True, "output_file": output_file},
                )
            else:
                return ToolResult(
                    tool_name="Bash",
                    success=check_result.get("exit_code", 1) == 0,
                    output=f"Process exited immediately.\nOutput: {output_file}\n\n{recent.strip()}",
                    raw_output=recent,
                    metadata={"background": True, "handle": bg_process.handle, "still_running": False, "output_file": output_file},
                )

        # ── Foreground mode (default) ────────────────────────────
        if not cmd:
            return ToolResult(
                tool_name="Bash",
                success=False,
                output="cmd is required for foreground execution.",
            )

        policy = resolve_bash_execution_policy(
            cmd=cmd,
            timeout=timeout,
            timeout_class=timeout_class,
        )
        result = await runtime.execute_bash(
            cmd,
            timeout=policy.max_runtime_seconds,
            idle_timeout=policy.idle_timeout_seconds,
            timeout_class=policy.timeout_class,
        )
        # If the command looks like it mutated files, invalidate read state
        # so any subsequent Edit forces a fresh Read.  Without this, agents
        # that run `sed -i`, `black`, `mv`, redirection, etc. between Read
        # and Edit hit silent `old_text not found` failures.  We reset
        # regardless of exit code — a command can partially mutate before
        # failing.
        if _command_could_mutate_files(cmd):
            _file_read_state.reset()
        metadata = dict(result.metadata)
        metadata.update(
            {
                "requested_timeout_seconds": policy.requested_timeout_seconds,
                "timeout_seconds": policy.max_runtime_seconds,
                "idle_timeout_seconds": policy.idle_timeout_seconds,
                "timeout_class": policy.timeout_class,
                "agent_elapsed_seconds": policy.agent_elapsed_seconds,
                "agent_wall_time_budget_seconds": policy.agent_wall_time_budget_seconds,
                "agent_wall_time_remaining_seconds": policy.agent_wall_time_remaining_seconds,
            }
        )
        return result.model_copy(update={"metadata": metadata})

    async def _stop_background(handle: str) -> ToolResult:
        """Stop a background process."""
        try:
            result = await runtime.stop_background(handle)
        except (KeyError, ValueError):
            return ToolResult(
                tool_name="Bash",
                success=False,
                output=f"Unknown handle: {handle}.",
            )

        final_output = result.get("final_output", "")
        exit_code = result.get("exit_code")

        return ToolResult(
            tool_name="Bash",
            success=True,
            output=(
                f"Process {handle} stopped (exit code: {exit_code}).\n\n"
                f"Final output:\n{final_output.strip()}"
            ),
            metadata={"handle": handle, "exit_code": exit_code},
        )

    registry.register(
        schema=ToolSchema(
            name="Bash",
            description=(
                "Execute a shell command. By default blocks until completion. "
                "Set background=true for long-running processes (servers, watchers) — returns a log file path. "
                "Use Read to check output. Use stop to terminate a background process."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": (
                            "Shell command to execute. "
                            "IMPORTANT: The parameter name is 'cmd', not 'command', 'code', or 'shell_command'. "
                            "Example: 'pytest tests/ -v --tb=short'"
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_BASH_TOOL_TIMEOUT_SECONDS,
                        "default": DEFAULT_BASH_TOOL_TIMEOUT_SECONDS,
                        "description": (
                            "Optional upper-bound hint in seconds. "
                            "The runtime may shorten it based on timeout_class and remaining agent wall-clock budget."
                        ),
                    },
                    "timeout_class": {
                        "type": "string",
                        "enum": ["short", "normal", "long"],
                        "default": _TIMEOUT_CLASS_DEFAULT,
                        "description": (
                            "Execution class for the command. "
                            "Use short for quick inspections, normal for typical commands, and long for bounded "
                            "test/build-style commands."
                        ),
                    },
                    "background": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "IMPORTANT: Set to true for any command that runs indefinitely "
                            "(servers, watchers, dev mode). Returns a log file path immediately. "
                            "Use Read(file_path=...) to check output. "
                            "You will be notified when the process exits."
                        ),
                    },
                    "stop": {
                        "type": "string",
                        "description": (
                            "Handle of a background process to stop (e.g. 'bg-001'). "
                            "Terminates the process and returns final output."
                        ),
                    },
                },
                "required": [],
            },
        ),
        fn=execute_bash,
        risk_level=RiskLevel.EXECUTE,
        output_formatter=lambda result: result.output,
    )
