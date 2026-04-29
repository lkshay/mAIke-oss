"""Local machine runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
import re
import shlex
import shutil
import time
from pathlib import Path

from maike.atoms.context import Checkpoint
from maike.atoms.tool import ToolResult
from maike.constants import DEFAULT_READ_FILE_CHAR_LIMIT, DEFAULT_READ_FILE_LINE_LIMIT
from maike.runtime.background import BackgroundProcess, BackgroundProcessManager


_SERVER_COMMAND_PATTERNS = re.compile(
    r"flask\s+run|uvicorn\s|gunicorn\s|http\.server|"
    r"npm\s+(start|run\s+dev)|pnpm\s+(start|dev)|yarn\s+(start|dev)|"
    r"node\b.*server|python.*\bapp\.py|streamlit\s+run|"
    r"next\s+dev|vite|webpack\s+serve|ng\s+serve",
    re.IGNORECASE,
)


def _looks_like_server_command(cmd: str) -> bool:
    """Return True if *cmd* looks like a long-running server process."""
    return bool(_SERVER_COMMAND_PATTERNS.search(cmd))


class WorkspacePathError(ValueError):
    """Raised when a path escapes the configured workspace."""


def _truncate_bash_section(raw: str, *, max_lines: int, hint: str) -> str:
    lines = raw.strip().splitlines()
    if len(lines) <= max_lines:
        return raw.strip()
    head_count = min(30, max_lines // 2)
    tail_count = min(30, max_lines - head_count)
    head = lines[:head_count]
    tail = lines[-tail_count:] if tail_count else []
    hidden = len(lines) - head_count - tail_count
    return "\n".join(
        [
            *head,
            "",
            f"... [{hidden} lines hidden - {hint}] ...",
            "",
            *tail,
        ]
    )


def format_bash_output(stdout: str, stderr: str, max_lines: int = 100) -> str:
    stdout = stdout.strip()
    stderr = stderr.strip()
    if not stderr:
        return _truncate_bash_section(
            stdout,
            max_lines=max_lines,
            hint="use grep_codebase or search_files for specifics",
        )

    sections: list[str] = []
    if stderr:
        sections.append(
            "[stderr]\n"
            + _truncate_bash_section(
                stderr,
                max_lines=min(60, max_lines),
                hint="stderr truncated; inspect the command directly if needed",
            )
        )
    if stdout:
        sections.append(
            "[stdout]\n"
            + _truncate_bash_section(
                stdout,
                max_lines=max(20, max_lines - 60),
                hint="stdout truncated; inspect the command directly if needed",
            )
        )
    return "\n\n".join(section for section in sections if section.strip())


@dataclass(frozen=True)
class RuntimeConfig:
    test_command: str = "pytest {path} --tb=short -q"
    install_command: str = "python3 -m pip install {package}"
    list_packages_command: str = "python3 -m pip list --format=columns"
    install_all_command: str | None = None
    lint_command: str | None = None
    package_file: str = "pyproject.toml"
    language: str = "python"
    package_manager: str | None = "pip"
    interpreter_command: str | None = "python3"
    project_root: str | None = None
    shell_env: dict[str, str] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    environment_confidence: str = "medium"
    environment_ready: bool = True

    @classmethod
    def for_language(cls, language: str) -> "RuntimeConfig":
        configs = {
            "python": cls(),
            "node": cls(
                test_command="npm test",
                install_command="npm install {package}",
                list_packages_command="npm ls --depth=0",
                install_all_command="npm install",
                package_file="package.json",
                language="node",
                package_manager="npm",
                interpreter_command="node",
            ),
            "rust": cls(
                test_command="cargo test {path}",
                install_command="cargo add {package}",
                list_packages_command="cargo tree --depth 1",
                install_all_command="cargo build",
                package_file="Cargo.toml",
                language="rust",
                package_manager="cargo",
                interpreter_command="cargo",
            ),
            "go": cls(
                test_command="go test {path}/...",
                install_command="go get {package}",
                list_packages_command="go list -m all",
                install_all_command="go mod tidy",
                package_file="go.mod",
                language="go",
                package_manager="go",
                interpreter_command="go",
            ),
        }
        return configs.get(language.lower(), cls())


class LocalRuntime:
    def __init__(
        self,
        workspace: Path,
        config: RuntimeConfig | None = None,
        manifest: EnvironmentManifest | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        # The manifest should already reflect precedence resolution before it
        # reaches the runtime. If present, it becomes the active config.
        if manifest is not None:
            self.config = manifest.to_runtime_config()
            self.manifest = manifest
        else:
            self.config = config or RuntimeConfig()
            self.manifest = None

        # Tracked background process manager for long-running commands.
        self.background_manager = BackgroundProcessManager()

        # Environment diagnostics — annotate failed tool results with
        # actionable hints (e.g., "ModuleNotFoundError → pip install X").
        from maike.runtime.diagnostics import EnvironmentDiagnostics
        self._diagnostics: EnvironmentDiagnostics | None = (
            EnvironmentDiagnostics(manifest) if manifest is not None else None
        )

    async def write_file(self, path: str, content: str) -> ToolResult:
        start = time.monotonic()
        try:
            target = self._resolve_path(path)
        except WorkspacePathError as exc:
            return self._path_error("Write", exc, start)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        execution_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(
            tool_name="Write",
            success=True,
            raw_output=str(target),
            output=f"Wrote {target.relative_to(self.workspace)}",
            execution_ms=execution_ms,
        )

    async def read_file_full(self, path: str) -> str | None:
        """Read a file's full content without truncation.

        Used internally by the Edit tool to match against the complete file.
        Returns the UTF-8 content or None if the file is missing/binary.
        """
        try:
            target = self._resolve_path(path)
        except WorkspacePathError:
            return None
        if not target.exists():
            return None
        raw_bytes = target.read_bytes()
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None

    async def read_file(self, path: str, *, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        start = time.monotonic()
        try:
            target = self._resolve_path(path)
        except WorkspacePathError as exc:
            return self._path_error("Read", exc, start)
        if not target.exists():
            error_msg = f"File not found: {target}"
            if target.parent.exists():
                import difflib
                siblings = [f.name for f in target.parent.iterdir() if f.is_file()]
                close = difflib.get_close_matches(target.name, siblings, n=3, cutoff=0.6)
                if close:
                    error_msg += f"\n\nDid you mean: {', '.join(close)}?"
            return ToolResult(
                tool_name="Read",
                success=False,
                error=error_msg,
                output=f"[ERROR] {error_msg}",
                execution_ms=int((time.monotonic() - start) * 1000),
            )
        raw_bytes = target.read_bytes()
        relative_path = target.relative_to(self.workspace).as_posix()
        metadata: dict[str, object] = {
            "path": relative_path,
            "size_bytes": len(raw_bytes),
            "binary": False,
            "truncated": False,
        }
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = f"[BINARY FILE] {relative_path} ({len(raw_bytes)} bytes)"
            metadata["binary"] = True
        else:
            total_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            metadata["total_lines"] = total_lines

            # Apply line-range slicing if requested.
            if start_line is not None or end_line is not None:
                lines = content.split("\n")
                sl = max((start_line or 1) - 1, 0)
                el = min(end_line or len(lines), len(lines))
                content = "\n".join(lines[sl:el])
                metadata["start_line"] = sl + 1
                metadata["end_line"] = el
                metadata["line_range"] = True
            elif total_lines > DEFAULT_READ_FILE_LINE_LIMIT:
                lines = content.split("\n")
                content = "\n".join(lines[:DEFAULT_READ_FILE_LINE_LIMIT])
                hidden_lines = total_lines - DEFAULT_READ_FILE_LINE_LIMIT
                next_start = DEFAULT_READ_FILE_LINE_LIMIT + 1
                next_end = min(total_lines, DEFAULT_READ_FILE_LINE_LIMIT + 200)
                content += (
                    f"\n\n... [{hidden_lines} more lines not shown — file has "
                    f"{total_lines} lines total. Use start_line/end_line to read "
                    f"specific sections, e.g. "
                    f'Read(path="{relative_path}", start_line={next_start}, '
                    f"end_line={next_end})]"
                )
                metadata["truncated"] = True
                metadata["total_lines"] = total_lines
            elif len(content) > DEFAULT_READ_FILE_CHAR_LIMIT:
                # Estimate how many lines were shown vs total.
                shown_lines = content[:DEFAULT_READ_FILE_CHAR_LIMIT].count("\n") + 1
                content = (
                    f"{content[:DEFAULT_READ_FILE_CHAR_LIMIT]}\n\n"
                    f"... [file truncated — showing ~{shown_lines} of {total_lines} lines. "
                    f"Use start_line/end_line to read specific sections, e.g. "
                    f'Read(path="{relative_path}", start_line={shown_lines + 1}, '
                    f"end_line={min(total_lines, shown_lines + 200)})]"
                )
                metadata["truncated"] = True
                metadata["total_lines"] = total_lines
        execution_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(
            tool_name="Read",
            success=True,
            raw_output=content,
            output=content,
            execution_ms=execution_ms,
            metadata=metadata,
        )

    async def execute_bash(
        self,
        cmd: str,
        timeout: float = 30,
        *,
        idle_timeout: float | None = None,
        timeout_class: str | None = None,
    ) -> ToolResult:
        start = time.monotonic()
        env = dict(os.environ)
        env.update(self.config.shell_env)
        process = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(self.workspace),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout_text, stderr_text, timeout_kind, returncode = await self._collect_process_output(
            process,
            timeout=timeout,
            idle_timeout=idle_timeout,
        )
        raw = "\n".join(part for part in [stdout_text, stderr_text] if part).strip()
        execution_ms = int((time.monotonic() - start) * 1000)
        metadata = {
            "returncode": returncode,
            "stderr_present": bool(stderr_text.strip()),
            "timeout_seconds": timeout,
            "idle_timeout_seconds": idle_timeout,
            "timeout_class": timeout_class,
            "package_manager": self.config.package_manager,
            "interpreter_command": self.config.interpreter_command,
            "environment_confidence": self.config.environment_confidence,
        }
        if timeout_kind is not None:
            if timeout_kind == "idle":
                error = f"Command idle timed out after {idle_timeout}s"
            else:
                error = f"Command timed out after {timeout}s"
            formatted = format_bash_output(stdout_text, stderr_text)
            output_parts = [f"[ERROR] {error}"]
            if formatted.strip():
                output_parts.append(formatted)

            # Actionable recovery guidance based on timeout type.
            if timeout_kind == "idle" and _looks_like_server_command(cmd):
                output_parts.append(
                    "This looks like a long-running server. "
                    "Re-run with background=true:\n"
                    f'  Bash(cmd="{cmd}", background=true)'
                )
            elif timeout_kind == "idle":
                output_parts.append(
                    f"The command produced no output for {idle_timeout}s so it "
                    f"was stopped. This does NOT mean the command failed — it "
                    f"may just be slow (e.g. installing packages, collecting "
                    f"tests, compiling). To fix:\n"
                    f'  - Use a longer timeout: Bash(cmd="{cmd}", timeout_class="long")\n'
                    f'  - Or run in background: Bash(cmd="{cmd}", background=true)'
                )
            else:
                output_parts.append(
                    f"The command exceeded the {timeout}s wall-clock limit. "
                    f"To fix:\n"
                    f'  - Use a longer timeout: Bash(cmd="{cmd}", timeout=180)\n'
                    f'  - Or run in background: Bash(cmd="{cmd}", background=true)'
                )
            metadata["timeout_kind"] = timeout_kind
            return ToolResult(
                tool_name="Bash",
                success=False,
                raw_output=raw,
                output="\n\n".join(output_parts),
                error=error,
                execution_ms=execution_ms,
                metadata=metadata,
            )
        result = ToolResult(
            tool_name="Bash",
            success=returncode == 0,
            raw_output=raw,
            output=format_bash_output(stdout_text, stderr_text),
            error=None if returncode == 0 else f"Command exited with {returncode}",
            execution_ms=execution_ms,
            metadata=metadata,
        )
        if not result.success and self._diagnostics:
            result = self._diagnostics.annotate_tool_result(result)
        return result

    async def install_package(self, package: str) -> ToolResult:
        cmd = self.config.install_command.format(package=shlex.quote(package))
        return await self.execute_bash(cmd, timeout=300)

    async def run_tests(self, path: str) -> ToolResult:
        target = shlex.quote(path)
        cmd = self.config.test_command.format(path=target)
        result = await self.execute_bash(cmd, timeout=300)
        returncode = result.metadata.get("returncode")
        raw = result.raw_output.lower()
        if returncode == 5 and ("no tests ran" in raw or "collected 0 items" in raw):
            return result.model_copy(
                update={
                    "tool_name": "run_tests",
                    "success": True,
                    "error": None,
                    "output": (
                        "No automated tests were found for this workspace. "
                        "Treating this as a non-fatal validation result.\n\n"
                        f"{result.output}"
                    ).strip(),
                }
            )
        return result.model_copy(update={"tool_name": "run_tests"})

    async def checkpoint(self, label: str, step: str) -> Checkpoint:
        if not await self.is_git_repo():
            raise RuntimeError("Workspace is not a git repository; checkpoint unavailable.")
        # Suppress git warnings about ignored files (e.g. .venv in .gitignore)
        # which cause execute_bash to report failure due to stderr output.
        add_result = await self.execute_bash(
            "git -c advice.addIgnoredFile=false add -A -- . "
            "':(exclude).maike' ':(exclude).venv' ':(exclude)venv' ':(exclude)node_modules'"
        )
        if not add_result.success:
            # Only fail on real errors, not warnings about ignored files
            err_text = (add_result.output or add_result.error or "").lower()
            if "ignored" not in err_text and "hint:" not in err_text:
                raise RuntimeError(add_result.output or add_result.error or "Failed to stage checkpoint changes.")
        untrack_maike = await self.execute_bash("git rm -r --cached --ignore-unmatch .maike")
        if not untrack_maike.success:
            raise RuntimeError(untrack_maike.output or untrack_maike.error or "Failed to exclude .maike from checkpoint.")
        commit = await self.execute_bash(
            f"git commit --allow-empty -m {shlex.quote(f'maike-checkpoint: {label} [{step}]')}"
        )
        if not commit.success:
            raise RuntimeError(commit.output or commit.error or "Failed to create checkpoint.")
        sha_result = await self.execute_bash("git rev-parse HEAD")
        if not sha_result.success:
            raise RuntimeError(sha_result.output or sha_result.error or "Failed to resolve checkpoint sha.")
        return Checkpoint(sha=sha_result.raw_output.strip(), label=label, step=step)

    async def restore(self, checkpoint: Checkpoint) -> None:
        result = await self.execute_bash(f"git reset --hard {shlex.quote(checkpoint.sha)}")
        if not result.success:
            raise RuntimeError(result.output or result.error or "Failed to restore checkpoint.")
        clean_result = await self.execute_bash(
            "git clean -fd -e .maike/ -e .venv/ -e venv/ -e node_modules/"
        )
        if not clean_result.success:
            raise RuntimeError(clean_result.output or clean_result.error or "Failed to clean workspace.")

    # Directories filtered from list_dir output to reduce noise.
    # Mirrors _STRUCTURE_SKIP_DIRS in probe.py.
    _LIST_DIR_SKIP = frozenset({
        ".git", ".svn", "node_modules", ".venv", "venv", "__pycache__",
        ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build",
        ".eggs", ".tox", ".nox", ".maike",
    })

    async def list_dir(self, path: str) -> ToolResult:
        start = time.monotonic()
        try:
            target = self._resolve_path(path)
        except WorkspacePathError as exc:
            return self._path_error("Bash", exc, start)
        if not target.exists():
            return ToolResult(
                tool_name="Bash",
                success=False,
                output=f"[ERROR] Directory not found: {target}",
                error=f"Directory not found: {target}",
                execution_ms=int((time.monotonic() - start) * 1000),
            )
        if target.is_file():
            entries = [target.name]
        else:
            all_children = list(target.iterdir())
            entries = sorted(
                child.name for child in all_children
                if child.name not in self._LIST_DIR_SKIP
            )
            hidden_count = len(all_children) - len(entries)

        raw = "\n".join(entries)
        if not target.is_file() and hidden_count > 0:
            raw += f"\n\n({hidden_count} noise directories hidden: .git, __pycache__, etc.)"
        return ToolResult(
            tool_name="Bash",
            success=True,
            raw_output=raw,
            output=raw,
            execution_ms=int((time.monotonic() - start) * 1000),
        )

    async def delete_file(self, path: str) -> ToolResult:
        start = time.monotonic()
        try:
            target = self._resolve_path(path)
        except WorkspacePathError as exc:
            return self._path_error("Bash", exc, start)
        if not target.exists():
            return ToolResult(
                tool_name="Bash",
                success=False,
                output=f"[ERROR] File not found: {target}",
                error=f"File not found: {target}",
                execution_ms=int((time.monotonic() - start) * 1000),
            )
        target.unlink()
        return ToolResult(
            tool_name="Bash",
            success=True,
            raw_output=str(target),
            output=f"Deleted {target.relative_to(self.workspace)}",
            execution_ms=int((time.monotonic() - start) * 1000),
        )

    async def git_available(self) -> bool:
        return shutil.which("git") is not None

    async def is_git_repo(self) -> bool:
        if not await self.git_available():
            return False
        result = await self.execute_bash("git rev-parse --is-inside-work-tree")
        return result.success and result.raw_output.strip() == "true"

    async def init_git_repo(self) -> ToolResult:
        result = await self.execute_bash("git init", timeout=30)
        if not result.success:
            return result

        email = await self.execute_bash("git config --get user.email", timeout=30)
        if not email.success or not email.raw_output.strip():
            configured = await self.execute_bash("git config user.email maike@local", timeout=30)
            if not configured.success:
                return configured

        name = await self.execute_bash("git config --get user.name", timeout=30)
        if not name.success or not name.raw_output.strip():
            configured = await self.execute_bash("git config user.name mAIke", timeout=30)
            if not configured.success:
                return configured

        return result

    async def start_background(self, cmd: str) -> BackgroundProcess:
        """Start a tracked background process in the workspace."""
        env = dict(os.environ)
        env.update(self.config.shell_env)
        return await self.background_manager.start(cmd, cwd=self.workspace, env=env)

    async def check_background(self, handle: str) -> dict:
        """Check status and recent output of a tracked background process."""
        return await self.background_manager.check(handle)

    async def stop_background(self, handle: str) -> dict:
        """Stop a tracked background process."""
        return await self.background_manager.stop(handle)

    async def _collect_process_output(
        self,
        process: asyncio.subprocess.Process,
        *,
        timeout: float,
        idle_timeout: float | None,
    ) -> tuple[str, str, str | None, int | None]:
        queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_open = True
        stderr_open = True
        start = time.monotonic()
        last_output_at = start
        deadline = start + timeout

        async def _pump(stream_name: str, stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                await queue.put((stream_name, None))
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    await queue.put((stream_name, None))
                    return
                await queue.put((stream_name, chunk))

        stdout_task = asyncio.create_task(_pump("stdout", process.stdout))
        stderr_task = asyncio.create_task(_pump("stderr", process.stderr))
        wait_task = asyncio.create_task(process.wait())
        timeout_kind: str | None = None

        try:
            while True:
                if wait_task.done() and not stdout_open and not stderr_open:
                    break

                now = time.monotonic()
                remaining_runtime = deadline - now
                if remaining_runtime <= 0:
                    timeout_kind = "max_runtime"
                    break

                if wait_task.done():
                    wait_timeout = remaining_runtime
                else:
                    if idle_timeout is None:
                        wait_timeout = remaining_runtime
                    else:
                        remaining_idle = (last_output_at + idle_timeout) - now
                        if remaining_idle <= 0:
                            timeout_kind = "idle"
                            break
                        wait_timeout = min(remaining_runtime, remaining_idle)

                queue_get = asyncio.create_task(queue.get())
                wait_set: set[asyncio.Task[object]] = {queue_get}
                if not wait_task.done():
                    wait_set.add(wait_task)
                done: set[asyncio.Task[object]] = set()
                try:
                    done, _ = await asyncio.wait(
                        wait_set,
                        timeout=wait_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    if queue_get not in done:
                        queue_get.cancel()
                        await asyncio.gather(queue_get, return_exceptions=True)

                if not done:
                    timeout_kind = "idle" if idle_timeout is not None and not wait_task.done() else "max_runtime"
                    break

                if queue_get in done:
                    stream_name, chunk = queue_get.result()
                    if chunk is None:
                        if stream_name == "stdout":
                            stdout_open = False
                        else:
                            stderr_open = False
                    else:
                        last_output_at = time.monotonic()
                        if stream_name == "stdout":
                            stdout_chunks.append(chunk)
                        else:
                            stderr_chunks.append(chunk)

            if timeout_kind is not None and not wait_task.done():
                # Kill the entire process group to clean up child processes
                # (e.g. servers spawned by shell commands).
                try:
                    os.killpg(process.pid, 9)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                await process.wait()
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            if not wait_task.done():
                await asyncio.gather(wait_task, return_exceptions=True)

        while not queue.empty():
            stream_name, chunk = queue.get_nowait()
            if chunk is None:
                continue
            if stream_name == "stdout":
                stdout_chunks.append(chunk)
            else:
                stderr_chunks.append(chunk)

        return (
            b"".join(stdout_chunks).decode(errors="replace"),
            b"".join(stderr_chunks).decode(errors="replace"),
            timeout_kind,
            process.returncode,
        )

    def _resolve_path(self, raw_path: str) -> Path:
        target = (self.workspace / raw_path).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise WorkspacePathError(f"Path escapes workspace: {raw_path}")
        return target

    def _path_error(self, tool_name: str, exc: WorkspacePathError, start: float) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            output=f"[ERROR] {exc}",
            error=str(exc),
            execution_ms=int((time.monotonic() - start) * 1000),
        )
