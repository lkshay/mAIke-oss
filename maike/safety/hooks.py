"""Safety-layer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
from typing import Any

from maike.atoms.context import AgentContext
from maike.atoms.tool import RiskLevel
from maike.safety.rules import BLOCKED_BASH_PATTERNS, Decision, classify_bash_command


@dataclass(frozen=True)
class SafetyAssessment:
    decision: Decision
    reason: str | None = None
    requires_checkpoint: bool = False
    approval_prompt: str | None = None


class SafetyLayer:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def assess(
        self,
        tool_name: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
        *,
        ctx: AgentContext | None = None,
    ) -> SafetyAssessment:
        raw_path = args.get("path")
        if raw_path is not None and not self._path_is_within_workspace(str(raw_path)):
            # Read-only tools outside workspace → ask permission, don't block.
            # Write/execute tools outside workspace → hard block.
            if risk_level == RiskLevel.READ:
                return SafetyAssessment(
                    decision=Decision.REQUIRE_APPROVAL,
                    reason=f"Path is outside workspace: {raw_path}",
                    approval_prompt=f"Read file outside workspace: {raw_path}? [y/N]: ",
                )
            return SafetyAssessment(
                decision=Decision.BLOCK,
                reason=f"Path escapes workspace: {raw_path}",
            )

        partition_guard = self._partition_scope_assessment(tool_name, args, ctx)
        if partition_guard is not None:
            return partition_guard

        if tool_name in {"Bash", "execute_bash"}:
            cmd = str(args.get("cmd", ""))
            lowered = cmd.lower()
            for label, pattern in BLOCKED_BASH_PATTERNS:
                if pattern.search(lowered):
                    return SafetyAssessment(
                        decision=Decision.BLOCK,
                        reason=f"Blocked bash pattern: {label}",
                    )
            # Per-command Bash risk classification.  The Bash tool registers
            # as EXECUTE, which would route every invocation through approval
            # — including verifiably read-only commands like `ls`, `cat`,
            # `git status`.  Classify the command and downgrade to READ when
            # we can prove safety via deterministic parsing.  Anything
            # ambiguous stays at EXECUTE (default-deny).  See
            # maike/safety/rules.py::classify_bash_command for the criteria.
            if classify_bash_command(cmd) == RiskLevel.READ:
                risk_level = RiskLevel.READ
            # Task-level behavioral constraints (e.g. "do not run tests") are
            # injected as a markdown <maike-constraints> block into the react
            # agent's system prompt — enforcement is prompt-level, not hard
            # block.  See maike/agents/constraints.py.

        # In react mode there are no stage checkpoints — skip this gate.
        pipeline = ctx.metadata.get("pipeline") if ctx else None
        requires_checkpoint = self._requires_checkpoint(tool_name, args, ctx)
        checkpoint_sha = self._checkpoint_sha(ctx)
        if requires_checkpoint and not checkpoint_sha and pipeline != "react":
            return SafetyAssessment(
                decision=Decision.BLOCK,
                reason=f"Stage checkpoint required before {tool_name}",
                requires_checkpoint=True,
            )

        # In react mode the user initiated the task — auto-approve WRITE
        # level tools (Write, Edit) but still require approval for EXECUTE
        # (Bash) and DESTRUCTIVE since Bash can do anything.
        if pipeline == "react":
            if risk_level == RiskLevel.WRITE:
                return SafetyAssessment(
                    decision=Decision.ALLOW,
                    requires_checkpoint=requires_checkpoint,
                )

        if risk_level in {RiskLevel.DESTRUCTIVE, RiskLevel.EXECUTE}:
            return SafetyAssessment(
                decision=Decision.REQUIRE_APPROVAL,
                requires_checkpoint=requires_checkpoint,
                approval_prompt=self._approval_prompt(tool_name, args, ctx, checkpoint_sha),
            )

        return SafetyAssessment(
            decision=Decision.ALLOW,
            requires_checkpoint=requires_checkpoint,
        )

    def _partition_scope_assessment(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: AgentContext | None,
    ) -> SafetyAssessment | None:
        if ctx is None:
            return None
        if str(ctx.metadata.get("coordination_mode")) != "partition":
            return None

        if tool_name in {"Bash", "execute_bash", "install_package", "git_commit"}:
            return SafetyAssessment(
                decision=Decision.BLOCK,
                reason=(
                    f"Tool '{tool_name}' is disabled for parallel partition agents. "
                    "Use scoped file tools and leave shared execution/integration to fan-in."
                ),
            )

        if tool_name not in {"Write", "write_file", "delete_file"}:
            return None
        raw_path = args.get("path")
        if raw_path is None:
            return None
        normalized = Path(str(raw_path)).as_posix() or "."
        scope = self._partition_scope(ctx)
        if not scope:
            return SafetyAssessment(
                decision=Decision.BLOCK,
                reason="Parallel partition agent has no file scope; refusing workspace mutation.",
            )
        if normalized not in scope:
            allowed = ", ".join(sorted(scope))
            return SafetyAssessment(
                decision=Decision.BLOCK,
                reason=(
                    f"Path '{normalized}' is outside this partition's file scope. "
                    f"Allowed files: {allowed}"
                ),
            )
        return None

    def pre_execute(
        self,
        tool_name: str,
        args: dict,
        risk_level: RiskLevel,
        *,
        ctx: AgentContext | None = None,
    ) -> Decision:
        return self.assess(tool_name, args, risk_level, ctx=ctx).decision

    def _path_is_within_workspace(self, raw_path: str) -> bool:
        candidate = (self.workspace / raw_path).resolve()
        return self.workspace == candidate or self.workspace in candidate.parents

    def _requires_checkpoint(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: AgentContext | None,
    ) -> bool:
        if tool_name in {"delete_file", "install_package", "git_commit"}:
            return True
        if tool_name in {"Write", "write_file"}:
            return self._is_multi_file_write(args, ctx)
        if tool_name in {"Bash", "execute_bash"}:
            return self._command_is_mutating(str(args.get("cmd", "")))
        return False

    def _is_multi_file_write(self, args: dict[str, Any], ctx: AgentContext | None) -> bool:
        if ctx is None:
            return False
        raw_path = args.get("path")
        if raw_path is None:
            return False
        normalized = Path(str(raw_path)).as_posix() or "."
        mutated_paths = {
            str(path)
            for path in ctx.metadata.get("mutated_paths", [])
            if path
        }
        return any(path != normalized for path in mutated_paths)

    def _checkpoint_sha(self, ctx: AgentContext | None) -> str | None:
        if ctx is None:
            return None
        value = ctx.metadata.get("stage_checkpoint_sha")
        return str(value) if value else None

    def _approval_prompt(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: AgentContext | None,
        checkpoint_sha: str | None,
    ) -> str:
        pipeline = ctx.metadata.get("pipeline") if ctx else None
        summary = self._summarize_args(args)

        # React mode: short, user-friendly prompt showing the command.
        if pipeline == "react":
            if summary:
                return f"Run {tool_name} ({summary})? [y/N]: "
            return f"Run {tool_name}? [y/N]: "

        role = ctx.role if ctx is not None else "agent"
        stage_name = ctx.stage_name if ctx is not None else "unknown"
        detail_parts = [summary] if summary else []
        if checkpoint_sha:
            detail_parts.append(f"checkpoint={checkpoint_sha[:7]}")
        detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
        return (
            f"Approve tool '{tool_name}' for agent {role} "
            f"during stage '{stage_name}'{detail}? [y/N]: "
        )

    def _summarize_args(self, args: dict[str, Any]) -> str:
        interesting_keys = ["path", "package", "message", "cmd", "timeout_class", "timeout"]
        parts: list[str] = []
        for key in interesting_keys:
            if key not in args:
                continue
            value = str(args[key]).strip()
            if len(value) > 80:
                value = f"{value[:77]}..."
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _command_is_mutating(self, cmd: str) -> bool:
        lowered = cmd.lower()
        regexes = [
            r"\bgit\s+(add|commit|reset|clean|checkout|switch|merge|rebase)\b",
            r"\b(rm|mv|cp|mkdir|touch)\b",
            r"\bsed\s+-i(?:\b|\s)",
            r"\btee\b",
            r"\b(python3?\s+-m\s+pip|pip3?|uv\s+pip|npm|pnpm|yarn|bun|poetry)\s+(install|add|remove|update)\b",
        ]
        if any(re.search(pattern, lowered) for pattern in regexes):
            return True
        try:
            tokens = shlex.split(cmd, posix=True)
        except ValueError:
            tokens = []
        benign_redirect_targets = {
            "&1",
            "&2",
            "/dev/null",
            "/dev/stdout",
            "/dev/stderr",
            "/dev/fd/1",
            "/dev/fd/2",
        }
        redirect_operators = {">", ">>", "1>", "1>>", "2>", "2>>"}
        for index, token in enumerate(tokens):
            if token in redirect_operators:
                target = tokens[index + 1] if index + 1 < len(tokens) else None
                if target is None or target not in benign_redirect_targets:
                    return True
                continue
            inline_redirect = re.match(r"^(?:\d?>>?|>>)(.+)$", token)
            if inline_redirect is not None:
                target = inline_redirect.group(1).strip()
                if not target or target not in benign_redirect_targets:
                    return True
        return bool(re.search(r"(?:^|[\s;&|])\d?>>?(?=\s|$)", cmd))

    def _partition_scope(self, ctx: AgentContext) -> set[str]:
        items = ctx.metadata.get("files_in_scope") or ctx.metadata.get("owned_deliverables") or []
        if not isinstance(items, list):
            return set()
        return {Path(str(item)).as_posix() or "." for item in items if str(item).strip()}
