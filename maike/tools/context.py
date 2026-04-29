"""Context variables for tool execution."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maike.atoms.context import AgentContext

if TYPE_CHECKING:
    from maike.agents.advisor import AdvisorSession
    from maike.agents.skill import SkillLoader
    from maike.context.telemetry import ContextTelemetry
    from maike.intelligence.code_index import CodeIndex
    from maike.plugins.hook_executor import HookExecutor
    from maike.plugins.lsp_manager import LSPManager


@dataclass
class AgentLoopState:
    """Live snapshot of the running AgentCore loop, exposed to tool handlers.

    The advisor tool needs to see the current conversation + iteration count
    at the moment the LLM calls ``Advisor(...)``. The AgentCore loop publishes
    this dataclass to ``CURRENT_AGENT_LOOP_STATE`` at the top of every
    iteration; tool handlers read it via ``peek_current_loop_state()``.

    ``conversation`` is kept as a live reference (not a copy) so the handler
    sees the latest messages even if the dataclass was published a few
    statements earlier in the same iteration.
    """

    conversation: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    agent_context: AgentContext | None = None


CURRENT_AGENT_CONTEXT: ContextVar[AgentContext | None] = ContextVar(
    "current_agent_context",
    default=None,
)

CURRENT_CONTEXT_TELEMETRY: ContextVar[ContextTelemetry | None] = ContextVar(
    "current_context_telemetry",
    default=None,
)

CURRENT_CODE_INDEX: ContextVar[CodeIndex | None] = ContextVar(
    "current_code_index",
    default=None,
)

CURRENT_SKILL_LOADER: ContextVar[SkillLoader | None] = ContextVar(
    "current_skill_loader",
    default=None,
)

CURRENT_HOOK_EXECUTOR: ContextVar[HookExecutor | None] = ContextVar(
    "current_hook_executor",
    default=None,
)

CURRENT_LSP_MANAGER: ContextVar[LSPManager | None] = ContextVar(
    "current_lsp_manager",
    default=None,
)

CURRENT_ADVISOR_SESSION: ContextVar["AdvisorSession | None"] = ContextVar(
    "current_advisor_session",
    default=None,
)

CURRENT_AGENT_LOOP_STATE: ContextVar[AgentLoopState | None] = ContextVar(
    "current_agent_loop_state",
    default=None,
)


def current_agent_context() -> AgentContext:
    ctx = CURRENT_AGENT_CONTEXT.get()
    if ctx is None:
        raise RuntimeError("No current agent context is active.")
    return ctx


def peek_current_agent_context() -> AgentContext | None:
    return CURRENT_AGENT_CONTEXT.get()


def peek_current_telemetry() -> ContextTelemetry | None:
    return CURRENT_CONTEXT_TELEMETRY.get()


def peek_current_code_index() -> CodeIndex | None:
    return CURRENT_CODE_INDEX.get()


def peek_current_skill_loader() -> SkillLoader | None:
    return CURRENT_SKILL_LOADER.get()


def peek_current_hook_executor() -> HookExecutor | None:
    return CURRENT_HOOK_EXECUTOR.get()


def peek_current_lsp_manager() -> LSPManager | None:
    return CURRENT_LSP_MANAGER.get()


def peek_current_advisor_session() -> "AdvisorSession | None":
    return CURRENT_ADVISOR_SESSION.get()


def peek_current_loop_state() -> AgentLoopState | None:
    return CURRENT_AGENT_LOOP_STATE.get()
