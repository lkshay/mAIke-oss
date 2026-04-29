"""Tool registration entrypoints."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from maike.atoms.tool import ToolResult
from maike.memory.session import SessionStore
from maike.runtime.protocol import ExecutionRuntime
from maike.tools.bash import register_bash_tools
from maike.tools.edit import register_edit_tools
from maike.tools.filesystem import register_filesystem_tools
from maike.tools.registry import ToolRegistry
from maike.tools.search import register_search_tools, register_semantic_search_tool
from maike.tools.user_input import register_user_input_tool
from maike.tools.web import register_web_tools

if TYPE_CHECKING:
    from maike.agents.skill import SkillLoader


def register_default_tools(
    registry: ToolRegistry,
    runtime: ExecutionRuntime,
    session: SessionStore | None = None,
    delegate_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    user_input_handler: Callable[[str, str], Awaitable[ToolResult]] | None = None,
    async_delegate_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    delegate_check_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    delegate_wait_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    delegate_send_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    team_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    advisor_handler: Callable[..., Awaitable[ToolResult]] | None = None,
    skill_loader: SkillLoader | None = None,
) -> ToolRegistry:
    register_filesystem_tools(registry, runtime, session=session)
    register_edit_tools(registry, runtime, session=session)
    register_bash_tools(registry, runtime)
    register_search_tools(registry, runtime)
    register_semantic_search_tool(registry, runtime)
    register_web_tools(registry, runtime)
    if delegate_handler is not None:
        from maike.tools.delegate import register_delegate_tool
        register_delegate_tool(
            registry,
            delegate_handler,
            async_handler=async_delegate_handler,
            check_handler=delegate_check_handler,
            wait_handler=delegate_wait_handler,
            send_handler=delegate_send_handler,
        )
    if team_handler is not None:
        from maike.tools.team import register_team_tool
        register_team_tool(registry, team_handler)
    if advisor_handler is not None:
        from maike.tools.advisor import register_advisor_tool
        register_advisor_tool(registry, advisor_handler)
    if user_input_handler is not None:
        register_user_input_tool(registry, user_input_handler)
    if skill_loader is not None:
        from maike.tools.skill import register_skill_tool
        register_skill_tool(
            registry, skill_loader,
            delegate_handler=delegate_handler,
        )
    return registry
