"""Delegate agent context builder.

Builds a minimal context for sub-agents spawned via the Delegate tool.
The sub-agent gets a focused task, optional context from the parent,
and tools appropriate for its agent_type.  No AskUser
(sub-agents work autonomously) and no Delegate (prevents recursion).

Supported agent types:
  - "implement" (default): full coding tools (Read, Write, Edit, Grep, Bash)
  - "explore": read-only codebase research (Read, Grep, SemanticSearch)
  - "verify": adversarial testing, read-only (Read, Grep, Bash)
"""

from __future__ import annotations

from pathlib import Path

from maike.agents.helpers import (
    build_context,
    build_environment_context_blocks,
    build_environment_metadata,
    build_messages,
    build_workspace_snapshot,
)
from maike.constants import DEFAULT_MODEL

# Budget for MAIKE.md content injected into delegate context.
_MAIKE_MD_CHAR_CAP = 2000


_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Mapping from agent_type to prompt filename and tool_profile.
_AGENT_TYPE_CONFIG: dict[str, tuple[str, str]] = {
    "explore":   ("delegate-explore.md",   "delegate_explore"),
    "plan":      ("delegate-plan.md",      "delegate_plan"),
    "implement": ("delegate-implement.md", "delegate"),
    "review":    ("delegate-review.md",    "delegate_review"),
    "verify":    ("delegate-verify.md",    "delegate_verify"),
    "debug":     ("delegate-debug.md",     "delegate_debug"),
    "test":      ("delegate-test.md",      "delegate"),
}

_DEFAULT_AGENT_TYPE = "implement"

# Cache loaded prompts to avoid re-reading files.
_prompt_cache: dict[str, str] = {}


def _load_delegate_prompt(agent_type: str) -> str:
    """Load the prompt for the given agent_type from the prompts directory.

    Falls back to "implement" for unknown types.
    """
    if agent_type not in _AGENT_TYPE_CONFIG:
        agent_type = _DEFAULT_AGENT_TYPE
    if agent_type in _prompt_cache:
        return _prompt_cache[agent_type]

    filename = _AGENT_TYPE_CONFIG[agent_type][0]
    prompt_path = _PROMPTS_DIR / filename
    prompt = prompt_path.read_text(encoding="utf-8")
    _prompt_cache[agent_type] = prompt
    return prompt


def tool_profile_for_agent_type(agent_type: str) -> str:
    """Return the tool_profile string for a given agent_type."""
    if agent_type not in _AGENT_TYPE_CONFIG:
        agent_type = _DEFAULT_AGENT_TYPE
    return _AGENT_TYPE_CONFIG[agent_type][1]


async def build_delegate_context(
    task: str,
    context: str,
    session,
    parent_id: str | None = None,
    model: str = DEFAULT_MODEL,
    adaptive_model: bool = True,
    system_prompt_override: str | None = None,
    agent_type: str = _DEFAULT_AGENT_TYPE,
    agent_def=None,
):
    """Build minimal context for a delegated sub-agent.

    The delegate gets:
      - task description
      - parent's context summary (if provided)
      - environment info (language, package manager)
      - type-specific system prompt (or plugin agent's prompt if overridden)
    No artifacts, no workspace snapshot — the agent explores on demand.
    """
    profile = tool_profile_for_agent_type(agent_type)
    ctx = build_context(
        role="delegate",
        task=task,
        stage_name="delegate",
        tool_profile=profile,
        parent_id=parent_id,
        model=model,
        input_artifacts=[],
        session_id=session.id,
        metadata={
            **build_environment_metadata(session),
            "pipeline": "react",
            "adaptive_model": adaptive_model,
            "agent_type": agent_type,
        },
    )

    # Inject critical_reminder for verification delegates so every LLM turn
    # receives a constraint nudge — prevents the verifier from drifting into
    # editing files or forgetting to emit a VERDICT.
    if agent_type == "verify":
        ctx.critical_reminder = (
            "CRITICAL: This is a VERIFICATION-ONLY task. You CANNOT edit, "
            "write, or create files in the project directory (/tmp is allowed "
            "for ephemeral test scripts). You MUST end with VERDICT: PASS, "
            "VERDICT: FAIL, or VERDICT: PARTIAL."
        )

    # Apply rich config from custom agent definitions.
    if agent_def is not None:
        if agent_def.critical_reminder:
            ctx.critical_reminder = agent_def.critical_reminder
        if agent_def.max_turns is not None:
            ctx.max_iterations = agent_def.max_turns

    # The agent-type-specific prompt is now loaded as the system prompt by
    # AgentCore._render_system_prompt() (using ctx.metadata["agent_type"]).
    # Only include it as a user-message context block when a custom
    # system_prompt_override is provided (custom/plugin agents).
    context_blocks: list[str] = []
    if system_prompt_override:
        context_blocks.append(system_prompt_override)
    context_blocks.extend(build_environment_context_blocks(session))

    # Inject project context so delegates aren't blind to the codebase.
    # MAIKE.md: project conventions, architecture, key decisions.
    from maike.agents.react import read_maike_md
    workspace = getattr(session, "workspace", None)
    if workspace:
        maike_content = read_maike_md(workspace)
        if maike_content:
            truncated = maike_content[:_MAIKE_MD_CHAR_CAP]
            if len(maike_content) > _MAIKE_MD_CHAR_CAP:
                truncated += "\n[...truncated]"
            context_blocks.append(
                f"## Project Context (MAIKE.md)\n\n{truncated}"
            )

    # Workspace file tree: top-level structure so the delegate can orient.
    snapshot = build_workspace_snapshot(session, "delegate")
    if snapshot:
        context_blocks.append(
            f"## Workspace Structure\n\n{snapshot}"
        )

    if context:
        context_blocks.append(
            f"## Context from parent agent\n\n{context}"
        )

    # Per-agent persistent memory: inject prior-session knowledge for
    # custom agents so they accumulate expertise across sessions.
    if agent_def is not None and workspace:
        try:
            from maike.constants import AGENT_MEMORY_PROJECT_SUBDIR
            from maike.memory.longterm import TypedLongTermMemory

            agent_mem_dir = workspace / AGENT_MEMORY_PROJECT_SUBDIR / agent_def.name
            if agent_mem_dir.is_dir():
                agent_memory = TypedLongTermMemory(
                    workspace, memory_dir=agent_mem_dir,
                )
                topics = agent_memory.load_topics_text(cap=4000)
                if topics:
                    context_blocks.append(
                        f"## Memory ({agent_def.name})\n\n{topics}"
                    )
                # Store for post-session extraction.
                ctx.metadata["agent_typed_memory"] = agent_memory
        except Exception:
            pass  # graceful degradation

    # Prepend initial_prompt from agent definition if present.
    effective_task = task
    if agent_def is not None and agent_def.initial_prompt:
        effective_task = f"{agent_def.initial_prompt}\n\n{task}"

    messages = build_messages(
        effective_task,
        [],
        context_blocks=context_blocks,
        role="delegate",
    )
    return ctx, messages


_FORK_DIRECTIVE_TEMPLATE = """\
<fork-context>
You have been forked from the parent agent to handle a specific subtask.
You have the parent's full context — all files read, reasoning, and discoveries.
Focus exclusively on:

{task}

Rules:
- Do NOT fork again (you are already a fork)
- Do NOT ask the user questions
- Complete the task and report results concisely
</fork-context>"""


def build_fork_context(
    parent_ctx,
    parent_conversation: list[dict],
    task: str,
    model: str = DEFAULT_MODEL,
):
    """Build context by copying the parent's state instead of starting fresh.

    The forked child inherits the parent's full conversation history plus
    a final user message with the fork directive.  This enables the child
    to leverage all the file contents, reasoning, and discoveries the
    parent has accumulated — without the cold-start penalty of a regular
    delegate.
    """
    import copy

    ctx = build_context(
        role="delegate",
        task=task,
        stage_name="delegate",
        tool_profile=parent_ctx.tool_profile,
        parent_id=parent_ctx.agent_id,
        model=model,
        input_artifacts=[],
        session_id=getattr(parent_ctx, "session_id", ""),
        metadata={
            **parent_ctx.metadata,
            "is_fork": True,
            "fork_parent_id": parent_ctx.agent_id,
        },
    )

    # Copy parent conversation and append fork directive.
    messages = copy.deepcopy(parent_conversation)
    messages.append({
        "role": "user",
        "content": _FORK_DIRECTIVE_TEMPLATE.format(task=task),
    })

    return ctx, messages
