"""Shared agent command logic for REPL and TUI surfaces.

Both the REPL (``maike/repl.py``) and TUI (``maike/tui/app.py``) need
to list and create agents.  This module provides the shared formatting
and file-writing logic so both surfaces call the same code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from maike.agents.agent_resolver import AgentResolver
from maike.constants import AGENTS_PROJECT_SUBDIR, AGENTS_USER_DIR

# All tools available for agent configuration.
ALL_TOOL_NAMES: list[str] = [
    "Read", "Write", "Edit", "Bash", "Grep", "SemanticSearch",
    "WebSearch", "WebFetch", "AskUser", "Delegate", "Skill",
]


# ---------------------------------------------------------------------------
# Agent listing
# ---------------------------------------------------------------------------

def format_agent_list(resolver: AgentResolver) -> list[str]:
    """Format the full agent list as display lines.

    Returns a list of plain-text lines (no ANSI / Rich markup) suitable
    for both REPL (printed to stderr) and TUI (added as info messages).
    """
    agents = resolver.list_available()
    if not agents:
        return [
            "No custom agents found.",
            "  Create one with /create-agent <name>",
            f"  Or place a markdown file in {AGENTS_USER_DIR}",
        ]

    lines: list[str] = [f"Custom Agents ({len(agents)}):"]
    for a in agents:
        tools = ", ".join(a.allowed_tools) if a.allowed_tools else "all"
        lines.append(f"  {a.name}  ({a.source})  model={a.model_tier}  tools={tools}")
        lines.append(f"    {a.description}")
    return lines


# ---------------------------------------------------------------------------
# Wizard data model
# ---------------------------------------------------------------------------

@dataclass
class AgentWizardData:
    """All fields needed to create a custom agent definition.

    Used as the single source of truth for both REPL and TUI wizard
    surfaces.  Passed to :func:`create_agent_file_v2` for persistence.
    """

    name: str
    description: str = ""
    system_prompt: str = ""
    model_tier: str = "default"
    max_turns: int = 30
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    skills: list[str] = field(default_factory=list)
    initial_prompt: str | None = None
    background: bool = False
    critical_reminder: str | None = None
    scope: str = "project"


# ---------------------------------------------------------------------------
# Agent name sanitisation
# ---------------------------------------------------------------------------

def sanitize_agent_name(name: str) -> str:
    """Normalise a user-provided name to a valid agent slug.

    Lowercase, ASCII alphanumerics and hyphens only.
    """
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "my-agent"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def preview_agent_markdown(data: AgentWizardData) -> str:
    """Render the agent markdown file content without writing to disk.

    Used by both REPL (confirmation preview) and TUI (live preview pane).
    """
    lines: list[str] = ["---"]
    lines.append(f"name: {data.name}")
    lines.append(f"description: {data.description}")
    lines.append(f"model: {data.model_tier}")
    lines.append(f"maxTurns: {data.max_turns}")

    # Optional fields — only emit when non-default.
    if data.allowed_tools is not None:
        lines.append(f"tools: {', '.join(data.allowed_tools)}")
    if data.disallowed_tools:
        lines.append(f"disallowedTools: {', '.join(data.disallowed_tools)}")
    if data.skills:
        lines.append(f"skills: {', '.join(data.skills)}")
    if data.initial_prompt:
        lines.append(f"initialPrompt: {data.initial_prompt}")
    if data.background:
        lines.append("background: true")
    if data.critical_reminder:
        lines.append(f"critical_reminder: {data.critical_reminder}")

    lines.append("---")
    lines.append("")

    # Body = system prompt or placeholder.
    if data.system_prompt.strip():
        lines.append(data.system_prompt.strip())
    else:
        lines.append(data.description or data.name)
        lines.append("")
        lines.append("## Guidelines")
        lines.append("")
        lines.append("[Edit this file to define your agent's system prompt and behavior.]")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def create_agent_file_v2(
    data: AgentWizardData,
    *,
    workspace: Path | None = None,
) -> Path:
    """Write an agent markdown file from wizard data. Returns the path."""
    if data.scope == "project" and workspace is not None:
        directory = workspace / AGENTS_PROJECT_SUBDIR
    else:
        directory = AGENTS_USER_DIR

    directory.mkdir(parents=True, exist_ok=True)

    slug = sanitize_agent_name(data.name)
    path = directory / f"{slug}.md"
    path.write_text(preview_agent_markdown(data), encoding="utf-8")
    return path


def create_agent_file(
    name: str,
    description: str,
    *,
    model_tier: str = "default",
    tools: str = "all",
    max_turns: int = 30,
    scope: str = "project",
    workspace: Path | None = None,
) -> Path:
    """Backward-compatible wrapper around :func:`create_agent_file_v2`."""
    allowed = None
    if tools and tools.strip().lower() != "all":
        allowed = [t.strip() for t in tools.split(",") if t.strip()]

    data = AgentWizardData(
        name=sanitize_agent_name(name),
        description=description,
        model_tier=model_tier,
        max_turns=max_turns,
        allowed_tools=allowed,
        scope=scope,
    )
    return create_agent_file_v2(data, workspace=workspace)


# ---------------------------------------------------------------------------
# LLM-assisted prompt generation
# ---------------------------------------------------------------------------

_PROMPT_GEN_SYSTEM = """\
You generate system prompts for AI coding sub-agents.
Given the agent's name, description, and available tools, produce a
concise system prompt that:
- States the agent's role and expertise clearly
- Lists the tools available and how to use them effectively
- Defines a clear numbered workflow (3-5 steps)
- Sets rules and constraints
- Specifies the expected output format

Keep it under 400 words. Be specific and actionable, not generic.
Do NOT wrap the output in markdown code fences — return raw text only.
"""

_PROMPT_GEN_USER = """\
Agent name: {name}
Description: {description}

Available tools: {tools}

Generate a system prompt for this agent.
"""


async def generate_agent_prompt(
    description: str,
    name: str,
    tools: list[str] | None,
    *,
    provider: str,
    model: str,
) -> str:
    """Call the cheap LLM to generate a system prompt for the agent.

    Follows the same pattern as ``init_helper.generate_maike_md()``.
    Returns the generated prompt text, or empty string on failure.
    """
    from maike.cost.tracker import CostTracker
    from maike.gateway.llm_gateway import LLMGateway
    from maike.observability.tracer import Tracer

    tools_str = ", ".join(tools) if tools else ", ".join(ALL_TOOL_NAMES)
    user_prompt = _PROMPT_GEN_USER.format(
        name=name,
        description=description,
        tools=tools_str,
    )

    cost_tracker = CostTracker(budget_usd=0.50)
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker, tracer, provider_name=provider)
    try:
        result = await gateway.call(
            system=_PROMPT_GEN_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[],
            model=model,
            max_tokens=1024,
        )
        return (result.content or "").strip()
    except Exception:
        return ""
    finally:
        await gateway.aclose()
