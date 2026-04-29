"""Skill tool — LLM-driven skill loading via explicit tool call."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from maike.agents.skill import SkillLoader, expand_skill_content
from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.tools.registry import ToolRegistry


def register_skill_tool(
    registry: ToolRegistry,
    skill_loader: SkillLoader,
    delegate_handler: Callable[..., Awaitable[ToolResult]] | None = None,
) -> None:
    """Register the ``Skill`` tool.

    The tool description includes a dynamically-generated catalog of
    available skills so the LLM knows what it can load.

    If *delegate_handler* is provided, fork-mode skills will run in
    an isolated subagent via the delegation system.
    """
    all_skills = skill_loader.load_all()
    catalog = skill_loader.build_catalog(all_skills)

    # Pre-compute the set of loadable skill names (excludes model-disabled).
    visible_skills = [s for s in all_skills if not s.disable_model_invocation]
    visible_names = [s.name for s in visible_skills]

    async def execute_skill(
        name: str | None = None,
        args: str | None = None,
    ) -> ToolResult:
        if not name:
            return ToolResult(
                tool_name="Skill",
                success=False,
                output=(
                    "name is required. Available skills:\n"
                    + "\n".join(f"- {n}" for n in visible_names)
                ),
            )

        skill = skill_loader.load_by_name(name, all_skills)
        if skill is None or skill.disable_model_invocation:
            return ToolResult(
                tool_name="Skill",
                success=False,
                output=(
                    f"Unknown skill: {name!r}. Available skills:\n"
                    + "\n".join(f"- {n}" for n in visible_names)
                ),
            )

        # Expand variables in skill content.
        from maike.tools.context import peek_current_agent_context
        ctx = peek_current_agent_context()
        session_id = ctx.metadata.get("session_id", "") if ctx else ""
        workspace = ctx.metadata.get("workspace", "") if ctx else ""
        expanded = expand_skill_content(
            skill,
            args=args or "",
            session_id=session_id,
            workspace=workspace,
        )

        # Fork mode: run in isolated subagent.
        if skill.context == "fork" and delegate_handler is not None:
            return await delegate_handler(
                task=expanded,
                context=f"Running skill: {skill.name}",
                model_tier=skill.model_override or "default",
                agent_type=skill.agent_type or "implement",
            )

        # Inline mode (default): return expanded content.
        parts: list[str] = [expanded]

        supporting = SkillLoader.load_supporting_content(skill)
        if supporting:
            parts.append(f"## Reference Material\n\n{supporting}")

        if skill.skill_dir is not None:
            refs_dir = skill.skill_dir / "references"
            if refs_dir.is_dir():
                ref_files = sorted(f.name for f in refs_dir.glob("*.md"))
                if ref_files:
                    files_list = ", ".join(
                        f"`references/{f}`" for f in ref_files
                    )
                    parts.append(
                        f"*For deeper detail, read these files in "
                        f"`{skill.skill_dir}`: {files_list}*"
                    )

        return ToolResult(
            tool_name="Skill",
            success=True,
            output="\n\n".join(parts),
        )

    registry.register(
        schema=ToolSchema(
            name="Skill",
            description=(
                "Load detailed guidance for a coding skill. Call this when "
                "you recognise a task that matches a skill's domain and you "
                "want the full reference. Only load a skill when you need "
                "its guidance — don't load skills speculatively.\n\n"
                "Available skills:\n" + catalog
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name from the catalog above.",
                    },
                    "args": {
                        "type": "string",
                        "description": (
                            "Optional arguments for the skill. "
                            "Check the skill's argument hint in the catalog."
                        ),
                    },
                },
                "required": ["name"],
            },
        ),
        fn=execute_skill,
        risk_level=RiskLevel.READ,
    )
