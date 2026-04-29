"""Progressive loading tool — lets agents fetch full content of summarized artifacts."""

from __future__ import annotations

import re

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.memory.session import SessionStore
from maike.tools.context import current_agent_context, peek_current_telemetry
from maike.tools.registry import ToolRegistry


def _extract_section(content: str, section: str) -> str | None:
    """Extract a markdown section by heading text.

    Returns the content under the first heading that matches *section*
    (case-insensitive), up to the next heading of equal or higher level.
    Returns ``None`` if no match is found.
    """
    lines = content.split("\n")
    collecting = False
    collected: list[str] = []
    match_level = 0

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            if not collecting and title.lower() == section.lower():
                collecting = True
                match_level = level
                collected.append(line)
                continue
            if collecting and level <= match_level:
                break
        if collecting:
            collected.append(line)

    return "\n".join(collected) if collected else None


def register_artifact_detail_tool(
    registry: ToolRegistry,
    session: SessionStore,
) -> None:
    """Register the ``fetch_artifact_detail`` tool.

    Uses the running agent's ``session_id`` (from agent context) and
    the *session* store to look up artifacts by logical name.
    """

    async def fetch_artifact_detail(
        artifact_name: str,
        section: str | None = None,
    ) -> ToolResult:
        ctx = current_agent_context()
        session_id = ctx.metadata["session_id"]
        artifact = await session.get_artifact_by_name(session_id, artifact_name)
        if artifact is None:
            return ToolResult(
                tool_name="fetch_artifact_detail",
                success=False,
                output="",
                raw_output="",
                error=f"Artifact '{artifact_name}' not found.",
            )

        content = artifact.content
        if section:
            extracted = _extract_section(content, section)
            if extracted is None:
                return ToolResult(
                    tool_name="fetch_artifact_detail",
                    success=False,
                    output="",
                    raw_output="",
                    error=f"Section '{section}' not found in artifact '{artifact_name}'.",
                )
            content = extracted

        telemetry = peek_current_telemetry()
        if telemetry is not None:
            telemetry.record_fetch()

        return ToolResult(
            tool_name="fetch_artifact_detail",
            success=True,
            output=content,
            raw_output=content,
            metadata={"artifact_name": artifact_name, "section": section},
        )

    registry.register(
        schema=ToolSchema(
            name="fetch_artifact_detail",
            description=(
                "Load the full content of a pipeline artifact that was summarized "
                "in your context. Use this when you need details that were omitted "
                "from a [SUMMARIZED] artifact. Optionally pass a section heading "
                "to retrieve only that part."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "artifact_name": {
                        "type": "string",
                        "description": (
                            "Logical name of the artifact, e.g. 'spec.md', 'plan.md', "
                            "'architecture.md'."
                        ),
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "Optional markdown heading to extract. Returns only that "
                            "section and its content."
                        ),
                    },
                },
                "required": ["artifact_name"],
            },
        ),
        fn=fetch_artifact_detail,
        risk_level=RiskLevel.READ,
        profiles=[
            "coding",
            "partition_coding",
            "testing",
            "debugging",
            "review",
            "acceptance",
        ],
    )
