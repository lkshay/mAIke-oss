"""Helper for `maike init` — on-demand LLM-assisted MAIKE.md generation."""

from __future__ import annotations

from pathlib import Path


_INIT_SYSTEM_PROMPT = """\
You generate concise MAIKE.md project context files for AI coding agents.
Be factual. Only describe what you can infer from the files provided.
Do not invent functionality, dependencies, or commands that aren't evident.
"""

_INIT_USER_TEMPLATE = """\
Generate a MAIKE.md for this project.

Language: {language}
Package manager: {package_manager}
Test command: {test_command}
Install command: {install_command}

Top-level source files:
{files_block}

{readme_section}

Produce a MAIKE.md with exactly these sections (omit a section if you have nothing factual to say):

## Overview
One paragraph describing what this project does.

## Commands
Key commands — test, run, install, lint. Use the values above as-is; do not guess.

## Architecture
2-5 bullet points naming key files/modules and their responsibility.

## Conventions
Coding style, patterns, or constraints the agent should follow.

Keep the whole file under 300 words. Be specific, not generic.
"""


def _collect_workspace_info(workspace: Path) -> dict:
    """Collect deterministic metadata from the workspace."""
    source_exts = {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".rb", ".java", ".kt", ".cs"}
    try:
        files = sorted(
            f.name for f in workspace.iterdir()
            if f.is_file() and f.suffix in source_exts
        )[:15]
    except OSError:
        files = []

    readme_content = ""
    for readme_name in ("README.md", "README.rst", "README.txt", "README"):
        readme_path = workspace / readme_name
        if readme_path.is_file():
            try:
                readme_content = readme_path.read_text(encoding="utf-8", errors="replace")[:1200]
            except OSError:
                pass
            break

    return {
        "files": files,
        "readme_content": readme_content,
    }


async def generate_maike_md(
    workspace: Path,
    *,
    provider: str,
    model: str,
    manifest,
) -> str:
    """Call the LLM to generate MAIKE.md content for the given workspace.

    Returns the raw generated text (does not write to disk).
    """
    from maike.cost.tracker import CostTracker
    from maike.gateway.llm_gateway import LLMGateway
    from maike.observability.tracer import Tracer

    info = _collect_workspace_info(workspace)

    language = getattr(manifest, "language", "unknown") or "unknown"
    package_manager = getattr(manifest, "package_manager", None) or "unknown"
    test_command = getattr(manifest, "test_command", None) or "unknown"
    install_command = (
        getattr(manifest, "install_all_command", None)
        or getattr(manifest, "install_command", None)
        or "unknown"
    )

    files_block = "\n".join(f"- {f}" for f in info["files"]) if info["files"] else "- (no source files detected)"
    readme_section = (
        f"README excerpt:\n{info['readme_content']}"
        if info["readme_content"]
        else "(no README found)"
    )

    user_prompt = _INIT_USER_TEMPLATE.format(
        language=language,
        package_manager=package_manager,
        test_command=test_command,
        install_command=install_command,
        files_block=files_block,
        readme_section=readme_section,
    )

    cost_tracker = CostTracker(budget_usd=1.0)
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker, tracer, provider_name=provider)
    try:
        result = await gateway.call(
            system=_INIT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[],
            model=model,
            max_tokens=1024,
        )
        return result.content or ""
    finally:
        await gateway.aclose()
