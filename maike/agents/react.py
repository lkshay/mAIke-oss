"""React agent context builder.

The react agent runs as a single flat ReAct loop with phased tool injection
and lazy context loading.  Initial context is deliberately minimal — the
agent uses exploration tools to load what it needs on demand.
"""

from __future__ import annotations

from pathlib import Path

from maike.agents.helpers import (
    build_context,
    build_environment_context_blocks,
    build_environment_metadata,
    build_messages,
)
from maike.constants import DEFAULT_MODEL
from maike.memory.learning import build_learning_context_blocks

# Source extensions counted for delegation opportunity heuristic.
_SOURCE_EXTENSIONS = frozenset({".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb"})

# Keywords that indicate a complex task (triggers planning reminder).
_COMPLEX_TASK_KEYWORDS = frozenset({
    "fix", "debug", "refactor", "add feature", "multiple", "across",
    "modules", "update all", "migrate", "breaking change",
})


def _should_inject_planning_reminder(task: str) -> bool:
    """Heuristic: is this task complex enough to warrant an explicit plan?"""
    if len(task) > 200:
        return True
    task_lower = task.lower()
    return any(kw in task_lower for kw in _COMPLEX_TASK_KEYWORDS)


# Keywords that signal a design/planning request (not an implementation request).
_DESIGN_KEYWORDS = frozenset({
    "design", "plan", "architect", "propose", "outline", "spec",
    "specification", "blueprint", "strategy", "approach",
})
# If ANY of these also appear, it's a build request even if design keywords match.
_BUILD_KEYWORDS = frozenset({
    "implement", "build", "create", "write code", "code it",
    "make it", "set up", "start coding", "develop",
})


def _is_design_request(task: str) -> bool:
    """Return True if the task is asking for a design/plan, not implementation."""
    task_lower = task.lower()
    has_design = any(kw in task_lower for kw in _DESIGN_KEYWORDS)
    has_build = any(kw in task_lower for kw in _BUILD_KEYWORDS)
    return has_design and not has_build


def _count_source_files(workspace: Path) -> int:
    """Count source files in the workspace (non-hidden, non-venv)."""
    skip = {".git", ".maike", "__pycache__", "node_modules", ".venv", "venv"}
    count = 0
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.relative_to(workspace).parts):
            continue
        if path.suffix in _SOURCE_EXTENSIONS:
            count += 1
    return count


def _strip_thought_signatures(messages: list) -> list:
    """Strip thought signatures from messages before persisting to thread history.

    Thought signatures are tied to a specific generation context and cannot
    be replayed across sessions.  This function removes them so that thread
    history can be safely loaded later.  Function-call blocks that had
    signatures are kept intact (the signature field is just removed) — the
    thread replay code handles unsigned blocks gracefully.
    """
    if not messages or not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and ("thought_signature" in block or "thoughtSignature" in block):
                block = {k: v for k, v in block.items() if k not in ("thought_signature", "thoughtSignature")}
            new_blocks.append(block)
        cleaned.append({**msg, "content": new_blocks})
    return cleaned


def _sanitize_thread_history_for_replay(messages: list) -> list:
    """Prepare thread history for replay into a new session.

    Cleans up two kinds of problems:

    1. **Unsigned tool_use blocks**: From sessions before thinking was enabled.
       Gemini requires signatures on all function-call parts. Drop these and
       their orphaned tool_result messages.

    2. **Legacy ``[Called ...]`` text markers**: From an earlier sanitization
       approach that converted tool_use blocks to text like ``[Called Read]``.
       These cause models to mimic the pattern instead of using real tool calls.
       Strip them from text content.
    """
    import re

    if not messages or not isinstance(messages, list):
        return messages

    # Regex to match legacy [Called ToolName] markers.
    _CALLED_RE = re.compile(r"\[Called \w+\]")

    # First pass: collect IDs of unsigned tool_use blocks.
    dropped_tool_ids: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and not block.get("thought_signature")):
                tool_id = block.get("id", "")
                if tool_id:
                    dropped_tool_ids.add(tool_id)

    # Second pass: clean messages.
    cleaned = []
    for msg in messages:
        content = msg.get("content")

        # String content: strip legacy [Called ...] markers.
        if isinstance(content, str):
            content = _CALLED_RE.sub("", content).strip()
            if content:
                cleaned.append({**msg, "content": content})
            continue

        # Non-list content: pass through.
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        # List content: drop unsigned tool_use + orphaned tool_result,
        # and strip [Called ...] from text blocks.
        new_blocks = []
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            btype = block.get("type", "")
            if btype == "tool_use" and not block.get("thought_signature"):
                continue
            if btype == "tool_result" and block.get("tool_use_id") in dropped_tool_ids:
                continue
            # Clean [Called ...] from text blocks too.
            if btype == "text" and block.get("text"):
                cleaned_text = _CALLED_RE.sub("", block["text"]).strip()
                if cleaned_text:
                    new_blocks.append({**block, "text": cleaned_text})
                continue
            new_blocks.append(block)

        if new_blocks:
            cleaned.append({**msg, "content": new_blocks})
        elif msg.get("role") == "assistant":
            cleaned.append({"role": "assistant", "content": ""})

    return cleaned
    return cleaned


def _has_prior_work(workspace) -> bool:
    """Check if the workspace contains source files from a prior session."""
    ws = Path(str(workspace))
    if not ws.is_dir():
        return False
    ignored = {".git", ".maike", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
    source_suffixes = {".py", ".js", ".ts", ".go", ".rs"}
    for entry in ws.iterdir():
        if entry.name in ignored or entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix in source_suffixes:
            return True
    return False

def read_maike_md(workspace) -> str | None:
    """Read MAIKE.md from the workspace root, if it exists."""
    ws = Path(str(workspace))
    maike_path = ws / "MAIKE.md"
    if maike_path.is_file():
        try:
            return maike_path.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


# ---------------------------------------------------------------------------
# Auto-memory topic loading
# ---------------------------------------------------------------------------



# Cap learning context at ~2000 tokens (estimated at ~4 chars/token).
_LEARNING_CHAR_CAP = 8000


def _inject_learning_context(
    session,
    task: str,
    context_blocks: list[str],
) -> None:
    """Query long-term memory for past session insights and append to context.

    Gracefully skips when the session has no learner or the learner returns
    nothing.  Output is truncated to ``_LEARNING_CHAR_CAP`` characters to
    keep token usage bounded.
    """
    learner = getattr(session, "learner", None)
    if learner is None:
        return

    manifest = getattr(session, "environment_manifest", None)
    lang_str = getattr(manifest, "language", "") if manifest else ""

    blocks = build_learning_context_blocks(learner, task, language=lang_str, role="react_agent")
    if not blocks:
        return

    combined = "\n\n".join(blocks)
    if len(combined) > _LEARNING_CHAR_CAP:
        combined = combined[:_LEARNING_CHAR_CAP].rsplit("\n", 1)[0] + "\n\n(truncated)"
    context_blocks.append(combined)


# Max chars of thread history to replay (~30K tokens).  Keeps the most
# recent messages and replaces older ones with a summary note.
_THREAD_HISTORY_CHAR_CAP = 120_000


def _cap_thread_history(messages: list) -> list:
    """Cap thread history to fit within a token budget.

    Keeps the most recent messages that fit within ``_THREAD_HISTORY_CHAR_CAP``.
    If the full history exceeds the cap, older messages are dropped and a
    summary note is prepended.
    """
    if not messages:
        return messages

    # Estimate total size
    total_chars = sum(
        len(m.get("content", "")) if isinstance(m.get("content"), str)
        else sum(len(str(b)) for b in m.get("content", []))
        for m in messages
    )
    if total_chars <= _THREAD_HISTORY_CHAR_CAP:
        return messages

    # Keep the most recent messages that fit within the cap.
    kept: list[dict] = []
    running = 0
    for msg in reversed(messages):
        content = msg.get("content", "")
        msg_chars = len(content) if isinstance(content, str) else sum(len(str(b)) for b in content)
        if running + msg_chars > _THREAD_HISTORY_CHAR_CAP and kept:
            break
        kept.append(msg)
        running += msg_chars

    kept.reverse()
    dropped = len(messages) - len(kept)

    if dropped > 0:
        summary = {
            "role": "user",
            "content": (
                f"[Thread history truncated: {dropped} older messages omitted "
                f"to fit context window. The {len(kept)} most recent messages follow.]"
            ),
        }
        return [summary] + kept
    return kept


def _strip_skill_tags(messages: list) -> list:
    """Remove session-scoped content from messages before thread persistence.

    Strips:
    - ``<maike-skill>`` blocks (re-selected fresh each session)
    - ``<maike-nudge>`` blocks (convergence/failure nudges create toxic context on replay)
    - ``<maike-status>`` blocks (file mutation notes are stale across sessions)
    - ``<maike-guidance>`` blocks (re-generated fresh from workspace state)
    - Old-format ``## Auto-loaded Skill:`` messages

    Keeps: task content, assistant responses, tool calls, tool results.
    """
    from maike.context.tags import strip_tag

    # Tags to strip — these are all session-scoped and should not be replayed.
    _STRIP_TAGS = ("maike-skill", "maike-nudge", "maike-status", "maike-guidance")

    if not messages or not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            # Strip XML-tagged session-scoped content
            for tag in _STRIP_TAGS:
                if f"<{tag}" in content:
                    content = strip_tag(content, tag)
            # Strip old-format skill messages
            if content.startswith("## Auto-loaded Skill:"):
                continue  # drop entire message
            if content.strip():
                cleaned.append({**msg, "content": content})
        else:
            cleaned.append(msg)
    return cleaned


def _sanitize_thread_history(messages: list) -> list:
    """Remove embedded system prompt from old thread histories.

    Early versions of build_react_context inserted REACT_SYSTEM_PROMPT into
    the first user message.  When that conversation was saved to a thread,
    the system prompt became permanent history.  This function strips it so
    old threads don't carry stale instructions.
    """
    if not messages:
        return messages
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "user":
        return messages
    content = first.get("content", "")
    if not isinstance(content, str):
        return messages
    # Detect old embedded system prompts by known signature strings.
    _SIGNATURES = (
        "== STEP 1: CLASSIFY THE TASK ==",
        "You are a software engineering agent.",
    )
    if not any(sig in content for sig in _SIGNATURES):
        return messages
    # Strip everything before the actual task/context.
    task_marker = "## Task\n"
    idx = content.find(task_marker)
    if idx >= 0:
        cleaned_content = content[idx + len(task_marker):]
        first = {**first, "content": cleaned_content}
        return [first] + messages[1:]
    return messages


async def build_react_context(
    task: str,
    session,
    parent_id: str | None = None,
    model: str = DEFAULT_MODEL,
    thread: dict | None = None,
    adaptive_model: bool = True,
):
    """Build minimal initial context for a react agent.

    The system prompt lives in ``react-agent.md`` and is loaded by AgentCore
    into the ``system`` parameter of the LLM call.  This builder only
    constructs the *user* messages:
      - environment summary (language, package manager, project type)
      - MAIKE.md content (if it exists) for project continuity
      - thread conversation history (if resuming a thread)
      - task description
    """
    # No artifacts loaded — react agent starts fresh and explores.
    ctx = build_context(
        role="react_agent",
        task=task,
        stage_name="react",
        tool_profile="react",
        parent_id=parent_id,
        model=model,
        input_artifacts=[],
        session_id=session.id,
        metadata={
            **build_environment_metadata(session),
            "pipeline": "react",
            "adaptive_model": adaptive_model,
        },
    )

    # Fresh context every session.  Thread continuity comes from compact
    # session summaries (not raw history replay).  This avoids the
    # problems of stripped tool_use blocks, recursive pruned summaries,
    # and duplicate environment blocks that made raw replay useless.
    #
    # Context blocks use ContextBlock instances for XML tagging.
    # The ordering is: environment → workspace snapshot → project → memory →
    # thread summaries → skills → guidance → task.
    from maike.context.tags import ContextBlock

    context_blocks: list[ContextBlock | str] = []

    # Environment context (language, package manager, toolchain).
    raw_env_blocks = build_environment_context_blocks(session)
    for env_block in raw_env_blocks:
        context_blocks.append(ContextBlock(
            tag="maike-environment", content=env_block, priority="high",
        ))

    # Workspace snapshot — file tree + git changes so the agent knows what
    # exists without running `ls`.  Pass *task* so the empty-workspace branch
    # can surface a clone-first hint when the task references a remote git
    # repository URL (avoids the misorientation that produces ls-loops).
    from maike.agents.helpers import build_workspace_snapshot
    workspace = getattr(session, "workspace", None)
    if workspace:
        snapshot = build_workspace_snapshot(session, "react_agent", task=task)
        if snapshot:
            context_blocks.append(ContextBlock(
                tag="maike-intelligence", content=snapshot, priority="high",
            ))

    # Pre-computed code intelligence: symbols, related files, repo map.
    from maike.agents.helpers import build_hot_context_block
    hot_context = build_hot_context_block(session, task, "react_agent")
    if hot_context:
        context_blocks.append(ContextBlock(
            tag="maike-intelligence", content=hot_context, priority="normal",
        ))

    # Task constraints — markdown "Task rules" block extracted by the
    # constraint extractor (plus MAIKE.md Protected Files, if any).
    # Injected verbatim at priority="high" so it survives context pruning.
    # The react agent reads and honors this natural-language guidance;
    # there is no structured-rule enforcement layer behind it.
    _tc_md = getattr(session, "task_constraints_markdown", None)
    if isinstance(_tc_md, str) and _tc_md.strip() and "(no explicit constraints)" not in _tc_md:
        context_blocks.append(ContextBlock(
            tag="maike-constraints",
            content=_tc_md,
            priority="high",
            attrs={"source": "task-extractor"},
        ))

    # MAIKE.md — project context from prior sessions.
    # Injected as a <system-reminder> in the first user message (not as a
    # ContextBlock) for prompt-cache-stable project context. The raw
    # content is forwarded to build_messages().
    maike_content = read_maike_md(workspace) if workspace else None

    # Persistent auto-memory: durable project knowledge from prior sessions.
    # Loaded from {workspace}/.maike/memories/MEMORY.md — survives across
    # sessions and prevents blind re-exploration of the codebase.
    typed_memory = getattr(session, "typed_memory", None)
    if typed_memory is not None:
        try:
            memory_index = typed_memory.read_index()
            if memory_index and memory_index.strip():
                # Load full topic files for richer context (cap at 8KB).
                topic_content = typed_memory.load_topics_text(cap=8000)
                auto_mem_content = memory_index
                if topic_content:
                    auto_mem_content += "\n\n" + topic_content
                context_blocks.append(ContextBlock(
                    tag="maike-memory",
                    content=auto_mem_content,
                    priority="high",
                    attrs={"source": "auto-memory"},
                ))
        except Exception:
            pass  # graceful degradation

    # Custom agent catalog: tell the main agent which specialists exist
    # so it can delegate to them proactively.
    _agent_resolver = getattr(session, "agent_resolver", None)
    if _agent_resolver is not None:
        try:
            catalog = _agent_resolver.build_catalog(cap=2000)
            if catalog:
                context_blocks.append(ContextBlock(
                    tag="maike-guidance",
                    content=(
                        "## Available Custom Agents\n"
                        "Use Delegate(agent=\"name\") to delegate to these specialists:\n\n"
                        + catalog
                    ),
                    priority="low",
                ))
        except Exception:
            pass  # graceful degradation

    # Team catalog: tell the main agent which teams it can invoke.
    _team_resolver = getattr(session, "team_resolver", None)
    if _team_resolver is not None:
        try:
            team_catalog = _team_resolver.build_catalog(cap=1500)
            if team_catalog:
                context_blocks.append(ContextBlock(
                    tag="maike-guidance",
                    content=(
                        "## Available Teams\n"
                        "Use Team(name=\"team-name\", task=\"...\") to invoke a team "
                        "(spawns all members in parallel, synthesizes results):\n\n"
                        + team_catalog
                    ),
                    priority="low",
                ))
        except Exception:
            pass  # graceful degradation

    # Advisor availability: when --advisor is enabled, tell the executor
    # the Advisor tool exists and WHEN to reach for it. This guidance is
    # directive (not just advisory) because cheap-model agents tend to skip
    # the tool unless the system prompt explicitly says to use it for
    # planning-heavy tasks.
    _advisor_session = getattr(session, "advisor_session", None)
    if _advisor_session is not None and getattr(_advisor_session, "enabled", False):
        try:
            cfg = _advisor_session.config
            context_blocks.append(ContextBlock(
                tag="maike-guidance",
                content=(
                    "## Advisor (frontier model) available\n"
                    f"A stronger model ({cfg.provider}/{cfg.model}) is available "
                    "as an Advisor for strategic decisions.\n\n"
                    "**Call `Advisor(question=\"...\", urgency=\"normal\"|\"stuck\")` when:**\n"
                    "- You are about to **add a new module, component, or feature** "
                    "(ask: \"Is my plan complete? Am I missing integration steps?\")\n"
                    "- You are about to **refactor across multiple files** "
                    "(ask: \"What hidden coupling should I watch for?\")\n"
                    "- You are **stuck** — the same test/command keeps failing "
                    "after 2+ fix attempts (urgency=\"stuck\")\n"
                    "- You are **uncertain which pattern to follow** from multiple "
                    "similar existing files\n\n"
                    "The advisor returns 1–3 sentences of strategic advice. It does "
                    "NOT execute code. The advisor ALSO fires automatically on "
                    "repeated failures, after exploration, and before your first "
                    "code change on planning-heavy tasks — so for simple bug fixes "
                    "you can ignore it entirely."
                ),
                priority="low",
            ))
        except Exception:
            pass  # graceful degradation

    # Long-term memory: past session learnings relevant to this task.
    # NOTE: session.long_term_memory is not set on OrchestratorSession;
    # learnings are injected via _inject_learning_context below instead.
    _learning_blocks: list[str] = []
    _inject_learning_context(session, task, _learning_blocks)
    for lb in _learning_blocks:
        context_blocks.append(ContextBlock(
            tag="maike-memory", content=lb, priority="normal",
        ))

    # Thread summaries — compact per-session summaries from prior work.
    # Injected at full quality; compressed lazily by ContextBudgetManager
    # only when context pressure demands it.
    effective_thread = thread or getattr(session, "thread", None)
    if effective_thread:
        summaries = effective_thread.get("summaries", [])
        if summaries:
            thread_name = effective_thread.get("name", "unknown")
            summary_content = (
                f"## Prior work on thread '{thread_name}' ({len(summaries)} sessions)\n\n"
                + "\n\n---\n\n".join(summaries)
            )
            # If the last session failed or was cancelled, add an explicit
            # note so the agent knows to resume/recover rather than starting
            # fresh when the user says "continue".
            last_summary = summaries[-1]
            if "Outcome: failure" in last_summary or "Outcome: cancelled" in last_summary:
                summary_content += (
                    "\n\n**NOTE: The previous session did not complete successfully. "
                    "If the user asks to 'continue', resume from where the last session "
                    "left off and address the issue that caused the failure.**"
                )
            context_blocks.append(ContextBlock(
                tag="maike-memory", content=summary_content, priority="high",
            ))
        elif effective_thread.get("conversation_history"):
            # Legacy thread: has raw history but no summaries.
            # Generate a one-time migration summary.
            try:
                from maike.memory.summary import SessionSummaryBuilder
                raw_hist = effective_thread["conversation_history"]
                if isinstance(raw_hist, list) and raw_hist:
                    _legacy_sb = SessionSummaryBuilder()
                    legacy_summary = _legacy_sb.build_summary(
                        messages=raw_hist,
                        task="(prior work — legacy thread)",
                        outcome="unknown",
                        session_id="legacy",
                        timestamp=effective_thread.get("created_at", "unknown"),
                    )
                    context_blocks.append(ContextBlock(
                        tag="maike-memory",
                        content=f"## Prior work (legacy thread)\n\n{legacy_summary}",
                        priority="high",
                    ))
            except Exception:
                pass  # graceful degradation

    # Skills — selected based on task text, injected inline right before
    # the task for maximum attention proximity.  This replaces the
    # separate-message injection that was in core.py.
    from maike.tools.context import peek_current_skill_loader
    _skill_loader = peek_current_skill_loader()
    injected_skill_names: list[str] = []
    if _skill_loader is not None:
        try:
            _all_skills = _skill_loader.load_all()
            _task_skills = _skill_loader.select_for_task(task, _all_skills)
            for skill in _task_skills:
                context_blocks.append(ContextBlock(
                    tag="maike-skill",
                    content=skill.content,
                    priority="critical",
                    attrs={"name": skill.name},
                ))
                injected_skill_names.append(skill.name)
        except Exception:
            pass  # graceful degradation

    # Store injected skill names in context metadata so core.py can
    # seed its _injected_skills set without re-querying.
    ctx = ctx.model_copy(update={
        "metadata": {**ctx.metadata, "injected_skills": injected_skill_names},
    })

    # Design-request detection: if the task is asking for a design/plan
    # (not implementation), tell the agent to respond with text, not code.
    # Critical priority so it survives compression.
    if _is_design_request(task):
        context_blocks.append(ContextBlock(
            tag="maike-guidance",
            content=(
                "## Design Request Detected\n"
                "This task is asking for a design, plan, or architecture proposal. "
                "Respond with a detailed design document using text only. "
                "Do NOT create files, run commands, or use tools unless the user "
                "explicitly asks you to implement or build something. "
                "Present your design first — the user will tell you when to start coding."
            ),
            priority="critical",
        ))

    # Guidance — planning reminders, delegation hints, prior work notes.
    # These are low-priority and dropped first during compression.
    prior_work = workspace and _has_prior_work(workspace)
    if prior_work:
        context_blocks.append(ContextBlock(
            tag="maike-guidance",
            content=(
                "NOTE: This workspace contains files from a prior session. "
                "Review what exists before making changes. Use Grep and "
                "Read to understand the current state before writing new code."
            ),
            priority="low",
        ))

    # Planning-reminder block removed: it fired on any task containing
    # "fix"/"debug"/"refactor" and pushed the model to always open with
    # a 3-step "1. State files 2. Describe changes 3. Proceed" preamble,
    # which then got re-injected by the cadence plan reminder. The core
    # prompt's "Uncertainty about the goal" bullet handles when a plan
    # is actually useful.

    if workspace:
        file_count = _count_source_files(workspace)
        if file_count >= 5 and _should_inject_planning_reminder(task):
            context_blocks.append(ContextBlock(
                tag="maike-guidance",
                content=(
                    f"## Delegation Opportunity\n"
                    f"This workspace has {file_count} source files. For independent sub-tasks,\n"
                    f"use Delegate(task='...') to work in parallel, then "
                    f"Delegate(action='wait', handle='...') to collect results."
                ),
                priority="low",
            ))

    messages = build_messages(
        task,
        [],
        context_blocks=context_blocks,
        role="react_agent",
        system_reminder=maike_content,
    )
    return ctx, messages
