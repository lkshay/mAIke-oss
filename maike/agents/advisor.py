"""Advisor pattern — frontier-model strategic advice for a local/cheap executor.

The executor runs the main agent loop on a cheap or local model. At key
decision points (explicit tool call, after exploration, when stuck) it asks
a frontier model (the advisor) for short strategic advice. The advisor reads
a compressed snapshot of the conversation and returns 1–3 sentences of
actionable guidance, which the executor injects into its own context.

Key invariant: the advisor gateway has its own provider adapter and silent
cost tracking. It never touches the executor's conversation state — advice
is delivered as a new user message tagged ``<maike-advisor>``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from maike.constants import (
    ADVISOR_COOLDOWN_ITERATIONS,
    ADVISOR_EXPLORATION_THRESHOLD,
    ADVISOR_MAX_CALLS_PER_SESSION,
    ADVISOR_MAX_OUTPUT_TOKENS,
    ADVISOR_RECENT_MESSAGES_CHARS,
    ADVISOR_TEMPERATURE,
    ADVISOR_TRANSCRIPT_MAX_CHARS,
)

if TYPE_CHECKING:
    from maike.atoms.agent import AgentContext
    from maike.gateway.llm_gateway import LLMGateway


logger = logging.getLogger(__name__)


class AdvisorUrgency(str, Enum):
    """Why the advisor is being called."""

    NORMAL = "normal"          # explicit tool call, low priority
    STUCK = "stuck"            # repeated failure / spinning detected
    PLAN_CHECK = "plan-check"  # after exploration, before implementation
    DONE_CHECK = "done-check"  # before completion (reserved for future use)


class AdvisorTrigger(str, Enum):
    """Which subsystem invoked the advisor."""

    TOOL = "tool"                          # Advisor(...) tool call
    ON_STUCK = "on_stuck"                  # auto-trigger: failure spike / spinning
    AFTER_EXPLORATION = "after_exploration"  # auto-trigger: N reads with no writes


@dataclass
class AdvisorVerdict:
    """The advisor's response. ``throttled=True`` means we skipped the call."""

    advice: str
    urgency: AdvisorUrgency
    trigger: AdvisorTrigger
    cost_usd: float = 0.0
    tokens_used: int = 0
    throttled: bool = False
    throttle_reason: str = ""

    @classmethod
    def throttled_verdict(
        cls,
        urgency: AdvisorUrgency,
        trigger: AdvisorTrigger,
        reason: str,
    ) -> AdvisorVerdict:
        return cls(
            advice="",
            urgency=urgency,
            trigger=trigger,
            throttled=True,
            throttle_reason=reason,
        )


@dataclass
class AdvisorConfig:
    """Resolved advisor configuration for a session."""

    enabled: bool = False
    provider: str = ""              # e.g. "gemini" — defaults to executor's provider
    model: str = ""                 # e.g. "gemini-2.5-pro" — defaults to strong tier
    budget_usd: float = 0.0         # absolute USD cap (fraction × session budget)
    max_calls: int = ADVISOR_MAX_CALLS_PER_SESSION
    cooldown_iterations: int = ADVISOR_COOLDOWN_ITERATIONS


class AdvisorSession:
    """Per-agent-session state for advisor calls.

    Tracks budget, call count, throttle state, and previous verdicts.
    Each call does budget/cooldown/max-calls checks, builds a compressed
    transcript, calls the silent advisor gateway, and records the verdict.
    """

    def __init__(
        self,
        gateway: LLMGateway | None,
        config: AdvisorConfig,
    ) -> None:
        self.gateway = gateway
        self.config = config
        self.call_count: int = 0
        self.cost_spent_usd: float = 0.0
        self.last_call_iteration: int = -999
        self.triggers_fired: set[str] = set()
        self.previous_verdicts: list[AdvisorVerdict] = []
        # Monotonic counter of tool-call failures observed this session.
        # Incremented by AgentCore whenever failure_tracker.record() sees a
        # failure; never reset. Used by the on_stuck trigger instead of the
        # clearable RepeatedFailureTracker._failure_hashes list — which gets
        # emptied after the tracker's own nudge fires, creating a blind spot.
        self.failures_seen: int = 0

    def record_failure(self) -> None:
        """Called by AgentCore when a tool-call failure is observed.

        Monotonic counter — never reset. Safe to call from anywhere; a no-op
        when the advisor is disabled but harmless to track anyway.
        """
        self.failures_seen += 1

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.gateway is not None)

    @property
    def budget_remaining_usd(self) -> float:
        """Remaining advisor budget in USD. 0.0 for budget_usd means unlimited."""
        if self.config.budget_usd <= 0:
            # Convention: 0 = unlimited (matches DEFAULT_RUN_BUDGET_USD).
            return float("inf")
        return max(0.0, self.config.budget_usd - self.cost_spent_usd)

    @property
    def _has_budget(self) -> bool:
        """True if we can afford another advisor call."""
        if self.config.budget_usd <= 0:
            return True  # unlimited
        return self.cost_spent_usd < self.config.budget_usd

    def _check_throttle(
        self,
        urgency: AdvisorUrgency,
        trigger: AdvisorTrigger,
        iteration_count: int,
    ) -> AdvisorVerdict | None:
        """Return a throttled verdict if we should skip, else None."""
        if not self.enabled:
            return AdvisorVerdict.throttled_verdict(urgency, trigger, "disabled")
        if self.call_count >= self.config.max_calls:
            return AdvisorVerdict.throttled_verdict(urgency, trigger, "max_calls_reached")
        if not self._has_budget:
            return AdvisorVerdict.throttled_verdict(urgency, trigger, "budget_exhausted")
        iters_since_last = iteration_count - self.last_call_iteration
        if iters_since_last < self.config.cooldown_iterations:
            return AdvisorVerdict.throttled_verdict(
                urgency, trigger, f"cooldown({iters_since_last}/{self.config.cooldown_iterations})",
            )
        return None

    async def advise(
        self,
        question: str,
        urgency: AdvisorUrgency,
        trigger: AdvisorTrigger,
        conversation: list[dict[str, Any]],
        ctx: AgentContext,
        iteration_count: int,
    ) -> AdvisorVerdict:
        """Produce an AdvisorVerdict.

        Returns a throttled verdict without calling the gateway if disabled,
        over budget, within cooldown, or at max-calls. Otherwise builds a
        compressed transcript and calls the advisor gateway.
        """
        throttled = self._check_throttle(urgency, trigger, iteration_count)
        if throttled is not None:
            logger.debug(
                "advisor throttled: trigger=%s reason=%s",
                trigger.value, throttled.throttle_reason,
            )
            return throttled

        assert self.gateway is not None  # enabled implies gateway

        try:
            system_prompt = load_advisor_prompt()
            user_context = build_advisor_context(
                question=question,
                urgency=urgency,
                trigger=trigger,
                conversation=conversation,
                ctx=ctx,
                previous_verdicts=self.previous_verdicts,
                iteration_count=iteration_count,
            )
            advice_text, tokens_used, cost_usd = await call_advisor_gateway(
                self,
                system_prompt=system_prompt,
                user_context=user_context,
            )
        except Exception as exc:
            logger.warning("advisor call failed: %s", exc)
            return AdvisorVerdict.throttled_verdict(
                urgency, trigger, f"call_failed:{type(exc).__name__}",
            )

        verdict = AdvisorVerdict(
            advice=advice_text.strip(),
            urgency=urgency,
            trigger=trigger,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
        )
        return verdict

    def record_verdict(self, verdict: AdvisorVerdict, iteration_count: int) -> None:
        """Update session state after a successful (non-throttled) advisor call."""
        if verdict.throttled:
            return
        self.call_count += 1
        self.cost_spent_usd += verdict.cost_usd
        self.last_call_iteration = iteration_count
        self.triggers_fired.add(verdict.trigger.value)
        self.previous_verdicts.append(verdict)


def resolve_advisor_config(
    *,
    enabled: bool,
    executor_provider: str,
    advisor_provider: str | None,
    advisor_model: str | None,
    session_budget_usd: float | None,
    budget_fraction: float,
) -> AdvisorConfig:
    """Resolve the advisor config from CLI/env/yaml precedence.

    The executor's provider and session budget are inherited. Overrides come
    from CLI flags. Defaults: advisor provider = executor provider, advisor
    model = strong tier of advisor provider, budget = fraction × session.
    """
    from maike.constants import model_for_tier
    from maike.gateway.providers import resolve_provider_name

    if not enabled:
        return AdvisorConfig(enabled=False)

    provider = advisor_provider or executor_provider
    resolved_provider = resolve_provider_name(provider_name=provider).value
    model = advisor_model or model_for_tier(resolved_provider, "strong")

    # Session budget of 0 or None = unlimited; give advisor a 0 budget too
    # (which means unlimited via budget_remaining_usd check — see below).
    if session_budget_usd is None or session_budget_usd <= 0:
        budget = 0.0  # 0 = unlimited
    else:
        budget = session_budget_usd * max(0.0, min(1.0, budget_fraction))

    return AdvisorConfig(
        enabled=True,
        provider=resolved_provider,
        model=model,
        budget_usd=budget,
    )


def before_first_edit_condition(
    conversation: list[dict[str, Any]],
    ctx: Any = None,  # kept for signature parity with other trigger helpers
) -> bool:
    """True when the agent has NOT yet emitted any Edit/Write tool call.

    Purely behavioral — no keyword matching, no char threshold, no task-text
    heuristics. Every session gets exactly one advisor checkpoint before
    its first code mutation, regardless of how the task is phrased.

    The trigger fires at most ONCE per session (``triggers_fired`` guard).
    """
    for msg in conversation:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") in {"Edit", "Write"}:
                return False
    return True


# Back-compat alias — will be removed in a future release.
before_first_write_condition = before_first_edit_condition


def before_completion_condition(
    conversation: list[dict[str, Any]],
    last_llm_had_tool_calls: bool,
) -> tuple[bool, str]:
    """True when the agent is about to end the turn prematurely.

    Fires when the last LLM response emitted NO tool calls (text-only end of
    turn) AND one of:
      (a) the agent made ZERO edits all session — "I'd do X" without doing X
      (b) the agent made edits but ran no verification Bash since the last
          edit — "done" without checking

    Returns ``(should_fire, reason_code)`` so the caller can phrase the
    advisor question specifically.
    """
    if last_llm_had_tool_calls:
        return False, ""

    last_edit_idx = -1
    last_verify_idx = -1
    any_edit = False

    for i, msg in enumerate(conversation):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            if name in {"Edit", "Write"}:
                last_edit_idx = i
                any_edit = True
            elif name == "Bash":
                last_verify_idx = i

    if not any_edit:
        return True, "zero_edits"
    if last_edit_idx > last_verify_idx:
        # Edit happened after (or without) the last Bash verification.
        return True, "unverified_edit"
    return False, ""


def exploration_threshold_met(
    conversation: list[dict[str, Any]],
    threshold: int = ADVISOR_EXPLORATION_THRESHOLD,
) -> bool:
    """True if the agent has run ≥ ``threshold`` read/grep calls AND has not
    yet mutated any files. Signals it has finished gathering context and is
    about to write — good moment for a plan check.

    Bash counts as exploration (running tests, ``ls``, ``which``) — it's only
    Edit/Write that mark the transition into the implementation phase.
    """
    read_count = 0
    for msg in conversation:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                name = block.get("name", "")
                if name in {"Read", "Grep", "SemanticSearch", "Glob"}:
                    read_count += 1
                elif name in {"Edit", "Write"}:
                    # Agent has started writing — too late for plan check.
                    return False
    return read_count >= threshold


# Lazy-loaded advisor prompt; populated the first time it's needed.
_advisor_prompt_cache: dict[str, str] = {}


def load_advisor_prompt() -> str:
    """Load the advisor system prompt. Concatenates core + guidance files."""
    if "prompt" in _advisor_prompt_cache:
        return _advisor_prompt_cache["prompt"]

    prompts_dir = Path(__file__).parent / "prompts"
    core_path = prompts_dir / "advisor-core.md"
    guidance_path = prompts_dir / "advisor-guidance.md"

    parts = []
    if core_path.exists():
        parts.append(core_path.read_text(encoding="utf-8"))
    if guidance_path.exists():
        parts.append(
            '<maike-guidance priority="low">\n'
            + guidance_path.read_text(encoding="utf-8")
            + "\n</maike-guidance>"
        )
    prompt = "\n\n".join(parts) if parts else _fallback_advisor_prompt()
    _advisor_prompt_cache["prompt"] = prompt
    return prompt


def _fallback_advisor_prompt() -> str:
    """Minimal prompt used if the prompt files are missing."""
    return (
        "You are a strategic advisor to a coding agent running on a cheaper model. "
        "You see a compressed transcript and a specific question. Output 1–3 short "
        "sentences with concrete, actionable next steps. No preamble, no markdown "
        "headers. If the agent's plan looks fine, say so and tell it to proceed."
    )


def build_advisor_context(
    question: str,
    urgency: AdvisorUrgency,
    trigger: AdvisorTrigger,
    conversation: list[dict[str, Any]],
    ctx: AgentContext,
    previous_verdicts: list[AdvisorVerdict],
    iteration_count: int,
    max_chars: int = ADVISOR_TRANSCRIPT_MAX_CHARS,
    recent_tail_chars: int = ADVISOR_RECENT_MESSAGES_CHARS,
) -> str:
    """Build the compressed transcript + framing sent to the advisor.

    Sections:
      1. Task framing (task, iteration, urgency, question)
      2. Previous advice (if any) — so advisor doesn't repeat itself
      3. Session summary (compact recap via SessionSummaryBuilder)
      4. Recent raw tail (last N chars of conversation verbatim)
    """
    parts: list[str] = []

    # 1. Task framing
    task = getattr(ctx, "task", "") or "(no task text available)"
    parts.append(
        f"# Task\n{task.strip()}\n\n"
        f"Iteration: {iteration_count}"
        f"{' / ' + str(ctx.max_iterations) if getattr(ctx, 'max_iterations', 0) else ''}\n"
        f"Urgency: {urgency.value}\n"
        f"Trigger: {trigger.value}"
    )

    # 2. Previous advice — avoid repeating
    if previous_verdicts:
        recent_advice = previous_verdicts[-2:]
        lines = ["# Previous advice (do not repeat)"]
        for v in recent_advice:
            lines.append(f"- [{v.trigger.value}] {v.advice.strip()}")
        parts.append("\n".join(lines))

    # 3. Session summary via SessionSummaryBuilder
    summary_text = _try_build_session_summary(ctx, conversation)
    if summary_text:
        parts.append(f"# Session summary\n{summary_text}")

    # 4. Recent raw tail
    tail = _recent_conversation_tail(conversation, recent_tail_chars)
    if tail:
        parts.append(f"# Recent conversation (raw)\n{tail}")

    # 5. Specific question
    parts.append(f"# Question\n{question.strip()}")

    context = "\n\n".join(parts)

    # Enforce hard cap — drop from the middle (keep framing + question).
    if len(context) > max_chars:
        head = parts[0]
        question_section = parts[-1]
        remaining_budget = max_chars - len(head) - len(question_section) - 100
        middle = "\n\n".join(parts[1:-1])
        if remaining_budget > 0 and len(middle) > remaining_budget:
            middle = middle[:remaining_budget] + "\n[... truncated ...]"
        context = "\n\n".join([head, middle, question_section])

    return context


def _try_build_session_summary(
    ctx: AgentContext,
    conversation: list[dict[str, Any]],
) -> str:
    """Best-effort session recap. Falls back to counting tool calls if the
    summary builder is unavailable or raises.
    """
    try:
        from maike.memory.summary import SessionSummaryBuilder

        builder = SessionSummaryBuilder()
        if hasattr(builder, "build_from_messages"):
            return builder.build_from_messages(conversation) or ""
    except Exception as exc:
        logger.debug("SessionSummaryBuilder unavailable: %s", exc)

    # Fallback: tool-call histogram
    tool_counts: dict[str, int] = {}
    for msg in conversation:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1
    if not tool_counts:
        return ""
    top = sorted(tool_counts.items(), key=lambda kv: -kv[1])[:8]
    return "Tool calls so far: " + ", ".join(f"{n}×{c}" for n, c in top)


def _recent_conversation_tail(
    conversation: list[dict[str, Any]],
    max_chars: int,
) -> str:
    """Render the most recent messages verbatim, up to ``max_chars``."""
    if max_chars <= 0 or not conversation:
        return ""
    rendered: list[str] = []
    total = 0
    for msg in reversed(conversation):
        text = _render_message(msg)
        if not text:
            continue
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                rendered.append(text[-remaining:])
            break
        rendered.append(text)
        total += len(text)
    return "\n---\n".join(reversed(rendered))


def _render_message(msg: dict[str, Any]) -> str:
    """Flatten a message dict into a short text rendering."""
    role = msg.get("role", "?")
    content = msg.get("content")
    if isinstance(content, str):
        return f"[{role}] {content}"
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = block.get("input", {})
            parts.append(f"→ {name}({_short_repr(inp)})")
        elif btype == "tool_result":
            output = block.get("content", "")
            if isinstance(output, list):
                output = " ".join(
                    b.get("text", "") for b in output if isinstance(b, dict)
                )
            parts.append(f"← {str(output)[:400]}")
    rendered = " ".join(p for p in parts if p).strip()
    if not rendered:
        return ""
    return f"[{role}] {rendered}"


def _short_repr(obj: Any, max_len: int = 120) -> str:
    s = repr(obj)
    if len(s) > max_len:
        s = s[:max_len] + "...)"
    return s


async def call_advisor_gateway(
    session: AdvisorSession,
    system_prompt: str,
    user_context: str,
) -> tuple[str, int, float]:
    """Call the advisor gateway. Returns (advice_text, tokens_used, cost_usd)."""
    if session.gateway is None:
        raise RuntimeError("advisor gateway is None — session is not enabled")

    messages = [{"role": "user", "content": user_context}]

    result = await session.gateway.call(
        system=system_prompt,
        messages=messages,
        tools=[],  # advisor has no tools — pure reasoning output
        model=session.config.model,
        temperature=ADVISOR_TEMPERATURE,
        max_tokens=ADVISOR_MAX_OUTPUT_TOKENS,
    )

    # Prefer LLMResult.cost_usd (per-call) and usage.total (per-call).
    cost_usd = float(getattr(result, "cost_usd", 0.0) or 0.0)
    usage = getattr(result, "usage", None)
    if usage is not None:
        tokens_used = int(usage.input_tokens + usage.output_tokens)
    else:
        tokens_used = 0

    # Extract text blocks. Adapters either populate content_blocks (preferred)
    # or the flat `content` field — handle both.
    text_parts: list[str] = []
    blocks = getattr(result, "content_blocks", None) or []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
    if not text_parts:
        flat = getattr(result, "content", None)
        if isinstance(flat, str) and flat:
            text_parts.append(flat)

    advice = "\n".join(t for t in text_parts if t).strip()
    return advice, tokens_used, cost_usd
