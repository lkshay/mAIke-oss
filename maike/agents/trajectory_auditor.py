"""TrajectoryAuditor — validates candidate nudges against task intent.

A deterministic detector (see ``stuck_detectors.py``) can detect a pattern
like "20 consecutive Reads without an Edit" but cannot tell whether that
pattern is **pathological** (executor is stuck) or **legitimate** (the task
is "summarize this codebase").

The auditor is a cheap-LLM validator.  Given the task text and a candidate
nudge, it returns:
  - ``approve`` — inject the nudge (optionally rephrased for task specificity)
  - ``veto`` — silently skip; the pattern is legitimate for this task

Design:
- Same silent ``_bg_gateway`` + Ollama-first pattern as ``constraints.py``
  and ``verdict.py`` — no primary-gateway calls, no TUI visibility.
- Budget + cooldown + max-calls throttle, modeled on ``advisor.AdvisorSession``.
- Per-session cache keyed by ``(kind, task_hash)`` so repeat firings for the
  same pattern on the same task cost one LLM call, not N.
- Fail-safe: any LLM error / JSON parse failure / timeout → approve with the
  original text.  Never raises; never introduces new failure modes.

Integration lives in ``AgentCore``: after the existing
``RepeatedFailureTracker`` injection block, call
``auditor.gate_and_inject(...)`` with each candidate nudge.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + verdict types
# ---------------------------------------------------------------------------


@dataclass
class AuditorConfig:
    enabled: bool = True
    # Absolute USD cap for auditor calls across a session.  Typically set to
    # a small fraction (5%) of the session budget by the orchestrator.
    budget_usd: float = 0.0
    max_calls: int = 6
    cooldown_iterations: int = 5
    # LLM call timeout.  Cheap tier is usually <2s; 10s is generous.
    timeout_s: float = 10.0


@dataclass
class AuditVerdict:
    """The auditor's decision for one candidate nudge."""

    decision: Literal["approve", "veto"]
    text: str | None = None  # final text to inject (None when veto)
    reason: str = ""
    cost_usd: float = 0.0
    throttled: bool = False
    source: Literal["llm", "cache", "fail_safe", "throttled"] = "fail_safe"


_SYSTEM_PROMPT = """\
You validate a procedural nudge for a coding agent.

A deterministic detector noticed a pattern (e.g. many Reads with no Edit) and
proposed a nudge.  You decide:

- If the pattern is LEGITIMATE for THIS specific task (e.g. task is
  "summarize this codebase" and the detector saw many Reads), emit NOTHING —
  an empty response.  The nudge will be suppressed.
- Otherwise, write a one- or two-sentence nudge, task-aware where possible
  (name the actual file, module, or verb the task cares about).  The nudge
  will be injected verbatim into the agent's prompt.

Format:
- Plain text.  No JSON.  No markdown fences.
- Keep under 50 words.
- When in doubt, write a nudge rather than veto — a small helpful reminder
  is cheaper than missing a pathological loop.
"""


# ---------------------------------------------------------------------------
# Auditor session
# ---------------------------------------------------------------------------


class TrajectoryAuditor:
    """Per-session auditor.  Create once per agent session, pass to AgentCore.

    Attribute ``config.enabled=False`` short-circuits everything — no calls,
    no budget usage, no state changes.  Same no-op behavior when
    ``config.budget_usd == 0`` and auditor is asked to fire — throttles out.
    """

    def __init__(
        self,
        *,
        gateway: Any,
        provider: str,
        config: AuditorConfig,
        tracer: Any = None,
    ) -> None:
        self.gateway = gateway
        self.provider = provider
        self.config = config
        self.tracer = tracer

        self.calls_made: int = 0
        self.cost_used_usd: float = 0.0
        self.last_call_iteration: int | None = None
        self._cache: dict[tuple[str, str], AuditVerdict] = {}

    # -------------------------------- gating
    def _throttle_reason(self, iteration: int | None) -> str | None:
        if not self.config.enabled:
            return "auditor disabled"
        if self.calls_made >= self.config.max_calls:
            return f"max_calls reached ({self.config.max_calls})"
        if self.config.budget_usd > 0 and self.cost_used_usd >= self.config.budget_usd:
            return f"budget exhausted (${self.cost_used_usd:.4f} / ${self.config.budget_usd:.4f})"
        if (
            iteration is not None
            and self.last_call_iteration is not None
            and iteration - self.last_call_iteration < self.config.cooldown_iterations
        ):
            return f"cooldown active (iter {iteration}, last {self.last_call_iteration})"
        return None

    # -------------------------------- public API
    async def audit(
        self,
        *,
        candidate: Any,  # stuck_detectors.CandidateNudge
        task: str,
        iteration: int | None = None,
    ) -> AuditVerdict:
        """Return an ``AuditVerdict`` for a candidate nudge.

        Fail-safe: any exception, timeout, or parse failure returns
        ``AuditVerdict(decision="approve", text=<original>, source="fail_safe")``.
        The auditor never introduces a new failure mode — worst case it falls
        back to the behavior you'd have without it.
        """
        original_text = getattr(candidate, "text", "") or ""

        # Cache check — same (kind, task_hash) twice → one LLM call.
        try:
            task_hash = candidate.task_hash(task)  # type: ignore[attr-defined]
            cache_key = (candidate.kind, task_hash)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return AuditVerdict(
                    decision=cached.decision,
                    text=cached.text,
                    reason=f"cache hit: {cached.reason}",
                    cost_usd=0.0,
                    source="cache",
                )
        except Exception:
            cache_key = None  # type: ignore[assignment]

        # Throttle check.
        throttle = self._throttle_reason(iteration)
        if throttle is not None:
            # Fail-safe: throttled → approve with original text (don't drop
            # a potentially useful nudge just because of budget).
            self._trace("audit_throttled", {"reason": throttle, "kind": candidate.kind})
            return AuditVerdict(
                decision="approve",
                text=original_text,
                reason=throttle,
                throttled=True,
                source="throttled",
            )

        # LLM call.
        try:
            verdict = await asyncio.wait_for(
                self._call_llm(candidate, task, original_text),
                timeout=self.config.timeout_s,
            )
            self.calls_made += 1
            self.last_call_iteration = iteration
            self.cost_used_usd += verdict.cost_usd
            if cache_key is not None:
                self._cache[cache_key] = verdict
            self._trace("audit_decision", {
                "kind": candidate.kind,
                "decision": verdict.decision,
                "reason": verdict.reason,
                "cost_usd": verdict.cost_usd,
            })
            return verdict
        except asyncio.TimeoutError:
            log.debug("Auditor timed out (%.1fs) for kind=%s", self.config.timeout_s, candidate.kind)
            return AuditVerdict(
                decision="approve", text=original_text,
                reason="timeout — fail-safe approve", source="fail_safe",
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Auditor error for kind=%s: %s", candidate.kind, exc)
            return AuditVerdict(
                decision="approve", text=original_text,
                reason=f"error — fail-safe approve: {exc}", source="fail_safe",
            )

    # -------------------------------- internals
    async def _call_llm(
        self,
        candidate: Any,
        task: str,
        original_text: str,
    ) -> AuditVerdict:
        """Ask the cheap model for a nudge string (approve) or empty (veto).

        No JSON parsing.  The system prompt tells the model to emit a nudge
        only if the pattern is pathological for THIS task, otherwise emit
        nothing.  Empty stdout → veto; non-empty → approve with that text.
        """
        evidence_str = ", ".join(
            f"{k}={v}" for k, v in (getattr(candidate, "evidence", {}) or {}).items()
        )[:300]
        user_msg = (
            f"Task:\n{task.strip()[:2000]}\n\n"
            f"Proposed nudge ({candidate.kind}):\n{original_text}\n\n"
            f"Evidence: {evidence_str or '(none)'}\n\n"
            "If the proposed nudge is helpful for THIS task, write a one- or "
            "two-sentence replacement that is task-specific (referring to the "
            "actual file or module where possible).  If the pattern is "
            "legitimate for this task (e.g. a read-only summarization task), "
            "write nothing — respond with an empty message."
        )

        from maike.agents.constraints import _select_extractor_gateway
        gateway, _provider, model = await _select_extractor_gateway(
            self.gateway, self.provider,
        )

        cost_before = self._gateway_cost(gateway)
        result = await gateway.call(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            model=model,
            temperature=0.0,
            max_tokens=256,
            tools=None,
        )
        cost_after = self._gateway_cost(gateway)
        call_cost = max(0.0, cost_after - cost_before)

        text = (result.content or "").strip()
        # Strip optional markdown fences.
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        if not text:
            return AuditVerdict(
                decision="veto", text=None,
                reason="model returned empty — pattern legitimate for task",
                cost_usd=call_cost, source="llm",
            )

        # Non-empty reply = approve with (task-aware) rephrasing.
        final_text = text[:1200]
        return AuditVerdict(
            decision="approve", text=final_text,
            reason="task-aware nudge",
            cost_usd=call_cost, source="llm",
        )

    @staticmethod
    def _gateway_cost(gateway: Any) -> float:
        """Best-effort cost snapshot — tolerates different gateway backends."""
        ct = getattr(gateway, "cost_tracker", None)
        if ct is None:
            return 0.0
        try:
            return float(getattr(ct, "session_total", 0.0))
        except Exception:
            return 0.0

    def _trace(self, name: str, payload: dict[str, Any]) -> None:
        if self.tracer is None:
            return
        try:
            # Tracer API used elsewhere in mAIke.
            log_method = getattr(self.tracer, "log_context_event", None)
            if callable(log_method):
                log_method(name, payload)
        except Exception:
            pass
