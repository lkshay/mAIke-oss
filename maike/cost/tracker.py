"""Cost tracking and budget enforcement."""

from __future__ import annotations

from maike.atoms.context import AgentContext
from maike.atoms.llm import LLMCallRecord


class BudgetExceededError(RuntimeError):
    """Raised when an agent or session exceeds budget."""


class CostTracker:
    def __init__(self, session_budget_usd: float | None = None) -> None:
        self._session_cost = 0.0
        self._records: list[LLMCallRecord] = []
        self._session_budget_usd = session_budget_usd

    def reset(self, session_budget_usd: float | None = None) -> None:
        self._session_cost = 0.0
        self._records = []
        self._session_budget_usd = session_budget_usd

    def configure_session_budget(self, session_budget_usd: float | None) -> None:
        self._session_budget_usd = session_budget_usd

    def record(self, record: LLMCallRecord) -> None:
        self._session_cost += record.cost_usd
        self._records.append(record)

    def check_session_budget(self) -> None:
        if not self._session_budget_usd:  # None or 0 = unlimited
            return
        if self._session_cost > self._session_budget_usd:
            raise BudgetExceededError(
                "Session cost budget exceeded "
                f"(${self._session_cost:.4f} / ${self._session_budget_usd:.2f})"
            )

    def check_projected_session_budget(
        self,
        projected_call_cost_usd: float,
        *,
        safety_margin: float,
    ) -> None:
        if not self._session_budget_usd:  # None or 0 = unlimited
            return
        remaining_budget = max(self._session_budget_usd - self._session_cost, 0.0)
        if projected_call_cost_usd > remaining_budget * safety_margin:
            raise BudgetExceededError(
                "Projected session cost budget exceeded before LLM call "
                f"(projected ${projected_call_cost_usd:.4f} vs remaining ${remaining_budget:.4f}, "
                f"margin {safety_margin:.2f})"
            )

    @property
    def session_total(self) -> float:
        return self._session_cost

    @property
    def records(self) -> list[LLMCallRecord]:
        return list(self._records)


class BudgetEnforcer:
    def check(
        self,
        ctx: AgentContext,
        *,
        reserved_tokens: int = 0,
        reserved_cost_usd: float = 0.0,
    ) -> None:
        if ctx.token_budget > 0:
            token_total = ctx.tokens_used + reserved_tokens
            if token_total > ctx.token_budget or (reserved_tokens == 0 and ctx.tokens_used >= ctx.token_budget):
                raise BudgetExceededError(
                    f"Agent {ctx.agent_id} exceeded token budget "
                    f"({ctx.tokens_used} + reserved {reserved_tokens} / {ctx.token_budget})"
                )
        if ctx.cost_budget_usd > 0:
            cost_total = ctx.cost_used_usd + reserved_cost_usd
            if cost_total > ctx.cost_budget_usd or (reserved_cost_usd == 0 and ctx.cost_used_usd >= ctx.cost_budget_usd):
                raise BudgetExceededError(
                    f"Agent {ctx.agent_id} exceeded cost budget "
                    f"(${ctx.cost_used_usd:.4f} + reserved ${reserved_cost_usd:.4f} / ${ctx.cost_budget_usd:.2f})"
                )
