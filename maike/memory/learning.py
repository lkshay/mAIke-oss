"""Session learning — extract and retrieve learnings from completed sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

from maike.memory.longterm import LongTermMemory
from maike.utils import utcnow


_LEARNING_COLLECTION = "session_learnings"


@dataclass
class SessionLearning:
    task_summary: str
    outcome: str  # "success" | "partial" | "failure"
    pipeline_used: str
    language: str
    cost_usd: float
    failure_reasons: list[str] = field(default_factory=list)
    successful_strategies: list[str] = field(default_factory=list)
    role_specific_learnings: dict[str, list[str]] = field(default_factory=dict)
    timestamp: str = ""

    def to_content(self) -> str:
        parts = [
            f"Task: {self.task_summary}",
            f"Outcome: {self.outcome}",
            f"Pipeline: {self.pipeline_used}",
            f"Language: {self.language}",
            f"Cost: ${self.cost_usd:.2f}",
        ]
        if self.failure_reasons:
            parts.append("Failures: " + "; ".join(self.failure_reasons))
        if self.successful_strategies:
            parts.append("Strategies: " + "; ".join(self.successful_strategies))
        if self.role_specific_learnings:
            for role, learnings in sorted(self.role_specific_learnings.items()):
                parts.append(f"Learnings ({role}): " + "; ".join(learnings))
        if self.timestamp:
            parts.append(f"When: {self.timestamp}")
        return "\n".join(parts)


class SessionLearner:
    """Records session outcomes and retrieves relevant learnings for future tasks."""

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    def record_session(
        self,
        *,
        session_id: str,
        task: str,
        outcome: str,
        pipeline: str,
        language: str = "unknown",
        cost_usd: float = 0.0,
        failure_reasons: list[str] | None = None,
        successful_strategies: list[str] | None = None,
        role_specific_learnings: dict[str, list[str]] | None = None,
    ) -> SessionLearning:
        learning = SessionLearning(
            task_summary=task[:200],
            outcome=outcome,
            pipeline_used=pipeline,
            language=language,
            cost_usd=cost_usd,
            failure_reasons=failure_reasons or [],
            successful_strategies=successful_strategies or [],
            role_specific_learnings=role_specific_learnings or {},
            timestamp=utcnow().isoformat(),
        )
        self._memory.add(
            collection=_LEARNING_COLLECTION,
            key=session_id,
            content=learning.to_content(),
        )
        return learning

    def retrieve_relevant_learnings(
        self,
        task: str,
        language: str = "",
        limit: int = 3,
    ) -> list[dict[str, str]]:
        query = f"{task} {language}".strip()
        return self._memory.query(
            collection=_LEARNING_COLLECTION,
            query=query,
            limit=limit,
        )


def build_learning_context_blocks(
    learner: SessionLearner | None,
    task: str,
    language: str = "",
    role: str = "",
) -> list[str]:
    """Build context blocks from past learnings for injection into agent prompts.

    All learnings are included without role-based filtering since the
    react-only architecture uses a single unified role.
    """
    if learner is None:
        return []
    learnings = learner.retrieve_relevant_learnings(task, language, limit=3)
    if not learnings:
        return []

    lines = ["## Lessons from Previous Sessions\n"]
    for i, learning in enumerate(learnings, 1):
        lines.append(f"### Session {i}")
        lines.append(learning.get("content", "(no content)"))
        lines.append("")
    return ["\n".join(lines)]
