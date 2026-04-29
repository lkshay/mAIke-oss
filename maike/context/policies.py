"""Per-stage context policies controlling which artifacts are inlined fully,
summarized, or omitted."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageContextPolicy:
    """Declares how artifacts should be presented to an agent in a given stage."""

    stage_name: str
    full_artifacts: tuple[str, ...] = ()
    summarized_artifacts: tuple[str, ...] = ()
    omitted_artifacts: tuple[str, ...] = ()


# --------------------------------------------------------------------- #
# Policy registry
# --------------------------------------------------------------------- #

CONTEXT_POLICIES: dict[str, StageContextPolicy] = {
    "coding": StageContextPolicy(
        stage_name="coding",
        full_artifacts=("acceptance-contract.md", "architecture.md", "plan.md"),
        summarized_artifacts=("spec.md",),
    ),
    "testing": StageContextPolicy(
        stage_name="testing",
        full_artifacts=("acceptance-contract.md", "code-summary.md", "fix-summary.md"),
        summarized_artifacts=("plan.md",),
        omitted_artifacts=("spec.md", "architecture.md"),
    ),
    "review": StageContextPolicy(
        stage_name="review",
        full_artifacts=("code-summary.md", "test-results.md"),
        summarized_artifacts=("spec.md", "acceptance-contract.md"),
    ),
    "acceptance": StageContextPolicy(
        stage_name="acceptance",
        full_artifacts=("acceptance-contract.md", "test-results.md", "review.md"),
        summarized_artifacts=("spec.md", "code-summary.md"),
    ),
    # Debugging pipeline — diagnosis stage gets raw task; fix stage gets diagnosis.
    "fix": StageContextPolicy(
        stage_name="fix",
        full_artifacts=("diagnosis.md",),
    ),
    # Editing pipeline reuses coding policy.
    "analysis": StageContextPolicy(
        stage_name="analysis",
        full_artifacts=(),
        summarized_artifacts=(),
    ),
}


def get_context_policy(stage_name: str) -> StageContextPolicy | None:
    """Look up the context policy for *stage_name*, or *None* if no policy is defined."""
    return CONTEXT_POLICIES.get(stage_name)
