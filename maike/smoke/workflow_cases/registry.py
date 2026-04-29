"""Registry assembly for workflow cases."""
from __future__ import annotations

WORKFLOW_CASES: dict[str, object] = {}

DEFAULT_WORKFLOW_NAMES = ("react-greenfield", "react-editing", "react-debugging")

_tiers_merged = False


def _ensure_tier_cases() -> None:
    global _tiers_merged
    if _tiers_merged:
        return
    _tiers_merged = True
    from maike.smoke.workflow_cases.tier1 import TIER1_EVAL_CASES
    from maike.smoke.workflow_cases.tier2 import TIER2_EVAL_CASES
    from maike.smoke.workflow_cases.tier3 import TIER3_EVAL_CASES
    from maike.smoke.workflow_cases.tier4 import TIER4_EVAL_CASES

    WORKFLOW_CASES.update(TIER1_EVAL_CASES)
    WORKFLOW_CASES.update(TIER2_EVAL_CASES)
    WORKFLOW_CASES.update(TIER3_EVAL_CASES)
    WORKFLOW_CASES.update(TIER4_EVAL_CASES)


TIER_WORKFLOW_NAMES: dict[str, tuple[str, ...]] = {}

_react_merged = False


def _ensure_react_cases() -> None:
    global _react_merged
    if _react_merged:
        return
    _react_merged = True
    from maike.smoke.workflow_cases.react_cases import REACT_EVAL_CASES
    WORKFLOW_CASES.update(REACT_EVAL_CASES)


_hard_merged = False


def _ensure_hard_cases() -> None:
    global _hard_merged
    if _hard_merged:
        return
    _hard_merged = True
    from maike.smoke.workflow_cases.hard_cases import HARD_EVAL_CASES
    WORKFLOW_CASES.update(HARD_EVAL_CASES)


_agentic_merged = False


def _ensure_agentic_cases() -> None:
    global _agentic_merged
    if _agentic_merged:
        return
    _agentic_merged = True
    from maike.smoke.workflow_cases.agentic_cases import AGENTIC_EVAL_CASES
    WORKFLOW_CASES.update(AGENTIC_EVAL_CASES)


_hard_agentic_merged = False


def _ensure_hard_agentic_cases() -> None:
    global _hard_agentic_merged
    if _hard_agentic_merged:
        return
    _hard_agentic_merged = True
    from maike.smoke.workflow_cases.hard_agentic_cases import HARD_AGENTIC_EVAL_CASES
    WORKFLOW_CASES.update(HARD_AGENTIC_EVAL_CASES)


def _ensure_all_cases() -> None:
    """Ensure both tier and react cases are loaded, and TIER_WORKFLOW_NAMES is populated."""
    _ensure_tier_cases()
    _ensure_react_cases()
    _ensure_hard_cases()
    _ensure_agentic_cases()
    _ensure_hard_agentic_cases()
    if not TIER_WORKFLOW_NAMES:
        for tier in ("tier1", "tier2", "tier3", "tier4"):
            TIER_WORKFLOW_NAMES[tier] = tuple(
                name for name, case in WORKFLOW_CASES.items() if tier in getattr(case, "tags", ())
            )
