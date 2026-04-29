"""Tests for maike.context.summarizer — ArtifactSummarizer."""

from maike.atoms.artifact import Artifact, ArtifactKind, ArtifactType
from maike.context.policies import CONTEXT_POLICIES, get_context_policy
from maike.context.summarizer import (
    SUMMARIZE_THRESHOLD,
    ArtifactSummarizer,
    summarize_artifacts,
)


def _make_artifact(
    name: str = "spec.md",
    content: str = "short content",
    stage: str = "requirements",
    kind: ArtifactKind = ArtifactKind.STAGE,
    artifact_type: ArtifactType = ArtifactType.SPEC,
) -> Artifact:
    return Artifact(
        logical_name=name,
        content=content,
        stage_name=stage,
        produced_by="test",
        kind=kind,
        type=artifact_type,
    )


def _long_spec() -> str:
    """Generate a spec >SUMMARIZE_THRESHOLD chars."""
    lines = ["# Requirements\n"]
    lines.append("This is the project specification.\n")
    for i in range(80):
        lines.append(f"- Requirement {i}: The system must handle case {i} correctly.")
    lines.append("\n# Non-Functional Requirements\n")
    lines.append("Performance must be acceptable under load.\n")
    for i in range(40):
        lines.append(f"- NFR {i}: Latency under {i * 10}ms for operation {i}.")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# should_summarize
# ------------------------------------------------------------------ #


def test_should_summarize_short_content_returns_false():
    summarizer = ArtifactSummarizer()
    artifact = _make_artifact(content="short")
    assert not summarizer.should_summarize(artifact)


def test_should_summarize_long_content_returns_true():
    summarizer = ArtifactSummarizer()
    artifact = _make_artifact(content="x" * (SUMMARIZE_THRESHOLD + 1))
    assert summarizer.should_summarize(artifact)


# ------------------------------------------------------------------ #
# summarize — spec
# ------------------------------------------------------------------ #


def test_summarize_spec_is_shorter():
    summarizer = ArtifactSummarizer()
    content = _long_spec()
    artifact = _make_artifact(content=content)
    summary = summarizer.summarize(artifact, role="coder")
    assert len(summary) < len(content)


def test_summarize_spec_preserves_headings():
    summarizer = ArtifactSummarizer()
    content = _long_spec()
    artifact = _make_artifact(content=content)
    summary = summarizer.summarize(artifact, role="coder")
    assert "# Requirements" in summary
    assert "# Non-Functional Requirements" in summary


def test_summarize_spec_preserves_bullets():
    summarizer = ArtifactSummarizer()
    content = _long_spec()
    artifact = _make_artifact(content=content)
    summary = summarizer.summarize(artifact, role="coder")
    assert "- Requirement 0:" in summary


# ------------------------------------------------------------------ #
# summarize — acceptance-contract is NEVER summarized
# ------------------------------------------------------------------ #


def test_acceptance_contract_never_summarized():
    summarizer = ArtifactSummarizer()
    content = "x" * 10_000
    artifact = _make_artifact(name="acceptance-contract.md", content=content)
    result = summarizer.summarize(artifact, role="coder")
    assert result == content


# ------------------------------------------------------------------ #
# summarize — plan
# ------------------------------------------------------------------ #


def test_summarize_plan_keeps_numbered_steps():
    summarizer = ArtifactSummarizer()
    content = "\n".join([
        "# Implementation Plan",
        "",
        "1. Create src/app.py with the main entry point.",
        "2. Implement the database layer in src/db.py.",
        "3. Write tests in tests/test_app.py.",
        "",
        "Some filler paragraph that should be excluded from the summary " * 20,
    ])
    artifact = _make_artifact(name="plan.md", content=content, artifact_type=ArtifactType.PLAN)
    summary = summarizer.summarize(artifact, role="coder")
    assert "1. Create src/app.py" in summary
    assert "2. Implement" in summary


# ------------------------------------------------------------------ #
# summarize — test-results
# ------------------------------------------------------------------ #


def test_summarize_test_results_keeps_pass_fail():
    summarizer = ArtifactSummarizer()
    content = "\n".join([
        "# Test Results",
        "",
        "Running pytest...",
        "collected 15 items",
        "",
        *[f"test_{i} PASSED" for i in range(12)],
        "test_13 FAILED - assertion error",
        "test_14 FAILED - timeout",
        "test_15 PASSED",
        "",
        "Summary: 13 passed, 2 failed",
    ])
    artifact = _make_artifact(name="test-results.md", content=content, artifact_type=ArtifactType.RESULT)
    summary = summarizer.summarize(artifact, role="reviewer")
    assert "FAILED" in summary
    assert "13 passed" in summary


# ------------------------------------------------------------------ #
# summarize — review
# ------------------------------------------------------------------ #


def test_summarize_review_keeps_findings():
    summarizer = ArtifactSummarizer()
    content = "\n".join([
        "# Code Review",
        "",
        "- [critical] Missing input validation in api.py:handle_request",
        "- [minor] Inconsistent naming in utils.py",
        "",
        "Long paragraph explaining the review methodology and background " * 20,
    ])
    artifact = _make_artifact(name="review.md", content=content, artifact_type=ArtifactType.REVIEW)
    summary = summarizer.summarize(artifact, role="acceptance")
    assert "[critical]" in summary
    assert "[minor]" in summary


# ------------------------------------------------------------------ #
# summarize_artifacts with policy
# ------------------------------------------------------------------ #


def test_summarize_artifacts_omits_artifacts():
    spec = _make_artifact(name="spec.md", content="long " * 1000)
    contract = _make_artifact(name="acceptance-contract.md", content="contract")

    result = summarize_artifacts(
        [spec, contract],
        omitted_names=("spec.md",),
        full_names=("acceptance-contract.md",),
    )
    assert len(result) == 1
    assert result[0].logical_name == "acceptance-contract.md"


def test_summarize_artifacts_keeps_full():
    content = "full content " * 500
    artifact = _make_artifact(name="architecture.md", content=content, artifact_type=ArtifactType.ARCHITECTURE)
    result = summarize_artifacts([artifact], full_names=("architecture.md",))
    assert result[0].content == content


def test_summarize_artifacts_summarizes_when_over_threshold():
    content = _long_spec()
    spec = _make_artifact(name="spec.md", content=content)
    result = summarize_artifacts([spec], summarized_names=("spec.md",), role="coder")
    assert len(result) == 1
    assert result[0].content.startswith("[SUMMARIZED]")
    assert len(result[0].content) < len(content)


def test_summarize_artifacts_skips_summarize_when_under_threshold():
    spec = _make_artifact(name="spec.md", content="short spec")
    result = summarize_artifacts([spec], summarized_names=("spec.md",), role="coder")
    assert result[0].content == "short spec"


# ------------------------------------------------------------------ #
# Policy integration
# ------------------------------------------------------------------ #


def test_coding_policy_exists():
    policy = get_context_policy("coding")
    assert policy is not None
    assert "acceptance-contract.md" in policy.full_artifacts


def test_testing_policy_omits_spec():
    policy = get_context_policy("testing")
    assert policy is not None
    assert "spec.md" in policy.omitted_artifacts


def test_acceptance_policy_keeps_contract_full():
    policy = get_context_policy("acceptance")
    assert policy is not None
    assert "acceptance-contract.md" in policy.full_artifacts


def test_unknown_stage_returns_none():
    assert get_context_policy("nonexistent") is None
