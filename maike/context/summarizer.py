"""Role-aware artifact summarization — pure string processing, no LLM calls."""

from __future__ import annotations

import re
from typing import Sequence

from maike.atoms.artifact import Artifact


# Characters below which we never summarize.
SUMMARIZE_THRESHOLD = 3000
# Maximum characters for a summary.
MAX_SUMMARY_CHARS = 3200


class ArtifactSummarizer:
    """Produces concise, role-aware summaries of pipeline artifacts.

    The caller decides *which* artifacts to summarize (via ``StageContextPolicy``).
    This class only knows *how* to compress each artifact type, optionally
    adapting the output to the consumer's role.
    """

    def should_summarize(self, artifact: Artifact, role: str = "") -> bool:  # noqa: ARG002
        """Heuristic: summarize if the content is large enough to benefit."""
        return len(artifact.content) > SUMMARIZE_THRESHOLD

    def summarize(
        self,
        artifact: Artifact,
        role: str = "",
        max_chars: int = MAX_SUMMARY_CHARS,
    ) -> str:
        """Return a shorter version of *artifact.content* tailored to *role*."""
        name = artifact.logical_name.lower()

        if "acceptance-contract" in name:
            # Never summarize — always critical for correctness.
            return artifact.content

        if "spec" in name:
            return self._summarize_spec(artifact.content, role, max_chars)
        if "plan" in name:
            return self._summarize_plan(artifact.content, role, max_chars)
        if "architecture" in name:
            return self._summarize_architecture(artifact.content, role, max_chars)
        if "code-summary" in name or "fix-summary" in name:
            return _head(artifact.content, max_chars)
        if "test-results" in name:
            return self._summarize_test_results(artifact.content, max_chars)
        if "review" in name:
            return self._summarize_review(artifact.content, max_chars)
        if "diagnosis" in name:
            return _head(artifact.content, max_chars)

        # Unknown artifact — simple head truncation.
        return _head(artifact.content, max_chars)

    # ------------------------------------------------------------------ #
    # Per-artifact strategies
    # ------------------------------------------------------------------ #

    def _summarize_spec(self, content: str, role: str, max_chars: int) -> str:
        """Headings + first sentence per section + bulleted items."""
        sections = _split_sections(content)
        parts: list[str] = []
        for heading, body in sections:
            parts.append(heading)
            bullets = _extract_bullets(body)
            first_sentence = _first_sentence(body)
            if role in ("coder", "debugger"):
                # Coder cares about file lists, constraints, interface contracts.
                parts.extend(bullets[:10])
                if not bullets and first_sentence:
                    parts.append(first_sentence)
            elif role == "tester":
                # Tester cares about acceptance criteria and test requirements.
                parts.extend(bullets[:10])
            elif role in ("reviewer", "acceptance"):
                parts.extend(bullets[:8])
                if first_sentence and not bullets:
                    parts.append(first_sentence)
            else:
                parts.extend(bullets[:6])
                if first_sentence and not bullets:
                    parts.append(first_sentence)
        return _cap("\n".join(parts), max_chars)

    def _summarize_plan(self, content: str, role: str, max_chars: int) -> str:  # noqa: ARG002
        """Numbered steps + file paths mentioned."""
        lines = content.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Keep headings, numbered items, and lines mentioning file paths.
            if (
                stripped.startswith("#")
                or re.match(r"^\d+[\.\)]\s", stripped)
                or re.match(r"^[-*]\s", stripped)
                or _mentions_path(stripped)
            ):
                kept.append(line)
        return _cap("\n".join(kept), max_chars)

    def _summarize_architecture(self, content: str, role: str, max_chars: int) -> str:  # noqa: ARG002
        """Component names, interfaces, data flow, key decisions."""
        sections = _split_sections(content)
        parts: list[str] = []
        for heading, body in sections:
            parts.append(heading)
            # Keep bullet points and lines with keywords.
            for line in body.split("\n"):
                stripped = line.strip()
                if (
                    stripped.startswith("-")
                    or stripped.startswith("*")
                    or _mentions_path(stripped)
                    or any(kw in stripped.lower() for kw in ("interface", "component", "flow", "contract", "api", "schema", "class", "module"))
                ):
                    parts.append(line)
        return _cap("\n".join(parts), max_chars)

    def _summarize_test_results(self, content: str, max_chars: int) -> str:
        """Pass/fail counts + failure details only."""
        lines = content.split("\n")
        kept: list[str] = []
        in_failure_block = False
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in ("passed", "failed", "error", "failure", "total", "summary", "result")):
                kept.append(line)
                in_failure_block = "fail" in lower or "error" in lower
            elif in_failure_block:
                kept.append(line)
                if not line.strip():
                    in_failure_block = False
            elif line.strip().startswith("#"):
                kept.append(line)
        return _cap("\n".join(kept), max_chars) if kept else _head(content, max_chars)

    def _summarize_review(self, content: str, max_chars: int) -> str:
        """Findings with severity only."""
        lines = content.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith("-")
                or stripped.startswith("*")
                or any(kw in stripped.lower() for kw in ("severity", "finding", "issue", "critical", "warning", "blocking", "minor", "major"))
            ):
                kept.append(line)
        return _cap("\n".join(kept), max_chars) if kept else _head(content, max_chars)


# ---------------------------------------------------------------------- #
# Text helpers
# ---------------------------------------------------------------------- #


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) pairs."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in text.split("\n"):
        if line.strip().startswith("#"):
            if current_heading or current_body:
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)

    if current_heading or current_body:
        sections.append((current_heading, "\n".join(current_body)))
    return sections


def _extract_bullets(text: str) -> list[str]:
    """Extract lines that start with a bullet marker."""
    return [
        line
        for line in text.split("\n")
        if line.strip().startswith("-") or line.strip().startswith("*") or re.match(r"^\s*\d+[\.\)]\s", line)
    ]


def _first_sentence(text: str) -> str:
    """Extract the first non-empty sentence."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("-") and not stripped.startswith("*"):
            # Take up to the first period or the whole line.
            dot = stripped.find(". ")
            if dot > 0:
                return stripped[: dot + 1]
            return stripped
    return ""


def _mentions_path(line: str) -> bool:
    """Heuristic: does the line mention a file path?"""
    return bool(re.search(r"\b\w+\.\w{1,5}\b", line) and "/" in line or line.strip().endswith((".py", ".js", ".ts", ".go", ".rs", ".md")))


def _head(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated]"


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Truncate at a line boundary.
    truncated = text[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars * 0.5:
        truncated = truncated[:last_nl]
    return truncated + "\n[...summarized]"


def summarize_artifacts(
    artifacts: Sequence[Artifact],
    *,
    full_names: Sequence[str] = (),
    summarized_names: Sequence[str] = (),
    omitted_names: Sequence[str] = (),
    role: str = "",
    max_chars: int = MAX_SUMMARY_CHARS,
) -> list[Artifact]:
    """Apply summarization policy to a list of artifacts, returning new Artifact
    objects with (possibly) compressed content.

    - ``full_names``: always keep full content.
    - ``summarized_names``: summarize if over threshold.
    - ``omitted_names``: exclude entirely.
    - Artifacts not in any list are kept as-is.
    """
    summarizer = ArtifactSummarizer()
    full_set = set(full_names)
    summarized_set = set(summarized_names)
    omitted_set = set(omitted_names)
    result: list[Artifact] = []

    for artifact in artifacts:
        name = artifact.logical_name
        if name in omitted_set:
            continue
        if name in full_set:
            result.append(artifact)
            continue
        if name in summarized_set and summarizer.should_summarize(artifact, role):
            new_content = summarizer.summarize(artifact, role, max_chars)
            result.append(artifact.model_copy(update={"content": f"[SUMMARIZED]\n{new_content}"}))
            continue
        # Default: keep as-is.
        result.append(artifact)

    return result
