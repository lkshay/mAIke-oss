"""Tests for ``Orchestrator._plan_partitions`` (item #4).

The partition planner now uses provider-native structured output — no
``json.loads`` on LLM text in the live path.  These tests verify:

  - Happy path: gateway returns ``LLMResult.parsed`` shaped like the
    ``PartitionPlan`` schema → orchestrator ingests cleanly and returns
    the legacy list-of-dict shape.
  - Edge cases: ``parsed=None`` (provider didn't honor schema) → returns
    None.  Single partition → returns None (not worth parallelizing).
    Overlapping file scopes → returns None.
  - Architectural invariant: the call path does NOT invoke ``json.loads``
    on LLM output.

``FakeGateway`` records whether ``response_schema`` was passed in, so
we also assert the orchestrator opts into structured output.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from maike.atoms.llm import LLMResult, StopReason, TokenUsage
from maike.orchestrator.orchestrator import Orchestrator
from maike.orchestrator.partition_schema import PartitionPlan


class _FakeGateway:
    """Records call kwargs + returns a canned LLMResult."""

    def __init__(self, parsed: dict[str, Any] | None, content: str = "") -> None:
        self._parsed = parsed
        self._content = content
        self.last_kwargs: dict[str, Any] = {}

    async def call(self, **kwargs) -> LLMResult:
        self.last_kwargs = kwargs
        return LLMResult(
            provider="fake",
            model=kwargs.get("model", "fake-model"),
            content=self._content,
            content_blocks=[],
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(input_tokens=10, output_tokens=20),
            parsed=self._parsed,
        )


def _make_orchestrator(tmp_path: Path) -> Orchestrator:
    """Build a minimal Orchestrator for testing _plan_partitions in isolation.
    _plan_partitions only needs ``self`` access; no session store etc."""
    return Orchestrator(base_path=tmp_path)


class TestPlanPartitionsHappyPath:
    def test_returns_list_of_dicts_from_parsed(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        parsed = {
            "partitions": [
                {"subtask": "implement A", "files": ["src/a.py"]},
                {"subtask": "implement B", "files": ["src/b.py"]},
            ],
        }
        gw = _FakeGateway(parsed=parsed)
        result = asyncio.run(orch._plan_partitions(
            task="build A and B independently",
            workspace=tmp_path,
            llm_gateway=gw,
            model="gemini-3-flash-preview",
        ))
        assert result is not None
        assert len(result) == 2
        assert result[0] == {"subtask": "implement A", "files": ["src/a.py"]}
        assert result[1] == {"subtask": "implement B", "files": ["src/b.py"]}

    def test_opts_into_structured_output(self, tmp_path):
        """The orchestrator MUST pass response_schema=PartitionPlan.  This
        is the architectural contract — skipping it would silently fall
        back to free-text JSON parsing which we removed."""
        orch = _make_orchestrator(tmp_path)
        gw = _FakeGateway(parsed={"partitions": [
            {"subtask": "x", "files": ["x.py"]},
            {"subtask": "y", "files": ["y.py"]},
        ]})
        asyncio.run(orch._plan_partitions(
            task="task",
            workspace=tmp_path,
            llm_gateway=gw,
            model="fake-model",
        ))
        assert gw.last_kwargs.get("response_schema") is PartitionPlan


class TestPlanPartitionsEdgeCases:
    def test_parsed_none_returns_none(self, tmp_path):
        """Provider didn't honor schema — orchestrator gives up cleanly."""
        orch = _make_orchestrator(tmp_path)
        gw = _FakeGateway(parsed=None, content="some free-text the adapter failed to parse")
        result = asyncio.run(orch._plan_partitions(
            task="task", workspace=tmp_path, llm_gateway=gw, model="fake-model",
        ))
        assert result is None

    def test_single_partition_returns_none(self, tmp_path):
        """One partition isn't worth parallelizing — bail to single-agent."""
        orch = _make_orchestrator(tmp_path)
        gw = _FakeGateway(parsed={
            "partitions": [{"subtask": "just do it", "files": ["one.py"]}],
        })
        result = asyncio.run(orch._plan_partitions(
            task="task", workspace=tmp_path, llm_gateway=gw, model="fake-model",
        ))
        assert result is None

    def test_empty_partitions_returns_none(self, tmp_path):
        """Model said: not partitionable."""
        orch = _make_orchestrator(tmp_path)
        gw = _FakeGateway(parsed={"partitions": []})
        result = asyncio.run(orch._plan_partitions(
            task="task", workspace=tmp_path, llm_gateway=gw, model="fake-model",
        ))
        assert result is None

    def test_overlapping_files_returns_none(self, tmp_path):
        """Overlap detection: parallel agents can't both own the same file."""
        orch = _make_orchestrator(tmp_path)
        gw = _FakeGateway(parsed={
            "partitions": [
                {"subtask": "a", "files": ["src/shared.py", "a.py"]},
                {"subtask": "b", "files": ["src/shared.py", "b.py"]},  # overlap!
            ],
        })
        result = asyncio.run(orch._plan_partitions(
            task="task", workspace=tmp_path, llm_gateway=gw, model="fake-model",
        ))
        assert result is None

    def test_caps_at_five_partitions(self, tmp_path):
        """The schema's max_length=5 enforces this at validation time;
        any overflow becomes a ValidationError → returns None."""
        orch = _make_orchestrator(tmp_path)
        # 6 partitions — Pydantic validation rejects it.
        parsed = {"partitions": [
            {"subtask": f"t{i}", "files": [f"f{i}.py"]}
            for i in range(6)
        ]}
        gw = _FakeGateway(parsed=parsed)
        result = asyncio.run(orch._plan_partitions(
            task="task", workspace=tmp_path, llm_gateway=gw, model="fake-model",
        ))
        assert result is None

    def test_returns_first_five_when_exactly_five(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        parsed = {"partitions": [
            {"subtask": f"t{i}", "files": [f"f{i}.py"]}
            for i in range(5)
        ]}
        gw = _FakeGateway(parsed=parsed)
        result = asyncio.run(orch._plan_partitions(
            task="task", workspace=tmp_path, llm_gateway=gw, model="fake-model",
        ))
        assert result is not None
        assert len(result) == 5

    def test_gateway_exception_returns_none(self, tmp_path):
        orch = _make_orchestrator(tmp_path)

        class _BrokenGateway:
            async def call(self, **kwargs):
                raise RuntimeError("network down")

        result = asyncio.run(orch._plan_partitions(
            task="task", workspace=tmp_path,
            llm_gateway=_BrokenGateway(), model="fake-model",
        ))
        assert result is None


class TestNoJsonLoadsOnLLMOutputInLivePath:
    """Architectural invariant — see .claude/handoff-2026-04-17-open-items.md.

    ``json.loads`` on LLM text output is banned in the live execution path
    because it fails silently at scale.  Provider-native structured output
    (this feature) is the approved alternative.
    """

    @staticmethod
    def _collect_offending_lines() -> list[str]:
        """Scan orchestrator/ and agents/ for ``json.loads`` calls."""
        import subprocess
        # git grep -E uses POSIX ERE which does NOT support \b.  The
        # pattern ``json\.loads`` is unambiguous enough on its own — we
        # don't also appear as a prefix/suffix of other identifiers in
        # the codebase.
        proc = subprocess.run(
            [
                "git", "grep", "-n", "-E", r"json\.loads",
                "--", "maike/orchestrator/", "maike/agents/",
            ],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 1:
            return []  # git grep exits 1 when no matches — that's the goal
        # Allow the match if the line carries an explicit "# json-loads-ok:" marker.
        offenders: list[str] = []
        for line in proc.stdout.splitlines():
            if "# json-loads-ok:" in line:
                continue
            # Skip comment-only references (docstring, plain comment).
            # Extract the code portion after "path:lineno:".  If it starts
            # with "#" (pure comment line), accept.
            try:
                _path, _lineno, rest = line.split(":", 2)
            except ValueError:
                continue
            stripped = rest.strip()
            if stripped.startswith("#"):
                continue
            # Accept mentions inside quoted strings (docstrings often
            # reference the name).  Heuristic: if the line has a triple
            # quote before json.loads, skip it.
            if '"""' in stripped or "'''" in stripped:
                continue
            # Accept if ``json.loads`` is inside a regular docstring-paragraph
            # continuation (line that's indented text, not code).  Skip if
            # not followed by "(".
            if "json.loads(" not in stripped:
                continue
            offenders.append(line)
        return offenders

    def test_no_json_loads_in_orchestrator_or_agents(self):
        offenders = self._collect_offending_lines()
        assert not offenders, (
            "Found json.loads() on LLM output in live execution path:\n"
            + "\n".join(offenders)
            + "\n\nArchitectural rule: no json.loads on LLM text.  Use "
              "provider-native structured output via LLMRequest.response_schema "
              "(see maike/gateway/structured.py).  If this usage is parsing "
              "known-good non-LLM JSON (cached metadata, tool arguments with "
              "structured guarantees, etc.), tag the line with a "
              "`# json-loads-ok: <reason>` comment to exempt it."
        )
