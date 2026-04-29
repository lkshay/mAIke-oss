"""Tests for deterministic stuck-pattern detectors."""

from __future__ import annotations

from typing import Any

from maike.agents.stuck_detectors import (
    CandidateNudge,
    detect_all_stuck_patterns,
    detect_reads_without_edit,
    detect_script_introspection_loop,
)


class _Ctx:
    """Minimal context stub — only needs a ``metadata`` dict."""
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers for building synthetic conversations
# ---------------------------------------------------------------------------


def _tool_use_msg(name: str, input_args: dict | None = None, tid: str = "t") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": name,
                     "input": input_args or {}}],
    }


def _tool_result_msg(tid: str = "t", is_error: bool = False) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid,
                     "is_error": is_error, "content": "ok"}],
    }


def _read_pair(path: str, tid: str) -> list[dict]:
    return [_tool_use_msg("Read", {"path": path}, tid=tid),
            _tool_result_msg(tid=tid)]


def _bash_pair(cmd: str, tid: str) -> list[dict]:
    return [_tool_use_msg("Bash", {"cmd": cmd}, tid=tid),
            _tool_result_msg(tid=tid)]


def _edit_pair(path: str, tid: str) -> list[dict]:
    return [_tool_use_msg("Edit", {"path": path}, tid=tid),
            _tool_result_msg(tid=tid)]


# ---------------------------------------------------------------------------
# Reads-without-Edit detector
# ---------------------------------------------------------------------------


class TestDetectReadsWithoutEdit:
    def test_under_threshold_no_fire(self):
        convo = []
        for i in range(14):
            convo.extend(_read_pair("widgets.py", tid=f"t{i}"))
        assert detect_reads_without_edit(convo, threshold=15) is None

    def test_at_threshold_fires(self):
        convo = []
        for i in range(15):
            convo.extend(_read_pair("widgets.py", tid=f"t{i}"))
        c = detect_reads_without_edit(convo, threshold=15)
        assert c is not None
        assert c.kind == "reads_without_edit"
        assert c.evidence["path"] == "widgets.py"
        assert c.evidence["count"] == 15

    def test_edit_resets_counter(self):
        convo = []
        for i in range(10):
            convo.extend(_read_pair("widgets.py", tid=f"pre{i}"))
        convo.extend(_edit_pair("widgets.py", tid="edit"))
        # 10 more Reads after Edit — should NOT fire (threshold 15)
        for i in range(10):
            convo.extend(_read_pair("widgets.py", tid=f"post{i}"))
        assert detect_reads_without_edit(convo, threshold=15) is None

    def test_mixed_paths_single_not_over_threshold(self):
        # 12 on one path, 12 on another — neither individually hits 15
        convo = []
        for i in range(12):
            convo.extend(_read_pair("a.py", tid=f"a{i}"))
            convo.extend(_read_pair("b.py", tid=f"b{i}"))
        # But total = 24 > 2*15=30? No, 24 < 30 → should NOT fire.
        assert detect_reads_without_edit(convo, threshold=15) is None

    def test_very_many_reads_across_files_fires(self):
        # Total reads ≥ 2*threshold but none individually over threshold —
        # exercises the "many reads spanning files" path.  Use 8 paths × 5
        # reads = 40 total, threshold 15 → no single path triggers.
        convo = []
        for file_idx in range(8):
            for read_idx in range(5):
                convo.extend(_read_pair(f"f{file_idx}.py", tid=f"f{file_idx}r{read_idx}"))
        c = detect_reads_without_edit(convo, threshold=15)
        assert c is not None
        assert c.evidence.get("total_contiguous_reads", 0) >= 30
        # Assert we didn't trip the single-path branch
        assert "path" not in c.evidence or c.evidence.get("count") is None

    def test_already_fired_suppresses(self):
        ctx = _Ctx()
        convo = [m for i in range(15) for m in _read_pair("x.py", tid=f"t{i}")]
        # First call fires
        assert detect_reads_without_edit(convo, ctx=ctx, threshold=15) is not None
        # Second call suppresses
        assert detect_reads_without_edit(convo, ctx=ctx, threshold=15) is None

    def test_empty_conversation_no_fire(self):
        assert detect_reads_without_edit([]) is None

    def test_grep_also_counts(self):
        convo = []
        for i in range(15):
            convo.append(_tool_use_msg("Grep", {"pattern": "foo", "path": "widgets.py"}, tid=f"t{i}"))
            convo.append(_tool_result_msg(tid=f"t{i}"))
        c = detect_reads_without_edit(convo, threshold=15)
        assert c is not None
        assert c.evidence["path"] == "widgets.py"


# ---------------------------------------------------------------------------
# Script-introspection loop detector
# ---------------------------------------------------------------------------


class TestDetectScriptIntrospectionLoop:
    def test_under_threshold_no_fire(self):
        convo = []
        for i in range(7):
            convo.extend(_bash_pair('python3 -c "from django.db import models; print(models.Manager)"', tid=f"t{i}"))
        assert detect_script_introspection_loop(convo, threshold=8) is None

    def test_at_threshold_fires(self):
        convo = []
        for i in range(8):
            convo.extend(_bash_pair('python3 -c "from django.db import models; print(models.Manager)"', tid=f"t{i}"))
        c = detect_script_introspection_loop(convo, threshold=8)
        assert c is not None
        assert c.kind == "script_introspection"
        assert c.evidence["target"] == "django"
        assert c.evidence["count"] == 8

    def test_python_dash_c_with_import_x(self):
        convo = []
        for i in range(8):
            convo.extend(_bash_pair('python3 -c "import sympy; print(sympy.__version__)"', tid=f"t{i}"))
        c = detect_script_introspection_loop(convo, threshold=8)
        assert c is not None
        assert c.evidence["target"] == "sympy"

    def test_non_introspection_bash_ignored(self):
        convo = []
        for i in range(8):
            convo.extend(_bash_pair("grep foo src/bar.py", tid=f"t{i}"))
            convo.extend(_bash_pair("ls src/", tid=f"t2_{i}"))
        assert detect_script_introspection_loop(convo, threshold=8) is None

    def test_node_dash_e_loop_fires(self):
        convo = []
        for i in range(8):
            convo.extend(_bash_pair("node -e \"const x = require('./lib/parser'); console.log(x)\"", tid=f"t{i}"))
        c = detect_script_introspection_loop(convo, threshold=8)
        assert c is not None
        # Target should be extracted from require() path
        assert "parser" in c.evidence["target"] or c.evidence["target"] == "<script>"

    def test_ruby_dash_e_loop_fires(self):
        convo = []
        for i in range(8):
            convo.extend(_bash_pair("ruby -e \"require 'json'; puts JSON.parse(x)\"", tid=f"t{i}"))
        c = detect_script_introspection_loop(convo, threshold=8)
        assert c is not None
        assert c.evidence["target"] == "json"

    def test_mixed_languages_do_not_cross_count(self):
        # 4 Python + 4 Node = 8 total, but neither individual target hits threshold
        convo = []
        for i in range(4):
            convo.extend(_bash_pair("python3 -c \"import foo\"", tid=f"py{i}"))
            convo.extend(_bash_pair("node -e \"require('./bar')\"", tid=f"js{i}"))
        assert detect_script_introspection_loop(convo, threshold=8) is None

    def test_edit_resets_counter(self):
        convo = []
        for i in range(5):
            convo.extend(_bash_pair('python3 -c "from django.db import models"', tid=f"pre{i}"))
        convo.extend(_edit_pair("widgets.py", tid="edit"))
        for i in range(5):
            convo.extend(_bash_pair('python3 -c "from django.db import models"', tid=f"post{i}"))
        # 5 post-edit introspections, threshold 8 → no fire
        assert detect_script_introspection_loop(convo, threshold=8) is None

    def test_mixed_modules_not_concentrated(self):
        convo = []
        for i in range(8):
            # alternate modules
            mod = "django" if i % 2 == 0 else "sympy"
            convo.extend(_bash_pair(f'python3 -c "import {mod}"', tid=f"t{i}"))
        # django count = 4, sympy count = 4 — below threshold
        assert detect_script_introspection_loop(convo, threshold=8) is None

    def test_already_fired_suppresses(self):
        ctx = _Ctx()
        convo = [m for i in range(8) for m in _bash_pair('python3 -c "import x"', tid=f"t{i}")]
        assert detect_script_introspection_loop(convo, ctx=ctx, threshold=8) is not None
        assert detect_script_introspection_loop(convo, ctx=ctx, threshold=8) is None

    def test_empty_conversation(self):
        assert detect_script_introspection_loop([]) is None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class TestDetectAllStuckPatterns:
    def test_both_detectors_run(self):
        ctx = _Ctx()
        # Build a conversation that trips both detectors.
        convo: list[dict] = []
        for i in range(15):
            convo.extend(_read_pair("widgets.py", tid=f"r{i}"))
        for i in range(8):
            convo.extend(_bash_pair('python3 -c "from django.db import models"', tid=f"b{i}"))
        out = detect_all_stuck_patterns(convo, ctx=ctx)
        kinds = {c.kind for c in out}
        assert "reads_without_edit" in kinds
        assert "script_introspection" in kinds

    def test_none_when_nothing_pathological(self):
        convo = []
        for i in range(3):
            convo.extend(_read_pair("a.py", tid=f"r{i}"))
            convo.extend(_edit_pair("a.py", tid=f"e{i}"))
        assert detect_all_stuck_patterns(convo) == []


# ---------------------------------------------------------------------------
# CandidateNudge
# ---------------------------------------------------------------------------


class TestCandidateNudgeTaskHash:
    def test_stable_for_same_task_and_kind(self):
        a = CandidateNudge(kind="reads_without_edit", text="x").task_hash("task A")
        b = CandidateNudge(kind="reads_without_edit", text="y").task_hash("task A")
        assert a == b

    def test_differs_for_different_kind(self):
        a = CandidateNudge(kind="reads_without_edit", text="x").task_hash("task A")
        b = CandidateNudge(kind="script_introspection", text="x").task_hash("task A")
        assert a != b

    def test_differs_for_different_task(self):
        a = CandidateNudge(kind="reads_without_edit", text="x").task_hash("task A")
        b = CandidateNudge(kind="reads_without_edit", text="x").task_hash("task B")
        assert a != b
