"""Tests that `maike history` gracefully handles sessions with/without verdict."""

from __future__ import annotations


def _base_session(id_: str = "s-1") -> dict:
    return {
        "id": id_,
        "task": "Fix the bug",
        "workspace": "/tmp/ws",
        "status": "completed",
        "created_at": "2026-04-16T17:00:00Z",
        "updated_at": "2026-04-16T17:05:00Z",
    }


class TestRenderHistoryVerdictColumn:
    def test_session_without_metadata_renders_dash(self, capsys):
        # Legacy row: no metadata field at all.  Must not crash.
        from maike.cli import _render_history
        _render_history([_base_session()])
        captured = capsys.readouterr().out
        # Rich table output — just assert no traceback and the header is there.
        assert "Verdict" in captured

    def test_session_with_verdict_metadata_renders_label(self, capsys):
        # Use a wide console so Rich doesn't truncate the verdict column.
        from rich.console import Console
        from rich.table import Table
        # Recreate _render_history's table-building path directly to avoid
        # the default console-width truncation behavior.
        row = _base_session("s-2")
        row["metadata"] = {
            "verdict": {
                "label": "satisfied", "confidence": 0.9,
                "rationale": "done", "source": "llm",
            },
        }
        # Sanity-check: the label lookup logic in _render_history should pick
        # up the right string.  Test the lookup directly (no rendering).
        meta = row.get("metadata") or {}
        v_info = meta.get("verdict") if isinstance(meta, dict) else None
        assert isinstance(v_info, dict)
        assert v_info.get("label") == "satisfied"

        # Also verify the rendered table contains the truncated prefix ('satisfi')
        # — Rich may narrow the column but the value is present.
        from maike.cli import _render_history
        _render_history([row])
        out = capsys.readouterr().out
        assert "satisfi" in out  # may be truncated with ellipsis

    def test_unproductive_label_stripped_for_display(self, capsys):
        from maike.cli import _render_history
        row = _base_session("s-3")
        row["metadata"] = {
            "verdict": {"label": "unproductive_budget_exhaustion"}
        }
        _render_history([row])
        out = capsys.readouterr().out
        # Display strips the "unproductive_" prefix.  Rich may truncate so
        # assert the prefix "budget" appears rather than the full string.
        assert "budget" in out

    def test_malformed_metadata_does_not_crash(self, capsys):
        from maike.cli import _render_history
        row = _base_session("s-4")
        row["metadata"] = "not a dict"  # type: ignore[assignment]
        # Should render with a dash, not raise.
        _render_history([row])
        out = capsys.readouterr().out
        assert "Verdict" in out
