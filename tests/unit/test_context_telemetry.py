"""Tests for maike.context.telemetry.ContextTelemetry."""

from maike.context.telemetry import ContextTelemetry


class TestContextTelemetryReport:
    def test_report_returns_all_fields(self):
        t = ContextTelemetry()
        report = t.report()
        expected_keys = {
            "initial_context_tokens",
            "workspace_snapshot_tokens",
            "artifact_tokens",
            "summarized_artifacts",
            "compression_applied",
            "compression_levels_used",
            "peak_conversation_tokens",
            "prune_events",
            "tokens_pruned",
            "on_demand_fetches",
            "convergence_nudge_injected",
            "tool_call_summary",
        }
        assert set(report.keys()) == expected_keys

    def test_report_default_values(self):
        report = ContextTelemetry().report()
        assert report["initial_context_tokens"] == 0
        assert report["workspace_snapshot_tokens"] == 0
        assert report["artifact_tokens"] == {}
        assert report["summarized_artifacts"] == []
        assert report["compression_applied"] is False
        assert report["compression_levels_used"] == []
        assert report["peak_conversation_tokens"] == 0
        assert report["prune_events"] == 0
        assert report["tokens_pruned"] == 0
        assert report["on_demand_fetches"] == 0

    def test_report_returns_copies(self):
        t = ContextTelemetry()
        t.artifact_tokens["spec.md"] = 100
        t.summarized_artifacts.append("spec.md")
        t.compression_levels_used.append("level1")
        report = t.report()
        # Mutating the report should not affect the telemetry.
        report["artifact_tokens"]["new"] = 999
        report["summarized_artifacts"].append("extra")
        report["compression_levels_used"].append("extra")
        assert "new" not in t.artifact_tokens
        assert "extra" not in t.summarized_artifacts
        assert "extra" not in t.compression_levels_used


class TestRecordPrune:
    def test_single_prune(self):
        t = ContextTelemetry()
        t.record_prune(1000, 800)
        assert t.prune_events == 1
        assert t.tokens_pruned == 200

    def test_multiple_prunes_accumulate(self):
        t = ContextTelemetry()
        t.record_prune(1000, 800)
        t.record_prune(900, 600)
        assert t.prune_events == 2
        assert t.tokens_pruned == 500

    def test_prune_no_negative_tokens(self):
        t = ContextTelemetry()
        t.record_prune(100, 200)
        assert t.tokens_pruned == 0
        assert t.prune_events == 1


class TestRecordCompression:
    def test_single_compression(self):
        t = ContextTelemetry()
        t.record_compression(["strip_tool_descriptions"])
        assert t.compression_applied is True
        assert t.compression_levels_used == ["strip_tool_descriptions"]

    def test_multiple_compressions_append(self):
        t = ContextTelemetry()
        t.record_compression(["level1", "level2"])
        t.record_compression(["level3"])
        assert t.compression_levels_used == ["level1", "level2", "level3"]

    def test_empty_levels_still_sets_flag(self):
        t = ContextTelemetry()
        t.record_compression([])
        assert t.compression_applied is True


class TestUpdatePeak:
    def test_tracks_maximum(self):
        t = ContextTelemetry()
        t.update_peak(100)
        t.update_peak(500)
        t.update_peak(300)
        assert t.peak_conversation_tokens == 500

    def test_initial_zero(self):
        t = ContextTelemetry()
        assert t.peak_conversation_tokens == 0

    def test_single_update(self):
        t = ContextTelemetry()
        t.update_peak(42)
        assert t.peak_conversation_tokens == 42


class TestRecordFetch:
    def test_increment(self):
        t = ContextTelemetry()
        t.record_fetch()
        t.record_fetch()
        t.record_fetch()
        assert t.on_demand_fetches == 3


class TestRecordToolCall:
    def test_single_tool_call(self):
        t = ContextTelemetry()
        t.record_tool_call("read_file", input_chars=50, output_chars=5000, compressed_chars=3000)
        assert len(t.tool_calls) == 1
        assert t.tool_calls[0].tool_name == "read_file"
        assert t.tool_calls[0].output_chars == 5000
        assert t.tool_calls[0].compressed_chars == 3000
        assert t.total_tool_output_chars == 5000
        assert t.total_compressed_chars == 3000

    def test_multiple_tool_calls_accumulate(self):
        t = ContextTelemetry()
        t.record_tool_call("read_file", input_chars=30, output_chars=3000, compressed_chars=2000)
        t.record_tool_call("grep_codebase", input_chars=20, output_chars=1000, compressed_chars=800)
        t.record_tool_call("read_file", input_chars=30, output_chars=4000, compressed_chars=2500)
        assert len(t.tool_calls) == 3
        assert t.total_tool_output_chars == 8000
        assert t.total_compressed_chars == 5300

    def test_tool_call_summary_in_report(self):
        t = ContextTelemetry()
        t.record_tool_call("read_file", input_chars=30, output_chars=3000, compressed_chars=2000)
        t.record_tool_call("read_file", input_chars=30, output_chars=4000, compressed_chars=2500)
        t.record_tool_call("grep_codebase", input_chars=20, output_chars=1000, compressed_chars=800)
        report = t.report()
        summary = report["tool_call_summary"]
        assert summary["total_calls"] == 3
        assert summary["total_output_chars"] == 8000
        assert summary["total_compressed_chars"] == 5300
        assert summary["by_tool"]["read_file"]["count"] == 2
        assert summary["by_tool"]["read_file"]["output_chars"] == 7000
        assert summary["by_tool"]["grep_codebase"]["count"] == 1


class TestTelemetryIntegration:
    def test_full_lifecycle(self):
        t = ContextTelemetry()
        t.initial_context_tokens = 5000
        t.workspace_snapshot_tokens = 200
        t.artifact_tokens = {"spec.md": 2000, "plan.md": 1500}
        t.summarized_artifacts = ["spec.md"]
        t.update_peak(5000)
        t.record_prune(5000, 3000)
        t.update_peak(6000)
        t.record_compression(["strip_descriptions", "truncate_artifacts"])
        t.record_fetch()

        report = t.report()
        assert report["initial_context_tokens"] == 5000
        assert report["workspace_snapshot_tokens"] == 200
        assert report["artifact_tokens"] == {"spec.md": 2000, "plan.md": 1500}
        assert report["summarized_artifacts"] == ["spec.md"]
        assert report["compression_applied"] is True
        assert report["compression_levels_used"] == ["strip_descriptions", "truncate_artifacts"]
        assert report["peak_conversation_tokens"] == 6000
        assert report["prune_events"] == 1
        assert report["tokens_pruned"] == 2000
        assert report["on_demand_fetches"] == 1
