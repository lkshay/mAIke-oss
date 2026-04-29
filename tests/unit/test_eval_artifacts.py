from pathlib import Path

from maike.eval.baselines import baseline_from_report, baseline_path, load_baseline, write_baseline
from maike.eval.contracts import EvalReport
from maike.eval.reporting import build_run_metadata, canonical_suite_key, write_report


def test_canonical_suite_key_normalizes_aliases_and_custom_lists():
    assert canonical_suite_key("control", ["greenfield", "editing"]) == "smoke"
    assert canonical_suite_key("semi-real", ["editing-multifile"]) == "dynamic"
    assert canonical_suite_key("editing,greenfield", ["greenfield", "editing"]) == "custom__editing+greenfield"


def test_write_report_persists_report_path_and_loads_baseline_roundtrip(tmp_path):
    report = EvalReport(
        schema_version=1,
        mode="baseline",
        suite="core",
        suite_key="core",
        provider="gemini",
        model="gemini-2.5-flash",
        budget=2.5,
        created_at="2026-03-18T12:00:00+00:00",
        git_sha="abc123",
        report_path=None,
        baseline_path=str(baseline_path(tmp_path, suite_key="core", provider="gemini", model="gemini-2.5-flash")),
        total=1,
        passed=1,
        failed=0,
        average_score=1.0,
        execution_failed=False,
        regression_failed=False,
        warning_count=0,
        results=[],
    )

    written_report = write_report(report, workspace_root=tmp_path)

    assert written_report.report_path is not None
    assert Path(written_report.report_path).exists()
    assert '"report_path"' in Path(written_report.report_path).read_text(encoding="utf-8")

    baseline = baseline_from_report(written_report, policy={"schema_version": 1})
    baseline = write_baseline(baseline, workspace_root=tmp_path)
    loaded = load_baseline(tmp_path, suite_key="core", provider="gemini", model="gemini-2.5-flash")

    assert baseline.baseline_path == loaded.baseline_path
    assert loaded.schema_version == 1
    assert loaded.report is not None
    assert loaded.report.report_path == written_report.report_path


def test_build_run_metadata_carries_expected_fields(tmp_path):
    metadata = build_run_metadata(
        suite="core",
        suite_key="core",
        provider="gemini",
        model="gemini-2.5-flash",
        budget=2.5,
        workspace_root=tmp_path,
    )

    assert metadata.suite == "core"
    assert metadata.suite_key == "core"
    assert metadata.provider == "gemini"
    assert metadata.model == "gemini-2.5-flash"
    assert metadata.budget == 2.5


def test_build_run_metadata_creates_missing_workspace_root(tmp_path):
    workspace_root = tmp_path / "fresh" / "eval-root"

    metadata = build_run_metadata(
        suite="all",
        suite_key="all",
        provider="gemini",
        model="gemini-2.5-flash",
        budget=5.0,
        workspace_root=workspace_root,
    )

    assert workspace_root.exists()
    assert metadata.git_sha is None
