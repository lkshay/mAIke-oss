"""Persistence helpers for eval baselines."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from maike.eval.contracts import EvalBaseline, EvalReport, eval_baseline_from_dict
from maike.eval.reporting import eval_storage_root, load_json, sanitize_path_component


def baselines_root(workspace_root: Path) -> Path:
    return eval_storage_root(workspace_root) / "baselines"


def baseline_path(
    workspace_root: Path,
    *,
    suite_key: str,
    provider: str,
    model: str,
) -> Path:
    return (
        baselines_root(workspace_root)
        / sanitize_path_component(suite_key)
        / sanitize_path_component(provider)
        / f"{sanitize_path_component(model)}.json"
    )


def load_baseline(
    workspace_root: Path,
    *,
    suite_key: str,
    provider: str,
    model: str,
) -> EvalBaseline:
    path = baseline_path(
        workspace_root,
        suite_key=suite_key,
        provider=provider,
        model=model,
    )
    data = load_json(path)
    baseline = eval_baseline_from_dict(data)
    return replace(baseline, baseline_path=str(path))


def write_baseline(
    baseline: EvalBaseline,
    *,
    workspace_root: Path,
) -> EvalBaseline:
    path = baseline_path(
        workspace_root,
        suite_key=baseline.suite_key,
        provider=baseline.provider,
        model=baseline.model,
    )
    from maike.eval.reporting import _write_json_atomic  # local import keeps helper private-ish

    _write_json_atomic(path, asdict(baseline))
    return replace(baseline, baseline_path=str(path))


def baseline_from_report(
    report: EvalReport,
    *,
    policy: dict[str, object],
) -> EvalBaseline:
    baseline_path_value = report.baseline_path or ""
    return EvalBaseline(
        schema_version=report.schema_version,
        captured_at=report.created_at,
        suite=report.suite,
        suite_key=report.suite_key,
        provider=report.provider,
        model=report.model,
        baseline_path=baseline_path_value,
        policy=policy,
        report=report,
    )
