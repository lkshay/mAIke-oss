"""Helpers for eval report metadata and JSON persistence."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from maike.eval.contracts import EvalReport, EvalRunMetadata
from maike.utils import utcnow


def eval_storage_root(workspace_root: Path) -> Path:
    return workspace_root / ".maike-eval"


def reports_root(workspace_root: Path) -> Path:
    return eval_storage_root(workspace_root) / "reports"


def sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._") or "value"


def canonical_suite_key(requested_suite: str, workflow_names: list[str]) -> str:
    normalized = requested_suite.strip().lower()
    aliases = {
        "smoke": "smoke",
        "control": "smoke",
        "core": "core",
        "dynamic": "dynamic",
        "semi-real": "dynamic",
        "tier1": "tier1",
        "tier2": "tier2",
        "tier3": "tier3",
        "tier4": "tier4",
        "all": "all",
        "agentic": "agentic",
        "hard-agentic": "hard-agentic",
    }
    if normalized in aliases:
        return aliases[normalized]
    ordered = sorted({name.strip().lower() for name in workflow_names if name.strip()})
    return f"custom__{'+'.join(ordered)}"


def build_run_metadata(
    *,
    suite: str,
    suite_key: str,
    provider: str,
    model: str,
    budget: float,
    workspace_root: Path,
    report_path: Path | None = None,
    baseline_path: Path | None = None,
) -> EvalRunMetadata:
    workspace_root.mkdir(parents=True, exist_ok=True)
    return EvalRunMetadata(
        suite=suite,
        suite_key=suite_key,
        provider=provider,
        model=model,
        budget=budget,
        created_at=utcnow().isoformat(),
        git_sha=_git_sha(workspace_root),
        report_path=str(report_path) if report_path is not None else None,
        baseline_path=str(baseline_path) if baseline_path is not None else None,
    )


def default_report_path(workspace_root: Path, metadata: EvalRunMetadata) -> Path:
    suite_component = sanitize_path_component(metadata.suite_key)
    # Truncate long suite keys (e.g. custom suites listing all case names) to
    # stay within filesystem filename limits (~255 bytes on most systems).
    if len(suite_component) > 80:
        import hashlib
        digest = hashlib.sha256(suite_component.encode()).hexdigest()[:8]
        suite_component = f"{suite_component[:70]}-{digest}"
    filename = "--".join(
        [
            sanitize_path_component(_timestamp_token(metadata.created_at)),
            suite_component,
            sanitize_path_component(metadata.provider),
            sanitize_path_component(metadata.model),
        ]
    )
    return reports_root(workspace_root) / f"{filename}.json"


def write_report(
    report: EvalReport,
    *,
    workspace_root: Path,
    output_path: Path | None = None,
) -> EvalReport:
    destination = output_path or default_report_path(
        workspace_root,
        EvalRunMetadata(
            suite=report.suite,
            suite_key=report.suite_key,
            provider=report.provider,
            model=report.model,
            budget=report.budget,
            created_at=report.created_at,
            git_sha=report.git_sha,
            report_path=report.report_path,
            baseline_path=report.baseline_path,
        ),
    )
    final_report = replace(report, report_path=str(destination))
    _write_json_atomic(destination, asdict(final_report))
    return final_report


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(obj: object) -> str:
    """Fallback serializer for ``json.dump``."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            temp_path = Path(handle.name)
        assert temp_path is not None
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _git_sha(workspace_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _timestamp_token(iso_value: str) -> str:
    return iso_value.replace(":", "").replace("+00:00", "Z")
