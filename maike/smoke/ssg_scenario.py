"""Multi-phase SSG (Static Site Generator) scenario for smoke testing.

Tests the full lifecycle of a real project built with mAIke's react pipeline:
  Phase 1: Build a static site generator from scratch
  Phase 2: Explain the code (read-only — should not modify files)
  Phase 3: Rate test quality (read-only — should not modify files)
  Phase 4: Add comprehensive unit tests

This exercises:
  - Bootstrap: venv creation, dep installation
  - MAIKE.md: generation and consumption across sessions
  - Behavior-based question detection: phases 2-3 should be read-only
  - Repeated failure tracker: phase 4 should not spiral
  - Checkpoint preservation: venv should survive across phases

Usage:
    python -m maike.smoke.ssg_scenario [--provider gemini] [--model gemini-2.5-flash] [--budget 10]

Or from Python:
    from maike.smoke.ssg_scenario import run_ssg_scenario
    results = run_ssg_scenario(provider="gemini", model="gemini-2.5-flash")
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from maike.cli import run_command
from maike.gateway.providers import resolve_model_name
from maike.smoke.workflows import default_live_provider, load_session_snapshot


@dataclass
class PhaseResult:
    """Result of a single scenario phase."""

    phase: int
    name: str
    task: str
    session_id: str
    status: str             # "completed" | "failed"
    cost_usd: float
    tokens_used: int
    files_modified: int     # 0 for read-only phases
    tests_passing: int
    tests_total: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "completed" and not self.errors


@dataclass
class SSGScenarioResult:
    """Aggregate result of the full SSG scenario."""

    provider: str
    model: str
    workspace: str
    timestamp: str
    phases: list[PhaseResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    @property
    def all_passed(self) -> bool:
        return all(p.passed for p in self.phases)

    def summary(self) -> str:
        lines = [
            f"SSG Scenario: {'PASS' if self.all_passed else 'FAIL'}",
            f"Provider: {self.provider} / {self.model}",
            f"Total cost: ${self.total_cost_usd:.4f}",
            f"Total tokens: {self.total_tokens:,}",
            "",
        ]
        for p in self.phases:
            status = "✓" if p.passed else "✗"
            lines.append(
                f"  {status} Phase {p.phase}: {p.name} "
                f"(${p.cost_usd:.4f}, {p.tokens_used:,} tokens, "
                f"{p.tests_passing}/{p.tests_total} tests, "
                f"{p.files_modified} files modified)"
            )
            for err in p.errors:
                lines.append(f"    ERROR: {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

_PHASES = [
    {
        "name": "build",
        "task": (
            "Build a Python static site generator CLI called 'forge'. "
            "1) Read markdown files from content/ directory. "
            "2) Parse YAML frontmatter (title, date, tags, template). "
            "3) Convert markdown to HTML. "
            "4) Render through Jinja2 templates from templates/ directory. "
            "5) Output to build/ preserving structure. "
            "6) CLI: 'forge build' and 'forge clean'. "
            "Use stdlib + markdown + jinja2 + pyyaml. "
            "Include pyproject.toml with [tool.setuptools] packages=['forge']. "
            "Write tests and verify end-to-end."
        ),
        "budget": 10.0,
        "expect_read_only": False,
    },
    {
        "name": "explain",
        "task": (
            "Can you explain the code we have generated so far? "
            "Walk me through the architecture and how the build process works."
        ),
        "budget": 5.0,
        "expect_read_only": True,
    },
    {
        "name": "rate-tests",
        "task": (
            "Are the existing unit tests covering the code well? "
            "Rate the test quality and tell me what's missing."
        ),
        "budget": 5.0,
        "expect_read_only": True,
    },
    {
        "name": "add-tests",
        "task": (
            "Add comprehensive unit tests for the forge project. "
            "Test the frontmatter parser edge cases, markdown-to-HTML conversion, "
            "template rendering, directory structure preservation in build output, "
            "and the CLI commands. Make sure all tests pass."
        ),
        "budget": 10.0,
        "expect_read_only": False,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_file_artifacts(workspace: Path, session_id: str) -> int:
    """Count file artifacts that were written (not just read) in a session."""
    import sqlite3
    db_path = workspace / ".maike" / "session.db"
    if not db_path.exists():
        return -1
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT COUNT(DISTINCT logical_name) FROM artifacts "
            "WHERE session_id = ? AND kind = 'file'",
            (session_id,),
        )
        return cursor.fetchone()[0]
    finally:
        conn.close()


def _run_tests_in_workspace(workspace: Path) -> tuple[int, int]:
    """Run pytest in the workspace and return (passing, total)."""
    # Ensure venv and deps
    venv_python = workspace / ".venv" / "bin" / "python"
    if not venv_python.exists():
        subprocess.run(
            ["python3", "-m", "venv", ".venv"],
            cwd=workspace, capture_output=True,
        )
        subprocess.run(
            [str(workspace / ".venv" / "bin" / "pip"), "install", "-e", ".", "pytest", "-q"],
            cwd=workspace, capture_output=True,
        )

    result = subprocess.run(
        [str(venv_python), "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
        cwd=workspace, capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr

    # Parse pytest summary: "23 passed" or "3 failed, 20 passed"
    import re
    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    error_match = re.search(r"(\d+) error", output)

    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(error_match.group(1)) if error_match else 0
    total = passed + failed + errors

    return passed, total


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ssg_scenario(
    *,
    provider: str | None = None,
    model: str | None = None,
    budget_override: float | None = None,
    workspace: Path | None = None,
    verbose: bool = False,
    phases: list[int] | None = None,
) -> SSGScenarioResult:
    """Run the multi-phase SSG scenario.

    Args:
        provider: LLM provider (default: auto-detect from env)
        model: Model name (default: provider default)
        budget_override: Override per-phase budget
        workspace: Workspace path (default: temp dir)
        verbose: Print verbose output
        phases: Which phases to run (1-indexed). None = all.

    Returns:
        SSGScenarioResult with per-phase outcomes
    """
    resolved_provider = provider or default_live_provider()
    resolved_model = resolve_model_name(resolved_provider, model)

    # Always use a temp directory unless explicitly overridden.
    # Guard against accidentally dirtying the mAIke repo.
    if workspace is not None:
        ws = workspace.resolve()
        maike_root = Path(__file__).resolve().parents[2]
        if ws == maike_root or maike_root in ws.parents:
            raise ValueError(
                f"Workspace {ws} is inside the mAIke repo ({maike_root}). "
                f"Use a path outside the repo or omit --workspace to use a temp directory."
            )
    else:
        ws = Path(tempfile.mkdtemp(prefix="maike-ssg-scenario-"))

    if not (ws / ".git").exists():
        subprocess.run(["git", "init"], cwd=ws, capture_output=True)

    print(f"SSG Scenario workspace: {ws}")

    result = SSGScenarioResult(
        provider=resolved_provider,
        model=resolved_model,
        workspace=str(ws),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    phases_to_run = phases or list(range(1, len(_PHASES) + 1))

    for phase_idx in phases_to_run:
        phase_def = _PHASES[phase_idx - 1]
        budget = budget_override or phase_def["budget"]

        print(f"\n{'='*60}")
        print(f"Phase {phase_idx}: {phase_def['name']}")
        print(f"{'='*60}")

        import time
        start = time.monotonic()
        errors: list[str] = []

        try:
            run_result = asyncio.run(
                run_command(
                    task=phase_def["task"],
                    workspace=ws,
                    provider=resolved_provider,
                    model=resolved_model,
                    language="python",
                    budget=budget,
                    yes=True,
                    verbose=verbose,
                )
            )
            session_id = run_result.session_id
            status = "completed"

            # Validate read-only phases
            if phase_def["expect_read_only"]:
                files_modified = _count_file_artifacts(ws, session_id)
                if files_modified > 0:
                    # File artifacts include reads tracked as artifacts,
                    # so check if any are actual writes by looking at versions > 1
                    import sqlite3
                    db_path = ws / ".maike" / "session.db"
                    conn = sqlite3.connect(str(db_path))
                    try:
                        cursor = conn.execute(
                            "SELECT COUNT(*) FROM artifacts "
                            "WHERE session_id = ? AND kind = 'file' AND version > 1",
                            (session_id,),
                        )
                        write_count = cursor.fetchone()[0]
                    finally:
                        conn.close()
                    if write_count > 0:
                        errors.append(
                            f"Read-only phase modified {write_count} file(s)"
                        )

        except Exception as exc:
            session_id = "unknown"
            status = "failed"
            errors.append(str(exc))

        elapsed = time.monotonic() - start
        files_modified = _count_file_artifacts(ws, session_id) if session_id != "unknown" else 0
        tests_passing, tests_total = _run_tests_in_workspace(ws)

        # Get cost/tokens from session DB
        cost_usd = 0.0
        tokens_used = 0
        if session_id != "unknown":
            try:
                snapshot = load_session_snapshot(ws, session_id)
                for run in snapshot.agent_runs:
                    cost_usd += run.get("cost_usd", 0)
                    tokens_used += run.get("tokens_used", 0)
            except Exception:
                pass

        phase_result = PhaseResult(
            phase=phase_idx,
            name=phase_def["name"],
            task=phase_def["task"][:100],
            session_id=session_id,
            status=status,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            files_modified=files_modified,
            tests_passing=tests_passing,
            tests_total=tests_total,
            duration_seconds=round(elapsed, 1),
            errors=errors,
        )
        result.phases.append(phase_result)
        result.total_cost_usd += cost_usd
        result.total_tokens += tokens_used

        print(f"  Status: {status}")
        print(f"  Cost: ${cost_usd:.4f} / Tokens: {tokens_used:,}")
        print(f"  Tests: {tests_passing}/{tests_total} passing")
        print(f"  Duration: {elapsed:.1f}s")
        if errors:
            for e in errors:
                print(f"  ERROR: {e}")

        # Stop if a phase fails hard
        if status == "failed" and not phase_def["expect_read_only"]:
            print(f"\nPhase {phase_idx} failed — stopping scenario.")
            break

    return result


def save_scenario_result(result: SSGScenarioResult, output_dir: Path | None = None) -> Path:
    """Save scenario result as JSON for regression tracking."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "ssg_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"ssg_{result.provider}_{result.model}_{timestamp}.json"
    output_path = output_dir / filename

    # Convert to serializable dict
    data = {
        "provider": result.provider,
        "model": result.model,
        "workspace": result.workspace,
        "timestamp": result.timestamp,
        "total_cost_usd": result.total_cost_usd,
        "total_tokens": result.total_tokens,
        "all_passed": result.all_passed,
        "phases": [asdict(p) for p in result.phases],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SSG multi-phase scenario")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--phases", type=str, default=None,
                        help="Comma-separated phase numbers to run (e.g. 1,4)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to maike/smoke/ssg_results/")
    args = parser.parse_args()

    phase_list = None
    if args.phases:
        phase_list = [int(p.strip()) for p in args.phases.split(",")]

    result = run_ssg_scenario(
        provider=args.provider,
        model=args.model,
        budget_override=args.budget,
        workspace=args.workspace,
        verbose=args.verbose,
        phases=phase_list,
    )

    print(f"\n{'='*60}")
    print(result.summary())
    print(f"{'='*60}")

    if args.save:
        path = save_scenario_result(result)
        print(f"\nResults saved to: {path}")
