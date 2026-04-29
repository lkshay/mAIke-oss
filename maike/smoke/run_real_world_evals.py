"""Run real-world eval cases against actual open source repos.

Usage:
    python -m maike.smoke.run_real_world_evals [--provider gemini] [--model gemini-2.5-pro] [--cases colorama-reset-bug]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from maike.smoke.real_world_cases import REAL_WORLD_CASES, RealWorldCase, _clone_repo


@dataclass
class RealWorldResult:
    name: str
    description: str
    status: str = "pending"
    passed: bool = False
    score: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    duration_seconds: float = 0.0
    workspace: str = ""
    error: str | None = None
    verification_error: str | None = None
    clone_ok: bool = False
    setup_ok: bool = False
    agent_ok: bool = False
    verify_ok: bool = False


def _run_maike_on_workspace(
    task: str,
    workspace: Path,
    provider: str,
    model: str,
    budget: float,
    verbose: bool = False,
) -> tuple[str, float, int]:
    """Run mAIke in a subprocess for full process isolation.

    Each case gets a fresh Python process so there are no event loop,
    httpx client, or module-level state leaks between cases.
    """
    # Write a small runner script that executes mAIke and prints results
    script = f'''
import asyncio, json, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from maike.cli import run_command
from maike.orchestrator.orchestrator import OrchestratorError
from maike.cost.tracker import BudgetExceededError

async def run():
    return await run_command(
        task={task!r},
        workspace=Path({str(workspace)!r}),
        provider={provider!r},
        model={model!r},
        budget={budget!r},
        yes=True,
        verbose={verbose!r},
    )

try:
    result = asyncio.run(run())
    session_id = getattr(result, "session_id", None)

    # Get cost/tokens from session store
    cost, tokens = 0.0, 0
    if session_id:
        from maike.memory.session import SessionStore
        async def get_cost():
            store = SessionStore(Path({str(workspace)!r}))
            await store.initialize()
            return await store.load_session_cost_summary(session_id)
        try:
            cs = asyncio.run(get_cost())
            if cs:
                cost = float(cs.get("total_cost_usd", 0) or 0)
                tokens = int(cs.get("total_tokens", 0) or 0)
        except Exception:
            pass

    print(json.dumps({{"status": "completed", "cost": cost, "tokens": tokens}}))
except (OrchestratorError, BudgetExceededError) as e:
    print(json.dumps({{"status": "failed", "cost": 0, "tokens": 0, "error": str(e)[:200]}}))
except Exception as e:
    print(json.dumps({{"status": "error", "cost": 0, "tokens": 0, "error": str(e)[:200]}}))
'''

    maike_root = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=maike_root,
        env={**os.environ, "PYTHONPATH": maike_root},
    )

    # Parse the JSON result from the last line of stdout
    output_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    for line in reversed(output_lines):
        try:
            data = json.loads(line)
            status = data.get("status", "error")
            error_msg = data.get("error", "")
            if error_msg and status != "completed":
                # Store error so the caller can display it
                _run_maike_on_workspace._last_error = error_msg
            return (
                status,
                float(data.get("cost", 0)),
                int(data.get("tokens", 0)),
            )
        except (json.JSONDecodeError, ValueError):
            continue

    # If we can't parse output, capture stderr
    stderr_tail = result.stderr.strip()[-300:] if result.stderr else ""
    if stderr_tail:
        _run_maike_on_workspace._last_error = stderr_tail
    if result.returncode == 0:
        return "completed", 0.0, 0
    return "failed", 0.0, 0

_run_maike_on_workspace._last_error = ""


def run_case(
    case: RealWorldCase,
    provider: str,
    model: str,
    verbose: bool = False,
    keep_workspace: bool = False,
) -> RealWorldResult:
    """Run a single real-world eval case."""
    result = RealWorldResult(name=case.name, description=case.description)
    workspace = Path(tempfile.mkdtemp(prefix=f"maike-rw-{case.name}-"))
    result.workspace = str(workspace)
    start = time.monotonic()

    try:
        # Step 1: Clone repo
        print(f"    Cloning {case.repo_url} @ {case.commit_sha}...")
        _clone_repo(case.repo_url, case.commit_sha, workspace)
        result.clone_ok = True

        # Step 2: Setup (install deps, inject bugs, etc.)
        if case.setup:
            print(f"    Setting up workspace...")
            case.setup(workspace)
        result.setup_ok = True

        # Step 3: Run mAIke agent
        print(f"    Running mAIke agent...")
        status, cost, tokens = _run_maike_on_workspace(
            case.task, workspace, provider, model, case.budget, verbose,
        )
        result.status = status
        result.cost_usd = cost
        result.tokens = tokens
        result.agent_ok = status == "completed"

        if status != "completed":
            agent_error = getattr(_run_maike_on_workspace, "_last_error", "") or ""
            result.error = f"Agent session {status}: {agent_error}" if agent_error else f"Agent session {status}"
            result.score = 0.1 if status == "failed" else 0.0
        else:
            # Step 4: Verify
            if case.verify:
                print(f"    Verifying workspace...")
                try:
                    case.verify(workspace)
                    result.verify_ok = True
                    result.passed = True
                    result.score = 1.0
                except (AssertionError, Exception) as exc:
                    result.verification_error = str(exc)
                    result.score = 0.4  # Completed but verification failed
            else:
                result.passed = True
                result.score = 0.8

    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.error = "Timed out"
    except subprocess.CalledProcessError as exc:
        result.status = "setup_error"
        result.error = f"Setup failed: {exc.stderr[:200] if exc.stderr else str(exc)}"
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)[:200]
    finally:
        elapsed = time.monotonic() - start
        result.duration_seconds = round(elapsed, 1)
        if not keep_workspace:
            try:
                shutil.rmtree(workspace, ignore_errors=True)
            except Exception:
                pass

    return result


def run_all(
    provider: str = "gemini",
    model: str = "gemini-2.5-pro",
    cases_filter: list[str] | None = None,
    verbose: bool = False,
    save: bool = False,
    keep_workspaces: bool = False,
) -> list[RealWorldResult]:
    """Run all real-world eval cases."""
    cases = REAL_WORLD_CASES
    if cases_filter:
        cases = {k: v for k, v in cases.items() if k in cases_filter}

    print(f"\n{'=' * 70}")
    print(f"Real-World Eval Suite — {provider}/{model}")
    print(f"Cases: {', '.join(cases.keys())}")
    print(f"{'=' * 70}\n")

    results: list[RealWorldResult] = []

    for name, case in cases.items():
        print(f"--- {name}: {case.description} ---")
        r = run_case(case, provider, model, verbose, keep_workspaces)
        results.append(r)

        icon = "✓" if r.passed else "✗"
        stages = []
        if r.clone_ok:
            stages.append("clone ✓")
        if r.setup_ok:
            stages.append("setup ✓")
        if r.agent_ok:
            stages.append("agent ✓")
        else:
            stages.append(f"agent ✗ ({r.status})")
        if r.verify_ok:
            stages.append("verify ✓")
        elif r.verification_error:
            stages.append("verify ✗")

        print(f"  {icon} {r.name}: {' → '.join(stages)}")
        print(f"    score={r.score:.1f} | ${r.cost_usd:.4f} | {r.tokens:,} tokens | {r.duration_seconds}s")
        if r.verification_error:
            print(f"    Verification: {r.verification_error[:150]}")
        if r.error:
            print(f"    Error: {r.error[:150]}")
        print()

    # Summary
    total_passed = sum(1 for r in results if r.passed)
    total_cost = sum(r.cost_usd for r in results)
    total_tokens = sum(r.tokens for r in results)

    print(f"{'=' * 70}")
    print(f"RESULTS: {total_passed}/{len(results)} passed")
    print(f"Total cost: ${total_cost:.4f} | Total tokens: {total_tokens:,}")
    print(f"{'=' * 70}")

    for r in results:
        icon = "✓" if r.passed else "✗"
        print(f"  {icon} {r.name}: score={r.score:.1f}, ${r.cost_usd:.4f}, {r.tokens:,} tokens, {r.duration_seconds}s")

    if save:
        results_dir = Path(__file__).parent / "real_world_results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"rw_{provider}_{model}_{timestamp}.json"
        result_data = {
            "provider": provider,
            "model": model,
            "timestamp": timestamp,
            "total_passed": total_passed,
            "total_cases": len(results),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "results": [asdict(r) for r in results],
        }
        result_file.write_text(json.dumps(result_data, indent=2))
        print(f"\nResults saved to: {result_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run real-world eval cases")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--cases", default=None, help="Comma-separated case names")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--keep-workspaces", action="store_true")
    args = parser.parse_args()

    cases_filter = args.cases.split(",") if args.cases else None
    run_all(
        provider=args.provider,
        model=args.model,
        cases_filter=cases_filter,
        verbose=args.verbose,
        save=args.save,
        keep_workspaces=args.keep_workspaces,
    )


if __name__ == "__main__":
    main()
