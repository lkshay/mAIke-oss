"""SWE-bench batch runner for mAIke.

Loads SWE-bench instances, runs the mAIke orchestrator on each, captures
git diffs, and writes a predictions JSONL file for the SWE-bench harness.

Usage::

    runner = SWEBenchRunner()
    report = await runner.run(
        variant="lite",
        provider="ollama",
        model="gemma4:26b",
    )
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from maike.eval.swebench import (
    Prediction,
    SWEBenchInstance,
    append_prediction,
    capture_patch,
    cleanup_workspace,
    load_completed_ids,
    load_dataset,
    setup_workspace,
)

logger = logging.getLogger(__name__)


@dataclass
class InstanceResult:
    """Result from running one SWE-bench instance."""

    instance_id: str
    patch_lines: int = 0
    cost_usd: float = 0.0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None
    skipped: bool = False


@dataclass
class SWEBenchReport:
    """Summary of a SWE-bench batch run."""

    variant: str
    provider: str
    model: str
    total_instances: int = 0
    attempted: int = 0
    patches_generated: int = 0  # Non-empty patches
    skipped: int = 0
    errors: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_elapsed_seconds: float = 0.0
    predictions_path: str = ""
    results: list[InstanceResult] = field(default_factory=list)


class SWEBenchRunner:
    """Runs mAIke on SWE-bench instances and collects predictions."""

    def __init__(
        self,
        workspace_root: Path = Path.cwd(),
        keep_workspaces: bool = False,
    ) -> None:
        self._workspace_root = workspace_root
        self._keep_workspaces = keep_workspaces

    async def run(
        self,
        *,
        variant: str = "lite",
        provider: str = "ollama",
        model: str = "gemma4:26b",
        budget_per_instance: float = 0.0,
        timeout_per_instance: int = 300,
        max_instances: int | None = None,
        instance_ids: list[str] | None = None,
        output_path: Path | None = None,
        resume_path: Path | None = None,
        auto_approve: bool = True,
        advisor_enabled: bool = False,
        advisor_provider: str | None = None,
        advisor_model: str | None = None,
        advisor_budget_pct: float | None = None,
    ) -> SWEBenchReport:
        """Run SWE-bench evaluation.

        Args:
            variant: ``"lite"``, ``"verified"``, or ``"full"``.
            provider: LLM provider name.
            model: Model name.
            budget_per_instance: USD budget per instance (0 = unlimited).
            timeout_per_instance: Seconds before giving up on an instance.
            max_instances: Cap the number of instances to run.
            instance_ids: Only run these specific instance IDs.
            output_path: Path for predictions JSONL.
            resume_path: Path to existing predictions to skip.
            auto_approve: Auto-approve tool calls (default True for eval).

        Returns:
            SWEBenchReport with results for each instance.
        """
        # Resolve output path.
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = self._workspace_root / f"swebench-{variant}-{model.replace(':', '-')}-{timestamp}.jsonl"

        # Load dataset.
        instances = load_dataset(variant=variant, instance_ids=instance_ids)
        if max_instances and len(instances) > max_instances:
            instances = instances[:max_instances]

        # Resume support — skip already-completed instances.
        completed_ids: set[str] = set()
        if resume_path and resume_path.exists():
            completed_ids = load_completed_ids(resume_path)
            # Copy existing predictions to output if different file.
            if resume_path != output_path:
                import shutil
                shutil.copy2(resume_path, output_path)
            logger.info("Resuming: %d instances already completed", len(completed_ids))

        report = SWEBenchReport(
            variant=variant,
            provider=provider,
            model=model,
            total_instances=len(instances),
            predictions_path=str(output_path),
        )

        model_label = f"{provider}/{model}"
        _write_progress(f"\nSWE-bench {variant} — {len(instances)} instances — {model_label}\n")

        clone_cache = self._workspace_root / ".swebench-repos"

        for idx, instance in enumerate(instances, 1):
            if instance.instance_id in completed_ids:
                report.skipped += 1
                report.results.append(InstanceResult(
                    instance_id=instance.instance_id, skipped=True,
                ))
                continue

            result = await self._run_instance(
                instance=instance,
                provider=provider,
                model=model,
                budget=budget_per_instance,
                timeout=timeout_per_instance,
                output_path=output_path,
                clone_cache=clone_cache,
                auto_approve=auto_approve,
                index=idx,
                total=len(instances),
                advisor_enabled=advisor_enabled,
                advisor_provider=advisor_provider,
                advisor_model=advisor_model,
                advisor_budget_pct=advisor_budget_pct,
            )
            report.results.append(result)
            report.attempted += 1
            if result.error:
                report.errors += 1
            if result.patch_lines > 0:
                report.patches_generated += 1
            report.total_cost_usd += result.cost_usd
            report.total_tokens += result.tokens_used
            report.total_elapsed_seconds += result.elapsed_seconds

        # Summary.
        _write_progress(
            f"\n{'='*60}\n"
            f"SWE-bench {variant} complete\n"
            f"  Attempted: {report.attempted}/{report.total_instances}\n"
            f"  Patches generated: {report.patches_generated}\n"
            f"  Errors: {report.errors}\n"
            f"  Skipped (resumed): {report.skipped}\n"
            f"  Total cost: ${report.total_cost_usd:.4f}\n"
            f"  Total tokens: {report.total_tokens:,}\n"
            f"  Predictions: {output_path}\n"
            f"{'='*60}\n"
        )

        return report

    async def _run_instance(
        self,
        *,
        instance: SWEBenchInstance,
        provider: str,
        model: str,
        budget: float,
        timeout: int,
        output_path: Path,
        clone_cache: Path,
        auto_approve: bool,
        index: int,
        total: int,
        advisor_enabled: bool = False,
        advisor_provider: str | None = None,
        advisor_model: str | None = None,
        advisor_budget_pct: float | None = None,
    ) -> InstanceResult:
        """Run a single SWE-bench instance."""
        start = time.monotonic()
        _write_progress(f"[{index}/{total}] {instance.instance_id} ... ")

        # Setup workspace.
        try:
            workspace = setup_workspace(
                instance, self._workspace_root, clone_cache=clone_cache,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            _write_progress(f"clone failed: {exc}\n")
            # Write empty prediction so we don't retry on resume.
            append_prediction(
                Prediction(instance.instance_id, f"{provider}/{model}", ""),
                output_path,
            )
            return InstanceResult(
                instance_id=instance.instance_id,
                elapsed_seconds=elapsed,
                error=f"clone failed: {exc}",
            )

        # Run the agent.
        error: str | None = None
        cost_usd = 0.0
        tokens_used = 0
        try:
            # Wrap the issue text with a prescriptive fix directive.
            # The agent must focus on editing source code, not setting
            # up environments or running tests (the SWE-bench harness
            # handles verification in its own Docker container).
            task = (
                "Fix the following GitHub issue by editing the source code "
                "in this repository.\n\n"
                "IMPORTANT RULES:\n"
                "1. Read the relevant source files to understand the bug\n"
                "2. Use Edit to make the minimal fix — do NOT rewrite entire files\n"
                "3. Do NOT modify test files\n"
                "4. Do NOT try to install dependencies or set up the environment\n"
                "5. Do NOT try to run the test suite — just make the code fix\n"
                "6. Focus on the source code change, nothing else\n\n"
                f"## Issue\n\n{instance.problem_statement}"
            )
            cost_usd, tokens_used = await asyncio.wait_for(
                self._run_agent(
                    task=task,
                    workspace=workspace,
                    provider=provider,
                    model=model,
                    budget=budget,
                    auto_approve=auto_approve,
                    advisor_enabled=advisor_enabled,
                    advisor_provider=advisor_provider,
                    advisor_model=advisor_model,
                    advisor_budget_pct=advisor_budget_pct,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            error = f"timeout after {timeout}s"
            _write_progress("TIMEOUT ")
        except Exception as exc:
            error = str(exc)[:200]
            _write_progress(f"ERROR({type(exc).__name__}) ")

        # Capture patch (diff against the instance's base commit so we pick up
        # checkpoint commits made during the run, not just unstaged changes).
        patch = capture_patch(workspace, base_commit=instance.base_commit)
        patch_lines = len(patch.splitlines()) if patch else 0

        # Write prediction incrementally.
        append_prediction(
            Prediction(instance.instance_id, f"{provider}/{model}", patch),
            output_path,
        )

        elapsed = time.monotonic() - start
        status = f"patch: {patch_lines} lines" if patch_lines > 0 else "no patch"
        _write_progress(f"{status}, ${cost_usd:.4f}, {elapsed:.0f}s\n")

        # Cleanup workspace.
        if not self._keep_workspaces:
            try:
                cleanup_workspace(instance, workspace, clone_cache=clone_cache)
            except Exception:
                pass

        return InstanceResult(
            instance_id=instance.instance_id,
            patch_lines=patch_lines,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            elapsed_seconds=elapsed,
            error=error,
        )

    async def _run_agent(
        self,
        *,
        task: str,
        workspace: Path,
        provider: str,
        model: str,
        budget: float,
        auto_approve: bool,
        advisor_enabled: bool = False,
        advisor_provider: str | None = None,
        advisor_model: str | None = None,
        advisor_budget_pct: float | None = None,
    ) -> tuple[float, int]:
        """Run the mAIke orchestrator on a task. Returns (cost_usd, tokens_used)."""
        from maike.cost.tracker import CostTracker
        from maike.gateway import LLMGateway
        from maike.observability.live import FileTraceSink
        from maike.observability.tracer import Tracer
        from maike.orchestrator.orchestrator import Orchestrator

        cost_tracker = CostTracker(session_budget_usd=budget if budget > 0 else None)
        # Per-instance trace log so timed-out sessions still leave a record.
        # Without this, a timed-out run leaves agent_runs/step_results empty
        # in session.db and there's no way to diagnose what the agent did.
        # FileTraceSink appends each event as a JSON line and flushes per
        # write — survives asyncio.CancelledError from the wait_for timeout.
        trace_path = workspace / ".maike" / "trace.jsonl"
        with FileTraceSink(log_path=trace_path) as trace_sink:
            tracer = Tracer(sink=trace_sink)
            orchestrator = Orchestrator(
                base_path=workspace,
                llm_gateway=LLMGateway(cost_tracker, tracer, provider_name=provider),
                cost_tracker=cost_tracker,
                tracer=tracer,
            )

            run_kwargs = dict(
                task=task,
                workspace=workspace,
                provider_name=provider,
                model=model,
                budget=budget if budget > 0 else None,
                language_override="python",
                auto_approve=auto_approve,
            )
            if advisor_enabled:
                run_kwargs["advisor_enabled"] = True
                run_kwargs["advisor_provider"] = advisor_provider
                run_kwargs["advisor_model"] = advisor_model
                if advisor_budget_pct is not None:
                    run_kwargs["advisor_budget_pct"] = advisor_budget_pct
            await orchestrator.run(**run_kwargs)

        return cost_tracker.session_total, sum(
            r.total_tokens for r in cost_tracker.records
        )


def _write_progress(msg: str) -> None:
    """Write progress to stderr (not captured by output redirection)."""
    sys.stderr.write(msg)
    sys.stderr.flush()
