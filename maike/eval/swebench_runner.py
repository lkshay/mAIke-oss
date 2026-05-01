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

        # Probe the repo for its test runner + target Python version so we
        # can hint at both in the prompt.  Falls back to a sane Python
        # default if probe finds nothing.
        target_python_version: str | None = None
        try:
            from maike.runtime.probe import EnvironmentProbe
            probe_manifest = EnvironmentProbe().probe(workspace)
            detected_test_cmd = (probe_manifest.test_command or "pytest -x --tb=short").strip()
            target_python_version = probe_manifest.target_language_version
        except Exception as _probe_exc:
            logger.debug("EnvironmentProbe failed for %s: %s", instance.instance_id, _probe_exc)
            detected_test_cmd = "pytest -x --tb=short"

        # Build the Python-mismatch warning block.  If the target Python
        # is older than the host, the agent will hit ``ModuleNotFoundError``
        # on stdlib modules removed in newer Pythons (cgi removed in 3.13,
        # asyncore in 3.12, imp in 3.12, distutils in 3.12) and waste budget
        # "fixing" them.  See django__django-11400 (24 Apr 2026) — Flash
        # spent extensive iteration patching a cgi import that the harness
        # never sees.  We tell the agent explicitly: trust the project's
        # python_requires, the harness uses the matching Python.
        import sys as _sys
        host_py = f"{_sys.version_info.major}.{_sys.version_info.minor}"
        if target_python_version:
            python_block = (
                "## Python version contract\n\n"
                f"- This project's `python_requires`: **`{target_python_version}`** "
                "(the harness will run against a Python in that range).\n"
                f"- Your local Python (host): **{host_py}**.\n"
                "- If your local `pytest` raises `ModuleNotFoundError` on stdlib "
                "modules (`cgi`, `asyncore`, `imp`, `distutils`, etc.), that is a "
                "host-vs-target version mismatch — those modules exist in the "
                "target Python and the harness will resolve them. Do NOT 'fix' "
                "the import. Skip that test locally and focus on the actual issue.\n\n"
            )
        else:
            python_block = (
                "## Python version contract\n\n"
                f"- Your local Python is **{host_py}**; the harness uses the project's "
                "target Python (often older).\n"
                "- If `pytest` fails locally with `ModuleNotFoundError` on stdlib "
                "modules (`cgi`, `asyncore`, `imp`, `distutils`), that's a host-vs-target "
                "mismatch — do NOT 'fix' the import. Focus on the actual issue.\n\n"
            )

        # Run the agent.
        error: str | None = None
        cost_usd = 0.0
        tokens_used = 0
        try:
            # Task prompt: explain the harness contract so the agent knows
            # local installs/runs don't affect grading.  This unblocks
            # test-driven self-correction (the dominant strategy in
            # leading agents like Claude Code, OpenHands) without the
            # agent overfitting to the FAIL_TO_PASS spec.
            task = (
                f"You are fixing a real bug in `{instance.repo}` "
                f"(commit `{instance.base_commit[:8]}`).\n\n"
                "## How the SWE-bench harness grades you\n\n"
                "- It runs hidden FAIL_TO_PASS tests on a *fresh clone* with your patch applied.\n"
                "- Whatever you install (`pip install ...`) or run (`pytest ...`) locally is "
                "discarded after grading. Use the local environment freely for self-correction.\n\n"
                f"{python_block}"
                "## Workflow\n\n"
                "**Bias toward action.** Even with imperfect information, make your best "
                "surgical edit and submit. An imperfect targeted fix is far better than "
                "an empty patch.\n\n"
                "1. **Start with the smallest possible edit.** Identify the function or "
                "class named in the issue and make the minimal change that addresses it. "
                "Do not explore broadly before editing.\n"
                "2. **Tests are for verification, not discovery.** "
                f"If `{detected_test_cmd}` runs easily, use it to confirm your specific fix. "
                "Don't aim to make every test pass — many may fail for unrelated environment "
                "reasons (Python version drift, missing optional deps). Target the test file "
                "named in the issue.\n"
                "3. **Stay in scope.** Touch only files clearly related to the issue. Cleaning "
                "up unrelated code, 'modernizing' adjacent helpers, or refactoring imports is "
                "OUT OF SCOPE and is the most common way to break tests the harness cares about.\n"
                "4. **Submit the smallest viable diff.** A 5-line targeted fix beats a 50-line "
                "refactor that 'covers more cases'. If your patch exceeds ~30 lines, ask "
                "whether you're doing more than the issue requires.\n\n"
                "## Constraints\n\n"
                "- Do NOT modify test files. The harness's `FAIL_TO_PASS` tests are hidden — "
                "modifying tests is cheating.\n"
                "- Do NOT modernize imports (e.g. `from collections import Mapping` → "
                "`from collections.abc import Mapping`). The harness's target Python may "
                "have different stdlib structure than your host.\n"
                "- Do NOT refactor adjacent code, rewrite helper functions, or 'clean up' "
                "logic that isn't named in the issue.\n"
                "- Do NOT try to enumerate which tests the harness will run. Focus on the issue text.\n"
                "- Do NOT run destructive git commands (`git reset --hard`, `git checkout .`, "
                "`git stash`) — these can silently lose your work before patch capture.\n\n"
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

        # Pre-flight: verify the patch is internally consistent with the
        # workspace state.  Validation failure doesn't kill submission —
        # the harness might still accept it — but we log a warning so the
        # failure mode is visible in the eval output.  See SWE-bench v2
        # astropy__astropy-13398: phantom no-op hunks made patch(1)
        # think the patch was reverse-applied.  _normalize_patch now
        # strips no-op hunks, but other malformations may remain.
        if patch:
            from maike.eval.swebench import validate_patch_applies
            ok, reason = validate_patch_applies(
                patch, instance.base_commit, workspace,
            )
            if not ok:
                logger.warning(
                    "patch validation failed for %s: %s",
                    instance.instance_id, reason,
                )

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
        # File trace sink disabled for SWE-bench eval runs.  On tough instances
        # the react loop can produce ~200K events/sec when stuck, generating
        # multi-GB trace files in minutes and filling disk on a host with
        # tight free space.  In-memory tracer events are still captured by
        # ``Tracer.events`` for the few callers that read them; what we drop
        # is the per-event JSONL write to disk.  Re-enable by setting
        # ``MAIKE_SWEBENCH_TRACE_FILES=1`` in the environment.
        import os as _os
        _enable_trace_files = _os.environ.get("MAIKE_SWEBENCH_TRACE_FILES") == "1"
        trace_path = workspace / ".maike" / "trace.jsonl"
        if _enable_trace_files:
            from contextlib import nullcontext
            sink_cm = FileTraceSink(log_path=trace_path)
            file_sink = sink_cm
        else:
            sink_cm = None
            file_sink = None

        try:
            if file_sink is not None:
                file_sink.__enter__()
            tracer = Tracer(sink=file_sink)
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
        finally:
            if file_sink is not None:
                file_sink.__exit__(None, None, None)

        return cost_tracker.session_total, sum(
            r.total_tokens for r in cost_tracker.records
        )


def _write_progress(msg: str) -> None:
    """Write progress to stderr (not captured by output redirection)."""
    sys.stderr.write(msg)
    sys.stderr.flush()
