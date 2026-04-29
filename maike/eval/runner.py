"""Workflow evaluation runner."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
import shutil

logger = logging.getLogger(__name__)
import subprocess
import tempfile
from pathlib import Path

from maike.eval.baselines import baseline_from_report, baseline_path, load_baseline, write_baseline
from maike.eval.contracts import (
    DynamicEvalMetrics,
    EvalCaseReport,
    EvalComparison,
    EvalCostSummary,
    EvalMetrics,
    EvalMode,
    EvalRegression,
    EvalReport,
    EvalRequest,
    EvalStageCostSummary,
)
from maike.eval.metrics import (
    COST_INCREASE_FAIL_ABS,
    COST_INCREASE_FAIL_RATIO,
    COST_INCREASE_WARN_ABS,
    COST_INCREASE_WARN_RATIO,
    TOKEN_INCREASE_FAIL_ABS,
    TOKEN_INCREASE_FAIL_RATIO,
    TOKEN_INCREASE_WARN_ABS,
    TOKEN_INCREASE_WARN_RATIO,
    dynamic_behavior_score,
    evaluate_increase,
    evaluate_retry_increase,
    evaluate_score_drop,
    policy_snapshot,
    react_eval_score,
    workflow_eval_score,
)
from maike.eval.reporting import build_run_metadata, canonical_suite_key, write_report
from maike.gateway.providers import resolve_model_name
from maike.smoke.workflow_cases.helpers import snapshot_workspace_files


def load_workflow_cases():
    from maike.smoke.workflow_cases.registry import _ensure_all_cases
    from maike.smoke.workflows import WORKFLOW_CASES

    _ensure_all_cases()
    return WORKFLOW_CASES


def latest_session(workspace: Path, session_id: str):
    from maike.smoke.workflows import _latest_session

    return _latest_session(workspace, session_id)


def load_session_snapshot(workspace: Path, session_id: str):
    from maike.smoke.workflows import load_session_snapshot as _load_session_snapshot

    return _load_session_snapshot(workspace, session_id)


def load_session_cost_summary(workspace: Path, session_id: str):
    from maike.memory.session import SessionStore

    async def _load():
        store = SessionStore(workspace)
        await store.initialize()
        return await store.get_session_cost(session_id)

    return asyncio.run(_load())


def default_live_provider() -> str:
    from maike.smoke.workflows import default_live_provider as _default_live_provider

    return _default_live_provider()


def provider_has_key(provider: str) -> bool:
    from maike.smoke.workflows import provider_has_key as _provider_has_key

    return _provider_has_key(provider)


def select_workflow_names(values) -> list[str]:
    from maike.smoke.workflows import select_workflow_names as _select_workflow_names

    return _select_workflow_names(values)


def default_workflow_names() -> tuple[str, ...]:
    from maike.smoke.workflows import DEFAULT_WORKFLOW_NAMES

    return DEFAULT_WORKFLOW_NAMES


def dynamic_workflow_names() -> tuple[str, ...]:
    cases = load_workflow_cases()
    return tuple(name for name, case in cases.items() if "dynamic" in getattr(case, "tags", ()))


def snapshot_artifact_names(snapshot) -> list[str]:
    latest_artifact_names = getattr(snapshot, "latest_artifact_names", None)
    if latest_artifact_names is not None:
        return list(latest_artifact_names)
    return list(getattr(snapshot, "artifact_names", ()))


class EvalRunner:
    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        budget: float = 5.0,
        keep_workspaces: bool = False,
        adaptive_model: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.budget = budget
        self.keep_workspaces = keep_workspaces
        self.adaptive_model = adaptive_model

    def run_suite(self, suite: str, workspace_root: Path) -> EvalReport:
        return self.run(
            EvalRequest(
                suite=suite,
                workspace_root=workspace_root,
                provider=self.provider,
                model=self.model,
                budget=self.budget,
                keep_workspaces=self.keep_workspaces,
                mode=EvalMode.RUN,
            )
        )

    def run(self, request: EvalRequest) -> EvalReport:
        workspace_root = Path(request.workspace_root).expanduser().resolve()
        workflow_names = self._workflow_names_for_suite(request.suite)
        suite_key = canonical_suite_key(request.suite, workflow_names)
        resolved_provider = request.provider or self.provider or default_live_provider()
        if not provider_has_key(resolved_provider):
            raise RuntimeError(f"No API key detected for provider '{resolved_provider}'")
        resolved_model = resolve_model_name(resolved_provider, request.model or self.model)

        baseline_ref = baseline_path(
            workspace_root,
            suite_key=suite_key,
            provider=resolved_provider,
            model=resolved_model,
        )
        baseline_load_error: str | None = None
        baseline = None
        if request.mode is EvalMode.COMPARE:
            try:
                baseline = load_baseline(
                    workspace_root,
                    suite_key=suite_key,
                    provider=resolved_provider,
                    model=resolved_model,
                )
            except FileNotFoundError:
                baseline_load_error = (
                    "Missing baseline for "
                    f"{suite_key} ({resolved_provider}/{resolved_model}) at {baseline_ref}"
                )
            except Exception as exc:
                baseline_load_error = f"Failed to load baseline at {baseline_ref}: {exc}"

        metadata = build_run_metadata(
            suite=request.suite,
            suite_key=suite_key,
            provider=resolved_provider,
            model=resolved_model,
            budget=request.budget,
            workspace_root=workspace_root,
            baseline_path=baseline_ref if request.mode in {EvalMode.COMPARE, EvalMode.BASELINE} else None,
        )
        keep_workspaces = request.keep_workspaces or self.keep_workspaces

        results = [
            self._run_case(
                workflow_name=workflow_name,
                provider=resolved_provider,
                model=resolved_model,
                workspace_root=workspace_root,
                keep_workspaces=keep_workspaces,
                budget=request.budget,
            )
            for workflow_name in workflow_names
        ]
        if request.mode is EvalMode.COMPARE:
            results = self._apply_baseline_comparison(
                results=results,
                baseline=baseline,
                baseline_load_error=baseline_load_error,
                baseline_path_value=str(baseline_ref),
            )

        total = len(results)
        passed = sum(1 for result in results if result.passed)
        failed = total - passed
        average_score = sum(result.score for result in results) / total if total else 0.0
        execution_failed = any(not result.execution_passed for result in results)
        regression_failed = any(not result.comparison_passed for result in results)
        warning_count = sum(len(result.warnings) for result in results)
        report = EvalReport(
            schema_version=1,
            mode=request.mode.value,
            suite=request.suite,
            suite_key=suite_key,
            provider=resolved_provider,
            model=resolved_model,
            budget=request.budget,
            created_at=metadata.created_at,
            git_sha=metadata.git_sha,
            report_path=metadata.report_path,
            baseline_path=metadata.baseline_path,
            total=total,
            passed=passed,
            failed=failed,
            average_score=average_score,
            execution_failed=execution_failed,
            regression_failed=regression_failed,
            warning_count=warning_count,
            results=results,
        )
        report = write_report(report, workspace_root=workspace_root, output_path=request.output_path)

        if request.mode is EvalMode.BASELINE and not report.execution_failed:
            baseline_payload = baseline_from_report(report, policy=policy_snapshot())
            baseline_payload = replace(baseline_payload, baseline_path=str(baseline_ref))
            write_baseline(baseline_payload, workspace_root=workspace_root)

        return report

    def _workflow_names_for_suite(self, suite: str) -> list[str]:
        normalized = suite.strip().lower()
        cases = load_workflow_cases()
        if normalized in {"smoke", "control"}:
            selected = [name for name in default_workflow_names() if name in cases]
            return selected or list(cases)
        if normalized == "core":
            selected = [name for name in ("greenfield", "editing", "debugging") if name in cases]
            return selected or list(cases)
        if normalized in {"dynamic", "semi-real"}:
            selected = [name for name in dynamic_workflow_names() if name in cases]
            return selected or list(cases)
        if normalized in {"tier1", "tier2", "tier3", "tier4"}:
            selected = [name for name, case in cases.items() if normalized in getattr(case, "tags", ())]
            return selected
        if normalized == "react":
            selected = [name for name, case in cases.items() if "react" in getattr(case, "tags", ())]
            return selected
        if normalized == "agentic":
            selected = [name for name, case in cases.items() if "agentic" in getattr(case, "tags", ())]
            return selected
        if normalized == "live-repo":
            selected = [name for name, case in cases.items() if "live-repo" in getattr(case, "tags", ())]
            return selected
        if normalized == "hard-agentic":
            selected = [name for name, case in cases.items()
                        if "agentic" in getattr(case, "tags", ())
                        and any(t in getattr(case, "tags", ()) for t in ("hard", "very-hard"))]
            return selected
        if normalized == "all":
            return list(cases)
        return select_workflow_names(item.strip() for item in suite.split(","))

    def _run_case(
        self,
        *,
        workflow_name: str,
        provider: str,
        model: str,
        workspace_root: Path,
        keep_workspaces: bool,
        budget: float,
    ) -> EvalCaseReport:
        case = load_workflow_cases()[workflow_name]
        is_react = hasattr(case, "is_react") and case.is_react
        is_multi_phase = hasattr(case, "is_multi_phase") and case.is_multi_phase

        if is_multi_phase:
            return self._run_multi_phase_case(
                workflow_name=workflow_name,
                case=case,
                provider=provider,
                model=model,
                workspace_root=workspace_root,
                keep_workspaces=keep_workspaces,
                budget=budget,
            )

        workspace_path = self._create_case_workspace(workspace_root, case.name)
        from maike.smoke.workflow_cases.helpers import snapshot_workspace_hashes
        # Run setup_workspace (seeder) early so pre-snapshot captures the
        # seeded state, not the empty workspace.  This way, only files the
        # AGENT modifies count as "modified" in the agentic metrics.
        case.setup_workspace(workspace_path)
        pre_files = snapshot_workspace_files(workspace_path)
        pre_hashes = snapshot_workspace_hashes(workspace_path)
        metrics = EvalMetrics()
        session_id: str | None = None
        observed_pipeline: str | None = None
        failure_reasons: list[str] = []
        cost_summary: EvalCostSummary | None = None

        try:
            run_records = self._run_case_runs(
                case=case,
                provider=provider,
                model=model,
                workspace_path=workspace_path,
                budget=budget,
                setup_already_done=True,
            )
            latest_run = run_records[-1]
            run_result = latest_run.run_result
            snapshot = latest_run.snapshot
            cost_summary = latest_run.cost_summary
            observed_pipeline = run_result.pipeline
            session_id = run_result.session_id

            if is_react:
                # React cases: skip pipeline/stage/artifact checks
                metrics = EvalMetrics(
                    session_completed=snapshot.status == "completed",
                    session_status_match=snapshot.status in getattr(case, "expected_session_statuses", ("completed",)),
                    pipeline_match=True,
                    stage_sequence_match=True,
                    persisted_stage_sequence_match=True,
                    required_artifacts_present=True,
                    required_stage_gates_present=True,
                    workspace_verified=False,
                )
                if not metrics.session_status_match:
                    failure_reasons.append(
                        f"expected session status in {getattr(case, 'expected_session_statuses', ('completed',))!r}, got {snapshot.status!r}"
                    )
            else:
                observed_stages = tuple(run_result.stage_results)
                dynamic_metrics = self._collect_dynamic_metrics(case=case, snapshot=snapshot)
                latest_artifact_names = snapshot_artifact_names(snapshot)
                required_stage_gates_present = True
                gate_failure_reasons = []
                metrics = EvalMetrics(
                    session_completed=snapshot.status == "completed",
                    session_status_match=snapshot.status in getattr(case, "expected_session_statuses", ("completed",)),
                    pipeline_match=run_result.pipeline == case.expected_pipeline,
                    stage_sequence_match=self._matches_stage_sequence(
                        observed_stages,
                        case.expected_stages,
                        getattr(case, "observed_stage_sequence_mode", "exact"),
                    ),
                    persisted_stage_sequence_match=self._matches_stage_sequence(
                        tuple(snapshot.stage_names),
                        case.expected_stages,
                        getattr(case, "persisted_stage_sequence_mode", "exact"),
                    ),
                    required_artifacts_present=all(
                        artifact_name in latest_artifact_names
                        for artifact_name in case.expected_stage_artifacts
                    ),
                    required_stage_gates_present=required_stage_gates_present,
                    workspace_verified=False,
                    dynamic=dynamic_metrics,
                )
                if not metrics.session_status_match:
                    failure_reasons.append(
                        f"expected session status in {getattr(case, 'expected_session_statuses', ('completed',))!r}, got {snapshot.status!r}"
                    )
                if not metrics.pipeline_match:
                    failure_reasons.append(
                        f"expected pipeline {case.expected_pipeline!r}, got {run_result.pipeline!r}"
                    )
                if not metrics.stage_sequence_match:
                    failure_reasons.append(
                        f"expected stages {case.expected_stages!r}, got {observed_stages!r}"
                    )
                if not metrics.persisted_stage_sequence_match:
                    failure_reasons.append(
                        f"expected persisted stages {case.expected_stages!r}, got {tuple(snapshot.stage_names)!r}"
                    )
                failure_reasons.extend(gate_failure_reasons)
                failure_reasons.extend(self._dynamic_failure_reasons(case, dynamic_metrics))

            # Workspace verification (common to both react and workflow)
            verifier = getattr(case, "verify_workspace", None)
            if verifier is not None:
                try:
                    verifier(workspace_path)
                    metrics = replace(metrics, workspace_verified=True)
                    verify_run = getattr(case, "verify_run", None)
                    if verify_run is not None:
                        from maike.smoke.workflows import WorkflowVerificationContext

                        verify_run(
                            WorkflowVerificationContext(
                                workspace=workspace_path,
                                runs=tuple(run_records),
                                latest_run=latest_run,
                            )
                        )
                except Exception as exc:
                    failure_reasons.append(f"workspace verification failed: {exc}")
            else:
                # No verifier — treat as verified (e.g. question tasks)
                metrics = replace(metrics, workspace_verified=True)
        except Exception as exc:
            failure_reasons.append(str(exc))

        # Collect agentic metrics if the case has the "agentic" tag.
        agentic = None
        is_agentic = "agentic" in getattr(case, "tags", ())
        if is_agentic and session_id:
            try:
                from maike.eval.agentic_metrics import collect_agentic_metrics
                post_files = snapshot_workspace_files(workspace_path)
                post_hashes = snapshot_workspace_hashes(workspace_path)
                agentic = collect_agentic_metrics(
                    workspace=workspace_path,
                    session_id=session_id,
                    cost_usd=cost_summary.total_cost_usd if cost_summary else 0.0,
                    passed=not failure_reasons,
                    difficulty_weight=getattr(case, "difficulty_weight", 1.0),
                    expected_modified_files=getattr(case, "expected_modified_files", ()),
                    pre_files=pre_files,
                    post_files=post_files,
                    pre_hashes=pre_hashes,
                    post_hashes=post_hashes,
                )
            except Exception as exc:
                logger.warning("Failed to collect agentic metrics: %s", exc)

        # Score using appropriate function
        if is_agentic and agentic is not None:
            from maike.eval.metrics import agentic_eval_score
            score = agentic_eval_score(
                session_completed=metrics.session_completed,
                workspace_verified=metrics.workspace_verified,
                error_recovery_rate=agentic.error_recovery_rate,
                wasted_call_ratio=agentic.wasted_call_ratio,
                change_minimality_score=agentic.change_minimality_score,
            )
            case_type = "react"
        elif is_react:
            score = react_eval_score(
                session_completed=metrics.session_completed,
                workspace_verified=metrics.workspace_verified,
            )
            case_type = "react"
        else:
            dynamic_score = self._dynamic_score(case, metrics.dynamic)
            score = workflow_eval_score(
                session_status_match=metrics.session_status_match,
                pipeline_match=metrics.pipeline_match,
                stage_sequence_match=metrics.stage_sequence_match,
                persisted_stage_sequence_match=metrics.persisted_stage_sequence_match,
                required_stage_gates_present=metrics.required_stage_gates_present,
                workspace_verified=metrics.workspace_verified,
                dynamic_score=dynamic_score,
            )
            case_type = "workflow"

        execution_passed = not failure_reasons
        passed = execution_passed

        if passed and not keep_workspaces:
            shutil.rmtree(workspace_path, ignore_errors=True)

        return EvalCaseReport(
            workflow_name=workflow_name,
            provider=provider,
            model=model,
            workspace=str(workspace_path),
            session_id=session_id,
            pipeline=observed_pipeline,
            passed=passed,
            execution_passed=execution_passed,
            comparison_passed=True,
            score=score,
            metrics=metrics,
            cost_summary=cost_summary,
            failure_reasons=failure_reasons,
            warnings=[],
            comparison=None,
            error="; ".join(failure_reasons) if failure_reasons else None,
            case_type=case_type,
            agentic_metrics=agentic,
        )

    def _run_multi_phase_case(
        self,
        *,
        workflow_name: str,
        case,
        provider: str,
        model: str,
        workspace_root: Path,
        keep_workspaces: bool,
        budget: float,
    ) -> EvalCaseReport:
        """Execute a multi-phase EvalCase, running each phase sequentially in the same workspace."""
        import time as _time
        from datetime import datetime, timezone
        from maike.eval.contracts import PhaseReport
        from maike.eval.metrics import classify_eval_error

        workspace_path = self._create_case_workspace(workspace_root, case.name)
        case.setup_workspace(workspace_path)

        phase_reports: list[PhaseReport] = []
        total_cost = 0.0
        total_tokens = 0
        all_session_ids: list[str] = []
        failure_reasons: list[str] = []
        session_completed = True

        for idx, phase in enumerate(case.phases):
            phase_start = _time.monotonic()
            phase_started_at = datetime.now(timezone.utc)
            phase_name = f"phase-{idx}"
            phase_budget = phase.budget if phase.budget is not None else budget

            # Snapshot files before phase (for read-only violation check)
            pre_files: set[str] = set()
            if phase.expect_read_only:
                pre_files = {
                    str(p.relative_to(workspace_path))
                    for p in workspace_path.rglob("*")
                    if p.is_file() and ".maike" not in p.parts and ".venv" not in p.parts and ".git" not in p.parts
                }

            try:
                run_result = self._execute_case_run(
                    case=case,
                    provider=provider,
                    model=model,
                    workspace_path=workspace_path,
                    budget=phase_budget,
                    task_override=phase.task,
                )
                session_id = run_result.session_id
                all_session_ids.append(session_id)
                cost_summary = self._load_cost_summary(workspace_path, session_id)
                phase_cost = cost_summary.total_cost_usd if cost_summary else 0.0
                phase_tokens = cost_summary.total_tokens if cost_summary else 0
                total_cost += phase_cost
                total_tokens += phase_tokens

                snapshot = load_session_snapshot(workspace_path, session_id)
                phase_status = "completed" if snapshot.status == "completed" else "failed"
                if snapshot.status != "completed":
                    session_completed = False
                    failure_reasons.append(f"{phase_name}: session status {snapshot.status!r}")

                # Check read-only violation
                read_only_violation = False
                files_written = 0
                if phase.expect_read_only:
                    post_files = {
                        str(p.relative_to(workspace_path))
                        for p in workspace_path.rglob("*")
                        if p.is_file() and ".maike" not in p.parts and ".venv" not in p.parts and ".git" not in p.parts
                    }
                    new_files = post_files - pre_files
                    if new_files:
                        read_only_violation = True
                        failure_reasons.append(
                            f"{phase_name}: read-only violation, created files: {new_files}"
                        )
                    files_written = len(new_files)

                phase_elapsed = _time.monotonic() - phase_start
                phase_ended_at = datetime.now(timezone.utc)
                phase_reports.append(PhaseReport(
                    phase_index=idx,
                    phase_name=phase_name,
                    session_id=session_id,
                    status=phase_status,
                    cost_usd=phase_cost,
                    tokens_used=phase_tokens,
                    files_written=files_written,
                    read_only_violation=read_only_violation,
                    duration_seconds=phase_elapsed,
                    wall_clock_seconds=phase_elapsed,
                    started_at=phase_started_at,
                    ended_at=phase_ended_at,
                ))
            except Exception as exc:
                phase_elapsed = _time.monotonic() - phase_start
                phase_ended_at = datetime.now(timezone.utc)
                error_msg = str(exc)
                phase_reports.append(PhaseReport(
                    phase_index=idx,
                    phase_name=phase_name,
                    status="error",
                    error=error_msg,
                    error_type=classify_eval_error(error_msg),
                    duration_seconds=phase_elapsed,
                    wall_clock_seconds=phase_elapsed,
                    started_at=phase_started_at,
                    ended_at=phase_ended_at,
                ))
                failure_reasons.append(f"{phase_name}: {exc}")
                session_completed = False
                break  # Stop executing further phases on error

        # Run phase verifier if provided
        workspace_verified = True
        phase_verifier = getattr(case, "verify_phases", None)
        if phase_verifier is not None:
            try:
                phase_verifier(phase_reports, workspace_path)
            except Exception as exc:
                workspace_verified = False
                failure_reasons.append(f"phase verification failed: {exc}")
        # Also run workspace verifier if provided
        ws_verifier = getattr(case, "verify_workspace", None)
        if ws_verifier is not None:
            try:
                ws_verifier(workspace_path)
            except Exception as exc:
                workspace_verified = False
                failure_reasons.append(f"workspace verification failed: {exc}")

        score = react_eval_score(
            session_completed=session_completed,
            workspace_verified=workspace_verified,
        )

        metrics = EvalMetrics(
            session_completed=session_completed,
            session_status_match=session_completed,
            pipeline_match=True,
            stage_sequence_match=True,
            persisted_stage_sequence_match=True,
            required_artifacts_present=True,
            required_stage_gates_present=True,
            workspace_verified=workspace_verified,
        )

        cost_summary_agg = EvalCostSummary(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
        )

        execution_passed = not failure_reasons
        passed = execution_passed

        if passed and not keep_workspaces:
            shutil.rmtree(workspace_path, ignore_errors=True)

        return EvalCaseReport(
            workflow_name=workflow_name,
            provider=provider,
            model=model,
            workspace=str(workspace_path),
            session_id=all_session_ids[-1] if all_session_ids else None,
            pipeline=None,
            passed=passed,
            execution_passed=execution_passed,
            comparison_passed=True,
            score=score,
            metrics=metrics,
            cost_summary=cost_summary_agg,
            failure_reasons=failure_reasons,
            warnings=[],
            comparison=None,
            error="; ".join(failure_reasons) if failure_reasons else None,
            case_type="multi_phase",
            phase_reports=phase_reports,
        )

    def _run_case_runs(
        self,
        *,
        case,
        provider: str,
        model: str,
        workspace_path: Path,
        budget: float,
        setup_already_done: bool = False,
    ) -> list[object]:
        from maike.smoke.workflows import WorkflowRunRecord

        runs: list[object] = []
        workspace_ready = setup_already_done
        case_budget = case.budget if getattr(case, "budget", None) is not None else budget
        for run_index in range(getattr(case, "repeat_runs", 1)):
            if run_index == 0 or not getattr(case, "reuse_workspace_between_runs", False):
                if run_index > 0 and workspace_path.exists():
                    shutil.rmtree(workspace_path, ignore_errors=True)
                    workspace_path.mkdir(parents=True, exist_ok=True)
                    self._initialize_git_repo(workspace_path)
                if not workspace_ready:
                    case.setup_workspace(workspace_path)
                workspace_ready = True
            elif not workspace_ready:
                case.setup_workspace(workspace_path)
                workspace_ready = True

            run_result = self._execute_case_run(
                case=case,
                provider=provider,
                model=model,
                workspace_path=workspace_path,
                budget=case_budget,
            )
            snapshot = load_session_snapshot(workspace_path, run_result.session_id)
            cost_summary = self._load_cost_summary(workspace_path, run_result.session_id)
            runs.append(
                WorkflowRunRecord(
                    run_result=run_result,
                    snapshot=snapshot,
                    cost_summary=cost_summary,
                    workspace_files=snapshot_workspace_files(workspace_path),
                )
            )
        return runs

    def _execute_case_run(
        self,
        *,
        case,
        provider: str,
        model: str,
        workspace_path: Path,
        budget: float,
        task_override: str | None = None,
    ):
        from maike.cli import run_command

        task = task_override if task_override is not None else case.task
        return asyncio.run(
            run_command(
                task=task,
                workspace=workspace_path,
                provider=provider,
                model=model,
                language=case.language,
                budget=budget,
                agent_token_budget=getattr(case, "agent_token_budget", None),
                yes=True,
                verbose=True,
                dynamic_agents_enabled=case.dynamic_agents_enabled,
                parallel_coding_enabled=case.parallel_coding_enabled,
                adaptive_model=self.adaptive_model,
            )
        )

    def _collect_dynamic_metrics(
        self,
        *,
        case,
        snapshot,
    ) -> DynamicEvalMetrics:
        agent_runs = snapshot.agent_runs
        partition_runs = [
            run for run in agent_runs if run["metadata"].get("spawn_reason") == "parallel_partition"
        ]
        specialist_runs = [
            run for run in agent_runs if run["metadata"].get("spawn_reason") == "specialist_needed"
        ]
        reflection_runs = [
            run for run in agent_runs if run["metadata"].get("spawn_reason") == "reflection"
        ]
        fan_in_runs = [
            run for run in agent_runs if run["metadata"].get("coordination_mode") == "fan_in"
        ]
        repair_runs = [
            run for run in agent_runs if run["metadata"].get("coordination_mode") == "reflection_repair"
        ]

        downstream_inputs = {
            artifact_id
            for run in agent_runs
            if run["metadata"].get("spawn_reason") != "specialist_needed"
            for artifact_id in run["metadata"].get("input_artifact_ids", [])
        }
        useful_specialist_count = sum(
            1
            for run in specialist_runs
            if any(
                artifact_id in downstream_inputs
                for artifact_id in run["metadata"].get("produced_artifact_ids", [])
            )
        )
        total_cost_usd = sum(float(run["cost_usd"] or 0.0) for run in agent_runs)
        total_tokens = sum(int(run["tokens_used"] or 0) for run in agent_runs)
        stage_attempts: dict[str, int] = {}
        for run in agent_runs:
            stage_attempt = int(run["metadata"].get("stage_attempt", 1) or 1)
            stage_attempts[run["stage_name"]] = max(stage_attempts.get(run["stage_name"], 1), stage_attempt)
        retry_count = sum(max(attempt - 1, 0) for attempt in stage_attempts.values())
        max_observed_spawn_depth = max(
            int(run["metadata"].get("spawn_depth", 0) or 0)
            for run in agent_runs
        ) if agent_runs else 0

        return DynamicEvalMetrics(
            partition_count=len(partition_runs),
            completed_partition_count=sum(1 for run in partition_runs if run["success"]),
            fan_in_completed=any(run["success"] for run in fan_in_runs),
            specialist_spawn_count=len(specialist_runs),
            useful_specialist_count=useful_specialist_count,
            reflection_count=len(reflection_runs),
            blocking_reflection_count=sum(
                1
                for run in reflection_runs
                if run["metadata"].get("reflection_blocking") is True
            ),
            repair_count=sum(1 for run in repair_runs if run["success"]),
            retry_count=retry_count,
            max_observed_spawn_depth=max_observed_spawn_depth,
            final_convergence=snapshot.status == "completed",
            total_cost_usd=total_cost_usd,
            total_tokens=total_tokens,
        )

    def _apply_baseline_comparison(
        self,
        *,
        results: list[EvalCaseReport],
        baseline,
        baseline_load_error: str | None,
        baseline_path_value: str,
    ) -> list[EvalCaseReport]:
        cases = load_workflow_cases()
        baseline_results = {
            result.workflow_name: result
            for result in ((baseline.report.results if baseline and baseline.report else []) or [])
        }
        updated: list[EvalCaseReport] = []
        for result in results:
            case = cases[result.workflow_name]
            warnings: list[EvalRegression] = []
            regressions: list[EvalRegression] = []
            baseline_case = baseline_results.get(result.workflow_name)

            if baseline_load_error:
                regressions.append(
                    EvalRegression(
                        metric="baseline",
                        severity="failure",
                        message=baseline_load_error,
                        current_value=result.workflow_name,
                        baseline_value=baseline_path_value,
                    )
                )
            elif baseline_case is None:
                regressions.append(
                    EvalRegression(
                        metric="baseline_case",
                        severity="failure",
                        message=f"Missing baseline case for workflow {result.workflow_name!r}",
                        current_value=result.workflow_name,
                        baseline_value=None,
                    )
                )
            elif not baseline_case.execution_passed:
                regressions.append(
                    EvalRegression(
                        metric="baseline_case",
                        severity="failure",
                        message=(
                            f"Baseline case {result.workflow_name!r} is not a passing baseline and cannot be compared"
                        ),
                        current_value=result.execution_passed,
                        baseline_value=baseline_case.execution_passed,
                    )
                )
            else:
                regressions.extend(self._correctness_regressions(result, baseline_case, case))
                warnings.extend(self._threshold_regressions(result, baseline_case))
                threshold_failures = [warning for warning in warnings if warning.severity == "failure"]
                regressions.extend(threshold_failures)
                warnings = [warning for warning in warnings if warning.severity != "failure"]
                result = self._attach_dynamic_baseline_metrics(result, baseline_case, case)

            comparison_passed = not regressions
            comparison = EvalComparison(
                baseline_workflow_name=baseline_case.workflow_name if baseline_case else None,
                passed=comparison_passed,
                regressions=regressions,
                warnings=warnings,
            )
            error_messages = list(result.failure_reasons)
            if regressions:
                error_messages.extend(regression.message for regression in regressions)
            updated.append(
                replace(
                    result,
                    comparison_passed=comparison_passed,
                    passed=result.execution_passed and comparison_passed,
                    warnings=warnings,
                    comparison=comparison,
                    error="; ".join(error_messages) if error_messages else None,
                )
            )
        return updated

    def _correctness_regressions(self, result: EvalCaseReport, baseline_case: EvalCaseReport, case) -> list[EvalRegression]:
        regressions: list[EvalRegression] = []
        if baseline_case.execution_passed and not result.execution_passed:
            regressions.append(
                EvalRegression(
                    metric="execution_passed",
                    severity="failure",
                    message="Workflow no longer completes successfully",
                    current_value=result.execution_passed,
                    baseline_value=baseline_case.execution_passed,
                )
            )

        checks = [
            ("session_status_match", "Session status no longer matches the case expectation"),
            ("pipeline_match", "Pipeline no longer matches expected workflow"),
            ("stage_sequence_match", "Observed stage sequence no longer matches expected workflow"),
            (
                "persisted_stage_sequence_match",
                "Persisted stage sequence no longer matches expected workflow",
            ),
            ("required_stage_gates_present", "Persisted stage-gate metadata is missing for one or more gated stages"),
            ("workspace_verified", "Workspace verification no longer passes"),
        ]
        for field_name, message in checks:
            baseline_value = getattr(baseline_case.metrics, field_name)
            current_value = getattr(result.metrics, field_name)
            if baseline_value and not current_value:
                regressions.append(
                    EvalRegression(
                        metric=field_name,
                        severity="failure",
                        message=message,
                        current_value=current_value,
                        baseline_value=baseline_value,
                    )
                )

        expectations = case.dynamic_expectations
        if expectations is None:
            return regressions
        if expectations.min_partitions and result.metrics.dynamic.partition_count < expectations.min_partitions:
            regressions.append(
                EvalRegression(
                    metric="partition_count",
                    severity="failure",
                    message="Dynamic partitioning no longer meets the required minimum",
                    current_value=result.metrics.dynamic.partition_count,
                    baseline_value=baseline_case.metrics.dynamic.partition_count,
                )
            )
        if expectations.require_fan_in and not result.metrics.dynamic.fan_in_completed:
            regressions.append(
                EvalRegression(
                    metric="fan_in_completed",
                    severity="failure",
                    message="Dynamic fan-in no longer completes successfully",
                    current_value=result.metrics.dynamic.fan_in_completed,
                    baseline_value=baseline_case.metrics.dynamic.fan_in_completed,
                )
            )
        if expectations.min_specialists and result.metrics.dynamic.specialist_spawn_count < expectations.min_specialists:
            regressions.append(
                EvalRegression(
                    metric="specialist_spawn_count",
                    severity="failure",
                    message="Required specialist delegation no longer occurs",
                    current_value=result.metrics.dynamic.specialist_spawn_count,
                    baseline_value=baseline_case.metrics.dynamic.specialist_spawn_count,
                )
            )
        if expectations.require_specialist_useful and result.metrics.dynamic.useful_specialist_count < 1:
            regressions.append(
                EvalRegression(
                    metric="useful_specialist_count",
                    severity="failure",
                    message="Specialist output is no longer incorporated downstream",
                    current_value=result.metrics.dynamic.useful_specialist_count,
                    baseline_value=baseline_case.metrics.dynamic.useful_specialist_count,
                )
            )
        if expectations.min_reflections and result.metrics.dynamic.reflection_count < expectations.min_reflections:
            regressions.append(
                EvalRegression(
                    metric="reflection_count",
                    severity="failure",
                    message="Required reflection behavior no longer occurs",
                    current_value=result.metrics.dynamic.reflection_count,
                    baseline_value=baseline_case.metrics.dynamic.reflection_count,
                )
            )
        if expectations.require_repair and result.metrics.dynamic.repair_count < 1:
            regressions.append(
                EvalRegression(
                    metric="repair_count",
                    severity="failure",
                    message="Reflection no longer produces a successful repair run",
                    current_value=result.metrics.dynamic.repair_count,
                    baseline_value=baseline_case.metrics.dynamic.repair_count,
                )
            )
        if expectations.min_retries and result.metrics.dynamic.retry_count < expectations.min_retries:
            regressions.append(
                EvalRegression(
                    metric="retry_count",
                    severity="failure",
                    message="Required retry behavior no longer occurs",
                    current_value=result.metrics.dynamic.retry_count,
                    baseline_value=baseline_case.metrics.dynamic.retry_count,
                )
            )
        if expectations.max_spawn_depth is not None and result.metrics.dynamic.max_observed_spawn_depth > expectations.max_spawn_depth:
            regressions.append(
                EvalRegression(
                    metric="max_observed_spawn_depth",
                    severity="failure",
                    message="Observed specialist depth exceeds the configured hard limit",
                    current_value=result.metrics.dynamic.max_observed_spawn_depth,
                    baseline_value=baseline_case.metrics.dynamic.max_observed_spawn_depth,
                )
            )
        return regressions

    def _threshold_regressions(
        self,
        result: EvalCaseReport,
        baseline_case: EvalCaseReport,
    ) -> list[EvalRegression]:
        regressions: list[EvalRegression] = []

        score_eval = evaluate_score_drop(result.score, baseline_case.score)
        if score_eval and score_eval.warning:
            regressions.append(
                EvalRegression(
                    metric="score",
                    severity="failure" if score_eval.failure else "warning",
                    message=f"Score dropped from {baseline_case.score:.3f} to {result.score:.3f}",
                    current_value=result.score,
                    baseline_value=baseline_case.score,
                    delta=score_eval.delta,
                    delta_ratio=score_eval.delta_ratio,
                )
            )

        if result.cost_summary and baseline_case.cost_summary:
            cost_eval = evaluate_increase(
                result.cost_summary.total_cost_usd,
                baseline_case.cost_summary.total_cost_usd,
                warning_ratio=COST_INCREASE_WARN_RATIO,
                warning_absolute=COST_INCREASE_WARN_ABS,
                failure_ratio=COST_INCREASE_FAIL_RATIO,
                failure_absolute=COST_INCREASE_FAIL_ABS,
            )
            if cost_eval and cost_eval.warning:
                regressions.append(
                    EvalRegression(
                        metric="total_cost_usd",
                        severity="failure" if cost_eval.failure else "warning",
                        message=(
                            "Total cost increased from "
                            f"{baseline_case.cost_summary.total_cost_usd:.4f} to "
                            f"{result.cost_summary.total_cost_usd:.4f}"
                        ),
                        current_value=result.cost_summary.total_cost_usd,
                        baseline_value=baseline_case.cost_summary.total_cost_usd,
                        delta=cost_eval.delta,
                        delta_ratio=cost_eval.delta_ratio,
                    )
                )

            token_eval = evaluate_increase(
                result.cost_summary.total_tokens,
                baseline_case.cost_summary.total_tokens,
                warning_ratio=TOKEN_INCREASE_WARN_RATIO,
                warning_absolute=TOKEN_INCREASE_WARN_ABS,
                failure_ratio=TOKEN_INCREASE_FAIL_RATIO,
                failure_absolute=TOKEN_INCREASE_FAIL_ABS,
            )
            if token_eval and token_eval.warning:
                regressions.append(
                    EvalRegression(
                        metric="total_tokens",
                        severity="failure" if token_eval.failure else "warning",
                        message=(
                            "Total tokens increased from "
                            f"{baseline_case.cost_summary.total_tokens} to "
                            f"{result.cost_summary.total_tokens}"
                        ),
                        current_value=result.cost_summary.total_tokens,
                        baseline_value=baseline_case.cost_summary.total_tokens,
                        delta=token_eval.delta,
                        delta_ratio=token_eval.delta_ratio,
                    )
                )

        retry_eval = evaluate_retry_increase(
            result.metrics.dynamic.retry_count,
            baseline_case.metrics.dynamic.retry_count,
        )
        if retry_eval and retry_eval.warning:
            regressions.append(
                EvalRegression(
                    metric="retry_count",
                    severity="failure" if retry_eval.failure else "warning",
                    message=(
                        "Retry count increased from "
                        f"{baseline_case.metrics.dynamic.retry_count} to "
                        f"{result.metrics.dynamic.retry_count}"
                    ),
                    current_value=result.metrics.dynamic.retry_count,
                    baseline_value=baseline_case.metrics.dynamic.retry_count,
                    delta=int(retry_eval.delta),
                    delta_ratio=retry_eval.delta_ratio,
                )
            )
        return regressions

    def _attach_dynamic_baseline_metrics(
        self,
        result: EvalCaseReport,
        baseline_case: EvalCaseReport,
        case,
    ) -> EvalCaseReport:
        expectations = case.dynamic_expectations
        if (
            expectations is None
            or not expectations.compare_to_deterministic_baseline
            or result.cost_summary is None
            or baseline_case.cost_summary is None
        ):
            return result
        dynamic = replace(
            result.metrics.dynamic,
            baseline_cost_usd=baseline_case.cost_summary.total_cost_usd,
            baseline_tokens=baseline_case.cost_summary.total_tokens,
            cost_delta_usd=result.cost_summary.total_cost_usd - baseline_case.cost_summary.total_cost_usd,
            token_delta=result.cost_summary.total_tokens - baseline_case.cost_summary.total_tokens,
        )
        return replace(result, metrics=replace(result.metrics, dynamic=dynamic))

    def _load_cost_summary(self, workspace_path: Path, session_id: str | None) -> EvalCostSummary | None:
        if session_id is None:
            return None
        summary = load_session_cost_summary(workspace_path, session_id)
        if summary is None:
            return None
        return EvalCostSummary(
            total_cost_usd=float(summary.get("total_cost_usd", 0.0) or 0.0),
            total_tokens=int(summary.get("total_tokens", 0) or 0),
            input_tokens=int(summary.get("input_tokens", 0) or 0),
            output_tokens=int(summary.get("output_tokens", 0) or 0),
            llm_calls=int(summary.get("llm_calls", 0) or 0),
            agent_runs=int(summary.get("agent_runs", 0) or 0),
            has_token_breakdown=bool(summary.get("has_token_breakdown", False)),
            per_stage=[
                EvalStageCostSummary(
                    stage_name=stage["stage_name"],
                    cost_usd=float(stage.get("cost_usd", 0.0) or 0.0),
                    tokens_used=int(stage.get("tokens_used", 0) or 0),
                    input_tokens=int(stage.get("input_tokens", 0) or 0),
                    output_tokens=int(stage.get("output_tokens", 0) or 0),
                    llm_calls=int(stage.get("llm_calls", 0) or 0),
                    agent_runs=int(stage.get("agent_runs", 0) or 0),
                    failed_runs=int(stage.get("failed_runs", 0) or 0),
                    has_token_breakdown=bool(stage.get("has_token_breakdown", False)),
                )
                for stage in summary.get("per_stage", [])
            ],
        )

    def _dynamic_failure_reasons(self, case, metrics: DynamicEvalMetrics) -> list[str]:
        expectations = case.dynamic_expectations
        if expectations is None:
            return []
        failures: list[str] = []
        if expectations.min_partitions and metrics.partition_count < expectations.min_partitions:
            failures.append(
                f"expected at least {expectations.min_partitions} partition runs, got {metrics.partition_count}"
            )
        if expectations.min_partitions and metrics.completed_partition_count < expectations.min_partitions:
            failures.append("not all expected partition runs completed successfully")
        if expectations.require_fan_in and not metrics.fan_in_completed:
            failures.append("expected a successful fan-in integration run")
        if expectations.min_specialists and metrics.specialist_spawn_count < expectations.min_specialists:
            failures.append(
                f"expected at least {expectations.min_specialists} specialist run, got {metrics.specialist_spawn_count}"
            )
        if expectations.require_specialist_useful and metrics.useful_specialist_count < 1:
            failures.append("expected specialist output to be incorporated into a later agent run")
        if expectations.min_reflections and metrics.reflection_count < expectations.min_reflections:
            failures.append(
                f"expected at least {expectations.min_reflections} reflection run, got {metrics.reflection_count}"
            )
        if expectations.require_repair and metrics.repair_count < 1:
            failures.append("expected reflection to trigger a successful repair run")
        if expectations.min_retries and metrics.retry_count < expectations.min_retries:
            failures.append(f"expected at least {expectations.min_retries} retry, got {metrics.retry_count}")
        if expectations.max_spawn_depth is not None and metrics.max_observed_spawn_depth > expectations.max_spawn_depth:
            failures.append(
                f"expected spawn depth <= {expectations.max_spawn_depth}, got {metrics.max_observed_spawn_depth}"
            )
        return failures

    def _dynamic_score(self, case, metrics: DynamicEvalMetrics) -> float | None:
        expectations = case.dynamic_expectations
        if expectations is None:
            return None
        checks: list[bool] = []
        if expectations.min_partitions:
            checks.append(metrics.partition_count >= expectations.min_partitions)
            checks.append(metrics.completed_partition_count >= expectations.min_partitions)
        if expectations.require_fan_in:
            checks.append(metrics.fan_in_completed)
        if expectations.min_specialists:
            checks.append(metrics.specialist_spawn_count >= expectations.min_specialists)
        if expectations.require_specialist_useful:
            checks.append(metrics.useful_specialist_count >= 1)
        if expectations.min_reflections:
            checks.append(metrics.reflection_count >= expectations.min_reflections)
        if expectations.require_repair:
            checks.append(metrics.repair_count >= 1)
            checks.append(metrics.final_convergence)
        if expectations.min_retries:
            checks.append(metrics.retry_count >= expectations.min_retries)
        if expectations.max_spawn_depth is not None:
            checks.append(metrics.max_observed_spawn_depth <= expectations.max_spawn_depth)
        return dynamic_behavior_score(checks)

    def _matches_stage_sequence(self, observed: tuple[str, ...], expected: tuple[str, ...], mode: str) -> bool:
        if mode == "exact":
            return observed == expected
        if mode == "prefix":
            return observed == expected[: len(observed)]
        raise ValueError(f"Unsupported stage sequence mode: {mode}")

    def _create_case_workspace(self, workspace_root: Path, case_name: str) -> Path:
        if self._is_within_git_worktree(workspace_root):
            workspace_path = Path(tempfile.mkdtemp(prefix=f"maike-eval-{case_name}-"))
        else:
            base_dir = workspace_root / ".maike-eval"
            base_dir.mkdir(parents=True, exist_ok=True)
            workspace_path = Path(tempfile.mkdtemp(prefix=f"{case_name}-", dir=base_dir))
        self._initialize_git_repo(workspace_path)
        return workspace_path

    def _is_within_git_worktree(self, path: Path) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    def _initialize_git_repo(self, workspace_path: Path) -> None:
        commands = [
            ["git", "init"],
            ["git", "config", "user.email", "maike@local"],
            ["git", "config", "user.name", "mAIke"],
        ]
        for command in commands:
            result = subprocess.run(
                command,
                cwd=workspace_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                output = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"Failed to initialize eval git repo at {workspace_path}: "
                    f"{' '.join(command)} -> {output or result.returncode}"
                )
