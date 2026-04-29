#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from maike.smoke.production_scenarios import (
    DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT,
    ProductionScenarioExecutionError,
    path_is_within,
    run_production_scenario,
    select_production_scenario_names,
)
from maike.smoke.workflows import default_live_provider, provider_has_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run production-grade live scenarios for mAIke.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Scenario to run: greenfield-expense-tracker, debugging-csv-import, or all. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument("--provider", default=None, help="Provider to use. Defaults to the first configured provider.")
    parser.add_argument("--model", default=None, help="Model override.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT,
        help="Root directory for scenario workspaces. Must be outside the mAIke repo.",
    )
    parser.add_argument(
        "--skip-agent-token-budget-check",
        action="store_true",
        help="Do not fail scenarios when agent token usage exceeds the scenario token budget.",
    )
    parser.add_argument("--keep-workspaces", action="store_true", help="Keep successful workspaces.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose live run output and persistence.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario_names = select_production_scenario_names(args.scenario or ["all"])
    provider = args.provider or default_live_provider()
    workspace_root = args.workspace_root.resolve()

    if path_is_within(ROOT_DIR, workspace_root):
        print(
            f"Refusing workspace root inside the repo: {workspace_root}. "
            f"Choose a path outside {ROOT_DIR}.",
            file=sys.stderr,
        )
        return 1

    if not provider_has_key(provider):
        print(
            f"No API key detected for provider '{provider}'. "
            "Set the provider key in the environment or .env file.",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for scenario_name in scenario_names:
        try:
            outcome = run_production_scenario(
                scenario_name,
                provider=provider,
                model=args.model,
                workspace=workspace_root / scenario_name,
                enforce_agent_token_budget=not args.skip_agent_token_budget_check,
                verbose=args.verbose,
            )
        except ProductionScenarioExecutionError as exc:
            print(f"[FAIL] {scenario_name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(
            f"[PASS] {scenario_name}: provider={outcome.provider} "
            f"model={outcome.model} pipeline={outcome.pipeline} workspace={outcome.workspace}"
        )
        if not args.keep_workspaces:
            shutil.rmtree(outcome.workspace, ignore_errors=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
