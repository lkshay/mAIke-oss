#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from maike.smoke.workflows import (
    WorkflowExecutionError,
    default_live_provider,
    provider_has_key,
    run_workflow_case,
    select_workflow_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live mAIke workflow smoke tests.")
    parser.add_argument(
        "--workflow",
        action="append",
        default=[],
        help="Workflow to run: greenfield, editing, debugging, or all. Can be provided multiple times.",
    )
    parser.add_argument("--provider", default=None, help="Provider to use. Defaults to Gemini if configured.")
    parser.add_argument("--model", default=None, help="Model override. Example: gemini-3-pro-preview")
    parser.add_argument("--budget", type=float, default=5.0, help="Per-workflow run budget in USD.")
    parser.add_argument("--keep-workspaces", action="store_true", help="Keep successful workspaces.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workflow_names = select_workflow_names(args.workflow or ["all"])
    provider = args.provider or default_live_provider()

    if not provider_has_key(provider):
        print(
            f"No API key detected for provider '{provider}'. "
            "Set the provider key in the environment or .env file.",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for workflow_name in workflow_names:
        try:
            outcome = run_workflow_case(
                workflow_name,
                provider=provider,
                model=args.model,
                budget=args.budget,
            )
        except WorkflowExecutionError as exc:
            print(f"[FAIL] {workflow_name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(
            f"[PASS] {workflow_name}: provider={outcome.provider} "
            f"model={outcome.model} pipeline={outcome.pipeline} workspace={outcome.workspace}"
        )
        if not args.keep_workspaces:
            shutil.rmtree(outcome.workspace, ignore_errors=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
