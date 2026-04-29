#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from maike.cli import run_command
from maike.gateway.providers import resolve_model_name
from maike.smoke.workflows import default_live_provider, provider_has_key

WorkspaceSetup = Callable[[Path], None]


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    task: str
    workspace_name: str
    dynamic_agents_enabled: bool
    parallel_coding_enabled: bool
    budget: float | None = None
    setup_workspace: WorkspaceSetup | None = None


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _init_git_repo(workspace: Path) -> None:
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@local"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=workspace, check=True, capture_output=True)


def _setup_specialist_workspace(workspace: Path) -> None:
    _write_text(
        workspace / "validator.py",
        """
        def validate_email(email: str) -> bool:
            \"\"\"Return True if the email looks valid.\"\"\"
            if not email or "@" not in email:
                return False
            local, domain = email.split("@")
            return bool(local) and "." in domain

        def validate_age(age: str) -> int:
            \"\"\"Parse age string, return int. Raise ValueError for invalid.\"\"\"
            value = int(age)
            if value < 0:
                raise ValueError("Age cannot be negative")
            return value
        """,
    )
    _write_text(
        workspace / "test_validator.py",
        """
        from validator import validate_email, validate_age
        import pytest

        def test_valid_email():
            assert validate_email("user@example.com") is True

        def test_email_with_multiple_ats():
            assert validate_email("user@name@example.com") is False

        def test_age_zero():
            assert validate_age("0") == 0

        def test_age_negative():
            with pytest.raises(ValueError):
                validate_age("-5")
        """,
    )
    _init_git_repo(workspace)


def _setup_reflection_workspace(workspace: Path) -> None:
    _write_text(
        workspace / "counter.py",
        """
        class Counter:
            def __init__(self):
                self.value = 0

            def increment(self):
                self.value += 1

            def decrement(self):
                self.value -= 1

            def reset(self):
                self.value = 0
        """,
    )
    _write_text(
        workspace / "test_counter.py",
        """
        from counter import Counter

        def test_increment():
            c = Counter()
            c.increment()
            assert c.value == 1

        def test_decrement():
            c = Counter()
            c.decrement()
            assert c.value == -1
        """,
    )
    _write_text(workspace / "README.md", "# Counter\nA simple counter class.")
    _init_git_repo(workspace)


SCENARIOS = {
    "baseline": DynamicScenario(
        name="baseline",
        workspace_name="baseline",
        task="Create a Python CLI calculator that supports add, subtract, multiply, divide. Include tests and a README.",
        dynamic_agents_enabled=False,
        parallel_coding_enabled=False,
    ),
    "fanout": DynamicScenario(
        name="fanout",
        workspace_name="fanout",
        task=(
            "Create a Python task tracker app with these files: task_store.py (JSON-backed CRUD), "
            "task_cli.py (add/list/done commands), test_task_store.py (pytest), and README.md. "
            "Keep it dependency-free."
        ),
        dynamic_agents_enabled=True,
        parallel_coding_enabled=True,
    ),
    "specialist": DynamicScenario(
        name="specialist",
        workspace_name="specialist",
        task=(
            "Fix the bugs in the validator module. The email validator crashes on addresses "
            "with multiple @ symbols, and the tests must all pass. Use the specialist path "
            "to verify your diagnosis before changing code."
        ),
        dynamic_agents_enabled=True,
        parallel_coding_enabled=False,
        setup_workspace=_setup_specialist_workspace,
    ),
    "reflection": DynamicScenario(
        name="reflection",
        workspace_name="reflection",
        task=(
            "Add a 'step' parameter to increment and decrement so they can change by N at a time. "
            "Also add an 'undo' method that reverts the last operation. Update tests and README."
        ),
        dynamic_agents_enabled=True,
        parallel_coding_enabled=True,
        setup_workspace=_setup_reflection_workspace,
    ),
    "budget": DynamicScenario(
        name="budget",
        workspace_name="budget",
        task=(
            "Build a full-featured inventory management system with product CRUD, "
            "category management, stock tracking, CSV import/export, and comprehensive tests."
        ),
        dynamic_agents_enabled=True,
        parallel_coding_enabled=True,
        budget=0.50,
    ),
    "retry": DynamicScenario(
        name="retry",
        workspace_name="retry",
        task=(
            "Create a Python module `roman.py` that converts integers (1-3999) to Roman numerals "
            "and back. Edge cases: 0 raises ValueError, 4000+ raises ValueError, "
            "invalid strings raise ValueError. Include thorough pytest coverage."
        ),
        dynamic_agents_enabled=True,
        parallel_coding_enabled=False,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dynamic-agent experiment scenarios for mAIke.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario to run: baseline, fanout, specialist, reflection, budget, retry, or all.",
    )
    parser.add_argument("--provider", default=None, help="Provider to use. Defaults to the first configured provider.")
    parser.add_argument("--model", default=None, help="Model override.")
    parser.add_argument("--budget", type=float, default=None, help="Override the scenario budget.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/tmp/maike-dynamic-experiments"),
        help="Root directory for experiment workspaces.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete each workspace after a successful run.")
    return parser


def select_scenarios(values: list[str]) -> list[DynamicScenario]:
    requested = [value.strip().lower() for value in values if value.strip()]
    if not requested or "all" in requested:
        return list(SCENARIOS.values())
    invalid = [value for value in requested if value not in SCENARIOS]
    if invalid:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario(s): {', '.join(invalid)}. Valid options: {valid}")
    return [SCENARIOS[value] for value in requested]


async def _run_scenario(
    scenario: DynamicScenario,
    *,
    workspace_root: Path,
    provider: str,
    model: str | None,
    budget_override: float | None,
) -> tuple[str, str, Path]:
    workspace = workspace_root / scenario.workspace_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if scenario.setup_workspace is not None:
        scenario.setup_workspace(workspace)

    resolved_model = resolve_model_name(provider, model)
    result = await run_command(
        task=scenario.task,
        workspace=workspace,
        provider=provider,
        model=resolved_model,
        budget=budget_override if budget_override is not None else scenario.budget or 5.0,
        yes=True,
        dynamic_agents_enabled=scenario.dynamic_agents_enabled,
        parallel_coding_enabled=scenario.parallel_coding_enabled,
    )
    return result.session_id, result.pipeline, workspace


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = select_scenarios(args.scenario)
    provider = args.provider or default_live_provider()

    if not provider_has_key(provider):
        print(
            f"No API key detected for provider '{provider}'. Set the provider key in the environment or .env file.",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for scenario in scenarios:
        try:
            session_id, pipeline, workspace = asyncio.run(
                _run_scenario(
                    scenario,
                    workspace_root=args.workspace_root,
                    provider=provider,
                    model=args.model,
                    budget_override=args.budget,
                )
            )
        except Exception as exc:  # pragma: no cover - live script behavior
            print(f"[FAIL] {scenario.name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(
            f"[PASS] {scenario.name}: provider={provider} model={resolve_model_name(provider, args.model)} "
            f"pipeline={pipeline} session={session_id} workspace={workspace}"
        )
        if args.clean:
            shutil.rmtree(workspace, ignore_errors=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
