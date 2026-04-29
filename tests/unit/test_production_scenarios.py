import importlib.util
from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from maike.smoke.production_scenarios import (
    DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT,
    PRODUCTION_SCENARIO_AGENT_TOKEN_BUDGET,
    PRODUCTION_SCENARIOS,
    path_is_within,
    run_production_scenario,
    select_production_scenario_names,
)
from maike.constants import DISABLED_AGENT_TOKEN_BUDGET


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_select_production_scenario_names_supports_all_and_explicit_values():
    assert select_production_scenario_names(["all"]) == list(PRODUCTION_SCENARIOS)
    assert select_production_scenario_names(["greenfield-expense-tracker"]) == ["greenfield-expense-tracker"]
    assert select_production_scenario_names(["debugging-csv-import"]) == ["debugging-csv-import"]


def test_select_production_scenario_names_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown production scenario"):
        select_production_scenario_names(["unknown"])


def test_production_scenario_registry_has_expected_pipelines():
    assert PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_pipeline == "greenfield"
    assert PRODUCTION_SCENARIOS["debugging-csv-import"].expected_pipeline == "debugging"
    assert PRODUCTION_SCENARIOS["debugging-csv-import"].require_specialist is True
    assert PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_stages == (
        "requirements",
        "planning",
        "architecture",
        "coding",
        "testing",
        "review",
        "acceptance",
    )


def test_default_production_workspace_root_uses_system_tmp():
    assert DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT == Path(tempfile.gettempdir()) / "maike-production-scenarios"
    assert PRODUCTION_SCENARIO_AGENT_TOKEN_BUDGET == 500_000


def test_path_is_within_detects_nested_paths(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert path_is_within(repo_root, repo_root) is True
    assert path_is_within(repo_root, repo_root / "nested" / "work") is True
    assert path_is_within(repo_root, tmp_path / "outside") is False


def test_greenfield_expense_tracker_verifier_accepts_workspace_with_existing_data(tmp_path):
    _write(
        tmp_path / "expense_store.py",
        """import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def _normalize_amount(value: str) -> str:
    amount = Decimal(value)
    if amount <= 0:
        raise ValueError("amount must be positive")
    return str(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class ExpenseStore:
    def __init__(self, path: str = "expenses.json") -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, amount: str, category: str, note: str) -> dict[str, str]:
        entry = {
            "amount": _normalize_amount(amount),
            "category": category,
            "note": note,
        }
        entries = self.load()
        entries.append(entry)
        self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return entry
""",
    )
    _write(
        tmp_path / "expense_reports.py",
        """from decimal import Decimal


def category_totals(entries: list[dict[str, str]]) -> dict[str, str]:
    totals: dict[str, Decimal] = {}
    for entry in entries:
        totals[entry["category"]] = totals.get(entry["category"], Decimal("0.00")) + Decimal(entry["amount"])
    return {category: f"{amount:.2f}" for category, amount in totals.items()}


def grand_total(entries: list[dict[str, str]]) -> str:
    total = sum((Decimal(entry["amount"]) for entry in entries), Decimal("0.00"))
    return f"{total:.2f}"
""",
    )
    _write(
        tmp_path / "expense_cli.py",
        """import argparse

from expense_reports import category_totals, grand_total
from expense_store import ExpenseStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--amount", required=True)
    add_parser.add_argument("--category", required=True)
    add_parser.add_argument("--note", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ExpenseStore()

    if args.command == "add":
        store.add(args.amount, args.category, args.note)
        print("added")
        return 0
    if args.command == "list":
        for index, entry in enumerate(store.load(), start=1):
            print(f"{index}. {entry['amount']} {entry['category']} {entry['note']}")
        return 0
    if args.command == "summary":
        entries = store.load()
        for category, amount in category_totals(entries).items():
            print(f"{category}: {amount}")
        print(f"total: {grand_total(entries)}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
""",
    )
    _write(
        tmp_path / "test_expense_store.py",
        """from expense_store import ExpenseStore
import pytest


def test_add_persists_entries(tmp_path):
    store = ExpenseStore(tmp_path / "expenses.json")
    entry = store.add("12.50", "groceries", "milk")

    assert entry["amount"] == "12.50"
    assert store.load()[0]["category"] == "groceries"


def test_negative_amount_is_rejected(tmp_path):
    store = ExpenseStore(tmp_path / "expenses.json")

    with pytest.raises(ValueError):
        store.add("-1.00", "bad", "nope")
""",
    )
    _write(
        tmp_path / "test_expense_cli.py",
        """from pathlib import Path
import json
import subprocess
import sys


CLI_PATH = Path(__file__).with_name("expense_cli.py")


def test_summary_command(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "add",
            "--amount",
            "5.00",
            "--category",
            "coffee",
            "--note",
            "latte",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0

    summary = subprocess.run(
        [sys.executable, str(CLI_PATH), "summary"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert summary.returncode == 0
    assert "coffee: 5.00" in summary.stdout.lower()


def test_add_writes_json(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "add",
            "--amount",
            "7.25",
            "--category",
            "groceries",
            "--note",
            "fruit",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    data = json.loads((tmp_path / "expenses.json").read_text(encoding="utf-8"))
    assert data[0]["note"] == "fruit"
""",
    )
    _write(
        tmp_path / "README.md",
        """# Expense Tracker

Run `python expense_cli.py add --amount 12.50 --category groceries --note milk`.
Run `python expense_cli.py list` or `python expense_cli.py summary`.
Run `pytest -q` for the test suite.
""",
    )
    _write(
        tmp_path / "expenses.json",
        """[
  {
    "amount": "10.50",
    "category": "food",
    "note": "lunch"
  },
  {
    "amount": "20.00",
    "category": "transport",
    "note": "train"
  }
]
""",
    )

    PRODUCTION_SCENARIOS["greenfield-expense-tracker"].verify_workspace(tmp_path)


def test_debugging_csv_import_verifier_accepts_passing_workspace(tmp_path):
    _write(
        tmp_path / "contacts_importer.py",
        """import csv
from pathlib import Path


def parse_contacts(csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(line for line in csv_text.splitlines() if line.strip())
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append(
            {
                "name": (row.get("name") or "").strip(),
                "email": (row.get("email") or "").strip().lower(),
                "team": (row.get("team") or "general").strip() or "general",
            }
        )
    return rows


def load_contacts(path: str) -> list[dict[str, str]]:
    return parse_contacts(Path(path).read_text(encoding="utf-8"))
""",
    )
    _write(
        tmp_path / "contacts_cli.py",
        """import sys

from contacts_importer import load_contacts


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: contacts_cli.py <csv-path>")
        return 1

    for contact in load_contacts(args[0]):
        print(f"{contact['name']} <{contact['email']}> [{contact['team']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
    )
    _write(
        tmp_path / "test_contacts_importer.py",
        """from contacts_importer import parse_contacts


def test_parse_name_with_quoted_comma():
    rows = parse_contacts('name,email,team\\n"Doe, Jane",Jane@example.com,ops\\n')

    assert rows[0]["name"] == "Doe, Jane"
    assert rows[0]["email"] == "jane@example.com"


def test_blank_lines_are_ignored():
    rows = parse_contacts(
        "name,email,team\\n\\nAlice Smith,alice@example.com,platform\\n\\n"
    )

    assert len(rows) == 1
""",
    )
    _write(
        tmp_path / "sample_contacts.csv",
        """name,email,team
"Doe, Jane",Jane@example.com,ops

Alice Smith,ALICE@example.com,platform
""",
    )
    _write(
        tmp_path / "README.md",
        """# Contacts Importer

Run `python contacts_cli.py sample_contacts.csv`.
Run `pytest -q` for the regression suite.
""",
    )

    PRODUCTION_SCENARIOS["debugging-csv-import"].verify_workspace(tmp_path)


def test_run_production_scenario_passes_agent_token_budget(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_run_command(**kwargs):
        captured.update(kwargs)
        return type(
            "Result",
            (),
            {
                "pipeline": "greenfield",
                "session_id": "session-1",
            },
        )()

    def fake_load_session_snapshot(workspace, session_id):
        assert workspace == tmp_path
        assert session_id == "session-1"
        return type(
            "Snapshot",
            (),
            {
                "status": "completed",
                "stage_names": PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_stages,
                "artifact_names": list(PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_stage_artifacts),
                "agent_runs": [],
            },
        )()

    monkeypatch.setattr("maike.smoke.production_scenarios.run_command", fake_run_command)
    monkeypatch.setattr("maike.smoke.production_scenarios.load_session_snapshot", fake_load_session_snapshot)
    monkeypatch.setitem(
        PRODUCTION_SCENARIOS,
        "greenfield-expense-tracker",
        replace(
            PRODUCTION_SCENARIOS["greenfield-expense-tracker"],
            verify_workspace=lambda workspace: None,
        ),
    )

    outcome = run_production_scenario(
        "greenfield-expense-tracker",
        provider="gemini",
        model="gemini-2.5-pro",
        workspace=tmp_path,
    )

    assert outcome.pipeline == "greenfield"
    assert captured["agent_token_budget"] == PRODUCTION_SCENARIO_AGENT_TOKEN_BUDGET


def test_run_production_scenario_can_skip_agent_token_budget_check(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_run_command(**kwargs):
        captured.update(kwargs)
        return type(
            "Result",
            (),
            {
                "pipeline": "greenfield",
                "session_id": "session-1",
            },
        )()

    def fake_load_session_snapshot(workspace, session_id):
        assert workspace == tmp_path
        assert session_id == "session-1"
        return type(
            "Snapshot",
            (),
            {
                "status": "completed",
                "stage_names": PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_stages,
                "artifact_names": list(PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_stage_artifacts),
                "agent_runs": [],
            },
        )()

    monkeypatch.setattr("maike.smoke.production_scenarios.run_command", fake_run_command)
    monkeypatch.setattr("maike.smoke.production_scenarios.load_session_snapshot", fake_load_session_snapshot)
    monkeypatch.setitem(
        PRODUCTION_SCENARIOS,
        "greenfield-expense-tracker",
        replace(
            PRODUCTION_SCENARIOS["greenfield-expense-tracker"],
            verify_workspace=lambda workspace: None,
        ),
    )

    outcome = run_production_scenario(
        "greenfield-expense-tracker",
        provider="gemini",
        model="gemini-2.5-pro",
        workspace=tmp_path,
        enforce_agent_token_budget=False,
    )

    assert outcome.pipeline == "greenfield"
    assert captured["agent_token_budget"] == DISABLED_AGENT_TOKEN_BUDGET


def test_run_production_scenario_forwards_verbose_flag(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_run_command(**kwargs):
        captured.update(kwargs)
        return type(
            "Result",
            (),
            {
                "pipeline": "greenfield",
                "session_id": "session-1",
            },
        )()

    def fake_load_session_snapshot(workspace, session_id):
        assert workspace == tmp_path
        assert session_id == "session-1"
        return type(
            "Snapshot",
            (),
            {
                "status": "completed",
                "stage_names": PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_stages,
                "artifact_names": list(PRODUCTION_SCENARIOS["greenfield-expense-tracker"].expected_stage_artifacts),
                "agent_runs": [],
            },
        )()

    monkeypatch.setattr("maike.smoke.production_scenarios.run_command", fake_run_command)
    monkeypatch.setattr("maike.smoke.production_scenarios.load_session_snapshot", fake_load_session_snapshot)
    monkeypatch.setitem(
        PRODUCTION_SCENARIOS,
        "greenfield-expense-tracker",
        replace(
            PRODUCTION_SCENARIOS["greenfield-expense-tracker"],
            verify_workspace=lambda workspace: None,
        ),
    )

    outcome = run_production_scenario(
        "greenfield-expense-tracker",
        provider="gemini",
        model="gemini-2.5-pro",
        workspace=tmp_path,
        verbose=True,
    )

    assert outcome.pipeline == "greenfield"
    assert captured["verbose"] is True


def test_run_production_scenarios_script_parser_accepts_verbose_flag():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_production_scenarios.py"
    spec = importlib.util.spec_from_file_location("run_production_scenarios_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(["--verbose"])

    assert args.verbose is True
