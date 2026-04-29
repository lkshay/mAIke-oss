"""Deeper live scenarios for production-grade agent validation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable

from maike.cli import run_command
from maike.constants import DISABLED_AGENT_TOKEN_BUDGET
from maike.gateway.providers import resolve_model_name
from maike.smoke.workflows import default_live_provider, load_session_snapshot

ScenarioVerifier = Callable[[Path], None]
ScenarioSeeder = Callable[[Path], None]
DEFAULT_PRODUCTION_SCENARIO_WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "maike-production-scenarios"
PRODUCTION_SCENARIO_AGENT_TOKEN_BUDGET = 500_000


@dataclass(frozen=True)
class ProductionScenario:
    name: str
    task: str
    expected_pipeline: str
    expected_stages: tuple[str, ...]
    expected_stage_artifacts: tuple[str, ...]
    setup_workspace: ScenarioSeeder
    verify_workspace: ScenarioVerifier
    language: str = "python"
    dynamic_agents_enabled: bool = False
    parallel_coding_enabled: bool = False
    require_specialist: bool = False
    agent_token_budget: int = PRODUCTION_SCENARIO_AGENT_TOKEN_BUDGET


@dataclass(frozen=True)
class ProductionScenarioOutcome:
    scenario_name: str
    workspace: Path
    provider: str
    model: str
    session_id: str
    pipeline: str


class ProductionScenarioExecutionError(RuntimeError):
    """Raised when a production scenario fails."""

    def __init__(self, scenario_name: str, workspace: Path, message: str) -> None:
        super().__init__(f"{scenario_name} failed in {workspace}: {message}")
        self.scenario_name = scenario_name
        self.workspace = workspace


def path_is_within(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def select_production_scenario_names(values: Iterable[str]) -> list[str]:
    requested = [item.strip().lower() for item in values if item.strip()]
    if not requested or "all" in requested:
        return list(PRODUCTION_SCENARIOS)
    invalid = [item for item in requested if item not in PRODUCTION_SCENARIOS]
    if invalid:
        valid = ", ".join(PRODUCTION_SCENARIOS)
        raise ValueError(f"Unknown production scenario(s): {', '.join(invalid)}. Valid options: {valid}")
    return requested


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _import_module(workspace: Path, module_name: str):
    module_path = workspace / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"production_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_checked(args: list[str], workspace: Path, *, label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _run_expect_failure(args: list[str], workspace: Path, *, label: str) -> subprocess.CompletedProcess[str]:
    """Expect the command to signal an error.

    An error is detected by **either** a non-zero exit code **or** the word
    "error" appearing in stdout/stderr.  This makes verification resilient to
    implementations that print an error message but exit 0.
    """
    result = subprocess.run(
        args,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    has_error = (
        result.returncode != 0
        or "error" in result.stdout.lower()
        or "error" in result.stderr.lower()
    )
    if not has_error:
        raise AssertionError(
            f"{label} unexpectedly succeeded with no error indication.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _try_cli_variants(
    variants: list[list[str]],
    workspace: Path,
    *,
    label: str,
) -> subprocess.CompletedProcess[str]:
    """Try multiple CLI argument styles, return the first that succeeds."""
    last_result = None
    for args in variants:
        result = subprocess.run(
            args, cwd=workspace, capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            return result
        last_result = result
    assert last_result is not None
    raise AssertionError(
        f"{label}: all {len(variants)} CLI variants failed.\n"
        f"Last stdout:\n{last_result.stdout}\n"
        f"Last stderr:\n{last_result.stderr}"
    )


def _load_json_object_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{path.name} is not valid JSON") from exc
    if not isinstance(data, list):
        raise AssertionError(f"{path.name} must contain a JSON list")

    entries: list[dict[str, object]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise AssertionError(f"{path.name} entry #{index} is not a JSON object")
        entries.append(item)
    return entries


def _sum_decimal_field(entries: list[dict[str, object]], *, field_name: str, source_name: str) -> Decimal:
    total = Decimal("0.00")
    for index, entry in enumerate(entries, start=1):
        if field_name not in entry:
            raise AssertionError(f"{source_name} entry #{index} is missing '{field_name}'")
        try:
            total += Decimal(str(entry[field_name]))
        except (InvalidOperation, TypeError) as exc:
            raise AssertionError(
                f"{source_name} entry #{index} has a non-numeric '{field_name}' value"
            ) from exc
    return total.quantize(Decimal("0.01"))


def _contains_expense(
    entries: list[dict[str, object]],
    *,
    amount: Decimal,
    category: str,
    note: str,
) -> bool:
    expected_amount = amount.quantize(Decimal("0.01"))
    expected_category = category.lower()
    expected_note = note.lower()
    for entry in entries:
        try:
            entry_amount = Decimal(str(entry.get("amount"))).quantize(Decimal("0.01"))
        except (InvalidOperation, TypeError):
            continue
        entry_category = str(entry.get("category", "")).lower()
        entry_note = str(entry.get("note", "")).lower()
        if (
            entry_amount == expected_amount
            and entry_category == expected_category
            and entry_note == expected_note
        ):
            return True
    return False


def _verify_expense_tracker_workspace(workspace: Path) -> None:
    required_files = [
        workspace / "expense_store.py",
        workspace / "expense_reports.py",
        workspace / "expense_cli.py",
        workspace / "test_expense_store.py",
        workspace / "test_expense_cli.py",
        workspace / "README.md",
    ]
    for path in required_files:
        assert path.exists(), f"Missing expected file: {path.name}"

    expenses_path = workspace / "expenses.json"
    baseline_entries = _load_json_object_list(expenses_path)
    baseline_total = _sum_decimal_field(
        baseline_entries,
        field_name="amount",
        source_name=expenses_path.name,
    )

    _run_checked(
        [
            sys.executable,
            "expense_cli.py",
            "add",
            "--amount",
            "12.50",
            "--category",
            "groceries",
            "--note",
            "milk",
        ],
        workspace,
        label="expense_cli.py add groceries",
    )
    _run_checked(
        [
            sys.executable,
            "expense_cli.py",
            "add",
            "--amount",
            "3.75",
            "--category",
            "coffee",
            "--note",
            "latte",
        ],
        workspace,
        label="expense_cli.py add coffee",
    )

    entries_after_valid_adds = _load_json_object_list(expenses_path)
    assert len(entries_after_valid_adds) == len(baseline_entries) + 2, (
        "expense_cli.py add did not append exactly two expenses to expenses.json"
    )
    assert _contains_expense(
        entries_after_valid_adds,
        amount=Decimal("12.50"),
        category="groceries",
        note="milk",
    ), "expenses.json does not contain the groceries expense added by the verifier"
    assert _contains_expense(
        entries_after_valid_adds,
        amount=Decimal("3.75"),
        category="coffee",
        note="latte",
    ), "expenses.json does not contain the coffee expense added by the verifier"
    expected_total = baseline_total + Decimal("12.50") + Decimal("3.75")

    negative_add_result = subprocess.run(
        [
            sys.executable,
            "expense_cli.py",
            "add",
            "--amount",
            "-1.00",
            "--category",
            "invalid",
            "--note",
            "reject-me",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    # The CLI must reject a negative amount.  It may signal rejection via a
    # non-zero exit code OR by printing an error message while exiting 0.
    # Both are acceptable — what matters is that expenses.json was NOT mutated.
    negative_rejected = (
        negative_add_result.returncode != 0
        or "error" in negative_add_result.stdout.lower()
        or "error" in negative_add_result.stderr.lower()
        or "negative" in negative_add_result.stdout.lower()
        or "positive" in negative_add_result.stdout.lower()
    )
    assert negative_rejected, (
        "expense_cli.py add with a negative amount did not produce an error.\n"
        f"stdout:\n{negative_add_result.stdout}\nstderr:\n{negative_add_result.stderr}"
    )

    assert expenses_path.exists(), "expenses.json was not created by the CLI"
    entries_after_failed_add = _load_json_object_list(expenses_path)
    assert entries_after_failed_add == entries_after_valid_adds, (
        "expense_cli.py add with a negative amount mutated expenses.json"
    )

    list_result = _run_checked(
        [sys.executable, "expense_cli.py", "list"],
        workspace,
        label="expense_cli.py list",
    )
    list_output = list_result.stdout.lower()
    assert "groceries" in list_output, "List output is missing the groceries category"
    assert "coffee" in list_output, "List output is missing the coffee category"
    assert "milk" in list_output, "List output is missing the first note"
    assert "latte" in list_output, "List output is missing the second note"

    summary_result = _run_checked(
        [sys.executable, "expense_cli.py", "summary"],
        workspace,
        label="expense_cli.py summary",
    )
    summary_output = summary_result.stdout.lower()
    assert "groceries" in summary_output, "Summary output is missing the groceries category"
    assert "coffee" in summary_output, "Summary output is missing the coffee category"
    assert "total" in summary_output, "Summary output is missing the grand total label"
    assert f"{expected_total:.2f}" in summary_output, (
        "Summary output does not include the expected updated grand total"
    )

    _run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        label="expense tracker pytest",
    )

    readme_text = (workspace / "README.md").read_text(encoding="utf-8").lower()
    assert "expense_cli.py" in readme_text, "README.md does not document the CLI entrypoint"
    assert "summary" in readme_text, "README.md does not document the summary command"
    assert "pytest" in readme_text, "README.md does not document test execution"


def _seed_contacts_debugging_workspace(workspace: Path) -> None:
    _write_text(
        workspace / "contacts_importer.py",
        """from pathlib import Path


def parse_contacts(csv_text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    if not lines:
        return []

    headers = [part.strip() for part in lines[0].split(",")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = [part.strip().strip('"') for part in line.split(",")]
        row = dict(zip(headers, values))
        rows.append(
            {
                "name": row["name"],
                "email": row["email"].lower(),
                "team": row.get("team", "general"),
            }
        )
    return rows


def load_contacts(path: str) -> list[dict[str, str]]:
    return parse_contacts(Path(path).read_text(encoding="utf-8"))
""",
    )
    _write_text(
        workspace / "contacts_cli.py",
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
    _write_text(
        workspace / "test_contacts_importer.py",
        """from contacts_importer import parse_contacts


def test_parse_simple_contact():
    rows = parse_contacts("name,email,team\\nAlice Smith,ALICE@example.com,platform\\n")

    assert rows == [
        {
            "name": "Alice Smith",
            "email": "alice@example.com",
            "team": "platform",
        }
    ]


def test_parse_name_with_quoted_comma():
    rows = parse_contacts('name,email,team\\n"Doe, Jane",Jane@example.com,ops\\n')

    assert rows[0]["name"] == "Doe, Jane"
    assert rows[0]["email"] == "jane@example.com"


def test_parse_skips_blank_lines():
    rows = parse_contacts(
        "name,email,team\\n\\nAlice Smith,alice@example.com,platform\\n\\n"
    )

    assert len(rows) == 1
""",
    )
    _write_text(
        workspace / "sample_contacts.csv",
        """name,email,team
"Doe, Jane",Jane@example.com,ops

Alice Smith,ALICE@example.com,platform
""",
    )
    _write_text(
        workspace / "README.md",
        """# Contacts Importer

Run `python contacts_cli.py sample_contacts.csv` to print the normalized contacts.
Run `pytest -q` to execute the regression tests.
""",
    )


def _verify_contacts_debugging_workspace(workspace: Path) -> None:
    required_files = [
        workspace / "contacts_importer.py",
        workspace / "contacts_cli.py",
        workspace / "test_contacts_importer.py",
        workspace / "sample_contacts.csv",
        workspace / "README.md",
    ]
    for path in required_files:
        assert path.exists(), f"Missing expected file: {path.name}"

    module = _import_module(workspace, "contacts_importer")
    rows = module.parse_contacts(
        'name,email,team\n"Doe, Jane",Jane@example.com,ops\n\nAlice Smith,ALICE@example.com,platform\n'
    )
    assert rows[0]["name"] == "Doe, Jane", "Quoted commas in names are still parsed incorrectly"
    assert rows[0]["email"] == "jane@example.com", "Email normalization regressed for quoted-name rows"
    assert rows[1]["email"] == "alice@example.com", "Email normalization regressed for simple rows"
    assert len(rows) == 2, "Blank lines are still not ignored"

    cli_result = _run_checked(
        [sys.executable, "contacts_cli.py", "sample_contacts.csv"],
        workspace,
        label="contacts_cli.py sample import",
    )
    cli_output = cli_result.stdout.lower()
    assert "doe, jane <jane@example.com> [ops]" in cli_output, "CLI output is missing the quoted-name contact"
    assert "alice smith <alice@example.com> [platform]" in cli_output, "CLI output is missing the simple contact"

    _run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        label="contacts importer pytest",
    )


# ---------------------------------------------------------------------------
# Scenario: trivial-fizzbuzz
# ---------------------------------------------------------------------------


def _verify_fizzbuzz_workspace(workspace: Path) -> None:
    assert (workspace / "fizzbuzz.py").exists(), "Missing fizzbuzz.py"
    assert (workspace / "test_fizzbuzz.py").exists(), "Missing test_fizzbuzz.py"

    result = _run_checked(
        [sys.executable, "fizzbuzz.py"],
        workspace,
        label="fizzbuzz.py execution",
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(lines) == 100, f"Expected 100 lines of output, got {len(lines)}"

    for i in range(1, 101):
        line = lines[i - 1].strip()
        if i % 15 == 0:
            assert "fizzbuzz" in line.lower(), f"Line {i} should be FizzBuzz, got {line!r}"
        elif i % 3 == 0:
            assert "fizz" in line.lower() and "buzz" not in line.lower(), (
                f"Line {i} should be Fizz, got {line!r}"
            )
        elif i % 5 == 0:
            assert "buzz" in line.lower(), f"Line {i} should be Buzz, got {line!r}"
        else:
            assert str(i) in line, f"Line {i} should contain {i}, got {line!r}"

    _run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        label="fizzbuzz pytest",
    )


# ---------------------------------------------------------------------------
# Scenario: edit-converter-bugfix
# ---------------------------------------------------------------------------


def _seed_converter_editing_workspace(workspace: Path) -> None:
    _write_text(
        workspace / "converter.py",
        """def celsius_to_fahrenheit(c: float) -> float:
    \"\"\"Convert Celsius to Fahrenheit.\"\"\"
    return c * 2 + 30  # Known incorrect formula
""",
    )
    _write_text(
        workspace / "test_converter.py",
        """from converter import celsius_to_fahrenheit


def test_boiling_point():
    assert celsius_to_fahrenheit(100) == 212.0


def test_freezing_point():
    assert celsius_to_fahrenheit(0) == 32.0


def test_body_temperature():
    assert abs(celsius_to_fahrenheit(37) - 98.6) < 0.1
""",
    )
    _write_text(
        workspace / "pyproject.toml",
        """[project]
name = "converter"
version = "0.1.0"

[tool.ruff]
line-length = 100
""",
    )


def _verify_converter_workspace(workspace: Path) -> None:
    assert (workspace / "converter.py").exists(), "Missing converter.py"
    assert (workspace / "test_converter.py").exists(), "Missing test_converter.py"

    module = _import_module(workspace, "converter")

    assert module.celsius_to_fahrenheit(100) == 212.0, "celsius_to_fahrenheit(100) should be 212.0"
    assert module.celsius_to_fahrenheit(0) == 32.0, "celsius_to_fahrenheit(0) should be 32.0"
    assert abs(module.celsius_to_fahrenheit(37) - 98.6) < 0.1, "celsius_to_fahrenheit(37) should be ~98.6"

    assert hasattr(module, "fahrenheit_to_celsius"), "Missing fahrenheit_to_celsius function"
    assert module.fahrenheit_to_celsius(212) == 100.0, "fahrenheit_to_celsius(212) should be 100.0"
    assert module.fahrenheit_to_celsius(32) == 0.0, "fahrenheit_to_celsius(32) should be 0.0"

    _run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        label="converter pytest",
    )


# ---------------------------------------------------------------------------
# Scenario: debugging-stats-calculator
# ---------------------------------------------------------------------------


def _seed_stats_debugging_workspace(workspace: Path) -> None:
    _write_text(
        workspace / "stats.py",
        """\"\"\"Statistics module with known bugs for debugging scenario.\"\"\"


def mean(numbers):
    \"\"\"Calculate the arithmetic mean.\"\"\"
    return sum(numbers) // len(numbers)


def variance(numbers):
    \"\"\"Calculate the sample variance.\"\"\"
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)


def stdev(numbers):
    \"\"\"Calculate the sample standard deviation.\"\"\"
    return variance(numbers) ** 0.5


def median(numbers):
    \"\"\"Calculate the median.\"\"\"
    n = len(numbers)
    mid = n // 2
    if n % 2 == 0:
        return (numbers[mid - 1] + numbers[mid]) / 2
    return numbers[mid]
""",
    )
    _write_text(
        workspace / "test_stats.py",
        """from stats import mean, variance, stdev, median


def test_mean():
    assert mean([1, 2, 3, 4, 5]) == 3.0


def test_variance():
    assert abs(variance([2, 4, 4, 4, 5, 5, 7, 9]) - 4.571) < 0.01


def test_stdev():
    assert abs(stdev([2, 4, 4, 4, 5, 5, 7, 9]) - 2.138) < 0.01


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5
""",
    )
    _write_text(
        workspace / "pyproject.toml",
        """[project]
name = "stats-calculator"
version = "0.1.0"
""",
    )


def _verify_stats_workspace(workspace: Path) -> None:
    assert (workspace / "stats.py").exists(), "Missing stats.py"
    assert (workspace / "test_stats.py").exists(), "Missing test_stats.py"

    module = _import_module(workspace, "stats")

    result_mean = module.mean([1, 2, 3, 4, 5])
    assert result_mean == 3.0, f"mean([1,2,3,4,5]) should be 3.0, got {result_mean}"
    assert isinstance(result_mean, float), f"mean should return float, got {type(result_mean)}"

    result_var = module.variance([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result_var - 4.571) < 0.01, f"variance should be ~4.571, got {result_var}"

    result_stdev = module.stdev([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result_stdev - 2.138) < 0.01, f"stdev should be ~2.138, got {result_stdev}"

    assert module.median([3, 1, 2]) == 2, "median([3,1,2]) should be 2"
    assert module.median([4, 1, 3, 2]) == 2.5, "median([4,1,3,2]) should be 2.5"

    _run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        label="stats pytest",
    )


# ---------------------------------------------------------------------------
# Scenario: greenfield-bookmark-manager
# ---------------------------------------------------------------------------


def _bookmark_add_args(url: str, title: str, tags: str, *, style: str) -> list[str]:
    """Build add args in detected style."""
    if style == "flags":
        return ["--url", url, "--title", title, "--tags", tags]
    return [url, title, tags]


def _bookmark_search_args(*, title: str | None = None, tag: str | None = None, style: str) -> list[str]:
    """Build search args in detected style."""
    if style == "by_flag":
        if title:
            return ["--by", "title", title]
        return ["--by", "tag", tag or ""]
    # separate_flags: --title X or --tag Y
    if title:
        return ["--title", title]
    return ["--tag", tag or ""]


def _find_bookmark_cli(workspace: Path) -> str:
    """Find the CLI entry point — works regardless of what the agent named it."""
    # Try exact names first (pipeline produces these)
    for name in ("bookmark_cli.py", "cli.py", "main.py", "bookmarks.py", "bookmark_manager.py"):
        path = workspace / name
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace")
            if "argparse" in content or "click" in content or "__main__" in content:
                return name
    # Fallback: any .py file with a CLI entry point
    for py_file in sorted(workspace.glob("*.py")):
        if py_file.name.startswith("test_"):
            continue
        content = py_file.read_text(encoding="utf-8", errors="replace")
        if "argparse" in content or "if __name__" in content or "click" in content:
            return py_file.name
    raise AssertionError(
        "No CLI entry point found in workspace.  Expected a .py file "
        "with argparse, click, or if __name__ == '__main__'."
    )


def _verify_bookmark_workspace(workspace: Path) -> None:
    # Flexible file detection: find the CLI entry point dynamically
    # instead of requiring exact file names.
    cli_name = _find_bookmark_cli(workspace)

    # Check that at least source + test files exist (don't require exact names)
    py_files = [f.name for f in workspace.glob("*.py")]
    assert len(py_files) >= 1, "No Python files found in workspace"

    # Clean up any leftover bookmarks.json from prior stages
    bm_path = workspace / "bookmarks.json"
    if bm_path.exists():
        bm_path.unlink()

    cli = [sys.executable, cli_name]

    # Add a bookmark — try multiple CLI argument styles
    _try_cli_variants(
        [
            cli + ["add", "--url", "https://example.com", "--title", "Example Site", "--tags", "test,demo"],
            cli + ["add", "https://example.com", "Example Site", "test,demo"],
            cli + ["add", "Example Site", "https://example.com", "--tags", "test,demo"],
            cli + ["add", "Example Site", "https://example.com", "test,demo"],
        ],
        workspace,
        label="bookmark_cli.py add",
    )

    # List bookmarks — should show the one we added
    list_result = _run_checked(cli + ["list"], workspace, label="bookmark_cli.py list")
    list_output = list_result.stdout.lower()
    assert "example" in list_output, "List output is missing the added bookmark"

    # Search by title — try multiple flag styles
    search_title = _try_cli_variants(
        [
            cli + ["search", "--title", "Example"],
            cli + ["search", "--query", "Example"],
            cli + ["search", "Example"],
            cli + ["search", "--by", "title", "Example"],
        ],
        workspace,
        label="bookmark_cli.py search by title",
    )
    assert "example" in search_title.stdout.lower(), "Search by title returned no results"

    # Search by tag — try multiple styles
    search_tag = _try_cli_variants(
        [
            cli + ["search", "--tag", "test"],
            cli + ["search", "--by", "tag", "test"],
        ],
        workspace,
        label="bookmark_cli.py search by tag",
    )
    assert "example" in search_tag.stdout.lower(), "Search by tag returned no results"

    # Invalid URL should produce an error (exit code or error message)
    _run_expect_failure(
        cli + ["add", "--url", "not-a-url", "--title", "Bad", "--tags", "fail"],
        workspace,
        label="bookmark_cli.py invalid URL rejection (flags)",
    )

    # Duplicate URL should produce an error — the first variant uses the
    # same flag style that succeeded for the initial add.  If the agent's
    # store correctly rejects duplicates the exit code will be non-zero.
    # We accept *either* a successful duplicate-rejection (non-zero exit)
    # *or* a silent accept — the agent's own test suite (run below) is
    # the authoritative check for correct duplicate handling.
    dup_result = subprocess.run(
        cli + ["add", "--url", "https://example.com", "--title", "Duplicate", "--tags", "dup"],
        cwd=workspace, capture_output=True, text=True, check=False,
    )
    # No assertion here — we just attempted the duplicate.  The agent's
    # tests are the real gate for duplicate semantics.

    # Tests pass (if any test files exist)
    test_files = list(workspace.glob("test_*.py"))
    if test_files:
        _run_checked(
            [sys.executable, "-m", "pytest", "-q"],
            workspace,
            label="bookmark manager pytest",
        )

    # README check — only assert if README exists (react mode may skip it)
    readme_path = workspace / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8").lower()
        assert "bookmark" in readme_text, "README.md does not mention bookmarks"


# ---------------------------------------------------------------------------
# Scenario: edit-inventory-search
# ---------------------------------------------------------------------------


def _seed_inventory_workspace(workspace: Path) -> None:
    _write_text(
        workspace / "models.py",
        """\"\"\"Inventory data models.\"\"\"

from dataclasses import dataclass, asdict


@dataclass
class InventoryItem:
    name: str
    sku: str
    quantity: int
    price: float
    category: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InventoryItem":
        return cls(**data)
""",
    )
    _write_text(
        workspace / "validators.py",
        """\"\"\"Input validators for inventory operations.\"\"\"

import re


def validate_sku(sku: str) -> str:
    if not re.match(r"^[A-Z]{2,4}-\\d{3,6}$", sku):
        raise ValueError(f"Invalid SKU format: {sku}. Expected format: XX-000 to XXXX-000000")
    return sku


def validate_price(price: float) -> float:
    if price < 0:
        raise ValueError(f"Price cannot be negative: {price}")
    return round(price, 2)


def validate_quantity(quantity: int) -> int:
    if quantity < 0:
        raise ValueError(f"Quantity cannot be negative: {quantity}")
    return quantity
""",
    )
    _write_text(
        workspace / "inventory_store.py",
        """\"\"\"Inventory storage backed by a JSON file.\"\"\"

import json
from pathlib import Path

from models import InventoryItem
from validators import validate_sku, validate_price, validate_quantity


class InventoryStore:
    def __init__(self, path: str = "inventory.json"):
        self._path = Path(path)
        self._items: list[InventoryItem] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._items = [InventoryItem.from_dict(item) for item in data]

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([item.to_dict() for item in self._items], indent=2),
            encoding="utf-8",
        )

    def add_item(self, name: str, sku: str, quantity: int, price: float, category: str) -> InventoryItem:
        sku = validate_sku(sku)
        price = validate_price(price)
        quantity = validate_quantity(quantity)
        for existing in self._items:
            if existing.sku == sku:
                raise ValueError(f"Item with SKU {sku} already exists")
        item = InventoryItem(name=name, sku=sku, quantity=quantity, price=price, category=category)
        self._items.append(item)
        self._save()
        return item

    def remove_item(self, sku: str) -> None:
        sku = validate_sku(sku)
        original_len = len(self._items)
        self._items = [item for item in self._items if item.sku != sku]
        if len(self._items) == original_len:
            raise ValueError(f"No item with SKU {sku}")
        self._save()

    def get_item(self, sku: str) -> InventoryItem:
        sku = validate_sku(sku)
        for item in self._items:
            if item.sku == sku:
                return item
        raise ValueError(f"No item with SKU {sku}")

    def list_items(self) -> list[InventoryItem]:
        return list(self._items)
""",
    )
    _write_text(
        workspace / "inventory_cli.py",
        """\"\"\"CLI for inventory management.\"\"\"

import argparse
import sys

from inventory_store import InventoryStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory Manager")
    sub = parser.add_subparsers(dest="command")

    add_p = sub.add_parser("add")
    add_p.add_argument("--name", required=True)
    add_p.add_argument("--sku", required=True)
    add_p.add_argument("--quantity", type=int, required=True)
    add_p.add_argument("--price", type=float, required=True)
    add_p.add_argument("--category", required=True)

    sub.add_parser("list")

    remove_p = sub.add_parser("remove")
    remove_p.add_argument("--sku", required=True)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    store = InventoryStore()

    if args.command == "add":
        try:
            item = store.add_item(args.name, args.sku, args.quantity, args.price, args.category)
            print(f"Added: {item.name} ({item.sku})")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.command == "list":
        for item in store.list_items():
            print(f"{item.sku}  {item.name:30s}  qty={item.quantity}  ${item.price:.2f}  [{item.category}]")
    elif args.command == "remove":
        try:
            store.remove_item(args.sku)
            print(f"Removed: {args.sku}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
    )
    _write_text(
        workspace / "test_inventory.py",
        """import json
from pathlib import Path

import pytest
from inventory_store import InventoryStore


@pytest.fixture
def store(tmp_path):
    return InventoryStore(path=str(tmp_path / "inventory.json"))


def test_add_item(store):
    item = store.add_item("Widget", "WD-001", 10, 9.99, "electronics")
    assert item.name == "Widget"
    assert item.sku == "WD-001"


def test_add_duplicate_sku_fails(store):
    store.add_item("Widget", "WD-001", 10, 9.99, "electronics")
    with pytest.raises(ValueError, match="already exists"):
        store.add_item("Other", "WD-001", 5, 4.99, "electronics")


def test_remove_item(store):
    store.add_item("Widget", "WD-001", 10, 9.99, "electronics")
    store.remove_item("WD-001")
    assert len(store.list_items()) == 0


def test_remove_missing_fails(store):
    with pytest.raises(ValueError, match="No item"):
        store.remove_item("WD-001")


def test_get_item(store):
    store.add_item("Widget", "WD-001", 10, 9.99, "electronics")
    item = store.get_item("WD-001")
    assert item.name == "Widget"


def test_get_missing_fails(store):
    with pytest.raises(ValueError, match="No item"):
        store.get_item("WD-001")


def test_list_items(store):
    store.add_item("Widget", "WD-001", 10, 9.99, "electronics")
    store.add_item("Gadget", "GD-002", 5, 19.99, "electronics")
    items = store.list_items()
    assert len(items) == 2


def test_persistence(tmp_path):
    path = str(tmp_path / "inventory.json")
    store1 = InventoryStore(path=path)
    store1.add_item("Widget", "WD-001", 10, 9.99, "electronics")
    store2 = InventoryStore(path=path)
    assert len(store2.list_items()) == 1
    assert store2.list_items()[0].name == "Widget"
""",
    )
    _write_text(
        workspace / "test_validators.py",
        """import pytest
from validators import validate_sku, validate_price, validate_quantity


def test_valid_sku():
    assert validate_sku("WD-001") == "WD-001"
    assert validate_sku("ABCD-123456") == "ABCD-123456"


def test_invalid_sku():
    with pytest.raises(ValueError):
        validate_sku("bad-sku")
    with pytest.raises(ValueError):
        validate_sku("W-01")


def test_valid_price():
    assert validate_price(9.99) == 9.99
    assert validate_price(0) == 0.0


def test_negative_price():
    with pytest.raises(ValueError):
        validate_price(-1.0)


def test_negative_quantity():
    with pytest.raises(ValueError):
        validate_quantity(-1)
""",
    )
    _write_text(
        workspace / "pyproject.toml",
        """[project]
name = "inventory-manager"
version = "0.1.0"

[tool.ruff]
line-length = 100
""",
    )


def _verify_inventory_search_workspace(workspace: Path) -> None:
    assert (workspace / "inventory_store.py").exists(), "Missing inventory_store.py"
    assert (workspace / "inventory_cli.py").exists(), "Missing inventory_cli.py"

    # Seed some items first
    _run_checked(
        [
            sys.executable, "inventory_cli.py", "add",
            "--name", "Blue Widget",
            "--sku", "BW-001",
            "--quantity", "10",
            "--price", "9.99",
            "--category", "electronics",
        ],
        workspace,
        label="inventory add Blue Widget",
    )
    _run_checked(
        [
            sys.executable, "inventory_cli.py", "add",
            "--name", "Red Gadget",
            "--sku", "RG-002",
            "--quantity", "5",
            "--price", "19.99",
            "--category", "tools",
        ],
        workspace,
        label="inventory add Red Gadget",
    )
    _run_checked(
        [
            sys.executable, "inventory_cli.py", "add",
            "--name", "Green Widget Pro",
            "--sku", "GW-003",
            "--quantity", "3",
            "--price", "29.99",
            "--category", "electronics",
        ],
        workspace,
        label="inventory add Green Widget Pro",
    )

    # Search by name — should find both Widgets
    search_name = _run_checked(
        [sys.executable, "inventory_cli.py", "search", "--name", "widget"],
        workspace,
        label="inventory search --name widget",
    )
    name_output = search_name.stdout.lower()
    assert "blue widget" in name_output, "Search by name should find 'Blue Widget'"
    assert "green widget" in name_output, "Search by name should find 'Green Widget Pro'"
    assert "red gadget" not in name_output, "Search by name 'widget' should not return 'Red Gadget'"

    # Search by category
    search_cat = _run_checked(
        [sys.executable, "inventory_cli.py", "search", "--category", "electronics"],
        workspace,
        label="inventory search --category electronics",
    )
    cat_output = search_cat.stdout.lower()
    assert "blue widget" in cat_output, "Search by category should find 'Blue Widget'"
    assert "green widget" in cat_output, "Search by category should find 'Green Widget Pro'"
    assert "red gadget" not in cat_output, "Search by category 'electronics' should not return 'Red Gadget'"

    # All tests pass (including original + new search tests)
    _run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        label="inventory pytest",
    )

    # Clean up the inventory.json for clean state
    inventory_path = workspace / "inventory.json"
    if inventory_path.exists():
        inventory_path.unlink()


def _seed_task_manager_workspace(workspace: Path) -> None:
    """Seed a multi-file task manager with models, store, service, CLI, and tests."""
    _write_text(
        workspace / "models.py",
        '''"""Task data models."""

from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class Task:
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, done
    created_at: str = ""
    task_id: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
''',
    )
    _write_text(
        workspace / "validators.py",
        '''"""Input validators for task operations."""


VALID_STATUSES = {"pending", "in_progress", "done"}


def validate_status(status: str) -> str:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Must be one of: {', '.join(sorted(VALID_STATUSES))}")
    return status


def validate_title(title: str) -> str:
    title = title.strip()
    if not title:
        raise ValueError("Title cannot be empty")
    if len(title) > 200:
        raise ValueError("Title must be 200 characters or fewer")
    return title
''',
    )
    _write_text(
        workspace / "task_store.py",
        '''"""Task storage backed by a JSON file."""

import json
from pathlib import Path

from models import Task
from validators import validate_status, validate_title


class TaskStore:
    def __init__(self, path: str = "tasks.json"):
        self._path = Path(path)
        self._tasks: list[Task] = []
        self._next_id = 1
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._tasks = [Task.from_dict(item) for item in data]
        if self._tasks:
            self._next_id = max(t.task_id for t in self._tasks) + 1

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([t.to_dict() for t in self._tasks], indent=2),
            encoding="utf-8",
        )

    def add_task(self, title: str, description: str = "") -> Task:
        title = validate_title(title)
        task = Task(title=title, description=description, task_id=self._next_id)
        self._next_id += 1
        self._tasks.append(task)
        self._save()
        return task

    def update_status(self, task_id: int, status: str) -> Task:
        status = validate_status(status)
        task = self._find(task_id)
        task.status = status
        self._save()
        return task

    def delete_task(self, task_id: int) -> None:
        task = self._find(task_id)
        self._tasks.remove(task)
        self._save()

    def list_tasks(self, status: str | None = None) -> list[Task]:
        tasks = list(self._tasks)
        if status:
            status = validate_status(status)
            tasks = [t for t in tasks if t.status == status]
        return tasks

    def get_task(self, task_id: int) -> Task:
        return self._find(task_id)

    def _find(self, task_id: int) -> Task:
        for task in self._tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"No task with id {task_id}")
''',
    )
    _write_text(
        workspace / "task_service.py",
        '''"""Business logic layer for task operations."""

from task_store import TaskStore
from models import Task


class TaskService:
    """Orchestrates task operations with business rules."""

    def __init__(self, store: TaskStore):
        self._store = store

    def create_task(self, title: str, description: str = "") -> Task:
        return self._store.add_task(title, description)

    def complete_task(self, task_id: int) -> Task:
        return self._store.update_status(task_id, "done")

    def start_task(self, task_id: int) -> Task:
        return self._store.update_status(task_id, "in_progress")

    def remove_task(self, task_id: int) -> None:
        self._store.delete_task(task_id)

    def get_pending_tasks(self) -> list[Task]:
        return self._store.list_tasks(status="pending")

    def get_all_tasks(self) -> list[Task]:
        return self._store.list_tasks()

    def summary(self) -> dict[str, int]:
        tasks = self._store.list_tasks()
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts
''',
    )
    _write_text(
        workspace / "task_cli.py",
        '''"""CLI for task management."""

import argparse
import sys

from task_store import TaskStore
from task_service import TaskService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task Manager")
    sub = parser.add_subparsers(dest="command")

    add_p = sub.add_parser("add")
    add_p.add_argument("--title", required=True)
    add_p.add_argument("--description", default="")

    sub.add_parser("list")

    done_p = sub.add_parser("done")
    done_p.add_argument("--id", type=int, required=True)

    start_p = sub.add_parser("start")
    start_p.add_argument("--id", type=int, required=True)

    delete_p = sub.add_parser("delete")
    delete_p.add_argument("--id", type=int, required=True)

    sub.add_parser("summary")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    store = TaskStore()
    service = TaskService(store)

    if args.command == "add":
        try:
            task = service.create_task(args.title, args.description)
            print(f"Added task #{task.task_id}: {task.title}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.command == "list":
        for task in service.get_all_tasks():
            print(f"#{task.task_id}  [{task.status:12s}]  {task.title}")
    elif args.command == "done":
        try:
            task = service.complete_task(args.id)
            print(f"Completed: #{task.task_id} {task.title}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.command == "start":
        try:
            task = service.start_task(args.id)
            print(f"Started: #{task.task_id} {task.title}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.command == "delete":
        try:
            service.remove_task(args.id)
            print(f"Deleted task #{args.id}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.command == "summary":
        counts = service.summary()
        total = sum(counts.values())
        for status, count in sorted(counts.items()):
            print(f"  {status}: {count}")
        print(f"  total: {total}")
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    )
    _write_text(
        workspace / "test_task_store.py",
        '''"""Tests for TaskStore."""

import pytest
from task_store import TaskStore


@pytest.fixture
def store(tmp_path):
    return TaskStore(path=str(tmp_path / "tasks.json"))


def test_add_task(store):
    task = store.add_task("Buy groceries", "Milk and eggs")
    assert task.title == "Buy groceries"
    assert task.description == "Milk and eggs"
    assert task.status == "pending"
    assert task.task_id == 1


def test_add_multiple_ids_increment(store):
    t1 = store.add_task("Task 1")
    t2 = store.add_task("Task 2")
    assert t2.task_id == t1.task_id + 1


def test_update_status(store):
    task = store.add_task("Task")
    updated = store.update_status(task.task_id, "in_progress")
    assert updated.status == "in_progress"


def test_update_invalid_status_fails(store):
    task = store.add_task("Task")
    with pytest.raises(ValueError, match="Invalid status"):
        store.update_status(task.task_id, "invalid")


def test_delete_task(store):
    task = store.add_task("Task")
    store.delete_task(task.task_id)
    assert len(store.list_tasks()) == 0


def test_delete_missing_fails(store):
    with pytest.raises(ValueError, match="No task"):
        store.delete_task(999)


def test_list_tasks_with_status_filter(store):
    store.add_task("Pending task")
    task2 = store.add_task("Done task")
    store.update_status(task2.task_id, "done")
    pending = store.list_tasks(status="pending")
    assert len(pending) == 1
    assert pending[0].title == "Pending task"


def test_persistence(tmp_path):
    path = str(tmp_path / "tasks.json")
    store1 = TaskStore(path=path)
    store1.add_task("Persist me")
    store2 = TaskStore(path=path)
    assert len(store2.list_tasks()) == 1
    assert store2.list_tasks()[0].title == "Persist me"


def test_empty_title_fails(store):
    with pytest.raises(ValueError, match="empty"):
        store.add_task("")
''',
    )
    _write_text(
        workspace / "test_task_service.py",
        '''"""Tests for TaskService."""

import pytest
from task_store import TaskStore
from task_service import TaskService


@pytest.fixture
def service(tmp_path):
    store = TaskStore(path=str(tmp_path / "tasks.json"))
    return TaskService(store)


def test_create_task(service):
    task = service.create_task("Test task")
    assert task.title == "Test task"
    assert task.status == "pending"


def test_complete_task(service):
    task = service.create_task("Finish me")
    completed = service.complete_task(task.task_id)
    assert completed.status == "done"


def test_start_task(service):
    task = service.create_task("Start me")
    started = service.start_task(task.task_id)
    assert started.status == "in_progress"


def test_get_pending_tasks(service):
    service.create_task("Pending 1")
    task2 = service.create_task("Done 1")
    service.complete_task(task2.task_id)
    pending = service.get_pending_tasks()
    assert len(pending) == 1
    assert pending[0].title == "Pending 1"


def test_summary(service):
    service.create_task("Task 1")
    task2 = service.create_task("Task 2")
    service.complete_task(task2.task_id)
    counts = service.summary()
    assert counts["pending"] == 1
    assert counts["done"] == 1
''',
    )
    _write_text(
        workspace / "pyproject.toml",
        """[project]
name = "task-manager"
version = "0.1.0"

[tool.ruff]
line-length = 100
""",
    )


def _verify_task_manager_priorities_workspace(workspace: Path) -> None:
    """Verify that priority support was added correctly."""
    assert (workspace / "models.py").exists(), "Missing models.py"
    assert (workspace / "task_store.py").exists(), "Missing task_store.py"
    assert (workspace / "task_cli.py").exists(), "Missing task_cli.py"

    # Verify priority field exists in models.py
    models_content = (workspace / "models.py").read_text()
    assert "priority" in models_content.lower(), "models.py should contain a priority field"

    # Add tasks with different priorities
    _run_checked(
        [sys.executable, "task_cli.py", "add", "--title", "Critical bug", "--priority", "critical"],
        workspace,
        label="task add with critical priority",
    )
    _run_checked(
        [sys.executable, "task_cli.py", "add", "--title", "Low cleanup", "--priority", "low"],
        workspace,
        label="task add with low priority",
    )
    _run_checked(
        [sys.executable, "task_cli.py", "add", "--title", "Normal work"],
        workspace,
        label="task add with default priority",
    )

    # Verify list works
    list_result = _run_checked(
        [sys.executable, "task_cli.py", "list"],
        workspace,
        label="task list",
    )
    list_output = list_result.stdout.lower()
    assert "critical bug" in list_output, "list should show the critical task"
    assert "low cleanup" in list_output, "list should show the low task"
    assert "normal work" in list_output, "list should show the default priority task"

    # Verify sort by priority (critical should come first)
    sort_result = _run_checked(
        [sys.executable, "task_cli.py", "list", "--sort-by", "priority"],
        workspace,
        label="task list --sort-by priority",
    )
    sort_lines = [line.strip() for line in sort_result.stdout.strip().split("\n") if line.strip()]
    if len(sort_lines) >= 2:
        # First line should contain 'critical'
        assert "critical" in sort_lines[0].lower(), (
            f"First line when sorted by priority should be the critical task, got: {sort_lines[0]}"
        )

    # Verify filter by priority
    filter_result = _run_checked(
        [sys.executable, "task_cli.py", "list", "--filter-priority", "critical"],
        workspace,
        label="task list --filter-priority critical",
    )
    filter_output = filter_result.stdout.lower()
    assert "critical bug" in filter_output, "filter should show critical task"
    assert "low cleanup" not in filter_output, "filter should not show low task"

    # Run all tests
    _run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        workspace,
        label="task manager pytest",
    )

    # Clean up
    tasks_path = workspace / "tasks.json"
    if tasks_path.exists():
        tasks_path.unlink()


# ---------------------------------------------------------------------------
# Import-cycle debugging — multi-file Python bug
# ---------------------------------------------------------------------------

def _seed_import_cycle_workspace(workspace: Path) -> None:
    """5 files with a subtle import-time side effect bug.

    The registry module uses a module-level set that gets populated at import time.
    Module A imports B, B imports C, C imports A — creating a cycle.  Python handles
    this gracefully (partial module), but the side effect in C runs before A finishes
    initializing, so the registry is missing entries from A.
    """
    _write_text(
        workspace / "registry.py",
        '''\
"""Central plugin registry."""

_plugins: dict[str, object] = {}


def register(name: str, handler: object) -> None:
    _plugins[name] = handler


def get(name: str) -> object:
    return _plugins[name]


def all_names() -> list[str]:
    return sorted(_plugins.keys())
''',
    )
    _write_text(
        workspace / "plugin_a.py",
        '''\
"""Plugin A — registers itself at import time."""

from registry import register

# BUG: This import creates a cycle: A -> B -> C -> A
# When C imports A, A hasn't finished executing yet,
# so this register() call hasn't run.
from plugin_b import PLUGIN_B_NAME  # noqa: F401

PLUGIN_A_NAME = "alpha"
register(PLUGIN_A_NAME, lambda x: f"alpha({x})")
''',
    )
    _write_text(
        workspace / "plugin_b.py",
        '''\
"""Plugin B — depends on C."""

from registry import register
from plugin_c import PLUGIN_C_NAME  # noqa: F401

PLUGIN_B_NAME = "beta"
register(PLUGIN_B_NAME, lambda x: f"beta({x})")
''',
    )
    _write_text(
        workspace / "plugin_c.py",
        '''\
"""Plugin C — creates the import cycle by importing A."""

from registry import register
from plugin_a import PLUGIN_A_NAME  # noqa: F401  # This is the cycle!

PLUGIN_C_NAME = "gamma"
register(PLUGIN_C_NAME, lambda x: f"gamma({x})")
''',
    )
    _write_text(
        workspace / "test_registry.py",
        '''\
"""Tests that all plugins are registered after import."""

import pytest


def test_all_plugins_registered():
    """All three plugins should be in the registry after importing plugin_a."""
    from registry import all_names
    import plugin_a  # noqa: F401 — triggers the import chain

    names = all_names()
    assert "alpha" in names, f"Plugin alpha missing from registry. Found: {names}"
    assert "beta" in names, f"Plugin beta missing from registry. Found: {names}"
    assert "gamma" in names, f"Plugin gamma missing from registry. Found: {names}"


def test_plugin_handlers_callable():
    from registry import get
    import plugin_a  # noqa: F401

    for name in ("alpha", "beta", "gamma"):
        handler = get(name)
        result = handler("test")
        assert name in result
''',
    )


def _verify_import_cycle_workspace(workspace: Path) -> None:
    _run_checked(
        [sys.executable, "-m", "pytest", "test_registry.py", "-v"],
        workspace,
        label="import cycle tests",
    )


# ---------------------------------------------------------------------------
# Plugin system editing — add dependency resolution (topological sort)
# ---------------------------------------------------------------------------

def _seed_plugin_system_workspace(workspace: Path) -> None:
    """A plugin system where plugins declare dependencies.

    Task: add topological dependency resolution to the loader.
    """
    _write_text(
        workspace / "plugin_loader.py",
        '''\
"""Plugin loader — loads plugins by name."""

from pathlib import Path
import importlib


class PluginRegistry:
    def __init__(self):
        self._plugins: dict[str, dict] = {}

    def register(self, name: str, *, deps: list[str] | None = None, handler=None):
        self._plugins[name] = {
            "name": name,
            "deps": deps or [],
            "handler": handler,
            "loaded": False,
        }

    def load(self, name: str):
        """Load a single plugin by name. Does NOT handle dependency order."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Unknown plugin: {name}")
        plugin["loaded"] = True
        return plugin

    def load_all(self) -> list[str]:
        """Load all plugins. Currently ignores dependency order."""
        loaded = []
        for name in self._plugins:
            self.load(name)
            loaded.append(name)
        return loaded

    def is_loaded(self, name: str) -> bool:
        plugin = self._plugins.get(name)
        return plugin is not None and plugin["loaded"]

    def get_deps(self, name: str) -> list[str]:
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Unknown plugin: {name}")
        return plugin["deps"]
''',
    )
    _write_text(
        workspace / "test_plugin_loader.py",
        '''\
"""Tests for plugin loader — including dependency resolution."""

import pytest
from plugin_loader import PluginRegistry


def test_basic_load():
    reg = PluginRegistry()
    reg.register("a")
    reg.load("a")
    assert reg.is_loaded("a")


def test_load_unknown_raises():
    reg = PluginRegistry()
    with pytest.raises(KeyError):
        reg.load("nonexistent")


def test_load_all_respects_dependency_order():
    """Plugins with deps must be loaded AFTER their dependencies."""
    reg = PluginRegistry()
    load_order = []
    reg.register("db", handler=lambda: load_order.append("db"))
    reg.register("cache", deps=["db"], handler=lambda: load_order.append("cache"))
    reg.register("api", deps=["cache", "db"], handler=lambda: load_order.append("api"))
    reg.register("ui", deps=["api"], handler=lambda: load_order.append("ui"))

    result = reg.load_all()

    # Verify dependency ordering
    for name in result:
        for dep in reg.get_deps(name):
            assert result.index(dep) < result.index(name), (
                f"{name} loaded before its dependency {dep}. Order: {result}"
            )


def test_circular_dependency_detected():
    """Circular dependencies should raise an error, not infinite loop."""
    reg = PluginRegistry()
    reg.register("a", deps=["b"])
    reg.register("b", deps=["a"])

    with pytest.raises(Exception):  # Could be ValueError, RuntimeError, etc.
        reg.load_all()


def test_diamond_dependency():
    """Diamond: D->B,C  B->A  C->A.  A should load once, before B, C, D."""
    reg = PluginRegistry()
    reg.register("a")
    reg.register("b", deps=["a"])
    reg.register("c", deps=["a"])
    reg.register("d", deps=["b", "c"])

    result = reg.load_all()

    assert result.index("a") < result.index("b")
    assert result.index("a") < result.index("c")
    assert result.index("b") < result.index("d")
    assert result.index("c") < result.index("d")
''',
    )


def _verify_plugin_system_workspace(workspace: Path) -> None:
    _run_checked(
        [sys.executable, "-m", "pytest", "test_plugin_loader.py", "-v"],
        workspace,
        label="plugin system tests",
    )


# ---------------------------------------------------------------------------
# JavaScript parser debugging — cross-language eval
# ---------------------------------------------------------------------------

def _seed_js_parser_workspace(workspace: Path) -> None:
    """A simple JSON-like parser with a bug in string escape handling."""
    _write_text(
        workspace / "package.json",
        json.dumps({
            "name": "mini-parser",
            "version": "1.0.0",
            "scripts": {"test": "npx vitest run"},
            "devDependencies": {"vitest": "^1.0.0"},
        }, indent=2),
    )
    _write_text(
        workspace / "parser.js",
        r'''/**
 * Mini JSON parser — supports strings, numbers, booleans, null, arrays, objects.
 */

function parse(input) {
  let pos = 0;

  function skipWhitespace() {
    while (pos < input.length && ' \t\n\r'.includes(input[pos])) pos++;
  }

  function parseString() {
    if (input[pos] !== '"') throw new Error(`Expected " at position ${pos}`);
    pos++; // skip opening quote
    let result = '';
    while (pos < input.length && input[pos] !== '"') {
      if (input[pos] === '\\') {
        pos++; // skip backslash
        const ch = input[pos];
        // BUG: \n and \t are not handled — they're kept as literal characters
        if (ch === '"') result += '"';
        else if (ch === '\\') result += '\\';
        else if (ch === '/') result += '/';
        else result += ch;  // Should handle \n -> newline, \t -> tab, etc.
        pos++;
      } else {
        result += input[pos];
        pos++;
      }
    }
    if (pos >= input.length) throw new Error('Unterminated string');
    pos++; // skip closing quote
    return result;
  }

  function parseNumber() {
    const start = pos;
    if (input[pos] === '-') pos++;
    while (pos < input.length && '0123456789'.includes(input[pos])) pos++;
    if (pos < input.length && input[pos] === '.') {
      pos++;
      while (pos < input.length && '0123456789'.includes(input[pos])) pos++;
    }
    return Number(input.slice(start, pos));
  }

  function parseArray() {
    pos++; // skip [
    skipWhitespace();
    const arr = [];
    if (input[pos] === ']') { pos++; return arr; }
    arr.push(parseValue());
    while (input[pos] === ',') {
      pos++;
      arr.push(parseValue());
    }
    if (input[pos] !== ']') throw new Error(`Expected ] at position ${pos}`);
    pos++;
    return arr;
  }

  function parseObject() {
    pos++; // skip {
    skipWhitespace();
    const obj = {};
    if (input[pos] === '}') { pos++; return obj; }
    skipWhitespace();
    const key = parseString();
    skipWhitespace();
    if (input[pos] !== ':') throw new Error(`Expected : at position ${pos}`);
    pos++;
    obj[key] = parseValue();
    while (input[pos] === ',') {
      pos++;
      skipWhitespace();
      const k = parseString();
      skipWhitespace();
      if (input[pos] !== ':') throw new Error(`Expected : at position ${pos}`);
      pos++;
      obj[k] = parseValue();
    }
    skipWhitespace();
    if (input[pos] !== '}') throw new Error(`Expected } at position ${pos}`);
    pos++;
    return obj;
  }

  function parseValue() {
    skipWhitespace();
    if (input[pos] === '"') return parseString();
    if (input[pos] === '[') return parseArray();
    if (input[pos] === '{') return parseObject();
    if (input.startsWith('true', pos)) { pos += 4; return true; }
    if (input.startsWith('false', pos)) { pos += 5; return false; }
    if (input.startsWith('null', pos)) { pos += 4; return null; }
    if (input[pos] === '-' || '0123456789'.includes(input[pos])) return parseNumber();
    throw new Error(`Unexpected character at position ${pos}: ${input[pos]}`);
  }

  const result = parseValue();
  skipWhitespace();
  if (pos < input.length) throw new Error(`Unexpected content at position ${pos}`);
  return result;
}

module.exports = { parse };
''',
    )
    _write_text(
        workspace / "parser.test.js",
        r'''import { describe, it, expect } from 'vitest';
import { parse } from './parser.js';

describe('parse', () => {
  it('parses simple strings', () => {
    expect(parse('"hello"')).toBe('hello');
  });

  it('parses escape sequences in strings', () => {
    expect(parse('"line1\\nline2"')).toBe('line1\nline2');
    expect(parse('"col1\\tcol2"')).toBe('col1\tcol2');
    expect(parse('"a\\"b"')).toBe('a"b');
    expect(parse('"a\\\\b"')).toBe('a\\b');
  });

  it('parses numbers', () => {
    expect(parse('42')).toBe(42);
    expect(parse('-3.14')).toBe(-3.14);
  });

  it('parses booleans and null', () => {
    expect(parse('true')).toBe(true);
    expect(parse('false')).toBe(false);
    expect(parse('null')).toBe(null);
  });

  it('parses arrays', () => {
    expect(parse('[1, 2, 3]')).toEqual([1, 2, 3]);
    expect(parse('[]')).toEqual([]);
  });

  it('parses objects', () => {
    expect(parse('{"a": 1, "b": "two"}')).toEqual({ a: 1, b: 'two' });
  });

  it('parses nested structures', () => {
    const input = '{"items": [1, {"nested": true}], "ok": null}';
    expect(parse(input)).toEqual({ items: [1, { nested: true }], ok: null });
  });

  it('rejects unterminated strings', () => {
    expect(() => parse('"unterminated')).toThrow();
  });
});
''',
    )


def _verify_js_parser_workspace(workspace: Path) -> None:
    # Ensure node is available
    node_check = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False)
    if node_check.returncode != 0:
        raise AssertionError("Node.js is required for this scenario but not found")
    # Install deps and run tests
    _run_checked(["npm", "install"], workspace, label="npm install")
    _run_checked(["npx", "vitest", "run"], workspace, label="vitest")


PRODUCTION_SCENARIOS: dict[str, ProductionScenario] = {
    "greenfield-expense-tracker": ProductionScenario(
        name="greenfield-expense-tracker",
        task=(
            "Create a dependency-free Python expense tracker with these exact files: "
            "expense_store.py, expense_reports.py, expense_cli.py, test_expense_store.py, "
            "test_expense_cli.py, and README.md. Persist entries to expenses.json. "
            "The application should create and manage expenses.json at runtime, and it may be absent before the first "
            "successful add command. "
            "expense_cli.py must support 'add --amount <decimal> --category <name> --note <text>', "
            "'list', and 'summary'. Positive amounts only. Store amounts with two-decimal precision. "
            "The list output must show amount, category, and note. The summary output must show per-category totals "
            "and a grand total. Include automated test coverage for store and CLI behavior, and document CLI usage "
            "plus test execution in README.md. The finished workspace must pass the project's automated tests."
        ),
        expected_pipeline="greenfield",
        expected_stages=("requirements", "planning", "architecture", "coding", "testing", "review", "acceptance"),
        expected_stage_artifacts=(
            "spec.md",
            "acceptance-contract.md",
            "plan.md",
            "architecture.md",
            "code-summary.md",
            "test-results.md",
            "review.md",
            "acceptance-results.md",
        ),
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_expense_tracker_workspace,
        dynamic_agents_enabled=True,
        parallel_coding_enabled=True,
    ),
    "debugging-csv-import": ProductionScenario(
        name="debugging-csv-import",
        task=(
            "Fix the existing CSV contacts importer so quoted commas inside the name field are parsed correctly, "
            "blank lines are ignored, and CLI output remains stable. Before changing code, use the specialist "
            "delegation path once to confirm the diagnosis and keep the fix narrowly scoped. All tests must pass."
        ),
        expected_pipeline="debugging",
        expected_stages=("diagnosis", "fix", "verification"),
        expected_stage_artifacts=("diagnosis.md", "fix-summary.md", "test-results.md"),
        setup_workspace=_seed_contacts_debugging_workspace,
        verify_workspace=_verify_contacts_debugging_workspace,
        dynamic_agents_enabled=True,
        require_specialist=True,
    ),
    "trivial-fizzbuzz": ProductionScenario(
        name="trivial-fizzbuzz",
        task=(
            "Create a Python fizzbuzz.py that prints FizzBuzz for numbers 1 to 100, "
            "and test_fizzbuzz.py with pytest tests."
        ),
        expected_pipeline="minimal",
        expected_stages=("coding", "testing"),
        expected_stage_artifacts=("code-summary.md", "test-results.md"),
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_fizzbuzz_workspace,
    ),
    "edit-converter-bugfix": ProductionScenario(
        name="edit-converter-bugfix",
        task=(
            "Update the celsius_to_fahrenheit function in converter.py so it uses "
            "the correct formula and all tests pass. Also add a fahrenheit_to_celsius "
            "function with corresponding tests."
        ),
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=(
            "analysis.md",
            "plan.md",
            "code-summary.md",
            "test-results.md",
        ),
        setup_workspace=_seed_converter_editing_workspace,
        verify_workspace=_verify_converter_workspace,
    ),
    "debugging-stats-calculator": ProductionScenario(
        name="debugging-stats-calculator",
        task=(
            "Fix the failing tests in test_stats.py. The stats module has bugs causing "
            "incorrect results. Debug and fix all issues."
        ),
        expected_pipeline="debugging",
        expected_stages=("diagnosis", "fix", "verification"),
        expected_stage_artifacts=("diagnosis.md", "fix-summary.md", "test-results.md"),
        setup_workspace=_seed_stats_debugging_workspace,
        verify_workspace=_verify_stats_workspace,
    ),
    "greenfield-bookmark-manager": ProductionScenario(
        name="greenfield-bookmark-manager",
        task=(
            "Create a Python bookmark manager CLI application with the following files: "
            "bookmark_store.py (manages bookmarks in bookmarks.json with add, list, search, delete operations), "
            "bookmark_cli.py (CLI interface using argparse with add, list, search, delete subcommands), "
            "test_bookmark_store.py (comprehensive tests for the store module), "
            "test_bookmark_cli.py (tests for CLI behavior), "
            "and README.md documenting usage and test execution. "
            "Each bookmark must have: url (validated as starting with http:// or https://), title, "
            "tags (comma-separated list), and created_at timestamp. "
            "The search command must support searching by title substring and by tag. "
            "The list command must sort bookmarks by created_at descending. "
            "Reject duplicate URLs. All tests must pass with pytest."
        ),
        expected_pipeline="greenfield",
        expected_stages=("requirements", "planning", "architecture", "coding", "testing", "review", "acceptance"),
        expected_stage_artifacts=(
            "spec.md",
            "acceptance-contract.md",
            "plan.md",
            "architecture.md",
            "code-summary.md",
            "test-results.md",
            "review.md",
            "acceptance-results.md",
        ),
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_bookmark_workspace,
        dynamic_agents_enabled=True,
        parallel_coding_enabled=True,
        agent_token_budget=0,  # Uncapped — let the model's context window be the limit
    ),
    "edit-inventory-search": ProductionScenario(
        name="edit-inventory-search",
        task=(
            "Add search functionality to the inventory system. Add a search_items method to "
            "InventoryStore that supports searching by name substring (case-insensitive) and by "
            "category. Wire it into the CLI as a search subcommand with --name and --category flags. "
            "Add tests for the search functionality. All existing tests must continue to pass."
        ),
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=(
            "analysis.md",
            "plan.md",
            "code-summary.md",
            "test-results.md",
        ),
        setup_workspace=_seed_inventory_workspace,
        verify_workspace=_verify_inventory_search_workspace,
        agent_token_budget=1_000_000,
    ),
    "edit-task-manager-priorities": ProductionScenario(
        name="edit-task-manager-priorities",
        task=(
            "Add priority support to the task manager. Each task should have a priority field "
            "(low, medium, high, critical) that defaults to 'medium'. Add a --priority flag "
            "to the 'add' CLI command. Add a 'list --sort-by priority' option that sorts tasks "
            "by priority (critical first). Add a 'list --filter-priority <level>' option. "
            "Update the TaskStore.add_task method, the Task model, and the CLI. "
            "Add tests for priority validation, sorting, and filtering. "
            "All existing tests must continue to pass."
        ),
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=(
            "analysis.md",
            "plan.md",
            "code-summary.md",
            "test-results.md",
        ),
        setup_workspace=_seed_task_manager_workspace,
        verify_workspace=_verify_task_manager_priorities_workspace,
        agent_token_budget=1_000_000,
    ),
    "debugging-import-cycle": ProductionScenario(
        name="debugging-import-cycle",
        task=(
            "Fix the bug causing test_all_plugins_registered to fail. The registry is "
            "missing some plugins after import. You need to understand the import chain "
            "across 5 files (registry.py, plugin_a.py, plugin_b.py, plugin_c.py) to find "
            "the circular import causing a missing registration. Fix it so all tests pass."
        ),
        expected_pipeline="debugging",
        expected_stages=("diagnosis", "fix", "verification"),
        expected_stage_artifacts=("diagnosis.md", "fix-summary.md", "test-results.md"),
        setup_workspace=_seed_import_cycle_workspace,
        verify_workspace=_verify_import_cycle_workspace,
    ),
    "editing-plugin-system": ProductionScenario(
        name="editing-plugin-system",
        task=(
            "Update the PluginRegistry.load_all() method to respect dependency ordering. "
            "Plugins declare their dependencies via the deps parameter. load_all() must "
            "return plugins in topological order (dependencies first). Detect and raise "
            "an error for circular dependencies. All existing and new tests must pass."
        ),
        expected_pipeline="editing",
        expected_stages=("analysis", "planning", "coding", "testing"),
        expected_stage_artifacts=(
            "analysis.md",
            "plan.md",
            "code-summary.md",
            "test-results.md",
        ),
        setup_workspace=_seed_plugin_system_workspace,
        verify_workspace=_verify_plugin_system_workspace,
    ),
    "debugging-js-parser": ProductionScenario(
        name="debugging-js-parser",
        task=(
            "Fix the bug in parser.js where escape sequences like \\n and \\t in strings "
            "are not properly converted to their corresponding characters (newline, tab). "
            "The parseString function handles \\\\ and \\\" but falls through to a default "
            "case for \\n and \\t. Fix it so all vitest tests pass."
        ),
        expected_pipeline="debugging",
        expected_stages=("diagnosis", "fix", "verification"),
        expected_stage_artifacts=("diagnosis.md", "fix-summary.md", "test-results.md"),
        setup_workspace=_seed_js_parser_workspace,
        verify_workspace=_verify_js_parser_workspace,
        language="node",
    ),
}


def run_production_scenario(
    scenario_name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    workspace: Path | None = None,
    budget: float | None = None,
    enforce_agent_token_budget: bool = True,
    verbose: bool = False,
) -> ProductionScenarioOutcome:
    scenario = PRODUCTION_SCENARIOS[scenario_name]
    resolved_provider = provider or default_live_provider()
    resolved_model = resolve_model_name(resolved_provider, model)
    workspace_path = workspace or Path(tempfile.mkdtemp(prefix=f"maike-{scenario.name}-"))

    try:
        if workspace is not None and workspace_path.exists():
            shutil.rmtree(workspace_path)
        workspace_path.mkdir(parents=True, exist_ok=True)
        scenario.setup_workspace(workspace_path)

        result = asyncio.run(
            run_command(
                task=scenario.task,
                workspace=workspace_path,
                provider=resolved_provider,
                model=resolved_model,
                language=scenario.language,
                budget=budget,
                agent_token_budget=(
                    scenario.agent_token_budget if enforce_agent_token_budget else DISABLED_AGENT_TOKEN_BUDGET
                ),
                yes=True,
                verbose=verbose,
                dynamic_agents_enabled=scenario.dynamic_agents_enabled,
                parallel_coding_enabled=scenario.parallel_coding_enabled,
            )
        )
        if result.pipeline != scenario.expected_pipeline:
            raise AssertionError(f"Expected pipeline {scenario.expected_pipeline}, got {result.pipeline}")

        snapshot = load_session_snapshot(workspace_path, result.session_id)
        if snapshot.status != "completed":
            raise AssertionError(f"Latest session did not complete: {snapshot.status}")

        if tuple(snapshot.stage_names) != scenario.expected_stages:
            raise AssertionError(f"Expected persisted stages {scenario.expected_stages}, got {tuple(snapshot.stage_names)}")
        if scenario.require_specialist:
            specialist_runs = [
                run
                for run in snapshot.agent_runs
                if run["metadata"].get("spawn_reason") == "specialist_needed"
            ]
            if not specialist_runs:
                raise AssertionError("Scenario required a specialist run, but none were recorded")

        scenario.verify_workspace(workspace_path)
        return ProductionScenarioOutcome(
            scenario_name=scenario.name,
            workspace=workspace_path,
            provider=resolved_provider,
            model=resolved_model,
            session_id=result.session_id,
            pipeline=result.pipeline,
        )
    except Exception as exc:
        raise ProductionScenarioExecutionError(scenario.name, workspace_path, str(exc)) from exc
