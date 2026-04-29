"""Agentic eval cases — multi-file, real-world coding tasks.

These cases test actual agent capabilities:
  - Cross-module debugging (trace failures across files)
  - Feature addition (coordinate changes across 3+ files)
  - Refactoring with test preservation (change API, update all callers)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.helpers import write_text, run_pytest, import_module


# =====================================================================
# Case 1: Cross-Module Debugging
# =====================================================================

_CASE1_TASK = """\
The API tests are failing. Run the test suite, find the root cause bug, and fix it.
The bug is NOT in the tests — it is in the application code. Do NOT modify the tests.
After fixing the bug, verify all tests pass.
"""


def _seed_cross_module_debug(workspace: Path) -> None:
    write_text(workspace / "models.py", '''\
"""Domain models."""

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class User:
    name: str
    email: str
    age: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True


@dataclass
class Product:
    name: str
    price: float
    stock: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
''')

    # BUG: len(name) >= max_length should be len(name) > max_length
    write_text(workspace / "validators.py", '''\
"""Validation functions for domain models."""

MAX_NAME_LENGTH = 50
MIN_AGE = 0
MAX_AGE = 150


def validate_name(name: str, max_length: int = MAX_NAME_LENGTH) -> None:
    """Validate a name string."""
    if not name or not name.strip():
        raise ValueError("Name cannot be empty or whitespace")
    if len(name) >= max_length:
        raise ValueError(f"Name must be less than {max_length} characters, got {len(name)}")


def validate_email(email: str) -> None:
    """Basic email validation."""
    if not email or "@" not in email:
        raise ValueError(f"Invalid email: {email!r}")
    local, domain = email.rsplit("@", 1)
    if not local or not domain or "." not in domain:
        raise ValueError(f"Invalid email: {email!r}")


def validate_age(age: int) -> None:
    """Validate age is within acceptable range."""
    if not isinstance(age, int):
        raise TypeError(f"Age must be an integer, got {type(age).__name__}")
    if age < MIN_AGE or age > MAX_AGE:
        raise ValueError(f"Age must be between {MIN_AGE} and {MAX_AGE}, got {age}")


def validate_price(price: float) -> None:
    """Validate price is positive."""
    if price < 0:
        raise ValueError(f"Price must be non-negative, got {price}")
''')

    write_text(workspace / "database.py", '''\
"""In-memory database store."""

from typing import Any, Optional


class Database:
    def __init__(self):
        self._users: dict[str, dict] = {}
        self._products: dict[str, dict] = {}

    def add_user(self, user_data: dict) -> dict:
        uid = user_data["id"]
        self._users[uid] = user_data
        return user_data

    def get_user(self, uid: str) -> Optional[dict]:
        return self._users.get(uid)

    def list_users(self) -> list[dict]:
        return list(self._users.values())

    def add_product(self, product_data: dict) -> dict:
        pid = product_data["id"]
        self._products[pid] = product_data
        return product_data

    def get_product(self, pid: str) -> Optional[dict]:
        return self._products.get(pid)
''')

    write_text(workspace / "api.py", '''\
"""API route handlers."""

from models import User, Product
from validators import validate_name, validate_email, validate_age, validate_price
from database import Database

db = Database()


def create_user(name: str, email: str, age: int) -> dict:
    """Create a new user after validation."""
    validate_name(name)
    validate_email(email)
    validate_age(age)
    user = User(name=name, email=email, age=age)
    return db.add_user({"id": user.id, "name": user.name, "email": user.email, "age": user.age})


def create_product(name: str, price: float, stock: int) -> dict:
    """Create a new product after validation."""
    validate_name(name)
    validate_price(price)
    product = Product(name=name, price=price, stock=stock)
    return db.add_product({"id": product.id, "name": product.name, "price": product.price, "stock": product.stock})


def get_user(uid: str) -> dict | None:
    return db.get_user(uid)
''')

    write_text(workspace / "test_api.py", '''\
"""API integration tests."""

import pytest
from api import create_user, create_product, get_user


class TestCreateUser:
    def test_basic_user(self):
        result = create_user("Alice", "alice@example.com", 30)
        assert result["name"] == "Alice"
        assert result["email"] == "alice@example.com"
        assert result["age"] == 30

    def test_boundary_name_at_max_length(self):
        """Name exactly at MAX_NAME_LENGTH (50) should be accepted."""
        name = "a" * 50
        result = create_user(name, "test@example.com", 25)
        assert result["name"] == name

    def test_name_over_max_rejected(self):
        with pytest.raises(ValueError, match="less than"):
            create_user("a" * 51, "test@example.com", 25)

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            create_user("", "test@example.com", 25)

    def test_invalid_email_rejected(self):
        with pytest.raises(ValueError, match="Invalid email"):
            create_user("Bob", "not-an-email", 25)

    def test_negative_age_rejected(self):
        with pytest.raises(ValueError, match="between"):
            create_user("Charlie", "charlie@example.com", -1)

    def test_age_over_max_rejected(self):
        with pytest.raises(ValueError, match="between"):
            create_user("Dave", "dave@example.com", 200)


class TestCreateProduct:
    def test_basic_product(self):
        result = create_product("Widget", 9.99, 100)
        assert result["name"] == "Widget"
        assert result["price"] == 9.99

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            create_product("Bad", -1.0, 10)


class TestGetUser:
    def test_get_existing(self):
        user = create_user("Eve", "eve@example.com", 28)
        found = get_user(user["id"])
        assert found is not None
        assert found["name"] == "Eve"

    def test_get_missing(self):
        assert get_user("nonexistent-id") is None
''')

    write_text(workspace / "conftest.py", '''\
"""Shared test fixtures."""

import pytest

@pytest.fixture(autouse=True)
def reset_db():
    """Reset the database before each test."""
    import api
    from database import Database
    api.db = Database()
''')


def _verify_cross_module_debug(workspace: Path) -> None:
    run_pytest(workspace, label="cross-module debug")
    validators = import_module(workspace, "validators")
    # Boundary: exactly at max_length should NOT raise
    validators.validate_name("a" * 50, max_length=50)
    # Over max: should still raise
    try:
        validators.validate_name("a" * 51, max_length=50)
        raise AssertionError("validate_name should reject names over max_length")
    except ValueError:
        pass


# =====================================================================
# Case 2: Feature Addition Across Files
# =====================================================================

_CASE2_TASK = """\
Add JSON export support to the report generator tool.

Requirements:
1. Add a JsonFormatter class in formatters.py that implements the same interface as TextFormatter and CsvFormatter
2. Register it in the FORMATTERS dict with key "json"
3. Update cli.py to accept --format json as a valid option
4. Add test coverage for JSON output in test_report.py (at least 2 tests)
5. The JSON output should be a valid JSON object with keys "title", "rows", and "summary"

Run all tests to verify nothing is broken and the new tests pass.
"""


def _seed_feature_addition(workspace: Path) -> None:
    write_text(workspace / "report.py", '''\
"""Report data model."""

from dataclasses import dataclass, field


@dataclass
class Report:
    title: str
    rows: list[dict] = field(default_factory=list)
    summary: str = ""

    def add_row(self, **kwargs) -> None:
        self.rows.append(kwargs)
''')

    write_text(workspace / "formatters.py", '''\
"""Report formatters — each converts a Report to a string."""

import csv
import io


class TextFormatter:
    def format(self, report) -> str:
        lines = [f"=== {report.title} ===", ""]
        for row in report.rows:
            lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        if report.summary:
            lines.extend(["", f"Summary: {report.summary}"])
        return "\\n".join(lines)


class CsvFormatter:
    def format(self, report) -> str:
        if not report.rows:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=report.rows[0].keys())
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row)
        return output.getvalue()


FORMATTERS = {
    "text": TextFormatter(),
    "csv": CsvFormatter(),
}
''')

    write_text(workspace / "cli.py", '''\
"""CLI for the report generator."""

import argparse
import sys

from report import Report
from formatters import FORMATTERS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate reports")
    parser.add_argument("--title", default="Report", help="Report title")
    parser.add_argument("--format", choices=["text", "csv"], default="text", help="Output format")
    parser.add_argument("--summary", default="", help="Report summary")
    args = parser.parse_args(argv)

    report = Report(title=args.title, summary=args.summary)
    report.add_row(name="Alice", score=95)
    report.add_row(name="Bob", score=87)
    report.add_row(name="Charlie", score=92)

    formatter = FORMATTERS[args.format]
    print(formatter.format(report))


if __name__ == "__main__":
    main()
''')

    write_text(workspace / "test_report.py", '''\
"""Tests for the report generator."""

import subprocess
import sys
import pytest
from report import Report
from formatters import FORMATTERS, TextFormatter, CsvFormatter


class TestReport:
    def test_add_row(self):
        r = Report(title="Test")
        r.add_row(a=1, b=2)
        assert len(r.rows) == 1
        assert r.rows[0] == {"a": 1, "b": 2}

    def test_empty_report(self):
        r = Report(title="Empty")
        assert r.rows == []


class TestTextFormatter:
    def test_basic_format(self):
        r = Report(title="Scores", summary="Done")
        r.add_row(name="Alice", score=95)
        output = TextFormatter().format(r)
        assert "=== Scores ===" in output
        assert "Alice" in output
        assert "Summary: Done" in output


class TestCsvFormatter:
    def test_basic_format(self):
        r = Report(title="Scores")
        r.add_row(name="Alice", score=95)
        r.add_row(name="Bob", score=87)
        output = CsvFormatter().format(r)
        assert "name,score" in output
        assert "Alice,95" in output

    def test_empty_report(self):
        r = Report(title="Empty")
        output = CsvFormatter().format(r)
        assert output == ""


class TestCli:
    def test_text_output(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "--format", "text", "--title", "Test"],
            capture_output=True, text=True, cwd=str(pytest.importorskip("pathlib").Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "=== Test ===" in result.stdout

    def test_csv_output(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "--format", "csv"],
            capture_output=True, text=True, cwd=str(pytest.importorskip("pathlib").Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "name,score" in result.stdout
''')

    write_text(workspace / "README.md", '''\
# Report Generator

A simple report generation tool supporting multiple output formats.

## Usage

```bash
python cli.py --title "My Report" --format text --summary "All done"
python cli.py --format csv
```

## Supported Formats

- **text**: Human-readable text output
- **csv**: Comma-separated values
''')


def _verify_feature_addition(workspace: Path) -> None:
    run_pytest(workspace, label="feature addition")
    formatters = import_module(workspace, "formatters")
    assert "json" in formatters.FORMATTERS, "FORMATTERS must include 'json'"
    # Test the formatter produces valid JSON
    report_mod = import_module(workspace, "report")
    r = report_mod.Report(title="Test", summary="ok")
    r.add_row(name="Alice", score=95)
    output = formatters.FORMATTERS["json"].format(r)
    parsed = json.loads(output)
    assert "title" in parsed
    assert "rows" in parsed
    # Test CLI accepts --format json
    result = subprocess.run(
        [sys.executable, "cli.py", "--format", "json"],
        capture_output=True, text=True, cwd=str(workspace),
    )
    assert result.returncode == 0
    json.loads(result.stdout)  # must be valid JSON


# =====================================================================
# Case 3: Refactor With Test Preservation
# =====================================================================

_CASE3_TASK = """\
Refactor the utils module: add a REQUIRED `encoding: str` parameter to `format_output()`.

Requirements:
1. Change the signature of `format_output(data)` to `format_output(data, encoding)` in utils.py
2. The `encoding` parameter must be REQUIRED (not optional, no default value)
3. Update ALL callers in processor.py, exporter.py, analyzer.py, and pipeline.py to pass encoding="utf-8"
4. Ensure ALL existing tests continue to pass
5. Add a new test in test_utils.py that verifies format_output works with encoding="ascii"

Run all tests to verify nothing is broken.
"""


def _seed_refactor_preservation(workspace: Path) -> None:
    write_text(workspace / "utils.py", '''\
"""Shared utility functions used by all modules."""


def format_output(data: dict) -> str:
    """Format a data dict into a readable string."""
    lines = []
    for key, value in sorted(data.items()):
        lines.append(f"{key}: {value}")
    return "\\n".join(lines)


def parse_input(raw: str) -> dict:
    """Parse a key=value string into a dict."""
    result = {}
    for line in raw.strip().split("\\n"):
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def validate_data(data: dict) -> list[str]:
    """Validate data and return a list of errors (empty = valid)."""
    errors = []
    if not data:
        errors.append("Data cannot be empty")
    for key, value in data.items():
        if not isinstance(key, str):
            errors.append(f"Key must be string, got {type(key).__name__}")
        if value is None:
            errors.append(f"Value for {key!r} cannot be None")
    return errors
''')

    write_text(workspace / "processor.py", '''\
"""Data processor module."""

from utils import format_output, validate_data


def process(data: dict) -> str:
    """Validate and format data."""
    errors = validate_data(data)
    if errors:
        return "ERRORS: " + "; ".join(errors)
    return format_output(data)
''')

    write_text(workspace / "exporter.py", '''\
"""Data exporter module."""

from utils import format_output, parse_input


def export_from_raw(raw: str) -> str:
    """Parse raw input and export as formatted string."""
    data = parse_input(raw)
    return format_output(data)


def export_dict(data: dict) -> str:
    """Export a dict as formatted string."""
    return format_output(data)
''')

    write_text(workspace / "analyzer.py", '''\
"""Data analysis module."""

from utils import format_output, parse_input, validate_data


def analyze(raw: str) -> str:
    """Parse, validate, and format data."""
    data = parse_input(raw)
    errors = validate_data(data)
    if errors:
        return "Analysis failed: " + "; ".join(errors)
    enriched = {**data, "_analyzed": "true", "_item_count": str(len(data))}
    return format_output(enriched)
''')

    write_text(workspace / "pipeline.py", '''\
"""Orchestration pipeline combining all modules."""

from processor import process
from exporter import export_from_raw, export_dict
from analyzer import analyze


def run_pipeline(data: dict | None = None, raw: str | None = None) -> dict:
    """Run the full pipeline and return results."""
    results = {}
    if data:
        results["processed"] = process(data)
        results["exported"] = export_dict(data)
    if raw:
        results["analyzed"] = analyze(raw)
        results["raw_exported"] = export_from_raw(raw)
    return results
''')

    write_text(workspace / "test_utils.py", '''\
"""Tests for utility functions."""

import pytest
from utils import format_output, parse_input, validate_data


class TestFormatOutput:
    def test_basic(self):
        result = format_output({"b": 2, "a": 1})
        assert "a: 1" in result
        assert "b: 2" in result
        # Keys should be sorted
        assert result.index("a:") < result.index("b:")

    def test_empty(self):
        result = format_output({})
        assert result == ""


class TestParseInput:
    def test_basic(self):
        result = parse_input("name=Alice\\nage=30")
        assert result == {"name": "Alice", "age": "30"}

    def test_empty(self):
        result = parse_input("")
        assert result == {}

    def test_strips_whitespace(self):
        result = parse_input("  key = value  ")
        assert result == {"key": "value"}


class TestValidateData:
    def test_valid(self):
        assert validate_data({"a": 1, "b": "two"}) == []

    def test_empty(self):
        errors = validate_data({})
        assert any("empty" in e.lower() for e in errors)

    def test_none_value(self):
        errors = validate_data({"x": None})
        assert any("none" in e.lower() for e in errors)
''')

    write_text(workspace / "test_pipeline.py", '''\
"""Integration tests for the pipeline."""

from pipeline import run_pipeline


class TestPipeline:
    def test_with_dict(self):
        results = run_pipeline(data={"name": "Alice", "age": "30"})
        assert "processed" in results
        assert "exported" in results
        assert "name: Alice" in results["processed"]
        assert "name: Alice" in results["exported"]

    def test_with_raw(self):
        results = run_pipeline(raw="color=blue\\nsize=large")
        assert "analyzed" in results
        assert "raw_exported" in results
        assert "color: blue" in results["analyzed"]

    def test_with_both(self):
        results = run_pipeline(
            data={"x": "1"},
            raw="y=2",
        )
        assert "processed" in results
        assert "analyzed" in results

    def test_empty(self):
        results = run_pipeline()
        assert results == {}
''')


def _verify_refactor_preservation(workspace: Path) -> None:
    run_pytest(workspace, label="refactor preservation")
    utils = import_module(workspace, "utils")
    # format_output with encoding should work
    result = utils.format_output({"a": 1}, encoding="utf-8")
    assert "a: 1" in result
    # Without encoding should raise TypeError (required param)
    try:
        utils.format_output({"a": 1})
        raise AssertionError("format_output() should require 'encoding' parameter")
    except TypeError:
        pass
    # ASCII encoding should work
    result = utils.format_output({"key": "value"}, encoding="ascii")
    assert "key: value" in result


# =====================================================================
# Case Registry
# =====================================================================

AGENTIC_EVAL_CASES: dict[str, EvalCase] = {
    "agentic-cross-module-debug": EvalCase(
        name="agentic-cross-module-debug",
        phases=(EvalPhase(task=_CASE1_TASK),),
        setup_workspace=_seed_cross_module_debug,
        verify_workspace=_verify_cross_module_debug,
        tags=("agentic", "react", "medium"),
        budget=5.0,
        difficulty_weight=2.0,
        expected_modified_files=("validators.py",),
    ),
    "agentic-feature-addition": EvalCase(
        name="agentic-feature-addition",
        phases=(EvalPhase(task=_CASE2_TASK),),
        setup_workspace=_seed_feature_addition,
        verify_workspace=_verify_feature_addition,
        tags=("agentic", "react", "hard"),
        budget=8.0,
        difficulty_weight=3.0,
        expected_modified_files=("formatters.py", "cli.py", "test_report.py"),
    ),
    "agentic-refactor-preservation": EvalCase(
        name="agentic-refactor-preservation",
        phases=(EvalPhase(task=_CASE3_TASK),),
        setup_workspace=_seed_refactor_preservation,
        verify_workspace=_verify_refactor_preservation,
        tags=("agentic", "react", "hard"),
        budget=8.0,
        difficulty_weight=3.0,
        expected_modified_files=("utils.py", "processor.py", "exporter.py", "analyzer.py", "pipeline.py", "test_utils.py"),
    ),
}
