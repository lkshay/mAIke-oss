"""Hard agentic eval cases — designed to stress agent capabilities.

These cases are significantly harder than the standard agentic cases:
  - Case 1: specsynth live repo (28K LOC, 12+ file navigation)
  - Case 2: LabelGuard TypeScript test suite creation (cross-language)
  - Case 3: Event-driven pipeline with 3 cascading bugs (1000 LOC, 10 files)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.helpers import write_text, run_pytest


# =====================================================================
# Case 3: Event-Driven Pipeline with Cascading Bugs
# =====================================================================

_CASE3_TASK = """\
The pipeline integration tests are failing. There are multiple bugs across \
different modules that interact with each other. Read the test output \
carefully, trace the failures to their root causes, and fix ALL bugs.

The bugs are in the application code, NOT in the tests. Do NOT modify \
any test files (test_pipeline.py, test_transforms.py).

After fixing, ALL tests must pass.
"""


def _seed_event_pipeline(workspace: Path) -> None:
    """Seed a 10-file event-driven data pipeline with 3 interconnected bugs."""

    write_text(workspace / "events.py", '''\
"""Event bus for pub/sub communication between pipeline components."""

from __future__ import annotations
from typing import Any, Callable


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable) -> None:
        """Register a handler for an event."""
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Remove a handler."""
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event to all registered handlers.

        BUG 3: Exceptions from handlers are silently swallowed.
        When the metrics handler crashes (division by zero on empty stats),
        the pipeline appears to succeed but metrics are wrong.
        """
        for handler in self._handlers.get(event, []):
            try:
                handler(event, data)
            except Exception:
                pass  # BUG: silently swallows exceptions

    def handler_count(self, event: str) -> int:
        return len(self._handlers.get(event, []))
''')

    write_text(workspace / "config.py", '''\
"""Pipeline configuration."""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    stages: list[str] = field(default_factory=lambda: [
        "clean", "normalize", "validate", "enrich"
    ])
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 0.1
    strict_validation: bool = True
''')

    write_text(workspace / "transforms.py", '''\
"""Data transform functions for the pipeline."""

from __future__ import annotations
from typing import Any


def clean(record: dict[str, Any]) -> dict[str, Any]:
    """Remove None values and strip whitespace from string fields."""
    result = {}
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        result[key] = value
    return result


def normalize(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize field names to lowercase and values to standard types.

    BUG 1: Lowercases field NAMES, but the validate transform checks
    for exact case-sensitive field names. When normalize runs before
    validate in the pipeline, validation fails on fields that were
    originally uppercase.
    """
    result = {}
    for key, value in record.items():
        normalized_key = key.lower()  # BUG: lowercases field names
        if isinstance(value, str):
            value = value.strip().lower()
        result[normalized_key] = value
    return result


def validate(record: dict[str, Any], required_fields: tuple[str, ...] = ("Name", "Email")) -> dict[str, Any]:
    """Validate that required fields are present.

    Checks for exact field names (case-sensitive).
    """
    missing = [f for f in required_fields if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    # Validate email format
    email = record.get("Email") or record.get("email", "")
    if email and "@" not in str(email):
        raise ValueError(f"Invalid email: {email}")
    return record


def enrich(record: dict[str, Any]) -> dict[str, Any]:
    """Add computed fields to the record."""
    enriched = dict(record)
    name = enriched.get("name") or enriched.get("Name", "")
    if name:
        parts = str(name).split()
        enriched["first_name"] = parts[0] if parts else ""
        enriched["last_name"] = parts[-1] if len(parts) > 1 else ""
    enriched["_processed"] = True
    return enriched
''')

    write_text(workspace / "validators.py", '''\
"""Schema validators that emit validation events."""

from __future__ import annotations
from typing import Any

from events import EventBus


class SchemaValidator:
    def __init__(self, bus: EventBus, schema: dict[str, type]) -> None:
        self.bus = bus
        self.schema = schema
        self.errors: list[str] = []

    def validate(self, record: dict[str, Any]) -> bool:
        """Validate a record against the schema. Emits events."""
        self.errors = []
        for field_name, expected_type in self.schema.items():
            value = record.get(field_name)
            if value is None:
                self.errors.append(f"Missing field: {field_name}")
            elif not isinstance(value, expected_type):
                self.errors.append(
                    f"Type mismatch for {field_name}: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
        if self.errors:
            self.bus.emit("validation_error", {
                "record": record,
                "errors": self.errors,
            })
            return False
        self.bus.emit("validation_success", {"record": record})
        return True
''')

    write_text(workspace / "sink.py", '''\
"""Output sink with batching and flush logic."""

from __future__ import annotations
from typing import Any

from events import EventBus


class BatchSink:
    def __init__(self, bus: EventBus, batch_size: int = 10) -> None:
        self.bus = bus
        self.batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._output: list[dict[str, Any]] = []
        self._flush_count = 0

    def write(self, record: dict[str, Any]) -> None:
        """Add a record to the buffer. Auto-flushes at batch_size."""
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered records to output.

        BUG 2: Clears the buffer BEFORE writing to output.
        Data is lost on every flush.
        """
        if not self._buffer:
            return
        self._buffer.clear()  # BUG: should be AFTER extending output
        self._output.extend(self._buffer)
        self._flush_count += 1
        self.bus.emit("sink_flush", {
            "count": len(self._buffer),
            "total": len(self._output),
        })

    @property
    def output(self) -> list[dict[str, Any]]:
        return list(self._output)

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        return len(self._output)
''')

    write_text(workspace / "metrics.py", '''\
"""Metrics collector listening to pipeline events."""

from __future__ import annotations
from typing import Any

from events import EventBus


class MetricsCollector:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.records_processed = 0
        self.validation_errors = 0
        self.validation_successes = 0
        self.flush_events = 0
        self._error_details: list[dict] = []

        # Register handlers
        bus.on("validation_error", self._on_validation_error)
        bus.on("validation_success", self._on_validation_success)
        bus.on("sink_flush", self._on_flush)
        bus.on("record_processed", self._on_processed)

    def _on_validation_error(self, event: str, data: Any) -> None:
        self.validation_errors += 1
        self._error_details.append(data)

    def _on_validation_success(self, event: str, data: Any) -> None:
        self.validation_successes += 1

    def _on_flush(self, event: str, data: Any) -> None:
        self.flush_events += 1

    def _on_processed(self, event: str, data: Any) -> None:
        self.records_processed += 1

    @property
    def error_rate(self) -> float:
        """Calculate validation error rate.

        Note: This divides by total validations. If no validations
        have occurred, it returns 0.0 (not division by zero).
        """
        total = self.validation_errors + self.validation_successes
        if total == 0:
            return 0.0
        return self.validation_errors / total
''')

    write_text(workspace / "retry.py", '''\
"""Retry decorator with exponential backoff for transforms."""

from __future__ import annotations

import time
from typing import Any, Callable


def with_retry(
    max_retries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator that retries a transform function on failure."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_exc  # type: ignore[misc]
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator
''')

    write_text(workspace / "pipeline.py", '''\
"""Pipeline orchestrator — registers stages and runs them in order."""

from __future__ import annotations
from typing import Any, Callable

from events import EventBus
from sink import BatchSink
from metrics import MetricsCollector
from config import PipelineConfig
import transforms


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.bus = EventBus()
        self.sink = BatchSink(self.bus, batch_size=self.config.batch_size)
        self.metrics = MetricsCollector(self.bus)
        self._stages: list[tuple[str, Callable]] = []
        self._setup_stages()

    def _setup_stages(self) -> None:
        """Register transform stages based on config."""
        stage_map: dict[str, Callable] = {
            "clean": transforms.clean,
            "normalize": transforms.normalize,
            "validate": lambda r: transforms.validate(r),
            "enrich": transforms.enrich,
        }
        for stage_name in self.config.stages:
            fn = stage_map.get(stage_name)
            if fn:
                self._stages.append((stage_name, fn))

    def run(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run all records through the pipeline stages."""
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for record in records:
            try:
                current = dict(record)
                for stage_name, transform in self._stages:
                    current = transform(current)
                self.sink.write(current)
                self.bus.emit("record_processed", {"record": current})
                results.append(current)
            except Exception as exc:
                errors.append({"record": record, "error": str(exc), "stage": stage_name})

        # Flush remaining records in the sink buffer.
        self.sink.flush()
        return results

    @property
    def processed_count(self) -> int:
        return self.metrics.records_processed

    @property
    def error_count(self) -> int:
        return self.metrics.validation_errors

    @property
    def output_records(self) -> list[dict[str, Any]]:
        return self.sink.output
''')

    write_text(workspace / "test_pipeline.py", '''\
"""Integration tests for the data pipeline."""

import pytest
from pipeline import Pipeline
from config import PipelineConfig


class TestPipelineBasic:
    def test_pipeline_processes_valid_records(self):
        """Records with proper fields should flow through all stages."""
        config = PipelineConfig(stages=["clean", "normalize", "validate", "enrich"])
        pipe = Pipeline(config)
        records = [
            {"Name": "Alice Smith", "Email": "alice@example.com", "Age": 30},
            {"Name": "Bob Jones", "Email": "bob@example.com", "Age": 25},
        ]
        results = pipe.run(records)
        assert len(results) == 2
        assert results[0].get("first_name") == "alice"
        assert results[0].get("_processed") is True

    def test_pipeline_output_is_flushed(self):
        """All processed records should appear in the sink output."""
        config = PipelineConfig(
            stages=["clean", "enrich"],
            batch_size=5,
        )
        pipe = Pipeline(config)
        records = [{"Name": f"User {i}", "Email": f"u{i}@test.com"} for i in range(8)]
        pipe.run(records)
        # All 8 records should be in output (5 auto-flushed + 3 final flush)
        assert len(pipe.output_records) == 8

    def test_pipeline_metrics_count(self):
        """Metrics collector should track processed records."""
        config = PipelineConfig(stages=["clean", "enrich"])
        pipe = Pipeline(config)
        records = [{"Name": "Test", "Email": "t@t.com"} for _ in range(5)]
        pipe.run(records)
        assert pipe.processed_count == 5


class TestPipelineValidation:
    def test_normalize_then_validate_preserves_fields(self):
        """When normalize runs before validate, field names should still
        be recognized by the validator (case-insensitive matching)."""
        config = PipelineConfig(stages=["clean", "normalize", "validate", "enrich"])
        pipe = Pipeline(config)
        records = [{"Name": "Alice", "Email": "alice@example.com"}]
        # This should NOT raise — normalize should preserve field
        # compatibility with validate.
        results = pipe.run(records)
        assert len(results) == 1

    def test_validation_errors_are_reported(self):
        """Records missing required fields should be caught, not silently dropped."""
        config = PipelineConfig(stages=["clean", "validate"])
        pipe = Pipeline(config)
        records = [{"Name": "Alice"}]  # missing Email
        results = pipe.run(records)
        # Record should fail validation
        assert len(results) == 0 or pipe.error_count > 0


class TestPipelineSink:
    def test_sink_preserves_all_records(self):
        """The sink must not lose any records during flush operations."""
        config = PipelineConfig(stages=["clean"], batch_size=3)
        pipe = Pipeline(config)
        records = [{"Name": f"User{i}", "data": i} for i in range(7)]
        pipe.run(records)
        # 3 auto-flushed + 3 auto-flushed + 1 final flush = 7
        assert len(pipe.output_records) == 7, (
            f"Expected 7 records in output, got {len(pipe.output_records)}. "
            f"Sink may be losing records during flush."
        )

    def test_small_batch_all_preserved(self):
        """Even with batch_size=2, all records should be preserved."""
        config = PipelineConfig(stages=["clean"], batch_size=2)
        pipe = Pipeline(config)
        records = [{"x": i} for i in range(5)]
        pipe.run(records)
        assert len(pipe.output_records) == 5


class TestPipelineEvents:
    def test_event_handler_errors_are_visible(self):
        """Errors in event handlers should be raised or logged,
        not silently swallowed."""
        from events import EventBus

        errors_seen = []

        def bad_handler(event, data):
            raise RuntimeError("handler crashed")

        def error_catcher(event, data):
            errors_seen.append(data)

        bus = EventBus()
        bus.on("test_event", bad_handler)

        # The bus should either raise the exception or provide
        # a mechanism to detect that a handler failed.
        # Silently swallowing is a bug.
        try:
            bus.emit("test_event", {"key": "value"})
            # If we get here, the exception was swallowed.
            # Check if there is an error tracking mechanism.
            assert False, (
                "EventBus.emit() silently swallowed a handler exception. "
                "Handler errors should be raised or at least logged/tracked."
            )
        except RuntimeError:
            pass  # Good — exception was properly propagated

    def test_metrics_track_validations(self):
        """Metrics collector should track both successes and errors."""
        config = PipelineConfig(stages=["clean", "enrich"])
        pipe = Pipeline(config)
        records = [{"Name": "Test", "Email": "t@t.com"} for _ in range(3)]
        pipe.run(records)
        assert pipe.metrics.records_processed == 3
''')

    write_text(workspace / "test_transforms.py", '''\
"""Unit tests for transform functions."""

import pytest
from transforms import clean, normalize, validate, enrich


class TestClean:
    def test_removes_none_values(self):
        assert clean({"a": 1, "b": None}) == {"a": 1}

    def test_strips_whitespace(self):
        assert clean({"name": "  Alice  "}) == {"name": "Alice"}

    def test_removes_empty_strings(self):
        assert clean({"name": "", "age": 25}) == {"age": 25}

    def test_empty_record(self):
        assert clean({}) == {}


class TestNormalize:
    def test_lowercases_values(self):
        result = normalize({"Name": "ALICE"})
        assert "alice" in result.values()

    def test_preserves_non_string_values(self):
        result = normalize({"count": 42})
        assert 42 in result.values()


class TestValidate:
    def test_valid_record(self):
        record = {"Name": "Alice", "Email": "alice@test.com"}
        result = validate(record)
        assert result == record

    def test_missing_required_field(self):
        with pytest.raises(ValueError, match="Missing required fields"):
            validate({"Name": "Alice"})

    def test_invalid_email(self):
        with pytest.raises(ValueError, match="Invalid email"):
            validate({"Name": "Alice", "Email": "not-email"})

    def test_custom_required_fields(self):
        record = {"id": 1, "type": "A"}
        result = validate(record, required_fields=("id", "type"))
        assert result == record


class TestEnrich:
    def test_adds_first_last_name(self):
        result = enrich({"name": "Alice Smith"})
        assert result["first_name"] == "Alice"
        assert result["last_name"] == "Smith"

    def test_adds_processed_flag(self):
        result = enrich({"name": "Test"})
        assert result["_processed"] is True

    def test_single_name(self):
        result = enrich({"name": "Madonna"})
        assert result["first_name"] == "Madonna"
        assert result.get("last_name") == ""
''')


def _verify_event_pipeline(workspace: Path) -> None:
    """Verify all pipeline bugs are fixed."""
    run_pytest(workspace, label="event pipeline")

    # Verify Bug 1 fix: normalize shouldn't break validate
    sys.path.insert(0, str(workspace))
    try:
        from transforms import normalize, validate
        record = {"Name": "Alice", "Email": "alice@test.com"}
        normalized = normalize(record)
        # After normalize, validate should still work
        # (either normalize preserves case, or validate is case-insensitive)
        validate(normalized)
    except ValueError:
        raise AssertionError(
            "Bug 1 not fixed: normalize breaks validate by lowercasing field names"
        )
    finally:
        sys.path.pop(0)
        for mod_name in list(sys.modules):
            if not mod_name.startswith(("_", "pytest", "pathlib", "os", "sys", "io", "collections")):
                mod = sys.modules[mod_name]
                if hasattr(mod, "__file__") and mod.__file__ and str(workspace) in str(mod.__file__):
                    del sys.modules[mod_name]


# =====================================================================
# Case 1: specsynth Live Repo — Failure Category Misrouting
# =====================================================================

_SPECSYNTH_SOURCE = Path("/Users/lkshay/specsynth")

_CASE1_TASK = """\
The test tests/test_failure_routing.py is failing. The failure category \
routing in the implementation agent is broken — the `_repair_agent_for_failure` \
method in specsynth_agents/implementation_agent/agent.py routes \
`syntax_or_compile` errors to the WRONG repair agent.

Find the bug in the routing logic and fix it. The fix should be in \
`_repair_agent_for_failure()`. Run the failing test to verify your fix.

Do NOT modify any test files. The bug is in the application code only.
"""

_SPECSYNTH_ROUTING_TEST = '''\
"""Test that failure categories route to the correct repair agent."""

import re
from pathlib import Path

# Locate project root (tests/ is one level below root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_mock_agent():
    """Build a minimal mock with the real _repair_agent_for_failure method."""
    class MockAgent:
        def __init__(self):
            self.harness_repair_agent = "harness_repair"
            self.unit_test_repair_agent = "unit_test_repair"
            self.integration_test_repair_agent = "integration_repair"

    agent_path = _PROJECT_ROOT / "specsynth_agents" / "implementation_agent" / "agent.py"
    source = agent_path.read_text(encoding="utf-8")

    # Extract method body.
    match = re.search(
        r"(    def _repair_agent_for_failure\\(self.*?\\n)(?=    def )",
        source,
        re.DOTALL,
    )
    assert match, "Could not find _repair_agent_for_failure in agent.py"
    method_src = match.group(0)
    ns = {}
    exec(f"class _Tmp:\\n{method_src}", ns)
    MockAgent._repair_agent_for_failure = ns["_Tmp"]._repair_agent_for_failure
    return MockAgent()


class TestFailureRouting:
    """Verify that failure categories route to the correct repair agent type."""

    HARNESS_CATEGORIES = [
        "fixture_or_collection",
        "import_or_symbol_resolution",
        "dependency_or_manifest",
        "syntax_or_compile",
        "boot_or_runtime",
        "permission_or_environment",
        "invalid_test_target",
        "zero_tests_collected",
        "unknown",
    ]

    BEHAVIOR_CATEGORIES = [
        "assertion_behavior",
        "interface_mismatch",
        "timeout_or_deadlock",
    ]

    def test_harness_categories_route_to_harness_agent(self):
        agent = _make_mock_agent()
        for cat in self.HARNESS_CATEGORIES:
            result = agent._repair_agent_for_failure(cat)
            assert result == "harness_repair", (
                f"Category {cat!r} should route to harness_repair, got {result!r}"
            )

    def test_behavior_categories_route_to_unit_test_agent(self):
        agent = _make_mock_agent()
        for cat in self.BEHAVIOR_CATEGORIES:
            result = agent._repair_agent_for_failure(cat)
            assert result == "unit_test_repair", (
                f"Category {cat!r} should route to unit_test_repair, got {result!r}"
            )

    def test_integration_flag_routes_to_integration_agent(self):
        agent = _make_mock_agent()
        result = agent._repair_agent_for_failure("assertion_behavior", integration=True)
        assert result == "integration_repair", (
            f"assertion_behavior with integration=True should route to integration_repair, got {result!r}"
        )

    def test_syntax_compile_is_not_behavior(self):
        """syntax_or_compile is an environment issue, not a code behavior issue."""
        agent = _make_mock_agent()
        result = agent._repair_agent_for_failure("syntax_or_compile")
        assert result != "unit_test_repair", (
            "syntax_or_compile should NOT route to unit_test_repair — it is a harness issue"
        )
'''


def _seed_specsynth_misrouting(workspace: Path) -> None:
    """Copy specsynth, install deps, inject routing bug, add failing test."""
    if not _SPECSYNTH_SOURCE.is_dir():
        raise RuntimeError(
            f"specsynth source not found at {_SPECSYNTH_SOURCE}. "
            "Clone it first: git clone <specsynth-repo> /Users/lkshay/specsynth"
        )

    # 1. Copy source tree (exclude heavy/transient dirs).
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".env", "web"}
    skip_suffixes = {".pyc", ".pyo", ".log", ".sqlite", ".db"}
    for src in _SPECSYNTH_SOURCE.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(_SPECSYNTH_SOURCE)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if src.suffix in skip_suffixes:
            continue
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # 2. Create venv and install deps.
    subprocess.run(
        [sys.executable, "-m", "venv", str(workspace / ".venv")],
        check=True, capture_output=True,
    )
    venv_pip = str(workspace / ".venv" / "bin" / "pip")
    subprocess.run(
        [venv_pip, "install", "-q", "-r", str(workspace / "requirements.txt")],
        check=True, capture_output=True, timeout=300,
    )

    # 3. Inject the bug: move "syntax_or_compile" out of the harness set.
    agent_file = workspace / "specsynth_agents" / "implementation_agent" / "agent.py"
    source = agent_file.read_text(encoding="utf-8")
    # Remove syntax_or_compile from the harness routing set.
    source = source.replace(
        '            "syntax_or_compile",\n            "boot_or_runtime",',
        '            "boot_or_runtime",',
    )
    agent_file.write_text(source, encoding="utf-8")

    # 4. Write the failing test.
    write_text(workspace / "tests" / "test_failure_routing.py", _SPECSYNTH_ROUTING_TEST)


def _verify_specsynth_misrouting(workspace: Path) -> None:
    """Verify the routing bug is fixed."""
    venv_python = str(workspace / ".venv" / "bin" / "python3")

    # The routing test must pass.
    result = subprocess.run(
        [venv_python, "-m", "pytest", "tests/test_failure_routing.py", "-v", "--tb=short"],
        cwd=workspace, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"test_failure_routing.py failed:\n{result.stdout}\n{result.stderr}"

    # Verify syntax_or_compile is back in the harness routing set.
    agent_source = (workspace / "specsynth_agents" / "implementation_agent" / "agent.py").read_text()
    # Find the _repair_agent_for_failure method and check syntax_or_compile is in the set.
    import re
    match = re.search(
        r"def _repair_agent_for_failure\(self.*?\n(.*?)(?=\n    def )",
        agent_source,
        re.DOTALL,
    )
    assert match, "_repair_agent_for_failure method not found"
    method_body = match.group(1)
    assert "syntax_or_compile" in method_body, (
        "syntax_or_compile must be in the harness routing set in _repair_agent_for_failure"
    )


# =====================================================================
# Case Registry
# =====================================================================

# =====================================================================
# Case 2: LabelGuard — Write Tests for Untested TypeScript Project
# =====================================================================

_LABELGUARD_SOURCE = Path("/Users/lkshay/ambition/LabelGuard")

_CASE2_TASK = """\
This TypeScript project has no tests. Set up a test framework (Vitest \
recommended) and write comprehensive tests for the two modules below.

**Module 1: src/analysis/scoring.ts**
- `determineVerdict(aiScore, labelChecks, imageResults)` — returns \
verdict + remediation. Test ALL code paths:
  - aiScore is null → compliant
  - aiScore >= 70, no labels → violation
  - aiScore >= 40, image issues → violation
  - aiScore 40-69, no labels → warning
  - aiScore < 40 → compliant
  - Various label combinations (hasVisibleLabel, hasMetaTag, etc.)
- `buildRemediation(checks, images)` — generates remediation steps

**Module 2: src/crawler/discover.ts**
- `normalizeUrl(raw, baseDomain)` — test with:
  - Valid same-domain URLs (with/without www)
  - Cross-domain URLs → null
  - Fragment stripping (#section)
  - Trailing slash removal
  - Malformed URLs → null
- `isAssetUrl(url)` — test with:
  - Asset extensions (.js, .css, .png, .jpg, .pdf, .mp4)
  - Non-asset URLs (/about, /blog/post)
  - Malformed URLs

Target: at least 25 passing tests across both modules. Run the tests \
to verify they all pass.

Do NOT modify the source files — only add test infrastructure and test files.
"""


def _seed_labelguard_tests(workspace: Path) -> None:
    """Copy LabelGuard source files, create minimal project, install deps."""
    if not _LABELGUARD_SOURCE.is_dir():
        raise RuntimeError(
            f"LabelGuard source not found at {_LABELGUARD_SOURCE}. "
            "Clone it or adjust the path."
        )

    # Copy only the files under test + their type dependencies
    src_dir = workspace / "src"
    (src_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (src_dir / "crawler").mkdir(parents=True, exist_ok=True)
    (src_dir / "utils").mkdir(parents=True, exist_ok=True)

    # Source files to test
    for rel in [
        "src/analysis/scoring.ts",
        "src/analysis/label-scanner.ts",
        "src/analysis/image-provenance.ts",
        "src/crawler/discover.ts",
        "src/crawler/rate-limiter.ts",
        "src/crawler/robots.ts",
    ]:
        src = _LABELGUARD_SOURCE / rel
        if src.exists():
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # Minimal logger stub so imports resolve
    write_text(workspace / "src/utils/logger.ts", '''\
export const logger = {
  info: (...args: any[]) => {},
  warn: (...args: any[]) => {},
  debug: (...args: any[]) => {},
  error: (...args: any[]) => {},
};
''')

    # Minimal package.json with vitest + typescript only
    write_text(workspace / "package.json", '''\
{
  "name": "labelguard-tests",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "test": "vitest run"
  },
  "devDependencies": {
    "typescript": "^5.7.0",
    "vitest": "^3.2.0",
    "@types/node": "^22.0.0"
  }
}
''')

    # tsconfig matching the real project
    write_text(workspace / "tsconfig.json", '''\
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "types": ["vitest/globals"]
  },
  "include": ["src/**/*", "tests/**/*"],
  "exclude": ["node_modules", "dist"]
}
''')

    # Type stubs for external dependencies the source files import
    types_dir = workspace / "src" / "types"
    types_dir.mkdir(parents=True, exist_ok=True)

    # better-sqlite3 stub (for calculateDomainScore)
    write_text(types_dir / "better-sqlite3.d.ts", '''\
declare module 'better-sqlite3' {
  interface Statement { get(...params: any[]): any; }
  interface Database { prepare(sql: string): Statement; }
  export default Database;
}
''')

    # cheerio stub (for discover.ts)
    write_text(types_dir / "cheerio.d.ts", '''\
declare module 'cheerio' {
  export function load(html: string, options?: any): any;
}
''')

    # playwright stub (for discover.ts)
    write_text(types_dir / "playwright.d.ts", '''\
declare module 'playwright' {
  export interface Page {
    goto(url: string, options?: any): Promise<any>;
    content(): Promise<string>;
    close(): Promise<void>;
  }
  export interface Browser {
    newPage(): Promise<Page>;
  }
}
''')

    # Install dependencies
    result = subprocess.run(
        ["npm", "install"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"npm install failed: {result.stderr[:500]}")


def _verify_labelguard_tests(workspace: Path) -> None:
    """Verify test infrastructure + at least 25 passing tests."""
    # Check vitest config exists (any form)
    has_config = any(
        (workspace / name).exists()
        for name in ["vitest.config.ts", "vitest.config.mts", "vitest.config.js"]
    )
    # Agent might configure vitest in package.json instead — that's also fine
    if not has_config:
        pkg = workspace / "package.json"
        if pkg.exists():
            import json
            data = json.loads(pkg.read_text())
            # vitest can be configured inline or the default config is sufficient
            has_config = "vitest" in str(data.get("scripts", {}).get("test", ""))

    # Run the test suite
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=json"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Parse test count from JSON output or fallback to counting "pass" lines
    passed_count = 0
    try:
        import json
        # vitest JSON reporter outputs to stdout
        json_output = result.stdout
        # Find JSON object in output (may have preamble)
        start = json_output.find('{"')
        if start >= 0:
            data = json.loads(json_output[start:])
            passed_count = data.get("numPassedTests", 0)
    except (json.JSONDecodeError, ValueError):
        # Fallback: count "✓" or "pass" lines in stderr/stdout
        for line in (result.stdout + result.stderr).splitlines():
            if "✓" in line or "pass" in line.lower():
                passed_count += 1

    # Also try running without JSON reporter for a simpler check
    if passed_count == 0:
        result2 = subprocess.run(
            ["npx", "vitest", "run"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = result2.stdout + result2.stderr
        # Look for "X passed" pattern
        import re
        m = re.search(r"(\d+)\s+pass", combined, re.IGNORECASE)
        if m:
            passed_count = int(m.group(1))

    assert passed_count >= 25, (
        f"Expected at least 25 passing tests, got {passed_count}. "
        f"stdout: {(result.stdout or '')[:500]}\n"
        f"stderr: {(result.stderr or '')[:500]}"
    )

    # Check test files exist for both modules
    test_files = list(workspace.rglob("*.test.ts")) + list(workspace.rglob("*.spec.ts"))
    if not test_files:
        test_files = list((workspace / "tests").rglob("*.ts")) if (workspace / "tests").exists() else []
    assert len(test_files) >= 1, "No test files found (expected *.test.ts or *.spec.ts)"

    # Source files should not be modified
    for src_file in ["src/analysis/scoring.ts", "src/crawler/discover.ts"]:
        original = _LABELGUARD_SOURCE / src_file
        current = workspace / src_file
        if original.exists() and current.exists():
            assert original.read_text() == current.read_text(), (
                f"Source file {src_file} was modified — tests should not change source files"
            )


HARD_AGENTIC_EVAL_CASES: dict[str, EvalCase] = {
    "hard-event-pipeline": EvalCase(
        name="hard-event-pipeline",
        phases=(EvalPhase(task=_CASE3_TASK),),
        setup_workspace=_seed_event_pipeline,
        verify_workspace=_verify_event_pipeline,
        tags=("agentic", "react", "very-hard"),
        budget=10.0,
        difficulty_weight=5.0,
        expected_modified_files=("transforms.py", "sink.py", "events.py"),
    ),
    "hard-specsynth-routing": EvalCase(
        name="hard-specsynth-routing",
        phases=(EvalPhase(task=_CASE1_TASK),),
        setup_workspace=_seed_specsynth_misrouting,
        verify_workspace=_verify_specsynth_misrouting,
        tags=("agentic", "react", "hard", "live-repo"),
        budget=10.0,
        difficulty_weight=4.0,
        expected_modified_files=("specsynth_agents/implementation_agent/agent.py",),
    ),
    "hard-labelguard-tests": EvalCase(
        name="hard-labelguard-tests",
        phases=(EvalPhase(task=_CASE2_TASK),),
        setup_workspace=_seed_labelguard_tests,
        verify_workspace=_verify_labelguard_tests,
        tags=("agentic", "react", "hard", "typescript"),
        budget=10.0,
        difficulty_weight=4.0,
        expected_modified_files=(),  # All new files — no expected modifications
    ),
}
