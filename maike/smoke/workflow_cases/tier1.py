"""Tier 1 production workflow cases."""

from __future__ import annotations

from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.helpers import assert_readme_mentions, import_module, run_pytest, write_text


def _verify_ambiguous_pipeline_workspace(workspace: Path) -> None:
    module = import_module(workspace, "error_utils")
    assert hasattr(module, "InvalidConfigError")
    assert hasattr(module, "MissingValueError")
    assert module.format_error(module.InvalidConfigError("missing value")).lower().startswith("invalidconfigerror")
    run_pytest(workspace, label="ambiguous pipeline pytest")
    assert_readme_mentions(workspace, "error_utils.py", "pytest")


def _seed_large_editing_workspace(workspace: Path) -> None:
    write_text(workspace / "app" / "__init__.py", "")
    write_text(workspace / "app" / "users.py", """def create_user(name: str, email: str) -> dict[str, str]:
    return {"name": name, "email": email}


def update_user(user_id: int, email: str) -> dict[str, object]:
    return {"id": user_id, "email": email}
""")
    write_text(workspace / "app" / "orders.py", """def create_order(customer_id: int, sku: str) -> dict[str, object]:
    return {"customer_id": customer_id, "sku": sku}


def cancel_order(order_id: int, reason: str) -> dict[str, object]:
    return {"id": order_id, "reason": reason}
""")
    write_text(workspace / "app" / "payments.py", """def capture_payment(amount: float, currency: str) -> dict[str, object]:
    return {"amount": amount, "currency": currency}


def refund_payment(payment_id: int, amount: float) -> dict[str, object]:
    return {"payment_id": payment_id, "amount": amount}
""")
    write_text(workspace / "app" / "coupons.py", """def validate_coupon(code: str, customer_id: int) -> bool:
    return bool(code) and customer_id > 0


def apply_coupon(code: str, total: float) -> float:
    return total * 0.9 if code else total
""")
    write_text(workspace / "app" / "routes.py", "ROUTES = ['users', 'orders', 'payments']\n")
    write_text(workspace / "app" / "config.py", "DEBUG = False\n")
    write_text(workspace / "app" / "templates.py", "TEMPLATE_NAME = 'base'\n")
    write_text(workspace / "README.md", "# Sample service\n\nRun `pytest -q`.\n")
    write_text(workspace / "docs" / "api.md", "# API docs\n")
    write_text(workspace / "scripts" / "seed_data.py", "print('seed')\n")
    write_text(workspace / "tests" / "test_users.py", """from app.users import create_user, update_user


def test_create_user_happy_path():
    assert create_user("Jane", "jane@example.com")["email"] == "jane@example.com"


def test_update_user_happy_path():
    assert update_user(1, "jane@example.com")["id"] == 1
""")
    write_text(workspace / "tests" / "test_orders.py", """from app.orders import create_order, cancel_order


def test_create_order_happy_path():
    assert create_order(1, "SKU")["sku"] == "SKU"


def test_cancel_order_happy_path():
    assert cancel_order(1, "mistake")["reason"] == "mistake"
""")
    write_text(workspace / "tests" / "test_payments.py", """from app.payments import capture_payment, refund_payment


def test_capture_payment_happy_path():
    assert capture_payment(10.0, "USD")["currency"] == "USD"


def test_refund_payment_happy_path():
    assert refund_payment(9, 2.5)["payment_id"] == 9
""")
    write_text(workspace / "tests" / "test_coupons.py", """from app.coupons import apply_coupon, validate_coupon


def test_validate_coupon_happy_path():
    assert validate_coupon("SAVE", 1) is True


def test_apply_coupon_happy_path():
    assert apply_coupon("SAVE", 100.0) == 90.0
""")


def _verify_large_editing_workspace(workspace: Path) -> None:
    coupons = import_module(workspace, "app.coupons")
    orders = import_module(workspace, "app.orders")
    payments = import_module(workspace, "app.payments")
    users = import_module(workspace, "app.users")

    invalid_calls = [
        (users.create_user, ("", "jane@example.com")),
        (users.create_user, ("Jane", "")),
        (users.update_user, (0, "jane@example.com")),
        (orders.create_order, (0, "SKU")),
        (orders.cancel_order, (0, "")),
        (payments.capture_payment, (0.0, "USD")),
        (payments.refund_payment, (0, 0.0)),
        (coupons.validate_coupon, ("", 0)),
        (coupons.apply_coupon, ("", -1.0)),
    ]
    for fn, args in invalid_calls:
        try:
            fn(*args)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} did not reject invalid input {args!r}")
    run_pytest(workspace, label="large editing pytest")


def _seed_debugging_multi_symptom_workspace(workspace: Path) -> None:
    write_text(workspace / "math_utils.py", "def multiply(a: int, b: int) -> int:\n    return a + b\n")
    write_text(workspace / "string_utils.py", "def normalize_email(value: str) -> str:\n    return value\n")
    write_text(workspace / "date_utils.py", "def next_day(day: int) -> int:\n    return day\n")
    write_text(workspace / "test_math_utils.py", "from math_utils import multiply\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n")
    write_text(workspace / "test_string_utils.py", "from string_utils import normalize_email\n\n\ndef test_normalize_email():\n    assert normalize_email('  Jane@Example.com ') == 'jane@example.com'\n")
    write_text(workspace / "test_date_utils.py", "from date_utils import next_day\n\n\ndef test_next_day():\n    assert next_day(7) == 8\n")


def _verify_debugging_multi_symptom_workspace(workspace: Path) -> None:
    assert import_module(workspace, "math_utils").multiply(3, 4) == 12
    assert import_module(workspace, "string_utils").normalize_email("  Jane@Example.com ") == "jane@example.com"
    assert import_module(workspace, "date_utils").next_day(7) == 8
    run_pytest(workspace, label="debugging multi symptom pytest")


def _verify_greenfield_dependencies_workspace(workspace: Path) -> None:
    module_path = workspace / "weather_cli.py"
    assert module_path.exists(), "weather_cli.py was not created"
    module_text = module_path.read_text(encoding="utf-8")
    assert "requests" in module_text, "weather_cli.py does not import requests"
    run_pytest(workspace, label="greenfield dependencies pytest")
    assert_readme_mentions(workspace, "weather_cli.py", "pytest")


TIER1_EVAL_CASES: dict[str, EvalCase] = {
    "ambiguous-pipeline-selection": EvalCase(
        name="ambiguous-pipeline-selection",
        phases=(
            EvalPhase(
                task=(
                    "Create error_utils.py with InvalidConfigError, MissingValueError, and format_error(exc). "
                    "Add test_error_utils.py and README.md. This is a new utility module, not a debugging task."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_ambiguous_pipeline_workspace,
        tags=("tier1",),
    ),
    "editing-on-large-codebase": EvalCase(
        name="editing-on-large-codebase",
        phases=(
            EvalPhase(
                task=(
                    "Add input validation to all eight public functions in this existing Python service. "
                    "Reject empty strings, non-positive numeric ids, and non-positive amounts with ValueError. "
                    "Keep the existing test suite passing and do not change file ownership outside the app/ package and tests."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_large_editing_workspace,
        verify_workspace=_verify_large_editing_workspace,
        tags=("tier1",),
    ),
    "debugging-multi-symptom": EvalCase(
        name="debugging-multi-symptom",
        phases=(
            EvalPhase(
                task="Fix the failing tests in this project. There are multiple bugs in separate files.",
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=_seed_debugging_multi_symptom_workspace,
        verify_workspace=_verify_debugging_multi_symptom_workspace,
        tags=("tier1",),
    ),
    "greenfield-with-dependencies": EvalCase(
        name="greenfield-with-dependencies",
        phases=(
            EvalPhase(
                task=(
                    "Create weather_cli.py, test_weather_cli.py, and README.md. "
                    "weather_cli.py must fetch weather data from wttr.in using requests and expose a small CLI. "
                    "The tests must mock requests.get and pass under pytest."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda workspace: None,
        verify_workspace=_verify_greenfield_dependencies_workspace,
        tags=("tier1",),
    ),
}
