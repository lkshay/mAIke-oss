"""Phase 5 wiring tests: CLI flags, react-context injection, /advisor command."""

from __future__ import annotations

from pathlib import Path

import pytest

from maike.cli import build_parser


# ── CLI flag parsing ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "subcommand,extra_args",
    [
        ("run", ["task"]),
        ("chat", []),
        ("resume", ["session-id", "--workspace", "/tmp"]),
        ("eval", []),
    ],
)
def test_advisor_flags_present_on_subcommand(subcommand, extra_args):
    parser = build_parser()
    args = parser.parse_args(
        [subcommand, *extra_args, "--advisor",
         "--advisor-provider", "anthropic",
         "--advisor-model", "claude-opus-4-6",
         "--advisor-budget-pct", "0.4"],
    )
    assert args.advisor is True
    assert args.advisor_provider == "anthropic"
    assert args.advisor_model == "claude-opus-4-6"
    assert args.advisor_budget_pct == 0.4


@pytest.mark.parametrize(
    "subcommand,extra_args",
    [
        ("run", ["task"]),
        ("chat", []),
        ("resume", ["session-id", "--workspace", "/tmp"]),
        ("eval", []),
    ],
)
def test_advisor_flags_default_disabled(subcommand, extra_args):
    parser = build_parser()
    args = parser.parse_args([subcommand, *extra_args])
    assert args.advisor is False
    assert args.advisor_provider is None
    assert args.advisor_model is None
    assert args.advisor_budget_pct == 0.20  # ADVISOR_BUDGET_FRACTION_DEFAULT


# ── react context injection ────────────────────────────────────────


def test_advisor_block_only_injected_when_session_enabled():
    """build_react_context should inject the Advisor availability block only
    when session.advisor_session is enabled."""
    import inspect
    from maike.agents import react

    src = inspect.getsource(react.build_react_context)
    # Both the block and the gate are present.
    assert "Advisor" in src and "frontier model" in src
    assert "advisor_session" in src
    assert "enabled" in src


# ── /advisor slash command ─────────────────────────────────────────


def test_advisor_command_registered():
    from maike.commands.builtins import register_builtins
    from maike.commands.registry import CommandRegistry

    reg = CommandRegistry()
    register_builtins(reg)
    assert "advisor" in reg.command_names


def test_advisor_command_status_when_disabled():
    """Calling /advisor on a session without advisor enabled prints status."""
    from maike.commands.builtins import _handle_advisor

    class _MockSession:
        advisor_enabled = False

    # Just verify it doesn't crash. (Output goes to stderr.)
    _handle_advisor(_MockSession(), [])


def test_advisor_command_status_when_enabled():
    from maike.commands.builtins import _handle_advisor

    class _MockSession:
        advisor_enabled = True
        advisor_provider = "gemini"
        advisor_model = "gemini-2.5-pro"
        advisor_budget_pct = 0.30

    # No crash, no exception.
    _handle_advisor(_MockSession(), [])


def test_repl_handle_advisor_status_runs():
    """The REPL's _handle_advisor method renders status without crashing."""
    from maike.repl import REPLSession

    sess = REPLSession(
        workspace=Path("/tmp"),
        provider="gemini",
        model="gemini-2.5-flash",
        budget=5.0,
        advisor_enabled=True,
        advisor_provider="anthropic",
        advisor_model="claude-opus-4-6",
        advisor_budget_pct=0.25,
    )
    sess._handle_advisor(["status"])
    sess._handle_advisor([])  # default → status
    sess._handle_advisor(["ask", "anything"])  # not yet wired path
    sess._handle_advisor(["unknown"])  # error path


# ── Tracer event kind ──────────────────────────────────────────────


def test_tracer_event_kinds_include_advisor():
    from maike.observability.tracer import TraceEventKind

    assert TraceEventKind.ADVISOR_CALL == "advisor_call"
    assert TraceEventKind.ADVISOR_THROTTLED == "advisor_throttled"


# ── OrchestratorSession field ──────────────────────────────────────


def test_orchestrator_session_has_advisor_session_field():
    from maike.orchestrator.session import OrchestratorSession
    fields = {f.name for f in OrchestratorSession.__dataclass_fields__.values()}
    assert "advisor_session" in fields
