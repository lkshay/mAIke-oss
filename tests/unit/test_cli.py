import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from maike.atoms.agent import AgentResult
from maike.atoms.context import Checkpoint
import maike.cli as cli
from maike.constants import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_RUN_BUDGET_USD
from maike.eval.contracts import EvalMode
from maike.memory.session import SessionStore
from maike.orchestrator.orchestrator import OrchestratorCancelled, OrchestratorError
from maike.runtime.local import RuntimeConfig

build_parser = cli.build_parser
cost_command = cli.cost_command
dry_run_command = cli.dry_run_command
eval_command = cli.eval_command
history_command = cli.history_command
invoke_resume_command = cli.invoke_resume_command
invoke_run_command = cli.invoke_run_command
resume_command = cli.resume_command
run_command = cli.run_command

if cli.typer is not None:
    from typer.testing import CliRunner

    runner = CliRunner()
else:  # pragma: no cover - depends on optional dependency
    CliRunner = None
    runner = None


def test_cli_run_defaults_to_pinned_model():
    parser = build_parser()
    args = parser.parse_args(["run", "build a todo app"])
    assert args.provider == DEFAULT_PROVIDER
    # --model is left None at the CLI layer so resolve_model_name() can pick
    # the chosen provider's default; verify the resolution lands on DEFAULT_MODEL.
    assert args.model is None
    from maike.gateway.providers import resolve_model_name
    assert resolve_model_name(args.provider, args.model) == DEFAULT_MODEL
    assert args.budget == DEFAULT_RUN_BUDGET_USD
    assert args.language == "python"
    # --show-constraints is opt-in; default off.
    assert args.show_constraints is False


def test_cli_run_parses_show_constraints_flag():
    parser = build_parser()
    args = parser.parse_args(["run", "fix foo.py", "--show-constraints"])
    assert args.show_constraints is True


def test_show_constraints_command_prints_extractor_output(monkeypatch, tmp_path, capsys):
    """Exercises the public helper: it should build a gateway, call
    build_task_constraints, print the markdown to stderr, and return the
    extracted text.  We stub the extractor + gateway so no real LLM call
    happens and no API key is required.
    """
    captured: dict = {}

    async def fake_build_task_constraints(
        *, task, workspace, session_bg_gateway, session_provider,
    ):
        captured["task"] = task
        captured["workspace"] = workspace
        captured["provider"] = session_provider
        return (
            "## Task rules\n\n- Do not run tests.\n- Touch foo.py only.\n",
            ["docs/**"],
        )

    monkeypatch.setattr(
        "maike.agents.constraints.build_task_constraints",
        fake_build_task_constraints,
    )

    class _FakeGateway:
        def __init__(self, *a, **kw):
            pass

        async def aclose(self):
            return None

    monkeypatch.setattr("maike.gateway.llm_gateway.LLMGateway", _FakeGateway)
    # Bypass API-key preflight — the helper will early-exit otherwise.
    monkeypatch.setattr(cli, "_ensure_configured", lambda _provider: None)

    markdown = cli.show_constraints_command(
        task="Fix the bug in foo.py. Do not run tests.",
        workspace=tmp_path,
        provider="gemini",
        model="gemini-2.5-flash",
    )

    assert "Task rules" in markdown
    err = capsys.readouterr().err
    assert "## Task rules" in err
    assert "Do not run tests." in err
    assert "Read-only patterns" in err
    assert "docs/**" in err
    assert captured["provider"] == "gemini"
    assert captured["task"].startswith("Fix the bug in foo.py")


def test_cli_eval_defaults_to_live_eval_options():
    parser = build_parser()
    args = parser.parse_args(["eval"])

    assert args.suite == "all"
    assert args.provider is None
    assert args.model is None
    assert args.budget == DEFAULT_RUN_BUDGET_USD
    assert args.keep_workspaces is False
    assert args.output is None
    assert args.compare_baseline is False
    assert args.update_baseline is False


def test_cli_resume_requires_workspace_and_accepts_session_id(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["resume", "session-123", "--workspace", str(tmp_path)])

    assert args.session_id == "session-123"
    assert args.workspace == tmp_path
    assert args.verbose is False
    assert args.log is None


def test_runtime_config_factory_supports_python_and_node():
    python_config = RuntimeConfig.for_language("python")
    node_config = RuntimeConfig.for_language("node")
    unknown_config = RuntimeConfig.for_language("ruby")

    assert python_config.language == "python"
    assert python_config.test_command == "pytest {path} --tb=short -q"
    assert node_config.language == "node"
    assert node_config.install_command == "npm install {package}"
    assert unknown_config.language == "python"


def test_run_command_resolves_provider_default_model(monkeypatch, tmp_path):
    captured = {}

    class FakeOrchestrator:
        def __init__(self, base_path, **kwargs):
            captured["base_path"] = base_path
            captured["init_kwargs"] = kwargs

        async def run(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    monkeypatch.setattr("maike.cli.Orchestrator", FakeOrchestrator)

    asyncio.run(
        run_command(
            task="build a todo app",
            workspace=tmp_path,
            provider="gemini",
            model=DEFAULT_MODEL,
            budget=1.25,
        )
    )

    assert captured["provider_name"] == "gemini"
    assert captured["model"] == DEFAULT_MODEL
    assert captured["budget"] == 1.25
    assert captured["language_override"] == "python"


def test_run_command_skips_language_override_when_not_explicit(monkeypatch, tmp_path):
    captured = {}

    class FakeOrchestrator:
        def __init__(self, base_path, **kwargs):
            del base_path, kwargs

        async def run(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    monkeypatch.setattr("maike.cli.Orchestrator", FakeOrchestrator)

    asyncio.run(
        run_command(
            task="build a todo app",
            workspace=tmp_path,
            language="python",
            language_explicit=False,
        )
    )

    assert captured["language_override"] is None


def test_invoke_run_command_prints_expected_cli_error(monkeypatch, tmp_path, capsys):
    async def fake_run_command(**kwargs):
        del kwargs
        raise OrchestratorError(
            "requirements failed: Session cost budget exceeded ($0.2500 / $0.00)"
        )

    monkeypatch.setattr("maike.cli.run_interactive", fake_run_command)

    with pytest.raises(SystemExit) as exc_info:
        invoke_run_command(
            task="build a todo app",
            workspace=tmp_path,
        )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Session cost budget exceeded" in captured.err
    assert "requirements failed" in captured.err


def test_invoke_run_command_exits_130_on_graceful_cancellation(monkeypatch, tmp_path):
    async def fake_run_command(**kwargs):
        del kwargs
        raise OrchestratorCancelled("session-123")

    monkeypatch.setattr("maike.cli.run_interactive", fake_run_command)

    with pytest.raises(SystemExit) as exc_info:
        invoke_run_command(
            task="build a todo app",
            workspace=tmp_path,
        )

    assert exc_info.value.code == 130


def test_resume_command_loads_stored_run_config(monkeypatch, tmp_path):
    captured = {}

    async def seed():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session(
            "ship feature",
            tmp_path,
            metadata={
                "run_config": {
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "budget": 2.5,
                    "agent_token_budget": 8000,
                    "language_override": "node",
                    "dynamic_agents_enabled": True,
                    "parallel_coding_enabled": True,
                    "auto_approve": True,
                }
            },
        )
        await store.store_checkpoint(
            session_id,
            Checkpoint(sha="abc123", label="pre-coding", step="coding"),
        )
        return session_id

    async def fake_run_command(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    session_id = asyncio.run(seed())
    monkeypatch.setattr("maike.cli.run_command", fake_run_command)

    asyncio.run(resume_command(session_id, tmp_path, verbose=True))

    assert captured["task"] == "ship feature"
    assert captured["provider"] == "gemini"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["budget"] == 2.5
    assert captured["agent_token_budget"] == 8000
    assert captured["language"] == "node"
    assert captured["language_explicit"] is True
    assert captured["yes"] is True
    assert captured["dynamic_agents_enabled"] is True
    assert captured["parallel_coding_enabled"] is True
    assert captured["session_id"] == session_id


def test_resume_command_rejects_sessions_without_run_config(tmp_path):
    async def seed():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("ship feature", tmp_path)
        await store.store_checkpoint(
            session_id,
            Checkpoint(sha="abc123", label="pre-coding", step="coding"),
        )
        return session_id

    session_id = asyncio.run(seed())

    with pytest.raises(OrchestratorError, match="missing stored run config"):
        asyncio.run(resume_command(session_id, tmp_path))


def test_eval_command_runs_eval_runner(monkeypatch, tmp_path):
    captured = {}

    @dataclass(frozen=True)
    class FakeReport:
        schema_version: int
        mode: str
        suite: str
        suite_key: str
        provider: str
        model: str
        budget: float
        created_at: str
        git_sha: str | None
        report_path: str | None
        baseline_path: str | None
        total: int
        passed: int
        failed: int
        average_score: float
        execution_failed: bool
        regression_failed: bool
        warning_count: int
        results: list

    class FakeRunner:
        def __init__(self, *, provider=None, model=None, budget=DEFAULT_RUN_BUDGET_USD, keep_workspaces=False, adaptive_model=False):
            captured["provider"] = provider
            captured["model"] = model
            captured["budget"] = budget
            captured["keep_workspaces"] = keep_workspaces

        def run(self, request):
            captured["request"] = request
            return FakeReport(
                schema_version=1,
                mode=request.mode.value,
                suite=request.suite,
                suite_key="smoke",
                provider="gemini",
                model="gemini-2.5-flash",
                budget=request.budget,
                created_at="2026-03-18T12:00:00+00:00",
                git_sha=None,
                report_path=str(request.output_path) if request.output_path else None,
                baseline_path=None,
                total=1,
                passed=1,
                failed=0,
                average_score=1.0,
                execution_failed=False,
                regression_failed=False,
                warning_count=0,
                results=[],
            )

    monkeypatch.setattr("maike.cli.EvalRunner", FakeRunner)
    output_path = tmp_path / "report.json"

    result = eval_command(
        suite="smoke",
        workspace=tmp_path,
        provider="gemini",
        model="gemini-2.5-flash",
        budget=2.5,
        keep_workspaces=True,
        output=output_path,
    )

    assert captured["provider"] == "gemini"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["budget"] == 2.5
    assert captured["keep_workspaces"] is True
    assert captured["request"].suite == "smoke"
    assert captured["request"].workspace_root == tmp_path
    assert captured["request"].mode is EvalMode.RUN
    assert captured["request"].output_path == output_path
    assert result["passed"] == 1
    assert result["average_score"] == 1.0
    assert result["report_path"] == str(output_path)


def test_eval_command_rejects_conflicting_baseline_flags(tmp_path):
    with pytest.raises(ValueError):
        eval_command(
            workspace=tmp_path,
            compare_baseline=True,
            update_baseline=True,
        )


def test_eval_command_propagates_compare_mode(monkeypatch, tmp_path):
    captured = {}

    @dataclass(frozen=True)
    class FakeReport:
        schema_version: int
        mode: str
        suite: str
        suite_key: str
        provider: str
        model: str
        budget: float
        created_at: str
        git_sha: str | None
        report_path: str | None
        baseline_path: str | None
        total: int
        passed: int
        failed: int
        average_score: float
        execution_failed: bool
        regression_failed: bool
        warning_count: int
        results: list

    class FakeRunner:
        def __init__(self, *, provider=None, model=None, budget=DEFAULT_RUN_BUDGET_USD, keep_workspaces=False, adaptive_model=False):
            del provider, model, budget, keep_workspaces, adaptive_model

        def run(self, request):
            captured["request"] = request
            return FakeReport(
                schema_version=1,
                mode=request.mode.value,
                suite=request.suite,
                suite_key="smoke",
                provider="gemini",
                model="gemini-2.5-flash",
                budget=request.budget,
                created_at="2026-03-18T12:00:00+00:00",
                git_sha=None,
                report_path=None,
                baseline_path=None,
                total=1,
                passed=0,
                failed=1,
                average_score=0.7,
                execution_failed=False,
                regression_failed=True,
                warning_count=0,
                results=[],
            )

    monkeypatch.setattr("maike.cli.EvalRunner", FakeRunner)

    result = eval_command(workspace=tmp_path, compare_baseline=True)

    assert captured["request"].mode is EvalMode.COMPARE
    assert result["regression_failed"] is True


def test_typer_eval_command_exits_nonzero_on_compare_regression(tmp_path, monkeypatch):
    if runner is None:
        pytest.skip("typer is not installed")

    monkeypatch.setattr(
        cli,
        "eval_command",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "mode": "compare",
            "suite": "all",
            "suite_key": "all",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "budget": 5.0,
            "created_at": "2026-03-18T12:00:00+00:00",
            "git_sha": None,
            "report_path": None,
            "baseline_path": None,
            "total": 1,
            "passed": 0,
            "failed": 1,
            "average_score": 0.5,
            "execution_failed": False,
            "regression_failed": True,
            "warning_count": 0,
            "results": [],
        },
    )

    result = runner.invoke(cli.app, ["eval", "--workspace", str(tmp_path), "--compare-baseline"])

    assert result.exit_code == 1
    assert '"regression_failed": true' in result.stdout.lower()


def test_history_and_cost_commands_query_persisted_session_summaries(tmp_path):
    async def seed():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("build api", tmp_path)
        await store.mark_session_status(session_id, "completed")
        await store.store_agent_run(
            session_id,
            AgentResult(
                agent_id="agent-1",
                role="coder",
                stage_name="coding",
                cost_usd=0.15,
                tokens_used=250,
                metadata={
                    "llm_usage": {
                        "calls": 2,
                        "input_tokens": 150,
                        "output_tokens": 100,
                        "total_tokens": 250,
                        "cost_usd": 0.15,
                    }
                },
            ),
        )
        return session_id

    session_id = asyncio.run(seed())

    history = history_command(workspace=tmp_path, limit=5)
    cost = cost_command(workspace=tmp_path, last=5)
    detail = cost_command(session_id=session_id, workspace=tmp_path)

    assert history[0]["id"] == session_id
    assert cost[0]["id"] == session_id
    assert cost[0]["total_cost_usd"] == pytest.approx(0.15)
    assert cost[0]["input_tokens"] == 150
    assert detail[0]["per_stage"][0]["stage_name"] == "coding"


def test_typer_history_command_renders_table(tmp_path):
    if runner is None:
        pytest.skip("typer is not installed")

    async def seed():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("ship feature", tmp_path)
        await store.mark_session_status(session_id, "completed")
        return session_id

    session_id = asyncio.run(seed())

    result = runner.invoke(cli.app, ["history", "--workspace", str(tmp_path), "--limit", "5"])

    assert result.exit_code == 0
    assert "Session History" in result.stdout
    assert session_id[:8] in result.stdout
    # Rich may ellipsize the status column when more columns are present;
    # the original "completed" text becomes "complet…".  Check the prefix
    # to keep this assertion robust against future column additions.
    assert "complet" in result.stdout.lower()


def test_typer_cost_command_renders_detailed_breakdown(tmp_path):
    if runner is None:
        pytest.skip("typer is not installed")

    async def seed():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("fix bug", tmp_path)
        await store.mark_session_status(session_id, "failed")
        await store.store_agent_run(
            session_id,
            AgentResult(
                agent_id="agent-9",
                role="tester",
                stage_name="verification",
                success=False,
                cost_usd=0.05,
                tokens_used=75,
                metadata={
                    "llm_usage": {
                        "calls": 1,
                        "input_tokens": 40,
                        "output_tokens": 35,
                        "total_tokens": 75,
                        "cost_usd": 0.05,
                    }
                },
            ),
        )
        return session_id

    session_id = asyncio.run(seed())

    result = runner.invoke(cli.app, ["cost", session_id, "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Per-Stage Breakdown" in result.stdout
    assert "verif" in result.stdout.lower()
    assert "Input tokens" in result.stdout


def test_dry_run_command_returns_estimate_without_session(tmp_path):
    """dry_run_command should produce cost estimates without creating sessions."""
    # Create a small workspace with a source file
    (tmp_path / "main.py").write_text("print('hello')\n")

    estimate = dry_run_command(
        task="build a todo app",
        workspace=tmp_path,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL,
        budget=5.0,
    )

    assert estimate["provider"] == DEFAULT_PROVIDER
    assert estimate["model"] is not None
    assert estimate["task_tokens"] > 0
    assert estimate["system_prompt_tokens"] == 2000
    assert estimate["per_iteration_tokens"] > 0
    assert estimate["total_estimated_tokens"] > 0
    assert estimate["total_estimated_cost_usd"] > 0
    assert estimate["budget_usd"] == 5.0
    assert estimate["file_count"] >= 1

    # Verify no session DB was created
    db_path = tmp_path / ".maike" / "session.db"
    assert not db_path.exists()


def test_typer_dry_run_flag_exits_zero(tmp_path):
    """The --dry-run flag via Typer should exit 0 and show estimate info."""
    if runner is None:
        pytest.skip("typer is not installed")

    (tmp_path / "app.py").write_text("x = 1\n")

    result = runner.invoke(
        cli.app,
        ["run", "build a todo app", "--workspace", str(tmp_path), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Dry Run Estimate" in result.stdout
    assert "Provider" in result.stdout
    assert "Model" in result.stdout
    assert "Estimated" in result.stdout


# ── REPL tests ──────────────────────────────────────────────────────────────

from maike.repl import REPLSession, classify_input, parse_input


def test_repl_session_initialization(tmp_path):
    session = REPLSession(
        workspace=tmp_path,
        provider="gemini",
        model="gemini-2.5-flash",
        budget=3.0,
    )
    assert session.workspace == tmp_path.resolve()
    assert session.provider == "gemini"
    assert session.model == "gemini-2.5-flash"
    assert session.budget == 3.0
    assert session.thread_id is None
    assert session.total_cost == 0.0


def test_repl_session_defaults(tmp_path):
    session = REPLSession(workspace=tmp_path)
    assert session.provider == DEFAULT_PROVIDER
    assert session.model == DEFAULT_MODEL
    assert session.budget == DEFAULT_RUN_BUDGET_USD


def test_classify_input_empty():
    assert classify_input("") == ("empty", "")
    assert classify_input("   ") == ("empty", "")


def test_classify_input_slash_commands():
    assert classify_input("/help") == ("slash", "/help")
    assert classify_input("/quit") == ("slash", "/quit")
    assert classify_input("/exit") == ("slash", "/exit")
    assert classify_input("/cost") == ("slash", "/cost")
    assert classify_input("/new") == ("slash", "/new")
    assert classify_input("/budget") == ("slash", "/budget")
    assert classify_input("/history") == ("slash", "/history")
    assert classify_input("  /HELP  ") == ("slash", "/HELP")
    # Multi-word slash commands return full text
    assert classify_input("/plugin list") == ("slash", "/plugin list")
    assert classify_input("/skill load python-style") == ("slash", "/skill load python-style")


def test_classify_input_task():
    assert classify_input("build a todo app") == ("task", "build a todo app")
    assert classify_input("  fix the bug  ") == ("task", "fix the bug")


def test_parse_input_simple():
    assert parse_input("hello world") == "hello world"


def test_parse_input_multiline_continuation():
    raw = "build a \\\ntodo app"
    assert parse_input(raw) == "build a todo app"


def test_parse_input_no_trailing_continuation():
    assert parse_input("just a line") == "just a line"


def test_parse_input_empty():
    assert parse_input("") == ""
    assert parse_input("   ") == ""


def test_repl_quit_command(monkeypatch):
    """REPL should exit when /quit is entered."""
    inputs = iter(["/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"))
    asyncio.run(session.run())
    # If we get here, the loop exited cleanly.


def test_repl_exit_command(monkeypatch):
    """REPL should exit when /exit is entered."""
    inputs = iter(["/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"))
    asyncio.run(session.run())


def test_repl_eof_exits(monkeypatch):
    """Ctrl+D (EOF) should exit the REPL."""
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError))

    session = REPLSession(workspace=Path("/tmp"))
    asyncio.run(session.run())


def test_repl_ctrl_c_continues(monkeypatch):
    """Ctrl+C should not exit; the next input should be processed."""
    call_count = 0

    def fake_input(prompt=""):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt
        return "/quit"

    monkeypatch.setattr("builtins.input", fake_input)

    session = REPLSession(workspace=Path("/tmp"))
    asyncio.run(session.run())
    assert call_count == 2


def test_repl_empty_input_skipped(monkeypatch):
    """Empty lines should be skipped without executing anything."""
    inputs = iter(["", "   ", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"))
    asyncio.run(session.run())


def test_repl_help_command(monkeypatch, capsys):
    """The /help command should print available commands."""
    inputs = iter(["/help", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"))
    asyncio.run(session.run())

    captured = capsys.readouterr()
    # Help output goes to stderr
    assert "/help" in captured.err
    assert "/quit" in captured.err
    assert "/cost" in captured.err


def test_repl_cost_command(monkeypatch, capsys):
    """The /cost command should display current session cost."""
    inputs = iter(["/cost", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"), budget=10.0)
    session.total_cost = 0.25
    asyncio.run(session.run())

    captured = capsys.readouterr()
    assert "$0.25" in captured.err
    assert "$10.00" in captured.err


def test_repl_budget_command(monkeypatch, capsys):
    """The /budget command should display remaining budget."""
    inputs = iter(["/budget", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"), budget=5.0)
    session.total_cost = 1.5
    asyncio.run(session.run())

    captured = capsys.readouterr()
    assert "$3.50" in captured.err or "$3.5" in captured.err


def test_repl_new_command_resets_thread(monkeypatch):
    """The /new command should clear the thread_id."""
    inputs = iter(["/new", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"))
    session.thread_id = "abc-123"
    asyncio.run(session.run())
    assert session.thread_id is None


def test_repl_unknown_slash_command(monkeypatch, capsys):
    """Unknown slash commands should print a warning."""
    inputs = iter(["/foobar", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    session = REPLSession(workspace=Path("/tmp"))
    asyncio.run(session.run())

    captured = capsys.readouterr()
    assert "Unknown command" in captured.err
    assert "/foobar" in captured.err


def test_repl_prompt_is_clean():
    """Prompt string should be a clean 'maike>' regardless of thread state."""
    session = REPLSession(workspace=Path("/tmp"))
    assert "maike>" in session._prompt_str()

    session.thread_id = "abcdef1234567890"
    prompt = session._prompt_str()
    assert "maike>" in prompt
    # Thread ID is NOT in the prompt — it's shown via /history instead
    assert "abcdef12" not in prompt


def test_chat_argparse_subcommand():
    """The argparse parser should accept the 'chat' subcommand."""
    parser = build_parser()
    args = parser.parse_args(["chat", "--workspace", "/tmp", "--budget", "3.0"])
    assert args.command == "chat"
    assert args.budget == 3.0
