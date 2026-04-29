"""CLI entrypoints for mAIke."""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import traceback

import getpass
from typing import Any

from dotenv import load_dotenv

from maike.config import (
    PROVIDERS,
    get_configured_provider,
    has_any_configured_provider,
    has_key_for_provider,
    load_user_config,
    save_provider_config,
)
from maike.constants import (
    ADVISOR_BUDGET_FRACTION_DEFAULT,
    ADVISOR_ENABLED_DEFAULT,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_RUN_BUDGET_USD,
    PROVIDER_LLM_CONFIG,
)
from maike.eval.contracts import EvalMode, EvalRequest
from maike.cost.tracker import BudgetExceededError
from maike.eval.runner import EvalRunner
from maike.gateway.providers import resolve_model_name
from maike.observability.live import CompositeTraceSink, FileTraceSink, RichLiveSink, VerboseConsoleSink
from maike.observability.tracer import Tracer
from maike.orchestrator.orchestrator import (
    CancellationController,
    Orchestrator,
    OrchestratorCancelled,
    OrchestratorError,
)
from maike.orchestrator.preflight import PreflightError

try:  # pragma: no cover - optional dependency
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Markdown = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import typer
except ImportError:  # pragma: no cover - exercised by fallback path
    typer = None


def _stdout_console() -> Console | None:
    if Console is None:
        return None
    return Console(stderr=False)


def _stderr_console() -> Console | None:
    if Console is None:
        return None
    return Console(stderr=True)


# ── ANSI colour constants (used by banner, wizard, and prompt) ───────────────

_DIM   = "\033[90m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_CYAN  = "\033[36m"

# Gradient palette: magenta → purple → blue → cyan
_GRAD = [
    "\033[38;5;199m",  # hot pink
    "\033[38;5;177m",  # magenta
    "\033[38;5;141m",  # purple
    "\033[38;5;105m",  # indigo
    "\033[38;5;75m",   # blue
    "\033[38;5;45m",   # cyan
    "\033[38;5;43m",   # teal
]

_W_GREEN  = "\033[32m"
_W_YELLOW = "\033[33m"
_W_RED    = "\033[31m"


def _load_all_dotenv() -> None:
    """Load user-level config then local .env (local takes precedence)."""
    load_user_config()   # ~/.config/maike/.env — does not override
    load_dotenv()        # local .env — overrides


def _w(text: str, colour: str = "") -> None:
    sys.stderr.write(f"{colour}{text}{_RESET}\n")
    sys.stderr.flush()


def _run_setup_wizard() -> str | None:
    """
    Interactive first-run setup.

    Guides the user through selecting an AI provider and entering their API
    key, then saves the result to ~/.config/maike/.env.

    Returns the selected provider name, or None if the user aborted.
    """
    _w("")
    _w(f"  {_BOLD}Welcome to mAIke!{_RESET}  Let's set up your AI provider.")
    _w("")

    # ── Which providers are already configured? ──────────────────────────────
    configured = [p["key"] for p in PROVIDERS if has_key_for_provider(p["key"])]
    if configured:
        _w(f"  {_DIM}Already configured:{_RESET}", "")
        sys.stderr.write("")
        for key in configured:
            info = next(p for p in PROVIDERS if p["key"] == key)
            sys.stderr.write(f"  {_W_GREEN}✓{_RESET}  {info['label']}\n")
        sys.stderr.flush()
        _w("")

    # ── Provider menu ────────────────────────────────────────────────────────
    _w(f"  {_BOLD}Select a provider to configure:{_RESET}")
    _w("")

    col_w = max(len(p["label"]) for p in PROVIDERS) + 2
    for i, p in enumerate(PROVIDERS, 1):
        check = f"{_W_GREEN}✓{_RESET} " if has_key_for_provider(p["key"]) else "  "
        label = f"{_BOLD}{p['label']}{_RESET}".ljust(col_w + len(_BOLD) + len(_RESET))
        sys.stderr.write(
            f"  {_DIM}{i}{_RESET}  {check}{label}"
            f"  {_DIM}{p['model']:22}{_RESET}"
            f"  {_DIM}{p['pricing']}{_RESET}\n"
        )
    sys.stderr.flush()
    _w("")
    _w(f"  {_DIM}q  Skip / quit{_RESET}")
    _w("")

    # ── Pick provider ────────────────────────────────────────────────────────
    while True:
        try:
            sys.stderr.write(f"  {_CYAN}Provider [1-{len(PROVIDERS)}]: {_RESET}")
            sys.stderr.flush()
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            _w("")
            return None

        if choice in {"q", "quit", "exit", ""}:
            return None

        if choice.isdigit() and 1 <= int(choice) <= len(PROVIDERS):
            provider_info = PROVIDERS[int(choice) - 1]
            break

        _w(f"  {_W_RED}Invalid choice — enter a number 1-{len(PROVIDERS)} or q to skip.{_RESET}")

    provider_key = provider_info["key"]
    env_key      = provider_info["env_key"]
    _w("")

    # Gemini accepts three auth paths: GEMINI_API_KEY, GOOGLE_API_KEY, or
    # Vertex AI via gcloud (GOOGLE_GENAI_USE_VERTEXAI=True + GOOGLE_CLOUD_PROJECT).
    # If the user already has one of the alternate paths wired up, let them
    # keep it instead of pasting a fresh key they may not have.
    gemini_alt_auth: str | None = None
    if provider_key == "gemini" and not os.getenv("GEMINI_API_KEY"):
        if os.getenv("GOOGLE_API_KEY"):
            gemini_alt_auth = "GOOGLE_API_KEY"
        elif (
            os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")
            and os.getenv("GOOGLE_CLOUD_PROJECT")
        ):
            gemini_alt_auth = "Vertex AI (GOOGLE_GENAI_USE_VERTEXAI + GOOGLE_CLOUD_PROJECT)"

    if gemini_alt_auth:
        _w(f"  {_W_GREEN}✓{_RESET}  {gemini_alt_auth} is already configured — Gemini will use it.")
        _w(f"  {_DIM}Press Enter to keep it, or paste a GEMINI_API_KEY to override.{_RESET}")
    else:
        _w(f"  {_DIM}Get your key at the provider's dashboard, then paste it below.{_RESET}")
        if provider_key == "gemini":
            _w(f"  {_DIM}Tip: Gemini also accepts GOOGLE_API_KEY or Vertex AI auth (gcloud).{_RESET}")
        _w(f"  {_DIM}Input is hidden.{_RESET}")
    _w("")

    # ── Enter API key ────────────────────────────────────────────────────────
    prompt_label = (
        f"Enter your {env_key} (or Enter to skip)"
        if gemini_alt_auth
        else f"Enter your {env_key}"
    )
    while True:
        try:
            sys.stderr.write(f"  {_CYAN}{prompt_label}: {_RESET}")
            sys.stderr.flush()
            api_key = getpass.getpass("").strip()
        except (EOFError, KeyboardInterrupt):
            _w("")
            return None

        if not api_key:
            if gemini_alt_auth:
                api_key = None  # Reuse the existing alternate auth path
                break
            _w(f"  {_W_RED}Key cannot be empty. Try again or press Ctrl+C to abort.{_RESET}")
            continue
        break

    # ── Save ─────────────────────────────────────────────────────────────────
    try:
        save_provider_config(provider_key, api_key)
    except Exception as exc:  # noqa: BLE001
        _w(f"  {_W_RED}Failed to save config: {exc}{_RESET}")
        return None

    _w("")
    from maike.config import CONFIG_FILE
    if api_key is not None:
        _w(f"  {_W_GREEN}✓{_RESET}  Key saved to {_DIM}{CONFIG_FILE}{_RESET}")
    else:
        _w(f"  {_W_GREEN}✓{_RESET}  Reusing existing {gemini_alt_auth} for Gemini")
    _w(f"  {_W_GREEN}✓{_RESET}  Default provider: {_BOLD}{provider_info['label']}{_RESET}")
    _w("")
    return provider_key


def _ensure_configured(provider: str | None = None) -> None:
    """
    Check that the requested *provider* has an API key.

    If no key is found, launches the setup wizard.  Raises SystemExit if the
    wizard is aborted and still no key is available.
    """
    target = provider or get_configured_provider() or DEFAULT_PROVIDER
    # Ollama runs locally — no API key needed.
    if target == "ollama":
        return
    if has_key_for_provider(target):
        return
    # Any provider will do if no specific one was requested.
    if provider is None and has_any_configured_provider():
        return

    _run_setup_wizard()

    # Re-check after wizard.
    if not (has_key_for_provider(target) or (provider is None and has_any_configured_provider())):
        _emit_cli_error("No API key configured. Run `maike setup` to add one.")
        raise SystemExit(1)


def _build_run_trace_sink(
    verbose: bool, log_path: Path | None, react_mode: bool = False,
) -> CompositeTraceSink | None:
    console = _stdout_console()
    if console is None and log_path is None:
        return None
    sinks = []
    if console is not None:
        sinks.append(RichLiveSink(console=console, react_mode=react_mode))
        if verbose:
            sinks.append(VerboseConsoleSink(console=console))
    if log_path is not None:
        sinks.append(FileTraceSink(log_path=log_path))
    return CompositeTraceSink(sinks=sinks)


def _build_orchestrator(
    *,
    workspace: Path,
    managed_sink,
    dynamic_agents_enabled: bool,
    parallel_coding_enabled: bool,
    stream_sink=None,
) -> Orchestrator:
    orchestrator_kwargs = {"base_path": workspace}
    if managed_sink is not None:
        orchestrator_kwargs["tracer"] = Tracer(sink=managed_sink)
    if dynamic_agents_enabled or parallel_coding_enabled:
        orchestrator_kwargs.update(
            {
                "dynamic_agents_enabled": dynamic_agents_enabled,
                "parallel_coding_enabled": parallel_coding_enabled,
            }
        )
    if stream_sink is not None:
        orchestrator_kwargs["stream_sink"] = stream_sink
    return Orchestrator(**orchestrator_kwargs)


@contextmanager
def _install_signal_handlers(controller: CancellationController):
    previous_handlers: list[tuple[int, object]] = []

    def _handler(signum, frame):  # pragma: no cover - exercised via integration path
        del frame
        if controller.cancel_requested:
            raise KeyboardInterrupt(f"Forced exit on signal {signum}")
        controller.request_cancel()

    supported_signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        supported_signals.append(signal.SIGTERM)
    try:
        for sig in supported_signals:
            previous_handlers.append((sig, signal.getsignal(sig)))
            signal.signal(sig, _handler)
        yield
    finally:
        for sig, previous in reversed(previous_handlers):
            signal.signal(sig, previous)


def _coerce_optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


async def run_single_turn(
    task: str,
    workspace: Path,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    language: str = "python",
    language_explicit: bool = True,
    budget: float | None = DEFAULT_RUN_BUDGET_USD,
    agent_token_budget: int | None = None,
    yes: bool = False,
    verbose: bool = False,
    dynamic_agents_enabled: bool = False,
    parallel_coding_enabled: bool = False,
    log_path: Path | None = None,
    session_id: str | None = None,
    cancellation: CancellationController | None = None,
    adaptive_model: bool = True,
    advisor_enabled: bool = ADVISOR_ENABLED_DEFAULT,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = ADVISOR_BUDGET_FRACTION_DEFAULT,
):
    """Single-turn run (for evals, production scenarios, programmatic use)."""
    _load_all_dotenv()
    trace_sink = _build_run_trace_sink(verbose, log_path, react_mode=True)
    sink_context = trace_sink if trace_sink is not None else nullcontext(None)

    import os as _os
    _stream_sink = None
    if _os.environ.get("MAIKE_STREAM", "").lower() in ("1", "true"):
        from maike.observability.live import StreamingRenderer
        _streaming_renderer = StreamingRenderer()
        _stream_sink = _streaming_renderer.on_chunk

    with sink_context as managed_sink:
        orchestrator = _build_orchestrator(
            workspace=workspace,
            managed_sink=managed_sink,
            dynamic_agents_enabled=dynamic_agents_enabled,
            parallel_coding_enabled=parallel_coding_enabled,
            stream_sink=_stream_sink,
        )
        resolved_model = resolve_model_name(provider, model)
        run_kwargs: dict[str, Any] = dict(
            task=task,
            workspace=workspace,
            provider_name=provider,
            model=resolved_model,
            language_override=language if language_explicit else None,
            budget=budget,
            agent_token_budget=agent_token_budget,
            auto_approve=yes,
            verbose=verbose,
            session_id=session_id,
            cancellation=cancellation,
            adaptive_model=adaptive_model,
        )
        if advisor_enabled:
            run_kwargs["advisor_enabled"] = advisor_enabled
            run_kwargs["advisor_provider"] = advisor_provider
            run_kwargs["advisor_model"] = advisor_model
            run_kwargs["advisor_budget_pct"] = advisor_budget_pct
        try:
            return await orchestrator.run(**run_kwargs)
        except TypeError:
            for _k in (
                "advisor_enabled", "advisor_provider",
                "advisor_model", "advisor_budget_pct",
            ):
                run_kwargs.pop(_k, None)
            return await orchestrator.run(**run_kwargs)


# Backward compat alias — evals and smoke tests import this name.
run_command = run_single_turn



_BANNER_LINES = [
    r"            █████╗ ██╗██╗  ██╗         ",
    r"  ████╗████╗██╔══██╗██║██║ ██╔╝██████╗ ",
    r"  ██╔███╔██║███████║██║█████╔╝ ██╔═══╝ ",
    r"  ██║╚█╚╝██║██╔══██║██║██╔═██╗ █████╗  ",
    r"  ██║ ╚╝ ██║██║  ██║██║██║ ╚██╗██╔══╝  ",
    r"  ╚═╝    ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═════╝",
    r"   your local coding agent        ░▒▓█  ",
]


def _render_gradient_banner() -> str:
    """Render the banner with a per-line colour gradient."""
    lines = []
    for i, line in enumerate(_BANNER_LINES):
        colour = _GRAD[i % len(_GRAD)]
        lines.append(f"{_BOLD}{colour}{line}{_RESET}")
    return "\n".join(lines)


def _print_banner_art() -> None:
    """Print just the ASCII art banner (on launch)."""
    import sys
    sys.stderr.write(f"\n{_render_gradient_banner()}\n\n")
    sys.stderr.flush()


def _print_banner(task: str, model: str, workspace: Path) -> None:
    """Print startup banner with task context."""
    import sys
    sys.stderr.write(f"\n{_render_gradient_banner()}\n\n")
    sys.stderr.write(f"{_DIM}  model: {model}  workspace: {workspace.name}{_RESET}\n")
    sys.stderr.write(f"{_DIM}  task:  {task[:80]}{'...' if len(task) > 80 else ''}{_RESET}\n")
    sys.stderr.write("\n")
    sys.stderr.flush()


async def run_interactive(
    task: str,
    workspace: Path,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    language: str = "python",
    language_explicit: bool = True,
    budget: float | None = DEFAULT_RUN_BUDGET_USD,
    agent_token_budget: int | None = None,
    yes: bool = False,
    verbose: bool = False,
    dynamic_agents_enabled: bool = False,
    parallel_coding_enabled: bool = False,
    log_path: Path | None = None,
    cancellation: CancellationController | None = None,
    thread_id: str | None = None,
    new_thread: bool = False,
    show_banner: bool = True,
    adaptive_model: bool = True,
    suppress_turn_callback: bool = False,
    advisor_enabled: bool = ADVISOR_ENABLED_DEFAULT,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = ADVISOR_BUDGET_FRACTION_DEFAULT,
):
    """Interactive run — prompts for follow-ups after each agent turn."""
    _load_all_dotenv()
    resolved_model = resolve_model_name(provider, model)
    if show_banner:
        _print_banner(task, resolved_model, workspace)
    trace_sink = _build_run_trace_sink(verbose, log_path, react_mode=True)
    sink_context = trace_sink if trace_sink is not None else nullcontext(None)

    # Streaming renderer is available but disabled by default until
    # Gemini streaming hang is resolved.  Set MAIKE_STREAM=1 to enable.
    import os as _os
    _stream_sink = None
    if _os.environ.get("MAIKE_STREAM", "").lower() in ("1", "true"):
        from maike.observability.live import StreamingRenderer
        _streaming_renderer = StreamingRenderer()
        _stream_sink = _streaming_renderer.on_chunk

    with sink_context as managed_sink:
        orchestrator = _build_orchestrator(
            workspace=workspace,
            managed_sink=managed_sink,
            dynamic_agents_enabled=dynamic_agents_enabled,
            parallel_coding_enabled=parallel_coding_enabled,
            stream_sink=_stream_sink,
        )

        async def turn_callback(result) -> str | None:
            _render_run_result(result, workspace)
            try:
                follow_up = input("\n─── maike ▸ ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not follow_up or follow_up.lower() in {"exit", "quit", "q"}:
                return None
            return follow_up

        orchestrator_kwargs: dict[str, Any] = dict(
            initial_task=task,
            workspace=workspace,
            provider_name=provider,
            model=resolved_model,
            language_override=language if language_explicit else None,
            budget=budget,
            agent_token_budget=agent_token_budget,
            auto_approve=yes,
            verbose=verbose,
            cancellation=cancellation,
            thread_id=thread_id,
            new_thread=new_thread,
            turn_callback=turn_callback if (not yes and not suppress_turn_callback) else None,
            adaptive_model=adaptive_model,
        )
        if advisor_enabled:
            orchestrator_kwargs["advisor_enabled"] = advisor_enabled
            orchestrator_kwargs["advisor_provider"] = advisor_provider
            orchestrator_kwargs["advisor_model"] = advisor_model
            orchestrator_kwargs["advisor_budget_pct"] = advisor_budget_pct
        result = await orchestrator.run_interactive(**orchestrator_kwargs)
        return result


async def resume_command(
    session_id: str,
    workspace: Path,
    *,
    verbose: bool = False,
    log_path: Path | None = None,
    cancellation: CancellationController | None = None,
    advisor_enabled: bool = ADVISOR_ENABLED_DEFAULT,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = ADVISOR_BUDGET_FRACTION_DEFAULT,
):
    _load_all_dotenv()
    from maike.memory.session import SessionStore

    store = SessionStore(workspace)
    await store.initialize()
    session = await store.get_session(session_id)
    if session is None:
        raise OrchestratorError(f"Session not found: {session_id}")

    stored_workspace = Path(str(session["workspace"])).resolve()
    if stored_workspace != workspace.resolve():
        raise OrchestratorError(
            f"Session workspace mismatch: expected {stored_workspace}, got {workspace.resolve()}"
        )
    if session["status"] == "completed":
        raise OrchestratorError(f"Session already completed: {session_id}")
    if session["status"] == "failed":
        raise OrchestratorError(f"Failed sessions cannot be resumed: {session_id}")

    metadata = session.get("metadata") or {}
    run_config = metadata.get("run_config")
    if not isinstance(run_config, dict):
        raise OrchestratorError(f"Session is missing stored run config: {session_id}")

    resume_kwargs: dict[str, Any] = dict(
        task=str(session["task"]),
        workspace=workspace,
        provider=str(run_config.get("provider") or DEFAULT_PROVIDER),
        model=str(run_config.get("model") or DEFAULT_MODEL),
        language=str(run_config.get("language_override") or "python"),
        language_explicit=run_config.get("language_override") is not None,
        budget=_coerce_optional_float(run_config.get("budget")),
        agent_token_budget=_coerce_optional_int(run_config.get("agent_token_budget")),
        yes=bool(run_config.get("auto_approve", False)),
        verbose=verbose,
        dynamic_agents_enabled=bool(run_config.get("dynamic_agents_enabled", False)),
        parallel_coding_enabled=bool(run_config.get("parallel_coding_enabled", False)),
        log_path=log_path,
        session_id=session_id,
        cancellation=cancellation,
    )
    if advisor_enabled:
        resume_kwargs["advisor_enabled"] = advisor_enabled
        resume_kwargs["advisor_provider"] = advisor_provider
        resume_kwargs["advisor_model"] = advisor_model
        resume_kwargs["advisor_budget_pct"] = advisor_budget_pct
    try:
        return await run_command(**resume_kwargs)
    except TypeError:
        for _k in (
            "advisor_enabled", "advisor_provider",
            "advisor_model", "advisor_budget_pct",
        ):
            resume_kwargs.pop(_k, None)
        return await run_command(**resume_kwargs)


def _list_threads(workspace: Path) -> None:
    """Display threads for the given workspace."""
    from maike.memory.session import SessionStore

    async def _run():
        store = SessionStore(workspace)
        async with store.use_shared_connection():
            return await store.list_threads(workspace)

    threads = asyncio.run(_run())
    if not threads:
        print("No threads found in this workspace.")
        return
    console = _stdout_console()
    if Table is not None and console is not None:
        table = Table(title=f"Threads in {workspace}")
        table.add_column("ID", style="cyan", max_width=12)
        table.add_column("Name", style="green")
        table.add_column("Status")
        table.add_column("Tokens", justify="right")
        table.add_column("Updated")
        for t in threads:
            table.add_row(
                t["id"][:12] + "...",
                t["name"],
                t["status"],
                f'{t["history_token_estimate"]:,}',
                t["updated_at"][:19] if t.get("updated_at") else "-",
            )
        console.print(table)
    else:
        for t in threads:
            print(f'  {t["id"][:12]}... {t["name"]} [{t["status"]}] tokens={t["history_token_estimate"]:,}')


def _emit_cli_error(message: str) -> None:
    console = _stderr_console()
    if console is not None:
        console.print(f"[bold red]Error:[/bold red] {message}", highlight=False)
        return
    if typer is not None:  # pragma: no cover - exercised indirectly via CLI path
        typer.echo(f"Error: {message}", err=True)
        return
    print(f"Error: {message}", file=sys.stderr)


def invoke_setup_command() -> None:
    """Interactive setup wizard — add or update provider API keys."""
    _load_all_dotenv()
    result = _run_setup_wizard()
    if result is None:
        _echo("Setup skipped.")


def invoke_init_command(
    workspace: Path,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    force: bool = False,
) -> None:
    """Generate (or regenerate) MAIKE.md for the workspace using the fast model."""
    import asyncio
    _load_all_dotenv()
    _ensure_configured(provider)

    maike_path = workspace / "MAIKE.md"
    if maike_path.exists() and not force:
        sys.stderr.write(
            f"{_DIM}  MAIKE.md already exists. Use --force to regenerate.{_RESET}\n"
        )
        sys.stderr.flush()
        return

    from maike.constants import cheap_model_for_provider
    from maike.runtime.probe import EnvironmentProbe
    from maike.agents.init_helper import generate_maike_md

    resolved_model = model or cheap_model_for_provider(provider or "anthropic")
    sys.stderr.write(f"{_DIM}  Generating MAIKE.md with {resolved_model}...{_RESET}\n")
    sys.stderr.flush()

    manifest = EnvironmentProbe().resolve(workspace)

    content = asyncio.run(
        generate_maike_md(workspace, provider=provider, model=resolved_model, manifest=manifest)
    )
    if not content.strip():
        sys.stderr.write(f"{_DIM}  No content generated — MAIKE.md not written.{_RESET}\n")
        sys.stderr.flush()
        return

    # Preserve any existing ## Environment section written by bootstrap
    env_block = ""
    if maike_path.exists():
        import re as _re
        existing = maike_path.read_text(encoding="utf-8")
        env_match = _re.search(r"(## Environment\n(?:.*\n)*)", existing)
        if env_match:
            env_block = "\n" + env_match.group(1).strip()

    maike_path.write_text(content.rstrip() + env_block + "\n", encoding="utf-8")
    sys.stderr.write(f"  Wrote {maike_path}\n")
    sys.stderr.flush()


def invoke_run_command(
    task: str,
    workspace: Path,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    language: str = "python",
    language_explicit: bool = True,
    budget: float | None = DEFAULT_RUN_BUDGET_USD,
    agent_token_budget: int | None = None,
    yes: bool = False,
    verbose: bool = False,
    dynamic_agents_enabled: bool = False,
    parallel_coding_enabled: bool = False,
    log_path: Path | None = None,
    thread_id: str | None = None,
    new_thread: bool = False,
    show_banner: bool = True,
    adaptive_model: bool = True,
    advisor_enabled: bool = ADVISOR_ENABLED_DEFAULT,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = ADVISOR_BUDGET_FRACTION_DEFAULT,
):
    _load_all_dotenv()
    # Use the user's saved default provider if none was explicitly requested.
    effective_provider = get_configured_provider() or DEFAULT_PROVIDER
    if provider == DEFAULT_PROVIDER:
        provider = effective_provider
    _ensure_configured(provider)
    controller = CancellationController()
    try:
        with _install_signal_handlers(controller):
            return asyncio.run(
                run_interactive(
                    task=task,
                    workspace=workspace,
                    provider=provider,
                    model=model,
                    language=language,
                    language_explicit=language_explicit,
                    budget=budget,
                    agent_token_budget=agent_token_budget,
                    yes=yes,
                    verbose=verbose,
                    dynamic_agents_enabled=dynamic_agents_enabled,
                    parallel_coding_enabled=parallel_coding_enabled,
                    log_path=log_path,
                    cancellation=controller,
                    thread_id=thread_id,
                    new_thread=new_thread,
                    show_banner=show_banner,
                    adaptive_model=adaptive_model,
                    advisor_enabled=advisor_enabled,
                    advisor_provider=advisor_provider,
                    advisor_model=advisor_model,
                    advisor_budget_pct=advisor_budget_pct,
                )
            )
    except (OrchestratorCancelled, KeyboardInterrupt) as exc:
        raise SystemExit(130) from exc
    except (BudgetExceededError, OrchestratorError, PreflightError) as exc:
        if verbose:
            traceback.print_exception(exc, file=sys.stderr)
        else:
            _emit_cli_error(str(exc))
        raise SystemExit(1) from exc


_SKIP_DIRS = {"node_modules", "__pycache__", "venv"}
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
    ".java", ".rb", ".cs", ".c", ".cpp", ".h",
})


def _count_source_files(workspace: Path) -> tuple[int, int]:
    """Return (file_count, total_lines) for source files under *workspace*."""
    file_count = 0
    total_lines = 0
    for root, _dirs, files in os.walk(workspace):
        root_path = Path(root)
        if any(part.startswith(".") or part in _SKIP_DIRS
               for part in root_path.relative_to(workspace).parts):
            continue
        for fname in files:
            fpath = root_path / fname
            if fpath.suffix in _SOURCE_EXTENSIONS:
                file_count += 1
                try:
                    total_lines += sum(1 for _ in fpath.open(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    return file_count, total_lines


def dry_run_command(
    task: str,
    workspace: Path,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    budget: float = DEFAULT_RUN_BUDGET_USD,
) -> dict:
    """Estimate tokens and cost without executing any LLM calls."""
    from maike.runtime.probe import EnvironmentProbe

    resolved_model = resolve_model_name(provider, model)
    manifest = EnvironmentProbe().resolve(workspace)
    file_count, total_lines = _count_source_files(workspace)

    task_tokens = int(len(task.split()) * 1.3)
    system_prompt_tokens = 2000
    workspace_context_tokens = min(file_count * 200, 50000)
    per_iteration_tokens = task_tokens + system_prompt_tokens + workspace_context_tokens
    estimated_iterations = 5
    total_estimated_tokens = per_iteration_tokens * estimated_iterations

    config = PROVIDER_LLM_CONFIG.get(provider.lower())
    cost_per_m_input = config.pricing.input_per_million_usd if config is not None else 1.0
    per_iteration_cost = (per_iteration_tokens / 1_000_000) * cost_per_m_input
    total_estimated_cost = per_iteration_cost * estimated_iterations

    estimate = {
        "provider": provider,
        "model": resolved_model,
        "language": manifest.language,
        "file_count": file_count,
        "total_lines": total_lines,
        "task_tokens": task_tokens,
        "system_prompt_tokens": system_prompt_tokens,
        "workspace_context_tokens": workspace_context_tokens,
        "per_iteration_tokens": per_iteration_tokens,
        "estimated_iterations": estimated_iterations,
        "total_estimated_tokens": total_estimated_tokens,
        "per_iteration_cost_usd": per_iteration_cost,
        "total_estimated_cost_usd": total_estimated_cost,
        "budget_usd": budget,
    }

    console = _stdout_console()
    if Table is not None and console is not None:
        table = Table(title="Dry Run Estimate")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Provider", provider)
        table.add_row("Model", resolved_model)
        table.add_row("Language", manifest.language)
        table.add_row("Source files", _format_int(file_count))
        table.add_row("Total lines", _format_int(total_lines))
        table.add_row("Estimated input tokens/iter", _format_int(per_iteration_tokens))
        table.add_row("Estimated cost/iter", _format_usd(per_iteration_cost))
        table.add_row("Estimated iterations", str(estimated_iterations))
        table.add_row("Total estimated tokens", _format_int(total_estimated_tokens))
        table.add_row("Total estimated cost", _format_usd(total_estimated_cost))
        table.add_row("Budget", f"${budget:.2f}")
        console.print(table)
    else:
        print(f"Provider: {provider}")
        print(f"Model: {resolved_model}")
        print(f"Estimated input tokens/iter: {_format_int(per_iteration_tokens)}")
        print(f"Estimated cost/iter: {_format_usd(per_iteration_cost)}")
        print(f"Total estimated cost: {_format_usd(total_estimated_cost)}")
        print(f"Budget: ${budget:.2f}")

    return estimate


def show_constraints_command(
    task: str,
    workspace: Path,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> str:
    """Extract task constraints and print the resulting markdown to stderr.

    Does not run a session.  Builds a silent ``LLMGateway`` exactly like
    the orchestrator's ``_bg_gateway``, calls ``build_task_constraints``,
    and writes the rendered markdown block to stderr so users can inspect
    what the ``<maike-constraints>`` context block will contain without
    burning session budget.

    Returns the extracted markdown text (for programmatic use in tests).
    """
    _load_all_dotenv()
    effective_provider = get_configured_provider() or DEFAULT_PROVIDER
    if provider == DEFAULT_PROVIDER:
        provider = effective_provider
    _ensure_configured(provider)

    from maike.agents.constraints import build_task_constraints
    from maike.cost.tracker import CostTracker
    from maike.gateway.llm_gateway import LLMGateway

    # Touch the model name through the same resolver the orchestrator uses,
    # so an unknown provider/model surfaces a clean error before we build the
    # gateway.  The extractor picks its own model (Ollama-first); this call
    # is validation-only.
    resolve_model_name(provider, model)

    async def _run() -> tuple[str, list[str]]:
        cost_tracker = CostTracker(session_budget_usd=0.0)
        tracer = Tracer()
        bg_gateway = LLMGateway(
            cost_tracker, tracer,
            provider_name=provider,
            silent=True,
        )
        try:
            return await build_task_constraints(
                task=task,
                workspace=workspace,
                session_bg_gateway=bg_gateway,
                session_provider=provider,
            )
        finally:
            try:
                await bg_gateway.aclose()
            except Exception:  # noqa: BLE001
                pass

    try:
        markdown, read_only_patterns = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 — never crash on extraction
        _emit_cli_error(f"constraint extraction failed: {exc}")
        raise SystemExit(1) from exc

    err = sys.stderr
    header = f"# Task constraints (provider={provider})\n"
    print(header, file=err)
    if markdown.strip():
        print(markdown, file=err)
    else:
        print("(no constraints extracted — model returned empty)", file=err)
    if read_only_patterns:
        print("\n# Read-only patterns (from MAIKE.md):", file=err)
        for pattern in read_only_patterns:
            print(f"- {pattern}", file=err)
    err.flush()
    return markdown


def invoke_resume_command(
    session_id: str,
    workspace: Path,
    *,
    verbose: bool = False,
    log_path: Path | None = None,
    advisor_enabled: bool = ADVISOR_ENABLED_DEFAULT,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = ADVISOR_BUDGET_FRACTION_DEFAULT,
):
    controller = CancellationController()
    try:
        with _install_signal_handlers(controller):
            return asyncio.run(
                resume_command(
                    session_id=session_id,
                    workspace=workspace,
                    verbose=verbose,
                    log_path=log_path,
                    cancellation=controller,
                    advisor_enabled=advisor_enabled,
                    advisor_provider=advisor_provider,
                    advisor_model=advisor_model,
                    advisor_budget_pct=advisor_budget_pct,
                )
            )
    except (OrchestratorCancelled, KeyboardInterrupt) as exc:
        raise SystemExit(130) from exc
    except (BudgetExceededError, OrchestratorError, PreflightError) as exc:
        if verbose:
            traceback.print_exception(exc, file=sys.stderr)
        else:
            _emit_cli_error(str(exc))
        raise SystemExit(1) from exc


def invoke_chat_command(
    workspace: Path = Path.cwd(),
    provider: str | None = None,
    model: str | None = None,
    budget: float = DEFAULT_RUN_BUDGET_USD,
    show_banner: bool = True,
    verbose: bool = False,
    yes: bool = False,
    no_tui: bool = False,
    advisor_enabled: bool = ADVISOR_ENABLED_DEFAULT,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = ADVISOR_BUDGET_FRACTION_DEFAULT,
) -> None:
    """Launch the interactive REPL (or TUI if available)."""
    _load_all_dotenv()
    effective_provider = provider or get_configured_provider() or DEFAULT_PROVIDER
    _ensure_configured(effective_provider)
    # When the caller passes a model, use it; otherwise fall back to the
    # chosen provider's default (resolve_model_name handles the None case).
    resolved_model = resolve_model_name(effective_provider, model)
    resolved_workspace = Path(workspace).resolve()

    # Use TUI by default when connected to an interactive terminal.
    use_tui = (
        not no_tui
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )

    if use_tui:
        try:
            from maike.tui import launch_tui

            launch_tui(
                workspace=resolved_workspace,
                provider=effective_provider,
                model=resolved_model,
                budget=budget,
                verbose=verbose,
                yes=yes,
                advisor_enabled=advisor_enabled,
                advisor_provider=advisor_provider,
                advisor_model=advisor_model,
                advisor_budget_pct=advisor_budget_pct,
            )
            return
        except ImportError:
            pass  # textual not installed — fall back to legacy REPL
        except TypeError:
            # Older launch_tui signature — fall back and proceed without advisor.
            from maike.tui import launch_tui

            launch_tui(
                workspace=resolved_workspace,
                provider=effective_provider,
                model=resolved_model,
                budget=budget,
                verbose=verbose,
                yes=yes,
            )
            return

    # Legacy REPL fallback.
    if show_banner:
        _print_banner_art()

    from maike.repl import REPLSession

    try:
        session = REPLSession(
            workspace=resolved_workspace,
            provider=effective_provider,
            model=resolved_model,
            budget=budget,
            verbose=verbose,
            yes=yes,
            advisor_enabled=advisor_enabled,
            advisor_provider=advisor_provider,
            advisor_model=advisor_model,
            advisor_budget_pct=advisor_budget_pct,
        )
    except TypeError:
        # Older REPLSession signature — fall back without advisor.
        session = REPLSession(
            workspace=resolved_workspace,
            provider=effective_provider,
            model=resolved_model,
            budget=budget,
            verbose=verbose,
            yes=yes,
        )
    asyncio.run(session.run())


def eval_command(
    suite: str = "all",
    workspace: Path = Path.cwd(),
    provider: str | None = None,
    model: str | None = None,
    budget: float = DEFAULT_RUN_BUDGET_USD,
    keep_workspaces: bool = False,
    output: Path | None = None,
    compare_baseline: bool = False,
    update_baseline: bool = False,
    adaptive_model: bool = False,
    advisor_enabled: bool = ADVISOR_ENABLED_DEFAULT,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = ADVISOR_BUDGET_FRACTION_DEFAULT,
) -> dict:
    _load_all_dotenv()
    if compare_baseline and update_baseline:
        raise ValueError("--compare-baseline and --update-baseline cannot be used together")

    mode = EvalMode.RUN
    if compare_baseline:
        mode = EvalMode.COMPARE
    elif update_baseline:
        mode = EvalMode.BASELINE

    runner_kwargs: dict[str, Any] = dict(
        provider=provider,
        model=model,
        budget=budget,
        keep_workspaces=keep_workspaces,
        adaptive_model=adaptive_model,
    )
    # EvalRunner may not yet accept advisor kwargs — pass only if supported.
    try:
        report = EvalRunner(
            **runner_kwargs,
            advisor_enabled=advisor_enabled,
            advisor_provider=advisor_provider,
            advisor_model=advisor_model,
            advisor_budget_pct=advisor_budget_pct,
        ).run(
            EvalRequest(
                suite=suite,
                workspace_root=workspace,
                provider=provider,
                model=model,
                budget=budget,
                keep_workspaces=keep_workspaces,
                mode=mode,
                output_path=output,
            )
        )
    except TypeError:
        report = EvalRunner(**runner_kwargs).run(
            EvalRequest(
                suite=suite,
                workspace_root=workspace,
                provider=provider,
                model=model,
                budget=budget,
                keep_workspaces=keep_workspaces,
                mode=mode,
                output_path=output,
            )
        )
    return asdict(report)


def _run_async(coro):
    """Run an async coroutine, handling the case where an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def cost_command(session_id: str | None = None, last: int = 10, workspace: Path = Path.cwd()) -> list[dict]:
    _load_all_dotenv()
    from maike.memory.session import SessionStore

    store = SessionStore(workspace)

    async def _load() -> list[dict]:
        await store.initialize()
        if session_id:
            summary = await store.get_session_cost(session_id)
            return [summary] if summary is not None else []

        summaries: list[dict] = []
        for session in await store.list_sessions(last):
            summary = await store.get_session_cost(session["id"])
            if summary is not None:
                summaries.append(summary)
        return summaries

    return _run_async(_load())


def history_command(workspace: Path = Path.cwd(), limit: int = 20) -> list[dict]:
    _load_all_dotenv()
    from maike.memory.session import SessionStore

    store = SessionStore(workspace)

    async def _load():
        await store.initialize()
        return await store.list_sessions(limit)

    return _run_async(_load())


def _format_usd(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"${float(value):.4f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _echo(message: str) -> None:
    console = _stdout_console()
    if console is not None:
        console.print(message, highlight=False)
        return
    print(message)


def _json_default(obj: object) -> str:
    """Fallback serializer for non-standard types."""
    from datetime import datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _print_json(data: dict) -> None:
    console = _stdout_console()
    if console is not None:
        console.print_json(json=json.dumps(data, default=_json_default))
        return
    print(json.dumps(data, indent=2, sort_keys=True, default=_json_default))


def _render_history(sessions: list[dict]) -> None:
    if not sessions:
        _echo("No sessions found.")
        return
    if Table is None:
        _print_json({"sessions": sessions})
        return

    table = Table(title="Session History")
    table.add_column("Session ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Verdict", style="yellow")
    table.add_column("Created")
    table.add_column("Updated")
    table.add_column("Task", overflow="fold")
    table.add_column("Workspace", overflow="fold")
    for session in sessions:
        meta = session.get("metadata") or {}
        verdict_info = meta.get("verdict") if isinstance(meta, dict) else None
        verdict_label = "—"
        if isinstance(verdict_info, dict):
            v_label = verdict_info.get("label")
            if isinstance(v_label, str) and v_label:
                # Trim unproductive_* for display; full label in the summary.
                verdict_label = v_label.replace("unproductive_", "")
        table.add_row(
            session["id"],
            session["status"],
            verdict_label,
            session["created_at"],
            session["updated_at"],
            session["task"],
            session["workspace"],
        )
    console = _stdout_console()
    assert console is not None
    console.print(table)


def _render_cost_list(summaries: list[dict]) -> None:
    if not summaries:
        _echo("No sessions found.")
        return
    if Table is None:
        _print_json({"sessions": summaries})
        return

    table = Table(title="Session Cost")
    table.add_column("Session ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Cost")
    table.add_column("Tokens")
    table.add_column("LLM Calls")
    table.add_column("Task", overflow="fold")
    for summary in summaries:
        table.add_row(
            summary["id"],
            summary["status"],
            _format_usd(summary["total_cost_usd"]),
            _format_int(summary["total_tokens"]),
            _format_int(summary["llm_calls"]),
            summary["task"],
        )
    console = _stdout_console()
    assert console is not None
    console.print(table)


def _render_cost_detail(summary: dict) -> None:
    if Table is None:
        _print_json(summary)
        return

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="cyan")
    summary_table.add_column()
    summary_table.add_row("Session", summary["id"])
    summary_table.add_row("Task", summary["task"])
    summary_table.add_row("Status", summary["status"])
    summary_table.add_row("Workspace", summary["workspace"])
    summary_table.add_row("Created", summary["created_at"])
    summary_table.add_row("Updated", summary["updated_at"])
    summary_table.add_row("Total cost", _format_usd(summary["total_cost_usd"]))
    summary_table.add_row("Total tokens", _format_int(summary["total_tokens"]))
    summary_table.add_row(
        "Input tokens",
        _format_int(summary["input_tokens"]) if summary["has_token_breakdown"] else "-",
    )
    summary_table.add_row(
        "Output tokens",
        _format_int(summary["output_tokens"]) if summary["has_token_breakdown"] else "-",
    )
    summary_table.add_row("LLM calls", _format_int(summary["llm_calls"]))
    summary_table.add_row("Agent runs", _format_int(summary["agent_runs"]))

    per_stage = Table(title="Per-Stage Breakdown")
    per_stage.add_column("Stage", style="green")
    per_stage.add_column("Cost")
    per_stage.add_column("Tokens")
    per_stage.add_column("Input")
    per_stage.add_column("Output")
    per_stage.add_column("LLM Calls")
    per_stage.add_column("Agent Runs")
    per_stage.add_column("Failures")
    for stage in summary["per_stage"]:
        per_stage.add_row(
            stage["stage_name"],
            _format_usd(stage["cost_usd"]),
            _format_int(stage["tokens_used"]),
            _format_int(stage["input_tokens"]) if stage["has_token_breakdown"] else "-",
            _format_int(stage["output_tokens"]) if stage["has_token_breakdown"] else "-",
            _format_int(stage["llm_calls"]),
            _format_int(stage["agent_runs"]),
            _format_int(stage["failed_runs"]),
        )

    console = _stdout_console()
    assert console is not None
    console.print(summary_table)
    console.print(per_stage)


def _render_run_result(result, workspace: Path, streamed: bool = False) -> None:
    # Print the agent's text output (answers, explanations, etc.)
    # When streamed=True, the text was already displayed token-by-token
    # during the LLM turn, so skip the re-render.
    if not streamed:
        agent_output = _extract_agent_output(result)
        if agent_output:
            print()
            print(agent_output)
            print()

    # Extract cost/tokens directly from the result.
    total_cost = 0.0
    total_tokens = 0
    for stage_results in result.stage_results.values():
        for agent_result in stage_results:
            total_cost += getattr(agent_result, "cost_usd", 0) or 0
            total_tokens += getattr(agent_result, "tokens_used", 0) or 0

    stages = ", ".join(result.stage_results) or "-"
    print(f"Session: {result.session_id}  Pipeline: {result.pipeline}  "
          f"Stages: {stages}  Cost: {_format_usd(total_cost)}  Tokens: {_format_int(total_tokens)}")


def _extract_agent_output(result) -> str:
    """Pull the final agent text output from the orchestrator result."""
    for stage_agents in result.stage_results.values():
        for agent_result in reversed(stage_agents):
            output = (agent_result.output or "").strip()
            if output:
                return output
    return ""


def _render_eval_summary(summary: dict) -> None:
    _print_json(summary)


def _invoke_plugin_command(args) -> None:
    """Handle `maike plugin` subcommands."""
    cmd = getattr(args, "plugin_command", None)
    if cmd == "list":
        _invoke_plugin_list(getattr(args, "workspace", Path.cwd()))
    elif cmd == "validate":
        _invoke_plugin_validate(getattr(args, "path", Path.cwd()))
    elif cmd == "install":
        _invoke_plugin_install(args.source, getattr(args, "install_dir", None), args.scope, getattr(args, "force", False))
    elif cmd == "enable":
        _invoke_plugin_enable(args.name)
    elif cmd == "disable":
        _invoke_plugin_disable(args.name)
    elif cmd == "update":
        _invoke_plugin_update(args.name)
    elif cmd == "uninstall":
        _invoke_plugin_uninstall(args.name, getattr(args, "remove_data", False))
    else:
        print("Usage: maike plugin {list,validate,install,enable,disable,update,uninstall}")


def _invoke_plugin_list(workspace: Path) -> None:
    """Display discovered plugins, their components, and standalone MCP/LSP servers."""
    from maike.constants import (
        LSP_PROJECT_CONFIG_NAME,
        PLUGIN_PROJECT_SUBDIR,
        PLUGIN_USER_DIR,
    )
    from maike.plugins.agent_loader import load_all_plugin_agents
    from maike.plugins.discovery import PluginDiscovery
    from maike.plugins.hooks import load_hook_configs
    from maike.plugins.loader import PluginLoader
    from maike.plugins.lsp_config import load_lsp_configs
    from maike.plugins.mcp_config import load_mcp_configs
    from maike.plugins.settings import load_settings

    # Discover plugins (unfiltered — list shows all, including disabled)
    search_dirs: list[Path] = []
    if PLUGIN_USER_DIR.is_dir():
        search_dirs.append(PLUGIN_USER_DIR)
    project_dir = workspace / PLUGIN_PROJECT_SUBDIR
    if project_dir.is_dir():
        search_dirs.append(project_dir)

    settings = load_settings()
    manifests = PluginDiscovery.discover(search_dirs) if search_dirs else []
    skills = PluginLoader.load_all_plugin_skills(manifests) if manifests else []
    agents = load_all_plugin_agents(manifests)
    hooks = load_hook_configs(manifests)
    mcp_configs = load_mcp_configs(workspace, manifests)
    lsp_configs = load_lsp_configs(
        manifests,
        project_lsp_path=(workspace / LSP_PROJECT_CONFIG_NAME),
    )

    if not manifests and not mcp_configs and not lsp_configs:
        print("No plugins, MCP servers, or LSP servers found.")
        print(f"\nPlugin directories scanned:")
        print(f"  User:    {PLUGIN_USER_DIR}")
        print(f"  Project: {project_dir}")
        return

    # Print plugins
    if manifests:
        print("Plugins:")
        for m in manifests:
            plugin_skills = [s for s in skills if s.namespace == m.name]
            plugin_agents = [a for a in agents if a.namespace == m.name]
            plugin_mcp = [c for c in mcp_configs if c.source == "plugin" and c.name.startswith(f"{m.name}:")]
            plugin_hooks_count = sum(
                1 for defs in hooks.hooks.values()
                for d in defs
                if d.source_plugin == m.name
            )

            state = " [disabled]" if m.name in settings.disabled else ""
            print(f"  {m.name} (v{m.version}){state} — {m.path}")
            if m.description:
                print(f"    {m.description}")
            if plugin_skills:
                print(f"    Skills:  {', '.join(s.name for s in plugin_skills)}")
            if plugin_agents:
                print(f"    Agents:  {', '.join(a.qualified_name for a in plugin_agents)}")
            if plugin_mcp:
                for c in plugin_mcp:
                    print(f"    MCP:     {c.name} (command: {c.command})")
            if plugin_hooks_count:
                print(f"    Hooks:   {plugin_hooks_count} hook(s)")
        print()

    # Print standalone MCP servers (not from plugins)
    standalone_mcp = [c for c in mcp_configs if c.source != "plugin"]
    if standalone_mcp:
        print("MCP Servers (standalone):")
        for c in standalone_mcp:
            print(f"  {c.name} — {c.source} (command: {c.command} {' '.join(c.args)})")
        print()

    # Print LSP servers
    if lsp_configs:
        print("LSP Servers:")
        for c in lsp_configs:
            exts = ", ".join(c.extension_to_language.keys())
            print(f"  {c.name} — {c.source} (command: {c.command}, extensions: {exts})")
        print()


def _invoke_plugin_validate(plugin_path: Path) -> None:
    """Validate a plugin directory."""
    from maike.plugins.manifest import parse_plugin_manifest

    manifest = parse_plugin_manifest(plugin_path)
    if manifest is None:
        print(f"Invalid plugin: no valid .maike-plugin/plugin.json found in {plugin_path}")
        raise SystemExit(1)

    print(f"Plugin: {manifest.name} v{manifest.version}")
    issues: list[str] = []

    # Check component directories
    if manifest.skills_dir.is_dir():
        skill_count = len(list(manifest.skills_dir.glob("*/SKILL.md")))
        print(f"  Skills:  {skill_count} found in {manifest.skills_dir}")
    else:
        print(f"  Skills:  none (no {manifest.skills_dir})")

    if manifest.agents_dir.is_dir():
        agent_count = len(list(manifest.agents_dir.glob("*.md")))
        print(f"  Agents:  {agent_count} found in {manifest.agents_dir}")
    else:
        print(f"  Agents:  none (no {manifest.agents_dir})")

    if manifest.hooks_file.is_file():
        print(f"  Hooks:   {manifest.hooks_file}")
    else:
        print(f"  Hooks:   none")

    if manifest.mcp_config_file.is_file():
        print(f"  MCP:     {manifest.mcp_config_file}")
    else:
        print(f"  MCP:     none")

    if manifest.lsp_config_file.is_file():
        print(f"  LSP:     {manifest.lsp_config_file}")
    else:
        print(f"  LSP:     none")

    if issues:
        print(f"\nIssues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    else:
        print("\nPlugin is valid.")


def _invoke_plugin_install(source: str, dir_preset: str | None, scope: str, force: bool = False) -> None:
    """Install a plugin from a local directory or git URL."""
    from maike.plugins.installer import (
        PluginInstallError,
        describe_plugin,
        install_plugin,
        resolve_install_dir,
    )

    target = resolve_install_dir(dir_preset, scope)
    try:
        manifest = install_plugin(source, target, scope, force=force)
        print(f"Installed plugin: {manifest.name} v{manifest.version}")
        print(f"  Location: {manifest.path}")
        summary = describe_plugin(manifest)
        if summary:
            print(summary)
    except PluginInstallError as exc:
        print(f"Install failed: {exc}")
        raise SystemExit(1)


def _invoke_plugin_enable(name: str) -> None:
    """Enable a disabled plugin."""
    from maike.plugins.settings import load_settings, save_settings

    settings = load_settings()
    if name not in settings.disabled:
        print(f"Plugin '{name}' is already enabled.")
        return
    settings.disabled.discard(name)
    save_settings(settings)
    print(f"Enabled plugin: {name}")


def _invoke_plugin_disable(name: str) -> None:
    """Disable a plugin."""
    from maike.plugins.settings import load_settings, save_settings

    settings = load_settings()
    settings.disabled.add(name)
    save_settings(settings)
    print(f"Disabled plugin: {name}")
    print("  (takes effect on next session)")


def _invoke_plugin_update(name: str) -> None:
    """Update a git-installed plugin."""
    from maike.plugins.installer import PluginInstallError, update_plugin

    try:
        manifest = update_plugin(name)
        if manifest is None:
            print(f"Plugin '{name}' was installed from a local directory — cannot auto-update.")
        else:
            print(f"Updated plugin: {manifest.name} v{manifest.version}")
    except PluginInstallError as exc:
        print(f"Update failed: {exc}")
        raise SystemExit(1)


def _invoke_plugin_uninstall(name: str, remove_data: bool = False) -> None:
    """Remove an installed plugin."""
    from maike.plugins.installer import PluginInstallError, uninstall_plugin

    try:
        uninstall_plugin(name, remove_data=remove_data)
        print(f"Uninstalled plugin: {name}")
        if remove_data:
            print("  (plugin data directory also removed)")
    except PluginInstallError as exc:
        print(f"Uninstall failed: {exc}")
        raise SystemExit(1)


def _invoke_skill_install(source: str, scope: str, force: bool = False) -> None:
    """Install standalone skill(s) from a local directory or git URL."""
    from maike.plugins.installer import PluginInstallError, install_skill

    try:
        names = install_skill(source, scope=scope, force=force)
        if len(names) == 1:
            print(f"Installed skill: {names[0]}")
        else:
            print(f"Installed {len(names)} skills: {', '.join(names)}")
    except PluginInstallError as exc:
        print(f"Install failed: {exc}")
        raise SystemExit(1)


def _invoke_worktree_command(args) -> None:
    """Dispatch `maike worktree {add|list|remove}`."""
    from maike.workflows.worktree import (
        WorktreeError,
        create_worktree,
        list_worktrees,
        remove_worktree,
    )

    action = getattr(args, "worktree_action", None)
    workspace = getattr(args, "workspace", Path.cwd())

    if action == "add":
        try:
            path = create_worktree(workspace, args.name, base=args.base)
        except WorktreeError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)
        print(f"Created worktree: {path}")
        print(f"  cd {path}")
        print("  maike chat  # run tasks isolated on this branch")
        return

    if action == "list":
        worktrees = list_worktrees(workspace)
        if not worktrees:
            print("No mAIke-managed worktrees found.")
            print("  Create one with: maike worktree add <name>")
            return
        print(f"Worktrees ({len(worktrees)}):")
        for wt in worktrees:
            print(f"  {wt.name}  branch={wt.branch}  head={wt.head}")
            print(f"    {wt.path}")
        return

    if action == "remove":
        try:
            remove_worktree(workspace, args.name, delete_branch=args.delete_branch)
        except WorktreeError as exc:
            print(f"Error: {exc}")
            raise SystemExit(1)
        extra = " (branch deleted)" if args.delete_branch else ""
        print(f"Removed worktree: {args.name}{extra}")
        return

    print("Usage: maike worktree {add <name> | list | remove <name>}")
    raise SystemExit(2)


def _add_advisor_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared --advisor/--advisor-* flags to a subparser."""
    parser.add_argument(
        "--advisor",
        action="store_true",
        default=ADVISOR_ENABLED_DEFAULT,
        help="Enable the advisor pattern: local/cheap executor + frontier-model advisor at key decision points.",
    )
    parser.add_argument(
        "--advisor-provider",
        default=None,
        help="Provider for advisor calls (defaults to the executor's provider).",
    )
    parser.add_argument(
        "--advisor-model",
        default=None,
        help="Advisor model (defaults to the strong-tier model for the advisor's provider).",
    )
    parser.add_argument(
        "--advisor-budget-pct",
        type=float,
        default=ADVISOR_BUDGET_FRACTION_DEFAULT,
        # argparse treats '%' specially in help strings — escape with '%%'.
        help=(
            "Advisor budget as a fraction of --budget "
            f"(default {int(ADVISOR_BUDGET_FRACTION_DEFAULT * 100)}%%)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maike")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", "-m", default=None)
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("task")
    run_parser.add_argument("--workspace", "-w", type=Path, default=Path.cwd())
    run_parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    run_parser.add_argument("--model", "-m", default=None)
    run_parser.add_argument("--language", default="python")
    run_parser.add_argument("--budget", "-b", type=float, default=DEFAULT_RUN_BUDGET_USD)
    run_parser.add_argument("--yes", "-y", action="store_true")
    run_parser.add_argument("--verbose", "-v", action="store_true")
    run_parser.add_argument("--dynamic-agents", action="store_true")
    run_parser.add_argument("--parallel-coding", action="store_true")
    run_parser.add_argument("--log", type=Path, default=None)
    run_parser.add_argument("--thread", type=str, default=None, help="Resume a specific thread by ID.")
    run_parser.add_argument("--new-thread", action="store_true", help="Force a new thread instead of resuming the last one.")
    run_parser.add_argument("--dry-run", "-n", action="store_true", help="Estimate tokens and cost without executing.")
    run_parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive model tier routing (use --model for all calls).")
    run_parser.add_argument(
        "--show-constraints", action="store_true",
        help=(
            "Run the task-constraint extractor, print the resulting "
            "markdown to stderr, and exit without running the session. "
            "Useful for inspecting what the <maike-constraints> block will "
            "contain without burning session budget."
        ),
    )
    _add_advisor_args(run_parser)

    threads_parser = subparsers.add_parser("threads", help="List threads in this workspace.")
    threads_parser.add_argument("--workspace", "-w", type=Path, default=Path.cwd())

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("session_id")
    resume_parser.add_argument("--workspace", "-w", type=Path, required=True)
    resume_parser.add_argument("--verbose", "-v", action="store_true")
    resume_parser.add_argument("--log", type=Path, default=None)
    _add_advisor_args(resume_parser)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--suite", default="all")
    eval_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    eval_parser.add_argument("--provider", default=None)
    eval_parser.add_argument("--model", default=None)
    eval_parser.add_argument("--budget", "-b", type=float, default=DEFAULT_RUN_BUDGET_USD)
    eval_parser.add_argument("--keep-workspaces", action="store_true")
    eval_parser.add_argument("--output", type=Path, default=None)
    eval_parser.add_argument("--adaptive", action="store_true", default=False, help="Enable adaptive model routing in evals (disabled by default).")
    _add_advisor_args(eval_parser)
    eval_mode_group = eval_parser.add_mutually_exclusive_group()
    eval_mode_group.add_argument("--compare-baseline", action="store_true")
    eval_mode_group.add_argument("--update-baseline", action="store_true")

    cost_parser = subparsers.add_parser("cost")
    cost_parser.add_argument("session_id", nargs="?")
    cost_parser.add_argument("--session-id", dest="session_id_option", default=None)
    cost_parser.add_argument("--last", type=int, default=10)
    cost_parser.add_argument("--workspace", type=Path, default=Path.cwd())

    history_parser = subparsers.add_parser("history")
    history_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    history_parser.add_argument("--limit", type=int, default=20)

    subparsers.add_parser("setup", help="Add or update AI provider API keys.")

    init_parser = subparsers.add_parser("init", help="Generate MAIKE.md for this workspace.")
    init_parser.add_argument("--workspace", "-w", type=Path, default=Path.cwd())
    init_parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    init_parser.add_argument("--model", "-m", default=None)
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing MAIKE.md.")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive coding session.")
    chat_parser.add_argument("--workspace", "-w", type=Path, default=Path.cwd())
    chat_parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    chat_parser.add_argument("--model", "-m", default=None)
    chat_parser.add_argument("--budget", "-b", type=float, default=DEFAULT_RUN_BUDGET_USD)
    chat_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed LLM and tool traces")
    chat_parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve tool calls")
    _add_advisor_args(chat_parser)

    plugin_parser = subparsers.add_parser("plugin", help="Manage plugins.")
    plugin_subs = plugin_parser.add_subparsers(dest="plugin_command")
    plugin_list_parser = plugin_subs.add_parser("list", help="List discovered plugins and their components.")
    plugin_list_parser.add_argument("--workspace", "-w", type=Path, default=Path.cwd())
    plugin_validate_parser = plugin_subs.add_parser("validate", help="Validate a plugin directory.")
    plugin_validate_parser.add_argument("path", type=Path, nargs="?", default=Path.cwd())
    plugin_install_parser = plugin_subs.add_parser("install", help="Install a plugin from a local directory or git URL.")
    plugin_install_parser.add_argument("source", help="Local directory path or git repository URL.")
    plugin_install_parser.add_argument("--dir", dest="install_dir", default=None, help="Custom target directory.")
    plugin_install_parser.add_argument("--scope", choices=["user", "project"], default="user")
    plugin_install_parser.add_argument("--force", action="store_true", help="Overwrite existing plugin.")
    plugin_enable_parser = plugin_subs.add_parser("enable", help="Enable a disabled plugin.")
    plugin_enable_parser.add_argument("name", help="Plugin name.")
    plugin_disable_parser = plugin_subs.add_parser("disable", help="Disable a plugin.")
    plugin_disable_parser.add_argument("name", help="Plugin name.")
    plugin_update_parser = plugin_subs.add_parser("update", help="Update a git-installed plugin.")
    plugin_update_parser.add_argument("name", help="Plugin name.")
    plugin_uninstall_parser = plugin_subs.add_parser("uninstall", help="Remove an installed plugin.")
    plugin_uninstall_parser.add_argument("name", help="Plugin name.")
    plugin_uninstall_parser.add_argument("--remove-data", action="store_true", help="Also remove plugin data directory.")

    # Skill subcommands
    skill_parser = subparsers.add_parser("skill", help="Manage skills.")
    skill_subs = skill_parser.add_subparsers(dest="skill_command")
    skill_install_parser = skill_subs.add_parser("install", help="Install skill(s) from a local directory or git URL.")
    skill_install_parser.add_argument("source", help="Local directory path or git repository URL.")
    skill_install_parser.add_argument("--scope", choices=["user", "project"], default="user")
    skill_install_parser.add_argument("--force", action="store_true", help="Overwrite existing skill.")

    # Worktree subcommands
    worktree_parser = subparsers.add_parser("worktree", help="Manage git worktrees for isolated branches.")
    worktree_subs = worktree_parser.add_subparsers(dest="worktree_action")
    wt_add = worktree_subs.add_parser("add", help="Create a new worktree on a new branch.")
    wt_add.add_argument("name", help="Branch/worktree name.")
    wt_add.add_argument("--base", default="HEAD", help="Base commit/branch (default: HEAD).")
    wt_add.add_argument("--workspace", "-w", type=Path, default=Path.cwd())
    wt_list = worktree_subs.add_parser("list", help="List mAIke-managed worktrees.")
    wt_list.add_argument("--workspace", "-w", type=Path, default=Path.cwd())
    wt_rm = worktree_subs.add_parser("remove", help="Remove a worktree.")
    wt_rm.add_argument("name", help="Worktree name to remove.")
    wt_rm.add_argument("--delete-branch", action="store_true", help="Also delete the associated branch.")
    wt_rm.add_argument("--workspace", "-w", type=Path, default=Path.cwd())

    return parser


def main(argv: list[str] | None = None):
    if typer is not None:  # pragma: no cover - optional dependency
        app()
        return

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.command is None:
        # Bare `maike` launches the interactive REPL (same as `maike chat`).
        invoke_chat_command(
            workspace=Path.cwd(),
            provider=args.provider,
            model=args.model,
        )
        return None


    if args.command == "setup":
        invoke_setup_command()
        return None

    if args.command == "init":
        invoke_init_command(
            args.workspace,
            provider=args.provider,
            model=args.model,
            force=args.force,
        )
        return None

    if args.command == "run":
        if args.show_constraints:
            # Inspection flag — prints extracted markdown to stderr and exits
            # without running a session.  Combines cleanly with --dry-run
            # (caller can invoke each separately to see both outputs).
            show_constraints_command(
                task=args.task,
                workspace=args.workspace,
                provider=args.provider,
                model=args.model,
            )
            return None
        if args.dry_run:
            return dry_run_command(
                task=args.task,
                workspace=args.workspace,
                provider=args.provider,
                model=args.model,
                budget=args.budget,
            )
        result = invoke_run_command(
            task=args.task,
            workspace=args.workspace,
            provider=args.provider,
            model=args.model,
            language=args.language,
            language_explicit="--language" in raw_argv,
            budget=args.budget,
            yes=args.yes,
            verbose=args.verbose,
            dynamic_agents_enabled=args.dynamic_agents,
            parallel_coding_enabled=args.parallel_coding,
            log_path=args.log,
            thread_id=args.thread,
            new_thread=args.new_thread,
            adaptive_model=not args.no_adaptive,
            advisor_enabled=args.advisor,
            advisor_provider=args.advisor_provider,
            advisor_model=args.advisor_model,
            advisor_budget_pct=args.advisor_budget_pct,
        )
        # Final summary (interactive mode already shows per-turn results)
        _render_run_result(result, args.workspace, streamed=True)
        return result
    if args.command == "threads":
        _list_threads(args.workspace)
        return None
    if args.command == "resume":
        result = invoke_resume_command(
            session_id=args.session_id,
            workspace=args.workspace,
            verbose=args.verbose,
            log_path=args.log,
            advisor_enabled=args.advisor,
            advisor_provider=args.advisor_provider,
            advisor_model=args.advisor_model,
            advisor_budget_pct=args.advisor_budget_pct,
        )
        _render_run_result(result, args.workspace, streamed=True)
        return result
    if args.command == "eval":
        summary = eval_command(
            args.suite,
            args.workspace,
            args.provider,
            args.model,
            args.budget,
            args.keep_workspaces,
            args.output,
            args.compare_baseline,
            args.update_baseline,
            adaptive_model=args.adaptive,
            advisor_enabled=args.advisor,
            advisor_provider=args.advisor_provider,
            advisor_model=args.advisor_model,
            advisor_budget_pct=args.advisor_budget_pct,
        )
        _render_eval_summary(summary)
        if args.compare_baseline and summary["regression_failed"]:
            raise SystemExit(1)
        return summary
    if args.command == "cost":
        session_id = args.session_id or args.session_id_option
        summaries = cost_command(session_id, args.last, args.workspace)
        if session_id and not summaries:
            _emit_cli_error(f"Session not found: {session_id}")
            raise SystemExit(1)
        if session_id and summaries:
            _render_cost_detail(summaries[0])
        else:
            _render_cost_list(summaries)
        return summaries
    if args.command == "history":
        sessions = history_command(args.workspace, args.limit)
        _render_history(sessions)
        return sessions
    if args.command == "chat":
        invoke_chat_command(
            workspace=args.workspace,
            provider=args.provider,
            model=args.model,
            budget=args.budget,
            verbose=args.verbose,
            yes=args.yes,
            advisor_enabled=args.advisor,
            advisor_provider=args.advisor_provider,
            advisor_model=args.advisor_model,
            advisor_budget_pct=args.advisor_budget_pct,
        )
        return None
    if args.command == "plugin":
        _invoke_plugin_command(args)
        return None
    if args.command == "skill":
        cmd = getattr(args, "skill_command", None)
        if cmd == "install":
            _invoke_skill_install(args.source, args.scope, getattr(args, "force", False))
        else:
            print("Usage: maike skill {install}")
        return None
    if args.command == "worktree":
        _invoke_worktree_command(args)
        return None
    raise SystemExit(2)


if typer is not None:  # pragma: no cover - optional dependency
    app = typer.Typer(name="maike", help="mAIke - your local coding agent", invoke_without_command=True)

    @app.callback(invoke_without_command=True)
    def _default_callback(
        ctx: typer.Context,
        provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", help="LLM provider"),
        model: str | None = typer.Option(None, "--model", "-m", help="Model name (defaults to the provider's default)"),
        budget: float = typer.Option(DEFAULT_RUN_BUDGET_USD, "--budget", "-b", help="Session budget in USD"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed LLM and tool traces"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve tool calls"),
    ):
        if ctx.invoked_subcommand is not None:
            return
        # Bare `maike` launches the interactive REPL (same as `maike chat`).
        invoke_chat_command(
            workspace=Path.cwd(),
            provider=provider,
            model=model,
            budget=budget,
            verbose=verbose,
            yes=yes,
        )

    @app.command()
    def setup():
        """Add or update AI provider API keys."""
        invoke_setup_command()

    @app.command()
    def init(
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        provider: str = typer.Option(DEFAULT_PROVIDER, "--provider"),
        model: str | None = typer.Option(None, "--model", "-m"),
        force: bool = typer.Option(False, "--force"),
    ):
        """Generate MAIKE.md for this workspace using the fast model."""
        invoke_init_command(workspace, provider=provider, model=model, force=force)

    @app.command()
    def run(
        ctx: typer.Context,
        task: str,
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        provider: str = typer.Option(DEFAULT_PROVIDER, "--provider"),
        model: str | None = typer.Option(None, "--model", "-m"),
        language: str = typer.Option("python", "--language"),
        budget: float = typer.Option(DEFAULT_RUN_BUDGET_USD, "--budget", "-b"),
        yes: bool = typer.Option(False, "--yes", "-y"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
        dynamic_agents_enabled: bool = typer.Option(False, "--dynamic-agents"),
        parallel_coding_enabled: bool = typer.Option(False, "--parallel-coding"),
        log: Path | None = typer.Option(None, "--log"),
        dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Estimate tokens and cost without executing."),
        no_adaptive: bool = typer.Option(False, "--no-adaptive", help="Disable adaptive model tier routing."),
        show_constraints: bool = typer.Option(
            False, "--show-constraints",
            help=(
                "Run the task-constraint extractor, print the resulting "
                "markdown to stderr, and exit without running the session."
            ),
        ),
        advisor: bool = typer.Option(False, "--advisor", help="Enable the advisor pattern."),
        advisor_provider: str | None = typer.Option(None, "--advisor-provider"),
        advisor_model: str | None = typer.Option(None, "--advisor-model"),
        advisor_budget_pct: float = typer.Option(ADVISOR_BUDGET_FRACTION_DEFAULT, "--advisor-budget-pct"),
    ):
        if show_constraints:
            # Inspection flag — prints extracted markdown to stderr and exits
            # without running a session.  See show_constraints_command.
            show_constraints_command(
                task=task,
                workspace=workspace,
                provider=provider,
                model=model,
            )
            return
        if dry_run:
            dry_run_command(
                task=task,
                workspace=workspace,
                provider=provider,
                model=model,
                budget=budget,
            )
            return
        result = invoke_run_command(
            task=task,
            workspace=workspace,
            provider=provider,
            model=model,
            language=language,
            language_explicit=(
                ctx.get_parameter_source("language").name == "COMMANDLINE"
                if hasattr(ctx, "get_parameter_source")
                else False
            ),
            budget=budget,
            yes=yes,
            verbose=verbose,
            dynamic_agents_enabled=dynamic_agents_enabled,
            parallel_coding_enabled=parallel_coding_enabled,
            log_path=log,
            adaptive_model=not no_adaptive,
            advisor_enabled=advisor,
            advisor_provider=advisor_provider,
            advisor_model=advisor_model,
            advisor_budget_pct=advisor_budget_pct,
        )
        _render_run_result(result, workspace)

    @app.command()
    def resume(
        session_id: str,
        workspace: Path = typer.Option(..., "--workspace", "-w"),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
        log: Path | None = typer.Option(None, "--log"),
    ):
        result = invoke_resume_command(
            session_id=session_id,
            workspace=workspace,
            verbose=verbose,
            log_path=log,
        )
        _render_run_result(result, workspace)

    @app.command()
    def eval(
        suite: str = typer.Option("all"),
        workspace: Path = typer.Option(Path.cwd()),
        provider: str | None = typer.Option(None, "--provider"),
        model: str | None = typer.Option(None, "--model"),
        budget: float = typer.Option(DEFAULT_RUN_BUDGET_USD, "--budget", "-b"),
        keep_workspaces: bool = typer.Option(False, "--keep-workspaces"),
        output: Path | None = typer.Option(None, "--output"),
        compare_baseline: bool = typer.Option(False, "--compare-baseline"),
        update_baseline: bool = typer.Option(False, "--update-baseline"),
        adaptive: bool = typer.Option(False, "--adaptive", help="Enable adaptive model routing (disabled by default in evals)."),
    ):
        if compare_baseline and update_baseline:
            raise typer.BadParameter("--compare-baseline and --update-baseline cannot be used together")
        summary = eval_command(
            suite,
            workspace,
            provider,
            model,
            budget,
            keep_workspaces,
            output,
            compare_baseline,
            update_baseline,
            adaptive_model=adaptive,
        )
        _render_eval_summary(summary)
        if compare_baseline and summary["regression_failed"]:
            raise typer.Exit(code=1)

    @app.command(name="swe-bench")
    def swe_bench(
        variant: str = typer.Option("lite", help="lite, verified, or full"),
        provider: str = typer.Option(None, "--provider"),
        model: str = typer.Option(None, "--model", "-m"),
        budget: float = typer.Option(0.0, "--budget", "-b", help="USD per instance (0=unlimited)"),
        max_instances: int = typer.Option(None, "--max-instances", help="Cap number of instances"),
        instance_ids: str = typer.Option(None, "--instance-ids", help="Comma-separated instance IDs"),
        output: Path = typer.Option(None, "--output", "-o", help="Predictions JSONL path"),
        resume: Path = typer.Option(None, "--resume", help="Resume from existing predictions"),
        timeout: int = typer.Option(300, "--timeout", help="Per-instance timeout in seconds"),
        keep_workspaces: bool = typer.Option(False, "--keep-workspaces"),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        advisor: bool = typer.Option(False, "--advisor", help="Enable frontier-model advisor co-pilot"),
        advisor_provider: str = typer.Option(None, "--advisor-provider", help="Advisor provider (defaults to executor provider)"),
        advisor_model: str = typer.Option(None, "--advisor-model", help="Advisor model (e.g. gemini-3.1-pro-preview)"),
        advisor_budget_pct: float = typer.Option(None, "--advisor-budget-pct", help="Advisor spend cap as fraction of session budget (default 0.2)"),
    ):
        """Run SWE-bench evaluation and generate predictions."""
        _load_all_dotenv()
        effective_provider = provider or get_configured_provider() or "ollama"
        _ensure_configured(effective_provider)
        resolved_model = model or DEFAULT_MODEL
        ids_list = [i.strip() for i in instance_ids.split(",")] if instance_ids else None

        from maike.eval.swebench_runner import SWEBenchRunner

        runner = SWEBenchRunner(
            workspace_root=workspace.resolve(),
            keep_workspaces=keep_workspaces,
        )
        report = asyncio.run(runner.run(
            variant=variant,
            provider=effective_provider,
            model=resolved_model,
            budget_per_instance=budget,
            timeout_per_instance=timeout,
            max_instances=max_instances,
            instance_ids=ids_list,
            output_path=output,
            resume_path=resume,
            advisor_enabled=advisor,
            advisor_provider=advisor_provider,
            advisor_model=advisor_model,
            advisor_budget_pct=advisor_budget_pct,
        ))
        sys.stderr.write(f"\nPredictions written to: {report.predictions_path}\n")
        sys.stderr.write(
            f"To evaluate: python -m swebench.harness.run_evaluation "
            f"--predictions_path {report.predictions_path} "
            f"--dataset_name princeton-nlp/SWE-bench_Lite --run_id maike-eval\n"
        )

    @app.command()
    def cost(
        session_id: str | None = typer.Argument(None),
        last: int = typer.Option(10),
        workspace: Path = typer.Option(Path.cwd()),
    ):
        summaries = cost_command(session_id, last, workspace)
        if session_id and not summaries:
            _emit_cli_error(f"Session not found: {session_id}")
            raise typer.Exit(code=1)
        if session_id and summaries:
            _render_cost_detail(summaries[0])
            return
        _render_cost_list(summaries)

    @app.command()
    def history(
        workspace: Path = typer.Option(Path.cwd()),
        limit: int = typer.Option(20),
    ):
        _render_history(history_command(workspace, limit))

    @app.command()
    def chat(
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
        provider: str = typer.Option(DEFAULT_PROVIDER, "--provider"),
        model: str | None = typer.Option(None, "--model", "-m"),
        budget: float = typer.Option(DEFAULT_RUN_BUDGET_USD, "--budget", "-b"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed LLM and tool traces"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve tool calls"),
        no_tui: bool = typer.Option(False, "--no-tui", help="Use legacy text REPL instead of TUI"),
        advisor: bool = typer.Option(False, "--advisor", help="Enable the advisor pattern."),
        advisor_provider: str | None = typer.Option(None, "--advisor-provider"),
        advisor_model: str | None = typer.Option(None, "--advisor-model"),
        advisor_budget_pct: float = typer.Option(ADVISOR_BUDGET_FRACTION_DEFAULT, "--advisor-budget-pct"),
    ):
        """Start an interactive coding session."""
        invoke_chat_command(
            workspace=workspace,
            provider=provider,
            model=model,
            budget=budget,
            verbose=verbose,
            yes=yes,
            no_tui=no_tui,
            advisor_enabled=advisor,
            advisor_provider=advisor_provider,
            advisor_model=advisor_model,
            advisor_budget_pct=advisor_budget_pct,
        )

    plugin_app = typer.Typer(name="plugin", help="Manage plugins.")
    app.add_typer(plugin_app, name="plugin")

    @plugin_app.command("list")
    def plugin_list(
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ):
        """List discovered plugins and their components."""
        _invoke_plugin_list(workspace)

    @plugin_app.command("validate")
    def plugin_validate(
        path: Path = typer.Argument(Path.cwd()),
    ):
        """Validate a plugin directory."""
        _invoke_plugin_validate(path)

    @plugin_app.command("install")
    def plugin_install_cmd(
        source: str = typer.Argument(..., help="Local directory path or git URL."),
        dir: str = typer.Option(None, "--dir", help="Custom target directory."),
        scope: str = typer.Option("user", "--scope", help="user or project."),
        force: bool = typer.Option(False, "--force", help="Overwrite existing plugin."),
    ):
        """Install a plugin from a local directory or git URL."""
        _invoke_plugin_install(source, dir, scope, force)

    @plugin_app.command("enable")
    def plugin_enable_cmd(name: str = typer.Argument(..., help="Plugin name.")):
        """Enable a disabled plugin."""
        _invoke_plugin_enable(name)

    @plugin_app.command("disable")
    def plugin_disable_cmd(name: str = typer.Argument(..., help="Plugin name.")):
        """Disable a plugin."""
        _invoke_plugin_disable(name)

    @plugin_app.command("update")
    def plugin_update_cmd(name: str = typer.Argument(..., help="Plugin name.")):
        """Update a git-installed plugin."""
        _invoke_plugin_update(name)

    @plugin_app.command("uninstall")
    def plugin_uninstall_cmd(
        name: str = typer.Argument(..., help="Plugin name."),
        remove_data: bool = typer.Option(False, "--remove-data", help="Also remove plugin data directory."),
    ):
        """Remove an installed plugin."""
        _invoke_plugin_uninstall(name, remove_data)

    # ── Skill commands ─────────────────────────────────────────────────
    skill_app = typer.Typer(name="skill", help="Manage skills.")
    app.add_typer(skill_app, name="skill")

    @skill_app.command("install")
    def skill_install_cmd(
        source: str = typer.Argument(..., help="Local directory path or git URL."),
        scope: str = typer.Option("user", "--scope", help="user or project."),
        force: bool = typer.Option(False, "--force", help="Overwrite existing skill."),
    ):
        """Install skill(s) from a local directory or git URL."""
        _invoke_skill_install(source, scope, force)

    # ── Worktree commands ──────────────────────────────────────────────
    worktree_app = typer.Typer(name="worktree", help="Manage git worktrees for isolated branches.")
    app.add_typer(worktree_app, name="worktree")

    @worktree_app.command("add")
    def worktree_add_cmd(
        name: str = typer.Argument(..., help="Branch/worktree name."),
        base: str = typer.Option("HEAD", "--base", help="Base commit/branch."),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ):
        """Create a new worktree on a new branch."""
        from types import SimpleNamespace
        _invoke_worktree_command(SimpleNamespace(
            worktree_action="add", name=name, base=base, workspace=workspace,
        ))

    @worktree_app.command("list")
    def worktree_list_cmd(
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ):
        """List mAIke-managed worktrees."""
        from types import SimpleNamespace
        _invoke_worktree_command(SimpleNamespace(
            worktree_action="list", workspace=workspace,
        ))

    @worktree_app.command("remove")
    def worktree_remove_cmd(
        name: str = typer.Argument(..., help="Worktree name to remove."),
        delete_branch: bool = typer.Option(False, "--delete-branch", help="Also delete the branch."),
        workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w"),
    ):
        """Remove a worktree."""
        from types import SimpleNamespace
        _invoke_worktree_command(SimpleNamespace(
            worktree_action="remove", name=name,
            delete_branch=delete_branch, workspace=workspace,
        ))
