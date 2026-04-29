"""Main Textual application for mAIke TUI."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from rich.markup import escape as _rich_escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, OptionList

from maike.tui.commands import SLASH_COMMANDS, find_command
from maike.tui.widgets.approval_list import APPROVE, APPROVE_ALWAYS, ApprovalList
from maike.tui.widgets.header_bar import HeaderBar
from maike.tui.widgets.input_area import PromptInput
from maike.tui.widgets.message_list import MessageList
from maike.tui.widgets.status_bar import StatusBar


def _copy_via_native_tool(text: str) -> bool:
    """Copy *text* to the system clipboard via a platform-native subprocess.

    macOS uses ``pbcopy``; Linux tries ``wl-copy`` (Wayland) then ``xclip``;
    Windows uses ``clip``.  Returns ``True`` on success.  Falls through to
    the caller (which should try OSC 52 next) on any failure — including
    missing tools, timeouts, or non-zero exits.

    Native tools are preferred over OSC 52 because OSC 52 is dropped
    silently in some common terminals (notably stock macOS Terminal.app).
    """
    if sys.platform == "darwin":
        cmd = ["pbcopy"]
    elif sys.platform.startswith("linux"):
        if shutil.which("wl-copy"):
            cmd = ["wl-copy"]
        elif shutil.which("xclip"):
            cmd = ["xclip", "-selection", "clipboard"]
        else:
            return False
    elif sys.platform == "win32":
        cmd = ["clip"]
    else:
        return False
    try:
        subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            timeout=2,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


class MaikeTUIApp(App):
    """The mAIke terminal user interface."""

    TITLE = "mAIke"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        # Ctrl+C does double duty: copy selection if one exists, else quit.
        # ``priority=True`` is REQUIRED — without it Textual's Screen-level
        # ``ctrl+c → screen.copy_text`` and TextArea's ``ctrl+c → copy``
        # bindings fire first, short-circuiting our pbcopy-backed handler
        # and falling back to OSC 52 (which Apple_Terminal silently drops).
        Binding("ctrl+c", "copy_or_quit", "Copy/Quit", priority=True),
        ("ctrl+d", "quit", "Quit"),
        Binding("pageup", "scroll_up", "Scroll Up", show=False),
        Binding("pagedown", "scroll_down", "Scroll Down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
        Binding("escape", "cancel_action", "Cancel", show=False),
        Binding("ctrl+l", "clear_view", "Clear", show=True),
        Binding("ctrl+n", "new_thread", "New Thread", show=True),
        Binding("ctrl+t", "toggle_dark", "Theme", show=True),
        Binding("ctrl+o", "view_full_output", "Expand", show=True),
    ]

    def __init__(
        self,
        workspace: Path,
        provider: str = "",
        model: str = "",
        budget: float = 5.0,
        verbose: bool = False,
        yes: bool = False,
        language: str = "python",
        adaptive_model: bool = True,
        advisor_enabled: bool = False,
        advisor_provider: str | None = None,
        advisor_model: str | None = None,
        advisor_budget_pct: float = 0.20,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.budget = budget
        self.verbose = verbose
        self.yes = yes
        self.language = language
        self.adaptive_model = adaptive_model
        self.advisor_enabled = advisor_enabled
        self.advisor_provider = advisor_provider
        self.advisor_model = advisor_model
        self.advisor_budget_pct = advisor_budget_pct

        self._orchestrator: Any = None
        self._trace_sink: Any = None
        self._stream_sink: Any = None
        self._input_future: asyncio.Future | None = None
        self._task_running = False
        self._session_id = ""
        self._conversation_thread_id: str | None = None
        self._approval_callback: Callable | None = None
        self._waiting_for_approval: bool = False
        self._cancellation: Any = None  # CancellationController
        self._worker: Any = None  # Active Textual Worker running _execute_task
        self._streaming_shown: bool = False

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        yield HeaderBar(
            workspace=str(self.workspace),
            model=self.model,
            provider=self.provider,
        )
        yield MessageList()
        # Bottom stack is a single docked Vertical container so we can
        # put the input + status bar in a normal vertical flow with real
        # gaps between them.  Docking each child separately doesn't work
        # — Textual collapses them all to the same row.
        with Vertical(id="bottom-stack"):
            yield PromptInput()
            yield StatusBar(model=self.model, provider=self.provider, budget=self.budget)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(PromptInput).focus()
        ml = self.query_one(MessageList)
        self._show_welcome(ml)

    # The same ASCII banner the legacy REPL prints on launch — kept as a
    # single source of truth so the TUI and REPL match byte-for-byte.
    _BANNER_LINES = (
        r"            █████╗ ██╗██╗  ██╗         ",
        r"  ████╗████╗██╔══██╗██║██║ ██╔╝██████╗ ",
        r"  ██╔███╔██║███████║██║█████╔╝ ██╔═══╝ ",
        r"  ██║╚█╚╝██║██╔══██║██║██╔═██╗ █████╗  ",
        r"  ██║ ╚╝ ██║██║  ██║██║██║ ╚██╗██╔══╝  ",
        r"  ╚═╝    ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═════╝",
        r"   your local coding agent        ░▒▓█  ",
    )
    # Per-line gradient matching maike.cli._GRAD (ANSI 256-color palette
    # entries 199, 177, 141, 105, 75, 45, 43).
    _BANNER_COLORS = (
        "#ff00af",  # hot pink
        "#d787d7",  # magenta
        "#af87d7",  # purple
        "#8787d7",  # indigo
        "#5fafd7",  # blue
        "#00d7ff",  # cyan
        "#00d7af",  # teal
    )

    def _show_welcome(self, ml) -> None:
        """Render the gradient ASCII banner plus a compact meta line."""
        from maike.tui.theme import MAIKE_DIM

        # Banner — each row painted with its gradient color, bold.  We
        # mount each line as a pre-escaped Static so the ``[`` and ``]``
        # inside the box-drawing art are not interpreted as Rich markup.
        from rich.text import Text as RichText
        from textual.widgets import Static

        ml.mount(Static(""))  # top spacer
        for line, color in zip(self._BANNER_LINES, self._BANNER_COLORS):
            ml.mount(
                Static(
                    RichText(line, style=f"bold {color}"),
                    classes="banner-line",
                )
            )

        # Compact meta line: model · budget · workspace · /help hint
        model_str = f"{self.provider}:{self.model}" if self.provider else self.model
        ws_short = str(self.workspace).rstrip("/").rsplit("/", 1)[-1] or "/"
        meta = (
            f"  [{MAIKE_DIM}]{model_str}  \u00b7  "
            f"${self.budget:.2f}  \u00b7  "
            f"ws:{ws_short}  \u00b7  "
            f"type [/{MAIKE_DIM}][bold]/help[/bold][{MAIKE_DIM}] to begin[/{MAIKE_DIM}]"
        )
        ml.add_info("")
        ml.add_info(meta)

    # ------------------------------------------------------------------
    # Scroll actions
    # ------------------------------------------------------------------

    def action_scroll_up(self) -> None:
        self.query_one(MessageList).scroll_page_up(animate=False)

    def action_scroll_down(self) -> None:
        self.query_one(MessageList).scroll_page_down(animate=False)

    def action_scroll_home(self) -> None:
        self.query_one(MessageList).scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        self.query_one(MessageList).scroll_end(animate=False)

    # ------------------------------------------------------------------
    # View actions
    # ------------------------------------------------------------------

    def action_clear_view(self) -> None:
        ml = self.query_one(MessageList)
        ml.clear_content()
        ml.add_info("[dim]Screen cleared.[/dim]")

    def action_new_thread(self) -> None:
        self._conversation_thread_id = None
        self.query_one(MessageList).add_info("Starting new conversation thread.")
        self.query_one(HeaderBar).update_session(thread="new")

    def action_cancel_action(self) -> None:
        if self._waiting_for_approval and self._approval_callback:
            self._dismiss_approval(False)
            return
        if not self._task_running:
            return

        # Signal the orchestrator (polled at stage gates) AND hard-cancel
        # the Textual worker.  The worker cancel raises CancelledError on
        # the next `await` — almost always an in-flight LLM stream_call —
        # so the LLM actually stops talking immediately instead of
        # finishing its current turn.  Without the worker cancel the
        # cancellation flag is only checked at orchestrator gates, which
        # is too coarse to feel responsive from the TUI.
        if self._cancellation is not None:
            self._cancellation.request_cancel()
        worker = self._worker
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass

        ml = self.query_one(MessageList)
        ml.stop_spinner()
        ml.finalize_streaming()
        ml.add_info("[yellow]Cancelled by user.[/yellow]")

        # Resolve any pending follow-up future so it doesn't leak.
        fut = self._input_future
        if fut is not None and not fut.done():
            fut.set_result(None)

        # Return control to the input immediately; the worker's finally
        # block will still run its own cleanup when the cancellation
        # propagates through, but the user shouldn't have to wait for it.
        self._task_running = False
        self._worker = None
        self._cancellation = None
        self._streaming_shown = False
        self._input_future = None
        self._enable_input(mode="task")

    def action_view_full_output(self) -> None:
        """Open the full output of the most recent tool or delegate in a modal.

        The TUI's inline collapsible shows a generous preview but caps at
        ``_INLINE_PREVIEW_MAX_LINES`` to keep the scroll buffer usable.
        This action retrieves the complete text from the most recent
        tool/delegate widget and shows it in a dedicated, scrollable
        modal — letting the user inspect anything truncated inline.
        """
        ml = self.query_one(MessageList)
        target = ml.latest_output_widget()
        if target is None or not getattr(target, "full_output", ""):
            ml.add_info(
                "[dim]No tool or delegate output available to expand yet.[/dim]"
            )
            return
        from maike.tui.screens.full_output_screen import FullOutputScreen

        name = getattr(target, "tool_name", "Output")
        self.push_screen(FullOutputScreen(title=name, content=target.full_output))

    def action_copy_or_quit(self) -> None:
        """Ctrl+C: copy the current text selection if any, else quit.

        Tries platform-native clipboard tools first (``pbcopy`` on macOS,
        ``wl-copy``/``xclip`` on Linux, ``clip`` on Windows).  Falls back
        to OSC 52 if the native path is unavailable or fails.
        """
        selection = self.screen.get_selected_text()
        if selection:
            if not _copy_via_native_tool(selection):
                self.copy_to_clipboard(selection)  # OSC 52 fallback
            self.screen.clear_selection()
            self.query_one(MessageList).add_info(
                f"[dim]Copied {len(selection)} chars to clipboard.[/dim]"
            )
            return
        self.action_quit()

    # ------------------------------------------------------------------
    # Approval — inline OptionList
    # ------------------------------------------------------------------

    def _show_approval_list(self, prompt: str, callback: Callable[[bool], None]) -> None:
        self._approval_callback = callback
        self._waiting_for_approval = True
        self.query_one(PromptInput).display = False
        approval = ApprovalList(prompt, self._on_approval_selected)
        self.mount(approval, before=self.query_one(Footer))

    def _on_approval_selected(self, value: str) -> None:
        approved = value in (APPROVE, APPROVE_ALWAYS)
        self._dismiss_approval(approved)

    def _dismiss_approval(self, approved: bool) -> None:
        self._waiting_for_approval = False
        for w in self.query(ApprovalList):
            w.remove()
        pi = self.query_one(PromptInput)
        pi.display = True
        pi.set_mode("task")
        cb = self._approval_callback
        self._approval_callback = None
        if cb:
            cb(approved)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if isinstance(event.option_list, ApprovalList):
            self._on_approval_selected(event.option_id or "deny")

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.stop()

        prompt = self.query_one(PromptInput)
        prompt.save_to_history(text)
        prompt.clear()

        if text.startswith("/") and self._try_handle_slash_command(text):
            return
        # If the line starts with `/` but didn't match a known command we
        # fall through and submit it as a prose task to the LLM — the
        # user is probably typing a sentence that happens to begin with
        # a slash (e.g. `/tmp/foo doesn't exist`), not an unknown command.

        # Text prompt response (AskUser tool waiting for free-text).
        if self._waiting_for_approval and self._approval_callback:
            cb = self._approval_callback
            self._approval_callback = None
            self._waiting_for_approval = False
            prompt.set_mode("task")
            cb(text)
            return

        ml = self.query_one(MessageList)
        # User follow-up: strong divider first so the new turn is visually distinct.
        is_follow_up = self._input_future is not None and not self._input_future.done()
        if is_follow_up:
            ml.add_user_turn_separator()
        ml.add_user_message(text)

        # Follow-up input between turns.
        if is_follow_up:
            self._input_future.set_result(text)
            self._disable_input()
            return

        # New task.
        self._disable_input()
        # Keep a handle to the worker so Esc can hard-cancel it.
        self._worker = self.run_worker(self._execute_task(text))

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def _execute_task(self, task: str) -> None:
        if self._orchestrator is None:
            self._orchestrator, self._trace_sink, self._stream_sink = (
                self._build_orchestrator()
            )

        self._task_running = True
        ml = self.query_one(MessageList)

        from maike.orchestrator.orchestrator import CancellationController
        self._cancellation = CancellationController()

        try:
            run_kwargs = dict(
                initial_task=task,
                workspace=self.workspace,
                provider_name=self.provider,
                model=self.model,
                language_override=self.language,
                budget=self.budget,
                auto_approve=self.yes,
                verbose=self.verbose,
                turn_callback=self._turn_callback,
                adaptive_model=self.adaptive_model,
                thread_id=self._conversation_thread_id,
                cancellation=self._cancellation,
            )
            if self.advisor_enabled:
                run_kwargs["advisor_enabled"] = self.advisor_enabled
                run_kwargs["advisor_provider"] = self.advisor_provider
                run_kwargs["advisor_model"] = self.advisor_model
                run_kwargs["advisor_budget_pct"] = self.advisor_budget_pct
            try:
                result = await self._orchestrator.run_interactive(**run_kwargs)
            except TypeError:
                # Older orchestrator without advisor kwargs.
                for _k in (
                    "advisor_enabled", "advisor_provider",
                    "advisor_model", "advisor_budget_pct",
                ):
                    run_kwargs.pop(_k, None)
                result = await self._orchestrator.run_interactive(**run_kwargs)

            ml.stop_spinner()
            ml.finalize_streaming()

            if result is not None:
                self._session_id = getattr(result, "session_id", "")
                self._conversation_thread_id = getattr(result, "thread_id", None)
                self.query_one(HeaderBar).update_session(
                    session_id=self._session_id,
                    thread=self._conversation_thread_id or "new",
                )
                # NOTE: no fallback add_assistant_message here.  The
                # orchestrator calls _turn_callback after *every* turn,
                # including the last one, and _turn_callback either
                # finalizes the streaming widget (display path #1) or
                # mounts a fallback Markdown widget (display path #2).
                # Calling add_assistant_message again here produced a
                # duplicate copy of the final turn's reply whenever
                # streaming did show content — _turn_callback resets
                # _streaming_shown to False before returning, so the
                # guard below was effectively always False.

        except asyncio.CancelledError:
            # Esc was pressed — action_cancel_action already updated the
            # UI and re-enabled input.  Swallow the cancellation quietly
            # so it doesn't surface as an error.
            ml.stop_spinner()
            ml.finalize_streaming()
            raise
        except Exception as exc:
            ml.stop_spinner()
            ml.add_error(str(exc))
        finally:
            self._task_running = False
            self._input_future = None
            self._cancellation = None
            self._worker = None
            self._streaming_shown = False
            self.call_later(self._enable_input)

    async def _turn_callback(self, result: Any) -> str | None:
        ml = self.query_one(MessageList)
        ml.stop_spinner()
        ml.finalize_streaming()

        # Show extracted output as fallback if streaming didn't capture it.
        if not self._streaming_shown:
            agent_output = self._extract_output(result)
            if agent_output:
                ml.add_assistant_message(agent_output)
        self._streaming_shown = False

        # Enable input and wait for follow-up.
        loop = asyncio.get_running_loop()
        self._input_future = loop.create_future()

        try:
            self._enable_input(mode="follow-up")
        except Exception:
            # If enabling input fails, still unblock by auto-resolving
            self._input_future.set_result(None)

        try:
            follow_up = await self._input_future
        except asyncio.CancelledError:
            return None

        if not follow_up or follow_up.lower() in {"exit", "quit", "q"}:
            return None

        # Don't add_user_message here — on_input_submitted already displayed it.
        self._disable_input()
        return follow_up

    # ------------------------------------------------------------------
    # Approval prompts
    # ------------------------------------------------------------------

    def show_approval(self, prompt: str, callback: Callable[[bool], None]) -> None:
        self._show_approval_list(prompt, callback)

    def show_text_prompt(self, prompt: str, callback: Callable[[str], None]) -> None:
        self.query_one(MessageList).add_info(
            f"[bold yellow]Input needed:[/bold yellow] {prompt}"
        )
        self._approval_callback = lambda text: callback(str(text))
        self._waiting_for_approval = True
        self._enable_input(mode="text-prompt")

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _try_handle_slash_command(self, text: str) -> bool:
        """Dispatch *text* as a slash command if it matches the registry.

        Returns ``True`` when the line was consumed as a command (valid
        name, handler fired); ``False`` means no known command matched
        and the caller should treat the line as regular prose to submit
        to the LLM.  This lets prompts like ``/tmp/foo doesn't exist``
        or ``/ssse hello`` flow through as tasks instead of surfacing a
        noisy "Unknown command" error.
        """
        parts = text[1:].split(None, 1)
        raw_name = parts[0].lower() if parts else ""
        args = parts[1].split() if len(parts) > 1 else []

        cmd = find_command(raw_name)
        if cmd is None:
            return False

        handler = self._slash_handlers().get(cmd.name)
        if handler is None:
            # Registered command with no handler wired up — still consume
            # the line and show a clear error rather than silently
            # submitting the raw text to the LLM.
            self.query_one(MessageList).add_info(
                f"[red]Command /{cmd.name} has no handler.[/red]"
            )
            return True
        handler(self.query_one(MessageList), args)
        return True

    def _slash_handlers(self) -> dict[str, Callable]:
        """Map canonical command name -> handler callable."""
        return {
            "help": self._cmd_help,
            "cost": self._cmd_cost,
            "status": self._cmd_status,
            "skill": self._cmd_skill,
            "agent": self._cmd_agent,
            "team": self._cmd_team,
            "plugin": self._cmd_plugin,
            "create-agent": self._cmd_create_agent,
            "create-team": self._cmd_create_team,
            "worktree": self._cmd_worktree,
            "new": lambda ml, args: self.action_new_thread(),
            "clear": lambda ml, args: self.action_clear_view(),
            "quit": lambda ml, args: self.exit(),
        }

    def _cmd_help(self, ml, args: list[str]) -> None:
        ml.add_info("[bold]Available commands:[/bold]")
        for cmd in SLASH_COMMANDS:
            aliases_str = ""
            if cmd.aliases:
                aliases_str = f"  [dim](aliases: {', '.join('/' + a for a in cmd.aliases)})[/dim]"
            ml.add_info(f"  /{cmd.name:<14}\u2014 {cmd.description}{aliases_str}")
        ml.add_info("")
        ml.add_info("[bold]Keybindings:[/bold]")
        ml.add_info("  PgUp/PgDn     \u2014 Scroll output")
        ml.add_info("  Home/End      \u2014 Jump to top/bottom")
        ml.add_info("  Tab (in /..)  \u2014 Complete command")
        ml.add_info("  \u2191 / \u2193 (popup) \u2014 Navigate completions")
        ml.add_info("  Escape        \u2014 Dismiss popup / cancel action")
        ml.add_info("  Ctrl+L        \u2014 Clear screen")
        ml.add_info("  Ctrl+N        \u2014 New conversation thread")
        ml.add_info("  Ctrl+T        \u2014 Toggle light/dark theme")

    def _cmd_cost(self, ml, args: list[str]) -> None:
        if self._trace_sink:
            ml.add_info(
                f"Cost: ${self._trace_sink._cumulative_cost:.4f} | "
                f"Tokens: {self._trace_sink._cumulative_tokens:,} | "
                f"Iterations: {self._trace_sink._iteration}"
            )
        else:
            ml.add_info("No session active yet.")

    def _cmd_status(self, ml, args: list[str]) -> None:
        ml.add_info(
            f"Provider: {self.provider} | Model: {self.model} | "
            f"Budget: ${self.budget:.2f}"
        )
        ml.add_info(f"Workspace: {self.workspace}")
        if self._session_id:
            ml.add_info(f"Session: {self._session_id[:8]}")

    # ------------------------------------------------------------------
    # /agent handler
    # ------------------------------------------------------------------

    def _cmd_agent(self, ml, args: list[str]) -> None:
        """Handle /agent [list]."""
        from pathlib import Path

        from maike.agents.agent_commands import format_agent_list
        from maike.agents.agent_resolver import AgentResolver
        from maike.constants import AGENTS_PROJECT_SUBDIR, AGENTS_USER_DIR

        workspace = Path(self.workspace)
        resolver = AgentResolver(
            user_dir=AGENTS_USER_DIR,
            project_dir=workspace / AGENTS_PROJECT_SUBDIR,
        )
        for line in format_agent_list(resolver):
            ml.add_info(line)

    def _cmd_team(self, ml, args: list[str]) -> None:
        """Handle /team [list]."""
        from pathlib import Path

        from maike.agents.team_commands import format_team_list
        from maike.agents.team_resolver import TeamResolver
        from maike.constants import TEAMS_PROJECT_SUBDIR, TEAMS_USER_DIR

        workspace = Path(self.workspace)
        resolver = TeamResolver(
            user_dir=TEAMS_USER_DIR,
            project_dir=workspace / TEAMS_PROJECT_SUBDIR,
        )
        for line in format_team_list(resolver):
            ml.add_info(line)

    def _cmd_create_team(self, ml, args: list[str]) -> None:
        """Handle /create-team <name>."""
        from pathlib import Path

        from maike.agents.team_commands import (
            TeamWizardData,
            create_team_file,
            sanitize_team_name,
        )

        if not args:
            ml.add_info("[dim]Usage: /create-team <name>[/dim]")
            return

        slug = sanitize_team_name(args[0])
        workspace = Path(self.workspace)

        from maike.tui.screens.team_wizard_screen import TeamWizardScreen

        def _on_wizard_done(data: TeamWizardData | None) -> None:
            if data is None:
                ml.add_info("[dim]Team creation cancelled.[/dim]")
                return
            path = create_team_file(data, workspace=workspace)
            ml.add_info(f"[green]Created team: {path}[/green]")
            ml.add_info(f'  Use Team(name="{slug}", task="...") to invoke it.')

        self.push_screen(
            TeamWizardScreen(team_name=slug),
            callback=_on_wizard_done,
        )

    def _cmd_worktree(self, ml, args: list[str]) -> None:
        """Handle /worktree [list | add <name> | remove <name>]."""
        from pathlib import Path

        from maike.workflows.worktree import (
            WorktreeError,
            create_worktree,
            list_worktrees,
            remove_worktree,
        )

        workspace = Path(self.workspace)
        action = args[0] if args else "list"

        if action == "list":
            worktrees = list_worktrees(workspace)
            if not worktrees:
                ml.add_info("[dim]No mAIke-managed worktrees found.[/dim]")
                ml.add_info("[dim]  Create one with: /worktree add <name>[/dim]")
                return
            ml.add_info(f"[bold]Worktrees ({len(worktrees)}):[/bold]")
            for wt in worktrees:
                ml.add_info(f"  {wt.name}  branch={wt.branch}  head={wt.head}")
                ml.add_info(f"    [dim]{_rich_escape(str(wt.path))}[/dim]")
            return

        if action == "add":
            if len(args) < 2:
                ml.add_info("[dim]Usage: /worktree add <name> \\[--base <ref>][/dim]")
                return
            name = args[1]
            base = "HEAD"
            if "--base" in args:
                idx = args.index("--base")
                if idx + 1 < len(args):
                    base = args[idx + 1]
            try:
                path = create_worktree(workspace, name, base=base)
            except WorktreeError as exc:
                ml.add_info(f"[red]Error: {exc}[/red]")
                return
            ml.add_info(f"[green]Created worktree: {path}[/green]")
            ml.add_info("[bold]To work in it:[/bold]")
            ml.add_info("  /quit")
            ml.add_info(f"  cd {path}")
            ml.add_info("  maike chat")
            return

        if action == "remove":
            if len(args) < 2:
                ml.add_info("[dim]Usage: /worktree remove <name> \\[--delete-branch][/dim]")
                return
            name = args[1]
            delete_branch = "--delete-branch" in args
            try:
                remove_worktree(workspace, name, delete_branch=delete_branch)
            except WorktreeError as exc:
                ml.add_info(f"[red]Error: {exc}[/red]")
                return
            extra = " (branch deleted)" if delete_branch else ""
            ml.add_info(f"[green]Removed worktree: {name}{extra}[/green]")
            return

        ml.add_info("[dim]Usage: /worktree \\[list | add <name> | remove <name>][/dim]")

    def _cmd_create_agent(self, ml, args: list[str]) -> None:
        """Handle /create-agent <name> [--scope user|project]."""
        from pathlib import Path

        from maike.agents.agent_commands import (
            AgentWizardData,
            create_agent_file_v2,
            sanitize_agent_name,
        )

        if not args:
            ml.add_info("[dim]Usage: /create-agent <name> \\[--scope user|project][/dim]")
            return

        name = args[0]
        scope = "project"
        if "--scope" in args:
            idx = args.index("--scope")
            if idx + 1 < len(args) and args[idx + 1] in ("user", "project"):
                scope = args[idx + 1]

        slug = sanitize_agent_name(name)
        workspace = Path(self.workspace)

        from maike.tui.screens.agent_wizard_screen import AgentWizardScreen

        def _on_wizard_done(data: AgentWizardData | None) -> None:
            if data is None:
                ml.add_info("[dim]Agent creation cancelled.[/dim]")
                return
            path = create_agent_file_v2(data, workspace=workspace)
            ml.add_info(f"[green]Created agent: {path}[/green]")
            ml.add_info(f'  Use Delegate(agent="{slug}") to invoke it.')

        self.push_screen(
            AgentWizardScreen(
                agent_name=slug,
                scope=scope,
                provider=self.provider,
                model=self.model,
            ),
            callback=_on_wizard_done,
        )

    # ------------------------------------------------------------------
    # /skill and /plugin handlers
    # ------------------------------------------------------------------

    def _cmd_skill(self, ml, args: list[str]) -> None:
        """Handle /skill [list|load <name>|install <source>]."""
        subcmd = args[0] if args else "list"

        if subcmd == "list":
            self._cmd_skill_list(ml)
        elif subcmd == "load" and len(args) >= 2:
            self._cmd_skill_load(ml, args[1])
        elif subcmd == "install" and len(args) >= 2:
            self._cmd_skill_install(ml, args[1])
        else:
            ml.add_info("[dim]Usage: /skill \\[list | load <name> | install <source>][/dim]")

    def _cmd_skill_list(self, ml) -> None:
        from pathlib import Path

        from maike.agents.knowledge import _KNOWLEDGE_DIR
        from maike.agents.skill import SkillLoader
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR, SKILL_PROJECT_SUBDIR, SKILL_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.loader import PluginLoader

        workspace = Path(self.workspace)
        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_pdir = workspace / PLUGIN_PROJECT_SUBDIR
        if project_pdir.is_dir():
            search_dirs.append(project_pdir)

        manifests = PluginDiscovery.discover_enabled(search_dirs)
        plugin_skills = PluginLoader.load_all_plugin_skills(manifests) if manifests else []

        user_dir = SKILL_USER_DIR if SKILL_USER_DIR.is_dir() else None
        proj_dir = workspace / SKILL_PROJECT_SUBDIR
        proj_dir = proj_dir if proj_dir.is_dir() else None

        loader = SkillLoader(
            builtin_dir=_KNOWLEDGE_DIR,
            user_dir=user_dir,
            project_dir=proj_dir,
            extra_skills=plugin_skills,
        )
        all_skills = loader.load_all()

        if not all_skills:
            ml.add_info("[dim]No skills found.[/dim]")
            return

        ml.add_info(f"[bold]Skills ({len(all_skills)}):[/bold]")
        for s in all_skills:
            auto = " [yellow](auto)[/yellow]" if s.auto_inject else ""
            desc = f"  [dim]{_rich_escape(s.description[:80])}[/dim]" if s.description else ""
            ml.add_info(f"  [cyan]{_rich_escape(s.name)}[/cyan]  [dim]{_rich_escape(s.source.value)}[/dim]{auto}")
            if desc:
                ml.add_info(desc)

    def _cmd_skill_load(self, ml, name: str) -> None:
        from pathlib import Path

        from maike.agents.knowledge import _KNOWLEDGE_DIR
        from maike.agents.skill import SkillLoader
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR, SKILL_PROJECT_SUBDIR, SKILL_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.loader import PluginLoader

        workspace = Path(self.workspace)
        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_pdir = workspace / PLUGIN_PROJECT_SUBDIR
        if project_pdir.is_dir():
            search_dirs.append(project_pdir)

        manifests = PluginDiscovery.discover_enabled(search_dirs)
        plugin_skills = PluginLoader.load_all_plugin_skills(manifests) if manifests else []
        user_dir = SKILL_USER_DIR if SKILL_USER_DIR.is_dir() else None
        proj_dir = workspace / SKILL_PROJECT_SUBDIR
        proj_dir = proj_dir if proj_dir.is_dir() else None

        loader = SkillLoader(
            builtin_dir=_KNOWLEDGE_DIR,
            user_dir=user_dir,
            project_dir=proj_dir,
            extra_skills=plugin_skills,
        )
        skill = loader.load_by_name(name)
        if skill is None:
            ml.add_error(f"Skill not found: {name}")
            return

        ml.add_info(f"[bold]{_rich_escape(skill.name)}[/bold] [dim]({_rich_escape(skill.source.value)})[/dim]")
        if skill.description:
            ml.add_info(f"  {skill.description[:120]}")
        # Show first 15 lines of content
        lines = skill.content.strip().splitlines()[:15]
        for line in lines:
            ml.add_info(f"  [dim]{_rich_escape(line)}[/dim]")
        if len(skill.content.strip().splitlines()) > 15:
            ml.add_info("  [dim]...(truncated)[/dim]")

    def _cmd_skill_install(self, ml, source: str) -> None:
        from maike.plugins.installer import PluginInstallError, install_skill

        try:
            names = install_skill(source)
            if len(names) == 1:
                ml.add_info(f"[green]Installed skill: {names[0]}[/green]")
            else:
                ml.add_info(f"[green]Installed {len(names)} skills: {', '.join(names)}[/green]")
        except PluginInstallError as exc:
            ml.add_error(f"Install failed: {exc}")

    def _cmd_plugin(self, ml, args: list[str]) -> None:
        """Handle /plugin [list|install <source>|uninstall <name>|enable <name>|disable <name>]."""
        subcmd = args[0] if args else "list"

        if subcmd == "list":
            self._cmd_plugin_list(ml)
        elif subcmd == "install" and len(args) >= 2:
            self._cmd_plugin_install(ml, args[1])
        elif subcmd == "uninstall" and len(args) >= 2:
            self._cmd_plugin_uninstall(ml, args[1])
        elif subcmd == "enable" and len(args) >= 2:
            self._cmd_plugin_enable(ml, args[1])
        elif subcmd == "disable" and len(args) >= 2:
            self._cmd_plugin_disable(ml, args[1])
        else:
            ml.add_info("[dim]Usage: /plugin \\[list | install <source> | uninstall <name> | enable <name> | disable <name>][/dim]")

    def _cmd_plugin_list(self, ml) -> None:
        from pathlib import Path

        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.installer import describe_plugin
        from maike.plugins.settings import load_settings

        workspace = Path(self.workspace)
        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_dir = workspace / PLUGIN_PROJECT_SUBDIR
        if project_dir.is_dir():
            search_dirs.append(project_dir)

        manifests = PluginDiscovery.discover(search_dirs) if search_dirs else []
        settings = load_settings()

        if not manifests:
            ml.add_info("[dim]No plugins found.[/dim]")
            return

        ml.add_info(f"[bold]Plugins ({len(manifests)}):[/bold]")
        for m in manifests:
            if m.name in settings.disabled:
                state = " [red](disabled)[/red]"
            else:
                state = " [green](enabled)[/green]"
            ml.add_info(f"  [cyan]{m.name}[/cyan] (v{m.version}){state}")
            if m.description:
                ml.add_info(f"    [dim]{_rich_escape(m.description[:100])}[/dim]")
            summary = describe_plugin(m)
            if summary and "no components" not in summary:
                for line in summary.strip().splitlines():
                    ml.add_info(f"    [dim]{_rich_escape(line.strip())}[/dim]")

    def _cmd_plugin_install(self, ml, source: str) -> None:
        from maike.plugins.installer import PluginInstallError, describe_plugin, install_plugin

        try:
            manifest = install_plugin(source)
            ml.add_info(f"[green]Installed: {manifest.name} v{manifest.version}[/green]")
            summary = describe_plugin(manifest)
            if summary:
                ml.add_info(f"  [dim]{_rich_escape(summary)}[/dim]")
        except PluginInstallError as exc:
            ml.add_error(f"Install failed: {exc}")

    def _cmd_plugin_uninstall(self, ml, name: str) -> None:
        from maike.plugins.installer import PluginInstallError, uninstall_plugin

        try:
            uninstall_plugin(name)
            ml.add_info(f"[green]Uninstalled: {name}[/green]")
        except PluginInstallError as exc:
            ml.add_error(f"Uninstall failed: {exc}")

    def _cmd_plugin_enable(self, ml, name: str) -> None:
        from maike.plugins.settings import load_settings, save_settings

        settings = load_settings()
        if name not in settings.disabled:
            ml.add_info(f"Plugin '{name}' is already enabled.")
            return
        settings.disabled.discard(name)
        save_settings(settings)
        ml.add_info(f"[green]Enabled: {name}[/green]")

    def _cmd_plugin_disable(self, ml, name: str) -> None:
        from maike.plugins.settings import load_settings, save_settings

        settings = load_settings()
        settings.disabled.add(name)
        save_settings(settings)
        ml.add_info(f"[yellow]Disabled: {name} (takes effect on next session)[/yellow]")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_orchestrator(self):
        from maike.cli import _load_all_dotenv
        from maike.observability.tracer import Tracer
        from maike.orchestrator.orchestrator import Orchestrator
        from maike.tui.bridge import TUIStreamSink, TUITraceSink

        _load_all_dotenv()
        trace_sink = TUITraceSink(self)
        stream_sink = TUIStreamSink(self, trace_sink)
        orchestrator = Orchestrator(
            base_path=self.workspace,
            tracer=Tracer(sink=trace_sink),
            stream_sink=stream_sink,
        )
        return orchestrator, trace_sink, stream_sink

    def _disable_input(self) -> None:
        self.query_one(PromptInput).disabled = True

    def _enable_input(self, mode: str = "task") -> None:
        try:
            prompt = self.query_one(PromptInput)
            prompt.disabled = False
            prompt.set_mode(mode)
            prompt.focus()
        except Exception:
            pass  # Never let input-enable failures freeze the TUI

    @staticmethod
    def _extract_output(result: Any) -> str:
        if result is None:
            return ""
        for stage_agents in getattr(result, "stage_results", {}).values():
            for agent_result in reversed(stage_agents):
                output = getattr(agent_result, "output", None)
                if output:
                    return output
        return ""
