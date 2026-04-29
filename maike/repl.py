"""Interactive REPL for multi-turn coding conversations."""

from __future__ import annotations

import sys
from pathlib import Path

from maike.constants import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_RUN_BUDGET_USD


# ── ANSI constants (matching cli.py palette) ────────────────────────────────

_DIM = "\033[90m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/cost": "Show session cost breakdown",
    "/history": "Show conversation history for current thread",
    "/new": "Start a new thread (fresh conversation)",
    "/budget": "Show remaining budget",
    "/plugin list": "List plugins with enabled/disabled state",
    "/plugin enable <name>": "Enable a disabled plugin",
    "/plugin disable <name>": "Disable a plugin",
    "/plugin install <source>": "Install from local dir or git URL",
    "/skill list": "List all available skills",
    "/skill load <name>": "Show skill content",
    "/agent list": "List available plugin agents",
    "/hook list": "List registered hooks",
    "/mcp list": "List MCP servers and tools",
    "/quit": "Exit the REPL",
    "/exit": "Exit the REPL",
}


def parse_input(raw: str) -> str:
    """Join backslash-continued lines into a single string.

    Lines ending with ``\\`` are merged with the following line.  The
    trailing backslash and the newline are replaced with a single space.
    """
    lines = raw.split("\n")
    merged: list[str] = []
    buf = ""
    for line in lines:
        if line.endswith("\\"):
            buf += line[:-1].rstrip() + " "
        else:
            buf += line
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return "\n".join(merged).strip()


def classify_input(text: str) -> tuple[str, str]:
    """Return ``(kind, value)`` for the given input line.

    ``kind`` is one of ``"empty"``, ``"slash"``, or ``"task"``.
    For slash commands ``value`` is the full stripped text (e.g.
    ``"/plugin list"``).  For tasks ``value`` is the full task text.
    """
    stripped = text.strip()
    if not stripped:
        return ("empty", "")
    if stripped.startswith("/"):
        return ("slash", stripped)
    return ("task", stripped)


class REPLSession:
    """Interactive REPL for multi-turn coding conversations."""

    def __init__(
        self,
        workspace: Path,
        provider: str | None = None,
        model: str | None = None,
        budget: float = DEFAULT_RUN_BUDGET_USD,
        verbose: bool = False,
        yes: bool = False,
        advisor_enabled: bool = False,
        advisor_provider: str | None = None,
        advisor_model: str | None = None,
        advisor_budget_pct: float = 0.20,
    ):
        self.workspace = workspace.resolve()
        self.provider = provider or DEFAULT_PROVIDER
        self.model = model or DEFAULT_MODEL
        self.budget = budget
        self.verbose = verbose
        self.yes = yes
        self.advisor_enabled = advisor_enabled
        self.advisor_provider = advisor_provider
        self.advisor_model = advisor_model
        self.advisor_budget_pct = advisor_budget_pct
        self.thread_id: str | None = None
        self.total_cost: float = 0.0
        self._force_new_thread: bool = False
        self._task_registry = None  # Set after first orchestrator run
        self._last_run_metadata: dict = {}  # Telemetry from last agent run

        # Initialize command registry.
        from maike.commands.registry import CommandRegistry
        from maike.commands.builtins import register_builtins
        self._command_registry = CommandRegistry()
        register_builtins(self._command_registry)

    # ── Prompt string ───────────────────────────────────────────────────

    def _prompt_str(self) -> str:
        return f"{_CYAN}maike>{_RESET} "

    # ── Built-in slash-command handlers ─────────────────────────────────

    def _handle_help(self) -> None:
        sys.stderr.write(f"\n{_BOLD}Available commands:{_RESET}\n")
        for cmd, desc in SLASH_COMMANDS.items():
            sys.stderr.write(f"  {_CYAN}{cmd:30}{_RESET}  {desc}\n")
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _handle_cost(self) -> None:
        budget_str = f"${self.budget:.2f}" if self.budget else "unlimited"
        self._write(
            f"\n{_BOLD}Session cost:{_RESET} {_GREEN}${self.total_cost:.4f}{_RESET}"
            f"  /  budget {budget_str}\n\n"
        )

    def _handle_budget(self) -> None:
        if not self.budget:
            self._write(f"\n{_BOLD}Budget:{_RESET} {_GREEN}unlimited{_RESET}  (used ${self.total_cost:.4f})\n\n")
            return
        remaining = self.budget - self.total_cost
        colour = _GREEN if remaining > self.budget * 0.2 else _YELLOW
        self._write(
            f"\n{_BOLD}Budget remaining:{_RESET} {colour}${remaining:.4f}{_RESET}"
            f"  (used ${self.total_cost:.4f} of ${self.budget:.2f})\n\n"
        )

    def _handle_history(self) -> None:
        if self.thread_id is None:
            self._write(f"\n{_DIM}No active thread yet. Start a task to begin.{_RESET}\n\n")
            return
        short = self.thread_id[:8]
        self._write(
            f"\n{_BOLD}Thread:{_RESET} {short}\n"
            f"{_DIM}  Full ID: {self.thread_id}{_RESET}\n"
            f"{_DIM}  Run /new to start a fresh conversation{_RESET}\n\n"
        )

    def _handle_new(self) -> None:
        self.thread_id = None
        self._force_new_thread = True
        sys.stderr.write(f"\n{_GREEN}Started a new conversation thread.{_RESET}\n\n")
        sys.stderr.flush()

    # ── Plugin slash commands ──────────────────────────────────────────

    def _handle_plugin(self, args: list[str]) -> None:
        subcmd = args[0] if args else "list"

        if subcmd == "list":
            self._plugin_list()
        elif subcmd == "enable" and len(args) >= 2:
            self._plugin_enable(args[1])
        elif subcmd == "disable" and len(args) >= 2:
            self._plugin_disable(args[1])
        elif subcmd == "install" and len(args) >= 2:
            self._plugin_install(args[1])
        elif subcmd == "uninstall" and len(args) >= 2:
            self._plugin_uninstall(args[1])
        else:
            self._write(f"{_YELLOW}Usage: /plugin list|enable|disable|install|uninstall{_RESET}\n")

    def _plugin_list(self) -> None:
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.settings import load_settings

        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_dir = self.workspace / PLUGIN_PROJECT_SUBDIR
        if project_dir.is_dir():
            search_dirs.append(project_dir)

        manifests = PluginDiscovery.discover(search_dirs) if search_dirs else []
        settings = load_settings()

        if not manifests:
            self._write(f"{_DIM}No plugins found.{_RESET}\n")
            return

        self._write(f"\n{_BOLD}Plugins:{_RESET}\n")
        for m in manifests:
            state = f" {_RED}[disabled]{_RESET}" if m.name in settings.disabled else f" {_GREEN}[enabled]{_RESET}"
            self._write(f"  {_CYAN}{m.name}{_RESET} (v{m.version}){state} — {m.path}\n")
            if m.description:
                self._write(f"    {_DIM}{m.description}{_RESET}\n")
        self._write("\n")

    def _plugin_enable(self, name: str) -> None:
        from maike.plugins.settings import load_settings, save_settings

        settings = load_settings()
        if name not in settings.disabled:
            self._write(f"Plugin '{name}' is already enabled.\n")
            return
        settings.disabled.discard(name)
        save_settings(settings)
        self._write(f"{_GREEN}Enabled plugin: {name}{_RESET}\n")

    def _plugin_disable(self, name: str) -> None:
        from maike.plugins.settings import load_settings, save_settings

        settings = load_settings()
        settings.disabled.add(name)
        save_settings(settings)
        self._write(f"{_YELLOW}Disabled plugin: {name} (takes effect on next session){_RESET}\n")

    def _plugin_install(self, source: str) -> None:
        from maike.plugins.installer import PluginInstallError, describe_plugin, install_plugin

        try:
            manifest = install_plugin(source)
            self._write(f"{_GREEN}Installed: {manifest.name} v{manifest.version}{_RESET}\n")
            summary = describe_plugin(manifest)
            if summary:
                self._write(f"{summary}\n")
        except PluginInstallError as exc:
            self._write(f"{_RED}Install failed: {exc}{_RESET}\n")

    def _plugin_uninstall(self, name: str) -> None:
        from maike.plugins.installer import PluginInstallError, uninstall_plugin

        try:
            uninstall_plugin(name)
            self._write(f"{_GREEN}Uninstalled: {name}{_RESET}\n")
        except PluginInstallError as exc:
            self._write(f"{_RED}Uninstall failed: {exc}{_RESET}\n")

    # ── Skill slash commands ───────────────────────────────────────────

    def _handle_skill(self, args: list[str]) -> None:
        subcmd = args[0] if args else "list"

        if subcmd == "list":
            self._skill_list()
        elif subcmd == "load" and len(args) >= 2:
            self._skill_load(args[1])
        elif subcmd == "install" and len(args) >= 2:
            self._skill_install(args[1])
        else:
            self._write(f"{_YELLOW}Usage: /skill list|load <name>|install <source>{_RESET}\n")

    def _skill_install(self, source: str) -> None:
        from maike.plugins.installer import PluginInstallError, install_skill

        try:
            names = install_skill(source)
            if len(names) == 1:
                self._write(f"{_GREEN}Installed skill: {names[0]}{_RESET}\n")
            else:
                self._write(f"{_GREEN}Installed {len(names)} skills: {', '.join(names)}{_RESET}\n")
        except PluginInstallError as exc:
            self._write(f"{_RED}Install failed: {exc}{_RESET}\n")

    def _skill_list(self) -> None:
        from maike.agents.knowledge import _KNOWLEDGE_DIR
        from maike.agents.skill import SkillLoader
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR, SKILL_PROJECT_SUBDIR, SKILL_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.loader import PluginLoader

        # Load from all sources
        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_dir = self.workspace / PLUGIN_PROJECT_SUBDIR
        if project_dir.is_dir():
            search_dirs.append(project_dir)

        manifests = PluginDiscovery.discover_enabled(search_dirs)
        plugin_skills = PluginLoader.load_all_plugin_skills(manifests) if manifests else []

        user_dir = SKILL_USER_DIR if SKILL_USER_DIR.is_dir() else None
        proj_skill_dir = self.workspace / SKILL_PROJECT_SUBDIR
        proj_skill_dir = proj_skill_dir if proj_skill_dir.is_dir() else None

        loader = SkillLoader(
            builtin_dir=_KNOWLEDGE_DIR,
            user_dir=user_dir,
            project_dir=proj_skill_dir,
            extra_skills=plugin_skills,
        )
        all_skills = loader.load_all()

        if not all_skills:
            self._write(f"{_DIM}No skills found.{_RESET}\n")
            return

        self._write(f"\n{_BOLD}Skills ({len(all_skills)}):{_RESET}\n")
        for s in all_skills:
            auto = " [auto]" if s.auto_inject else ""
            self._write(f"  {_CYAN}{s.name}{_RESET}  {_DIM}{s.source.value}{auto}{_RESET}\n")
            if s.description:
                self._write(f"    {s.description}\n")
        self._write("\n")

    def _skill_load(self, name: str) -> None:
        from maike.agents.knowledge import _KNOWLEDGE_DIR
        from maike.agents.skill import SkillLoader
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR, SKILL_PROJECT_SUBDIR, SKILL_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.loader import PluginLoader

        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_dir = self.workspace / PLUGIN_PROJECT_SUBDIR
        if project_dir.is_dir():
            search_dirs.append(project_dir)

        manifests = PluginDiscovery.discover_enabled(search_dirs)
        plugin_skills = PluginLoader.load_all_plugin_skills(manifests) if manifests else []
        user_dir = SKILL_USER_DIR if SKILL_USER_DIR.is_dir() else None
        proj_skill_dir = self.workspace / SKILL_PROJECT_SUBDIR
        proj_skill_dir = proj_skill_dir if proj_skill_dir.is_dir() else None

        loader = SkillLoader(
            builtin_dir=_KNOWLEDGE_DIR,
            user_dir=user_dir,
            project_dir=proj_skill_dir,
            extra_skills=plugin_skills,
        )
        all_skills = loader.load_all()

        match = None
        for s in all_skills:
            if s.name == name or s.name.endswith(f":{name}"):
                match = s
                break
        if match is None:
            self._write(f"{_RED}Skill '{name}' not found.{_RESET}\n")
            return

        self._write(f"\n{_BOLD}Skill: {match.name}{_RESET}\n")
        self._write(f"{_DIM}{match.description}{_RESET}\n\n")
        # Show first 30 lines of content
        lines = match.content.splitlines()
        for line in lines[:30]:
            self._write(f"  {line}\n")
        if len(lines) > 30:
            self._write(f"  {_DIM}... ({len(lines) - 30} more lines){_RESET}\n")
        self._write("\n")

    # ── Agent slash commands ───────────────────────────────────────────

    def _handle_agent(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            self._agent_list()
        else:
            self._write(f"{_YELLOW}Usage: /agent list{_RESET}\n")

    def _agent_list(self) -> None:
        from maike.agents.agent_commands import format_agent_list

        resolver = self._build_agent_resolver()
        self._write("\n")
        for line in format_agent_list(resolver):
            self._write(f"{line}\n")
        self._write("\n")

    async def _handle_create_agent(self, args: list[str]) -> None:
        from maike.agents.agent_commands import (
            ALL_TOOL_NAMES,
            AgentWizardData,
            create_agent_file_v2,
            preview_agent_markdown,
            sanitize_agent_name,
        )

        if not args:
            self._write(f"{_YELLOW}Usage: /create-agent <name> [--scope user|project]{_RESET}\n")
            return

        name = args[0]
        scope = "project"
        if "--scope" in args:
            idx = args.index("--scope")
            if idx + 1 < len(args) and args[idx + 1] in ("user", "project"):
                scope = args[idx + 1]

        slug = sanitize_agent_name(name)
        scope_label = ".maike/agents/" if scope == "project" else "~/.config/maike/agents/"
        self._write(f"\n{_BOLD}Creating agent: {_CYAN}{slug}{_RESET}\n")
        self._write(f"{_DIM}  Scope: {scope} ({scope_label}){_RESET}\n\n")

        try:
            # 1. Description
            self._write(f"{_BOLD}1. Description{_RESET} {_DIM}— when should mAIke auto-delegate to this agent?{_RESET}\n")
            description = input("   > ").strip()
            if not description:
                description = f"A custom {slug} agent"

            # 2. Model tier
            self._write(f"\n{_BOLD}2. Model tier{_RESET} {_DIM}— cheap (fast/$), default (balanced), strong (max capability){_RESET}\n")
            model_tier = input("   [default]: ").strip() or "default"
            if model_tier not in ("cheap", "default", "strong"):
                model_tier = "default"

            # 3. Max turns
            self._write(f"\n{_BOLD}3. Max turns{_RESET} {_DIM}— iteration budget (10=focused, 30=typical, 50+=complex){_RESET}\n")
            max_turns_str = input("   [30]: ").strip() or "30"
            try:
                max_turns = int(max_turns_str)
            except ValueError:
                max_turns = 30

            # 4. System prompt
            self._write(f"\n{_BOLD}4. System prompt{_RESET} {_DIM}— defines the agent's behavior and expertise{_RESET}\n")
            self._write(f"{_DIM}   (Multi-line: type your prompt, blank line to finish. Leave empty for placeholder.){_RESET}\n")

            # Offer LLM generation.
            gen_choice = input("   Generate from description? [y/N]: ").strip().lower()
            system_prompt = ""
            if gen_choice in ("y", "yes"):
                self._write(f"{_DIM}   Generating...{_RESET}\n")
                try:
                    from maike.agents.agent_commands import generate_agent_prompt
                    from maike.constants import cheap_model_for_provider
                    cheap_model = cheap_model_for_provider(self.provider)
                    system_prompt = await generate_agent_prompt(
                        description=description,
                        name=slug,
                        tools=None,
                        provider=self.provider,
                        model=cheap_model,
                    )
                    if system_prompt:
                        self._write(f"{_GREEN}   Generated.{_RESET} You can edit the file later to refine.\n")
                    else:
                        self._write(f"{_YELLOW}   Generation failed — write your prompt manually.{_RESET}\n")
                except Exception:
                    self._write(f"{_YELLOW}   Generation failed — write your prompt manually.{_RESET}\n")

            if not system_prompt:
                prompt_lines: list[str] = []
                while True:
                    line = input("   > ")
                    if line == "":
                        break
                    prompt_lines.append(line)
                system_prompt = "\n".join(prompt_lines)

            # 5. Tools
            tools_display = ", ".join(ALL_TOOL_NAMES)
            self._write(f"\n{_BOLD}5. Tools{_RESET} {_DIM}— which tools can this agent use?{_RESET}\n")
            self._write(f"{_DIM}   Available: {tools_display}{_RESET}\n")
            self._write(f"{_DIM}   Enter: 'all', comma-separated list, or '-Write,-Delegate' to exclude{_RESET}\n")
            tools_input = input("   [all]: ").strip() or "all"

            allowed_tools: list[str] | None = None
            disallowed_tools: list[str] | None = None
            if tools_input.lower() != "all":
                if tools_input.startswith("-"):
                    # Exclusion mode: "-Write,-Delegate"
                    excluded = [t.strip().lstrip("-") for t in tools_input.split(",") if t.strip()]
                    disallowed_tools = [t for t in excluded if t in ALL_TOOL_NAMES]
                else:
                    # Inclusion mode: "Read, Grep, Bash"
                    selected = [t.strip() for t in tools_input.split(",") if t.strip()]
                    allowed_tools = [t for t in selected if t in ALL_TOOL_NAMES]

            # 6. Skills
            self._write(f"\n{_BOLD}6. Skills{_RESET} {_DIM}— auto-injected knowledge modules (comma-separated, or empty){_RESET}\n")
            skills_input = input("   []: ").strip()
            skills = [s.strip() for s in skills_input.split(",") if s.strip()] if skills_input else []

            # 7. Initial prompt
            self._write(f"\n{_BOLD}7. Initial prompt{_RESET} {_DIM}— prepended to every task (e.g. \"Read ARCHITECTURE.md first\"){_RESET}\n")
            initial_prompt = input("   []: ").strip() or None

            # 8. Critical reminder
            self._write(f"\n{_BOLD}8. Critical reminder{_RESET} {_DIM}— constraint injected every LLM turn (e.g. \"Never modify prod data\"){_RESET}\n")
            critical_reminder = input("   []: ").strip() or None

            # 9. Background
            self._write(f"\n{_BOLD}9. Background{_RESET} {_DIM}— run this agent async by default?{_RESET}\n")
            bg_input = input("   [no]: ").strip().lower()
            background = bg_input in ("y", "yes", "true")

        except (EOFError, KeyboardInterrupt):
            self._write(f"\n{_DIM}Cancelled.{_RESET}\n")
            return

        # Build wizard data.
        data = AgentWizardData(
            name=slug,
            description=description,
            system_prompt=system_prompt,
            model_tier=model_tier,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            skills=skills,
            initial_prompt=initial_prompt,
            background=background,
            critical_reminder=critical_reminder,
            scope=scope,
        )

        # Preview.
        self._write(f"\n{_BOLD}Preview:{_RESET}\n")
        self._write(f"{_DIM}{'─' * 50}{_RESET}\n")
        for line in preview_agent_markdown(data).splitlines():
            self._write(f"  {line}\n")
        self._write(f"{_DIM}{'─' * 50}{_RESET}\n")

        # Confirm.
        confirm = input("\n  Create agent? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            self._write(f"{_DIM}Cancelled.{_RESET}\n")
            return

        path = create_agent_file_v2(data, workspace=self.workspace)
        self._write(f"\n{_GREEN}Created agent: {path}{_RESET}\n")
        self._write(f"  Use Delegate(agent=\"{slug}\") to invoke it.\n\n")

    def _build_agent_resolver(self):
        """Construct an AgentResolver from the current workspace."""
        from maike.agents.agent_resolver import AgentResolver
        from maike.constants import AGENTS_PROJECT_SUBDIR, AGENTS_USER_DIR

        return AgentResolver(
            user_dir=AGENTS_USER_DIR,
            project_dir=self.workspace / AGENTS_PROJECT_SUBDIR,
        )

    # ── Team slash commands ─────────────────────────────────────────────

    def _handle_team(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            self._team_list()
        else:
            self._write(f"{_YELLOW}Usage: /team list{_RESET}\n")

    def _handle_advisor(self, args: list[str]) -> None:
        """`/advisor` (status) or `/advisor ask <question>` (manual call)."""
        from maike.constants import (
            ADVISOR_BUDGET_FRACTION_DEFAULT,
            ADVISOR_MAX_CALLS_PER_SESSION,
        )

        if not args or args[0] == "status":
            if not self.advisor_enabled:
                self._write(
                    f"\n{_DIM}Advisor is not enabled. Restart with --advisor "
                    f"to turn it on.{_RESET}\n\n"
                )
                return
            provider = self.advisor_provider or "(same as executor)"
            model = self.advisor_model or "(strong-tier default)"
            pct = self.advisor_budget_pct or ADVISOR_BUDGET_FRACTION_DEFAULT
            self._write(
                f"\n{_BOLD}Advisor enabled{_RESET}\n"
                f"  Provider: {_CYAN}{provider}{_RESET}\n"
                f"  Model: {_CYAN}{model}{_RESET}\n"
                f"  Budget: {int(pct * 100)}% of session "
                f"(max {ADVISOR_MAX_CALLS_PER_SESSION} calls)\n\n"
            )
            return
        if args[0] == "ask":
            self._write(
                f"\n{_DIM}Manual /advisor ask is not yet wired in the REPL — "
                f"prompt the agent with: 'Use the Advisor tool to ...' instead.{_RESET}\n\n"
            )
            return
        self._write(f"{_YELLOW}Usage: /advisor [status|ask <question>]{_RESET}\n")

    def _team_list(self) -> None:
        from maike.agents.team_commands import format_team_list
        from maike.agents.team_resolver import TeamResolver
        from maike.constants import TEAMS_PROJECT_SUBDIR, TEAMS_USER_DIR

        resolver = TeamResolver(
            user_dir=TEAMS_USER_DIR,
            project_dir=self.workspace / TEAMS_PROJECT_SUBDIR,
        )
        self._write("\n")
        for line in format_team_list(resolver):
            self._write(f"{line}\n")
        self._write("\n")

    def _handle_create_team(self, args: list[str]) -> None:
        from maike.agents.team_commands import (
            TeamWizardData,
            TeamWizardMemberData,
            create_team_file,
            preview_team_markdown,
            sanitize_team_name,
        )

        if not args:
            self._write(f"{_YELLOW}Usage: /create-team <name>{_RESET}\n")
            return

        slug = sanitize_team_name(args[0])
        self._write(f"\n{_BOLD}Creating team: {_CYAN}{slug}{_RESET}\n\n")

        try:
            description = input("  Description: ").strip()
            if not description:
                description = f"A custom team: {slug}"

            process_type = input("  Process (parallel/sequential) [parallel]: ").strip() or "parallel"
            if process_type not in ("parallel", "sequential"):
                process_type = "parallel"

            on_failure = input("  On failure (continue/abort/retry) [continue]: ").strip() or "continue"
            if on_failure not in ("continue", "abort", "retry"):
                on_failure = "continue"

            # Collect members.
            members: list[TeamWizardMemberData] = []
            self._write(f"\n{_BOLD}Add members{_RESET} {_DIM}(blank agent name to finish):{_RESET}\n")
            while True:
                agent = input(f"  Member {len(members) + 1} agent name (or blank to finish): ").strip()
                if not agent:
                    if not members:
                        self._write(f"{_YELLOW}  At least one member is required.{_RESET}\n")
                        continue
                    break
                role = input("  Role description: ").strip()
                members.append(TeamWizardMemberData(agent=agent, role=role or None))

            synthesis = input("\n  Synthesis prompt (blank for default): ").strip()

        except (EOFError, KeyboardInterrupt):
            self._write(f"\n{_DIM}Cancelled.{_RESET}\n")
            return

        data = TeamWizardData(
            name=slug,
            description=description,
            process_type=process_type,
            on_failure=on_failure,
            members=members,
            synthesis_prompt=synthesis,
        )

        # Preview.
        self._write(f"\n{_BOLD}Preview:{_RESET}\n")
        self._write(f"{_DIM}{'─' * 50}{_RESET}\n")
        for line in preview_team_markdown(data).splitlines():
            self._write(f"  {line}\n")
        self._write(f"{_DIM}{'─' * 50}{_RESET}\n")

        confirm = input("\n  Create team? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            self._write(f"{_DIM}Cancelled.{_RESET}\n")
            return

        path = create_team_file(data, workspace=self.workspace)
        self._write(f"\n{_GREEN}Created team: {path}{_RESET}\n")
        self._write(f"  Use Team(name=\"{slug}\", task=\"...\") to invoke it.\n\n")

    # ── Worktree slash commands ─────────────────────────────────────────

    def _handle_worktree(self, args: list[str]) -> None:
        from maike.workflows.worktree import (
            WorktreeError,
            create_worktree,
            list_worktrees,
            remove_worktree,
        )

        action = args[0] if args else "list"

        if action == "list":
            worktrees = list_worktrees(self.workspace)
            self._write("\n")
            if not worktrees:
                self._write(f"{_DIM}No mAIke-managed worktrees found.{_RESET}\n")
                self._write(f"{_DIM}  Create one with: /worktree add <name>{_RESET}\n\n")
                return
            self._write(f"{_BOLD}Worktrees ({len(worktrees)}):{_RESET}\n")
            for wt in worktrees:
                self._write(f"  {_CYAN}{wt.name}{_RESET}  branch={wt.branch}  head={wt.head}\n")
                self._write(f"    {_DIM}{wt.path}{_RESET}\n")
            self._write("\n")
            return

        if action == "add":
            if len(args) < 2:
                self._write(f"{_YELLOW}Usage: /worktree add <name> [--base <ref>]{_RESET}\n")
                return
            name = args[1]
            base = "HEAD"
            if "--base" in args:
                idx = args.index("--base")
                if idx + 1 < len(args):
                    base = args[idx + 1]
            try:
                path = create_worktree(self.workspace, name, base=base)
            except WorktreeError as exc:
                self._write(f"{_RED}Error: {exc}{_RESET}\n")
                return
            self._write(f"\n{_GREEN}Created worktree: {path}{_RESET}\n")
            self._write(f"\n{_BOLD}To work in it:{_RESET}\n")
            self._write("  /quit\n")
            self._write(f"  cd {path}\n")
            self._write("  maike chat\n\n")
            return

        if action == "remove":
            if len(args) < 2:
                self._write(f"{_YELLOW}Usage: /worktree remove <name> [--delete-branch]{_RESET}\n")
                return
            name = args[1]
            delete_branch = "--delete-branch" in args
            try:
                remove_worktree(self.workspace, name, delete_branch=delete_branch)
            except WorktreeError as exc:
                self._write(f"{_RED}Error: {exc}{_RESET}\n")
                return
            extra = " (branch deleted)" if delete_branch else ""
            self._write(f"\n{_GREEN}Removed worktree: {name}{extra}{_RESET}\n\n")
            return

        self._write(f"{_YELLOW}Usage: /worktree [list | add <name> | remove <name>]{_RESET}\n")

    # ── Hook slash commands ────────────────────────────────────────────

    def _handle_hook(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            self._hook_list()
        else:
            self._write(f"{_YELLOW}Usage: /hook list{_RESET}\n")

    def _hook_list(self) -> None:
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.hooks import load_hook_configs

        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_dir = self.workspace / PLUGIN_PROJECT_SUBDIR
        if project_dir.is_dir():
            search_dirs.append(project_dir)

        manifests = PluginDiscovery.discover_enabled(search_dirs)
        hook_config = load_hook_configs(manifests)

        total = sum(len(defs) for defs in hook_config.hooks.values())
        if total == 0:
            self._write(f"{_DIM}No hooks registered.{_RESET}\n")
            return

        self._write(f"\n{_BOLD}Hooks ({total}):{_RESET}\n")
        for event, defs in sorted(hook_config.hooks.items(), key=lambda x: x[0].value):
            for d in defs:
                matcher = f" matcher={d.matcher}" if d.matcher else ""
                plugin = f" ({d.source_plugin})" if d.source_plugin else ""
                self._write(f"  {_CYAN}{event.value}{_RESET}{matcher}{plugin}\n")
                self._write(f"    {_DIM}{d.command}{_RESET}\n")
        self._write("\n")

    # ── MCP slash commands ─────────────────────────────────────────────

    def _handle_mcp(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            self._mcp_list()
        else:
            self._write(f"{_YELLOW}Usage: /mcp list{_RESET}\n")

    def _mcp_list(self) -> None:
        from maike.constants import PLUGIN_PROJECT_SUBDIR, PLUGIN_USER_DIR
        from maike.plugins.discovery import PluginDiscovery
        from maike.plugins.mcp_config import load_mcp_configs

        search_dirs: list[Path] = []
        if PLUGIN_USER_DIR.is_dir():
            search_dirs.append(PLUGIN_USER_DIR)
        project_dir = self.workspace / PLUGIN_PROJECT_SUBDIR
        if project_dir.is_dir():
            search_dirs.append(project_dir)

        manifests = PluginDiscovery.discover_enabled(search_dirs)
        configs = load_mcp_configs(self.workspace, manifests)

        if not configs:
            self._write(f"{_DIM}No MCP servers configured.{_RESET}\n")
            return

        self._write(f"\n{_BOLD}MCP Servers ({len(configs)}):{_RESET}\n")
        for c in configs:
            args_str = " ".join(c.args[:3])
            if len(c.args) > 3:
                args_str += " ..."
            self._write(f"  {_CYAN}{c.name}{_RESET}  ({c.source})\n")
            self._write(f"    {_DIM}{c.command} {args_str}{_RESET}\n")
        self._write("\n")

    # ── Main loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main REPL loop."""
        self._print_welcome()

        while True:
            try:
                raw = input(self._prompt_str())
            except EOFError:
                # Ctrl+D — exit
                sys.stderr.write("\n")
                sys.stderr.flush()
                break
            except KeyboardInterrupt:
                # Ctrl+C — cancel current input, stay in REPL
                sys.stderr.write("\n")
                sys.stderr.flush()
                continue

            text = parse_input(raw)
            kind, value = classify_input(text)

            if kind == "empty":
                continue

            # Bare quit/exit/q without slash prefix
            if kind == "task" and value.lower() in ("quit", "exit", "q"):
                break

            if kind == "slash":
                try:
                    if not await self._command_registry.async_dispatch(value, self):
                        self._write(
                            f"{_YELLOW}Unknown command: {value.split()[0]}  "
                            f"(type /help for available commands){_RESET}\n"
                        )
                except SystemExit:
                    break
                except KeyboardInterrupt:
                    self._write(f"\n{_DIM}Cancelled.{_RESET}\n")
                except Exception as exc:
                    self._write(f"{_RED}Command failed: {exc}{_RESET}\n")
                continue

            # It's a task — run via the orchestrator.
            await self._run_task(value)

    async def _run_task(self, task: str) -> None:
        """Execute a single task through the orchestrator."""
        from maike.cli import run_interactive

        try:
            # Thread resolution strategy:
            # - thread_id set (from a previous turn)  → reuse that thread
            # - /new was called (_force_new_thread)    → force a new thread
            # - neither (first task, or after restart) → let the orchestrator
            #   auto-resume the last active thread, or create one if none exists
            force_new = self._force_new_thread
            self._force_new_thread = False

            result = await run_interactive(
                task=task,
                workspace=self.workspace,
                provider=self.provider,
                model=self.model,
                budget=self.budget,
                language="python",
                language_explicit=False,
                thread_id=self.thread_id,
                new_thread=force_new,
                show_banner=False,
                yes=self.yes,
                verbose=self.verbose,
                # Suppress the inner turn_callback — the REPL manages its
                # own follow-up loop.  Without this, run_interactive shows
                # its own "─── maike ▸" prompt, creating a confusing
                # double-prompt after the first response.
                suppress_turn_callback=True,
                advisor_enabled=self.advisor_enabled,
                advisor_provider=self.advisor_provider,
                advisor_model=self.advisor_model,
                advisor_budget_pct=self.advisor_budget_pct,
            )
        except (SystemExit, KeyboardInterrupt):
            return
        except Exception as exc:
            # Extract the most informative error message — some SDKs
            # (Gemini, OpenAI) nest the real message in attributes.
            err_msg = str(exc).strip()
            if not err_msg:
                err_msg = getattr(exc, "message", "") or ""
            if not err_msg and hasattr(exc, "args") and exc.args:
                err_msg = str(exc.args[0]).strip()
            if not err_msg:
                err_msg = type(exc).__name__
            self._write(f"{_RED}Error: {err_msg}{_RESET}\n")
            return

        if result is None:
            return

        # Print the agent's text output.
        agent_output = self._extract_agent_output(result)
        if agent_output:
            sys.stdout.write(f"\n{agent_output}\n")
            sys.stdout.flush()

        # Extract thread_id from the result for continuity.
        new_tid = self._extract_thread_id(result)
        if new_tid and self.thread_id is None:
            self.thread_id = new_tid

        # Accumulate cost and store metadata from the result.
        for stage_results in getattr(result, "stage_results", {}).values():
            for agent_result in stage_results:
                self.total_cost += getattr(agent_result, "cost_usd", 0) or 0
                meta = getattr(agent_result, "metadata", None)
                if meta:
                    self._last_run_metadata = meta

        budget_str = f"${self.budget:.2f}" if self.budget else "unlimited"
        self._write(f"{_DIM}  cost: ${self.total_cost:.4f} / {budget_str}{_RESET}\n")

    async def _run_plan_task(self, task: str) -> None:
        """Execute a task in plan mode — read-only tools, structured output."""
        from maike.cli import run_interactive

        self._write(f"{_DIM}  [plan mode — read-only]{_RESET}\n")
        try:
            result = await run_interactive(
                task=task,
                workspace=self.workspace,
                provider=self.provider,
                model=self.model,
                budget=self.budget,
                language="python",
                language_explicit=False,
                thread_id=self.thread_id,
                show_banner=False,
                yes=self.yes,
                verbose=self.verbose,
                suppress_turn_callback=True,
                tool_profile_override="delegate_plan",
                advisor_enabled=self.advisor_enabled,
                advisor_provider=self.advisor_provider,
                advisor_model=self.advisor_model,
                advisor_budget_pct=self.advisor_budget_pct,
            )
        except (SystemExit, KeyboardInterrupt):
            return
        except Exception as exc:
            self._write(f"{_RED}Error: {exc}{_RESET}\n")
            return

        if result is None:
            return

        agent_output = self._extract_agent_output(result)
        if agent_output:
            sys.stdout.write(f"\n{agent_output}\n")
            sys.stdout.flush()

        for stage_results in getattr(result, "stage_results", {}).values():
            for agent_result in stage_results:
                self.total_cost += getattr(agent_result, "cost_usd", 0) or 0

        self._write(f"{_DIM}  cost: ${self.total_cost:.4f}{_RESET}\n")

    @staticmethod
    def _extract_agent_output(result) -> str:
        """Pull the final agent text output from the orchestrator result."""
        stage_results = getattr(result, "stage_results", {})
        for stage_agents in stage_results.values():
            for agent_result in reversed(stage_agents):
                output = getattr(agent_result, "output", None)
                if output and output.strip():
                    return output.strip()
        return ""

    @staticmethod
    def _extract_thread_id(result) -> str | None:
        """Pull thread_id from the orchestrator result."""
        # OrchestratorResult now has a thread_id field directly.
        tid = getattr(result, "thread_id", None)
        if tid:
            return tid
        # Fallback: check metadata (legacy).
        metadata = getattr(result, "metadata", None) or {}
        return metadata.get("thread_id")

    def _write(self, text: str) -> None:
        """Write to stderr (where REPL output goes)."""
        sys.stderr.write(text)
        sys.stderr.flush()

    async def _cleanup_stale_sessions(self) -> None:
        """Mark any orphaned 'running' sessions as 'cancelled'.

        Sessions can get stuck in 'running' state if the process was killed
        (Ctrl+C, crash, etc.).  Clean them up on REPL start.
        """
        try:
            from maike.memory.session import SessionStore
            store = SessionStore(self.workspace / ".maike")
            await store.initialize()
            async with store.use_shared_connection():
                # Direct query for running sessions in this workspace
                async with store._db() as db:
                    cursor = await db.execute(
                        "SELECT id FROM sessions WHERE status = 'running' AND workspace = ?",
                        (str(self.workspace),),
                    )
                    stale_rows = await cursor.fetchall()
                count = 0
                for (sid,) in stale_rows:
                    await store.mark_session_status(sid, "cancelled")
                    count += 1
                if count:
                    self._write(f"{_DIM}  cleaned up {count} stale session(s){_RESET}\n")
        except Exception:
            pass  # best-effort

    def _print_welcome(self) -> None:
        sys.stderr.write(
            f"\n{_BOLD}mAIke interactive session{_RESET}\n"
            f"{_DIM}  workspace: {self.workspace}{_RESET}\n"
            f"{_DIM}  provider:  {self.provider}  model: {self.model}{_RESET}\n"
            f"{_DIM}  budget:    {'${:.2f}'.format(self.budget) if self.budget else 'unlimited'}{_RESET}\n"
            f"{_DIM}  Type /help for commands, /quit to exit{_RESET}\n\n"
        )
        sys.stderr.flush()
