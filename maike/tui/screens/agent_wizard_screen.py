"""Full-screen agent creation wizard for the TUI.

Two-column layout: scrollable form on the left, live markdown preview
on the right.  All agent definition fields are surfaced with guidance
text so users can create effective agents without reading docs.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    SelectionList,
    Static,
    Switch,
    TextArea,
)

from maike.agents.agent_commands import (
    ALL_TOOL_NAMES,
    AgentWizardData,
    preview_agent_markdown,
)


class AgentWizardScreen(ModalScreen[AgentWizardData | None]):
    """Modal wizard for creating a custom agent definition."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    AgentWizardScreen {
        align: center middle;
    }

    #wizard-container {
        width: 96%;
        max-width: 120;
        height: 90%;
        border: heavy #8b5cf6;
        background: $surface;
        padding: 1 2;
    }

    #wizard-title {
        text-style: bold;
        color: #8b5cf6;
        margin-bottom: 1;
    }

    #wizard-columns {
        height: 1fr;
    }

    #form-scroll {
        width: 1fr;
        min-width: 40;
        margin-right: 1;
    }

    #preview-pane {
        width: 1fr;
        min-width: 30;
        border-left: solid #444;
        padding: 0 1;
        overflow-y: auto;
    }

    #preview-label {
        text-style: bold;
        color: #8b5cf6;
        margin-bottom: 1;
    }

    #preview-content {
        color: $text-muted;
    }

    .field-label {
        text-style: bold;
        margin-top: 1;
    }

    .field-hint {
        color: $text-muted;
        margin-bottom: 0;
    }

    #system-prompt-area {
        height: 8;
        min-height: 5;
    }

    #tool-mode-radio {
        margin: 0;
        height: auto;
    }

    #tool-selection {
        height: auto;
        max-height: 13;
    }

    #button-row {
        margin-top: 1;
        height: auto;
        align: center middle;
    }

    #button-row Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        agent_name: str,
        scope: str = "project",
        provider: str = "gemini",
        model: str = "",
    ) -> None:
        super().__init__()
        self._agent_name = agent_name
        self._scope = scope
        self._provider = provider
        self._model = model

    def compose(self) -> ComposeResult:
        with Vertical(id="wizard-container"):
            yield Static(
                f"Create Agent: [bold]{self._agent_name}[/bold]",
                id="wizard-title",
            )

            with Horizontal(id="wizard-columns"):
                # ── Left column: form ──
                with VerticalScroll(id="form-scroll"):
                    # Description
                    yield Label("Description", classes="field-label")
                    yield Static(
                        "When should mAIke auto-delegate to this agent?",
                        classes="field-hint",
                    )
                    yield Input(
                        placeholder="e.g. Reviews code for security and correctness",
                        id="description-input",
                    )

                    # Model tier
                    yield Label("Model Tier", classes="field-label")
                    yield Static(
                        "cheap = fast/$, default = balanced, strong = max capability",
                        classes="field-hint",
                    )
                    with RadioSet(id="model-tier-radio"):
                        yield RadioButton("cheap", id="tier-cheap")
                        yield RadioButton("default", id="tier-default", value=True)
                        yield RadioButton("strong", id="tier-strong")

                    # Max turns
                    yield Label("Max Turns", classes="field-label")
                    yield Static(
                        "Iteration budget: 10=focused, 30=typical, 50+=complex",
                        classes="field-hint",
                    )
                    yield Input(value="30", id="max-turns-input")

                    # System prompt
                    yield Label("System Prompt", classes="field-label")
                    yield Static(
                        "Defines the agent's behavior — the most important field.",
                        classes="field-hint",
                    )
                    yield Button(
                        "Generate from description",
                        variant="default",
                        id="generate-btn",
                    )
                    yield TextArea(id="system-prompt-area")

                    # Tools
                    yield Label("Tool Access", classes="field-label")
                    yield Static(
                        "Which tools can this agent use?",
                        classes="field-hint",
                    )
                    with RadioSet(id="tool-mode-radio"):
                        yield RadioButton("All tools", id="tools-all", value=True)
                        yield RadioButton("Only selected", id="tools-selected")
                        yield RadioButton("All except selected", id="tools-excluded")

                    yield SelectionList[str](
                        *((name, name, True) for name in ALL_TOOL_NAMES),
                        id="tool-selection",
                    )

                    # Skills
                    yield Label("Skills", classes="field-label")
                    yield Static(
                        "Auto-injected knowledge modules (comma-separated)",
                        classes="field-hint",
                    )
                    yield Input(placeholder="e.g. test-methodology, debugging", id="skills-input")

                    # Initial prompt
                    yield Label("Initial Prompt", classes="field-label")
                    yield Static(
                        "Prepended to every task (e.g. \"Read ARCHITECTURE.md first\")",
                        classes="field-hint",
                    )
                    yield Input(id="initial-prompt-input")

                    # Critical reminder
                    yield Label("Critical Reminder", classes="field-label")
                    yield Static(
                        "Constraint injected every LLM turn (e.g. \"Never modify prod data\")",
                        classes="field-hint",
                    )
                    yield Input(id="critical-reminder-input")

                    # Background
                    yield Label("Background", classes="field-label")
                    yield Static(
                        "Run this agent async by default?",
                        classes="field-hint",
                    )
                    yield Switch(id="background-switch", value=False)

                    # Buttons
                    with Horizontal(id="button-row"):
                        yield Button("Create Agent", variant="primary", id="create-btn")
                        yield Button("Cancel", variant="error", id="cancel-btn")

                # ── Right column: live preview ──
                with VerticalScroll(id="preview-pane"):
                    yield Static("Preview", id="preview-label")
                    yield Static("", id="preview-content")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self) -> AgentWizardData:
        """Read all form fields and return wizard data."""
        description = self.query_one("#description-input", Input).value.strip()

        # Model tier from radio set.
        model_tier = "default"
        model_radio = self.query_one("#model-tier-radio", RadioSet)
        if model_radio.pressed_index == 0:
            model_tier = "cheap"
        elif model_radio.pressed_index == 2:
            model_tier = "strong"

        # Max turns.
        try:
            max_turns = int(self.query_one("#max-turns-input", Input).value)
        except (ValueError, TypeError):
            max_turns = 30

        system_prompt = self.query_one("#system-prompt-area", TextArea).text.strip()

        # Tools.
        tool_mode = self.query_one("#tool-mode-radio", RadioSet)
        tool_sel = self.query_one("#tool-selection", SelectionList)
        selected_tools = list(tool_sel.selected)

        allowed_tools: list[str] | None = None
        disallowed_tools: list[str] | None = None
        if tool_mode.pressed_index == 1:
            # "Only selected"
            allowed_tools = selected_tools
        elif tool_mode.pressed_index == 2:
            # "All except selected"
            disallowed_tools = selected_tools

        # Skills.
        skills_raw = self.query_one("#skills-input", Input).value.strip()
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

        initial_prompt = self.query_one("#initial-prompt-input", Input).value.strip() or None
        critical_reminder = self.query_one("#critical-reminder-input", Input).value.strip() or None
        background = self.query_one("#background-switch", Switch).value

        return AgentWizardData(
            name=self._agent_name,
            description=description or f"A custom {self._agent_name} agent",
            system_prompt=system_prompt,
            model_tier=model_tier,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            skills=skills,
            initial_prompt=initial_prompt,
            background=background,
            critical_reminder=critical_reminder,
            scope=self._scope,
        )

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        """Regenerate the preview pane from current form state."""
        try:
            data = self._collect_data()
            md = preview_agent_markdown(data)
            self.query_one("#preview-content", Static).update(md)
        except Exception:
            pass  # form may be partially composed

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_preview()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._refresh_preview()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        # Show/hide tool selection based on tool mode.
        tool_mode = self.query_one("#tool-mode-radio", RadioSet)
        tool_sel = self.query_one("#tool-selection", SelectionList)
        tool_sel.display = tool_mode.pressed_index != 0  # hide when "All tools"
        self._refresh_preview()

    def on_selection_list_selected_changed(self) -> None:
        self._refresh_preview()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._refresh_preview()

    def on_mount(self) -> None:
        # Hide tool selection initially (default = "All tools").
        self.query_one("#tool-selection", SelectionList).display = False
        self._refresh_preview()
        self.query_one("#description-input", Input).focus()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-btn":
            self.dismiss(self._collect_data())
        elif event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "generate-btn":
            self._generate_prompt()

    def _generate_prompt(self) -> None:
        """Run LLM generation in a background worker."""
        description = self.query_one("#description-input", Input).value.strip()
        if not description:
            self.query_one("#system-prompt-area", TextArea).text = (
                "# Enter a description first, then click Generate."
            )
            return

        btn = self.query_one("#generate-btn", Button)
        btn.label = "Generating..."
        btn.disabled = True

        async def _do_generate() -> str:
            from maike.agents.agent_commands import generate_agent_prompt
            from maike.constants import cheap_model_for_provider

            cheap_model = cheap_model_for_provider(self._provider)
            return await generate_agent_prompt(
                description=description,
                name=self._agent_name,
                tools=None,
                provider=self._provider,
                model=self._model or cheap_model,
            )

        def _on_done(result: str) -> None:
            btn.label = "Generate from description"
            btn.disabled = False
            if result:
                self.query_one("#system-prompt-area", TextArea).text = result
                self._refresh_preview()

        self.app.run_worker(
            _do_generate,
            thread=True,
            exit_on_error=False,
        ).then(_on_done)

    def action_cancel(self) -> None:
        self.dismiss(None)
