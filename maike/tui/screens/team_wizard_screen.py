"""Team creation wizard for the TUI.

Two-column layout: scrollable form on the left, live markdown preview
on the right.  Members can be added/removed dynamically.
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
    Static,
    TextArea,
)

from maike.agents.team_commands import (
    TeamWizardData,
    TeamWizardMemberData,
    preview_team_markdown,
)


class TeamWizardScreen(ModalScreen[TeamWizardData | None]):
    """Modal wizard for creating a team definition."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    TeamWizardScreen {
        align: center middle;
    }

    #team-wizard-container {
        width: 96%;
        max-width: 120;
        height: 90%;
        border: heavy #3dbfa0;
        background: $surface;
        padding: 1 2;
    }

    #team-wizard-title {
        text-style: bold;
        color: #3dbfa0;
        margin-bottom: 1;
    }

    #team-wizard-columns {
        height: 1fr;
    }

    #team-form-scroll {
        width: 1fr;
        min-width: 40;
        margin-right: 1;
    }

    #team-preview-pane {
        width: 1fr;
        min-width: 30;
        border-left: solid #444;
        padding: 0 1;
        overflow-y: auto;
    }

    #team-preview-label {
        text-style: bold;
        color: #3dbfa0;
        margin-bottom: 1;
    }

    #team-preview-content {
        color: $text-muted;
    }

    .team-field-label {
        text-style: bold;
        margin-top: 1;
    }

    .team-field-hint {
        color: $text-muted;
        margin-bottom: 0;
    }

    #synthesis-area {
        height: 5;
        min-height: 3;
    }

    .member-row {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
        border: solid #444;
    }

    .member-row Input {
        margin: 0;
    }

    #team-button-row {
        margin-top: 1;
        height: auto;
        align: center middle;
    }

    #team-button-row Button {
        margin: 0 1;
    }

    #add-member-btn {
        margin-top: 1;
    }
    """

    def __init__(self, team_name: str, scope: str = "project") -> None:
        super().__init__()
        self._team_name = team_name
        self._scope = scope
        self._member_count = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="team-wizard-container"):
            yield Static(
                f"Create Team: [bold]{self._team_name}[/bold]",
                id="team-wizard-title",
            )

            with Horizontal(id="team-wizard-columns"):
                # ── Left column: form ──
                with VerticalScroll(id="team-form-scroll"):
                    # Description
                    yield Label("Description", classes="team-field-label")
                    yield Static(
                        "What does this team do? Shown in the team catalog.",
                        classes="team-field-hint",
                    )
                    yield Input(
                        placeholder="e.g. Comprehensive code review pipeline",
                        id="team-description",
                    )

                    # Process type
                    yield Label("Process Type", classes="team-field-label")
                    yield Static(
                        "parallel = all at once, sequential = output chains",
                        classes="team-field-hint",
                    )
                    with RadioSet(id="team-process-radio"):
                        yield RadioButton("parallel", id="proc-parallel", value=True)
                        yield RadioButton("sequential", id="proc-sequential")

                    # On failure
                    yield Label("On Failure", classes="team-field-label")
                    yield Static(
                        "What happens when a member fails?",
                        classes="team-field-hint",
                    )
                    with RadioSet(id="team-failure-radio"):
                        yield RadioButton("continue", id="fail-continue", value=True)
                        yield RadioButton("abort", id="fail-abort")
                        yield RadioButton("retry", id="fail-retry")

                    # Members
                    yield Label("Members", classes="team-field-label")
                    yield Static(
                        "Add agents or inline roles to the team.",
                        classes="team-field-hint",
                    )
                    # Member rows are mounted before this button dynamically.
                    yield Button(
                        "+ Add Member",
                        variant="default",
                        id="add-member-btn",
                    )

                    # Synthesis
                    yield Label("Synthesis Prompt", classes="team-field-label")
                    yield Static(
                        "How should results be combined? Leave empty for default.",
                        classes="team-field-hint",
                    )
                    yield TextArea(id="synthesis-area")

                    # Buttons
                    with Horizontal(id="team-button-row"):
                        yield Button("Create Team", variant="primary", id="team-create-btn")
                        yield Button("Cancel", variant="error", id="team-cancel-btn")

                # ── Right column: live preview ──
                with VerticalScroll(id="team-preview-pane"):
                    yield Static("Preview", id="team-preview-label")
                    yield Static("", id="team-preview-content")

    # ------------------------------------------------------------------
    # Member management
    # ------------------------------------------------------------------

    def _add_member_row(self) -> None:
        """Add a new member input row to the form."""
        idx = self._member_count
        self._member_count += 1

        # Mount before the "Add Member" button so new rows appear above it.
        add_btn = self.query_one("#add-member-btn")
        label = Static(
            f"[bold]Member {idx + 1}[/bold]",
            id=f"member-label-{idx}",
        )
        agent_input = Input(
            placeholder="agent name (or leave empty for inline role)",
            id=f"member-agent-{idx}",
        )
        role_input = Input(
            placeholder="role description",
            id=f"member-role-{idx}",
        )
        remove_btn = Button(
            f"Remove member {idx + 1}",
            variant="error",
            id=f"remove-member-{idx}",
        )
        parent = add_btn.parent
        parent.mount(label, before=add_btn)
        parent.mount(agent_input, before=add_btn)
        parent.mount(role_input, before=add_btn)
        parent.mount(remove_btn, before=add_btn)
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self) -> TeamWizardData:
        """Read all form fields and return wizard data."""
        description = self.query_one("#team-description", Input).value.strip()

        process_radio = self.query_one("#team-process-radio", RadioSet)
        process_type = "sequential" if process_radio.pressed_index == 1 else "parallel"

        failure_radio = self.query_one("#team-failure-radio", RadioSet)
        on_failure = ["continue", "abort", "retry"][failure_radio.pressed_index or 0]

        # Collect members from dynamic inputs.
        members: list[TeamWizardMemberData] = []
        for i in range(self._member_count):
            try:
                agent_input = self.query_one(f"#member-agent-{i}", Input)
            except Exception:
                continue  # removed member
            if agent_input.display is False:
                continue
            role_input = self.query_one(f"#member-role-{i}", Input)
            agent = agent_input.value.strip() or None
            role = role_input.value.strip() or None
            if agent or role:
                members.append(TeamWizardMemberData(agent=agent, role=role))

        synthesis = self.query_one("#synthesis-area", TextArea).text.strip()

        return TeamWizardData(
            name=self._team_name,
            description=description or f"A custom team: {self._team_name}",
            process_type=process_type,
            on_failure=on_failure,
            members=members,
            synthesis_prompt=synthesis,
            scope=self._scope,
        )

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        try:
            data = self._collect_data()
            md = preview_team_markdown(data)
            self.query_one("#team-preview-content", Static).update(md)
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_preview()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._refresh_preview()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._refresh_preview()

    def on_mount(self) -> None:
        # Start with one empty member row.
        self._add_member_row()
        self._refresh_preview()
        self.query_one("#team-description", Input).focus()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        if btn_id == "team-create-btn":
            self.dismiss(self._collect_data())
        elif btn_id == "team-cancel-btn":
            self.dismiss(None)
        elif btn_id == "add-member-btn":
            self._add_member_row()
        elif btn_id.startswith("remove-member-"):
            idx = btn_id.replace("remove-member-", "")
            try:
                for widget_id in (f"member-label-{idx}", f"member-agent-{idx}",
                                  f"member-role-{idx}", f"remove-member-{idx}"):
                    self.query_one(f"#{widget_id}").display = False
                self._refresh_preview()
            except Exception:
                pass

    def action_cancel(self) -> None:
        self.dismiss(None)
