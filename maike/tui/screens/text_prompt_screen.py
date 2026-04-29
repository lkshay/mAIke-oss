"""Modal screen for free-text input prompts.

Used by the AskUser tool and other interactions that need free-text
from the user rather than a simple yes/no.
"""

from __future__ import annotations

from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class TextPromptScreen(ModalScreen[str]):
    """Modal dialog for collecting free-text input from the user."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    TextPromptScreen {
        align: center middle;
    }
    #text-prompt-dialog {
        width: 70;
        max-width: 90%;
        padding: 1 2;
        border: heavy #8b5cf6;
        background: $surface;
    }
    #text-prompt-title {
        margin: 0 0 1 0;
    }
    #text-prompt-message {
        margin: 0 0 1 0;
        max-height: 10;
    }
    #text-prompt-input {
        margin: 0 0 1 0;
    }
    #text-prompt-dialog Button {
        margin: 0 1;
        width: auto;
    }
    """

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self.prompt_text = prompt

    def compose(self):
        yield Vertical(
            Static(
                "[bold #8b5cf6]\u270f Input Required[/bold #8b5cf6]",
                id="text-prompt-title",
            ),
            Static(self.prompt_text, id="text-prompt-message"),
            Input(placeholder="Type your response ...", id="text-prompt-input"),
            Button("Submit", variant="primary", id="submit-btn"),
            id="text-prompt-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#text-prompt-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text:
            self.dismiss(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        text = self.query_one("#text-prompt-input", Input).value.strip()
        self.dismiss(text or "cancelled")

    def action_cancel(self) -> None:
        self.dismiss("cancelled")
