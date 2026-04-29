"""Inline approval widget — navigable option list for y/n/feedback.

Replaces the PromptInput area during approval prompts.  The user
navigates options with Up/Down and selects with Enter.
"""

from __future__ import annotations

import itertools
from typing import Callable

from textual.widgets import OptionList
from textual.widgets.option_list import Option


# Approval option values — used to map selection back to result
APPROVE = "approve"
DENY = "deny"
APPROVE_ALWAYS = "approve_always"

# Unique ID counter — prevents duplicate widget ID errors when
# multiple approval requests fire concurrently (shouldn't happen
# with proper delegate auto-approve, but safety net).
_id_counter = itertools.count(1)


class ApprovalList(OptionList):
    """Inline navigable option list for approval prompts."""

    DEFAULT_CSS = """
    ApprovalList {
        dock: bottom;
        height: auto;
        max-height: 6;
        margin: 0 1;
        border: heavy $warning;
        background: $surface;
    }
    ApprovalList:focus {
        border: heavy $warning;
    }
    ApprovalList > .option-list--option-highlighted {
        background: $warning 30%;
    }
    """

    def __init__(self, prompt: str, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._prompt = prompt
        uid = next(_id_counter)
        super().__init__(
            Option(f"\u2714 Allow              {prompt[:40]}", id=APPROVE),
            Option("\u2718 Deny", id=DENY),
            Option("\u2714 Always allow this session", id=APPROVE_ALWAYS),
            id=f"approval-list-{uid}",
        )

    def on_mount(self) -> None:
        self.highlighted = 0
        self.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        value = event.option_id or DENY
        self._callback(value)
