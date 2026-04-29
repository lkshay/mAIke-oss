"""Full-output modal — show the untruncated text of a tool / delegate / message.

Invoked when the user wants to see the complete output of an item that was
displayed with truncation in the main message list.  Press ``q`` or
``Escape`` to close.  Supports text selection and scrolling.
"""

from __future__ import annotations

from rich.syntax import Syntax
from rich.text import Text as RichText
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Static

from maike.tui.theme import MAIKE_ACCENT, MAIKE_DIM


class FullOutputScreen(ModalScreen[None]):
    """A modal that renders a block of text in full, scrollable."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
        Binding("pageup", "page_up", "Page Up", show=False),
        Binding("pagedown", "page_down", "Page Down", show=False),
        Binding("home", "home", "Top", show=False),
        Binding("end", "end", "Bottom", show=False),
        Binding("g", "home", "Top", show=False),
        Binding("G", "end", "Bottom", show=False),
    ]

    DEFAULT_CSS = """
    FullOutputScreen {
        align: center middle;
    }
    FullOutputScreen > #full-output-box {
        width: 90%;
        height: 85%;
        background: $surface;
        border: thick $accent 80%;
        padding: 0 1;
    }
    FullOutputScreen #full-output-header {
        height: 1;
        background: $primary-background;
        color: $text-muted;
        padding: 0 1;
    }
    FullOutputScreen #full-output-body {
        height: 1fr;
        padding: 1 1;
        scrollbar-gutter: stable;
    }
    FullOutputScreen Footer {
        dock: bottom;
    }
    """

    def __init__(self, title: str, content: str, language: str | None = None) -> None:
        super().__init__()
        self._title = title
        self._content = content or "(empty)"
        self._language = language

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(id="full-output-box"):
            yield Static(
                RichText.from_markup(
                    f"  [bold {MAIKE_ACCENT}]{self._title}[/bold {MAIKE_ACCENT}]  "
                    f"[{MAIKE_DIM}]· {len(self._content):,} chars · "
                    f"{self._content.count(chr(10)) + 1:,} lines · Esc/q to close[/{MAIKE_DIM}]"
                ),
                id="full-output-header",
            )
            with VerticalScroll(id="full-output-body"):
                if self._language:
                    yield Static(
                        Syntax(
                            self._content,
                            self._language,
                            line_numbers=True,
                            word_wrap=False,
                            theme="monokai",
                        )
                    )
                else:
                    # Plain text — preserve whitespace & allow selection
                    yield Static(self._content, markup=False)
        yield Footer()

    def action_dismiss(self) -> None:
        self.app.pop_screen()

    def action_page_up(self) -> None:
        self.query_one("#full-output-body", VerticalScroll).scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self.query_one("#full-output-body", VerticalScroll).scroll_page_down(animate=False)

    def action_home(self) -> None:
        self.query_one("#full-output-body", VerticalScroll).scroll_home(animate=False)

    def action_end(self) -> None:
        self.query_one("#full-output-body", VerticalScroll).scroll_end(animate=False)
