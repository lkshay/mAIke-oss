"""TUI widget exports.

Imports are guarded so that importing this package does not fail when
``textual`` is not installed (e.g. in minimal test environments).
The widgets are re-exported lazily — they're available when accessed,
but don't force a ``textual`` import at package-load time.
"""

__all__ = [
    "DelegateWidget",
    "HeaderBar",
    "MessageList",
    "PromptInput",
    "StatusBar",
    "ToolCallWidget",
    "TurnSeparator",
]


def __getattr__(name: str):
    """Lazy imports — only load widget modules when accessed."""
    _map = {
        "DelegateWidget": "maike.tui.widgets.delegate_widget",
        "HeaderBar": "maike.tui.widgets.header_bar",
        "MessageList": "maike.tui.widgets.message_list",
        "PromptInput": "maike.tui.widgets.input_area",
        "StatusBar": "maike.tui.widgets.status_bar",
        "ToolCallWidget": "maike.tui.widgets.tool_widget",
        "TurnSeparator": "maike.tui.widgets.turn_separator",
    }
    module_path = _map.get(name)
    if module_path is not None:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
