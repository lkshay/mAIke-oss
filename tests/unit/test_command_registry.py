"""Tests for the slash command registry."""

from types import SimpleNamespace

from maike.commands.registry import CommandRegistry


def _make_session():
    return SimpleNamespace(total_cost=0.5, budget=5.0, thread_id=None)


class TestCommandRegistry:
    def test_register_and_dispatch(self):
        reg = CommandRegistry()
        captured = {}

        def handler(session, args):
            captured["session"] = session
            captured["args"] = args

        reg.register("test", "A test command", handler)
        session = _make_session()
        result = reg.dispatch("/test arg1 arg2", session)
        assert result is True
        assert captured["session"] is session
        assert captured["args"] == ["arg1", "arg2"]

    def test_dispatch_unknown_returns_false(self):
        reg = CommandRegistry()
        assert reg.dispatch("/unknown", _make_session()) is False

    def test_dispatch_no_args(self):
        reg = CommandRegistry()
        captured = {}

        def handler(session, args):
            captured["args"] = args

        reg.register("help", "Show help", handler)
        reg.dispatch("/help", _make_session())
        assert captured["args"] == []

    def test_help_text(self):
        reg = CommandRegistry()
        reg.register("help", "Show help", lambda s, a: None)
        reg.register("cost", "Show cost", lambda s, a: None)
        text = reg.help_text()
        assert "/help" in text
        assert "/cost" in text
        assert "Show help" in text

    def test_command_names(self):
        reg = CommandRegistry()
        reg.register("help", "Show help", lambda s, a: None)
        reg.register("cost", "Show cost", lambda s, a: None)
        assert reg.command_names == ["cost", "help"]

    def test_case_insensitive(self):
        reg = CommandRegistry()
        captured = {}

        def handler(session, args):
            captured["called"] = True

        reg.register("Help", "Show help", handler)
        assert reg.dispatch("/HELP", _make_session()) is True
        assert captured.get("called") is True


class TestBuiltinRegistration:
    def test_builtins_register(self):
        from maike.commands.builtins import register_builtins

        reg = CommandRegistry()
        register_builtins(reg)
        # Check core commands exist.
        assert "help" in reg.command_names
        assert "cost" in reg.command_names
        assert "status" in reg.command_names
        assert "tasks" in reg.command_names
        assert "quit" in reg.command_names
        assert "new" in reg.command_names
