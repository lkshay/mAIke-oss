import asyncio

from maike.orchestrator.preflight import PreflightChecker
from maike.runtime.local import RuntimeConfig


class FakeRuntime:
    def __init__(self, *, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()

    async def git_available(self):
        return True

    async def is_git_repo(self):
        return True

    async def init_git_repo(self):
        raise AssertionError("init_git_repo should not be called in this test")


class FakeApprovalGate:
    async def confirm(self, prompt: str):
        del prompt
        return True


def test_preflight_accepts_gemini_key_without_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    report = asyncio.run(
        PreflightChecker(FakeRuntime(), FakeApprovalGate()).ensure_ready("gemini")
    )

    assert report.provider_name == "gemini"
    assert report.api_key_present is True
    assert report.environment_language == "python"
    assert report.environment_ready is True


def test_preflight_rejects_unready_runtime(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    runtime = FakeRuntime(
        config=RuntimeConfig(
            language="python",
            environment_ready=False,
            diagnostics=["No Python interpreter was found on PATH."],
        )
    )

    try:
        asyncio.run(PreflightChecker(runtime, FakeApprovalGate()).ensure_ready("gemini"))
    except RuntimeError as exc:
        assert str(exc) == "No Python interpreter was found on PATH."
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected preflight to fail for an unready runtime")
