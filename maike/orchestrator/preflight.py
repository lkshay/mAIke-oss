"""Preflight checks for CLI execution."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from maike.gateway.providers import ProviderName, resolve_provider_name
from maike.runtime.protocol import ExecutionRuntime
from maike.safety.approval import ApprovalGate


class PreflightError(RuntimeError):
    """Raised when the local environment is not ready for a run."""


@dataclass
class PreflightReport:
    python_ok: bool
    api_key_present: bool
    provider_name: str
    git_available: bool
    git_repo_ready: bool
    environment_language: str
    package_manager: str | None
    environment_confidence: str
    environment_ready: bool
    diagnostics: list[str]


class PreflightChecker:
    def __init__(self, runtime: ExecutionRuntime, approval_gate: ApprovalGate) -> None:
        self.runtime = runtime
        self.approval_gate = approval_gate

    async def ensure_ready(self, provider_name: str) -> PreflightReport:
        provider = resolve_provider_name(provider_name)
        python_ok = sys.version_info >= (3, 11)
        api_key_present = bool(self._provider_api_key(provider))
        git_available = await self.runtime.git_available()
        git_repo_ready = False
        config = getattr(self.runtime, "config", None)
        environment_language = getattr(config, "language", "unknown")
        package_manager = getattr(config, "package_manager", None)
        environment_confidence = getattr(config, "environment_confidence", "low")
        environment_ready = bool(getattr(config, "environment_ready", True))
        diagnostics = list(getattr(config, "diagnostics", []))

        if not python_ok:
            raise PreflightError("Python 3.11+ is required.")
        if not api_key_present:
            raise PreflightError(self._missing_key_message(provider))
        if not git_available:
            raise PreflightError("git is required for checkpoints and rollback.")
        if not environment_ready:
            detail = diagnostics[0] if diagnostics else f"{environment_language} runtime is not ready."
            raise PreflightError(detail)

        # Report git state without blocking.
        git_repo_ready = await self.runtime.is_git_repo()

        return PreflightReport(
            python_ok=python_ok,
            api_key_present=api_key_present,
            provider_name=provider.value,
            git_available=git_available,
            git_repo_ready=git_repo_ready,
            environment_language=environment_language,
            package_manager=package_manager,
            environment_confidence=environment_confidence,
            environment_ready=environment_ready,
            diagnostics=diagnostics,
        )

    def _provider_api_key(self, provider: ProviderName) -> str | None:
        # Ollama runs locally — no API key needed.
        if provider == ProviderName.OLLAMA:
            return "local"
        env_keys = {
            ProviderName.ANTHROPIC: ("ANTHROPIC_API_KEY",),
            ProviderName.OPENAI: ("OPENAI_API_KEY",),
            ProviderName.GEMINI: ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        }
        # Vertex AI uses Application Default Credentials — no API key needed.
        if provider == ProviderName.GEMINI:
            if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1"):
                return "vertex-ai-adc"
        for key in env_keys.get(provider, ()):
            value = os.getenv(key)
            if value:
                return value
        return None

    def _missing_key_message(self, provider: ProviderName) -> str:
        messages = {
            ProviderName.ANTHROPIC: "ANTHROPIC_API_KEY is required for Anthropic runs.",
            ProviderName.OPENAI: "OPENAI_API_KEY is required for OpenAI runs.",
            ProviderName.GEMINI: "GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini runs.",
            ProviderName.OLLAMA: "Ollama must be running locally (ollama serve).",
        }
        return messages.get(provider, f"API key required for {provider.value}.")
