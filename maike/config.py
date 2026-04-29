"""User configuration management for mAIke.

Keys and default provider are stored in ~/.config/maike/.env so that
`maike` works from any directory without requiring a local .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "maike"
CONFIG_FILE = CONFIG_DIR / ".env"

# Ordered list used for display in the setup wizard.
PROVIDERS: list[dict] = [
    {
        "key": "anthropic",
        "label": "Anthropic",
        "tagline": "Claude — best reasoning & coding",
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "pricing": "$15 / $75 per M tokens",
    },
    {
        "key": "openai",
        "label": "OpenAI",
        "tagline": "GPT — fast and widely supported",
        "model": "gpt-5.5",
        "env_key": "OPENAI_API_KEY",
        "pricing": "$2.50 / $15 per M tokens",
    },
    {
        "key": "gemini",
        "label": "Google Gemini",
        "tagline": "Gemini Flash — huge context, lowest cost",
        "model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "pricing": "$0.30 / $2.50 per M tokens",
    },
    {
        "key": "ollama",
        "label": "Ollama (local)",
        "tagline": "Local models — free, private, no API key",
        "model": "gemma4:26b",
        "env_key": None,  # No API key needed
        "pricing": "Free (local)",
    },
]

_PROVIDER_BY_KEY: dict[str, dict] = {p["key"]: p for p in PROVIDERS}


def load_user_config() -> None:
    """Load ~/.config/maike/.env into the process environment (does not override already-set vars)."""
    if CONFIG_FILE.exists():
        try:
            from dotenv import load_dotenv as _load
            _load(CONFIG_FILE, override=False)
        except ImportError:
            pass


def get_configured_provider() -> str | None:
    """Return the provider saved by the setup wizard, or None."""
    return os.getenv("MAIKE_DEFAULT_PROVIDER")


def has_key_for_provider(provider: str) -> bool:
    """Return True if credentials for *provider* are present in the environment."""
    info = _PROVIDER_BY_KEY.get(provider)
    if info is None:
        return False
    env_key = info.get("env_key")
    if not env_key:
        # Provider needs no API key (e.g., Ollama runs locally).
        return provider == "ollama"
    if os.getenv(env_key):
        return True
    # Gemini accepts GOOGLE_API_KEY, or Vertex AI auth via gcloud.
    if provider == "gemini":
        if os.getenv("GOOGLE_API_KEY"):
            return True
        if (
            os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")
            and os.getenv("GOOGLE_CLOUD_PROJECT")
        ):
            return True
    return False


def has_any_configured_provider() -> bool:
    """Return True if at least one provider API key is available."""
    return any(has_key_for_provider(p["key"]) for p in PROVIDERS)


def save_provider_config(provider: str, api_key: str | None = None) -> None:
    """Persist *provider* (and optionally *api_key*) to ~/.config/maike/.env.

    When *api_key* is ``None``, only ``MAIKE_DEFAULT_PROVIDER`` is written —
    useful when a key for *provider* is already available under an alternate
    env name (e.g. Gemini accepts ``GOOGLE_API_KEY`` in place of
    ``GEMINI_API_KEY``).
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    info = _PROVIDER_BY_KEY[provider]
    env_key = info["env_key"]

    # Preserve any existing keys in the file.
    existing: dict[str, str] = {}
    if CONFIG_FILE.exists():
        try:
            from dotenv import dotenv_values
            existing = dict(dotenv_values(CONFIG_FILE))
        except ImportError:
            pass

    if api_key is not None and env_key:
        existing[env_key] = api_key
    existing["MAIKE_DEFAULT_PROVIDER"] = provider

    CONFIG_FILE.write_text(
        "".join(f'{k}="{v}"\n' for k, v in existing.items()),
        encoding="utf-8",
    )

    # Apply to the running process so the current run works immediately.
    if api_key is not None and env_key:
        os.environ[env_key] = api_key
    os.environ["MAIKE_DEFAULT_PROVIDER"] = provider
