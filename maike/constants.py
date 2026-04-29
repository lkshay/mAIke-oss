"""Shared constants and config defaults for mAIke."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "gemini"  # Updateable constant: verify when upgrading providers/models.
DEFAULT_MODEL = "gemini-3-flash-preview"  # Updateable constant: verify when upgrading models/pricing.
COST_PER_M_INPUT_USD = 15.00  # Updateable constant: verify when upgrading models/pricing.
COST_PER_M_OUTPUT_USD = 75.00  # Updateable constant: verify when upgrading models/pricing.
DEFAULT_OPENAI_MODEL = "gpt-5.5"  # Updateable constant: verify when upgrading providers/models.
OPENAI_COST_PER_M_INPUT_USD = 2.50  # Updateable constant: verify when upgrading models/pricing.
OPENAI_COST_PER_M_OUTPUT_USD = 15.00  # Updateable constant: verify when upgrading models/pricing.
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"  # Updateable constant: verify when upgrading providers/models.
GEMINI_COST_PER_M_INPUT_USD = 0.30  # Updateable constant: verify when upgrading models/pricing.
GEMINI_COST_PER_M_OUTPUT_USD = 2.50  # Updateable constant: verify when upgrading models/pricing.

# Cheap models — lightweight, low-cost; used for summaries, classification, simple subtasks.
DEFAULT_ANTHROPIC_CHEAP_MODEL = "claude-haiku-4-5-20251001"  # Updateable constant
ANTHROPIC_CHEAP_COST_PER_M_INPUT_USD = 0.80  # Updateable constant
ANTHROPIC_CHEAP_COST_PER_M_OUTPUT_USD = 4.00  # Updateable constant
DEFAULT_OPENAI_CHEAP_MODEL = "gpt-5.5-mini"  # Updateable constant
OPENAI_CHEAP_COST_PER_M_INPUT_USD = 0.40  # Updateable constant
OPENAI_CHEAP_COST_PER_M_OUTPUT_USD = 1.60  # Updateable constant
DEFAULT_GEMINI_CHEAP_MODEL = "gemini-2.5-flash"  # Updateable constant
GEMINI_CHEAP_COST_PER_M_INPUT_USD = 0.30  # Updateable constant
GEMINI_CHEAP_COST_PER_M_OUTPUT_USD = 2.50  # Updateable constant

# Strong models — highest capability; used for complex reasoning, architecture, hard debugging.
DEFAULT_ANTHROPIC_STRONG_MODEL = "claude-opus-4-20250514"  # Updateable constant (Opus — kept as the higher-capability tier above the Sonnet default)
ANTHROPIC_STRONG_COST_PER_M_INPUT_USD = 15.00  # Updateable constant
ANTHROPIC_STRONG_COST_PER_M_OUTPUT_USD = 75.00  # Updateable constant
DEFAULT_OPENAI_STRONG_MODEL = "gpt-5.5"  # Updateable constant (same as default for OpenAI)
OPENAI_STRONG_COST_PER_M_INPUT_USD = 2.50  # Updateable constant
OPENAI_STRONG_COST_PER_M_OUTPUT_USD = 15.00  # Updateable constant
DEFAULT_GEMINI_STRONG_MODEL = "gemini-2.5-pro"  # Updateable constant
GEMINI_STRONG_COST_PER_M_INPUT_USD = 1.25  # Updateable constant
GEMINI_STRONG_COST_PER_M_OUTPUT_USD = 10.00  # Updateable constant

# Ollama (local models) — zero cost, served via Ollama on localhost.
DEFAULT_OLLAMA_MODEL = "gemma4:latest"  # Updateable constant: uses whatever gemma4 tag is installed
DEFAULT_OLLAMA_CHEAP_MODEL = "gemma4:latest"  # Updateable constant: same model for cheap (local = free)
DEFAULT_OLLAMA_STRONG_MODEL = "gemma4:latest"  # Updateable constant: override with gemma4:31b if installed
OLLAMA_COST_PER_M_INPUT_USD = 0.0   # Local models are free
OLLAMA_COST_PER_M_OUTPUT_USD = 0.0

DEFAULT_RUN_BUDGET_USD = 0.0  # 0 = unlimited; set via --budget or MAIKE_DEFAULT_BUDGET_USD
DEFAULT_LLM_TEMPERATURE = 0.20
LOW_TEMPERATURE = 0.10
DETERMINISTIC_TEMPERATURE = 0.0


@dataclass(frozen=True)
class ModelPricing:
    input_per_million_usd: float
    output_per_million_usd: float


@dataclass(frozen=True)
class ProviderLLMConfig:
    provider_name: str
    default_model: str
    pricing: ModelPricing
    cheap_model: str | None = None
    cheap_pricing: ModelPricing | None = None
    strong_model: str | None = None
    strong_pricing: ModelPricing | None = None


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"  # Updateable constant: verify when upgrading models/pricing.

_log = logging.getLogger(__name__)

# ── YAML config loading ──────────────────────────────────────────
_BUNDLED_YAML = Path(__file__).parent / "models_default.yaml"
_USER_YAML = Path.home() / ".config" / "maike" / "models.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_model_config(
    bundled_path: Path = _BUNDLED_YAML,
    user_path: Path = _USER_YAML,
) -> dict[str, Any]:
    """Load model config from the bundled YAML, deep-merged with user overrides.

    Falls back to an empty dict when the bundled file is missing or unparseable
    (the caller is expected to fall back to hardcoded constants).
    """
    try:
        import yaml  # noqa: WPS433
    except ImportError:
        _log.debug("PyYAML not installed; falling back to hardcoded model config")
        return {}

    config: dict[str, Any] = {}
    if bundled_path.is_file():
        try:
            with open(bundled_path) as fh:
                raw = yaml.safe_load(fh)
                config = raw if isinstance(raw, dict) else {}
        except Exception:
            _log.warning("Failed to parse bundled models YAML at %s", bundled_path, exc_info=True)
            return {}
    else:
        _log.debug("Bundled models YAML not found at %s", bundled_path)
        return {}

    if user_path.is_file():
        try:
            with open(user_path) as fh:
                raw = yaml.safe_load(fh)
                user_config = raw if isinstance(raw, dict) else {}
            if user_config:
                config = _deep_merge(config, user_config)
        except Exception:
            _log.warning("Failed to parse user models YAML at %s; ignoring", user_path, exc_info=True)

    return config


def _pricing_from_dict(d: dict[str, Any]) -> ModelPricing:
    return ModelPricing(
        input_per_million_usd=float(d["input_per_million_usd"]),
        output_per_million_usd=float(d["output_per_million_usd"]),
    )


def _build_provider_config(name: str, prov: dict[str, Any]) -> ProviderLLMConfig:
    """Build a :class:`ProviderLLMConfig` from a parsed YAML provider block."""
    cheap_model = prov.get("cheap_model")
    cheap_pricing = _pricing_from_dict(prov["cheap_pricing"]) if prov.get("cheap_pricing") else None
    strong_model = prov.get("strong_model")
    strong_pricing = _pricing_from_dict(prov["strong_pricing"]) if prov.get("strong_pricing") else None
    return ProviderLLMConfig(
        provider_name=name,
        default_model=prov["default_model"],
        pricing=_pricing_from_dict(prov["pricing"]),
        cheap_model=cheap_model,
        cheap_pricing=cheap_pricing,
        strong_model=strong_model,
        strong_pricing=strong_pricing,
    )


def _hardcoded_provider_config() -> dict[str, ProviderLLMConfig]:
    """Return the hardcoded provider config (fallback when YAML is unavailable)."""
    return {
        "anthropic": ProviderLLMConfig(
            provider_name="anthropic",
            default_model=DEFAULT_ANTHROPIC_MODEL,
            pricing=ModelPricing(
                input_per_million_usd=COST_PER_M_INPUT_USD,
                output_per_million_usd=COST_PER_M_OUTPUT_USD,
            ),
            cheap_model=DEFAULT_ANTHROPIC_CHEAP_MODEL,
            cheap_pricing=ModelPricing(
                input_per_million_usd=ANTHROPIC_CHEAP_COST_PER_M_INPUT_USD,
                output_per_million_usd=ANTHROPIC_CHEAP_COST_PER_M_OUTPUT_USD,
            ),
            strong_model=DEFAULT_ANTHROPIC_STRONG_MODEL,
            strong_pricing=ModelPricing(
                input_per_million_usd=ANTHROPIC_STRONG_COST_PER_M_INPUT_USD,
                output_per_million_usd=ANTHROPIC_STRONG_COST_PER_M_OUTPUT_USD,
            ),
        ),
        "openai": ProviderLLMConfig(
            provider_name="openai",
            default_model=DEFAULT_OPENAI_MODEL,
            pricing=ModelPricing(
                input_per_million_usd=OPENAI_COST_PER_M_INPUT_USD,
                output_per_million_usd=OPENAI_COST_PER_M_OUTPUT_USD,
            ),
            cheap_model=DEFAULT_OPENAI_CHEAP_MODEL,
            cheap_pricing=ModelPricing(
                input_per_million_usd=OPENAI_CHEAP_COST_PER_M_INPUT_USD,
                output_per_million_usd=OPENAI_CHEAP_COST_PER_M_OUTPUT_USD,
            ),
            strong_model=DEFAULT_OPENAI_STRONG_MODEL,
            strong_pricing=ModelPricing(
                input_per_million_usd=OPENAI_STRONG_COST_PER_M_INPUT_USD,
                output_per_million_usd=OPENAI_STRONG_COST_PER_M_OUTPUT_USD,
            ),
        ),
        "gemini": ProviderLLMConfig(
            provider_name="gemini",
            default_model=DEFAULT_GEMINI_MODEL,
            pricing=ModelPricing(
                input_per_million_usd=GEMINI_COST_PER_M_INPUT_USD,
                output_per_million_usd=GEMINI_COST_PER_M_OUTPUT_USD,
            ),
            cheap_model=DEFAULT_GEMINI_CHEAP_MODEL,
            cheap_pricing=ModelPricing(
                input_per_million_usd=GEMINI_CHEAP_COST_PER_M_INPUT_USD,
                output_per_million_usd=GEMINI_CHEAP_COST_PER_M_OUTPUT_USD,
            ),
            strong_model=DEFAULT_GEMINI_STRONG_MODEL,
            strong_pricing=ModelPricing(
                input_per_million_usd=GEMINI_STRONG_COST_PER_M_INPUT_USD,
                output_per_million_usd=GEMINI_STRONG_COST_PER_M_OUTPUT_USD,
            ),
        ),
        "ollama": ProviderLLMConfig(
            provider_name="ollama",
            default_model=DEFAULT_OLLAMA_MODEL,
            pricing=ModelPricing(
                input_per_million_usd=OLLAMA_COST_PER_M_INPUT_USD,
                output_per_million_usd=OLLAMA_COST_PER_M_OUTPUT_USD,
            ),
            cheap_model=DEFAULT_OLLAMA_CHEAP_MODEL,
            cheap_pricing=ModelPricing(
                input_per_million_usd=OLLAMA_COST_PER_M_INPUT_USD,
                output_per_million_usd=OLLAMA_COST_PER_M_OUTPUT_USD,
            ),
            strong_model=DEFAULT_OLLAMA_STRONG_MODEL,
            strong_pricing=ModelPricing(
                input_per_million_usd=OLLAMA_COST_PER_M_INPUT_USD,
                output_per_million_usd=OLLAMA_COST_PER_M_OUTPUT_USD,
            ),
        ),
    }


def get_provider_llm_config(
    *,
    bundled_path: Path | None = None,
    user_path: Path | None = None,
) -> dict[str, ProviderLLMConfig]:
    """Load provider LLM config from YAML (bundled + user overrides).

    Falls back to hardcoded constants if YAML loading fails.
    The result is cached after the first successful call (with default paths).
    """
    bp = bundled_path if bundled_path is not None else _BUNDLED_YAML
    up = user_path if user_path is not None else _USER_YAML

    raw = load_model_config(bundled_path=bp, user_path=up)
    providers_raw = raw.get("providers")
    if not isinstance(providers_raw, dict) or not providers_raw:
        _log.debug("YAML providers block missing or empty; using hardcoded config")
        return _hardcoded_provider_config()

    try:
        result: dict[str, ProviderLLMConfig] = {}
        for name, prov in providers_raw.items():
            result[name] = _build_provider_config(name, prov)
        return result
    except Exception:
        _log.warning("Failed to build provider config from YAML; using hardcoded fallback", exc_info=True)
        return _hardcoded_provider_config()


# ── Cached singleton ─────────────────────────────────────────────
_provider_llm_config_cache: dict[str, ProviderLLMConfig] | None = None
_raw_yaml_config_cache: dict[str, Any] | None = None


def _get_cached_provider_llm_config() -> dict[str, ProviderLLMConfig]:
    global _provider_llm_config_cache  # noqa: PLW0603
    if _provider_llm_config_cache is None:
        _provider_llm_config_cache = get_provider_llm_config()
    return _provider_llm_config_cache


def _get_cached_raw_yaml_config() -> dict[str, Any]:
    global _raw_yaml_config_cache  # noqa: PLW0603
    if _raw_yaml_config_cache is None:
        _raw_yaml_config_cache = load_model_config()
    return _raw_yaml_config_cache


def _reset_provider_llm_config_cache() -> None:
    """Clear cached config (for testing)."""
    global _provider_llm_config_cache, _raw_yaml_config_cache  # noqa: PLW0603
    _provider_llm_config_cache = None
    _raw_yaml_config_cache = None


class _LazyDictProxy(dict):  # type: ignore[type-arg]
    """Base class for dict proxies that populate themselves on first access."""

    def _load(self) -> dict:  # type: ignore[type-arg]
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if not super().__len__():
            super().update(self._load())

    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_loaded()
        return super().__len__()

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return super().get(key, default)

    def values(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().values()

    def items(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().items()

    def keys(self):  # type: ignore[override]
        self._ensure_loaded()
        return super().keys()


class _ProviderConfigProxy(_LazyDictProxy):
    """Lazily loads provider config from YAML on first access."""

    def _load(self) -> dict[str, ProviderLLMConfig]:
        return _get_cached_provider_llm_config()


PROVIDER_LLM_CONFIG: dict[str, ProviderLLMConfig] = _ProviderConfigProxy()


def _build_pricing_map() -> dict[str, ModelPricing]:
    """Build model->pricing map covering default, cheap, and strong tiers."""
    result: dict[str, ModelPricing] = {}
    for config in PROVIDER_LLM_CONFIG.values():
        result[config.default_model] = config.pricing
        if config.cheap_model and config.cheap_pricing:
            result.setdefault(config.cheap_model, config.cheap_pricing)
        if config.strong_model and config.strong_pricing:
            result.setdefault(config.strong_model, config.strong_pricing)
    return result


def _build_extra_pricing() -> dict[str, ModelPricing]:
    """Extract extra_model_pricing entries from cached YAML config."""
    raw = _get_cached_raw_yaml_config()
    extra: dict[str, ModelPricing] = {}
    extra_raw = raw.get("extra_model_pricing")
    if isinstance(extra_raw, dict):
        for model_name, pricing_dict in extra_raw.items():
            try:
                extra[model_name] = _pricing_from_dict(pricing_dict)
            except Exception:
                _log.debug("Skipping invalid extra_model_pricing entry: %s", model_name)
    return extra


_HARDCODED_EXTRA_MODELS = {
    "gemini-2.0-flash": ModelPricing(
        input_per_million_usd=GEMINI_COST_PER_M_INPUT_USD,
        output_per_million_usd=GEMINI_COST_PER_M_OUTPUT_USD,
    ),
    "gemini-3-flash-preview": ModelPricing(
        input_per_million_usd=GEMINI_COST_PER_M_INPUT_USD,
        output_per_million_usd=GEMINI_COST_PER_M_OUTPUT_USD,
    ),
    # Gemini 3.1 preview models — pricing estimates pinned here until Google
    # publishes final values. Updateable constants: re-verify when promoted out of preview.
    "gemini-3.1-flash-lite-preview": ModelPricing(
        # Flash Lite is the cheap tier — estimated 2x cheaper than 2.5 Flash.
        input_per_million_usd=0.10,
        output_per_million_usd=0.40,
    ),
    "gemini-3.1-pro-preview": ModelPricing(
        # Pro tier — matches 2.5 Pro pricing as a conservative estimate.
        input_per_million_usd=GEMINI_STRONG_COST_PER_M_INPUT_USD,
        output_per_million_usd=GEMINI_STRONG_COST_PER_M_OUTPUT_USD,
    ),
}


class _PricingMapProxy(_LazyDictProxy):
    """Lazy-loading pricing map that builds itself from PROVIDER_LLM_CONFIG + extras."""

    def _load(self) -> dict[str, ModelPricing]:
        result = _build_pricing_map()
        extras = _build_extra_pricing()
        fallback = extras if extras else _HARDCODED_EXTRA_MODELS
        for k, v in fallback.items():
            result.setdefault(k, v)
        return result


MODEL_PRICING_USD_PER_MTOKEN: dict[str, ModelPricing] = _PricingMapProxy()


def default_model_for_provider(provider_name: str) -> str:
    config = PROVIDER_LLM_CONFIG.get(str(provider_name).lower())
    if config is None:
        return DEFAULT_MODEL
    return config.default_model


def cheap_model_for_provider(provider_name: str) -> str:
    """Return the cheap/lightweight model for *provider_name*, falling back to default."""
    config = PROVIDER_LLM_CONFIG.get(str(provider_name).lower())
    if config is None or config.cheap_model is None:
        return default_model_for_provider(provider_name)
    return config.cheap_model


def strong_model_for_provider(provider_name: str) -> str:
    """Return the strong/high-capability model for *provider_name*, falling back to default."""
    config = PROVIDER_LLM_CONFIG.get(str(provider_name).lower())
    if config is None or config.strong_model is None:
        return default_model_for_provider(provider_name)
    return config.strong_model


def model_for_tier(provider_name: str, tier: str) -> str:
    """Return the model for *provider_name* at the given *tier* ('default', 'cheap', 'strong')."""
    if tier == "cheap":
        return cheap_model_for_provider(provider_name)
    if tier == "strong":
        return strong_model_for_provider(provider_name)
    return default_model_for_provider(provider_name)


def pricing_for_model(model: str) -> ModelPricing | None:
    return MODEL_PRICING_USD_PER_MTOKEN.get(model)


DEFAULT_AGENT_TOKEN_BUDGET = 0  # Updateable constant — 0 = unlimited; set >0 to enforce
DISABLED_AGENT_TOKEN_BUDGET = 0
DEFAULT_AGENT_COST_BUDGET_USD = 0.0  # Updateable constant — 0 = unlimited; set >0 to enforce
DEFAULT_AGENT_MAX_ITERATIONS = 0  # 0 = unlimited (budget is the constraint); set >0 to enforce
DEFAULT_REACT_MAX_ITERATIONS = 200  # Safety cap for react agent loops; budget is the real constraint
DEFAULT_FACTORY_TOKEN_BUDGET = 50_000
DEFAULT_FACTORY_COST_BUDGET_USD = 1.00
DEFAULT_LLM_MAX_TOKENS = 8_096
DEFAULT_LLM_TIMEOUT_SECONDS = 120
DEFAULT_BASH_TOOL_TIMEOUT_SECONDS = 60

# Retry configuration  (Updateable constant)
LLM_RETRY_MAX_ATTEMPTS = 5
LLM_RETRY_BASE_DELAY_SECONDS = 2.0
LLM_RETRY_MAX_DELAY_SECONDS = 60.0
LLM_RETRY_JITTER_SECONDS = 1.0
LLM_RETRY_503_COOLDOWN_SECONDS = 45.0  # Extra wait after exhausting fast retries on 503
MAX_BASH_TOOL_TIMEOUT_SECONDS = 180
DEFAULT_READ_FILE_CHAR_LIMIT = 32_000
DEFAULT_READ_FILE_LINE_LIMIT = 2000
SESSION_BUDGET_PRECHECK_MARGIN = 0.95

MODEL_CONTEXT_LIMIT = 200_000
PRUNE_THRESHOLD = 150_000
PRUNE_CONTEXT_FRACTION = 0.40  # Prune when conversation reaches 40% of model context limit.
KEEP_RECENT_MESSAGES = 10

# Context budget enforcement
CONTEXT_BUDGET_SAFETY_MARGIN = 0.90  # Target 90% of context limit to leave room for format overhead.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # OpenAI
    "gpt-5.4": 128_000,
    "gpt-5.4-mini": 128_000,
    "gpt-5.5": 128_000,
    "gpt-5.5-mini": 128_000,
    # Gemini  (Updateable constant: verify when upgrading providers/models.)
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-3-flash-preview": 1_048_576,
    # Ollama (local models)
    "gemma4:e4b": 128_000,
    "gemma4:26b": 256_000,
    "gemma4:31b": 256_000,
    "gemma4:latest": 128_000,
    "llama3.1:latest": 128_000,
}


def context_limit_for_model(model: str) -> int:
    """Return the context window size for *model*, falling back to MODEL_CONTEXT_LIMIT."""
    return MODEL_CONTEXT_LIMITS.get(model, MODEL_CONTEXT_LIMIT)


def prune_threshold_for_model(model: str) -> int:
    """Model-aware prune trigger: 75% of context, capped at 800K.

    Gemini (1M) → 786K.  Claude (200K) → 150K.  GPT (128K) → 96K.
    Falls back to ``PRUNE_THRESHOLD`` for unknown models.
    """
    limit = context_limit_for_model(model)
    return min(int(limit * 0.75), 800_000)


def prune_fraction_for_model(model: str) -> float:
    """Target fraction of context to keep after pruning.

    Larger context windows can afford to keep a higher fraction:
      - ≥500K (Gemini): keep 60% → target ~470K tokens
      - ≥150K (Claude): keep 40% → target ~80K tokens (current behaviour)
      - <150K (GPT):    keep 35% → target ~33K tokens (leave headroom)
    """
    limit = context_limit_for_model(model)
    if limit >= 500_000:
        return 0.60
    if limit >= 150_000:
        return 0.40
    return 0.35


MAX_SPAWN_DEPTH = 2
MAX_CONCURRENT_AGENTS = 10
SESSION_AGENT_CAP = 25
SPAWN_BUDGET_CAP_USD = 1.00

# Delegation constants
DELEGATION_MAX_DEPTH = 1  # Sub-agents cannot delegate further
DELEGATION_BUDGET_FRACTION = 0.30  # Sub-agent gets 30% of parent's remaining budget
DELEGATION_BUDGET_CAP_USD = 1.00  # Hard cap on delegation budget
DELEGATION_TOKEN_BUDGET = 500_000  # Token budget for delegated agents
DELEGATION_MAX_ITERATIONS = 30  # Iteration cap for delegated agents

# Agent definition directories
AGENTS_USER_DIR = Path.home() / ".config" / "maike" / "agents"
AGENTS_PROJECT_SUBDIR = ".maike/agents"

# Per-agent memory directories (+ /{agent-name} appended at runtime)
AGENT_MEMORY_USER_DIR = Path.home() / ".config" / "maike" / "agent-memory"
AGENT_MEMORY_PROJECT_SUBDIR = ".maike/agent-memory"

# Team definition directories
TEAMS_USER_DIR = Path.home() / ".config" / "maike" / "teams"
TEAMS_PROJECT_SUBDIR = ".maike/teams"
MAX_TEAM_MEMBERS = 5
TEAM_SYNTHESIS_BUDGET_FRACTION = 0.15
TEAM_BUDGET_CAP_USD = 3.00

DEFAULT_DB_RELATIVE_PATH = ".maike/session.db"
DEFAULT_VECTOR_STORE_RELATIVE_PATH = ".maike/vector_store.json"

MAX_PRUNED_EVENTS = 48
TOOL_RESULT_TRUNCATE_LIMIT = 200

# Phase 5 — Improved pruning
DESIGN_DECISION_KEYWORDS: frozenset[str] = frozenset({
    "decided", "chose", "because", "trade-off", "approach",
    "design", "architecture", "pattern", "interface", "contract",
})
VARIABLE_RECENT_WINDOW_MIN = 10
VARIABLE_RECENT_WINDOW_MAX = 30
VARIABLE_RECENT_WINDOW_FRACTION = 0.15

# Code intelligence
CODE_INDEX_MAX_FILES = 5_000
CODE_INDEX_MAX_FILE_SIZE = 500_000
SMART_REPO_MAP_CAP = 12_000
FIND_SYMBOL_MAX_RESULTS = 30
FIND_REFERENCES_MAX_RESULTS = 50
SEMANTIC_SEARCH_MAX_RESULTS = 15
HOT_CONTEXT_BUDGET_FRACTION = 0.15

# ── Skill & Plugin directories ────────────────────────────────────
SKILL_USER_DIR = Path.home() / ".config" / "maike" / "skills"
PLUGIN_USER_DIR = Path.home() / ".config" / "maike" / "plugins"
SKILL_PROJECT_SUBDIR = ".maike/skills"    # relative to workspace
PLUGIN_PROJECT_SUBDIR = ".maike/plugins"  # relative to workspace
MAX_MID_SESSION_SKILLS = 2
MAX_ASYNC_DELEGATES = 5  # max concurrent async delegates per parent

# ── Advisor pattern ──────────────────────────────────────────────
# Local/cheap executor + frontier advisor. Opt-in via --advisor flag.
# Advisor auto-selects strong tier of the same provider unless
# --advisor-provider/--advisor-model overrides are set.
ADVISOR_ENABLED_DEFAULT = False
ADVISOR_BUDGET_FRACTION_DEFAULT = 0.20   # advisor spend capped at 20% of session budget
ADVISOR_MAX_CALLS_PER_SESSION = 10       # hard cap on number of advisor calls
ADVISOR_COOLDOWN_ITERATIONS = 3          # min iterations between advisor calls
ADVISOR_TRANSCRIPT_MAX_CHARS = 12_000    # total chars sent to advisor (summary + tail)
ADVISOR_RECENT_MESSAGES_CHARS = 6_000    # raw recent conversation tail included verbatim
ADVISOR_EXPLORATION_THRESHOLD = 2        # read/grep calls before after_exploration trigger fires
# before_first_edit is purely behavioral (no task-text heuristics).
# Kept as historical docs — values unused.  Removed in a future release.
ADVISOR_PLANNING_KEYWORDS: frozenset[str] = frozenset()
ADVISOR_PLANNING_MIN_TASK_CHARS = 0
ADVISOR_MAX_OUTPUT_TOKENS = 3000         # advisor output cap — big enough to absorb Gemini-style
                                         # thinking tokens AND leave room for the short visible advice
ADVISOR_TEMPERATURE = 0.3                # advisor temperature — slightly warmer for judgment

# ── Session Memory (live structured notes) ───────────────────────
SESSION_MEMORY_INIT_THRESHOLD = 30_000  # tokens before first memory update
SESSION_MEMORY_UPDATE_INTERVAL = 5      # tool calls between memory updates
SESSION_MEMORY_TOKEN_THRESHOLD = 15_000  # tokens-since-last-update trigger
SESSION_MEMORY_MAX_TOKENS = 4_000       # max tokens in session memory file
SESSION_MEMORY_FILENAME = "session_memory.md"  # relative to .maike/

# ── Post-Compaction Recovery ─────────────────────────────────────
POST_COMPACT_MAX_FILES = 5              # max recently-read files to re-inject
POST_COMPACT_TOKEN_BUDGET = 30_000      # total budget for re-injected files
POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000  # per-file cap

# ── Settings & external agent paths ────────────────────────────────
SETTINGS_PATH = Path.home() / ".config" / "maike" / "settings.json"

# ── MCP server defaults ────────────────────────────────────────────
MCP_USER_CONFIG = Path.home() / ".config" / "maike" / ".mcp.json"
MCP_PROJECT_CONFIG_NAME = ".mcp.json"  # relative to workspace
MCP_TOOL_CALL_TIMEOUT_S = 30
MCP_SERVER_START_TIMEOUT_S = 10

# ── LSP server defaults ───────────────────────────────────────────
LSP_PROJECT_CONFIG_NAME = ".lsp.json"  # relative to workspace

# ── Adaptive model selection ─────────────────────────────────────
ADAPTIVE_MODEL_ENABLED = True  # Updateable constant: toggle adaptive tier routing
ADAPTIVE_CHEAP_MAX_MESSAGES = 5  # Max conversation messages for "cheap" tier eligibility
ADAPTIVE_STRONG_FAILURE_THRESHOLD = 3  # Consecutive failures before escalating to "strong"
ADAPTIVE_STRONG_ITERATION_RATIO = 0.7  # Fraction of max_iterations that triggers "strong"

# Provider fallback order — when a provider fails with a non-retryable error (401, 403,
# ConnectionRefusedError), try the next available provider in this order.
PROVIDER_FALLBACK_ORDER: list[str] = ["anthropic", "openai", "gemini"]

# Env-var names used to detect whether a provider's API key is configured.
PROVIDER_API_KEY_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


# ── Runtime model config mutation ────────────────────────────────────

def _apply_model_config(data: dict) -> None:
    """Merge parsed YAML *data* into the module-level config maps.

    Useful for runtime overrides (e.g. workspace-local ``models.yaml``).
    Uses the same ``input_per_million_usd`` / ``output_per_million_usd`` keys
    as the bundled YAML and ``_pricing_from_dict``.
    """
    providers = data.get("providers", {})
    for provider_name, pconfig in providers.items():
        provider_name = str(provider_name).lower()
        existing = PROVIDER_LLM_CONFIG.get(provider_name)
        if existing is None:
            continue

        new_default = pconfig.get("default_model", existing.default_model)
        new_cheap = pconfig.get("cheap_model", existing.cheap_model)
        new_strong = pconfig.get("strong_model", existing.strong_model)

        new_pricing = existing.pricing
        new_cheap_pricing = existing.cheap_pricing
        new_strong_pricing = existing.strong_pricing

        # Accept both top-level pricing block and per-tier blocks
        if "pricing" in pconfig and isinstance(pconfig["pricing"], dict):
            try:
                new_pricing = _pricing_from_dict(pconfig["pricing"])
            except (KeyError, TypeError, ValueError):
                pass
        if "cheap_pricing" in pconfig and isinstance(pconfig["cheap_pricing"], dict):
            try:
                new_cheap_pricing = _pricing_from_dict(pconfig["cheap_pricing"])
            except (KeyError, TypeError, ValueError):
                pass
        if "strong_pricing" in pconfig and isinstance(pconfig["strong_pricing"], dict):
            try:
                new_strong_pricing = _pricing_from_dict(pconfig["strong_pricing"])
            except (KeyError, TypeError, ValueError):
                pass

        PROVIDER_LLM_CONFIG[provider_name] = ProviderLLMConfig(
            provider_name=provider_name,
            default_model=new_default,
            pricing=new_pricing,
            cheap_model=new_cheap,
            cheap_pricing=new_cheap_pricing,
            strong_model=new_strong,
            strong_pricing=new_strong_pricing,
        )

        # Update the global pricing map (strong/cheap last so they override
        # the default entry when the model names differ; skip when same as
        # default to avoid clobbering the updated default pricing).
        if new_default:
            MODEL_PRICING_USD_PER_MTOKEN[new_default] = new_pricing
        if new_cheap and new_cheap_pricing and new_cheap != new_default:
            MODEL_PRICING_USD_PER_MTOKEN[new_cheap] = new_cheap_pricing
        if new_strong and new_strong_pricing and new_strong != new_default:
            MODEL_PRICING_USD_PER_MTOKEN[new_strong] = new_strong_pricing

    # Context limits
    for model_name, limit in data.get("context_limits", {}).items():
        MODEL_CONTEXT_LIMITS[model_name] = int(limit)
