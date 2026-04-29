from __future__ import annotations

from pathlib import Path

from maike.constants import (
    COST_PER_M_INPUT_USD,
    COST_PER_M_OUTPUT_USD,
    DEFAULT_ANTHROPIC_CHEAP_MODEL,
    DEFAULT_ANTHROPIC_STRONG_MODEL,
    DEFAULT_GEMINI_CHEAP_MODEL,
    DEFAULT_GEMINI_STRONG_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_CHEAP_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_STRONG_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_RUN_BUDGET_USD,
    DETERMINISTIC_TEMPERATURE,
    LOW_TEMPERATURE,
    MAX_PRUNED_EVENTS,
    TOOL_RESULT_TRUNCATE_LIMIT,
    ModelPricing,
    ProviderLLMConfig,
    cheap_model_for_provider,
    default_model_for_provider,
    model_for_tier,
    pricing_for_model,
    strong_model_for_provider,
)


def test_default_model_and_pricing_constants_are_pinned():
    assert DEFAULT_PROVIDER == "gemini"
    assert DEFAULT_MODEL == "gemini-3-flash-preview"
    assert DEFAULT_OPENAI_MODEL == "gpt-5.5"
    assert DEFAULT_GEMINI_MODEL == "gemini-3-flash-preview"
    assert COST_PER_M_INPUT_USD == 15.00
    assert COST_PER_M_OUTPUT_USD == 75.00
    assert DEFAULT_RUN_BUDGET_USD == 0.0  # unlimited by default
    assert DEFAULT_LLM_TEMPERATURE == 0.20
    assert LOW_TEMPERATURE == 0.10
    assert DETERMINISTIC_TEMPERATURE == 0.0
    assert MAX_PRUNED_EVENTS == 48
    assert TOOL_RESULT_TRUNCATE_LIMIT == 200


def test_provider_defaults_and_pricing_are_resolved_from_shared_config():
    from maike.constants import DEFAULT_ANTHROPIC_MODEL
    anthropic_pricing = pricing_for_model(DEFAULT_ANTHROPIC_MODEL)
    openai_pricing = pricing_for_model(DEFAULT_OPENAI_MODEL)
    gemini_pricing = pricing_for_model(DEFAULT_GEMINI_MODEL)

    assert default_model_for_provider("anthropic") == DEFAULT_ANTHROPIC_MODEL
    assert default_model_for_provider("openai") == DEFAULT_OPENAI_MODEL
    assert default_model_for_provider("gemini") == DEFAULT_GEMINI_MODEL
    assert anthropic_pricing.input_per_million_usd == COST_PER_M_INPUT_USD
    assert anthropic_pricing.output_per_million_usd == COST_PER_M_OUTPUT_USD
    assert openai_pricing is not None
    assert gemini_pricing is not None


def test_cheap_model_for_provider_returns_correct_models():
    assert cheap_model_for_provider("anthropic") == DEFAULT_ANTHROPIC_CHEAP_MODEL
    assert cheap_model_for_provider("openai") == DEFAULT_OPENAI_CHEAP_MODEL
    assert cheap_model_for_provider("gemini") == DEFAULT_GEMINI_CHEAP_MODEL


def test_strong_model_for_provider_returns_correct_models():
    assert strong_model_for_provider("anthropic") == DEFAULT_ANTHROPIC_STRONG_MODEL
    assert strong_model_for_provider("openai") == DEFAULT_OPENAI_STRONG_MODEL
    assert strong_model_for_provider("gemini") == DEFAULT_GEMINI_STRONG_MODEL


def test_model_for_tier_dispatches_correctly():
    from maike.constants import DEFAULT_ANTHROPIC_MODEL
    assert model_for_tier("anthropic", "default") == DEFAULT_ANTHROPIC_MODEL
    assert model_for_tier("anthropic", "cheap") == DEFAULT_ANTHROPIC_CHEAP_MODEL
    assert model_for_tier("anthropic", "strong") == DEFAULT_ANTHROPIC_STRONG_MODEL
    assert model_for_tier("openai", "cheap") == DEFAULT_OPENAI_CHEAP_MODEL
    assert model_for_tier("gemini", "strong") == DEFAULT_GEMINI_STRONG_MODEL


def test_tier_helpers_fallback_for_unknown_provider():
    assert cheap_model_for_provider("unknown") == DEFAULT_MODEL
    assert strong_model_for_provider("unknown") == DEFAULT_MODEL
    assert model_for_tier("unknown", "cheap") == DEFAULT_MODEL


def test_pricing_map_includes_all_tier_models():
    assert pricing_for_model(DEFAULT_ANTHROPIC_CHEAP_MODEL) is not None
    assert pricing_for_model(DEFAULT_ANTHROPIC_STRONG_MODEL) is not None
    assert pricing_for_model(DEFAULT_OPENAI_CHEAP_MODEL) is not None
    assert pricing_for_model(DEFAULT_OPENAI_STRONG_MODEL) is not None
    assert pricing_for_model(DEFAULT_GEMINI_CHEAP_MODEL) is not None
    assert pricing_for_model(DEFAULT_GEMINI_STRONG_MODEL) is not None


# ── YAML config loading tests ───────────────────────────────────


def test_load_bundled_default_yaml():
    """The bundled models_default.yaml loads and contains all three providers."""
    from maike.constants import _BUNDLED_YAML, load_model_config

    config = load_model_config(bundled_path=_BUNDLED_YAML, user_path=Path("/nonexistent"))
    assert "providers" in config
    providers = config["providers"]
    assert "anthropic" in providers
    assert "openai" in providers
    assert "gemini" in providers
    # Check structure of one provider
    anthropic = providers["anthropic"]
    assert "default_model" in anthropic
    assert "pricing" in anthropic
    assert anthropic["pricing"]["input_per_million_usd"] == 15.00
    assert anthropic["pricing"]["output_per_million_usd"] == 75.00


def test_user_override_merges_correctly(tmp_path: Path):
    """User YAML overrides are deep-merged into bundled defaults."""
    from maike.constants import _BUNDLED_YAML, load_model_config

    user_yaml = tmp_path / "models.yaml"
    user_yaml.write_text(
        "providers:\n"
        "  anthropic:\n"
        "    pricing:\n"
        "      input_per_million_usd: 99.99\n"
        "  new_provider:\n"
        "    default_model: new-model-v1\n"
        "    pricing:\n"
        "      input_per_million_usd: 1.00\n"
        "      output_per_million_usd: 2.00\n"
    )
    config = load_model_config(bundled_path=_BUNDLED_YAML, user_path=user_yaml)
    providers = config["providers"]

    # Overridden value
    assert providers["anthropic"]["pricing"]["input_per_million_usd"] == 99.99
    # Non-overridden value preserved from bundled defaults
    assert providers["anthropic"]["pricing"]["output_per_million_usd"] == 75.00
    # Original provider still present
    assert "openai" in providers
    # New provider added by user
    assert providers["new_provider"]["default_model"] == "new-model-v1"


def test_fallback_to_hardcoded_when_yaml_missing(tmp_path: Path):
    """When both bundled and user YAML are missing, hardcoded fallback is used."""
    from maike.constants import get_provider_llm_config

    config = get_provider_llm_config(
        bundled_path=tmp_path / "nonexistent.yaml",
        user_path=tmp_path / "also_nonexistent.yaml",
    )
    assert "anthropic" in config
    assert "openai" in config
    assert "gemini" in config
    # Verify values match the hardcoded constants
    assert config["anthropic"].pricing.input_per_million_usd == COST_PER_M_INPUT_USD


def test_fallback_when_yaml_is_malformed(tmp_path: Path):
    """Malformed YAML triggers fallback to hardcoded config."""
    from maike.constants import get_provider_llm_config

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":::not valid yaml{{{}}}][")
    config = get_provider_llm_config(
        bundled_path=bad_yaml,
        user_path=tmp_path / "nonexistent.yaml",
    )
    assert "anthropic" in config
    assert isinstance(config["anthropic"], ProviderLLMConfig)


def test_get_provider_llm_config_returns_valid_objects():
    """get_provider_llm_config() returns ProviderLLMConfig for each provider."""
    from maike.constants import get_provider_llm_config

    config = get_provider_llm_config()
    for name, prov_config in config.items():
        assert isinstance(prov_config, ProviderLLMConfig), f"{name} is not ProviderLLMConfig"
        assert prov_config.provider_name == name
        assert isinstance(prov_config.pricing, ModelPricing)
        assert prov_config.pricing.input_per_million_usd >= 0  # 0 is valid (local models)
        assert prov_config.pricing.output_per_million_usd >= 0
        if prov_config.cheap_model is not None:
            assert isinstance(prov_config.cheap_pricing, ModelPricing)
        if prov_config.strong_model is not None:
            assert isinstance(prov_config.strong_pricing, ModelPricing)


def test_get_provider_llm_config_with_user_overrides(tmp_path: Path):
    """User YAML overrides propagate into ProviderLLMConfig objects."""
    from maike.constants import _BUNDLED_YAML, get_provider_llm_config

    user_yaml = tmp_path / "models.yaml"
    user_yaml.write_text(
        "providers:\n"
        "  anthropic:\n"
        "    default_model: custom-model-v2\n"
        "    pricing:\n"
        "      input_per_million_usd: 42.00\n"
        "      output_per_million_usd: 84.00\n"
    )
    config = get_provider_llm_config(
        bundled_path=_BUNDLED_YAML,
        user_path=user_yaml,
    )
    assert config["anthropic"].default_model == "custom-model-v2"
    assert config["anthropic"].pricing.input_per_million_usd == 42.00
    assert config["anthropic"].pricing.output_per_million_usd == 84.00
    # Other providers still present from bundled
    assert "openai" in config
    assert "gemini" in config


def test_deep_merge_utility():
    """_deep_merge performs recursive dict merging."""
    from maike.constants import _deep_merge

    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99, "z": 100}, "c": 4}
    result = _deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3, "c": 4}
    # Original dicts are not mutated
    assert base["a"]["y"] == 2


def test_bundled_yaml_loads_context_limits():
    """Context limits from the bundled YAML are loaded correctly."""
    from maike.constants import _BUNDLED_YAML, load_model_config

    config = load_model_config(bundled_path=_BUNDLED_YAML, user_path=Path("/nonexistent"))
    context_limits = config.get("context_limits", {})
    assert "gemini-2.5-flash" in context_limits
    assert context_limits["gemini-2.5-flash"] == 1048576
    assert "gemini-3-flash-preview" in context_limits


def test_proxy_dict_behaves_like_regular_dict():
    """PROVIDER_LLM_CONFIG proxy supports standard dict operations."""
    from maike.constants import PROVIDER_LLM_CONFIG

    # len
    assert len(PROVIDER_LLM_CONFIG) >= 3
    # __contains__
    assert "anthropic" in PROVIDER_LLM_CONFIG
    assert "nonexistent" not in PROVIDER_LLM_CONFIG
    # .get() with default
    assert PROVIDER_LLM_CONFIG.get("nonexistent") is None
    assert PROVIDER_LLM_CONFIG.get("nonexistent", "fallback") == "fallback"
    # iteration
    keys = list(PROVIDER_LLM_CONFIG.keys())
    assert "anthropic" in keys
    # values and items
    assert len(list(PROVIDER_LLM_CONFIG.values())) >= 3
    assert len(list(PROVIDER_LLM_CONFIG.items())) >= 3
