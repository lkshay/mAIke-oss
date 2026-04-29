"""Tests for runtime model configuration mutation via _apply_model_config."""

from maike.constants import (
    ModelPricing,
    PROVIDER_LLM_CONFIG,
    MODEL_PRICING_USD_PER_MTOKEN,
    MODEL_CONTEXT_LIMITS,
    _apply_model_config,
    pricing_for_model,
    context_limit_for_model,
)


def test_apply_model_config_updates_pricing():
    """Applying config should update pricing for known providers."""
    original = pricing_for_model("claude-sonnet-4-6")
    assert original is not None

    config = {
        "providers": {
            "anthropic": {
                "pricing": {
                    "input_per_million_usd": 10.00,
                    "output_per_million_usd": 50.00,
                },
            },
        },
    }
    _apply_model_config(config)

    updated = pricing_for_model("claude-sonnet-4-6")
    assert updated is not None
    assert updated.input_per_million_usd == 10.00
    assert updated.output_per_million_usd == 50.00

    # Restore original
    MODEL_PRICING_USD_PER_MTOKEN["claude-sonnet-4-6"] = original
    # Also restore provider config
    existing = PROVIDER_LLM_CONFIG["anthropic"]
    PROVIDER_LLM_CONFIG["anthropic"] = existing.__class__(
        provider_name=existing.provider_name,
        default_model=existing.default_model,
        pricing=original,
        cheap_model=existing.cheap_model,
        cheap_pricing=existing.cheap_pricing,
        strong_model=existing.strong_model,
        strong_pricing=existing.strong_pricing,
    )


def test_apply_model_config_adds_new_model():
    """Config can update the default model for a provider."""
    original_config = PROVIDER_LLM_CONFIG.get("anthropic")
    assert original_config is not None
    original_pricing = MODEL_PRICING_USD_PER_MTOKEN.get(original_config.default_model)

    config = {
        "providers": {
            "anthropic": {
                "default_model": "claude-test-model",
                "pricing": {
                    "input_per_million_usd": 1.00,
                    "output_per_million_usd": 5.00,
                },
            },
        },
    }
    _apply_model_config(config)
    new_pricing = pricing_for_model("claude-test-model")
    assert new_pricing is not None
    assert new_pricing.input_per_million_usd == 1.00

    # Cleanup — restore
    PROVIDER_LLM_CONFIG["anthropic"] = original_config
    MODEL_PRICING_USD_PER_MTOKEN.pop("claude-test-model", None)
    if original_pricing:
        MODEL_PRICING_USD_PER_MTOKEN[original_config.default_model] = original_pricing


def test_apply_context_limits():
    config = {
        "context_limits": {
            "test-model-ctx": 500_000,
        },
    }
    _apply_model_config(config)
    assert context_limit_for_model("test-model-ctx") == 500_000

    # Cleanup
    MODEL_CONTEXT_LIMITS.pop("test-model-ctx", None)


def test_unknown_provider_ignored():
    """Unknown providers in config should not crash."""
    config = {
        "providers": {
            "unknown_provider": {
                "default_model": "foo",
                "pricing": {"input_per_million_usd": 1.0, "output_per_million_usd": 2.0},
            },
        },
    }
    _apply_model_config(config)  # Should not raise
