"""Helpers for provider-native structured LLM output.

The partition planner (item #4 in .claude/handoff-2026-04-17-open-items.md)
drove the initial need: we wanted to remove the last ``json.loads`` on LLM
output in the live execution path.  Rather than parse free-text JSON from
the model and hope for the best, we ask each provider to guarantee valid
JSON via their native structured-output API:

  - Gemini:   ``GenerateContentConfig(response_schema=..., response_mime_type="application/json")``
  - OpenAI:   ``response_format={"type":"json_schema", "json_schema":{...}}``
  - Anthropic: forced tool-use with the schema as ``input_schema``
  - Ollama:   ``response_format={"type":"json_object"}`` (best-effort; no
              schema enforcement at the provider)

Callers pass a Pydantic model class via ``LLMRequest.response_schema``.
Each adapter does the provider-specific translation and populates
``LLMResult.parsed`` with a dict.  Consumers then call
``MyModel.model_validate(result.parsed)`` for full type safety.

Why not parse into the Pydantic object directly in the adapter?  Keeps
``maike.atoms.llm.LLMResult`` free of upstream Pydantic-user imports; the
result object needs to be serialisable through the tracing layer, and
pydantic instances don't always round-trip cleanly (e.g. models with
forward refs that aren't importable in the tracing process).
"""

from __future__ import annotations

from typing import Any


def schema_dict(model_cls: type) -> dict[str, Any]:
    """Return the JSON Schema for a Pydantic model as a plain dict.

    Strips unsupported ``$defs`` indirections so providers that handle
    references inconsistently (looking at you, Gemini) see a flat
    schema.  For models that don't use ``$ref`` internally this is a
    no-op beyond the ``ref_template`` parameter.
    """
    # model_json_schema() is Pydantic v2 API; caller is expected to pass
    # a ``BaseModel`` subclass.  We don't import pydantic here to keep
    # this module side-effect-free in non-LLM contexts.
    if not hasattr(model_cls, "model_json_schema"):
        raise TypeError(
            f"schema_dict() expected a Pydantic BaseModel subclass, got "
            f"{model_cls!r}"
        )
    return model_cls.model_json_schema(
        ref_template="#/definitions/{model}",
    )


def schema_name(model_cls: type) -> str:
    """Return a stable schema identifier for use in provider APIs.

    Uses the model class's own ``__name__`` — good enough for OpenAI
    ``json_schema.name`` and Anthropic tool names.
    """
    return getattr(model_cls, "__name__", "structured_output")
