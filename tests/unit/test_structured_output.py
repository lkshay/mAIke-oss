"""Tests for provider-native structured LLM output (item #4).

Each adapter maps ``LLMRequest.response_schema`` (a Pydantic model class)
to its provider's native structured-output API.  On the happy path,
``LLMResult.parsed`` is populated with the parsed dict.

These tests use fake clients — no network, no real LLM, no API keys.
They verify:
  - The adapter passes the right-shaped kwarg to the SDK (schema plumbing).
  - The result's ``parsed`` field is populated from a valid JSON response.
  - Graceful degradation when the response isn't parseable.

``PartitionPlan`` from ``maike.orchestrator.partition_schema`` is used
as the representative schema throughout — it's the first real consumer
of this infrastructure and covers the list-of-objects case that's
tricky for some providers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from maike.gateway.providers import (
    AnthropicAdapter,
    GeminiAdapter,
    LLMRequest,
    OllamaAdapter,
    OpenAIAdapter,
)
from maike.gateway.structured import schema_dict, schema_name
from maike.orchestrator.partition_schema import PartitionPlan


# ---------------------------------------------------------------------------
# schema_dict / schema_name helpers
# ---------------------------------------------------------------------------


class TestSchemaHelpers:
    def test_schema_dict_returns_json_schema_for_pydantic(self):
        result = schema_dict(PartitionPlan)
        assert isinstance(result, dict)
        # Pydantic JSON schema always carries type + properties.
        assert "properties" in result
        assert "partitions" in result["properties"]

    def test_schema_name_uses_class_name(self):
        assert schema_name(PartitionPlan) == "PartitionPlan"

    def test_schema_dict_rejects_non_pydantic(self):
        class NotAModel:
            pass
        with pytest.raises(TypeError):
            schema_dict(NotAModel)


# ---------------------------------------------------------------------------
# LLMRequest / LLMResult fields
# ---------------------------------------------------------------------------


class TestLLMRequestResponseSchemaField:
    def test_response_schema_defaults_to_none(self):
        req = LLMRequest(
            system="", messages=[], tools=[],
            model="x", temperature=0.0, max_tokens=10,
            timeout_seconds=30,
        )
        assert req.response_schema is None

    def test_response_schema_accepts_pydantic_cls(self):
        req = LLMRequest(
            system="", messages=[], tools=[],
            model="x", temperature=0.0, max_tokens=10,
            timeout_seconds=30, response_schema=PartitionPlan,
        )
        assert req.response_schema is PartitionPlan


def test_llm_result_parsed_defaults_to_none():
    from maike.atoms.llm import LLMResult
    r = LLMResult(model="x")
    assert r.parsed is None


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------


class TestGeminiStructuredOutput:
    def test_build_config_sets_response_schema_and_drops_tools(self):
        adapter = GeminiAdapter()
        req = LLMRequest(
            system="sys", messages=[],
            tools=[{"name": "foo", "input_schema": {}}],  # intentionally present
            model="gemini-3-flash-preview",
            temperature=0.0, max_tokens=512, timeout_seconds=30,
            response_schema=PartitionPlan,
        )
        config = adapter._build_config(req)
        # With genai_types available, config is a GenerateContentConfig.
        # Without it (fallback path), it's a dict.  Handle both.
        if isinstance(config, dict):
            assert config.get("response_schema") is not None
            assert config.get("response_mime_type") == "application/json"
            assert config.get("tools") in (None, [])
        else:
            rs = getattr(config, "response_schema", None)
            mime = getattr(config, "response_mime_type", None)
            tools = getattr(config, "tools", None)
            assert rs is not None
            assert mime == "application/json"
            assert not tools  # schema path drops tools

    def test_build_config_no_schema_keeps_tools(self):
        adapter = GeminiAdapter()
        req = LLMRequest(
            system="sys", messages=[], tools=[],
            model="gemini-3-flash-preview",
            temperature=0.0, max_tokens=512, timeout_seconds=30,
        )
        config = adapter._build_config(req)
        if isinstance(config, dict):
            assert "response_schema" not in config
        else:
            rs = getattr(config, "response_schema", None)
            mime = getattr(config, "response_mime_type", None)
            assert rs is None
            assert mime is None

    def test_parse_response_populates_parsed_on_structured_path(self):
        """Simulate the adapter flow that populates LLMResult.parsed when
        response_schema was set.  We can't easily invoke create() end-to-end
        without a live client, but we can exercise the parse helper + the
        post-parse model_copy shim by constructing the result path directly."""
        import json

        adapter = GeminiAdapter()
        # Response with JSON content that matches PartitionPlan shape.
        plan_json = json.dumps({
            "partitions": [
                {"subtask": "implement A", "files": ["a.py"]},
                {"subtask": "implement B", "files": ["b.py"]},
            ],
        })
        fake_response = SimpleNamespace(
            candidates=[SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(parts=[
                    SimpleNamespace(text=plan_json, thought=False),
                ]),
            )],
            text=None,
            function_calls=[],
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
        )
        result = adapter._parse_response(fake_response, model="gemini-3-flash-preview")
        # _parse_response does not populate .parsed — that happens in create().
        # But the text IS the JSON, so we can verify the round-trip by parsing.
        assert result.content == plan_json
        parsed = json.loads(result.content)
        plan = PartitionPlan.model_validate(parsed)
        assert len(plan.partitions) == 2
        assert plan.partitions[0].files == ["a.py"]


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class _FakeOpenAIClient:
    """Captures the kwargs passed to responses.create so tests can assert
    schema plumbing without talking to the real API."""

    def __init__(self, response: Any):
        self._response = response
        self.last_kwargs: dict[str, Any] = {}
        self.responses = self  # client.responses.create(...) ↦ self.create

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class TestOpenAIStructuredOutput:
    def test_create_passes_text_format_json_schema(self):
        import asyncio
        import json

        plan_json = json.dumps({
            "partitions": [{"subtask": "a", "files": ["a.py"]}, {"subtask": "b", "files": ["b.py"]}],
        })
        # OpenAI Responses API response shape: output = [{type:"message", content:[{type:"output_text", text:"..."}]}]
        fake_response = SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=plan_json)],
            )],
            output_text=plan_json,
            status="completed",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
        client = _FakeOpenAIClient(fake_response)
        adapter = OpenAIAdapter(client=client)
        req = LLMRequest(
            system="sys",
            messages=[{"role": "user", "content": "plan it"}],
            tools=[],
            model="gpt-5-mini",
            temperature=0.0, max_tokens=2000, timeout_seconds=30,
            response_schema=PartitionPlan,
        )
        result = asyncio.run(adapter.create(req))

        # Plumbing assertions: schema was passed via text.format.json_schema.
        text_kwarg = client.last_kwargs.get("text")
        assert isinstance(text_kwarg, dict)
        fmt = text_kwarg.get("format") or {}
        assert fmt.get("type") == "json_schema"
        assert fmt.get("name") == "PartitionPlan"
        assert fmt.get("strict") is True
        assert "properties" in (fmt.get("schema") or {})
        # Tools dropped.
        assert client.last_kwargs.get("tools") is None

        # Parsed dict populated from valid JSON text.
        assert result.parsed is not None
        plan = PartitionPlan.model_validate(result.parsed)
        assert len(plan.partitions) == 2

    def test_create_without_schema_does_not_set_text_kwarg(self):
        import asyncio

        fake_response = SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="hello")],
            )],
            output_text="hello",
            status="completed",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        client = _FakeOpenAIClient(fake_response)
        adapter = OpenAIAdapter(client=client)
        req = LLMRequest(
            system="sys", messages=[{"role": "user", "content": "hi"}],
            tools=[], model="gpt-5-mini",
            temperature=0.0, max_tokens=10, timeout_seconds=30,
        )
        result = asyncio.run(adapter.create(req))
        assert "text" not in client.last_kwargs
        assert result.parsed is None

    def test_create_degrades_gracefully_on_invalid_json(self):
        """Provider should populate parsed=None if the text doesn't parse."""
        import asyncio

        fake_response = SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="not-json-at-all")],
            )],
            output_text="not-json-at-all",
            status="completed",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        client = _FakeOpenAIClient(fake_response)
        adapter = OpenAIAdapter(client=client)
        req = LLMRequest(
            system="sys", messages=[{"role": "user", "content": "x"}],
            tools=[], model="gpt-5-mini",
            temperature=0.0, max_tokens=10, timeout_seconds=30,
            response_schema=PartitionPlan,
        )
        result = asyncio.run(adapter.create(req))
        assert result.parsed is None


# ---------------------------------------------------------------------------
# Anthropic adapter (forced tool-use)
# ---------------------------------------------------------------------------


class _FakeAnthropicClient:
    """Mimics client.messages.create(...)."""

    def __init__(self, response: Any):
        self._response = response
        self.last_kwargs: dict[str, Any] = {}
        # client.messages.create(...)
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class TestAnthropicStructuredOutput:
    def test_create_forces_tool_use_and_surfaces_input(self):
        import asyncio

        plan_dict = {
            "partitions": [
                {"subtask": "a", "files": ["a.py"]},
                {"subtask": "b", "files": ["b.py"]},
            ],
        }
        # Anthropic response shape: content=[{type:"tool_use", id, name, input}]
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(
                type="tool_use",
                id="tu_1",
                name="PartitionPlan",
                input=plan_dict,
            )],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
        client = _FakeAnthropicClient(fake_response)
        adapter = AnthropicAdapter(client=client)
        req = LLMRequest(
            system="sys",
            messages=[{"role": "user", "content": "plan"}],
            tools=[],
            model="claude-sonnet-5",
            temperature=0.0, max_tokens=2000, timeout_seconds=30,
            response_schema=PartitionPlan,
        )
        result = asyncio.run(adapter.create(req))

        # Tool was synthesized with schema as input_schema.
        tools_param = client.last_kwargs.get("tools") or []
        assert len(tools_param) == 1
        assert tools_param[0]["name"] == "PartitionPlan"
        assert "properties" in tools_param[0]["input_schema"]
        # Tool-use was forced.
        tc = client.last_kwargs.get("tool_choice")
        assert tc == {"type": "tool", "name": "PartitionPlan"}
        # Parsed surfaces the tool_call input dict.
        assert result.parsed == plan_dict
        plan = PartitionPlan.model_validate(result.parsed)
        assert len(plan.partitions) == 2

    def test_create_without_schema_no_tool_choice(self):
        import asyncio

        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        client = _FakeAnthropicClient(fake_response)
        adapter = AnthropicAdapter(client=client)
        req = LLMRequest(
            system="sys", messages=[{"role": "user", "content": "hi"}],
            tools=[], model="claude-sonnet-5",
            temperature=0.0, max_tokens=10, timeout_seconds=30,
        )
        result = asyncio.run(adapter.create(req))
        assert "tool_choice" not in client.last_kwargs
        assert result.parsed is None


# ---------------------------------------------------------------------------
# Ollama adapter (best-effort json_object)
# ---------------------------------------------------------------------------


class _FakeOllamaClient:
    """Mimics the OpenAI SDK pointed at Ollama (client.chat.completions.create)."""

    def __init__(self, response: Any):
        self._response = response
        self.last_kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class TestOllamaStructuredOutput:
    def test_create_sets_response_format_json_object(self):
        import asyncio
        import json

        plan_json = json.dumps({
            "partitions": [
                {"subtask": "a", "files": ["a.py"]},
                {"subtask": "b", "files": ["b.py"]},
            ],
        })
        # OpenAI chat.completions response shape.
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=plan_json, role="assistant", tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        client = _FakeOllamaClient(fake_response)
        adapter = OllamaAdapter(client=client)
        req = LLMRequest(
            system="sys",
            messages=[{"role": "user", "content": "plan"}],
            tools=[],
            model="gemma4:31b",
            temperature=0.0, max_tokens=2000, timeout_seconds=30,
            response_schema=PartitionPlan,
        )
        result = asyncio.run(adapter.create(req))
        assert client.last_kwargs.get("response_format") == {"type": "json_object"}
        # Parsed populated.
        assert result.parsed is not None
        plan = PartitionPlan.model_validate(result.parsed)
        assert len(plan.partitions) == 2

    def test_create_without_schema_no_response_format(self):
        import asyncio

        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="hi", role="assistant", tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        client = _FakeOllamaClient(fake_response)
        adapter = OllamaAdapter(client=client)
        req = LLMRequest(
            system="sys", messages=[{"role": "user", "content": "hi"}],
            tools=[], model="gemma4:26b",
            temperature=0.0, max_tokens=10, timeout_seconds=30,
        )
        result = asyncio.run(adapter.create(req))
        assert "response_format" not in client.last_kwargs
        assert result.parsed is None

    def test_create_degrades_gracefully_on_invalid_json(self):
        import asyncio

        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="not valid json", role="assistant", tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        client = _FakeOllamaClient(fake_response)
        adapter = OllamaAdapter(client=client)
        req = LLMRequest(
            system="sys", messages=[{"role": "user", "content": "x"}],
            tools=[], model="gemma4:26b",
            temperature=0.0, max_tokens=10, timeout_seconds=30,
            response_schema=PartitionPlan,
        )
        result = asyncio.run(adapter.create(req))
        assert result.parsed is None
