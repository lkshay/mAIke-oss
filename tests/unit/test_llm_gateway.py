import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types as genai_types

import logging

from maike.constants import (
    COST_PER_M_INPUT_USD,
    COST_PER_M_OUTPUT_USD,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    GEMINI_COST_PER_M_INPUT_USD,
    GEMINI_COST_PER_M_OUTPUT_USD,
    LLM_RETRY_503_COOLDOWN_SECONDS,
    LLM_RETRY_BASE_DELAY_SECONDS,
    LLM_RETRY_JITTER_SECONDS,
    LLM_RETRY_MAX_ATTEMPTS,
    LLM_RETRY_MAX_DELAY_SECONDS,
    MODEL_CONTEXT_LIMITS,
    OPENAI_COST_PER_M_INPUT_USD,
    OPENAI_COST_PER_M_OUTPUT_USD,
    pricing_for_model,
)
from maike.atoms.llm import LLMCallRecord, LLMResult, StopReason, StreamChunk, TokenUsage
from maike.cost.tracker import CostTracker
from maike.gateway.llm_gateway import LLMGateway
from maike.gateway.providers import BaseProviderAdapter, GeminiAdapter, LLMRequest, ProviderName, normalize_gemini_model, resolve_model_name, resolve_provider_name
from maike.observability.tracer import Tracer


class FakeMessagesAPI:
    async def create(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=2_000_000),
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="hello world")],
        )


class FakeClient:
    def __init__(self):
        self.messages = FakeMessagesAPI()


class SequencedMessagesAPI:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **kwargs):
        del kwargs
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SequencedAnthropicClient:
    def __init__(self, outcomes):
        self.messages = SequencedMessagesAPI(outcomes)


class FakeStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


def make_record(*, cost_usd: float, input_tokens: int = 100, output_tokens: int = 50) -> LLMCallRecord:
    return LLMCallRecord(
        provider="anthropic",
        model=DEFAULT_ANTHROPIC_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=12,
        stop_reason="end_turn",
    )


def test_llm_gateway_uses_hardcoded_cost_constants():
    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=FakeClient())

    result = __import__("asyncio").run(
        gateway.call(system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL)
    )

    expected = COST_PER_M_INPUT_USD + (2 * COST_PER_M_OUTPUT_USD)
    assert result.cost_usd == expected
    assert tracker.session_total == expected
    assert tracker.records[-1].provider == "anthropic"
    assert tracker.records[-1].model == DEFAULT_ANTHROPIC_MODEL
    assert tracker.records[-1].input_tokens == 1_000_000
    assert tracker.records[-1].output_tokens == 2_000_000
    assert tracker.records[-1].total_tokens == 3_000_000
    assert tracker.records[-1].cost_usd == expected
    assert tracker.records[-1].stop_reason == "end_turn"
    assert tracer.events[-1]["model"] == DEFAULT_ANTHROPIC_MODEL
    assert tracer.events[-1]["provider"] == "anthropic"
    assert tracer.events[-1]["input_tokens"] == 1_000_000
    assert tracer.events[-1]["output_tokens"] == 2_000_000
    assert tracer.events[-1]["total_tokens"] == 3_000_000
    assert tracer.events[-1]["cost_usd"] == expected
    assert tracer.events[-1]["stop_reason"] == "end_turn"
    assert tracer.events[-1]["latency_ms"] == result.latency_ms


def test_llm_gateway_rejects_projected_budget_over_margin():
    tracker = CostTracker(session_budget_usd=1.00)
    tracker.record(
        make_record(
            cost_usd=0.90,
            input_tokens=10,
            output_tokens=10,
        )
    )
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=FakeClient())

    with pytest.raises(RuntimeError, match="budget exceeded"):
        asyncio.run(
            gateway.call(
                system="sys",
                messages=[{"role": "user", "content": "Build a large app"}],
                model=DEFAULT_ANTHROPIC_MODEL,
            )
        )


class FakeOpenAIResponsesAPI:
    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            status="completed",
            usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=2_000_000),
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="openai text")],
                ),
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="read_file",
                    arguments='{"path":"README.md"}',
                ),
            ],
        )


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeOpenAIResponsesAPI()


class FakeGeminiModelsAPI:
    async def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            text="gemini text",
            function_calls=[SimpleNamespace(name="list_dir", args={"path": "."})],
            usage_metadata=SimpleNamespace(prompt_token_count=1_000_000, candidates_token_count=2_000_000),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )


class FakeGeminiAio:
    def __init__(self):
        self.models = FakeGeminiModelsAPI()


class FakeGeminiClient:
    def __init__(self):
        self.aio = FakeGeminiAio()


def test_openai_gateway_normalizes_responses_and_uses_openai_pricing():
    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="openai", client=FakeOpenAIClient())

    result = __import__("asyncio").run(
        gateway.call(system="sys", messages=[], model=DEFAULT_OPENAI_MODEL)
    )

    expected = OPENAI_COST_PER_M_INPUT_USD + (2 * OPENAI_COST_PER_M_OUTPUT_USD)
    assert result.provider == "openai"
    assert result.model == DEFAULT_OPENAI_MODEL
    assert result.tool_calls == [{"id": "call_1", "name": "read_file", "input": {"path": "README.md"}}]
    assert result.stop_reason.value == "tool_use"
    assert result.cost_usd == expected
    assert tracer.events[-1]["provider"] == "openai"
    assert tracker.records[-1].model == DEFAULT_OPENAI_MODEL
    assert tracker.records[-1].stop_reason == "tool_use"
    assert tracer.events[-1]["total_tokens"] == 3_000_000


def test_gemini_gateway_normalizes_responses_and_uses_gemini_pricing():
    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="gemini", client=FakeGeminiClient())

    result = __import__("asyncio").run(
        gateway.call(system="sys", messages=[], model=DEFAULT_GEMINI_MODEL)
    )

    expected = GEMINI_COST_PER_M_INPUT_USD + (2 * GEMINI_COST_PER_M_OUTPUT_USD)
    assert result.provider == "gemini"
    assert result.model == DEFAULT_GEMINI_MODEL
    assert result.content == "gemini text"
    assert result.tool_calls == [{"id": "list_dir", "name": "list_dir", "input": {"path": "."}}]
    assert result.cost_usd == expected
    assert tracker.records[-1].model == DEFAULT_GEMINI_MODEL
    assert tracker.records[-1].stop_reason == "tool_use"
    assert tracer.events[-1]["provider"] == "gemini"
    assert tracer.events[-1]["total_tokens"] == 3_000_000


def test_gemini_gateway_preserves_thought_signatures_across_tool_turns():
    class SignatureGeminiModelsAPI:
        async def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
                candidates=[
                    SimpleNamespace(
                        finish_reason="STOP",
                        content=genai_types.Content(
                            role="model",
                            parts=[
                                genai_types.Part(
                                    functionCall=genai_types.FunctionCall(name="write_file", args={"path": "README.md"}),
                                    thoughtSignature=b"sig-123",
                                )
                            ],
                        ),
                    )
                ],
                function_calls=[SimpleNamespace(name="write_file", args={"path": "README.md"})],
            )

    class SignatureGeminiClient:
        def __init__(self):
            self.aio = SimpleNamespace(models=SignatureGeminiModelsAPI())

    tracker = CostTracker()
    tracer = Tracer()
    client = SignatureGeminiClient()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="gemini", client=client)

    first_result = __import__("asyncio").run(
        gateway.call(system="sys", messages=[], model=DEFAULT_MODEL)
    )

    assert first_result.content_blocks[0].thought_signature == b"sig-123"

    __import__("asyncio").run(
        gateway.call(
            system="sys",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        block.model_dump(exclude_none=True)
                        for block in first_result.content_blocks
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "write_file",
                            "tool_name": "write_file",
                            "content": "ok",
                            "is_error": False,
                        }
                    ],
                },
            ],
            model=DEFAULT_MODEL,
        )
    )

    assistant_content = client.aio.models.kwargs["contents"][0]
    assistant_part = assistant_content.parts[0]
    assert assistant_part.function_call.name == "write_file"


def test_llm_gateway_retries_transport_timeout_with_backoff(monkeypatch):
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.25)

    tracker = CostTracker()
    tracer = Tracer()
    client = SequencedAnthropicClient(
        [asyncio.TimeoutError("timeout"), asyncio.run(FakeMessagesAPI().create())]
    )
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert result.model == DEFAULT_MODEL
    assert client.messages.calls == 2
    # New formula: LLM_RETRY_BASE_DELAY_SECONDS ^ (attempt + 1) + jitter = 2.0^2 + 0.25 = 4.25
    assert sleep_calls == [4.25]
    assert [event["kind"] for event in tracer.events] == ["llm_start", "llm_retry", "llm_start", "llm_call"]
    assert tracer.events[1]["payload"]["attempt"] == 1
    assert tracer.events[1]["payload"]["delay_seconds"] == 4.25
    assert tracer.events[-1]["payload"]["attempt"] == 2


def test_llm_gateway_retries_retryable_status_codes_up_to_third_attempt(monkeypatch):
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.5)

    tracker = CostTracker()
    tracer = Tracer()
    client = SequencedAnthropicClient(
        [
            FakeStatusError(429),
            FakeStatusError(503),
            asyncio.run(FakeMessagesAPI().create()),
        ]
    )
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert result.model == DEFAULT_MODEL
    assert client.messages.calls == 3
    # New formula: attempt=1: 2.0^2 + 0.5 = 4.5, attempt=2: 2.0^3 + 0.5 = 8.5
    assert sleep_calls == [4.5, 8.5]
    retry_events = [event for event in tracer.events if event["kind"] == "llm_retry"]
    assert [event["payload"]["status_code"] for event in retry_events] == [429, 503]


def test_llm_gateway_does_not_retry_non_retryable_status_codes(monkeypatch):
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)

    tracker = CostTracker()
    tracer = Tracer()
    client = SequencedAnthropicClient([FakeStatusError(400)])
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    with pytest.raises(FakeStatusError, match="http 400"):
        asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert client.messages.calls == 1
    assert sleep_calls == []
    assert tracer.events[-1]["kind"] == "llm_error"
    assert tracer.events[-1]["payload"]["retryable"] is False


def test_provider_adapter_closes_owned_default_client():
    events = {"closed": 0}

    class OwnedClient:
        async def aclose(self):
            events["closed"] += 1

    class OwnedClientAdapter(BaseProviderAdapter):
        provider_name = ProviderName.GEMINI

        def _build_default_client(self):
            return OwnedClient()

        async def create(self, request):
            del request
            self._get_client()
            return None

    async def scenario():
        adapter = OwnedClientAdapter()
        first = adapter._get_client()
        second = adapter._get_client()
        await adapter.aclose()
        await adapter.aclose()
        return first is second, adapter._owned_client

    reused, owned_client = asyncio.run(scenario())

    assert reused is True
    assert events["closed"] == 1
    assert owned_client is None


def test_provider_resolution_uses_model_or_provider_hints():
    assert resolve_provider_name(model="claude-opus-4-20250514").value == "anthropic"
    assert resolve_provider_name(model="gpt-5.4").value == "openai"
    assert resolve_provider_name(model="gemini-2.5-flash").value == "gemini"
    # resolve_model_name passes explicit models through unchanged
    assert resolve_model_name("openai", DEFAULT_OPENAI_MODEL) == DEFAULT_OPENAI_MODEL
    assert resolve_model_name("gemini", DEFAULT_GEMINI_MODEL) == DEFAULT_GEMINI_MODEL
    assert resolve_model_name("gemini", DEFAULT_MODEL) == DEFAULT_MODEL


def test_llm_gateway_applies_provider_timeout(monkeypatch):
    class SlowMessagesAPI:
        async def create(self, **kwargs):
            del kwargs
            await asyncio.sleep(0.02)
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="late")],
            )

    class SlowClient:
        def __init__(self):
            self.messages = SlowMessagesAPI()

    monkeypatch.setattr("maike.gateway.llm_gateway.DEFAULT_LLM_TIMEOUT_SECONDS", 0.001)
    gateway = LLMGateway(cost_tracker=CostTracker(), tracer=Tracer(), provider_name="anthropic", client=SlowClient())

    with pytest.raises(TimeoutError):
        asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))


# ---------- New retry-improvement tests ----------


class FakeStatusErrorWithHeaders(RuntimeError):
    """Exception that carries a response with headers, simulating provider SDK errors."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None):
        super().__init__(f"http {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=headers or {},
        )


def test_retry_respects_retry_after_header(monkeypatch):
    """When the server sends a Retry-After header, the gateway should use that value as delay."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.5)

    tracker = CostTracker()
    tracer = Tracer()
    # First call: 429 with Retry-After: 10 seconds, second call: success
    client = SequencedAnthropicClient(
        [
            FakeStatusErrorWithHeaders(429, {"retry-after": "10"}),
            asyncio.run(FakeMessagesAPI().create()),
        ]
    )
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert result.model == DEFAULT_MODEL
    assert client.messages.calls == 2
    # Delay should be exactly 10.0 (from Retry-After header), not the exponential backoff
    assert sleep_calls == [10.0]
    retry_events = [e for e in tracer.events if e["kind"] == "llm_retry"]
    assert len(retry_events) == 1
    assert retry_events[0]["payload"]["retry_after"] == 10.0
    assert retry_events[0]["payload"]["delay_seconds"] == 10.0


def test_retry_respects_retry_after_ms_header(monkeypatch):
    """Anthropic sends retry-after-ms; the gateway should convert ms to seconds."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.5)

    tracker = CostTracker()
    tracer = Tracer()
    client = SequencedAnthropicClient(
        [
            FakeStatusErrorWithHeaders(429, {"retry-after-ms": "5000"}),
            asyncio.run(FakeMessagesAPI().create()),
        ]
    )
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert result.model == DEFAULT_MODEL
    # 5000ms = 5.0 seconds
    assert sleep_calls == [5.0]


def test_retry_503_cooldown(monkeypatch):
    """After exhausting all fast retries on 503, the gateway does one extra attempt after cooldown."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.0)

    tracker = CostTracker()
    tracer = Tracer()
    # LLM_RETRY_MAX_ATTEMPTS (5) 503 errors, then success on cooldown attempt
    outcomes = [FakeStatusError(503) for _ in range(LLM_RETRY_MAX_ATTEMPTS)] + [
        asyncio.run(FakeMessagesAPI().create())
    ]
    client = SequencedAnthropicClient(outcomes)
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert result.model == DEFAULT_MODEL
    # Should have used max_attempts + 1 total calls (the extra cooldown attempt)
    assert client.messages.calls == LLM_RETRY_MAX_ATTEMPTS + 1
    # The last sleep should be the cooldown delay
    assert sleep_calls[-1] == LLM_RETRY_503_COOLDOWN_SECONDS


def test_retry_503_cooldown_still_fails(monkeypatch):
    """If the 503 cooldown attempt also fails, the error is raised."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.0)

    tracker = CostTracker()
    tracer = Tracer()
    # All attempts fail with 503, including the cooldown attempt
    outcomes = [FakeStatusError(503) for _ in range(LLM_RETRY_MAX_ATTEMPTS + 1)]
    client = SequencedAnthropicClient(outcomes)
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    with pytest.raises(FakeStatusError, match="http 503"):
        asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    # All attempts used including the cooldown
    assert client.messages.calls == LLM_RETRY_MAX_ATTEMPTS + 1
    assert sleep_calls[-1] == LLM_RETRY_503_COOLDOWN_SECONDS
    # Final event should be an error
    assert tracer.events[-1]["kind"] == "llm_error"


def test_retry_max_delay_cap(monkeypatch):
    """Delay should never exceed LLM_RETRY_MAX_DELAY_SECONDS even at high attempt counts."""
    gateway = LLMGateway(
        cost_tracker=CostTracker(),
        tracer=Tracer(),
        provider_name="anthropic",
        client=FakeClient(),
    )
    # With jitter at max
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: LLM_RETRY_JITTER_SECONDS)

    # Test with very high attempt numbers that would produce huge exponential values
    for attempt in range(1, 20):
        delay = gateway._calculate_delay(attempt, 500, None)
        assert delay <= LLM_RETRY_MAX_DELAY_SECONDS, (
            f"attempt={attempt}: delay {delay} exceeds max {LLM_RETRY_MAX_DELAY_SECONDS}"
        )


def test_retry_uses_configurable_constants(monkeypatch):
    """Verify the gateway uses the configurable constants, not hardcoded values."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.0)

    tracker = CostTracker()
    tracer = Tracer()

    # Verify max_attempts is configurable: with LLM_RETRY_MAX_ATTEMPTS errors (non-503)
    # the gateway should try exactly that many times
    outcomes = [FakeStatusError(429) for _ in range(LLM_RETRY_MAX_ATTEMPTS)]
    client = SequencedAnthropicClient(outcomes)
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    with pytest.raises(FakeStatusError, match="http 429"):
        asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert client.messages.calls == LLM_RETRY_MAX_ATTEMPTS

    # Verify the backoff uses LLM_RETRY_BASE_DELAY_SECONDS
    # For attempt=1 with jitter=0: LLM_RETRY_BASE_DELAY_SECONDS ^ (1+1)
    expected_first_delay = LLM_RETRY_BASE_DELAY_SECONDS ** 2
    assert sleep_calls[0] == expected_first_delay

    # For attempt=2 with jitter=0: LLM_RETRY_BASE_DELAY_SECONDS ^ (2+1)
    expected_second_delay = LLM_RETRY_BASE_DELAY_SECONDS ** 3
    assert sleep_calls[1] == expected_second_delay


def test_retry_after_header_ignored_when_exceeds_max(monkeypatch):
    """If Retry-After exceeds LLM_RETRY_MAX_DELAY_SECONDS, use exponential backoff instead."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.0)

    tracker = CostTracker()
    tracer = Tracer()
    # Retry-After of 9999 exceeds LLM_RETRY_MAX_DELAY_SECONDS (60)
    client = SequencedAnthropicClient(
        [
            FakeStatusErrorWithHeaders(429, {"retry-after": "9999"}),
            asyncio.run(FakeMessagesAPI().create()),
        ]
    )
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_MODEL))

    assert result.model == DEFAULT_MODEL
    # Should fall back to exponential backoff, not use the huge Retry-After
    expected_delay = LLM_RETRY_BASE_DELAY_SECONDS ** 2  # attempt=1, jitter=0
    assert sleep_calls == [expected_delay]


def test_extract_retry_after_handles_missing_response():
    """_extract_retry_after returns None when exception has no response."""
    gateway = LLMGateway(
        cost_tracker=CostTracker(),
        tracer=Tracer(),
        provider_name="anthropic",
        client=FakeClient(),
    )
    # Plain exception with no response attribute
    assert gateway._extract_retry_after(RuntimeError("plain error")) is None
    # Exception with response but no headers
    exc_no_headers = RuntimeError("no headers")
    exc_no_headers.response = SimpleNamespace(status_code=429)
    assert gateway._extract_retry_after(exc_no_headers) is None


# ---------- Gemini model support tests ----------


def test_gemini_3_flash_preview_in_context_limits():
    """Verify newer Gemini models are registered in MODEL_CONTEXT_LIMITS."""
    assert "gemini-3-flash-preview" in MODEL_CONTEXT_LIMITS
    assert MODEL_CONTEXT_LIMITS["gemini-3-flash-preview"] == 1_048_576
    assert "gemini-2.0-flash" in MODEL_CONTEXT_LIMITS
    assert MODEL_CONTEXT_LIMITS["gemini-2.0-flash"] == 1_048_576


def test_gemini_new_models_have_pricing():
    """Verify newer Gemini models have pricing entries."""
    pricing_flash = pricing_for_model("gemini-3-flash-preview")
    assert pricing_flash is not None
    assert pricing_flash.input_per_million_usd == GEMINI_COST_PER_M_INPUT_USD
    assert pricing_flash.output_per_million_usd == GEMINI_COST_PER_M_OUTPUT_USD

    pricing_2 = pricing_for_model("gemini-2.0-flash")
    assert pricing_2 is not None
    assert pricing_2.input_per_million_usd == GEMINI_COST_PER_M_INPUT_USD


def test_gemini_model_name_normalization():
    """Verify aliases resolve correctly via normalize_gemini_model."""
    assert normalize_gemini_model("gemini-3-flash") == "gemini-3-flash-preview"
    assert normalize_gemini_model("gemini-3.0-flash") == "gemini-3-flash-preview"
    assert normalize_gemini_model("gemini-2.0-flash") == "gemini-2.0-flash"
    # Unknown models pass through unchanged
    assert normalize_gemini_model("gemini-2.5-flash") == "gemini-2.5-flash"
    assert normalize_gemini_model("gemini-99-turbo") == "gemini-99-turbo"


def test_gemini_empty_response_raises_for_retry(caplog):
    """Empty Gemini response with no finish_reason raises ValueError for retry."""
    adapter = GeminiAdapter()
    # No candidates, no text — finish_reason is None (transient), should raise
    fake_response = SimpleNamespace(
        candidates=[],
        text=None,
        function_calls=[],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50),
        extra_data="x" * 100,
    )
    import pytest
    with caplog.at_level(logging.WARNING, logger="maike.gateway.providers"):
        with pytest.raises(ValueError, match="empty text/tool_calls"):
            adapter._parse_response(fake_response, model="gemini-3-flash-preview")
    assert any(
        "Gemini response parsing produced empty result" in record.message
        for record in caplog.records
    )


def test_gemini_empty_response_with_stop_does_not_raise(caplog):
    """Empty Gemini response with finish_reason=STOP is accepted (model chose silence)."""
    adapter = GeminiAdapter()
    # Has a candidate with finish_reason=STOP but empty text — deliberate, not transient
    fake_response = SimpleNamespace(
        candidates=[SimpleNamespace(
            finish_reason="STOP",
            content=SimpleNamespace(parts=[SimpleNamespace(text="", thought=False)]),
        )],
        text=None,
        function_calls=[],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50),
        extra_data="x" * 100,
    )
    with caplog.at_level(logging.WARNING, logger="maike.gateway.providers"):
        result = adapter._parse_response(fake_response, model="gemini-3-flash-preview")
    # Should return empty result, not raise
    assert result.content is None
    assert result.tool_calls == []


class _CapturingTracer:
    """Minimal tracer stub that records log_event calls for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, kind: str, **payload) -> None:
        self.events.append((kind, payload))


def test_gemini_empty_response_emits_telemetry_with_stop():
    """Deliberate-STOP empty response should emit gemini_empty_response event
    with ``deliberate_stop=True`` and the expected diagnostic payload.

    Part of #3 Step 1 — diagnostic telemetry for empty responses.
    """
    tracer = _CapturingTracer()
    adapter = GeminiAdapter(tracer=tracer)
    fake_response = SimpleNamespace(
        candidates=[SimpleNamespace(
            finish_reason="STOP",
            content=SimpleNamespace(parts=[
                SimpleNamespace(text="", thought=False),
                SimpleNamespace(text="", thought=False),
            ]),
        )],
        text=None,
        function_calls=[],
        usage_metadata=SimpleNamespace(prompt_token_count=120, candidates_token_count=0),
        extra_data="x" * 100,  # response preview has length > 50 so warn path fires
    )
    result = adapter._parse_response(
        fake_response, model="gemini-3-pro-preview", attempt_number=2,
    )
    # Still returns empty result (not a raise) — deliberate STOP.
    assert result.content is None
    assert result.tool_calls == []

    assert len(tracer.events) == 1
    kind, payload = tracer.events[0]
    assert kind == "gemini_empty_response"
    assert payload["deliberate_stop"] is True
    assert payload["model"] == "gemini-3-pro-preview"
    assert payload["finish_reason"] == "STOP"
    assert payload["candidate_count"] == 1
    assert payload["parts_count"] == 2
    assert payload["part_text_lengths"] == [0, 0]
    assert payload["has_any_function_call"] is False
    assert payload["input_token_estimate"] == 120
    assert payload["native_history_len"] == 0  # adapter is fresh
    assert payload["attempt_number"] == 2
    assert isinstance(payload["response_preview"], str)


def test_gemini_empty_response_emits_telemetry_before_raise():
    """Transient empty response (no finish_reason) should emit the event AND raise."""
    import pytest

    tracer = _CapturingTracer()
    adapter = GeminiAdapter(tracer=tracer)
    fake_response = SimpleNamespace(
        candidates=[],  # no candidates at all — transient
        text=None,
        function_calls=[],
        usage_metadata=SimpleNamespace(prompt_token_count=200, candidates_token_count=0),
        extra_data="x" * 100,
    )
    with pytest.raises(ValueError, match="empty text/tool_calls"):
        adapter._parse_response(
            fake_response, model="gemini-3-pro-preview", attempt_number=1,
        )
    # Event emitted BEFORE the raise
    assert len(tracer.events) == 1
    kind, payload = tracer.events[0]
    assert kind == "gemini_empty_response"
    assert payload["deliberate_stop"] is False
    assert payload["candidate_count"] == 0
    assert payload["parts_count"] == 0
    assert payload["attempt_number"] == 1


def test_gemini_empty_response_without_tracer_is_no_op():
    """Adapter with no tracer (tests / programmatic use) must not crash."""
    adapter = GeminiAdapter()  # no tracer
    fake_response = SimpleNamespace(
        candidates=[SimpleNamespace(
            finish_reason="STOP",
            content=SimpleNamespace(parts=[SimpleNamespace(text="", thought=False)]),
        )],
        text=None,
        function_calls=[],
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=0),
        extra_data="x" * 100,
    )
    # Should not raise even though no tracer was supplied.
    result = adapter._parse_response(fake_response, model="gemini-3-flash-preview")
    assert result.content is None


def test_gemini_retry_invalidates_native_history():
    """#3 Step 2 — thought-signature-drift mitigation.

    On a retry (attempt_number > 1), the adapter should clear
    ``_native_history`` before the call so the subsequent
    ``_convert_messages`` fallback rebuilds fresh contents (stripping
    stale ``thought_signature`` blocks).  First-attempt calls must NOT
    clear the cache — the native-history fast path is important for
    multi-turn performance."""
    import asyncio

    class _RecordingClient:
        def __init__(self):
            self.contents_seen: list = []

            async def gen(*, model, contents, config):
                # Record which contents list was passed — native_history
                # reuses the cached list object, so identity matters.
                self.contents_seen.append(contents)
                return SimpleNamespace(
                    candidates=[SimpleNamespace(
                        finish_reason="STOP",
                        content=SimpleNamespace(parts=[SimpleNamespace(
                            text="ok", thought=False,
                        )]),
                    )],
                    text=None,
                    function_calls=[],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=5, candidates_token_count=5,
                    ),
                )

            self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

    adapter = GeminiAdapter(client=_RecordingClient())
    # Seed native_history so we can observe the invalidation.
    adapter._native_history = [{"role": "user", "parts": [{"text": "seed"}]}]
    adapter._native_history_msg_count = 1

    # Attempt 1: should reuse native_history (identity match).
    req1 = LLMRequest(
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="gemini-3-pro-preview",
        temperature=0.0, max_tokens=100, timeout_seconds=30,
        attempt_number=1,
    )
    asyncio.run(adapter.create(req1))

    # Attempt 2: same request, but attempt_number=2 → should invalidate
    # _native_history, forcing _convert_messages to rebuild.
    # Re-seed history to simulate state at retry time.
    adapter._native_history = [{"role": "user", "parts": [{"text": "seed"}]}]
    adapter._native_history_msg_count = 1
    first_native_id = id(adapter._native_history[0])

    req2 = LLMRequest(
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="gemini-3-pro-preview",
        temperature=0.0, max_tokens=100, timeout_seconds=30,
        attempt_number=2,
    )
    asyncio.run(adapter.create(req2))
    # After attempt 2, contents_seen[1] should be a FRESH list built by
    # _convert_messages — NOT the seeded native_history we set before.
    rebuilt_contents = adapter.client.contents_seen[1]
    # Native history was rebuilt with the converted messages, so the
    # seed-object's identity no longer appears.
    assert all(id(c) != first_native_id for c in rebuilt_contents), (
        "Retry should have dropped the stale native_history seed, but the "
        "original object still appears in the converted contents."
    )


def test_gemini_first_attempt_does_not_clear_history():
    """Complement to the retry test: attempt_number=1 MUST preserve the
    native-history fast path.  Otherwise we'd invalidate on every single
    call and lose the thought-signature continuity for happy-path turns."""
    import asyncio
    from types import SimpleNamespace

    class _RecordingClient:
        def __init__(self):
            async def gen(*, model, contents, config):
                return SimpleNamespace(
                    candidates=[SimpleNamespace(
                        finish_reason="STOP",
                        content=SimpleNamespace(parts=[SimpleNamespace(
                            text="ok", thought=False,
                        )]),
                    )],
                    text=None,
                    function_calls=[],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=5, candidates_token_count=5,
                    ),
                )
            self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=gen))

    adapter = GeminiAdapter(client=_RecordingClient())
    # Seed a native_history that matches the message count — attempt 1
    # should honor it.
    seeded_entry = {"role": "user", "parts": [{"text": "seed"}]}
    adapter._native_history = [seeded_entry]
    adapter._native_history_msg_count = 1

    req = LLMRequest(
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="gemini-3-pro-preview",
        temperature=0.0, max_tokens=100, timeout_seconds=30,
        attempt_number=1,
    )
    asyncio.run(adapter.create(req))
    # The seed entry is still in native_history (plus the new response
    # was appended).
    assert seeded_entry in adapter._native_history


def test_llm_request_attempt_number_defaults_to_one():
    """LLMRequest.attempt_number is a new optional field; default must be 1."""
    req = LLMRequest(
        system="",
        messages=[],
        tools=[],
        model="x",
        temperature=0.0,
        max_tokens=10,
        timeout_seconds=30,
    )
    assert req.attempt_number == 1


def test_llm_request_attempt_number_threaded_through_gateway_retry(monkeypatch):
    """Gateway's retry loop should stamp attempt_number on each LLMRequest
    via dataclasses.replace, not leave it at the default 1."""
    seen_attempts: list[int] = []

    class _FakeAdapter:
        provider_name = ProviderName.GEMINI

        async def create(self, *, request):
            seen_attempts.append(request.attempt_number)
            if len(seen_attempts) == 1:
                # First attempt — fail with a retryable error so the loop retries
                raise ValueError("transient empty response")
            # Second attempt — succeed
            from maike.atoms.llm import LLMResult, TokenUsage, StopReason
            return LLMResult(
                provider="gemini",
                model=request.model,
                content="ok",
                content_blocks=[],
                tool_calls=[],
                stop_reason=StopReason.END_TURN,
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                latency_ms=0,
            )

        async def aclose(self):
            return None

    from maike.cost.tracker import CostTracker
    from maike.gateway.llm_gateway import LLMGateway
    from maike.observability.tracer import Tracer

    # Intercept build_provider_adapter so the gateway uses our fake.
    monkeypatch.setattr(
        "maike.gateway.llm_gateway.build_provider_adapter",
        lambda *a, **kw: _FakeAdapter(),
    )
    # Also mark ValueError as retryable in this test's scope.
    gw = LLMGateway(CostTracker(), Tracer(), provider_name="gemini", silent=True)
    # Force the gateway to treat ValueError as retryable.
    monkeypatch.setattr(gw, "_is_retryable_exception", lambda exc, status_code=None: True)
    monkeypatch.setattr(gw, "_calculate_delay", lambda *a, **kw: 0.0)

    import asyncio
    asyncio.run(gw.call(system="", messages=[{"role": "user", "content": "hi"}],
                         model="gemini-3-flash-preview", max_tokens=100))

    # First call: attempt_number=1; second: attempt_number=2.
    assert seen_attempts == [1, 2]


def test_gemini_adapter_normalizes_model_in_create():
    """GeminiAdapter.create() should normalize model aliases before calling the API."""

    class CapturingGeminiModelsAPI:
        async def generate_content(self, **kwargs):
            self.captured_model = kwargs.get("model")
            return SimpleNamespace(
                text="ok",
                function_calls=[],
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
                candidates=[SimpleNamespace(finish_reason="STOP")],
            )

    class CapturingGeminiClient:
        def __init__(self):
            self.aio = SimpleNamespace(models=CapturingGeminiModelsAPI())

    from maike.gateway.providers import LLMRequest

    client = CapturingGeminiClient()
    adapter = GeminiAdapter(client=client)
    request = LLMRequest(
        system="sys",
        messages=[],
        tools=[],
        model="gemini-3-flash",  # alias
        temperature=0.2,
        max_tokens=1024,
        timeout_seconds=30,
    )
    result = asyncio.run(adapter.create(request))
    # The model sent to the API should be the canonical name
    assert client.aio.models.captured_model == "gemini-3-flash-preview"
    # The model in the result should also be canonical
    assert result.model == "gemini-3-flash-preview"


# ---------- Provider fallback tests ----------


class FakeOpenAIFallbackClient:
    """Minimal OpenAI client that always succeeds, for fallback tests."""

    def __init__(self):
        self.responses = FakeOpenAIResponsesAPI()


def _patch_fallback_env(monkeypatch, *, anthropic=False, openai=False, gemini=False):
    """Set or unset provider API key env vars for fallback availability tests."""
    if anthropic:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if openai:
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if gemini:
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_fallback_triggers_on_401_after_retry_exhaustion(monkeypatch):
    """When primary provider fails with 401 (non-retryable), gateway falls back to an alternative."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    _patch_fallback_env(monkeypatch, anthropic=True, openai=True)

    # Patch build_provider_adapter so the fallback adapter uses our fake OpenAI client
    original_build = __import__("maike.gateway.providers", fromlist=["build_provider_adapter"]).build_provider_adapter

    def patched_build(provider_name, client=None, tracer=None):
        if str(provider_name) == "openai" and client is None:
            return original_build(provider_name, client=FakeOpenAIFallbackClient())
        return original_build(provider_name, client=client)

    monkeypatch.setattr("maike.gateway.llm_gateway.build_provider_adapter", patched_build)

    tracker = CostTracker()
    tracer = Tracer()
    # Primary (anthropic) always fails with 401 — non-retryable, so no retries, straight to fallback
    client = SequencedAnthropicClient([FakeStatusError(401)])
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL))

    # The result should come from the fallback provider
    assert result.provider == "openai"
    assert result.model == DEFAULT_OPENAI_MODEL
    assert result.metadata.get("fallback_provider") == "openai"
    # Primary provider was called exactly once (401 is not retried)
    assert client.messages.calls == 1
    # Verify tracer recorded fallback event
    fallback_events = [e for e in tracer.events if e["kind"] == "llm_fallback"]
    assert len(fallback_events) == 1
    assert fallback_events[0]["payload"]["original_provider"] == "anthropic"
    assert fallback_events[0]["payload"]["fallback_provider"] == "openai"


def test_fallback_does_not_trigger_on_429(monkeypatch):
    """429 is retryable — the gateway should retry (not fall back) and eventually raise."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("maike.gateway.llm_gateway.random.uniform", lambda a, b: 0.0)
    _patch_fallback_env(monkeypatch, anthropic=True, openai=True)

    tracker = CostTracker()
    tracer = Tracer()
    # All attempts fail with 429 (retryable) — should exhaust retries then raise, NOT fallback
    outcomes = [FakeStatusError(429) for _ in range(LLM_RETRY_MAX_ATTEMPTS)]
    client = SequencedAnthropicClient(outcomes)
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    with pytest.raises(FakeStatusError, match="http 429"):
        asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL))

    # All retries used, no fallback
    assert client.messages.calls == LLM_RETRY_MAX_ATTEMPTS
    fallback_events = [e for e in tracer.events if e["kind"] == "llm_fallback"]
    assert len(fallback_events) == 0


def test_fallback_requires_alternative_api_key(monkeypatch):
    """When no alternative provider has an API key, the original error is raised."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    # Only anthropic key set — no fallback targets available
    _patch_fallback_env(monkeypatch, anthropic=True, openai=False, gemini=False)

    tracker = CostTracker()
    tracer = Tracer()
    client = SequencedAnthropicClient([FakeStatusError(401)])
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    with pytest.raises(FakeStatusError, match="http 401"):
        asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL))

    # No fallback event logged
    fallback_events = [e for e in tracer.events if e["kind"] == "llm_fallback"]
    assert len(fallback_events) == 0


def test_fallback_only_happens_once(monkeypatch):
    """Fallback should not chain — if the fallback provider also returns 401, the original error is raised."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", fake_sleep)
    _patch_fallback_env(monkeypatch, anthropic=True, openai=True, gemini=True)

    # Make the fallback (openai) also fail
    class FailingOpenAIResponsesAPI:
        async def create(self, **kwargs):
            raise FakeStatusError(401)

    class FailingOpenAIClient:
        def __init__(self):
            self.responses = FailingOpenAIResponsesAPI()

    original_build = __import__("maike.gateway.providers", fromlist=["build_provider_adapter"]).build_provider_adapter

    def patched_build(provider_name, client=None, tracer=None):
        if str(provider_name) == "openai" and client is None:
            return original_build(provider_name, client=FailingOpenAIClient())
        return original_build(provider_name, client=client)

    monkeypatch.setattr("maike.gateway.llm_gateway.build_provider_adapter", patched_build)

    tracker = CostTracker()
    tracer = Tracer()
    client = SequencedAnthropicClient([FakeStatusError(401)])
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    # The original 401 error should be raised (not a chain of fallbacks)
    with pytest.raises(FakeStatusError, match="http 401"):
        asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL))

    # Only one fallback attempt was made
    fallback_events = [e for e in tracer.events if e["kind"] == "llm_fallback"]
    assert len(fallback_events) == 1


def test_fallback_metadata_in_result(monkeypatch):
    """When fallback succeeds, result.metadata contains 'fallback_provider'."""
    monkeypatch.setattr("maike.gateway.llm_gateway.asyncio.sleep", lambda d: asyncio.sleep(0))
    _patch_fallback_env(monkeypatch, anthropic=True, gemini=True)

    original_build = __import__("maike.gateway.providers", fromlist=["build_provider_adapter"]).build_provider_adapter

    def patched_build(provider_name, client=None, tracer=None):
        if str(provider_name) == "gemini" and client is None:
            return original_build(provider_name, client=FakeGeminiClient())
        return original_build(provider_name, client=client)

    monkeypatch.setattr("maike.gateway.llm_gateway.build_provider_adapter", patched_build)

    tracker = CostTracker()
    tracer = Tracer()
    # Primary fails with ConnectionRefusedError (fallback-eligible)
    client = SequencedAnthropicClient([ConnectionRefusedError("connection refused")])
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic", client=client)

    # OpenAI key not set, so fallback goes to gemini (next in PROVIDER_FALLBACK_ORDER with a key)
    result = asyncio.run(gateway.call(system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL))

    assert result.metadata == {"fallback_provider": "gemini"}
    assert result.provider == "gemini"
    assert result.model == DEFAULT_GEMINI_MODEL


# ---------- Streaming tests ----------


def test_stream_chunk_data_model():
    """StreamChunk fields have sensible defaults."""
    chunk = StreamChunk()
    assert chunk.text_delta == ""
    assert chunk.tool_use_delta is None
    assert chunk.usage_update is None
    assert chunk.is_final is False
    assert chunk.stop_reason is None
    assert chunk.accumulated_result is None

    # With text
    chunk = StreamChunk(text_delta="hello")
    assert chunk.text_delta == "hello"

    # With tool_use_delta
    chunk = StreamChunk(tool_use_delta={"id": "t1", "name": "Bash", "input": {"cmd": "ls"}})
    assert chunk.tool_use_delta["name"] == "Bash"

    # Final chunk
    chunk = StreamChunk(is_final=True, stop_reason="end_turn")
    assert chunk.is_final is True
    assert chunk.stop_reason == "end_turn"


class FakeStreamingAdapter(BaseProviderAdapter):
    """Mock adapter that supports stream_create, yielding pre-defined chunks."""

    provider_name = ProviderName.ANTHROPIC

    def __init__(self, chunks: list[StreamChunk], *, fail_on_create: bool = False):
        super().__init__()
        self._chunks = chunks
        self._fail_on_create = fail_on_create

    def _build_default_client(self) -> Any:
        return None

    async def create(self, request: LLMRequest) -> LLMResult:
        if self._fail_on_create:
            raise RuntimeError("create should not be called")
        return LLMResult(
            provider="anthropic",
            content="fallback content",
            content_blocks=[],
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.0,
            latency_ms=1,
            model=DEFAULT_ANTHROPIC_MODEL,
        )

    async def stream_create(self, request: LLMRequest):
        for chunk in self._chunks:
            yield chunk


def test_stream_call_accumulates_text_into_final_result():
    """stream_call yields intermediate chunks and a final chunk with accumulated LLMResult."""
    chunks = [
        StreamChunk(text_delta="Hello"),
        StreamChunk(text_delta=" world"),
        StreamChunk(
            usage_update={"input_tokens": 100, "output_tokens": 20},
            stop_reason="end_turn",
            is_final=True,
        ),
    ]
    adapter = FakeStreamingAdapter(chunks, fail_on_create=True)

    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic")
    gateway.adapter = adapter

    collected: list[StreamChunk] = []

    async def run():
        async for chunk in gateway.stream_call(
            system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL,
        ):
            collected.append(chunk)

    asyncio.run(run())

    # Should have intermediate chunks plus final
    assert len(collected) >= 2
    # Final chunk
    final = collected[-1]
    assert final.is_final is True
    assert final.accumulated_result is not None
    result = final.accumulated_result
    assert result.content == "Hello world"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.stop_reason == StopReason.END_TURN
    # Cost should be recorded
    assert tracker.session_total > 0
    assert len(tracker.records) == 1


def test_stream_call_accumulates_tool_calls():
    """stream_call correctly accumulates tool call deltas into the final result."""
    chunks = [
        StreamChunk(text_delta="Let me check."),
        StreamChunk(tool_use_delta={"id": "tc_1", "name": "Bash", "input": {"cmd": "ls"}}),
        StreamChunk(
            usage_update={"input_tokens": 50, "output_tokens": 30},
            stop_reason="end_turn",
            is_final=True,
        ),
    ]
    adapter = FakeStreamingAdapter(chunks, fail_on_create=True)

    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic")
    gateway.adapter = adapter

    async def run():
        final = None
        async for chunk in gateway.stream_call(
            system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL,
        ):
            if chunk.is_final:
                final = chunk
        return final

    final = asyncio.run(run())
    assert final is not None
    result = final.accumulated_result
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "Bash"
    assert result.tool_calls[0]["input"] == {"cmd": "ls"}
    # With tool calls, stop_reason should be TOOL_USE
    assert result.stop_reason == StopReason.TOOL_USE


class FailingStreamAdapter(BaseProviderAdapter):
    """Adapter whose stream_create raises mid-stream."""

    provider_name = ProviderName.ANTHROPIC

    def __init__(self):
        super().__init__()

    def _build_default_client(self) -> Any:
        return None

    async def create(self, request: LLMRequest) -> LLMResult:
        return LLMResult(
            provider="anthropic",
            content="fallback response",
            content_blocks=[],
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.0,
            latency_ms=1,
            model=DEFAULT_ANTHROPIC_MODEL,
        )

    async def stream_create(self, request: LLMRequest):
        yield StreamChunk(text_delta="partial")
        raise ConnectionError("stream broken")


def test_stream_call_falls_back_on_streaming_failure():
    """When streaming fails mid-response, stream_call falls back to non-streaming call."""
    adapter = FailingStreamAdapter()

    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic")
    gateway.adapter = adapter

    collected: list[StreamChunk] = []

    async def run():
        async for chunk in gateway.stream_call(
            system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL,
        ):
            collected.append(chunk)

    asyncio.run(run())

    # After fallback, we should get a single final chunk from the non-streaming call
    assert len(collected) >= 1
    final = collected[-1]
    assert final.is_final is True
    assert final.accumulated_result is not None
    assert final.accumulated_result.content == "fallback response"


class NonStreamingAdapter(BaseProviderAdapter):
    """Adapter that does NOT have stream_create."""

    provider_name = ProviderName.ANTHROPIC

    def __init__(self):
        super().__init__()

    def _build_default_client(self) -> Any:
        return None

    async def create(self, request: LLMRequest) -> LLMResult:
        return LLMResult(
            provider="anthropic",
            content="non-streaming response",
            content_blocks=[],
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.0,
            latency_ms=1,
            model=DEFAULT_ANTHROPIC_MODEL,
        )


def test_stream_call_falls_back_when_adapter_lacks_streaming():
    """When the adapter has no stream_create method, stream_call falls back to call()."""
    adapter = NonStreamingAdapter()

    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic")
    gateway.adapter = adapter

    collected: list[StreamChunk] = []

    async def run():
        async for chunk in gateway.stream_call(
            system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL,
        ):
            collected.append(chunk)

    asyncio.run(run())

    # Should get exactly one final chunk from the fallback
    assert len(collected) == 1
    final = collected[0]
    assert final.is_final is True
    assert final.accumulated_result is not None
    assert final.accumulated_result.content == "non-streaming response"


def test_stream_call_cost_tracking_with_streaming():
    """Verify that cost tracking works correctly with streaming calls."""
    chunks = [
        StreamChunk(text_delta="response text"),
        StreamChunk(
            usage_update={"input_tokens": 1_000_000, "output_tokens": 2_000_000},
            stop_reason="end_turn",
            is_final=True,
        ),
    ]
    adapter = FakeStreamingAdapter(chunks, fail_on_create=True)

    tracker = CostTracker()
    tracer = Tracer()
    gateway = LLMGateway(cost_tracker=tracker, tracer=tracer, provider_name="anthropic")
    gateway.adapter = adapter

    async def run():
        async for chunk in gateway.stream_call(
            system="sys", messages=[], model=DEFAULT_ANTHROPIC_MODEL,
        ):
            pass

    asyncio.run(run())

    # Cost should match what we'd get from a non-streaming call with same tokens
    expected = COST_PER_M_INPUT_USD + (2 * COST_PER_M_OUTPUT_USD)
    assert tracker.session_total == expected
    assert len(tracker.records) == 1
    assert tracker.records[0].input_tokens == 1_000_000
    assert tracker.records[0].output_tokens == 2_000_000
    assert tracker.records[0].cost_usd == expected
    # Tracer should have streaming=True in its payload
    llm_call_events = [e for e in tracer.events if e["kind"] == "llm_call"]
    assert len(llm_call_events) == 1
    assert llm_call_events[0]["payload"]["streaming"] is True
