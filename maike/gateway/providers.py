"""Provider adapters for Anthropic, OpenAI, and Gemini."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)

from maike.atoms.llm import LLMContentBlock, LLMResult, StopReason, StreamChunk, TokenUsage
from maike.constants import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    default_model_for_provider as default_model_for_provider_config,
)

try:  # pragma: no cover - optional dependency
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None

try:  # pragma: no cover - optional dependency
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None

try:  # pragma: no cover - optional dependency
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None


class ProviderName(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class LLMRequest:
    system: str | list[dict[str, Any]]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    # Retry counter — gateway bumps this before each attempt so adapters can
    # emit diagnostic events correlated with retry position.  Diagnostic
    # only; no behavior hinges on it.
    attempt_number: int = 1
    # Optional Pydantic model class.  When set, adapters route the call
    # through their provider-native structured-output API (Gemini
    # response_schema, OpenAI json_schema, Anthropic forced tool_use,
    # Ollama json_object).  ``LLMResult.parsed`` is then populated with the
    # parsed dict.  See maike/gateway/structured.py for the helper used to
    # derive the JSON Schema.  Non-frozen ``type`` reference is fine — we
    # never mutate the class, just call ``.model_json_schema()`` on it.
    response_schema: Any = None  # type: ignore[name-defined]  — type[BaseModel] | None


class ProviderAdapter(Protocol):
    provider_name: ProviderName

    async def create(self, request: LLMRequest) -> LLMResult: ...
    async def aclose(self) -> None: ...

    async def stream(self, request: LLMRequest):
        """Yield StreamChunk objects. Default: fall back to non-streaming."""
        result = await self.create(request)
        from maike.atoms.llm import StreamChunk
        if result.content:
            yield StreamChunk(type="text_delta", text=result.content)
        yield StreamChunk(type="done")


def resolve_provider_name(provider_name: str | None = None, model: str | None = None) -> ProviderName:
    if provider_name:
        if isinstance(provider_name, ProviderName):
            return provider_name
        return ProviderName(str(provider_name).lower())
    if model:
        lowered = model.lower()
        if lowered.startswith("claude"):
            return ProviderName.ANTHROPIC
        if lowered.startswith("gemini"):
            return ProviderName.GEMINI
        if lowered.startswith(("gpt", "o", "chatgpt")):
            return ProviderName.OPENAI
        # Local model names served by Ollama.
        if lowered.startswith(("gemma", "llama", "qwen", "phi", "mistral", "deepseek", "codellama")):
            return ProviderName.OLLAMA
    return ProviderName(DEFAULT_PROVIDER)


def default_model_for_provider(provider_name: str | ProviderName) -> str:
    provider = resolve_provider_name(provider_name)
    return default_model_for_provider_config(provider.value)


def resolve_model_name(provider_name: str | None, model: str | None) -> str:
    provider = resolve_provider_name(provider_name=provider_name, model=model)
    if not model:
        return default_model_for_provider(provider)
    return model


_CACHE_BOUNDARY = "<!-- CACHE_BOUNDARY -->"


def _flatten_system(system: str | list[dict[str, Any]]) -> str:
    """Collapse a system prompt to a plain string (for non-Anthropic providers)."""
    if isinstance(system, str):
        return system.replace(_CACHE_BOUNDARY, "")
    return "\n".join(block.get("text", "") for block in system if block.get("text"))


def _deduplicate_repeated_text(text: str, threshold: int = 3) -> str:
    """Remove repeated sentences/lines from LLM output.

    Gemini models sometimes degenerate and repeat the same sentence many
    times in a single response.  This function detects runs of identical
    lines (after stripping whitespace) and collapses them to a single
    occurrence when the run length meets or exceeds *threshold*.
    """
    lines = text.split("\n")
    if len(lines) < threshold:
        return text

    result: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            result.append(lines[i])
            i += 1
            continue
        # Count consecutive identical lines.
        run = 1
        while i + run < len(lines) and lines[i + run].strip() == stripped:
            run += 1
        result.append(lines[i])
        if run >= threshold:
            # Skip the duplicates — keep only the first occurrence.
            logger.debug(
                "Collapsed %d repeated lines: %s", run, stripped[:80],
            )
        else:
            # Keep all lines in short runs (could be intentional like a list).
            for j in range(1, run):
                result.append(lines[i + j])
        i += run

    return "\n".join(result)


def _anthropic_system_blocks(system: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build Anthropic-style system content blocks with cache_control hints.

    If *system* is already a list of dicts, return it as-is (caller controls
    caching).  If it is a string, split on the ``<!-- CACHE_BOUNDARY -->``
    marker: the prefix gets ``cache_control`` so Anthropic can cache it across
    turns, and the suffix (which changes each turn) does not.
    """
    if isinstance(system, list):
        return system

    marker = _CACHE_BOUNDARY
    if marker in system:
        prefix, suffix = system.split(marker, 1)
        blocks: list[dict[str, Any]] = []
        if prefix.strip():
            blocks.append({
                "type": "text",
                "text": prefix,
                "cache_control": {"type": "ephemeral"},
            })
        if suffix.strip():
            blocks.append({"type": "text", "text": suffix})
        return blocks or [{"type": "text", "text": system.replace(marker, "")}]

    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


class BaseProviderAdapter:
    provider_name: ProviderName

    def __init__(self, client: Any | None = None, tracer: Any | None = None) -> None:
        self.client = client
        self._owned_client: Any | None = None
        # Optional tracer for adapter-layer diagnostic events.  None is
        # supported so tests can instantiate adapters without wiring
        # observability.  When present, must expose ``log_event(kind, **payload)``.
        self._tracer = tracer

    async def _awaitable(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return await asyncio.to_thread(lambda: value)

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self._owned_client is None:
            self._owned_client = self._build_default_client()
        return self._owned_client

    async def aclose(self) -> None:
        client = self._owned_client
        if client is None:
            return
        self._owned_client = None
        close = getattr(client, "aclose", None)
        if callable(close):
            await self._awaitable(close())
            return
        close = getattr(client, "close", None)
        if callable(close):
            await self._awaitable(close())

    # Critical response fields — log a warning when these fall back to defaults.
    _CRITICAL_FIELDS: frozenset[str] = frozenset({"candidates", "content", "parts"})

    def _extract_value(self, item: Any, key: str, default: Any = None) -> Any:
        if item is None:
            return default
        # Primary: dict access or attribute access
        if isinstance(item, dict):
            value = item.get(key)
            if value is not None:
                return value
        else:
            value = getattr(item, key, None)
            if value is not None:
                return value
        # Fallback: try nested access via `result` attribute (newer Gemini API shapes).
        nested = getattr(item, "result", None)
        if nested is not None:
            nested_val = self._extract_value(nested, key)
            if nested_val is not None:
                return nested_val
        # Log when a critical field returns its default
        if key in self._CRITICAL_FIELDS and default is not None:
            logger.debug(
                "Provider field %r returned default on %s; response type: %s",
                key,
                type(item).__name__,
                type(item).__name__,
            )
        return default

    def _extract_usage(self, response: Any, *, input_keys: tuple[str, ...], output_keys: tuple[str, ...]) -> TokenUsage:
        usage_obj = self._extract_value(response, "usage", None) or self._extract_value(response, "usage_metadata", None)
        input_tokens = 0
        output_tokens = 0
        for key in input_keys:
            value = self._extract_value(usage_obj, key, None)
            if value is not None:
                input_tokens = int(value)
                break
        for key in output_keys:
            value = self._extract_value(usage_obj, key, None)
            if value is not None:
                output_tokens = int(value)
                break
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def _normalize_stop_reason(self, raw_stop_reason: str | None, *, tool_calls: list[dict[str, Any]]) -> StopReason:
        if tool_calls:
            return StopReason.TOOL_USE
        if raw_stop_reason in {StopReason.END_TURN.value, "completed", "stop"}:
            return StopReason.END_TURN
        if raw_stop_reason in {StopReason.MAX_TOKENS.value, "max_output_tokens", "length"}:
            return StopReason.MAX_TOKENS
        if raw_stop_reason == StopReason.STOP_SEQUENCE.value:
            return StopReason.STOP_SEQUENCE
        return StopReason.END_TURN


class AnthropicAdapter(BaseProviderAdapter):
    provider_name = ProviderName.ANTHROPIC

    def _build_default_client(self) -> Any:
        if AsyncAnthropic is None:
            raise RuntimeError(
                "anthropic is not installed. Install project dependencies or inject an Anthropic client."
            )
        return AsyncAnthropic()

    async def stream(self, request: LLMRequest):
        """Stream response from Anthropic API, yielding StreamChunk objects."""
        from maike.atoms.llm import StreamChunk
        client = self._get_client()
        stream_cm = client.messages.stream(
            model=request.model,
            system=_anthropic_system_blocks(request.system),
            messages=self._convert_messages(request.messages),
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        async with stream_cm as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield StreamChunk(type="text_delta", text=getattr(delta, "text", ""))
                    elif delta_type == "input_json_delta":
                        yield StreamChunk(
                            type="tool_use_delta",
                            tool_input_json=getattr(delta, "partial_json", ""),
                        )
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        yield StreamChunk(
                            type="tool_use_start",
                            tool_call_id=getattr(block, "id", None),
                            tool_name=getattr(block, "name", None),
                        )
            # After stream completes, get the final message for usage/result
            final_message = await stream.get_final_message()
            result = self._parse_response(final_message, model=request.model)
            yield StreamChunk(type="done")
            # Attach the final result as an attribute on the last chunk
            # The gateway will collect it
            yield result  # type: ignore[misc]

    async def create(self, request: LLMRequest) -> LLMResult:
        client = self._get_client()
        # Structured output: Anthropic has no response_schema, so we
        # synthesize a single tool named after the Pydantic class with the
        # schema as input_schema, then force ``tool_choice`` to that tool.
        # The parsed args arrive via the normal tool_calls path and get
        # surfaced in ``LLMResult.parsed`` below.
        tools_param = request.tools
        tool_choice_kwarg: dict[str, Any] | None = None
        if request.response_schema is not None:
            from maike.gateway.structured import schema_dict, schema_name
            name = schema_name(request.response_schema)
            tools_param = [
                {
                    "name": name,
                    "description": (
                        "Return a value conforming to the required schema. "
                        "Call this tool exactly once with the full structured output."
                    ),
                    "input_schema": schema_dict(request.response_schema),
                }
            ]
            tool_choice_kwarg = {"type": "tool", "name": name}
        create_kwargs: dict[str, Any] = {
            "model": request.model,
            "system": _anthropic_system_blocks(request.system),
            "messages": self._convert_messages(request.messages),
            "tools": tools_param,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if tool_choice_kwarg is not None:
            create_kwargs["tool_choice"] = tool_choice_kwarg
        response = await asyncio.wait_for(
            client.messages.create(**create_kwargs),
            timeout=request.timeout_seconds,
        )
        result = self._parse_response(response, model=request.model)
        # Populate parsed from the forced tool_use block.  Anthropic
        # guarantees arguments match input_schema when tool_choice is set.
        if request.response_schema is not None and result.tool_calls:
            forced_name = schema_name(request.response_schema)
            for tc in result.tool_calls:
                if tc.get("name") == forced_name:
                    inp = tc.get("input")
                    if isinstance(inp, dict):
                        result = result.model_copy(update={"parsed": inp})
                    break
        return result

    async def stream_create(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Stream Anthropic response using the messages.stream API."""
        client = self._get_client()
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        async with client.messages.stream(
            model=request.model,
            system=_anthropic_system_blocks(request.system),
            messages=self._convert_messages(request.messages),
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield StreamChunk(text_delta=getattr(delta, "text", ""))
                    elif delta_type == "input_json_delta":
                        idx = getattr(event, "index", 0)
                        partial = getattr(delta, "partial_json", "")
                        entry = accumulated_tool_calls.setdefault(idx, {"json_parts": []})
                        entry["json_parts"].append(partial)
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        idx = getattr(event, "index", 0)
                        accumulated_tool_calls[idx] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "json_parts": [],
                        }

            # After stream ends, get the final message for usage/stop_reason
            final_message = await stream.get_final_message()

        # Extract usage
        usage_obj = getattr(final_message, "usage", None)
        usage_dict = {
            "input_tokens": getattr(usage_obj, "input_tokens", 0),
            "output_tokens": getattr(usage_obj, "output_tokens", 0),
        }

        # Build complete tool calls
        for idx, tc_data in accumulated_tool_calls.items():
            json_str = "".join(tc_data.get("json_parts", []))
            try:
                input_data = json.loads(json_str) if json_str else {}
            except json.JSONDecodeError:
                input_data = {}
            yield StreamChunk(
                tool_use_delta={
                    "id": tc_data.get("id", ""),
                    "name": tc_data.get("name", ""),
                    "input": input_data,
                },
            )

        yield StreamChunk(
            usage_update=usage_dict,
            stop_reason=getattr(final_message, "stop_reason", "end_turn"),
            is_final=True,
        )

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                converted_content: Any = content
            else:
                converted_content = []
                for block in content:
                    block_type = block.get("type")
                    if block_type == "tool_result":
                        converted_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block["tool_use_id"],
                                "content": block["content"],
                            }
                        )
                    elif block_type == "tool_use":
                        converted_content.append(
                            {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": block["input"],
                            }
                        )
                    elif block_type in {"text", "output_text"}:
                        converted_content.append({"type": "text", "text": block.get("text") or block.get("content", "")})
            converted.append({"role": message["role"], "content": converted_content})
        return converted

    def _parse_response(self, response: Any, *, model: str) -> LLMResult:
        usage = self._extract_usage(response, input_keys=("input_tokens",), output_keys=("output_tokens",))
        content_blocks: list[LLMContentBlock] = []
        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in self._extract_value(response, "content", []) or []:
            parsed = LLMContentBlock(
                type=self._extract_value(block, "type"),
                text=self._extract_value(block, "text"),
                id=self._extract_value(block, "id"),
                name=self._extract_value(block, "name"),
                input=self._extract_value(block, "input", {}) or {},
            )
            content_blocks.append(parsed)
            if parsed.type == "text" and parsed.text:
                text_parts.append(parsed.text)
            if parsed.type == "tool_use":
                tool_calls.append({"id": parsed.id, "name": parsed.name, "input": parsed.input})
        return LLMResult(
            provider=self.provider_name.value,
            content="\n".join(text_parts) if text_parts else None,
            content_blocks=content_blocks,
            tool_calls=tool_calls,
            stop_reason=self._normalize_stop_reason(
                self._extract_value(response, "stop_reason"),
                tool_calls=tool_calls,
            ),
            usage=usage,
            latency_ms=0,
            model=model,
        )


class OpenAIAdapter(BaseProviderAdapter):
    provider_name = ProviderName.OPENAI

    def _build_default_client(self) -> Any:
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai is not installed. Install project dependencies or inject an OpenAI client."
            )
        return AsyncOpenAI()

    async def create(self, request: LLMRequest) -> LLMResult:
        client = self._get_client()
        # Structured output: OpenAI Responses API accepts
        # ``text={"format": {"type": "json_schema", ...}}``.  Tools are
        # dropped because schema and tools are mutually exclusive here.
        text_kwarg: dict[str, Any] | None = None
        tools_param = self._convert_tools(request.tools) or None
        if request.response_schema is not None:
            from maike.gateway.structured import schema_dict, schema_name
            text_kwarg = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name(request.response_schema),
                    "schema": schema_dict(request.response_schema),
                    "strict": True,
                },
            }
            tools_param = None
        create_kwargs: dict[str, Any] = {
            "model": request.model,
            "instructions": _flatten_system(request.system),
            "input": self._convert_messages(request.messages),
            "tools": tools_param,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if text_kwarg is not None:
            create_kwargs["text"] = text_kwarg
        response = await asyncio.wait_for(
            client.responses.create(**create_kwargs),
            timeout=request.timeout_seconds,
        )
        result = self._parse_response(response, model=request.model)
        # Populate LLMResult.parsed for structured-output path.  Happy path:
        # OpenAI guarantees the text is valid JSON matching the schema.
        # Degrade gracefully if parsing fails for any reason.
        if request.response_schema is not None and result.content:
            try:
                import json as _json
                result = result.model_copy(
                    update={"parsed": _json.loads(result.content)},
                )
            except (ValueError, TypeError) as exc:
                logger.debug(
                    "OpenAI structured-output JSON parse failed (non-fatal): %s",
                    exc,
                )
        return result

    async def stream_create(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Stream OpenAI response using the responses.create streaming API."""
        client = self._get_client()
        stream = client.responses.create(
            model=request.model,
            instructions=_flatten_system(request.system),
            input=self._convert_messages(request.messages),
            tools=self._convert_tools(request.tools) or None,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            stream=True,
        )
        usage_dict: dict[str, Any] | None = None
        stop_reason: str | None = None

        async for event in await stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield StreamChunk(text_delta=delta)
            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", "")
                if item_type == "function_call":
                    arguments = getattr(item, "arguments", "{}")
                    parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
                    yield StreamChunk(tool_use_delta={
                        "id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                        "name": getattr(item, "name", ""),
                        "input": parsed_args,
                    })
            elif event_type == "response.completed":
                response_obj = getattr(event, "response", None)
                if response_obj is not None:
                    usage_obj = getattr(response_obj, "usage", None)
                    if usage_obj is not None:
                        usage_dict = {
                            "input_tokens": getattr(usage_obj, "input_tokens", 0),
                            "output_tokens": getattr(usage_obj, "output_tokens", 0),
                        }
                    stop_reason = getattr(response_obj, "status", "completed")

        yield StreamChunk(
            usage_update=usage_dict,
            stop_reason=stop_reason or "completed",
            is_final=True,
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
            for tool in tools
        ]

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if isinstance(content, str):
                items.append({"role": role, "content": content})
                continue
            text_parts: list[str] = []

            def flush_text() -> None:
                if text_parts:
                    items.append({"role": role, "content": "\n".join(text_parts)})
                    text_parts.clear()

            for block in content:
                block_type = block.get("type")
                if block_type in {"text", "output_text"}:
                    text = block.get("text") or block.get("content", "")
                    if text:
                        text_parts.append(text)
                elif block_type == "tool_use":
                    flush_text()
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": block["id"],
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        }
                    )
                elif block_type == "tool_result":
                    flush_text()
                    output = block.get("content", "")
                    if not isinstance(output, str):
                        output = json.dumps(output)
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": output,
                        }
                    )
            flush_text()
        return items

    def _parse_response(self, response: Any, *, model: str) -> LLMResult:
        usage = self._extract_usage(response, input_keys=("input_tokens",), output_keys=("output_tokens",))
        content_blocks: list[LLMContentBlock] = []
        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []
        output_items = self._extract_value(response, "output", []) or []
        for item in output_items:
            item_type = self._extract_value(item, "type")
            if item_type == "message":
                for part in self._extract_value(item, "content", []) or []:
                    part_type = self._extract_value(part, "type")
                    if part_type in {"output_text", "text", "input_text"}:
                        text = self._extract_value(part, "text")
                        if text:
                            text_parts.append(text)
                            content_blocks.append(LLMContentBlock(type="text", text=text))
            elif item_type == "function_call":
                arguments = self._extract_value(item, "arguments", "{}")
                parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                call_id = self._extract_value(item, "call_id") or self._extract_value(item, "id")
                name = self._extract_value(item, "name")
                tool_calls.append({"id": call_id, "name": name, "input": parsed_arguments})
                content_blocks.append(
                    LLMContentBlock(
                        type="tool_use",
                        id=call_id,
                        name=name,
                        input=parsed_arguments or {},
                    )
                )
        return LLMResult(
            provider=self.provider_name.value,
            content="\n".join(text_parts) if text_parts else self._extract_value(response, "output_text"),
            content_blocks=content_blocks,
            tool_calls=tool_calls,
            stop_reason=self._normalize_stop_reason(
                self._extract_value(response, "status"),
                tool_calls=tool_calls,
            ),
            usage=usage,
            latency_ms=0,
            model=model,
        )


GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-3.0-flash": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-2.0-flash": "gemini-2.0-flash",
}


def normalize_gemini_model(model: str) -> str:
    """Allow flexible model naming for Gemini models.

    Maps common aliases (e.g. ``gemini-3-flash``) to canonical API model names.
    Returns the input unchanged when no alias matches.
    """
    return GEMINI_MODEL_ALIASES.get(model, model)


class GeminiAdapter(BaseProviderAdapter):
    provider_name = ProviderName.GEMINI

    def __init__(self, client: Any | None = None, tracer: Any | None = None) -> None:
        super().__init__(client, tracer=tracer)
        # Native Gemini Content history — preserves thought signatures
        # across turns without base64 round-trips.  Cleared on pruning,
        # error recovery, or session boundaries.  When empty, create()
        # falls back to _convert_messages().
        self._native_history: list[Any] = []
        self._native_history_msg_count: int = 0  # messages synced so far

    def clear_native_history(self) -> None:
        """Invalidate native history — next call falls back to _convert_messages()."""
        self._native_history.clear()
        self._native_history_msg_count = 0

    def notify_messages_changed(self, new_count: int) -> None:
        """Called when the conversation is modified (pruning, injection).

        If the count doesn't match, native history is invalidated and the
        next create() call will rebuild from _convert_messages().
        """
        if new_count != self._native_history_msg_count:
            self.clear_native_history()

    def _build_default_client(self) -> Any:
        if genai is None:
            raise RuntimeError(
                "google-genai is not installed. Install project dependencies or inject a Gemini client."
            )
        # When GOOGLE_GENAI_USE_VERTEXAI is set, pass HttpOptions for Vertex AI.
        import os
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1"):
            from google.genai.types import HttpOptions
            return genai.Client(http_options=HttpOptions(api_version="v1"))
        return genai.Client()

    async def create(self, request: LLMRequest) -> LLMResult:
        client = self._get_client()
        model = normalize_gemini_model(request.model)
        config = self._build_config(request)

        # #3 Step 2 — thought-signature-drift mitigation.  The previous
        # attempt raised (likely an empty-response-for-retry case from
        # ``_parse_response``), and the leading hypothesis is that stale
        # ``thought_signature`` blocks carried forward in ``_native_history``
        # cause Gemini to silently refuse on the next request.  Invalidate
        # native history before the retry so the fallback builds fresh
        # ``contents`` via ``_convert_messages`` — that path strips
        # signatures as it goes.
        #
        # Narrow trigger (``attempt_number > 1``) keeps this scoped to real
        # recovery scenarios; first-attempt calls still benefit from the
        # native-history fast path.  The Step 1 ``gemini_empty_response``
        # telemetry will show whether this actually reduces the retry rate.
        if request.attempt_number > 1 and self._native_history:
            logger.debug(
                "Gemini retry (attempt %d): invalidating native_history "
                "to strip potentially stale thought_signatures",
                request.attempt_number,
            )
            self.clear_native_history()

        # Use native history when it's in sync with the message list.
        # When messages are pruned/modified externally, _native_history_msg_count
        # won't match and we fall back to _convert_messages().
        msg_count = len(request.messages)
        if self._native_history and self._native_history_msg_count == msg_count:
            contents = self._native_history
        else:
            contents = self._convert_messages(request.messages)
            # Seed native history from the converted messages.
            self._native_history = list(contents)
            self._native_history_msg_count = msg_count

        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                ),
                timeout=request.timeout_seconds,
            )
        else:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config,
                ),
                timeout=request.timeout_seconds,
            )

        # Append the native response Content to history for next turn.
        candidates = self._extract_value(response, "candidates", []) or []
        if candidates:
            response_content = self._extract_value(candidates[0], "content", None)
            if response_content is not None:
                self._native_history.append(response_content)
                self._native_history_msg_count += 1  # assistant message added

        result = self._parse_response(
            response, model=model, attempt_number=request.attempt_number,
        )
        # Populate LLMResult.parsed when structured output was requested.
        # Gemini guarantees the response text is valid JSON when
        # response_mime_type="application/json" was set, so json.loads is
        # safe here — not a parse-fail risk.  If the text is still empty
        # (deliberate STOP path already handled), .parsed stays None.
        if request.response_schema is not None and result.content:
            try:
                import json as _json
                result = result.model_copy(
                    update={"parsed": _json.loads(result.content)},
                )
            except (ValueError, TypeError) as exc:
                logger.debug(
                    "Gemini structured-output JSON parse failed (non-fatal): %s",
                    exc,
                )
        return result

    async def stream_create(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Stream Gemini response using generate_content_stream."""
        # Streaming doesn't yield a complete Content object, so we can't
        # maintain native history through streaming turns.  Invalidate it
        # and let the next create() call rebuild from _convert_messages().
        self.clear_native_history()
        client = self._get_client()
        model = normalize_gemini_model(request.model)
        contents = self._convert_messages(request.messages)
        config = self._build_config(request)

        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            response = await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            async for chunk in response:
                text = self._extract_value(chunk, "text", "")
                if text:
                    yield StreamChunk(text_delta=text)

                # Extract function calls from chunk
                for fc in self._extract_value(chunk, "function_calls", []) or []:
                    name = self._extract_value(fc, "name", "")
                    args = self._extract_value(fc, "args", {}) or {}
                    call_id = self._extract_value(fc, "id") or name
                    yield StreamChunk(tool_use_delta={"id": call_id, "name": name, "input": args})

                # Extract usage if present
                usage_meta = self._extract_value(chunk, "usage_metadata", None)
                if usage_meta is not None:
                    yield StreamChunk(usage_update={
                        "input_tokens": self._extract_value(usage_meta, "prompt_token_count", 0) or 0,
                        "output_tokens": self._extract_value(usage_meta, "candidates_token_count", 0) or 0,
                    })
        else:
            # Sync client — run in thread, but can't truly stream
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
            parsed = self._parse_response(
                response, model=model, attempt_number=request.attempt_number,
            )
            yield StreamChunk(
                text_delta=parsed.content or "",
                is_final=True,
                stop_reason=parsed.stop_reason.value,
            )
            return

        yield StreamChunk(is_final=True, stop_reason="end_turn")

    def _build_config(self, request: LLMRequest) -> Any:
        system_text = _flatten_system(request.system)
        # Structured output: Gemini's response_schema + response_mime_type
        # path.  Mutually exclusive with tool-use, so we skip tool
        # conversion entirely when a schema is requested.  (Partition-
        # planner style calls don't use tools anyway.)
        schema_cls = request.response_schema
        tools = [] if schema_cls is not None else self._convert_tools(request.tools)
        if genai_types is None:
            cfg: dict[str, Any] = {
                "system_instruction": system_text,
                "temperature": request.temperature,
                "max_output_tokens": request.max_tokens,
                "tools": None if schema_cls is not None else tools,
            }
            if schema_cls is not None:
                from maike.gateway.structured import schema_dict
                cfg["response_mime_type"] = "application/json"
                cfg["response_schema"] = schema_dict(schema_cls)
            return cfg
        # Enable thinking for models that support it (Gemini 2.5+, 3+).
        # Disabled when response_schema is set — structured-output path
        # does not support thinking on current Gemini models.
        thinking_config = None
        if (
            schema_cls is None
            and genai_types is not None
            and hasattr(genai_types, "ThinkingConfig")
        ):
            thinking_config = genai_types.ThinkingConfig(include_thoughts=True)

        # Penalty params are only supported by certain Gemini models;
        # flash/preview variants reject them with 400 INVALID_ARGUMENT.
        penalty_kwargs = {}
        model_lower = (request.model or "").lower()
        if "flash" not in model_lower and "preview" not in model_lower:
            penalty_kwargs["frequencyPenalty"] = 0.3
            penalty_kwargs["presencePenalty"] = 0.1

        structured_kwargs: dict[str, Any] = {}
        if schema_cls is not None:
            from maike.gateway.structured import schema_dict
            structured_kwargs["response_mime_type"] = "application/json"
            structured_kwargs["response_schema"] = schema_dict(schema_cls)
            # Drop tools — schema path is incompatible with tool-use.
            tools = []

        return genai_types.GenerateContentConfig(
            system_instruction=system_text,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            tools=tools or None,
            thinking_config=thinking_config,
            **penalty_kwargs,
            **structured_kwargs,
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[Any]:
        if not tools:
            return []
        if genai_types is None:
            return [
                {
                    "function_declarations": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["input_schema"],
                        }
                        for tool in tools
                    ]
                }
            ]
        declarations = [
            genai_types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"],
            )
            for tool in tools
        ]
        return [genai_types.Tool(function_declarations=declarations)]

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        contents: list[Any] = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            content = message.get("content", "")
            parts: list[Any] = []
            if isinstance(content, str):
                parts.append(self._text_part(content))
            else:
                for block in content:
                    block_type = block.get("type")
                    if block_type in {"text", "output_text", "thinking"}:
                        text = block.get("text") or block.get("content", "")
                        if text:
                            parts.append(
                                self._text_part(
                                    text,
                                    thought_signature=block.get("thought_signature"),
                                )
                            )
                    elif block_type == "tool_use":
                        parts.append(
                            self._function_call_part(
                                block["name"],
                                block.get("input", {}),
                                thought_signature=block.get("thought_signature"),
                            )
                        )
                    elif block_type == "tool_result":
                        parts.append(
                            self._function_response_part(
                                block.get("tool_name", "tool"),
                                block.get("content", ""),
                                is_error=block.get("is_error", False),
                            )
                        )
            if parts:
                contents.append(self._content(role, parts))
        return contents

    def _content(self, role: str, parts: list[Any]) -> Any:
        if genai_types is None:
            return {"role": role, "parts": parts}
        return genai_types.Content(role=role, parts=parts)

    @staticmethod
    def _decode_thought_signature(sig: Any) -> bytes | None:
        """Decode a thought_signature from its stored form.

        Signatures are base64-encoded strings after JSON round-trips
        (thread history). Raw bytes are passed through as-is.
        Returns None if the input is None or invalid.
        """
        if sig is None:
            return None
        if isinstance(sig, bytes):
            return sig
        if isinstance(sig, str):
            import base64
            try:
                return base64.b64decode(sig)
            except Exception:
                # Not valid base64 — might be a raw string from older format.
                try:
                    return sig.encode("utf-8", errors="surrogateescape")
                except Exception:
                    return None
        return None

    @staticmethod
    def _normalize_thought_signature(sig: Any) -> bytes | None:
        """Ensure thought_signature is proper bytes for genai_types.Part.

        Gemini returns raw bytes in a str container.  The genai SDK's
        Pydantic model expects *base64-encoded* bytes (or raw bytes).
        Normalise to raw bytes here so we can base64-encode when needed.
        """
        if sig is None:
            return None
        if isinstance(sig, bytes):
            return sig
        if isinstance(sig, str):
            try:
                return sig.encode("latin-1")
            except UnicodeEncodeError:
                # Contains Unicode replacement chars or other non-latin-1;
                # fall back to UTF-8 which can encode anything.
                return sig.encode("utf-8", errors="surrogateescape")
        try:
            return bytes(sig)
        except (TypeError, ValueError):
            return None

    def _text_part(self, text: str, *, thought_signature: Any = None) -> Any:
        # Text parts: thought_signature is optional, include if present.
        thought_signature = self._decode_thought_signature(thought_signature)
        if genai_types is None:
            payload: dict[str, Any] = {"text": text}
            if thought_signature is not None:
                payload["thoughtSignature"] = thought_signature
            return payload
        kwargs: dict[str, Any] = {"text": text}
        if thought_signature is not None:
            kwargs["thoughtSignature"] = thought_signature
        return genai_types.Part(**kwargs)

    def _function_call_part(
        self,
        name: str,
        args: dict[str, Any],
        *,
        thought_signature: Any = None,
    ) -> Any:
        thought_signature = self._decode_thought_signature(thought_signature)
        if genai_types is None or not hasattr(genai_types, "FunctionCall"):
            payload: dict[str, Any] = {"functionCall": {"name": name, "args": args}}
            if thought_signature is not None:
                payload["thoughtSignature"] = thought_signature
            return payload
        kwargs: dict[str, Any] = {
            "functionCall": genai_types.FunctionCall(name=name, args=args),
        }
        if thought_signature is not None:
            kwargs["thoughtSignature"] = thought_signature
        return genai_types.Part(**kwargs)

    def _function_response_part(self, name: str, content: Any, *, is_error: bool) -> Any:
        payload = {"result": content, "is_error": is_error}
        if genai_types is None:
            return {"functionResponse": {"name": name, "response": payload}}
        if hasattr(genai_types.Part, "from_function_response"):
            return genai_types.Part.from_function_response(name=name, response=payload)
        return {"functionResponse": {"name": name, "response": payload}}

    def _emit_empty_response_event(
        self,
        *,
        model: str,
        finish_reason: Any,
        is_deliberate_stop: bool,
        candidate_count: int,
        parts_count: int,
        part_text_lengths: list[int],
        has_any_function_call: bool,
        input_token_estimate: int | None,
        response_preview: str,
        attempt_number: int,
    ) -> None:
        """Emit a structured trace event for empty Gemini responses.

        Fires for both deliberate-STOP (``is_deliberate_stop=True``) and
        transient cases.  See ``_parse_response`` for the distinction.

        No-op if no tracer was wired into the adapter (tests etc.).
        """
        if self._tracer is None:
            return
        try:
            self._tracer.log_event(
                "gemini_empty_response",
                model=model,
                finish_reason=str(finish_reason) if finish_reason is not None else None,
                deliberate_stop=is_deliberate_stop,
                candidate_count=candidate_count,
                parts_count=parts_count,
                part_text_lengths=part_text_lengths,
                has_any_function_call=has_any_function_call,
                input_token_estimate=input_token_estimate,
                native_history_len=len(self._native_history),
                response_preview=response_preview,
                attempt_number=attempt_number,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostics must never break the call
            logger.debug("Failed to emit gemini_empty_response event (non-fatal): %s", exc)

    def _parse_response(
        self,
        response: Any,
        *,
        model: str,
        attempt_number: int = 1,
    ) -> LLMResult:
        usage = self._extract_usage(
            response,
            input_keys=("prompt_token_count", "input_tokens"),
            output_keys=("candidates_token_count", "output_tokens"),
        )
        content_blocks: list[LLMContentBlock] = []
        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []
        part_text_lengths: list[int] = []  # diagnostic — per-part text len
        finish_reason = None
        candidates = self._extract_value(response, "candidates", []) or []
        candidate_content = None
        parts_iter: list[Any] = []
        if candidates:
            first_candidate = candidates[0]
            finish_reason = self._extract_value(first_candidate, "finish_reason")
            candidate_content = self._extract_value(first_candidate, "content", None)
            parts_iter = self._extract_value(candidate_content, "parts", []) or []
            for part in parts_iter:
                # Extract thought signature — pass through as-is (no normalization).
                # Gemini 3 requires these echoed back exactly as received.
                thought_signature = self._extract_value(part, "thought_signature")
                if thought_signature is None:
                    thought_signature = self._extract_value(part, "thoughtSignature")

                text = self._extract_value(part, "text")
                is_thought = bool(self._extract_value(part, "thought"))
                # Track per-part text length (including empty strings) for
                # empty-response diagnostics — see the telemetry block below.
                part_text_lengths.append(len(text or ""))
                if text:
                    if not is_thought:
                        text_parts.append(text)
                    content_blocks.append(
                        LLMContentBlock(
                            type="thinking" if is_thought else "text",
                            text=text,
                            thought_signature=thought_signature,
                        )
                    )

                function_call = self._extract_value(part, "function_call")
                if function_call is None:
                    function_call = self._extract_value(part, "functionCall")
                if function_call is not None:
                    name = self._extract_value(function_call, "name")
                    args = self._extract_value(function_call, "args", {}) or {}
                    call_id = self._extract_value(function_call, "id") or name
                    tool_calls.append({"id": call_id, "name": name, "input": args})
                    content_blocks.append(
                        LLMContentBlock(
                            type="tool_use",
                            id=call_id,
                            name=name,
                            input=args,
                            thought_signature=thought_signature,
                        )
                    )

        if not content_blocks:
            raw_text = self._extract_value(response, "text", None)
            if raw_text:
                text_parts.append(raw_text)
                content_blocks.append(LLMContentBlock(type="text", text=raw_text))

        if not tool_calls:
            for function_call in self._extract_value(response, "function_calls", []) or []:
                name = self._extract_value(function_call, "name")
                args = self._extract_value(function_call, "args", {}) or {}
                call_id = self._extract_value(function_call, "id") or name
                tool_calls.append({"id": call_id, "name": name, "input": args})
                content_blocks.append(
                    LLMContentBlock(type="tool_use", id=call_id, name=name, input=args)
                )

        # Validate: if response was non-empty but we got nothing, decide
        # whether to retry or accept.  Gemini 3.1 Pro sometimes returns
        # parts with text='' — if finish_reason is STOP, the model
        # deliberately chose to say nothing (not transient).  Only retry
        # when finish_reason is missing or indicates truncation.
        if not content_blocks and not tool_calls:
            raw_str = str(response)[:500]
            finish_str = str(finish_reason).upper() if finish_reason else ""
            is_deliberate_stop = "STOP" in finish_str
            if len(raw_str) > 50:  # Response had substantial content
                # Diagnostic event — fires for BOTH deliberate-STOP (accepted)
                # and transient (raised-for-retry) cases, with
                # ``deliberate_stop`` distinguishing them.  Used by the
                # #3 investigation to distinguish stale-signature drift from
                # context-length masquerade from SDK-layer pathology.
                #
                # Payload is structured-only; DO NOT include raw prompt text
                # or signature bytes (privacy + size).
                self._emit_empty_response_event(
                    model=model,
                    finish_reason=finish_reason,
                    is_deliberate_stop=is_deliberate_stop,
                    candidate_count=len(candidates),
                    parts_count=len(parts_iter),
                    part_text_lengths=part_text_lengths,
                    has_any_function_call=any(
                        self._extract_value(p, "function_call") is not None
                        or self._extract_value(p, "functionCall") is not None
                        for p in parts_iter
                    ),
                    input_token_estimate=(
                        usage.input_tokens if usage is not None else None
                    ),
                    response_preview=raw_str[:300],
                    attempt_number=attempt_number,
                )
                logger.warning(
                    "Gemini response parsing produced empty result from non-empty response. "
                    "Model: %s. finish_reason: %s. Response preview: %s",
                    model,
                    finish_reason,
                    raw_str[:200],
                )
                if not is_deliberate_stop:
                    # Transient: no finish_reason or truncation — retry.
                    raise ValueError(
                        f"Gemini returned empty text/tool_calls despite non-empty response "
                        f"(model={model}, finish_reason={finish_reason}). "
                        f"This is a transient API issue — retrying."
                    )

        # Deduplicate repeated text — Gemini sometimes generates the same
        # sentence or paragraph multiple times in a single response.
        joined_text = "\n".join(text_parts) if text_parts else None
        if joined_text:
            joined_text = _deduplicate_repeated_text(joined_text)

        return LLMResult(
            provider=self.provider_name.value,
            content=joined_text,
            content_blocks=content_blocks,
            tool_calls=tool_calls,
            stop_reason=self._normalize_stop_reason(finish_reason, tool_calls=tool_calls),
            usage=usage,
            latency_ms=0,
            model=model,
        )


# ---------------------------------------------------------------------------
# Ollama adapter (local models via OpenAI-compatible /v1 endpoint)
# ---------------------------------------------------------------------------


class OllamaAdapter(BaseProviderAdapter):
    """Adapter for Ollama-served local models (Gemma, Llama, Qwen, etc.).

    Uses the OpenAI Python SDK pointed at Ollama's ``/v1/chat/completions``
    endpoint.  No additional packages needed — the ``openai`` SDK handles
    everything including tool calling and streaming.
    """

    provider_name = ProviderName.OLLAMA

    def _build_default_client(self) -> Any:
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai is not installed. The Ollama adapter uses the OpenAI SDK "
                "pointed at Ollama's /v1 endpoint.  Install: pip install openai"
            )
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return AsyncOpenAI(
            api_key="ollama",  # Ollama requires no auth; placeholder for SDK
            base_url=f"{host}/v1",
        )

    async def create(self, request: LLMRequest) -> LLMResult:
        client = self._get_client()
        messages = self._convert_messages(request.messages)
        # Chat Completions API takes system prompt as first message.
        system_text = _flatten_system(request.system)
        if system_text:
            messages = [{"role": "system", "content": system_text}, *messages]
        tools = self._convert_tools(request.tools)
        # Structured output: Ollama's /v1 endpoint supports the OpenAI
        # ``response_format={"type":"json_object"}`` shape.  This does not
        # enforce a schema at the provider — newer Ollama models honor
        # schema-shaped prompts but older ones may not.  We append a
        # schema hint to the last user message as a fallback.
        create_kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "tools": tools or openai_NOT_GIVEN,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_schema is not None:
            create_kwargs["response_format"] = {"type": "json_object"}
            # Ollama ignores unknown tools+response_format combo differently
            # per model; drop tools to give the schema path the best shot.
            create_kwargs["tools"] = openai_NOT_GIVEN
        response = await asyncio.wait_for(
            client.chat.completions.create(**create_kwargs),
            timeout=request.timeout_seconds,
        )
        result = self._parse_response(response, model=request.model)
        # Best-effort JSON parse — model may produce close-but-not-quite
        # valid JSON.  Degrade to parsed=None gracefully.
        if request.response_schema is not None and result.content:
            try:
                import json as _json
                result = result.model_copy(
                    update={"parsed": _json.loads(result.content)},
                )
            except (ValueError, TypeError) as exc:
                logger.debug(
                    "Ollama structured-output JSON parse failed (non-fatal): %s",
                    exc,
                )
        return result

    async def stream_create(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        client = self._get_client()
        messages = self._convert_messages(request.messages)
        system_text = _flatten_system(request.system)
        if system_text:
            messages = [{"role": "system", "content": system_text}, *messages]
        tools = self._convert_tools(request.tools)
        # stream_options.include_usage asks Ollama (via the OpenAI-compatible
        # endpoint) to emit a final chunk carrying the prompt/completion
        # token counts.  Without it, intermediate chunks have no usage
        # data and the gateway's `final_usage` stays None — leading to
        # output_tokens=0 in agent_runs metadata.
        stream = await client.chat.completions.create(
            model=request.model,
            messages=messages,
            tools=tools or openai_NOT_GIVEN,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        text_parts: list[str] = []
        tool_call_acc: dict[int, dict[str, Any]] = {}
        last_usage: dict[str, int] | None = None
        async for chunk in stream:
            # The usage-only final chunk has empty `choices` — drain it for
            # the token counts before the loop's choices[0] access would
            # IndexError.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                last_usage = {
                    "input_tokens": getattr(chunk_usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(chunk_usage, "completion_tokens", 0) or 0,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            # Text delta.
            if delta.content:
                text_parts.append(delta.content)
                yield StreamChunk(text_delta=delta.content)
            # Tool call deltas — accumulate across chunks.
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {
                            "id": tc.id or f"call_{idx}",
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": "",
                        }
                    if tc.function and tc.function.arguments:
                        tool_call_acc[idx]["arguments"] += tc.function.arguments
            # Check for finish — but keep iterating to drain the trailing
            # usage chunk that follows when stream_options.include_usage
            # is set.
            finish = chunk.choices[0].finish_reason

        # Build final tool calls.
        tool_calls: list[dict[str, Any]] = []
        for acc in tool_call_acc.values():
            try:
                parsed_args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append({
                "id": acc["id"], "name": acc["name"], "input": parsed_args,
            })
            yield StreamChunk(tool_use_delta={
                "id": acc["id"], "name": acc["name"], "input": parsed_args,
            })

        # Emit usage on the FINAL chunk so the gateway's final_usage gets
        # populated.  Fall back to a length-based estimate if the server
        # didn't return usage (shouldn't happen with include_usage=True,
        # but Ollama versions vary).
        if last_usage is None:
            joined = "".join(text_parts)
            last_usage = {"input_tokens": 0, "output_tokens": max(1, len(joined) // 4) if joined else 0}

        stop_reason_value = (StopReason.TOOL_USE if tool_calls else StopReason.END_TURN).value
        yield StreamChunk(
            usage_update=last_usage,
            is_final=True,
            stop_reason=stop_reason_value,
        )

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert mAIke's block-based message format to Chat Completions format."""
        result: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue
            # Block-based content.
            text_parts: list[str] = []
            tool_calls_out: list[dict] = []
            tool_results: list[dict] = []

            for block in content:
                block_type = block.get("type")
                if block_type in ("text", "output_text", "thinking"):
                    text = block.get("text") or block.get("content", "")
                    if text:
                        text_parts.append(text)
                elif block_type == "tool_use":
                    tool_calls_out.append({
                        "id": block.get("id", f"call_{block.get('name', 'unknown')}"),
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if not isinstance(tool_content, str):
                        tool_content = json.dumps(tool_content)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", block.get("id", "")),
                        "content": tool_content,
                    })

            # Emit assistant message with tool calls.
            if tool_calls_out:
                msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls_out}
                if text_parts:
                    msg["content"] = "\n".join(text_parts)
                result.append(msg)
            elif tool_results:
                # Tool results are individual messages in Chat Completions format.
                if text_parts:
                    result.append({"role": role, "content": "\n".join(text_parts)})
                result.extend(tool_results)
            elif text_parts:
                result.append({"role": role, "content": "\n".join(text_parts)})

        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Convert mAIke tool schemas to OpenAI function-calling format."""
        if not tools:
            return None
        converted = []
        for tool in tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return converted

    def _parse_response(self, response: Any, *, model: str) -> LLMResult:
        """Parse a Chat Completions response into LLMResult."""
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        text = message.content if message and message.content else None
        finish_reason = choice.finish_reason if choice else "stop"

        # Parse tool calls.
        tool_calls: list[dict[str, Any]] = []
        content_blocks: list[LLMContentBlock] = []
        if text:
            content_blocks.append(LLMContentBlock(type="text", text=text))
        if message and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id or tc.function.name,
                    "name": tc.function.name,
                    "input": args,
                })
                content_blocks.append(LLMContentBlock(
                    type="tool_use",
                    id=tc.id or tc.function.name,
                    name=tc.function.name,
                    input=args,
                ))

        # Usage.
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return LLMResult(
            provider=self.provider_name.value,
            content=text,
            content_blocks=content_blocks,
            tool_calls=tool_calls,
            stop_reason=self._normalize_stop_reason(finish_reason, tool_calls=tool_calls),
            usage=usage,
            latency_ms=0,
            model=model,
        )

    async def aclose(self) -> None:
        """No persistent connection to close for Ollama."""
        pass


# Sentinel for OpenAI SDK's "not given" parameter.
try:
    from openai import NOT_GIVEN as openai_NOT_GIVEN
except ImportError:
    openai_NOT_GIVEN = None  # type: ignore[assignment]


def build_provider_adapter(
    provider_name: str | ProviderName,
    client: Any | None = None,
    tracer: Any | None = None,
) -> ProviderAdapter:
    provider = resolve_provider_name(provider_name)
    mapping = {
        ProviderName.ANTHROPIC: AnthropicAdapter,
        ProviderName.OPENAI: OpenAIAdapter,
        ProviderName.GEMINI: GeminiAdapter,
        ProviderName.OLLAMA: OllamaAdapter,
    }
    return mapping[provider](client=client, tracer=tracer)
