"""Normalized gateway over provider-specific SDK adapters."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections.abc import AsyncGenerator
from typing import Any

from maike.atoms.llm import LLMCallRecord, LLMContentBlock, LLMResult, StreamChunk, StopReason, TokenUsage
from maike.constants import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    LLM_RETRY_BASE_DELAY_SECONDS,
    LLM_RETRY_JITTER_SECONDS,
    LLM_RETRY_MAX_ATTEMPTS,
    LLM_RETRY_MAX_DELAY_SECONDS,
    LLM_RETRY_503_COOLDOWN_SECONDS,
    PROVIDER_API_KEY_ENV_VARS,
    PROVIDER_FALLBACK_ORDER,
    SESSION_BUDGET_PRECHECK_MARGIN,
    context_limit_for_model,
    default_model_for_provider,
    model_for_tier,
    pricing_for_model,
)
from maike.cost.tracker import CostTracker
from maike.gateway.providers import (
    LLMRequest,
    build_provider_adapter,
    resolve_model_name,
    resolve_provider_name,
)
from maike.observability.tracer import Tracer
from maike.tools.context import peek_current_agent_context
from maike.utils import dedupe_preserve_order, estimate_message_tokens

logger = logging.getLogger(__name__)


class LLMGateway:
    def __init__(
        self,
        cost_tracker: CostTracker,
        tracer: Tracer,
        provider_name: str = DEFAULT_PROVIDER,
        client: Any | None = None,
        fallback_providers: list[str] | None = None,
        silent: bool = False,
    ) -> None:
        self.cost_tracker = cost_tracker
        self.tracer = tracer
        self.provider_name = resolve_provider_name(provider_name).value
        # Pass the tracer into the adapter so adapters can emit structured
        # diagnostic events (e.g. GeminiAdapter's gemini_empty_response).
        # silent=True gateways still forward to the tracer — silence only
        # suppresses llm_start/llm_call TUI events, not low-level diagnostics.
        self.adapter = build_provider_adapter(
            self.provider_name, client=client, tracer=tracer,
        )
        self._fallback_providers = fallback_providers or []
        self._fallback_adapters: dict[str, Any] = {}
        self._silent = silent

    async def aclose(self) -> None:
        await self.adapter.aclose()
        for adapter in self._fallback_adapters.values():
            try:
                await adapter.aclose()
            except Exception:
                pass

    def clear_native_history(self) -> None:
        """Invalidate provider-native conversation history (Gemini only)."""
        if hasattr(self.adapter, "clear_native_history"):
            self.adapter.clear_native_history()

    def notify_messages_changed(self, count: int) -> None:
        """Notify adapter that conversation messages were modified externally."""
        if hasattr(self.adapter, "notify_messages_changed"):
            self.adapter.notify_messages_changed(count)

    def resolve_model_for_tier(self, tier: str) -> str:
        """Return the model name for *tier* ('default', 'cheap', 'strong') on this gateway's provider."""
        return model_for_tier(self.provider_name, tier)

    async def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        response_schema: Any = None,
    ) -> LLMResult:
        resolved_model = resolve_model_name(self.provider_name, model)
        estimated_input_tokens = estimate_message_tokens(messages)

        # Context window guard — fail early with a clear message instead of
        # letting the provider return an opaque error.
        model_limit = context_limit_for_model(resolved_model)
        if estimated_input_tokens + max_tokens > model_limit:
            raise RuntimeError(
                f"Estimated payload ({estimated_input_tokens:,} input + "
                f"{max_tokens:,} output = {estimated_input_tokens + max_tokens:,} tokens) "
                f"exceeds the context window of {resolved_model} "
                f"({model_limit:,} tokens). "
                "Consider reducing artifact sizes or enabling context compression."
            )

        projected_call_cost = self._calculate_cost_usd(resolved_model, estimated_input_tokens, max_tokens)
        self.cost_tracker.check_projected_session_budget(
            projected_call_cost,
            safety_margin=SESSION_BUDGET_PRECHECK_MARGIN,
        )
        max_attempts = LLM_RETRY_MAX_ATTEMPTS
        result: LLMResult | None = None
        latency_ms = 0
        attempt = 0
        last_status_code: int | None = None
        request = LLMRequest(
            system=system,
            messages=messages,
            tools=tools or [],
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            response_schema=response_schema,
        )
        while attempt < max_attempts:
            attempt += 1
            # Stamp the retry counter on the request so adapters can
            # correlate diagnostic events with retry position.  Safe to
            # replace() each iteration — LLMRequest is frozen.
            from dataclasses import replace as _dc_replace
            attempt_request = _dc_replace(request, attempt_number=attempt)
            if not self._silent:
                self.tracer.log_llm_start(
                    self.provider_name,
                    resolved_model,
                    payload={
                        "message_count": len(messages),
                        "tool_count": len(tools or []),
                        "estimated_input_tokens": estimated_input_tokens,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                )
            start = time.monotonic()
            try:
                result = await self.adapter.create(request=attempt_request)
                latency_ms = int((time.monotonic() - start) * 1000)
                break
            except Exception as exc:
                latency_ms = int((time.monotonic() - start) * 1000)
                status_code = self._status_code_for_exception(exc)
                last_status_code = status_code
                retryable = self._is_retryable_exception(exc, status_code=status_code)
                if not retryable or attempt >= max_attempts:
                    # 503 cooldown: one extra attempt after a longer wait
                    if retryable and status_code == 503 and attempt == max_attempts:
                        retry_after = self._extract_retry_after(exc)
                        delay_seconds = LLM_RETRY_503_COOLDOWN_SECONDS
                        self.tracer.log_event(
                            "llm_retry",
                            provider=self.provider_name,
                            model=resolved_model,
                            attempt=attempt,
                            max_attempts=max_attempts + 1,
                            status_code=status_code,
                            latency_ms=latency_ms,
                            delay_seconds=delay_seconds,
                            retry_after=retry_after,
                            error=str(exc),
                        )
                        await asyncio.sleep(delay_seconds)
                        # One final attempt
                        attempt += 1
                        if not self._silent:
                            self.tracer.log_llm_start(
                                self.provider_name,
                                resolved_model,
                                payload={
                                    "message_count": len(messages),
                                    "tool_count": len(tools or []),
                                    "estimated_input_tokens": estimated_input_tokens,
                                    "attempt": attempt,
                                    "max_attempts": max_attempts + 1,
                                },
                            )
                        start = time.monotonic()
                        try:
                            attempt_request = _dc_replace(request, attempt_number=attempt)
                            result = await self.adapter.create(request=attempt_request)
                            latency_ms = int((time.monotonic() - start) * 1000)
                            break
                        except Exception as final_exc:
                            latency_ms = int((time.monotonic() - start) * 1000)
                            final_status = self._status_code_for_exception(final_exc)
                            self.tracer.log_event(
                                "llm_error",
                                provider=self.provider_name,
                                model=resolved_model,
                                attempt=attempt,
                                max_attempts=max_attempts + 1,
                                retryable=False,
                                status_code=final_status,
                                latency_ms=latency_ms,
                                error=str(final_exc),
                            )
                            if self._is_fallback_eligible(final_exc, final_status):
                                return await self._attempt_fallback(
                                    original_exc=final_exc,
                                    system=system,
                                    messages=messages,
                                    tools=tools,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                )
                            raise
                    else:
                        self.tracer.log_event(
                            "llm_error",
                            provider=self.provider_name,
                            model=resolved_model,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            retryable=retryable,
                            status_code=status_code,
                            latency_ms=latency_ms,
                            error=str(exc),
                        )
                        if self._is_fallback_eligible(exc, status_code):
                            return await self._attempt_fallback(
                                original_exc=exc,
                                system=system,
                                messages=messages,
                                tools=tools,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                        raise
                retry_after = self._extract_retry_after(exc)
                delay_seconds = self._calculate_delay(attempt, status_code, retry_after)
                self.tracer.log_event(
                    "llm_retry",
                    provider=self.provider_name,
                    model=resolved_model,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    delay_seconds=delay_seconds,
                    retry_after=retry_after,
                    error=str(exc),
                )
                await asyncio.sleep(delay_seconds)
        assert result is not None
        cost_usd = self._calculate_cost_usd(resolved_model, result.usage.input_tokens, result.usage.output_tokens)
        result = result.model_copy(
            update={
                "provider": self.provider_name,
                "model": resolved_model,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            }
        )
        record = self._build_call_record(result)
        self._record_context_usage(record)
        self.cost_tracker.record(record)
        if not self._silent:
            self.tracer.log_llm_call(
                record,
                payload={
                    "assistant_output": self._assistant_output(result),
                    "thinking": self._thinking_output(result),
                    "tool_calls": result.tool_calls,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                },
            )
        self.cost_tracker.check_session_budget()
        return result

    async def stream_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream LLM response token by token.

        Yields ``StreamChunk`` objects as they arrive.  The final chunk has
        ``is_final=True`` and carries the fully-accumulated ``LLMResult`` in
        ``accumulated_result``.  Cost tracking and tracing happen at the end,
        identical to ``call()``.

        If the adapter does not support streaming, or if streaming fails
        mid-response, falls back to the non-streaming ``call()`` and yields
        the complete result as a single final chunk.
        """
        resolved_model = resolve_model_name(self.provider_name, model)
        estimated_input_tokens = estimate_message_tokens(messages)

        # Context window guard
        model_limit = context_limit_for_model(resolved_model)
        if estimated_input_tokens + max_tokens > model_limit:
            raise RuntimeError(
                f"Estimated payload ({estimated_input_tokens:,} input + "
                f"{max_tokens:,} output = {estimated_input_tokens + max_tokens:,} tokens) "
                f"exceeds the context window of {resolved_model} "
                f"({model_limit:,} tokens). "
                "Consider reducing artifact sizes or enabling context compression."
            )

        projected_call_cost = self._calculate_cost_usd(resolved_model, estimated_input_tokens, max_tokens)
        self.cost_tracker.check_projected_session_budget(
            projected_call_cost,
            safety_margin=SESSION_BUDGET_PRECHECK_MARGIN,
        )

        # Check if adapter supports streaming
        if not hasattr(self.adapter, "stream_create"):
            result = await self.call(
                system=system, messages=messages, tools=tools,
                model=model, temperature=temperature, max_tokens=max_tokens,
            )
            yield StreamChunk(
                text_delta=result.content or "",
                is_final=True,
                stop_reason=result.stop_reason.value,
                accumulated_result=result,
            )
            return

        request = LLMRequest(
            system=system,
            messages=messages,
            tools=tools or [],
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
        )

        if not self._silent:
            self.tracer.log_llm_start(
                self.provider_name,
                resolved_model,
                payload={
                    "message_count": len(messages),
                    "tool_count": len(tools or []),
                    "estimated_input_tokens": estimated_input_tokens,
                "attempt": 1,
                "max_attempts": 1,
                "streaming": True,
            },
        )

        start = time.monotonic()
        text_parts: list[str] = []
        tool_call_deltas: list[dict[str, Any]] = []
        final_usage: dict[str, Any] | None = None
        final_stop_reason: str | None = None

        try:
            stream_iter = self.adapter.stream_create(request=request).__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
                    )
                except StopAsyncIteration:
                    break
                if chunk.text_delta:
                    text_parts.append(chunk.text_delta)
                if chunk.tool_use_delta is not None:
                    tool_call_deltas.append(chunk.tool_use_delta)
                if chunk.usage_update is not None:
                    final_usage = chunk.usage_update
                if chunk.stop_reason is not None:
                    final_stop_reason = chunk.stop_reason
                if not chunk.is_final:
                    yield chunk
        except asyncio.TimeoutError:
            logger.warning(
                "Streaming timed out for %s/%s after %ds — no chunk received.",
                self.provider_name, resolved_model, DEFAULT_LLM_TIMEOUT_SECONDS,
            )
            # If we already have partial content, yield what we have
            if text_parts:
                yield StreamChunk(
                    text_delta="",
                    is_final=True,
                    stop_reason="timeout",
                )
                final_stop_reason = "timeout"
            else:
                raise  # re-raise so fallback below handles it
        except Exception as stream_exc:
            # Streaming failed mid-response — fall back to non-streaming
            logger.warning(
                "Streaming failed for %s/%s: %s. Falling back to non-streaming call.",
                self.provider_name, resolved_model, stream_exc,
            )
            result = await self.call(
                system=system, messages=messages, tools=tools,
                model=model, temperature=temperature, max_tokens=max_tokens,
            )
            yield StreamChunk(
                text_delta=result.content or "",
                is_final=True,
                stop_reason=result.stop_reason.value,
                accumulated_result=result,
            )
            return

        latency_ms = int((time.monotonic() - start) * 1000)

        content_text = "".join(text_parts) if text_parts else None
        content_blocks: list[LLMContentBlock] = []
        if content_text:
            content_blocks.append(LLMContentBlock(type="text", text=content_text))

        tool_calls: list[dict[str, Any]] = []
        for tc in tool_call_deltas:
            tool_calls.append(tc)
            content_blocks.append(LLMContentBlock(
                type="tool_use",
                id=tc.get("id"),
                name=tc.get("name"),
                input=tc.get("input", {}),
            ))

        usage = TokenUsage(
            input_tokens=(final_usage or {}).get("input_tokens", estimated_input_tokens),
            output_tokens=(final_usage or {}).get("output_tokens", 0),
        )

        stop_reason = self._normalize_stream_stop_reason(final_stop_reason, tool_calls=tool_calls)

        cost_usd = self._calculate_cost_usd(resolved_model, usage.input_tokens, usage.output_tokens)
        result = LLMResult(
            provider=self.provider_name,
            content=content_text,
            content_blocks=content_blocks,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            model=resolved_model,
        )

        record = self._build_call_record(result)
        self._record_context_usage(record)
        self.cost_tracker.record(record)
        if not self._silent:
            self.tracer.log_llm_call(
                record,
                payload={
                    "assistant_output": self._assistant_output(result),
                    "thinking": self._thinking_output(result),
                    "tool_calls": result.tool_calls,
                    "attempt": 1,
                    "max_attempts": 1,
                    "streaming": True,
                },
            )
        self.cost_tracker.check_session_budget()

        yield StreamChunk(
            text_delta="",
            is_final=True,
            stop_reason=result.stop_reason.value,
            accumulated_result=result,
        )

    def _normalize_stream_stop_reason(
        self, raw: str | None, *, tool_calls: list[dict[str, Any]]
    ) -> StopReason:
        """Delegate to the adapter's existing stop-reason normalizer."""
        return self.adapter._normalize_stop_reason(raw, tool_calls=tool_calls)

    def _build_call_record(self, result: LLMResult) -> LLMCallRecord:
        ctx = peek_current_agent_context()
        session_id = None
        agent_id = None
        stage_name = None
        tool_profile = None
        if ctx is not None:
            session_id = ctx.metadata.get("session_id")
            agent_id = ctx.agent_id
            stage_name = ctx.stage_name
            tool_profile = ctx.tool_profile
        return LLMCallRecord(
            provider=result.provider or self.provider_name,
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            stop_reason=result.stop_reason.value,
            session_id=session_id,
            agent_id=agent_id,
            stage_name=stage_name,
            tool_profile=tool_profile,
        )

    def _calculate_cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = pricing_for_model(model)
        if pricing is None:
            return 0.0
        return (
            input_tokens / 1_000_000 * pricing.input_per_million_usd
            + output_tokens / 1_000_000 * pricing.output_per_million_usd
        )

    def _record_context_usage(self, record: LLMCallRecord) -> None:
        ctx = peek_current_agent_context()
        if ctx is None:
            return
        usage = ctx.metadata.setdefault(
            "llm_usage",
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "providers": [],
                "models": [],
            },
        )
        usage["calls"] = int(usage.get("calls", 0)) + 1
        usage["input_tokens"] = int(usage.get("input_tokens", 0)) + record.input_tokens
        usage["output_tokens"] = int(usage.get("output_tokens", 0)) + record.output_tokens
        usage["total_tokens"] = int(usage.get("total_tokens", 0)) + record.total_tokens
        usage["cost_usd"] = float(usage.get("cost_usd", 0.0)) + record.cost_usd
        usage["providers"] = dedupe_preserve_order([*usage.get("providers", []), record.provider])
        usage["models"] = dedupe_preserve_order([*usage.get("models", []), record.model])

    def _assistant_output(self, result: LLMResult) -> str:
        if result.content:
            return result.content
        blocks = [block.text for block in result.content_blocks if block.text and block.type != "thinking"]
        return "\n".join(blocks).strip()

    def _thinking_output(self, result: LLMResult) -> str:
        blocks = [block.text for block in result.content_blocks if block.text and block.type == "thinking"]
        return "\n".join(blocks).strip()

    def _status_code_for_exception(self, exc: Exception) -> int | None:
        # Try common attribute names across providers.
        for attr in ("status_code", "code", "status"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val
        # Some SDKs nest it under .response.status_code
        response = getattr(exc, "response", None)
        nested = getattr(response, "status_code", None)
        if isinstance(nested, int):
            return nested
        return None

    def _is_retryable_exception(self, exc: Exception, *, status_code: int | None) -> bool:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        if status_code in {408, 429}:
            return True
        # Gemini empty response parsing: transient API issue, worth retrying.
        if isinstance(exc, ValueError) and "empty text/tool_calls" in str(exc):
            return True
        return bool(status_code is not None and status_code >= 500)

    def _extract_retry_after(self, exc: Exception) -> float | None:
        """Extract Retry-After delay from exception response headers, if present."""
        try:
            response = getattr(exc, "response", None)
            if response is None:
                return None
            headers = getattr(response, "headers", None)
            if headers is None:
                return None
            # Anthropic uses retry-after-ms
            retry_after_ms = headers.get("retry-after-ms")
            if retry_after_ms is not None:
                return float(retry_after_ms) / 1000.0
            # Standard Retry-After header (case-insensitive lookup)
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after is not None:
                return float(retry_after)
        except Exception:
            pass
        return None

    def _is_fallover_worthy(self, exc: Exception, status_code: int | None) -> bool:
        """Return True if this error warrants trying a fallback provider."""
        if status_code in {401, 403}:
            return True
        if isinstance(exc, (ConnectionError, ConnectionRefusedError, ConnectionResetError)):
            return True
        exc_name = type(exc).__name__.lower()
        if "connection" in exc_name or "apiconnection" in exc_name:
            return True
        return False

    async def _try_fallback(
        self,
        request: LLMRequest,
        original_error: Exception,
        estimated_input_tokens: int,
    ) -> LLMResult | None:
        """Attempt the call on fallback providers. Returns None if all fail."""
        for fb_provider in self._fallback_providers:
            fb_name = resolve_provider_name(fb_provider).value
            if fb_name == self.provider_name:
                continue
            try:
                if fb_name not in self._fallback_adapters:
                    self._fallback_adapters[fb_name] = build_provider_adapter(fb_name)
                fb_adapter = self._fallback_adapters[fb_name]
                fb_model = resolve_model_name(fb_name, None)
                fb_request = LLMRequest(
                    system=request.system,
                    messages=request.messages,
                    tools=request.tools,
                    model=fb_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    timeout_seconds=request.timeout_seconds,
                )
                self.tracer.log_event(
                    "provider_fallover",
                    from_provider=self.provider_name,
                    to_provider=fb_name,
                    to_model=fb_model,
                    reason=str(original_error),
                )
                start = time.monotonic()
                result = await fb_adapter.create(request=fb_request)
                latency_ms = int((time.monotonic() - start) * 1000)
                cost_usd = self._calculate_cost_usd(fb_model, result.usage.input_tokens, result.usage.output_tokens)
                return result.model_copy(
                    update={
                        "provider": fb_name,
                        "model": fb_model,
                        "latency_ms": latency_ms,
                        "cost_usd": cost_usd,
                    }
                )
            except Exception as fb_exc:
                self.tracer.log_event(
                    "provider_fallover_failed",
                    provider=fb_name,
                    error=str(fb_exc),
                )
                continue
        return None

    def _calculate_delay(self, attempt: int, status_code: int | None, retry_after: float | None) -> float:
        """Compute the delay before the next retry attempt."""
        # If server told us when to retry, respect it (capped at max_delay)
        if retry_after is not None and 0 < retry_after <= LLM_RETRY_MAX_DELAY_SECONDS:
            return retry_after

        # Exponential backoff with jitter
        delay = min(
            LLM_RETRY_BASE_DELAY_SECONDS ** (attempt + 1) + random.uniform(0, LLM_RETRY_JITTER_SECONDS),
            LLM_RETRY_MAX_DELAY_SECONDS,
        )
        return delay

    # ── Provider fallback ────────────────────────────────────────────

    @staticmethod
    def _is_fallback_eligible(exc: Exception, status_code: int | None) -> bool:
        """Return True if the error warrants trying a different provider."""
        if isinstance(exc, ConnectionError):
            return True
        return status_code in {401, 403}

    @staticmethod
    def _available_fallback_providers(current_provider: str) -> list[str]:
        """Return providers from the fallback order that have an API key set, excluding *current_provider*."""
        available: list[str] = []
        for provider in PROVIDER_FALLBACK_ORDER:
            if provider == current_provider:
                continue
            env_vars = PROVIDER_API_KEY_ENV_VARS.get(provider, [])
            if any(os.environ.get(var) for var in env_vars):
                available.append(provider)
        return available

    async def _attempt_fallback(
        self,
        *,
        original_exc: Exception,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        """Try the call once on the first available fallback provider.

        Raises the *original_exc* if no fallback is available or the fallback also fails.
        """
        fallback_providers = self._available_fallback_providers(self.provider_name)
        if not fallback_providers:
            raise original_exc

        fallback_name = fallback_providers[0]
        fallback_model = default_model_for_provider(fallback_name)

        logger.warning(
            "Provider %s failed with %s. Falling back to %s.",
            self.provider_name,
            original_exc,
            fallback_name,
        )
        self.tracer.log_event(
            "llm_fallback",
            original_provider=self.provider_name,
            fallback_provider=fallback_name,
            fallback_model=fallback_model,
            error=str(original_exc),
        )

        fallback_adapter = build_provider_adapter(fallback_name)
        try:
            request = LLMRequest(
                system=system,
                messages=messages,
                tools=tools or [],
                model=fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            )
            start = time.monotonic()
            result = await fallback_adapter.create(request=request)
            latency_ms = int((time.monotonic() - start) * 1000)

            cost_usd = self._calculate_cost_usd(fallback_model, result.usage.input_tokens, result.usage.output_tokens)
            result = result.model_copy(
                update={
                    "provider": fallback_name,
                    "model": fallback_model,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "metadata": {"fallback_provider": fallback_name},
                }
            )

            record = self._build_call_record(result)
            self._record_context_usage(record)
            self.cost_tracker.record(record)
            if not self._silent:
                self.tracer.log_llm_call(
                    record,
                    payload={
                        "assistant_output": self._assistant_output(result),
                        "tool_calls": result.tool_calls,
                        "attempt": 1,
                        "max_attempts": 1,
                        "fallback_provider": fallback_name,
                    },
                )
            self.cost_tracker.check_session_budget()
            return result
        except Exception:
            # Fallback also failed — raise the original error so callers see the root cause.
            raise original_exc
        finally:
            await fallback_adapter.aclose()
