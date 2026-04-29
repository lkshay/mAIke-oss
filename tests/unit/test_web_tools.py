"""Tests for WebSearch and WebFetch tools."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maike.atoms.tool import RiskLevel, ToolResult
from maike.tools.registry import ToolRegistry
from maike.tools.web import (
    _html_to_markdown,
    _normalize_fetch_url,
    _normalize_search_query,
    register_web_tools,
)


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_web_tools(registry, MagicMock())
    return registry


class TestRegistration:
    def test_websearch_registered(self):
        registry = _make_registry()
        tool = registry.get("WebSearch")
        assert tool is not None
        assert tool.risk_level == RiskLevel.READ

    def test_webfetch_registered(self):
        registry = _make_registry()
        tool = registry.get("WebFetch")
        assert tool is not None
        assert tool.risk_level == RiskLevel.READ


class TestWebSearchNoKey:
    """When no API key is configured, WebSearch falls back to the DuckDuckGo
    HTML scrape.  This test confirms the fallback is wired — it does NOT
    make a real network call (we patch _ddg_search)."""

    def test_no_key_falls_through_to_ddg(self):
        registry = ToolRegistry()
        register_web_tools(registry, MagicMock())
        tool = registry.get("WebSearch")

        env = {k: v for k, v in os.environ.items()
               if k not in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY",
                            "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID")}
        fake_ok = MagicMock(success=True, output="## Search results for: test\n\n1. **X**\n   https://x\n")
        mock_ddg = AsyncMock(return_value=fake_ok)
        with patch.dict(os.environ, env, clear=True), \
             patch("maike.tools.web._ddg_search", new=mock_ddg):
            result = asyncio.run(tool.fn(query="test"))
            assert mock_ddg.called

    def test_no_key_ddg_parse_failure_surfaces_cleanly(self):
        """If DDG returns unparseable HTML, the tool returns a clear error
        that points the user at the keyed-backend options.
        """
        registry = ToolRegistry()
        register_web_tools(registry, MagicMock())
        tool = registry.get("WebSearch")

        env = {k: v for k, v in os.environ.items()
               if k not in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY",
                            "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID")}

        class _FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k):
                class _R:
                    text = "<html><body>no results here</body></html>"
                    status_code = 200
                    def raise_for_status(self): pass
                return _R()

        with patch.dict(os.environ, env, clear=True), \
             patch("httpx.AsyncClient", _FakeClient):
            result = asyncio.run(tool.fn(query="test"))
            assert result.success is False
            assert "TAVILY_API_KEY" in result.output
            assert "BRAVE_SEARCH_API_KEY" in result.output


class TestWebSearchRateLimit:
    def test_rate_limit_after_5_calls(self):
        registry = ToolRegistry()
        register_web_tools(registry, MagicMock())
        tool = registry.get("WebSearch")

        # Mock DDG (the no-key fallback) so we don't make real network calls.
        env = {k: v for k, v in os.environ.items()
               if k not in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY",
                            "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID")}
        fake_ok = MagicMock(success=True, output="ok")
        with patch.dict(os.environ, env, clear=True), \
             patch("maike.tools.web._ddg_search", new=AsyncMock(return_value=fake_ok)):
            results = []
            for i in range(7):
                r = asyncio.run(tool.fn(query=f"test {i}"))
                results.append(r)

            # 6th and 7th should fail with rate limit message.
            assert "rate limit" in results[5].output.lower()
            assert "rate limit" in results[6].output.lower()


class TestWebFetchUrlValidation:
    def test_rejects_file_scheme(self):
        registry = _make_registry()
        tool = registry.get("WebFetch")
        result = asyncio.run(tool.fn(url="file:///etc/passwd"))
        assert result.success is False
        assert "scheme" in result.output.lower()

    def test_rejects_ftp_scheme(self):
        registry = _make_registry()
        tool = registry.get("WebFetch")
        result = asyncio.run(tool.fn(url="ftp://example.com/file"))
        assert result.success is False
        assert "scheme" in result.output.lower()

    def test_accepts_https(self):
        """https URLs should pass scheme validation (may fail at DNS/fetch)."""
        registry = _make_registry()
        tool = registry.get("WebFetch")
        # Will fail at DNS but should pass URL validation
        result = asyncio.run(tool.fn(url="https://this-domain-does-not-exist-xyzzy.example"))
        assert result.success is False
        # Should fail at DNS, not scheme validation
        assert "scheme" not in result.output.lower()


class TestWebFetchPrivateIp:
    def test_blocks_loopback(self):
        registry = _make_registry()
        tool = registry.get("WebFetch")

        with patch("maike.tools.web.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 0)),
            ]
            result = asyncio.run(tool.fn(url="https://internal.example.com"))
            assert result.success is False
            assert "private" in result.output.lower() or "reserved" in result.output.lower()

    def test_blocks_private_ip(self):
        registry = _make_registry()
        tool = registry.get("WebFetch")

        with patch("maike.tools.web.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (2, 1, 6, "", ("10.0.0.1", 0)),
            ]
            result = asyncio.run(tool.fn(url="https://internal.example.com"))
            assert result.success is False


class TestHtmlToMarkdown:
    def test_strips_script_tags(self):
        html = "<p>Hello</p><script>alert('xss')</script><p>World</p>"
        md = _html_to_markdown(html)
        assert "alert" not in md
        assert "Hello" in md
        assert "World" in md

    def test_converts_headings(self):
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        md = _html_to_markdown(html)
        assert "# Title" in md
        assert "## Subtitle" in md

    def test_converts_links(self):
        html = '<a href="https://example.com">Click here</a>'
        md = _html_to_markdown(html)
        assert "[Click here](https://example.com)" in md

    def test_converts_bold(self):
        html = "<strong>important</strong>"
        md = _html_to_markdown(html)
        assert "**important**" in md

    def test_converts_code(self):
        html = "<code>x = 1</code>"
        md = _html_to_markdown(html)
        assert "`x = 1`" in md

    def test_strips_remaining_tags(self):
        html = "<div><span>text</span></div>"
        md = _html_to_markdown(html)
        assert "<" not in md
        assert "text" in md

    def test_decodes_entities(self):
        html = "&amp; &lt; &gt; &quot;"
        md = _html_to_markdown(html)
        assert "& < > \"" in md


class TestWebFetchTruncation:
    def test_truncates_at_max_chars(self):
        registry = _make_registry()
        tool = registry.get("WebFetch")

        long_content = "x" * 20000
        mock_response = MagicMock()
        mock_response.text = long_content
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = MagicMock()

        with patch("maike.tools.web.socket.getaddrinfo") as mock_gai, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_gai.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = asyncio.run(tool.fn(url="https://example.com", max_chars=500))
            assert result.success is True
            assert len(result.output) <= 600  # 500 + truncation message
            assert "Truncated" in result.output


# ---------------------------------------------------------------------------
# Tavily backend
# ---------------------------------------------------------------------------


class TestTavilyBackend:
    def test_tavily_success(self):
        """Non-empty Tavily response is rendered with title/url/content."""
        from maike.tools.web import _tavily_search

        tavily_payload = {
            "query": "asyncio tutorial",
            "results": [
                {"title": "asyncio docs", "url": "https://docs.python.org/3/library/asyncio.html",
                 "content": "asyncio is a library to write concurrent code.", "score": 0.9},
                {"title": "Real Python", "url": "https://realpython.com/async-io/",
                 "content": "A complete walkthrough.", "score": 0.8},
            ],
        }

        class _FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return tavily_payload

        class _FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k): return _FakeResp()

        with patch("httpx.AsyncClient", _FakeClient):
            result = asyncio.run(_tavily_search("asyncio tutorial", 2, "fake-key"))

        assert result.success is True
        assert "asyncio docs" in result.output
        assert "https://docs.python.org" in result.output
        assert "concurrent code" in result.output
        assert result.metadata["backend"] == "tavily"

    def test_tavily_empty_results(self):
        from maike.tools.web import _tavily_search

        class _FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"results": []}

        class _FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k): return _FakeResp()

        with patch("httpx.AsyncClient", _FakeClient):
            result = asyncio.run(_tavily_search("no hits", 3, "fake-key"))
        assert result.success is True
        assert "No results" in result.output

    def test_tavily_truncates_long_content(self):
        """Tavily returns cleaned content that can be multi-KB per result.
        Ensure we cap each result's content to ~500 chars."""
        from maike.tools.web import _tavily_search

        long_content = "X" * 3000
        payload = {"results": [
            {"title": "big", "url": "https://x", "content": long_content},
        ]}

        class _FakeResp:
            def raise_for_status(self): pass
            def json(self): return payload

        class _FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k): return _FakeResp()

        with patch("httpx.AsyncClient", _FakeClient):
            result = asyncio.run(_tavily_search("big", 1, "fake-key"))
        # Content truncated to 500 + "…" suffix.
        assert "X" * 3000 not in result.output
        assert "…" in result.output

    def test_tavily_http_error(self):
        import httpx
        from maike.tools.web import _tavily_search

        class _FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k):
                raise httpx.HTTPError("connection refused")

        with patch("httpx.AsyncClient", _FakeClient):
            result = asyncio.run(_tavily_search("x", 1, "fake-key"))
        assert result.success is False
        assert "Tavily" in result.output


# ---------------------------------------------------------------------------
# DuckDuckGo fallback
# ---------------------------------------------------------------------------


class TestDDGBackend:
    SAMPLE_HTML = """
    <html><body>
    <div class="results">
      <div class="result">
        <h2 class="result__title">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">One Title</a>
        </h2>
        <a class="result__snippet" href="//x">First snippet content.</a>
      </div>
      <div class="result">
        <h2 class="result__title">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo">Two Title</a>
        </h2>
        <a class="result__snippet" href="//x">Second snippet content.</a>
      </div>
    </div>
    </body></html>
    """

    def _patch_ddg_with(self, html):
        class _FakeResp:
            text = html
            status_code = 200
            def raise_for_status(self): pass

        class _FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k): return _FakeResp()

        return patch("httpx.AsyncClient", _FakeClient)

    def test_ddg_success(self):
        from maike.tools.web import _ddg_search
        with self._patch_ddg_with(self.SAMPLE_HTML):
            result = asyncio.run(_ddg_search("example", 5))
        assert result.success is True
        assert "One Title" in result.output
        assert "Two Title" in result.output
        assert "https://example.com/one" in result.output
        assert "First snippet" in result.output
        assert result.metadata["backend"] == "duckduckgo"
        assert result.metadata["result_count"] == 2

    def test_ddg_respects_num_results(self):
        from maike.tools.web import _ddg_search
        with self._patch_ddg_with(self.SAMPLE_HTML):
            result = asyncio.run(_ddg_search("example", 1))
        assert result.metadata["result_count"] == 1
        assert "Two Title" not in result.output

    def test_ddg_unwraps_redirects(self):
        from maike.tools.web import _ddg_unwrap_url
        wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath&rut=abc"
        assert _ddg_unwrap_url(wrapped) == "https://example.com/path"

    def test_ddg_no_parseable_results(self):
        from maike.tools.web import _ddg_search
        with self._patch_ddg_with("<html><body>DDG removed</body></html>"):
            result = asyncio.run(_ddg_search("x", 3))
        assert result.success is False
        assert "HTML structure may have changed" in result.output
        # Points user at keyed backends
        assert "TAVILY_API_KEY" in result.output

    def test_ddg_http_error(self):
        import httpx
        from maike.tools.web import _ddg_search

        class _FakeClient:
            def __init__(self, *a, **k): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k):
                raise httpx.HTTPError("network unreachable")

        with patch("httpx.AsyncClient", _FakeClient):
            result = asyncio.run(_ddg_search("x", 3))
        assert result.success is False
        assert "DuckDuckGo" in result.output
        # Steers users toward keyed backends.
        assert "TAVILY_API_KEY" in result.output


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------


class TestBackendPriority:
    """Verify the priority order: Tavily > Brave > Google > DDG."""

    def _make_tool(self):
        registry = ToolRegistry()
        register_web_tools(registry, MagicMock())
        return registry.get("WebSearch")

    def test_tavily_wins_over_brave_and_google(self):
        tool = self._make_tool()
        t = AsyncMock(return_value=MagicMock(success=True, output="tavily"))
        b = AsyncMock(return_value=MagicMock(success=True, output="brave"))
        g = AsyncMock(return_value=MagicMock(success=True, output="google"))
        with patch.dict(os.environ, {
            "TAVILY_API_KEY": "t",
            "BRAVE_SEARCH_API_KEY": "b",
            "GOOGLE_SEARCH_API_KEY": "g",
            "GOOGLE_SEARCH_ENGINE_ID": "cx",
        }, clear=True), \
             patch("maike.tools.web._tavily_search", new=t), \
             patch("maike.tools.web._brave_search", new=b), \
             patch("maike.tools.web._google_search", new=g):
            asyncio.run(tool.fn(query="q"))
        assert t.called
        assert not b.called
        assert not g.called

    def test_brave_wins_over_google_when_tavily_absent(self):
        tool = self._make_tool()
        b = AsyncMock(return_value=MagicMock(success=True, output="brave"))
        g = AsyncMock(return_value=MagicMock(success=True, output="google"))
        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        env.update({
            "BRAVE_SEARCH_API_KEY": "b",
            "GOOGLE_SEARCH_API_KEY": "g",
            "GOOGLE_SEARCH_ENGINE_ID": "cx",
        })
        with patch.dict(os.environ, env, clear=True), \
             patch("maike.tools.web._brave_search", new=b), \
             patch("maike.tools.web._google_search", new=g):
            asyncio.run(tool.fn(query="q"))
        assert b.called
        assert not g.called

    def test_ddg_fires_when_all_keys_absent(self):
        tool = self._make_tool()
        d = AsyncMock(return_value=MagicMock(success=True, output="ddg"))
        env = {k: v for k, v in os.environ.items()
               if k not in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY",
                            "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID")}
        with patch.dict(os.environ, env, clear=True), \
             patch("maike.tools.web._ddg_search", new=d):
            asyncio.run(tool.fn(query="q"))
        assert d.called

    def test_google_requires_both_keys(self):
        """GOOGLE_SEARCH_API_KEY without GOOGLE_SEARCH_ENGINE_ID → falls to DDG."""
        tool = self._make_tool()
        g = AsyncMock(return_value=MagicMock(success=True, output="google"))
        d = AsyncMock(return_value=MagicMock(success=True, output="ddg"))
        env = {k: v for k, v in os.environ.items()
               if k not in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY",
                            "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID")}
        env["GOOGLE_SEARCH_API_KEY"] = "g"  # only one of the pair
        with patch.dict(os.environ, env, clear=True), \
             patch("maike.tools.web._google_search", new=g), \
             patch("maike.tools.web._ddg_search", new=d):
            asyncio.run(tool.fn(query="q"))
        assert not g.called
        assert d.called


# ===========================================================================
# Fix 3: query/URL normalization helpers
# ===========================================================================


class TestNormalizeSearchQuery:
    def test_lowercases(self):
        assert _normalize_search_query("Python AsyncIO") == "python asyncio"

    def test_collapses_whitespace(self):
        assert _normalize_search_query("python   asyncio") == "python asyncio"
        assert _normalize_search_query("  python\tasyncio\n") == "python asyncio"

    def test_strips_trailing_punctuation(self):
        assert _normalize_search_query("python asyncio?") == "python asyncio"
        assert _normalize_search_query("python asyncio.") == "python asyncio"
        assert _normalize_search_query("python asyncio!") == "python asyncio"

    def test_handles_empty(self):
        assert _normalize_search_query("") == ""
        assert _normalize_search_query("   ") == ""

    def test_normalizes_to_same_for_equivalent_queries(self):
        """Variations a real agent produces should collide on cache lookup."""
        assert (
            _normalize_search_query("Python AsyncIO tutorial")
            == _normalize_search_query("python asyncio tutorial")
            == _normalize_search_query("python   asyncio   tutorial?")
        )


class TestNormalizeFetchUrl:
    def test_strips_fragment(self):
        assert (
            _normalize_fetch_url("https://example.com/foo#section")
            == "https://example.com/foo"
        )

    def test_preserves_query_string(self):
        # Query strings DO affect server response — preserve them.
        u = "https://example.com/foo?bar=1"
        assert _normalize_fetch_url(u) == u

    def test_preserves_trailing_slash(self):
        # Some servers serve different content for / vs no-/.
        assert (
            _normalize_fetch_url("https://example.com/foo/")
            != _normalize_fetch_url("https://example.com/foo")
        )

    def test_handles_empty(self):
        assert _normalize_fetch_url("") == ""


# ===========================================================================
# Fix 3: WebSearch caching — duplicate queries don't burn budget
# ===========================================================================


def _no_key_env() -> dict:
    """Env scrubbed of search backend keys → DDG fallback path."""
    return {
        k: v for k, v in os.environ.items()
        if k not in (
            "TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY",
            "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID",
        )
    }


def _make_tool(name: str):
    registry = ToolRegistry()
    register_web_tools(registry, MagicMock())
    return registry.get(name)


class TestWebSearchCaching:
    """The transcript bug: turns 19 + 20 fired the same query twice — burning
    2 of 5 search credits.  Same query in a session must hit cache."""

    def _make_search_result(self, output: str = "## Search results for: q\n") -> ToolResult:
        return ToolResult(
            tool_name="WebSearch",
            success=True,
            output=output,
            raw_output=output,
            metadata={"query": "q"},
        )

    def test_identical_query_returns_cached(self):
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._make_search_result("## Search results for: q\nrow1"))
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            r1 = asyncio.run(tool.fn(query="python asyncio"))
            r2 = asyncio.run(tool.fn(query="python asyncio"))

        # First call hits the backend; second is served from cache.
        assert ddg.call_count == 1
        assert r1.success is True
        assert r2.success is True
        # Cached response must be flagged so the LLM knows.
        assert r2.metadata.get("cached") is True
        assert "Cached" in r2.output
        # Original content is preserved in the cached response.
        assert "row1" in r2.output

    def test_normalized_query_collisions_hit_cache(self):
        """Variations the agent produces should collide on cache lookup."""
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._make_search_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            asyncio.run(tool.fn(query="Python AsyncIO"))
            asyncio.run(tool.fn(query="python   asyncio"))
            asyncio.run(tool.fn(query="python asyncio?"))

        # Three variations, ONE backend call.
        assert ddg.call_count == 1

    def test_cache_does_not_burn_rate_limit(self):
        """The fix for the original bug (turns 19-20): duplicate queries
        must not count against the 5-call budget.

        The transcript fired the same query twice and burned 2/5 credits.
        After this fix, 10 duplicates burn 1 credit — leaving 4 credits
        for distinct queries (total 5 = budget cap)."""
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._make_search_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            # 10 duplicates — only the first hits the backend.
            for _ in range(10):
                asyncio.run(tool.fn(query="same query"))
            # 4 distinct queries now succeed (budget 1 + 4 = 5).
            distinct_results = [
                asyncio.run(tool.fn(query=f"distinct {i}"))
                for i in range(4)
            ]
            # The 5th distinct query exhausts budget.
            limit_result = asyncio.run(tool.fn(query="distinct 4"))

        # 1 backend call for the duplicate + 4 distinct that fit = 5.
        # The over-budget call did NOT hit the backend.
        assert ddg.call_count == 5
        assert all(r.success for r in distinct_results)
        assert limit_result.success is False
        assert "rate limit" in limit_result.output.lower()

    def test_different_num_results_distinct_cache_keys(self):
        """num_results is part of the cache key — different counts re-fetch."""
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._make_search_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            asyncio.run(tool.fn(query="q", num_results=3))
            asyncio.run(tool.fn(query="q", num_results=5))

        assert ddg.call_count == 2

    def test_failed_search_not_cached(self):
        """Transient failures shouldn't poison the cache — agent should be
        able to retry after a network blip."""
        tool = _make_tool("WebSearch")
        fail = ToolResult(tool_name="WebSearch", success=False, output="error")
        ok = self._make_search_result("good")
        # First call fails, second call succeeds.
        ddg = AsyncMock(side_effect=[fail, ok])
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            r1 = asyncio.run(tool.fn(query="q"))
            r2 = asyncio.run(tool.fn(query="q"))

        assert ddg.call_count == 2  # retry happened
        assert r1.success is False
        assert r2.success is True
        assert r2.metadata.get("cached") is not True

    def test_failed_search_does_not_burn_budget(self):
        """Failures don't decrement the budget either — agent gets full 5
        successful searches even after intermittent failures."""
        tool = _make_tool("WebSearch")
        fail = ToolResult(tool_name="WebSearch", success=False, output="error")
        ok = self._make_search_result("good")
        # 3 failures, then 5 successes — should all pass.
        ddg = AsyncMock(side_effect=[fail, fail, fail, ok, ok, ok, ok, ok])
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            results = [
                asyncio.run(tool.fn(query=f"q{i}"))
                for i in range(8)
            ]

        # 5 successful searches happened — budget intact despite failures.
        assert sum(1 for r in results if r.success) == 5

    def test_per_session_cache_isolated(self):
        """Each register_web_tools call gets its own cache — sessions don't
        share."""
        tool_a = _make_tool("WebSearch")
        tool_b = _make_tool("WebSearch")  # second registration = second session
        ddg = AsyncMock(return_value=self._make_search_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            asyncio.run(tool_a.fn(query="same"))
            asyncio.run(tool_b.fn(query="same"))

        # Each session called the backend independently.
        assert ddg.call_count == 2


class TestWebSearchGraduatedNudge:
    """Calls 3 and 4 should append a nudge encouraging Grep / AskUser before
    the budget is exhausted."""

    def _ok_result(self, output: str = "ok") -> ToolResult:
        return ToolResult(
            tool_name="WebSearch", success=True, output=output, raw_output=output,
        )

    def test_no_nudge_on_first_two_calls(self):
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._ok_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            r1 = asyncio.run(tool.fn(query="q1"))
            r2 = asyncio.run(tool.fn(query="q2"))

        assert "Note:" not in r1.output
        assert "Note:" not in r2.output

    def test_nudge_on_call_three(self):
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._ok_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            asyncio.run(tool.fn(query="q1"))
            asyncio.run(tool.fn(query="q2"))
            r3 = asyncio.run(tool.fn(query="q3"))

        assert "3/5" in r3.output
        assert "Grep" in r3.output

    def test_nudge_on_call_four(self):
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._ok_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            for i in range(3):
                asyncio.run(tool.fn(query=f"q{i}"))
            r4 = asyncio.run(tool.fn(query="q4"))

        assert "4/5" in r4.output
        assert "Grep" in r4.output

    def test_no_nudge_on_call_five(self):
        """Call 5 is the last — no point nudging there."""
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._ok_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            for i in range(4):
                asyncio.run(tool.fn(query=f"q{i}"))
            r5 = asyncio.run(tool.fn(query="q5"))

        assert "Note:" not in r5.output

    def test_no_nudge_on_cache_hit(self):
        """Cached responses don't re-emit the nudge — the agent has already
        seen it once."""
        tool = _make_tool("WebSearch")
        ddg = AsyncMock(return_value=self._ok_result())
        with patch.dict(os.environ, _no_key_env(), clear=True), \
             patch("maike.tools.web._ddg_search", new=ddg):
            # Burn 3 calls.
            asyncio.run(tool.fn(query="q1"))
            asyncio.run(tool.fn(query="q2"))
            r3 = asyncio.run(tool.fn(query="q3"))
            # Re-fire q3 — cached return, no additional nudge.
            r3_cached = asyncio.run(tool.fn(query="q3"))

        assert "3/5" in r3.output
        assert "Note:" not in r3_cached.output.replace(r3.output, "")
        assert r3_cached.metadata.get("cached") is True


# ===========================================================================
# Fix 3: WebFetch caching — duplicate URLs don't re-download
# ===========================================================================


class TestWebFetchCaching:
    """Transcript turns 21-22: same README fetched twice in a row."""

    def _patch_fetch(self, body: str = "<p>Hello</p>"):
        """Set up DNS + httpx mocks so a fetch returns *body*."""
        gai = patch("maike.tools.web.socket.getaddrinfo",
                    return_value=[(2, 1, 6, "", ("93.184.216.34", 0))])
        client_cls = MagicMock()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.text = body
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        client_cls.return_value = mock_client
        return gai, patch("httpx.AsyncClient", new=client_cls), mock_client

    def test_identical_url_hits_cache(self):
        tool = _make_tool("WebFetch")
        gai, client_patch, mock_client = self._patch_fetch("<p>readme content</p>")
        with gai, client_patch:
            r1 = asyncio.run(tool.fn(url="https://example.com/readme"))
            r2 = asyncio.run(tool.fn(url="https://example.com/readme"))

        assert mock_client.get.call_count == 1
        assert r1.success is True
        assert r2.success is True
        assert r2.metadata.get("cached") is True
        assert "Cached" in r2.output
        assert "readme content" in r2.output

    def test_fragment_normalized(self):
        """Same URL with different fragments hits cache."""
        tool = _make_tool("WebFetch")
        gai, client_patch, mock_client = self._patch_fetch()
        with gai, client_patch:
            asyncio.run(tool.fn(url="https://example.com/page#one"))
            asyncio.run(tool.fn(url="https://example.com/page#two"))

        assert mock_client.get.call_count == 1

    def test_different_path_distinct_cache(self):
        tool = _make_tool("WebFetch")
        gai, client_patch, mock_client = self._patch_fetch()
        with gai, client_patch:
            asyncio.run(tool.fn(url="https://example.com/a"))
            asyncio.run(tool.fn(url="https://example.com/b"))

        assert mock_client.get.call_count == 2

    def test_failed_fetch_not_cached(self):
        """Network errors / 404s should not poison the cache."""
        tool = _make_tool("WebFetch")
        # First fetch: scheme rejection (no network call).  Second: success.
        gai, client_patch, mock_client = self._patch_fetch("<p>good</p>")
        with gai, client_patch:
            # Fail with bad scheme.
            r1 = asyncio.run(tool.fn(url="ftp://example.com/foo"))
            assert r1.success is False
            # Re-fetch with a valid URL — should hit the network, not the cache.
            r2 = asyncio.run(tool.fn(url="https://example.com/foo"))
            assert r2.success is True
            # Re-fetch the SAME good URL — should hit cache now.
            r3 = asyncio.run(tool.fn(url="https://example.com/foo"))
            assert r3.metadata.get("cached") is True

        assert mock_client.get.call_count == 1

    def test_cache_eviction_fifo(self):
        """Cache cap prevents unbounded growth — oldest entry evicted first."""
        from maike.tools.web import _FETCH_CACHE_MAX

        tool = _make_tool("WebFetch")
        gai, client_patch, mock_client = self._patch_fetch()
        with gai, client_patch:
            # Fill cache + 1 to trigger eviction of the first entry.
            for i in range(_FETCH_CACHE_MAX + 1):
                asyncio.run(tool.fn(url=f"https://example.com/page-{i}"))
            # The first URL was evicted — re-fetching it should hit the network.
            asyncio.run(tool.fn(url="https://example.com/page-0"))

        # All initial fills + the evicted re-fetch.
        assert mock_client.get.call_count == _FETCH_CACHE_MAX + 2

    def test_per_session_fetch_cache_isolated(self):
        tool_a = _make_tool("WebFetch")
        tool_b = _make_tool("WebFetch")
        gai, client_patch, mock_client = self._patch_fetch()
        with gai, client_patch:
            asyncio.run(tool_a.fn(url="https://example.com/foo"))
            asyncio.run(tool_b.fn(url="https://example.com/foo"))

        # Different sessions = independent caches.
        assert mock_client.get.call_count == 2
