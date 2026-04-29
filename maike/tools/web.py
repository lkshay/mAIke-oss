"""Web search and fetch tools for external knowledge access."""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from urllib.parse import urlparse, urlunparse

from maike.atoms.tool import RiskLevel, ToolResult, ToolSchema
from maike.runtime.protocol import ExecutionRuntime
from maike.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SEARCH_RATE_LIMIT = 5
_FETCH_TIMEOUT = 15.0
_DEFAULT_NUM_RESULTS = 5
_MAX_NUM_RESULTS = 10
_DEFAULT_MAX_CHARS = 10_000
_MAX_MAX_CHARS = 50_000

# WebFetch cache cap.  Bounded so a long session walking many doc pages
# doesn't grow unbounded memory; FIFO eviction keeps the most recent N.
_FETCH_CACHE_MAX = 20

# Calls at which WebSearch appends a graduated rate-limit nudge to the
# successful result.  Fires once per call (idempotent — cached responses
# don't re-add the nudge).  The goal is to warn the agent BEFORE it burns
# the entire budget on duplicates / unfocused queries.
_SEARCH_NUDGE_CALLS: frozenset[int] = frozenset({3, 4})


def _normalize_search_query(query: str) -> str:
    """Normalize a search query for cache lookup.

    Lowercase, collapse internal whitespace, strip trailing sentence
    punctuation.  Designed to catch the dominant "agent fired the same query
    twice" pattern (turns 19-20 of the 76-turn transcript) without being so
    aggressive that semantically distinct queries collide.
    """
    if not query:
        return ""
    return " ".join(query.lower().split()).rstrip(".,;:!?")


def _normalize_fetch_url(url: str) -> str:
    """Normalize a URL for cache lookup.

    Strips the fragment (``#anchor``) since fragments don't affect server
    response.  Does NOT normalize trailing slash — some servers return
    different content for ``/foo`` vs ``/foo/``.  Does NOT lower-case the
    path — paths are case-sensitive on most servers.
    """
    if not url:
        return ""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment=""))


def _mark_result_cached(result: ToolResult, kind: str) -> ToolResult:
    """Return a clone of *result* with a `cached` marker in metadata + output.

    The output marker is what the LLM actually sees — it signals "you already
    asked this; look elsewhere" without pretending the data is new.
    """
    metadata = dict(result.metadata or {})
    metadata["cached"] = True
    cached_marker = (
        f"\n\n[Cached: identical {kind} already performed this session — "
        "no API call made.  If you need fresh content, vary the "
        f"{'query' if kind == 'search' else 'URL'}.]"
    )
    new_output = (result.output or "") + cached_marker
    return result.model_copy(update={"output": new_output, "metadata": metadata})


def _append_search_nudge(result: ToolResult, call_number: int) -> ToolResult:
    """Append a graduated rate-limit nudge to a successful search result.

    Fires at calls 3 and 4 (of 5) so the agent sees a budget warning while
    there's still room to pivot to Grep / AskUser.  No-op for failed results
    or for cached returns (which don't increment the counter).
    """
    nudge = (
        f"\n\n[Note: this is your {call_number}/{_SEARCH_RATE_LIMIT} web "
        "search this session.  Prefer Grep on the local codebase, or "
        "AskUser for clarification — once you reach "
        f"{_SEARCH_RATE_LIMIT} you cannot search again.]"
    )
    new_output = (result.output or "") + nudge
    return result.model_copy(update={"output": new_output})


def register_web_tools(registry: ToolRegistry, runtime: ExecutionRuntime) -> None:
    """Register WebSearch and WebFetch tools.

    Each registration creates fresh per-session state:
      * a search call counter and result cache (5-call budget; identical
        queries hit the cache for free),
      * a fetch URL cache (no call budget; cache prevents redundant
        re-downloads of the same URL).
    """

    _search_call_count = 0
    # Cache key: (normalized_query, num_results).  Misses re-search and
    # increment the budget; hits return the cached payload with a `cached`
    # marker and DO NOT count against the budget.
    _search_cache: dict[tuple[str, int], ToolResult] = {}
    # Cache key: normalized URL.  No call budget for fetch — the cache
    # alone prevents the redundant re-download we saw at turns 21-22 of the
    # transcript (same README fetched twice).
    _fetch_cache: dict[str, ToolResult] = {}

    async def web_search(query: str, num_results: int = _DEFAULT_NUM_RESULTS) -> ToolResult:
        """Search the web.

        Backend priority (highest-signal first):
          1. Tavily (LLM-optimized, cleaned content) — if ``TAVILY_API_KEY`` set.
          2. Brave Search — if ``BRAVE_SEARCH_API_KEY`` set.
          3. Google Custom Search — if both ``GOOGLE_SEARCH_API_KEY`` and
             ``GOOGLE_SEARCH_ENGINE_ID`` are set.
          4. DuckDuckGo HTML fallback — no key required.  Fragile (scraped
             HTML, no SLA) but means ``WebSearch`` always has SOMETHING to
             offer even in a fresh install.  Users wanting reliable search
             should set one of the keyed backends above.

        Identical queries within a session return cached results without
        incrementing the rate-limit counter.  Calls 3 and 4 also receive a
        graduated nudge encouraging Grep / AskUser before exhaustion.
        """
        nonlocal _search_call_count

        num_results = min(max(1, num_results), _MAX_NUM_RESULTS)

        # Cache hit — return without consuming budget.  Catches the dominant
        # "agent fired the same query twice in a row" failure mode.
        normalized = _normalize_search_query(query)
        cache_key = (normalized, num_results)
        cached = _search_cache.get(cache_key)
        if cached is not None:
            return _mark_result_cached(cached, "search")

        # Cache miss — check budget BEFORE incrementing.
        prospective_count = _search_call_count + 1
        if prospective_count > _SEARCH_RATE_LIMIT:
            return ToolResult(
                tool_name="WebSearch",
                success=False,
                output=(
                    f"WebSearch rate limit reached ({_SEARCH_RATE_LIMIT} calls "
                    "per session).  Use Grep for local codebase search, or "
                    "AskUser to clarify what you actually need."
                ),
            )

        try:
            import httpx  # noqa: F401 — import-probe; backends import their own httpx
        except ImportError:
            return ToolResult(
                tool_name="WebSearch",
                success=False,
                output="WebSearch unavailable — httpx is not installed. Run: pip install httpx",
            )

        tavily_key = os.environ.get("TAVILY_API_KEY")
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        google_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
        google_cx = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")

        if tavily_key:
            result = await _tavily_search(query, num_results, tavily_key)
        elif brave_key:
            result = await _brave_search(query, num_results, brave_key)
        elif google_key and google_cx:
            result = await _google_search(query, num_results, google_key, google_cx)
        else:
            # No-key fallback — DDG HTML scrape.  Always available; lower quality.
            result = await _ddg_search(query, num_results)

        # Only count and cache successful searches.  Failed searches (network
        # error, parse failure) shouldn't burn budget — the agent will likely
        # want to retry or pivot.
        if result.success:
            _search_call_count = prospective_count
            _search_cache[cache_key] = result
            if prospective_count in _SEARCH_NUDGE_CALLS:
                result = _append_search_nudge(result, prospective_count)
        return result

    async def web_fetch(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> ToolResult:
        """Fetch a URL and return its content as markdown.

        Identical URLs within a session return cached content without
        re-fetching.  Catches the redundant-fetch pattern at turns 21-22 of
        the transcript (same README fetched twice in a row).
        """
        max_chars = min(max(100, max_chars), _MAX_MAX_CHARS)

        # Cache hit — return cached content without re-fetching.
        cache_key = _normalize_fetch_url(url)
        cached = _fetch_cache.get(cache_key)
        if cached is not None:
            return _mark_result_cached(cached, "fetch")

        try:
            import httpx
        except ImportError:
            return ToolResult(
                tool_name="WebFetch",
                success=False,
                output="WebFetch unavailable — httpx is not installed. Run: pip install httpx",
            )

        # Validate URL scheme.
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                tool_name="WebFetch",
                success=False,
                output=f"Invalid URL scheme: {parsed.scheme!r}. Only http:// and https:// are allowed.",
            )

        if not parsed.hostname:
            return ToolResult(
                tool_name="WebFetch",
                success=False,
                output="Invalid URL: no hostname found.",
            )

        # Block private/loopback IPs (SSRF protection).
        try:
            addrs = socket.getaddrinfo(parsed.hostname, None)
            for _, _, _, _, sockaddr in addrs:
                ip = ipaddress.ip_address(sockaddr[0])
                if ip.is_private or ip.is_loopback or ip.is_reserved:
                    return ToolResult(
                        tool_name="WebFetch",
                        success=False,
                        output=f"Blocked: {parsed.hostname} resolves to private/reserved IP {ip}.",
                    )
        except (socket.gaierror, ValueError) as exc:
            return ToolResult(
                tool_name="WebFetch",
                success=False,
                output=f"DNS resolution failed for {parsed.hostname}: {exc}",
            )

        try:
            async with httpx.AsyncClient(
                timeout=_FETCH_TIMEOUT,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                resp = await client.get(url, headers={"User-Agent": "mAIke-agent/1.0"})
                resp.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult(
                tool_name="WebFetch",
                success=False,
                output=f"Request timed out after {_FETCH_TIMEOUT}s: {url}",
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                tool_name="WebFetch",
                success=False,
                output=f"HTTP {exc.response.status_code} for {url}",
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                tool_name="WebFetch",
                success=False,
                output=f"HTTP error fetching {url}: {exc}",
            )

        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            text = _html_to_markdown(resp.text)
        elif "application/json" in content_type:
            text = resp.text
        else:
            text = resp.text

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Truncated at {max_chars:,} characters]"

        result = ToolResult(
            tool_name="WebFetch",
            success=True,
            output=text,
            raw_output=text,
            metadata={"url": url, "status_code": resp.status_code, "chars": len(text)},
        )
        # Cache the successful fetch.  Bounded FIFO — cap prevents long
        # sessions from growing the cache without bound.
        if len(_fetch_cache) >= _FETCH_CACHE_MAX:
            _fetch_cache.pop(next(iter(_fetch_cache)))
        _fetch_cache[cache_key] = result
        return result

    # ── Registration ──────────────────────────────────────────────

    registry.register(
        ToolSchema(
            name="WebSearch",
            description=(
                "Search the web for information. Use for looking up API documentation, "
                "library usage, error messages, or any external knowledge. "
                "Returns titles, URLs, and snippets. Limited to "
                f"{_SEARCH_RATE_LIMIT} calls per session."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": f"Number of results (1-{_MAX_NUM_RESULTS}, default {_DEFAULT_NUM_RESULTS}).",
                    },
                },
                "required": ["query"],
            },
        ),
        fn=web_search,
        risk_level=RiskLevel.READ,
    )

    registry.register(
        ToolSchema(
            name="WebFetch",
            description=(
                "Fetch the content of a URL and return it as text/markdown. "
                "Use for reading documentation pages, API references, or any web content. "
                "HTML is converted to markdown. Output is truncated to max_chars."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (http:// or https://).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": f"Maximum characters to return (default {_DEFAULT_MAX_CHARS:,}, max {_MAX_MAX_CHARS:,}).",
                    },
                },
                "required": ["url"],
            },
        ),
        fn=web_fetch,
        risk_level=RiskLevel.READ,
    )


# ── Search backend implementations ────────────────────────────────


async def _brave_search(query: str, num_results: int, api_key: str) -> ToolResult:
    """Call Brave Search API."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return ToolResult(
            tool_name="WebSearch",
            success=False,
            output=f"Brave Search error: {exc}",
        )

    data = resp.json()
    results = data.get("web", {}).get("results", [])
    if not results:
        return ToolResult(
            tool_name="WebSearch",
            success=True,
            output=f"No results found for: {query}",
        )

    lines = [f"## Search results for: {query}\n"]
    for i, item in enumerate(results[:num_results], 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        snippet = item.get("description", "")
        lines.append(f"{i}. **{title}**\n   {url}\n   {snippet}\n")

    output = "\n".join(lines)
    return ToolResult(
        tool_name="WebSearch",
        success=True,
        output=output,
        raw_output=output,
        metadata={"query": query, "result_count": len(results)},
    )


async def _google_search(
    query: str, num_results: int, api_key: str, engine_id: str
) -> ToolResult:
    """Call Google Custom Search API."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "q": query,
                    "key": api_key,
                    "cx": engine_id,
                    "num": num_results,
                },
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return ToolResult(
            tool_name="WebSearch",
            success=False,
            output=f"Google Search error: {exc}",
        )

    data = resp.json()
    items = data.get("items", [])
    if not items:
        return ToolResult(
            tool_name="WebSearch",
            success=True,
            output=f"No results found for: {query}",
        )

    lines = [f"## Search results for: {query}\n"]
    for i, item in enumerate(items[:num_results], 1):
        title = item.get("title", "Untitled")
        url = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"{i}. **{title}**\n   {url}\n   {snippet}\n")

    output = "\n".join(lines)
    return ToolResult(
        tool_name="WebSearch",
        success=True,
        output=output,
        raw_output=output,
        metadata={"query": query, "result_count": len(items)},
    )


async def _tavily_search(query: str, num_results: int, api_key: str) -> ToolResult:
    """Call Tavily Search API.

    Tavily is LLM-optimized: results include a cleaned ``content`` excerpt
    (not just a snippet), which often removes the need for a follow-up
    ``WebFetch`` to the same URL.  Free tier: 1,000 queries/month.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": num_results,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return ToolResult(
            tool_name="WebSearch",
            success=False,
            output=f"Tavily Search error: {exc}",
        )

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return ToolResult(
            tool_name="WebSearch",
            success=True,
            output=f"No results found for: {query}",
        )

    lines = [f"## Search results for: {query}\n"]
    for i, item in enumerate(results[:num_results], 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        # Tavily's 'content' is a cleaned excerpt (bigger than a snippet).
        # Cap it so individual results don't dominate the output.
        content = (item.get("content") or "").strip()
        if len(content) > 500:
            content = content[:500] + "…"
        lines.append(f"{i}. **{title}**\n   {url}\n   {content}\n")

    output = "\n".join(lines)
    return ToolResult(
        tool_name="WebSearch",
        success=True,
        output=output,
        raw_output=output,
        metadata={"query": query, "result_count": len(results), "backend": "tavily"},
    )


# DDG result-block regex — match the HTML structure returned by
# html.duckduckgo.com/html.  Captures title+href; snippet captured
# separately via _DDG_SNIPPET_RE to avoid lazy/optional interactions.
# This IS fragile — DDG can change their HTML at any time.
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
# Snippet lives inside the same result block but in a separate anchor.
# We match the broader block then pull the snippet text out.
_DDG_BLOCK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'(.*?)'  # block content between title and next result
    r'(?=<a[^>]+class="result__a"|</body|$)',
    re.DOTALL | re.IGNORECASE,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_DDG_REDIRECT_RE = re.compile(r"uddg=([^&]+)", re.IGNORECASE)


def _ddg_unwrap_url(raw_url: str) -> str:
    """DDG wraps result URLs in a redirect.  Unwrap to the real target."""
    from urllib.parse import unquote
    m = _DDG_REDIRECT_RE.search(raw_url)
    if m:
        return unquote(m.group(1))
    # Sometimes the URL is already direct or a protocol-relative link.
    if raw_url.startswith("//"):
        return "https:" + raw_url
    return raw_url


def _ddg_strip_tags(html: str) -> str:
    """Strip HTML tags + decode common entities for a text snippet."""
    text = re.sub(r"<[^>]+>", "", html or "")
    text = (
        text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#39;", "'").replace("&apos;", "'")
            .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


async def _ddg_search(query: str, num_results: int) -> ToolResult:
    """DuckDuckGo HTML-endpoint fallback.  No API key required.

    Fragile — scrapes HTML from ``html.duckduckgo.com``.  DDG can change
    their markup at any time.  Use only as a "works out of the box"
    fallback; prefer Tavily/Brave/Google for reliable search.
    """
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True,
        ) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "us-en"},
                headers={
                    # DDG filters non-browser UAs aggressively.
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/121.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        return ToolResult(
            tool_name="WebSearch",
            success=False,
            output=(
                f"DuckDuckGo fallback error: {exc}.  "
                "No WebSearch API key is configured — set TAVILY_API_KEY, "
                "BRAVE_SEARCH_API_KEY, or GOOGLE_SEARCH_API_KEY for a "
                "reliable backend."
            ),
        )

    html = resp.text
    # Walk block-by-block so we can pick up title+snippet from the same
    # result container (avoids lazy/optional regex interactions).
    matches = _DDG_BLOCK_RE.findall(html)
    if not matches:
        return ToolResult(
            tool_name="WebSearch",
            success=False,
            output=(
                "DuckDuckGo returned no parseable results.  Their HTML "
                "structure may have changed.  Set TAVILY_API_KEY, "
                "BRAVE_SEARCH_API_KEY, or GOOGLE_SEARCH_API_KEY for a "
                "reliable backend."
            ),
        )

    lines = [f"## Search results for: {query}\n"]
    kept = 0
    seen_urls: set[str] = set()
    for raw_url, title_html, block_content in matches:
        if kept >= num_results:
            break
        url = _ddg_unwrap_url(raw_url)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = _ddg_strip_tags(title_html) or "Untitled"
        snippet_m = _DDG_SNIPPET_RE.search(block_content)
        snippet = _ddg_strip_tags(snippet_m.group(1)) if snippet_m else ""
        kept += 1
        lines.append(f"{kept}. **{title}**\n   {url}\n   {snippet}\n")

    if kept == 0:
        return ToolResult(
            tool_name="WebSearch",
            success=True,
            output=f"No results found for: {query}",
        )

    output = "\n".join(lines)
    return ToolResult(
        tool_name="WebSearch",
        success=True,
        output=output,
        raw_output=output,
        metadata={"query": query, "result_count": kept, "backend": "duckduckgo"},
    )


# ── HTML-to-markdown converter ────────────────────────────────────


def _html_to_markdown(html: str) -> str:
    """Lightweight HTML to markdown conversion using regex.

    Not a full parser — good enough for extracting readable content
    from documentation pages and articles.
    """
    # Remove script and style blocks entirely.
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert headings.
    for level in range(1, 7):
        text = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            lambda m, l=level: f"\n{'#' * l} {m.group(1).strip()}\n",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Convert links.
    text = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert list items.
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert paragraphs and line breaks.
    text = re.sub(r"<p[^>]*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Convert bold and italic.
    text = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert code blocks.
    text = re.sub(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", r"\n```\n\1\n```\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip all remaining HTML tags.
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common HTML entities.
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&apos;", "'").replace("&nbsp;", " ")

    # Collapse excessive whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
