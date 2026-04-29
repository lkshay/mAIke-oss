"""Hard eval cases — substantial real-world tasks with tight verifiers."""

from __future__ import annotations

import sys
from pathlib import Path

from maike.eval.case_protocol import EvalCase, EvalPhase
from maike.smoke.workflow_cases.helpers import import_module, run_pytest


# ---------------------------------------------------------------------------
# react-lru-cache
# ---------------------------------------------------------------------------

def _verify_lru_cache_workspace(workspace: Path) -> None:
    run_pytest(workspace, label="lru cache pytest")

    mod = import_module(workspace, "lru_cache")
    LRUCache = getattr(mod, "LRUCache")

    # --- Capacity and LRU eviction order ---
    cache = LRUCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")          # access "a" → makes "b" the LRU
    cache.put("d", 4)       # should evict "b"
    assert cache.get("b") is None, "Eviction: 'b' (LRU) should have been evicted"
    assert cache.get("a") == 1, "Eviction: 'a' (recently accessed) should survive"
    assert cache.get("c") == 3, "Eviction: 'c' should survive"
    assert cache.get("d") == 4, "Eviction: 'd' (newest) should survive"

    # --- Overwrite preserves key, updates value ---
    cache2 = LRUCache(max_size=2)
    cache2.put("x", 10)
    cache2.put("x", 20)   # overwrite
    assert cache2.get("x") == 20, "Overwrite: updated value should be returned"
    assert len(cache2) == 1, "Overwrite: should not double-count same key"

    # --- __contains__ and __len__ ---
    cache3 = LRUCache(max_size=2)
    cache3.put("p", 100)
    cache3.put("q", 200)
    assert "p" in cache3
    assert "q" in cache3
    assert "z" not in cache3
    assert len(cache3) == 2

    # --- Statistics: hits and misses ---
    cache4 = LRUCache(max_size=5)
    cache4.put("k", 99)
    cache4.get("k")     # hit
    cache4.get("k")     # hit
    cache4.get("miss")  # miss
    assert getattr(cache4, "hits", None) is not None, "Cache must expose a 'hits' attribute"
    assert getattr(cache4, "misses", None) is not None, "Cache must expose a 'misses' attribute"
    assert cache4.hits >= 2, f"Expected >= 2 hits, got {cache4.hits}"
    assert cache4.misses >= 1, f"Expected >= 1 miss, got {cache4.misses}"

    # --- Eviction count ---
    cache5 = LRUCache(max_size=2)
    cache5.put("1", 1)
    cache5.put("2", 2)
    cache5.put("3", 3)   # evicts "1"
    cache5.put("4", 4)   # evicts "2"
    evictions = getattr(cache5, "evictions", None)
    assert evictions is not None, "Cache must expose an 'evictions' attribute"
    assert evictions >= 2, f"Expected >= 2 evictions, got {evictions}"

    # --- delete ---
    cache6 = LRUCache(max_size=3)
    cache6.put("a", 1)
    cache6.put("b", 2)
    deleted = cache6.delete("a")
    assert deleted is True or deleted == 1, "delete should return truthy for existing key"
    assert cache6.get("a") is None, "Deleted key should return None"
    assert len(cache6) == 1

    # --- clear ---
    cache7 = LRUCache(max_size=3)
    cache7.put("a", 1)
    cache7.put("b", 2)
    cache7.clear()
    assert len(cache7) == 0, "clear() should empty the cache"
    assert cache7.get("a") is None


HARD_EVAL_CASES: dict[str, EvalCase] = {
    "react-lru-cache": EvalCase(
        name="react-lru-cache",
        phases=(
            EvalPhase(
                task=(
                    "Build a production-ready LRU (Least Recently Used) cache in Python without using "
                    "functools.lru_cache or any external caching library. "
                    "Create lru_cache.py with a LRUCache(max_size) class that has: "
                    "get(key) returning None on miss, "
                    "put(key, value) with LRU eviction when at capacity, "
                    "delete(key) returning True if found, "
                    "clear() to empty the cache, "
                    "__len__ and __contains__ support, "
                    "and three integer attributes: hits, misses, evictions. "
                    "Also create test_lru_cache.py with pytest tests covering: "
                    "LRU eviction order (most recent survives), capacity enforcement, "
                    "overwrite (same key updated not duplicated), hit/miss/eviction counters, "
                    "delete and clear. "
                    "Add README.md with usage examples. "
                    "All tests must pass."
                ),
            ),
        ),
        expected_pipeline=None,
        expected_stages=None,
        expected_stage_artifacts=None,
        setup_workspace=lambda _: None,
        verify_workspace=_verify_lru_cache_workspace,
        tags=("hard", "react"),
        budget=8.0,
    ),
}
