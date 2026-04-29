"""Tests for thread CRUD operations and thread name generation."""

import asyncio
from pathlib import Path

import pytest

from maike.memory.session import SessionStore, generate_thread_name


@pytest.fixture()
def store(tmp_path):
    """Create a fresh SessionStore in a temp directory."""
    return SessionStore(tmp_path)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


class TestGenerateThreadName:
    def test_basic_task(self):
        assert generate_thread_name("Create a Python CLI calculator") == "create-python-cli-calculator"

    def test_strips_stop_words(self):
        name = generate_thread_name("Fix the bug in the auth middleware")
        assert "the" not in name.split("-")
        assert "in" not in name.split("-")

    def test_limits_to_five_words(self):
        name = generate_thread_name("Create very complex large web application server")
        # 5 meaningful words after stop-word removal: create, very, complex, large, web
        assert len(name.split("-")) <= 5

    def test_strips_punctuation(self):
        name = generate_thread_name("Can you tell me how to run this?.")
        assert "?" not in name
        assert "." not in name

    def test_empty_task(self):
        assert generate_thread_name("") == "default-thread"

    def test_all_stop_words(self):
        assert generate_thread_name("can you do this for me?") == "default-thread"


class TestThreadCRUD:
    def test_create_and_get_thread(self, store):
        async def run():
            async with store.use_shared_connection():
                tid = await store.create_thread(Path("/workspace"), "test-thread")
                thread = await store.get_thread(tid)
                assert thread is not None
                assert thread["name"] == "test-thread"
                assert thread["workspace"] == "/workspace"
                assert thread["status"] == "active"
                assert thread["conversation_history"] == []
                assert thread["history_token_estimate"] == 0
        _run(run())

    def test_get_active_thread(self, store):
        async def run():
            async with store.use_shared_connection():
                ws = Path("/workspace")
                tid1 = await store.create_thread(ws, "thread-1")
                tid2 = await store.create_thread(ws, "thread-2")
                active = await store.get_active_thread(ws)
                assert active is not None
                # Most recently created should be returned
                assert active["id"] == tid2
        _run(run())

    def test_get_active_thread_returns_none_for_empty_workspace(self, store):
        async def run():
            async with store.use_shared_connection():
                result = await store.get_active_thread(Path("/nonexistent"))
                assert result is None
        _run(run())

    def test_append_thread_history(self, store):
        async def run():
            async with store.use_shared_connection():
                tid = await store.create_thread(Path("/workspace"), "test")
                messages = [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
                await store.append_thread_history(tid, messages)
                thread = await store.get_thread(tid)
                assert len(thread["conversation_history"]) == 2
                assert thread["history_token_estimate"] > 0

                # Append more
                await store.append_thread_history(tid, [{"role": "user", "content": "more"}])
                thread = await store.get_thread(tid)
                assert len(thread["conversation_history"]) == 3
        _run(run())

    def test_replace_thread_history(self, store):
        async def run():
            async with store.use_shared_connection():
                tid = await store.create_thread(Path("/workspace"), "test")
                await store.append_thread_history(tid, [
                    {"role": "user", "content": "old message 1"},
                    {"role": "assistant", "content": "old message 2"},
                ])
                # Replace with compressed version
                compressed = [{"role": "user", "content": "[SUMMARY] old conversation"}]
                await store.replace_thread_history(tid, compressed, 50)
                thread = await store.get_thread(tid)
                assert len(thread["conversation_history"]) == 1
                assert thread["history_token_estimate"] == 50
        _run(run())

    def test_list_threads(self, store):
        async def run():
            async with store.use_shared_connection():
                ws = Path("/workspace")
                await store.create_thread(ws, "thread-a")
                await store.create_thread(ws, "thread-b")
                await store.create_thread(Path("/other"), "thread-c")
                threads = await store.list_threads(ws)
                assert len(threads) == 2
                names = {t["name"] for t in threads}
                assert names == {"thread-a", "thread-b"}
        _run(run())

    def test_get_nonexistent_thread_returns_none(self, store):
        async def run():
            async with store.use_shared_connection():
                result = await store.get_thread("nonexistent-id")
                assert result is None
        _run(run())
