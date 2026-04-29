# FastAPI Testing Reference

## TestClient Basics

Use `httpx`-backed `TestClient` (FastAPI >= 0.100). The old `requests`-based
client is deprecated.

```python
from fastapi.testclient import TestClient
from myapp.main import app

client = TestClient(app)

def test_read_item():
    response = client.get("/items/1")
    assert response.status_code == 200
    assert response.json()["id"] == 1
```

**Gotcha:** `TestClient` runs the ASGI app synchronously. If your endpoint
uses `async def`, it still works, but background tasks won't complete unless
you explicitly await them in tests.

## Async Test Patterns

For true async tests (e.g., testing WebSocket or SSE endpoints):

```python
import pytest
from httpx import ASGITransport, AsyncClient
from myapp.main import app

@pytest.mark.anyio
async def test_async_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/items/1")
    assert response.status_code == 200
```

**Gotcha:** Use `pytest-anyio` (not `pytest-asyncio`) for FastAPI >= 0.100.
The `anyio` backend handles both asyncio and trio.

## Dependency Injection Overrides

Override dependencies to isolate tests from databases, external APIs, etc.:

```python
from myapp.main import app
from myapp.deps import get_db

def fake_db():
    return FakeDatabase()

app.dependency_overrides[get_db] = fake_db

# IMPORTANT: Clean up after tests
# app.dependency_overrides.clear()
```

**Anti-pattern:** Forgetting to clear overrides between tests causes
cross-test pollution. Use a fixture:

```python
@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()
```

## Pydantic v2 Validation in Tests

FastAPI uses Pydantic for request/response validation. In v2:

- `ValidationError` messages changed format. Don't assert exact error strings.
- Use `.model_dump()` instead of `.dict()` (deprecated).
- `response_model_exclude_unset=True` behavior changed; test actual JSON output.

```python
def test_validation_error():
    response = client.post("/items/", json={"name": 123})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(e["type"] == "string_type" for e in errors)
```

## Common Anti-Patterns

1. **Testing the framework, not your code.** Don't test that FastAPI returns
   422 for missing fields -- that's FastAPI's job. Test your business logic.

2. **Not testing error paths.** Always test 404, 422, 401/403, and 500
   scenarios. Use `raises` parameter in `HTTPException` tests.

3. **Sharing app state across tests.** FastAPI `app` is module-level. If you
   mutate `app.state`, tests leak state. Use fixtures to reset.

4. **Ignoring lifespan events.** If your app uses `@app.on_event("startup")`,
   `TestClient` runs them. If startup fails, your tests fail cryptically.
   Use `TestClient(app, raise_server_exceptions=True)` to surface errors.

5. **Mocking at the wrong layer.** Mock the dependency (via overrides), not
   the HTTP client. FastAPI DI is designed for testability -- use it.
