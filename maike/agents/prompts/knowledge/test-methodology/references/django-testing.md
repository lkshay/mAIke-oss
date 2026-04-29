# Django Testing Reference

## TestCase vs pytest-django

Django's `TestCase` wraps each test in a transaction that rolls back.
`pytest-django` provides the same via `@pytest.mark.django_db`.

```python
# Django-native
from django.test import TestCase

class UserTest(TestCase):
    def test_create_user(self):
        user = User.objects.create(username="alice")
        self.assertEqual(User.objects.count(), 1)

# pytest-django (preferred for mAIke agents)
import pytest

@pytest.mark.django_db
def test_create_user():
    user = User.objects.create(username="alice")
    assert User.objects.count() == 1
```

**Gotcha:** `@pytest.mark.django_db` is required for any test touching the
ORM. Without it, you get `DatabaseAccessNotAllowed`. Use
`@pytest.mark.django_db(transaction=True)` when testing code that uses
`transaction.atomic()` or `on_commit()`.

## Database Access Patterns

- `TransactionTestCase`: flushes the DB between tests (slow). Use only when
  testing signals, `on_commit`, or raw SQL.
- `TestCase`: wraps in a savepoint (fast). Default choice.
- `SimpleTestCase`: no DB access at all. Use for pure logic tests.

**Anti-pattern:** Using `TransactionTestCase` everywhere "to be safe" makes
the test suite 10-50x slower. Only use it when you actually need commits.

## Fixtures

Django fixtures (JSON/YAML) are fragile. Prefer factory functions:

```python
# Bad: fixtures that break when models change
# fixtures = ["users.json"]

# Good: factory functions
def make_user(**kwargs):
    defaults = {"username": "testuser", "email": "test@example.com"}
    defaults.update(kwargs)
    return User.objects.create(**defaults)

@pytest.mark.django_db
def test_user_profile():
    user = make_user(username="alice")
    assert user.profile is not None
```

For complex object graphs, use `factory_boy` with `DjangoModelFactory`.

## Testing Views and URLs

```python
from django.test import Client

@pytest.mark.django_db
def test_home_page():
    client = Client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome" in response.content
```

**Gotcha:** `Client.login()` requires the `django.contrib.sessions` and
`django.contrib.auth` middleware. If login silently fails, check
`MIDDLEWARE` and `AUTHENTICATION_BACKENDS`.

## Common Anti-Patterns

1. **Forgetting `django_db` marker.** Tests silently pass with zero
   assertions because the DB call raises before the assert.

2. **Testing with `DEBUG=True`.** Django's test runner sets `DEBUG=False`.
   Template errors that are silenced in debug mode will surface in tests.
   This is intentional -- don't override it.

3. **Not isolating celery tasks.** Use `@override_settings(CELERY_ALWAYS_EAGER=True)`
   or mock `task.delay()` to prevent actual task execution in tests.

4. **Overusing `setUp`.** Large `setUp` methods create tightly coupled tests.
   Each test should create only the data it needs.

5. **Asserting on auto-generated IDs.** PKs can vary between test runs.
   Assert on meaningful fields (username, email) not database IDs.

6. **Missing `override_settings` cleanup.** Use it as a decorator or context
   manager, never by mutating `django.conf.settings` directly.
