# Django Debugging Reference

## ORM Query Debugging

### Seeing the SQL

```python
from django.db import connection

# After running queries:
for query in connection.queries[-5:]:
    print(query["sql"], query["time"])

# For a specific queryset:
qs = User.objects.filter(is_active=True)
print(qs.query)  # prints the SQL (approximate, not parameterized)
```

**Gotcha:** `connection.queries` is only populated when `DEBUG=True`.
In production or tests (where `DEBUG=False`), use `django.db.reset_queries()`
and wrap code with `from django.test.utils import override_settings`.

### N+1 Query Detection

The most common Django performance bug:

```python
# BAD: N+1 -- one query per author
for book in Book.objects.all():
    print(book.author.name)  # each .author triggers a query

# GOOD: select_related for ForeignKey
for book in Book.objects.select_related("author").all():
    print(book.author.name)  # no extra queries

# GOOD: prefetch_related for ManyToMany / reverse FK
for author in Author.objects.prefetch_related("books").all():
    print([b.title for b in author.books.all()])
```

Use `django-debug-toolbar` or `nplusone` to catch N+1 issues automatically.

### QuerySet Gotchas

- `qs.count()` is `SELECT COUNT(*)` -- efficient. `len(qs)` fetches all rows.
- `qs.exists()` is `SELECT 1 ... LIMIT 1` -- use it instead of `if qs:`.
- Chaining `.filter()` is lazy. The query runs on iteration, `list()`, or slicing.
- `qs.update()` does a single `UPDATE` statement. Iterating + `.save()` is N queries.

## Middleware Debugging

When requests behave unexpectedly, check middleware ordering:

```python
# settings.py MIDDLEWARE order matters:
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",    # 1st: security headers
    "django.contrib.sessions.middleware.SessionMiddleware",  # before auth
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",        # before auth
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]
```

**Debugging technique:** Add a temporary middleware to log request/response:

```python
class DebugMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        print(f"[REQ] {request.method} {request.path} user={request.user}")
        response = self.get_response(request)
        print(f"[RES] {response.status_code} {request.path}")
        return response
```

**Common issues:**
- 403 on POST: missing CSRF token or `CsrfViewMiddleware` ordering.
- `AnonymousUser` when user should be logged in: `AuthenticationMiddleware`
  is missing or after middleware that reads `request.user`.
- Sessions not persisting: `SessionMiddleware` must be before `AuthenticationMiddleware`.

## Template Context Debugging

When templates render blank or wrong data:

```python
# In the view, print context before rendering:
def my_view(request):
    context = {"items": Item.objects.all()}
    print(f"[DEBUG] context keys: {list(context.keys())}")
    print(f"[DEBUG] items count: {context['items'].count()}")
    return render(request, "myapp/items.html", context)
```

**Common issues:**
- Variable name mismatch between view context and template `{{ variable }}`.
- QuerySet is empty because of wrong filter. Print `.query` to check SQL.
- Template inheritance: `{% block %}` not defined in child template,
  so parent's default content shows instead.
- `{% if items %}` is False for empty QuerySets. This is correct but
  surprising when you expected data.

## Migration Debugging

- `python manage.py showmigrations` -- see which migrations have run.
- `python manage.py sqlmigrate app_name 0001` -- see the SQL a migration generates.
- Circular dependency: split the migration or use `RunPython` with
  `apps.get_model()` to avoid import-time model references.
- `InconsistentMigrationHistory`: the migration table says it ran, but the
  schema doesn't match. Use `--fake` cautiously to reconcile.
