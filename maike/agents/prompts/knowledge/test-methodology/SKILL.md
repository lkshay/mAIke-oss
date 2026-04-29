---
name: test-methodology
description: "Testing strategy guidance — helps choose between TDD for small tasks and code-first for large multi-component tasks. Includes rules for testing existing code without modifying source files."
compatibility: "mAIke coding agent"
metadata:
  triggers: "tdd, test-driven, test driven, red green, test, testing strategy, testing approach, test plan, write tests, add tests, vitest, jest, pytest, django, fastapi, react, pytest-django, TestClient"
  paths: "**/test_*.py, **/tests/**, **/*.test.ts, **/*.spec.ts"
  auto_inject: "false"
---

## Test Methodology

### Golden Rule: Test Through the Public API

**Never modify source files to make internals testable.** If a function is not
exported, test its behavior indirectly through the public functions that call it.

Wrong: adding `export` to a private function so you can import it in tests.
Right: calling the public function with inputs that exercise the private path.

This applies to all languages:
- Python: don't add `__all__` entries or make `_private` methods public
- TypeScript/JavaScript: don't add `export` to module-internal functions
- Go: don't capitalize unexported functions

### Choose Strategy Based on Task Scope

**TDD (small tasks):** Adding a single function, fixing a bug, one well-defined feature.
See [references/tdd-workflow.md](references/tdd-workflow.md) for red-green-refactor.

**Code-first (large tasks):** Building 3+ interrelated components, design needs exploration.
See [references/large-task-workflow.md](references/large-task-workflow.md) for build-then-test.

**Decision rule:**
- Can you define the full API/behavior in one test file before writing code? → TDD
- Do you need to build 3+ interacting components before tests make sense? → Code-first
- Are you under iteration pressure (>50% of iteration budget used)? → Code-first

### Writing Tests for Existing Code

When the task is "add tests to existing code":
1. **Read the source** to understand all code paths and edge cases.
2. **Identify the public API** — these are the functions you will test.
3. **Do NOT modify the source.** If you need to test internal behavior, construct
   inputs to the public API that exercise the internal code path.
4. **Set up test infrastructure first** — install the test runner (pytest, vitest,
   jest), create config, verify a trivial test passes before writing real tests.
5. **Cover all branches** — trace each conditional in the source and write a test
   that hits each branch via the public API.

### TypeScript / JavaScript Testing

When setting up tests for a TypeScript project:
- **Vitest** is the recommended test runner (fast, ESM-native, TypeScript support).
- Create `vitest.config.ts` at the project root.
- Test files: `*.test.ts` or `*.spec.ts` next to the source file.
- For type-only imports, create minimal type stubs rather than installing heavy deps.
- Use `vi.mock()` for external dependencies (database, HTTP, filesystem).

### Anti-Pattern: Don't Write Tests for Unimplemented Code

Never write a full test suite for functions that don't exist yet. If you catch
yourself writing tests that all fail because the implementation is missing, stop
and switch to code-first.

### Framework-Specific References
- [references/fastapi-testing.md](references/fastapi-testing.md) -- FastAPI TestClient, async tests, dependency overrides
- [references/django-testing.md](references/django-testing.md) -- Django TestCase vs pytest-django, DB access, fixtures
- [references/react-testing.md](references/react-testing.md) -- React Testing Library patterns, async queries, mocking
