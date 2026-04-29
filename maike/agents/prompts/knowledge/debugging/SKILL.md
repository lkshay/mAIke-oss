---
name: debugging
description: "Systematic debugging methodology — observe before guessing, binary search for bugs, isolate failures. Use when debugging errors, diagnosing issues, or analyzing tracebacks."
compatibility: "mAIke coding agent"
metadata:
  triggers: "debug, debugging, diagnose, root cause, traceback, stack trace, django, orm, middleware"
  auto_inject: "false"
---

## Systematic Debugging

When fixing bugs or investigating failures:

1. **Reproduce first** — before changing anything, run the failing test or command
   to see the exact error. Read the full traceback.

2. **Observe, don't guess** — add print statements or assertions at the point of
   failure to see actual values. The bug is usually not where you think it is.

3. **Binary search** — for complex bugs, comment out half the code. If the bug
   disappears, it's in the half you removed. Narrow down iteratively.

4. **One change at a time** — make a single change, run the test, verify. Don't
   stack multiple fixes before testing.

5. **Check assumptions** — the most common bugs are wrong assumptions:
   - Variable is None when you expected a value
   - List is empty when you expected elements
   - String has unexpected whitespace or encoding
   - Off-by-one in index or range
   - Type mismatch (str vs int, list vs tuple)

6. **Read the error message** — Python tracebacks tell you the exact file, line,
   and expression that failed. Start there, not somewhere else.

## References

- `references/strategies.md` -- Advanced debugging strategies and techniques
- `references/django-debugging.md` -- Django ORM queries, middleware, template context
