# Advanced Debugging Strategies

## Binary Search Bisection

When a bug is somewhere in a large codebase and you can't pinpoint it:

1. Comment out (or bypass) roughly half the code.
2. Run the test. Does the bug still happen?
   - **Yes** — the bug is in the remaining half. Comment out half of that.
   - **No** — the bug is in the half you removed. Restore it and comment out the other half.
3. Repeat until you've isolated the exact function or line.

This finds any bug in O(log n) steps, even in code you don't understand.

## Print-Based Tracing with Structured Output

When you need to trace execution flow, use structured print statements:

```python
def process(data):
    print(f"[TRACE] process() called with {type(data).__name__}, len={len(data)}")
    for i, item in enumerate(data):
        print(f"[TRACE]   item[{i}] = {item!r}")
        result = transform(item)
        print(f"[TRACE]   result[{i}] = {result!r}")
```

Tips:
- Use a consistent prefix like `[TRACE]` or `[DEBUG]` so you can grep the output.
- Print types, not just values — `type(x).__name__` catches type mismatches.
- Use `!r` in f-strings to see exact string representations (whitespace, quotes).
- Print at function entry and exit to trace control flow.

## Assertion-Based Debugging

Insert assertions at key points to catch invalid state early:

```python
def calculate_price(quantity, unit_price, discount):
    assert quantity > 0, f"quantity must be positive, got {quantity}"
    assert 0 <= discount <= 1, f"discount must be 0-1, got {discount}"
    total = quantity * unit_price * (1 - discount)
    assert total >= 0, f"total should never be negative, got {total}"
    return total
```

Assertions are better than print statements when:
- You know what the correct state should be.
- You want the program to fail immediately at the point of corruption, not later.
- You're debugging a data flow issue (value gets corrupted somewhere upstream).

## Git Bisect for Regression Hunting

When something used to work but now doesn't, and you have a git history:

```bash
git bisect start
git bisect bad              # current commit is broken
git bisect good abc123      # this older commit was working
# Git checks out a middle commit — test it
# Then tell git:
git bisect good  # or  git bisect bad
# Repeat until git identifies the exact commit that introduced the bug
git bisect reset  # when done
```

Automate with a test script:
```bash
git bisect run pytest tests/test_specific.py -x
```

## Rubber Duck Debugging

When you're stuck, explain the problem out loud (or in writing) step by step:

1. State exactly what the code is supposed to do.
2. Walk through the code line by line, explaining what each line does.
3. At each step, state what you expect the values to be.
4. When your explanation diverges from what the code actually does, you've found the bug.

This works because forcing yourself to articulate each step exposes hidden assumptions.

## When to Revert and Start Fresh

Sometimes the fastest fix is to undo and redo. Consider reverting when:

- You've made many changes and lost track of which ones matter.
- The bug was introduced by your own changes (git diff shows what changed).
- You've been debugging for more than 15-20 minutes without progress.
- The code has become harder to understand than when you started.

Steps:
1. `git stash` your changes (don't lose them entirely).
2. Verify the original code works (or has only the original bug).
3. Reapply changes one at a time, testing after each.
4. The bug will reappear when you apply the problematic change.
