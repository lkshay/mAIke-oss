## Test-Driven Development

When building new functionality from scratch:

1. **Red** — write the simplest failing test that defines the next piece of behavior.
   Start with the most basic case (e.g., empty input, single element, trivial path).

2. **Green** — write the minimum code to make that test pass. Don't over-build.

3. **Refactor** — clean up duplication, extract helpers, improve names.
   Run tests again to verify nothing broke.

4. **Repeat** — add the next test case, slightly more complex than the last.
   Build complexity gradually: base case → simple case → edge case → error case.

5. **When to break the cycle** — if you realize the design is wrong after several
   passing tests, stop and refactor the design. Don't fight the tests.

This approach prevents the common failure mode of writing 200 lines of untested
code and then spending 10 iterations debugging it. Small, verified steps are faster.
