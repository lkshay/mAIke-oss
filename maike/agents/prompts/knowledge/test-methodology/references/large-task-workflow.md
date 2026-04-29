## Large Task Workflow — Code-First with Incremental Testing

When building multiple interrelated components or exploring a design:

1. **Sketch the structure** — create stub files/classes for all major components.
   Get the module structure right before filling in logic.

2. **Implement the core algorithm** — write the main logic first, without tests.
   Focus on getting the happy path working end-to-end.

3. **Add a smoke test** — once the core works, add ONE test that exercises
   the main path. Run it to verify basic correctness.

4. **Iterate: implement then test** — for each component you implement,
   add targeted tests immediately after: happy path, edge cases, error cases.
   Do NOT write tests for components you haven't built yet.

5. **Don't test-drive multiple unknowns** — if you don't know how components
   will interact, implement them first, then write integration tests.

6. **Budget your iterations** — if the task has N components and you have M
   iterations remaining, allocate roughly M/N iterations per component.
   Prioritize getting all components implemented over perfecting any one.

7. **When to switch to TDD** — once the architecture is stable and you're
   adding individual features to a working system, switch to TDD.
