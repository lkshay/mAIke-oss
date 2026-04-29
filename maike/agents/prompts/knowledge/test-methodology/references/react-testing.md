# React Testing Reference

## React Testing Library (RTL) Core Patterns

RTL encourages testing behavior, not implementation. Query by what the user
sees, not by component internals.

```jsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Counter } from "./Counter";

test("increments counter on click", async () => {
  const user = userEvent.setup();
  render(<Counter />);

  await user.click(screen.getByRole("button", { name: /increment/i }));

  expect(screen.getByText("Count: 1")).toBeInTheDocument();
});
```

## Query Priority

Use queries in this order (most to least preferred):

1. `getByRole` -- accessible role + name (best for buttons, links, headings)
2. `getByLabelText` -- form inputs with labels
3. `getByPlaceholderText` -- inputs without visible labels
4. `getByText` -- non-interactive content
5. `getByTestId` -- last resort, requires `data-testid` attribute

**Anti-pattern:** Using `getByTestId` for everything. It couples tests to
implementation details. If you need a test ID, the component may have an
accessibility problem.

## Async Patterns

For components that fetch data or update state asynchronously:

```jsx
test("loads user data", async () => {
  render(<UserProfile userId={1} />);

  // waitFor retries until the assertion passes or times out
  expect(await screen.findByText("Alice")).toBeInTheDocument();
});
```

**Gotcha:** `findBy*` queries are sugar for `waitFor` + `getBy*`. Use
`findBy*` when waiting for an element to appear. Use `waitFor` when
asserting on an element that's already present but changing.

**Anti-pattern:** Using `waitFor` with side effects inside:
```jsx
// BAD: side effects in waitFor can run multiple times
await waitFor(() => {
  fireEvent.click(button); // may click multiple times!
  expect(result).toBe("done");
});
```

## Mocking Patterns

### API Mocking with MSW

Prefer `msw` (Mock Service Worker) over mocking fetch/axios directly:

```jsx
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

const server = setupServer(
  http.get("/api/users/1", () => {
    return HttpResponse.json({ id: 1, name: "Alice" });
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

### Module Mocking

```jsx
// Mock a hook
jest.mock("./useAuth", () => ({
  useAuth: () => ({ user: { name: "Alice" }, isLoggedIn: true }),
}));
```

**Anti-pattern:** Mocking React internals (`useState`, `useEffect`). If you
need to mock these, the component is too complex -- extract logic into a
custom hook and test that separately.

## Common Anti-Patterns

1. **Testing implementation details.** Don't assert on state values,
   component instances, or internal methods. Test what the user sees.

2. **Snapshot overuse.** Large snapshots break on every trivial change and
   get rubber-stamp approved. Use targeted assertions instead.

3. **Not wrapping state updates in `act`.** RTL handles this for its own
   utilities, but manual `setState` calls in tests need `act()`.

4. **Ignoring cleanup.** RTL auto-cleans after each test with Jest, but
   custom subscriptions or timers need manual cleanup.

5. **Using `container.querySelector`.** Reaching into the DOM via CSS
   selectors defeats the purpose of RTL. Use role-based queries.

6. **Forgetting `user-event` setup.** Always call `userEvent.setup()` before
   render. It configures pointer and keyboard event simulation correctly.
