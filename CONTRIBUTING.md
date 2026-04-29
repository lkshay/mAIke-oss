# Contributing

Issues and PRs welcome. mAIke is alpha — expect breaking changes — and the surface area is large, so please open an issue before starting non-trivial work.

## Before you submit a PR

```bash
pip install -e .
ruff check maike/ tests/
pytest tests/
```

- All tests must pass locally.
- New behavior should come with a unit or integration test in `tests/`.
- Lint with `ruff` (config in `pyproject.toml`) — no separate formatter.
- Keep commit messages concise; one-line subjects, optional body for context.

## Areas where help is appreciated

- Provider adapters (Bedrock, Cerebras, Groq, OpenRouter)
- LSP integrations beyond the existing surface
- SWE-bench harness automation (running the official Docker harness end-to-end)
- TUI polish (themes, palette UX)
- Eval suite expansions

## Bug reports

Include the provider/model, the command you ran, the output of `maike --version`, and (if reproducible) the session ID from `~/.maike/sessions/`. Redact API keys before pasting logs.
