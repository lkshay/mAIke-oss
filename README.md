# mAIke

A local-first coding agent that runs on Anthropic, OpenAI, Gemini, or local Ollama models.

## Why mAIke?

- **Multi-provider** — switch between Claude, GPT, Gemini, and local Ollama models any time
- **Cost-aware** — every call is priced and projected against the budget you set, before it fires
- **Local-first** — your code, sessions, cost ledger, and memory all live on your machine
- **Safe by default** — read-before-edit, tool risk-tiering, inline approvals for execution
- **Persistent** — auto-memory carries project knowledge across sessions; threads keep multiple conversations per workspace
- **Recoverable** — built-in error recovery, with an optional frontier-model "advisor" co-pilot when you want a stronger brain on call

## Install

```bash
pip install -e .
```

## Configure

```bash
maike setup                  # interactive — writes ~/.config/maike/.env
# or set keys directly:
export GEMINI_API_KEY="..."  # ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY also accepted
# Ollama needs no key — just run `ollama serve` locally

maike init                   # one-time per workspace; generates MAIKE.md
```

## Launch

`maike` on its own opens the interactive TUI in the current directory.

```bash
# Open the TUI with defaults
maike

# Override provider / model / budget
maike --provider gemini --model gemini-2.5-flash --budget 5.00

# With auto-approve and verbose tracing
maike --yes --verbose

# Quit with /quit, /exit
```

| Flag | Purpose |
|------|---------|
| `--provider` | `anthropic` / `openai` / `gemini` / `ollama` |
| `--model` / `-m` | Model name (overrides the provider default) |
| `--budget` / `-b` | Session budget in USD (default 5.00) |
| `--verbose` / `-v` | Show LLM and tool traces inline |
| `--yes` / `-y` | Auto-approve tool calls |

## Inside the TUI

- Streamed output with collapsible tool-call widgets
- `@`-mention to attach files or directories to your message
- Inline approval widgets — APPROVE / APPROVE_ALWAYS / DENY, with an optional typed reason
- Ctrl+C copies the selection (or quits when nothing is selected)

**Slash commands**:

| Command | Purpose |
|---------|---------|
| `/help` | List commands |
| `/cost`, `/budget` | Cost breakdown / remaining budget |
| `/history`, `/new` | Show / reset the current thread |
| `/status`, `/tasks`, `/context` | Session state, background tasks, token usage |
| `/agent`, `/create-agent` | List / create custom agents |
| `/team`, `/create-team` | List / create agent teams |
| `/advisor` | Show advisor status or ask the advisor a question |
| `/worktree` | Manage git worktrees |
| `/plugin`, `/skill`, `/mcp`, `/hook` | Manage extensibility surfaces |
| `/quit`, `/exit` | Exit |

## Other commands

```bash
# Multiple conversations per workspace
maike threads                                # list threads
maike --thread <id>                          # continue a specific thread
maike --new-thread                           # force a brand-new thread

# Resume a specific session by ID
maike resume <session-id> --workspace <path>

# One-shot / scripted run (non-interactive — for CI, automation)
maike run "Add input validation to the User model"
maike run "Fix failing tests" --provider gemini --budget 2.00 --yes

# Inspect cost and history
maike cost                                   # current/last session
maike cost <session-id>                      # a specific session
maike history                                # workspace session history
```

## Threads & Worktrees

A workspace can hold many concurrent **threads** — each keeps its own message history, plan, and cost ledger. By default `maike` resumes the most recent thread; use `--new-thread` for a fresh start, or `--thread <id>` to continue a specific one.

For isolated branch work, `maike worktree add/list/remove` wraps `git worktree` so an agent can operate safely on a side branch without touching your main checkout.

## Advisor

Pair a cheap fast executor with a frontier-model "advisor" that gets called at key moments — long exploration, repeated failures, before the first edit, before completion:

```bash
maike --provider gemini --model gemini-2.5-flash \
      --advisor --advisor-provider anthropic --advisor-model claude-opus-4-20250514
```

The advisor never runs tools itself — it just returns 1–3 sentences of guidance that mAIke injects into the executor's next turn. It has its own budget cap (default 20% of the session).

## Auto-Memory

mAIke remembers what it learns. At the end of every session it writes a project overview, key architectural decisions, and resolved pitfalls into `<workspace>/.maike/memories/`. The next session starts with that knowledge already in context — no re-exploration of the same files.

## Extensibility

| Surface | What it gives you | Install |
|---------|-------------------|---------|
| **Skills** | Reusable Markdown procedures the agent invokes on demand | `maike skill install <path-or-url>` |
| **Plugins** | Bundles of skills, agents, hooks, MCP and LSP configs | `maike plugin install <path-or-url>` |
| **MCP** | Any Model Context Protocol server appears as a first-class tool | Configure in `.mcp.json` |
| **LSP** | Language servers feed diagnostics and symbols into the agent's context | Declare in a plugin manifest |

Custom agents and teams can also be created interactively with `/create-agent` and `/create-team`.

## Evaluation

```bash
maike eval --suite all                                   # built-in agentic test cases
maike eval --suite hard-agentic --keep-workspaces        # preserve workspaces for inspection
maike swe-bench --variant lite                           # SWE-bench Lite (300 instances)
maike swe-bench --variant verified                       # SWE-bench Verified (500 instances)
maike swe-bench --resume predictions.jsonl               # resume an interrupted run
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Provider keys |
| `GOOGLE_GENAI_USE_VERTEXAI=True` + `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` | Use Gemini via Vertex AI (auths through `gcloud`, no API key needed) |
| `OLLAMA_HOST` | Override the local Ollama endpoint |
| `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` / `GOOGLE_SEARCH_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID` | Web search backends (used in priority order) |
| `MAIKE_DEFAULT_BUDGET_USD` | Default session budget (default 5.00) |
| `MAIKE_WORKSPACE` | Default workspace path |
| `MAIKE_LOG_LEVEL` | Logging level |

## MAIKE.md

`maike init` writes a per-workspace `MAIKE.md` with project context and a `## Protected Files` section. Files listed there cannot be modified by the agent. MAIKE.md is yours to edit — mAIke reads it but never writes to it.
