<h1 align="center">mAIke</h1>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg?style=flat-square"></a>
  <img alt="Providers" src="https://img.shields.io/badge/providers-Anthropic%20%7C%20OpenAI%20%7C%20Gemini%20%7C%20Ollama-orange?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-yellow?style=flat-square">
</p>

<p align="center"><b>A local-first ReAct coding agent with strict tool gating, pre-call cost projection, and persistent project memory.</b></p>

<p align="center">Runs Claude, GPT, Gemini, or a local Ollama model — switchable mid-session. Your code, sessions, cost ledger, and memory never leave your machine.</p>

---

## Why mAIke?

- **Multi-provider** — Anthropic, OpenAI, Gemini (direct API or Vertex), and any Ollama model. Configured per session via `--provider` / `--model`.
- **Pre-call cost projection** — every API call is priced *before* it fires. Sessions abort gracefully at 95% of your budget instead of blowing past it.
- **Tool risk gating** — every tool is tiered `READ → WRITE → EXECUTE → DESTRUCTIVE`. Writes need a git checkpoint; execution needs inline approval.
- **Read-before-edit enforcement** — the Edit tool refuses to run on a file the agent hasn't Read this turn, killing a whole class of cascading edit bugs.
- **Auto-memory** — at session end, mAIke distills project overview / key decisions / resolved pitfalls into `.maike/memories/`. The next session starts with that knowledge already in context.
- **Async sub-agents** — spawn read-only delegates that run in parallel; the main agent collects results with `Delegate(action="check"|"wait")`.
- **Advisor pattern** — pair a cheap fast executor with a frontier "advisor" that fires only at exploration / first-edit / stuck moments — guidance, not tool calls.
- **SWE-bench built in** — `maike swe-bench --variant lite|verified|full` runs on real instances with clone caching and resume support.
- **Threads & worktrees** — multiple concurrent conversations per workspace, isolated git worktrees for branch work.
- **Extensible** — Skills, Plugins, MCP servers, and LSP language servers as first-class surfaces.

## Table of Contents

- [Quick Start](#quick-start)
- [Providers & Models](#providers--models)
- [Interactive Mode](#interactive-mode)
- [One-shot & Scripted Runs](#one-shot--scripted-runs)
- [Cost & Budgets](#cost--budgets)
- [Tool Safety Model](#tool-safety-model)
- [Auto-Memory](#auto-memory)
- [Threads & Worktrees](#threads--worktrees)
- [Advisor](#advisor)
- [Sub-Agents & Delegation](#sub-agents--delegation)
- [Extensibility](#extensibility)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Architecture at a Glance](#architecture-at-a-glance)
- [CLI Reference](#cli-reference)

---

## Quick Start

```bash
# Python 3.11+
pip install -e .

# Set keys interactively (writes ~/.config/maike/.env)
maike setup

# Or export directly
export GEMINI_API_KEY="..."        # ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY also accepted
# Ollama? Just `ollama serve` — no key needed.

# One-time per workspace — generates MAIKE.md (project context + protected files)
maike init

# Launch the TUI
maike
```

---

## Providers & Models

| Provider | Auth | Notes |
|----------|------|-------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude family |
| **OpenAI** | `OPENAI_API_KEY` | GPT family |
| **Gemini (direct)** | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Native thought-signature support, 1M context |
| **Gemini (Vertex)** | `gcloud auth application-default login` + `GOOGLE_GENAI_USE_VERTEXAI=True` | No API key needed |
| **Ollama (local)** | none | Set `OLLAMA_HOST` for remote daemons. Zero pricing. |

Default model and pricing live in `maike/constants.py` and `maike/models_default.yaml`. Override per-session with `--provider` / `--model`, or persist to `~/.config/maike/models.yaml` (deep-merged with defaults).

```bash
maike --provider gemini --model gemini-2.5-flash
maike --provider anthropic --model claude-opus-4-20250514
maike --provider ollama --model gemma4:26b
```

---

## Interactive Mode

`maike` on its own opens the Textual-based TUI in the current directory.

```bash
maike                                       # defaults
maike --provider gemini --budget 5.00       # override provider + budget
maike --yes --verbose                       # auto-approve + inline traces
```

| Flag | Purpose |
|------|---------|
| `--provider` | `anthropic` / `openai` / `gemini` / `ollama` |
| `--model` / `-m` | Override the provider default |
| `--budget` / `-b` | Session budget in USD (default 5.00) |
| `--yes` / `-y` | Auto-approve tool calls |
| `--verbose` / `-v` | Stream LLM and tool traces inline |
| `--new-thread` | Force a brand-new thread |
| `--thread <id>` | Continue a specific thread |

**Inside the TUI**

- Streamed output with collapsible tool-call widgets
- `@`-mention to attach files or directories
- Inline approval widgets — APPROVE / APPROVE_ALWAYS / DENY, with optional typed reason
- Ctrl+C copies selection (or quits when nothing is selected)

**Slash commands**

| Command | Purpose |
|---------|---------|
| `/help` | Show help and keybindings |
| `/cost` | Session cost and tokens |
| `/status` | Provider, model, budget, workspace |
| `/new` | Start a new conversation thread |
| `/clear` | Clear the screen |
| `/agent`, `/create-agent` | List / create custom agents |
| `/team`, `/create-team` | List / create agent teams |
| `/skill` | List, load, or install skills |
| `/plugin` | List / install / enable / disable plugins |
| `/worktree` | Manage git worktrees |
| `/quit` | Exit (aliases: `/exit`, `/q`) |

---

## One-shot & Scripted Runs

For CI, automation, or `cron` jobs:

```bash
maike run "Add input validation to the User model"
maike run "Fix failing tests" --provider gemini --budget 2.00 --yes
maike resume <session-id> --workspace <path>
```

`maike run` is non-interactive — the agent runs to completion (or budget exhaustion) and exits with a status code.

---

## Cost & Budgets

mAIke prices every call **before firing it**. Two enforcement layers:

- **`CostTracker`** (session-level) — runs `check_projected_session_budget()` against the next call's expected cost. If the projection would push the session past 95% of budget, the call is blocked and the session terminates gracefully.
- **`BudgetEnforcer`** (per-agent) — sub-agents and delegates have their own caps so a runaway delegate can't drain the parent's budget.

```bash
maike --budget 5.00                  # USD per session
maike cost                           # last/current session breakdown
maike cost <session-id>              # specific session
maike history                        # workspace session history
```

Sessions persist a structured cost ledger in `.maike/` for later analysis.

---

## Tool Safety Model

Every built-in tool carries a `RiskLevel`. The `SafetyLayer` intercepts every tool call and gates it:

| Level | Examples | Gate |
|-------|----------|------|
| **READ** | `Read`, `Grep`, `SemanticSearch`, `WebSearch`, `WebFetch` | runs freely; READ-safe tools execute in parallel |
| **WRITE** | `Write`, `Edit` | requires a git checkpoint |
| **EXECUTE** | `Bash` | requires checkpoint + inline approval (skipped with `--yes`) |
| **DESTRUCTIVE** | `rm -rf`, drop-table style | always prompts; pattern-matched in `safety/rules.py` |

**Read-before-edit:** the Edit tool refuses to run on a file the agent hasn't Read in this turn. After a successful edit the read state clears — the next Edit on that file requires a fresh Read. This kills the cascading-edit bug pattern where edit #2 operates on stale content.

**Bash idle timeout:** commands are killed if they produce no output for `idle_timeout` seconds (floor 10s). Errors include actionable recovery hints (`use timeout_class="long"` or `background=true`).

---

## Auto-Memory

mAIke distills durable project knowledge at the end of every session — synchronously, no LLM call, immune to TUI cancellation. Three memory types land in `<workspace>/.maike/memories/`:

- **`project_overview`** — purpose, tech stack, key modules
- **`key_decisions`** — architectural choices the agent made or learned
- **`pitfalls`** — errors encountered and how they were resolved

The next session reads `MEMORY.md` + topic files at startup and injects them as a high-priority context block. No re-exploration of the same files. No re-discovery of the same pitfalls.

---

## Threads & Worktrees

A workspace can hold many concurrent **threads** — each keeps its own message history, plan, and cost ledger.

```bash
maike threads                    # list threads
maike --thread <id>              # continue a specific thread
maike --new-thread               # force a brand-new thread
```

For isolated branch work, `maike worktree add/list/remove` wraps `git worktree` so an agent can operate on a side branch without touching your main checkout.

---

## Advisor

Pair a cheap fast executor (e.g. Gemini Flash Lite) with a frontier-model **advisor** that fires only at decisive moments — long exploration, repeated failures, before the first edit, before completion.

```bash
maike --provider gemini --model gemini-2.5-flash \
      --advisor --advisor-provider anthropic --advisor-model claude-opus-4-20250514 \
      --advisor-budget-pct 0.2
```

The advisor never runs tools. It returns 1–3 sentences of guidance that mAIke injects into the executor's next turn as a `<maike-advisor>` context block. It has its own budget cap (default 20% of session). Trigger conditions live in `maike/agents/advisor.py`.

---

## Sub-Agents & Delegation

The `Delegate` tool spawns read-only sub-agents in the background.

```python
# Inside an agent's tool call (illustrative)
Delegate(
    task="Find all callers of `parse_config`",
    agent_type="explore",      # explore | plan | verify | review | debug | implement | test
    background=True,           # default — returns a handle
)
# Later:
Delegate(action="check", handle="...")
Delegate(action="wait", handle="...")
```

- **Tool profiles** scope what each delegate type can do — `explore` gets `Read`/`Grep`/`SemanticSearch`/read-only `Bash`; `debug` adds `Edit`; `implement`/`test` get everything.
- **Result delivery** is inline (up to 3000 chars) — not file pointers — and persisted as `agent_runs` in the session DB.
- **Auto-resume** waits for *all* running delegates to complete before injecting results in one batch — preventing the parent from re-spawning delegates for already-covered work.
- `MAX_ASYNC_DELEGATES = 5`. The error message guides the LLM to use `action="wait"` if hit.

**Quality validation:** `assess_delegate_quality()` checks delegate outputs. Zero-tool-call delegates auto-retry once with a nudge. Quality flags (`good`/`suspect`/`hallucinated`) appear in notifications.

---

## Extensibility

| Surface | What it gives you | Install |
|---------|-------------------|---------|
| **Skills** | Reusable Markdown procedures the agent invokes on demand | `maike skill install <path-or-url>` |
| **Plugins** | Bundles of skills, agents, hooks, MCP and LSP configs | `maike plugin install <path-or-url>` |
| **MCP** | Any Model Context Protocol server appears as a first-class tool | `.mcp.json` |
| **LSP** | Language servers feed diagnostics and symbols into context | declared in plugin manifest |
| **Custom agents / teams** | Build via `/create-agent` and `/create-team` interactively | TUI |

---

## Evaluation

Two complementary suites:

```bash
# Built-in agentic test cases (seeders + verifiers + minimality scoring)
maike eval --suite all
maike eval --suite hard-agentic --keep-workspaces

# SWE-bench — clones repos at correct commit, runs the orchestrator, captures diff
maike swe-bench --variant lite                  # 300 instances
maike swe-bench --variant verified              # 500 instances
maike swe-bench --variant full                  # 2294 instances
maike swe-bench --instance-ids django__django-11019 sympy__sympy-20590
maike swe-bench --resume predictions.jsonl      # resume an interrupted run
```

The `agentic_eval_score()` weights: workspace verified (35%) · session completed (25%) · tests passing (15%) · error recovery (10%) · change minimality (10%) · wasted-call efficiency (5%).

SWE-bench output is a JSONL of predictions ready for the official Docker harness.

---

## Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Provider keys |
| `GOOGLE_GENAI_USE_VERTEXAI=True` + `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` | Use Gemini via Vertex AI |
| `OLLAMA_HOST` | Override the local Ollama endpoint |
| `BRAVE_SEARCH_API_KEY` / `GOOGLE_SEARCH_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID` | Web search backends |
| `MAIKE_DEFAULT_BUDGET_USD` | Default session budget (default 5.00) |
| `MAIKE_WORKSPACE` | Default workspace path |
| `MAIKE_LOG_LEVEL` | Logging level |

### MAIKE.md

`maike init` writes a per-workspace `MAIKE.md` with:

- Project context (auto-detected language, build commands, test commands)
- A `## Protected Files` section — files matching these patterns cannot be modified by the agent (post-hoc enforcement in `agents/core.py`)

MAIKE.md is yours to edit — mAIke reads it but never writes to it.

---

## Architecture at a Glance

```
CLI (cli.py)
  → Orchestrator (orchestrator/)         pipeline · state machine · partition fan-out · async delegates
    → AgentCore (agents/core.py)          ReAct loop:
        LLMGateway (gateway/)             provider adapters · retry · adaptive routing · token budgeting
        ToolRegistry (tools/)             Read · Write · Edit · Bash · Grep · Skill · SemanticSearch ·
                                          WebSearch · WebFetch · Delegate · Blackboard
        SafetyLayer (safety/)             risk-tier gating · checkpoint / approval gates · pattern blocks
        CostTracker (cost/)               per-call projection · 95% budget cutoff
        SessionToolTracker                inline waste/efficiency nudges (capped at 10/session)
        RepeatedFailureTracker            13-category error classification · recovery strategies
```

Pydantic v2 models in `atoms/`. SQLite-backed session store in `memory/`. ChromaDB for long-term memory. Tracing + Rich/file-based sinks in `observability/`. Textual TUI in `tui/`.

---

## CLI Reference

```bash
maike                                            # interactive TUI
maike setup                                      # write API keys to ~/.config/maike/.env
maike init                                       # generate MAIKE.md for the current workspace
maike run <task> [--provider --model --budget --yes]   # one-shot non-interactive
maike resume <session-id> --workspace <path>     # resume a specific session
maike threads                                    # list threads in this workspace
maike cost [<session-id>]                        # cost breakdown
maike history                                    # workspace session history
maike eval --suite <all|agentic|hard-agentic|live-repo> [--keep-workspaces]
maike swe-bench --variant <lite|verified|full> [--instance-ids ... | --resume FILE]
maike worktree <add|list|remove> ...             # git worktree management
maike skill <install|list|remove>
maike plugin <install|list|remove>
```

---

## License

[Apache 2.0](LICENSE) — see the LICENSE file for the full text.

## Status

mAIke is **alpha**. The CLI surface is stabilizing but expect breaking changes. Bug reports and PRs welcome.
