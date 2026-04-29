# Architecture

This document covers *why* each major component of mAIke exists. For *what* each one does at a high level, the README's [Architecture](../README.md#architecture) sketch is enough; this is for readers who want the design reasoning.

```
CLI → Orchestrator → AgentCore (ReAct loop)
                       ├─ LLMGateway      provider adapters · retry · adaptive routing
                       ├─ ToolRegistry    Read · Write · Edit · Bash · Grep · …
                       ├─ SafetyLayer     risk-tier gating · checkpoint gates
                       ├─ CostTracker     pre-call projection · 95% cutoff
                       └─ Memory          SQLite session store · ChromaDB long-term
```

---

## Orchestrator (`orchestrator/`)

**What it does:** owns the pipeline state machine — session boot, agent dispatch, partition fan-out for parallel coding, async delegate management, session summary writing, and auto-memory extraction at session end.

**Why it exists:** to keep `AgentCore` focused on the ReAct loop and nothing else. The Orchestrator is the deterministic shell around the non-deterministic agent — it handles all the bookkeeping (cost ledger writes, session DB persistence, delegate result collection) so the agent code can stay narrow.

**Non-obvious tradeoff:** the auto-memory extractor runs *synchronously* in the `finally` block, not as an async task. This is intentional: TUI cancellation propagates aggressively through asyncio, and we want auto-memory to land even when the user hits Ctrl+C. The cost is a 0.5–2s pause at session end; the benefit is that no Ctrl+C can ever skip memory extraction.

---

## AgentCore (`agents/core.py`)

**What it does:** runs the ReAct loop. The LLM proposes tool calls, AgentCore resolves them through `ToolRegistry`, gates them through `SafetyLayer`, executes them through `LocalRuntime`, and feeds results back. Loop terminates on final response, budget exhaustion, iteration cap, or cancellation.

**Why it exists:** because every provider's chat-completion API has different quirks (Gemini's thought-signatures, Anthropic's tool_use blocks, OpenAI's function-calling JSON), and abstracting those out into one universal runner means the rest of the system never has to care which provider is in use.

**Non-obvious tradeoff:** adaptive model routing is *disabled for Gemini*. Gemini maintains a `thought_signature` chain across turns; switching models mid-conversation invalidates the chain and produces 400 errors. We chose to forfeit the cheap-to-strong escalation feature on Gemini rather than degrade reliability. Other providers route freely.

---

## LLMGateway (`gateway/`)

**What it does:** provider-agnostic LLM interface. Adapters for Anthropic, OpenAI, Gemini, and Ollama all conform to the same `generate()` shape, with retry/backoff, token budgeting, and pricing lookups handled centrally.

**Why it exists:** so adding a new provider is a one-file change (a new adapter), not a refactor across the codebase.

**Non-obvious tradeoff:** the gateway maintains a separate `_bg_gateway = LLMGateway(silent=True)` for cheap background calls (compaction, constraint extraction, advisor). The `silent` flag suppresses tracing so background calls don't clutter the TUI. Costs are still tracked. The duplication is intentional — sharing one adapter caused Gemini's native history to corrupt under concurrent reads.

---

## ToolRegistry (`tools/`)

**What it does:** registers all built-in tools (Read, Write, Edit, Bash, Grep, SemanticSearch, WebSearch, WebFetch, Delegate, Blackboard, Skill) plus any user-installed Skills/Plugins/MCP tools, and exposes a `resolve()` API that maps an LLM-emitted name to a concrete tool.

**Why it exists:** LLMs hallucinate tool names. The registry's 3-tier name resolution (exact → normalized → alias) catches `read_file`, `read-file`, `Read`, etc. and routes them all to the same tool. Without this, every name miss is a wasted round-trip.

**Non-obvious tradeoff:** READ-safe tools execute in parallel via `asyncio.gather()`; mixed-risk batches run sequentially. This is the only place we trade safety for latency, and only after verifying the tool is genuinely side-effect-free.

---

## SafetyLayer (`safety/`)

**What it does:** intercepts every tool call, checks its risk level (`READ → WRITE → EXECUTE → DESTRUCTIVE`), and gates accordingly — checkpoints before WRITE, approval prompts before EXECUTE, hard pattern blocks for known-destructive Bash patterns (`rm -rf /`, `:(){:|:&};:`, etc.).

**Why it exists:** because a coding agent that can shell out to bash is exactly as dangerous as a junior engineer with sudo. The risk tiers force a conscious decision at each level.

**Non-obvious tradeoff:** the protected-files check is *post-hoc*, not pre-flight. Agents can attempt to mutate a protected file, but the check happens after the edit and rejects the attempt. We tried pre-flight and the false-positive rate (legitimate edits blocked by glob patterns) was worse than the post-hoc rollback.

---

## CostTracker (`cost/`)

**What it does:** prices every API call before it fires. Two layers — `CostTracker` for session-level enforcement, `BudgetEnforcer` for per-agent caps so a runaway delegate can't drain the parent.

**Why it exists:** because every other coding agent I've used surprised me with the bill. mAIke surfaces cost before the call, not after — and aborts the session at 95% of budget rather than overshooting.

**Non-obvious tradeoff:** the projected cost uses the *next-call estimate* (input tokens × input price + max-output × output price), which is a worst-case bound. Real costs are lower because agents rarely hit max output. This means we abort sessions slightly earlier than strictly necessary; we chose conservative over surprise-billed.

---

## Memory (`memory/`)

**What it does:** three layers — `SessionStore` (SQLite, durable session state), `WorkingMemory` (in-RAM, content-aware pruning during the loop), `LongTermMemory` (ChromaDB vector store for semantic recall) and `TypedLongTermMemory` (Markdown files for auto-memory).

**Why it exists:** the working memory has to fit in the model's context window; the session store has to survive crashes; the long-term memory has to persist project knowledge across sessions. Three different problems, three different storage layers.

**Non-obvious tradeoff:** auto-memory writes Markdown files, not vector embeddings, even though we have ChromaDB. Reason: Markdown is human-readable and human-editable. Users *will* go look at `.maike/memories/` and want to delete a wrong claim. They cannot do that with a vector embedding. Embeddings are reserved for fuzzy semantic recall during a session; persistent project knowledge is plain text.

---

## Context Management (`context/`)

**What it does:** progressive compression cascade when payload exceeds the model's effective limit. Four levels: strip tool schema descriptions → truncate artifacts → strip environment blocks → aggressive artifact truncation.

**Why it exists:** the alternative — failing the request and asking the user to reduce input — is worse. Compression should be invisible until it isn't enough.

**Non-obvious tradeoff:** the compression cascade is destructive within a turn but doesn't modify session memory. If you `--resume` a session, the original messages are intact; the compression only affected what got sent to the LLM that one turn.

---

## Observability (`observability/`)

Tracing emits structured events (`tool_start`, `tool_result`, `llm_start`, `llm_call`, `agent_complete`) consumed by the TUI bridge, file-based sinks for offline analysis, and the eval harness. Event kinds are typed constants — `TraceEventKind` — not raw strings, to keep the consumer registry honest.

---

## What's not here

- **Plugins, Skills, MCP, LSP** are described in the [README's Extensibility section](../README.md#extensibility). They're surfaces, not core architecture.
- **The TUI** (`tui/`) is intentionally out of scope here — it's a Textual app with a handler-registry bridge to the trace stream, and its design choices are mostly UX, not architecture.
- **The eval framework** (`eval/`) lives in its own world; see [SWE-bench Integration](../README.md#evaluation) for the user-facing slice.
