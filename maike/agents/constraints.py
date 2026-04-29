"""Task-constraint extraction — produces a markdown "Task rules" block.

The react agent is itself an LLM and reads natural-language guidance well.
We used to extract constraints as structured JSON and feed them to
``SafetyLayer`` for hard tool blocks, but the JSON-parse step was fragile
(see commit 6eee296 for the verdict-classifier equivalent).  This module
now produces markdown text that is injected verbatim into the react agent's
system prompt.  The agent reads and honors the rules.

What still exists:
  - MAIKE.md ``## Protected Files`` parsing → ``read_only_patterns`` list
    (deterministic, non-LLM).  These feed a post-hoc "you mutated a
    protected file" nag in ``agents/core.py`` that has always lived there.
  - ``check_path_against_constraints`` — the list-based glob check used by
    that post-hoc nag.  Simple and deterministic.
  - ``_select_extractor_gateway`` — Ollama-first gateway selector, shared
    with the verdict classifier.

What was removed (commit <this>):
  - ``TaskConstraints`` dataclass with 5 array fields.
  - ``_sanitize_constraints`` (defensive filter for hallucinated broad
    patterns).
  - ``check_bash_against_constraints`` / ``check_write_path_against_constraints``
    (regex + glob helpers consumed by SafetyLayer).
  - ``SafetyLayer._task_constraint_bash_check`` / ``_task_constraint_path_check``
    (hard tool blocks derived from extracted JSON).
  - ``_extract_bash_write_targets`` (Bash command parser).

The v3 eval already showed the prompt block was the dominant behavior-change
mechanism (xarray-6992 needed zero SafetyLayer blocks; django-11019 had only
2-3 blocks against a 132-line productive diff).  Markdown-only keeps that
mechanism and drops the fragile translator.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown extractor prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You read a coding task description and rewrite its CONSTRAINTS as a short,
directive-style markdown block for a downstream coding agent.

The task may target ANY language / ecosystem (Python, Node/TypeScript, Rust,
Go, Ruby, Java, etc.).  Your output must use the SAME ecosystem's vocabulary
as the task.  Do not default to Python examples unless the task is Python.

Rules for your output:
- Emit ONLY the markdown block.  No preamble, no commentary, no trailing text.
- Start with the heading: "## Task rules"
- Use imperative bullets.  Name specific forbidden commands from the task's
  language/ecosystem:
    * Python  → pytest, python -m pytest, tox, pip install
    * Node/TS → npm test, yarn test, jest, mocha, npm install
    * Rust    → cargo test, cargo check --tests
    * Go      → go test, go vet
    * Ruby    → rspec, bundle exec, gem install
- Infer implicit constraints using the task's context:
  * "sandboxed environment" / "offline" / "no internet" → forbid curl, wget,
    git fetch/clone/pull, git apply, git am (language-agnostic).
  * "no test environment" / "do not run tests" / "no venv configured" →
    forbid the project's test runner (see list above; pick by language).
  * "do not install dependencies" → forbid the project's package manager's
    install verb (pip install, npm install, cargo add, go get, gem install…).
  * "do not add test files" → forbid creating files matching the project's
    test-file convention (Python: test_*.py; Node: *.test.{js,ts},
    *.spec.{js,ts}; Rust: tests/*.rs; Go: *_test.go).
- Keep the block under 200 words.  Prefer specific forbidden-command examples
  over abstract rules.
- If there are no explicit or clearly-implied constraints, output exactly:
  "## Task rules\\n\\n(no explicit constraints)"
"""


_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # Python task — pytest/venv vocabulary.
    {
        "role": "user",
        "content": (
            'Task: "Fix the bug in residue_ntheory.py.  Do NOT run tests — no venv '
            'is configured.  Verify changes via `python3 -c "import ast; '
            'ast.parse(...)"` on modified files.  Do NOT install dependencies.  '
            'This is a sandboxed offline environment — no internet access."'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Task rules\n\n"
            "- Do not run tests.  No venv is configured.  Do not invoke "
            "`pytest`, `python -m pytest`, `tox`, or run `test_*.py` scripts.\n"
            "- Do not install dependencies.  Avoid `pip install`, "
            "`apt-get install`, `conda install`.\n"
            "- Sandboxed offline environment.  Do not use `curl`, `wget`, "
            "`git fetch`, `git clone`, `git pull`, `git apply`, or `git am`.\n"
            "- Verify syntactic correctness with "
            "`python3 -c \"import ast; ast.parse(open('<file>').read())\"` "
            "on every modified file."
        ),
    },
    # Node/TypeScript task — npm/jest vocabulary, different test-file convention.
    {
        "role": "user",
        "content": (
            'Task: "Add input validation in src/models/User.ts.  Do not run '
            '`npm test` or `jest` — the test harness is broken on this branch.  '
            'Do not install new packages; do not add new spec files."'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Task rules\n\n"
            "- Do not run tests.  Avoid `npm test`, `yarn test`, `jest`, or "
            "`mocha` — the harness is known-broken on this branch.\n"
            "- Do not install new packages.  Avoid `npm install`, "
            "`yarn add`, `pnpm add`.\n"
            "- Do not create new spec/test files (e.g. `*.test.ts`, "
            "`*.spec.ts`, `*.test.js`).\n"
            "- Verify with `tsc --noEmit` on modified files instead of "
            "running tests."
        ),
    },
    # Rust task — cargo vocabulary.
    {
        "role": "user",
        "content": (
            'Task: "Refactor the parser in src/lex.rs.  Do not run `cargo test` — '
            'the integration tests have an external dependency.  Do not modify '
            '`Cargo.toml`; verify with `cargo check` only."'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Task rules\n\n"
            "- Do not run tests.  Avoid `cargo test`, `cargo test --all`.  "
            "Integration tests require an external dependency not available.\n"
            "- Do not modify `Cargo.toml` or `Cargo.lock`.\n"
            "- Do not add new test files under `tests/` or "
            "`src/**/*_test.rs`.\n"
            "- Verify with `cargo check` on the modified crate."
        ),
    },
    # Go task — go test / go.mod vocabulary.
    {
        "role": "user",
        "content": (
            'Task: "Fix the race condition in internal/cache/cache.go.  Do not '
            'run `go test`; verify with `go vet` only.  Do not add new *_test.go '
            'files."'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "## Task rules\n\n"
            "- Do not run tests.  Avoid `go test`, `go test ./...`.\n"
            "- Do not add new `*_test.go` files.\n"
            "- Do not modify `go.mod` or `go.sum`.\n"
            "- Verify with `go vet ./...` on the modified package."
        ),
    },
    # Control: no constraints → emit the empty marker.
    {
        "role": "user",
        "content": 'Task: "Add a utility function for parsing ISO-8601 dates."',
    },
    {
        "role": "assistant",
        "content": "## Task rules\n\n(no explicit constraints)",
    },
]


# ---------------------------------------------------------------------------
# Ollama-first gateway selection (unchanged from prior version)
# ---------------------------------------------------------------------------


_OLLAMA_PROBE_CACHE: dict[tuple[str, str], bool] = {}
_OLLAMA_PROBE_TIMEOUT_S = 0.5
_DEFAULT_OLLAMA_MODEL = "gemma3:4b"


async def _ollama_reachable_with_model(host: str, model: str) -> bool:
    """Probe Ollama /api/tags with a short timeout.  True iff reachable AND model present."""
    cache_key = (host, model)
    if cache_key in _OLLAMA_PROBE_CACHE:
        return _OLLAMA_PROBE_CACHE[cache_key]

    reachable = False
    try:
        import httpx

        url = host.rstrip("/") + "/api/tags"
        async with httpx.AsyncClient(timeout=_OLLAMA_PROBE_TIMEOUT_S) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                names = {entry.get("name", "") for entry in data.get("models", [])}
                reachable = (
                    model in names
                    or f"{model}:latest" in names
                    or any(n.split(":")[0] == model.split(":")[0] for n in names)
                )
    except Exception as exc:
        log.debug("Ollama probe failed (non-fatal): %s", exc)
        reachable = False

    _OLLAMA_PROBE_CACHE[cache_key] = reachable
    return reachable


async def _select_extractor_gateway(
    session_bg_gateway: Any,
    session_provider: str,
) -> tuple[Any, str, str]:
    """Returns (gateway, provider, model).  Prefers local Ollama, falls back
    to the session's ``_bg_gateway`` (executor's cheap tier)."""
    from maike.constants import cheap_model_for_provider

    override_provider = os.getenv("MAIKE_CONSTRAINT_EXTRACTOR_PROVIDER", "").strip()
    override_model = os.getenv("MAIKE_CONSTRAINT_EXTRACTOR_MODEL", "").strip()
    ollama_host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    if ollama_host and not ollama_host.startswith(("http://", "https://")):
        ollama_host = "http://" + ollama_host

    if override_provider and override_model:
        try:
            from maike.gateway import LLMGateway
            gw = LLMGateway(
                cost_tracker=getattr(session_bg_gateway, "cost_tracker", None),
                tracer=getattr(session_bg_gateway, "tracer", None),
                provider_name=override_provider,
                silent=True,
            )
            return gw, override_provider, override_model
        except Exception as exc:
            log.debug("Explicit extractor gateway override failed: %s", exc)

    preferred_model = override_model or _DEFAULT_OLLAMA_MODEL
    try:
        if await _ollama_reachable_with_model(ollama_host, preferred_model):
            from maike.gateway import LLMGateway
            gw = LLMGateway(
                cost_tracker=getattr(session_bg_gateway, "cost_tracker", None),
                tracer=getattr(session_bg_gateway, "tracer", None),
                provider_name="ollama",
                silent=True,
            )
            return gw, "ollama", preferred_model
    except Exception as exc:
        log.debug("Ollama extractor path failed, falling back: %s", exc)

    return session_bg_gateway, session_provider, cheap_model_for_provider(session_provider)


# ---------------------------------------------------------------------------
# Conversational-prompt short circuit (unchanged)
# ---------------------------------------------------------------------------


def _is_conversational(task: str) -> bool:
    """Return True if the task is a question/conversation, not a coding task.

    Extraction is skipped for conversational prompts — no files to protect and
    no constraints to infer.
    """
    stripped = task.strip().lower()
    action_words = {
        "create", "write", "build", "implement", "add", "fix",
        "update", "modify", "refactor", "delete", "remove",
    }
    words = set(stripped.split())

    if stripped in ("continue", "go ahead", "keep going", "yes", "ok", "proceed", "do it", "go"):
        return True
    if len(words) <= 8 and stripped.endswith("?") and not (words & action_words):
        return True
    conversational_prefixes = (
        "what ", "how ", "why ", "when ", "where ", "who ",
        "can you ", "could you ", "tell me ", "explain ",
        "show me ", "list ", "describe ", "summarize ",
        "help me understand", "what's ", "what is ", "what are ",
        "how do ", "how can ", "how much ", "how many ",
    )
    if any(stripped.startswith(p) for p in conversational_prefixes):
        if not (words & action_words):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API — extract markdown
# ---------------------------------------------------------------------------


_EMPTY_MARKDOWN = "## Task rules\n\n(no explicit constraints)"
_MAX_MARKDOWN_CHARS = 1500  # ~200 words; truncation point if the model goes long


def _normalize_markdown(text: str) -> str:
    """Strip preamble, code fences, and cap length.

    The extractor prompt forbids preamble, but models sometimes ignore that.
    We defensively trim anything before the ``## Task rules`` heading and
    truncate at ``_MAX_MARKDOWN_CHARS``.
    """
    text = (text or "").strip()
    if not text:
        return _EMPTY_MARKDOWN
    # Strip markdown fences if present.
    if text.startswith("```"):
        # drop first line + trailing fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    # If the model wrote preamble before the heading, drop it.
    idx = text.find("## Task rules")
    if idx > 0:
        text = text[idx:]
    # Truncate.
    if len(text) > _MAX_MARKDOWN_CHARS:
        text = text[:_MAX_MARKDOWN_CHARS].rstrip() + "\n\n(truncated)"
    return text


async def extract_task_constraints(
    task: str,
    *,
    session_bg_gateway: Any,
    session_provider: str,
) -> str:
    """Return a markdown ``## Task rules`` block for the given task.

    Always returns a non-empty string.  On any error (network, timeout,
    conversational prompt), returns ``_EMPTY_MARKDOWN`` — the downstream
    injection is still safe; the agent just sees an empty rules block.
    """
    if _is_conversational(task):
        return _EMPTY_MARKDOWN

    gateway, provider, model = await _select_extractor_gateway(
        session_bg_gateway, session_provider,
    )
    log.debug("Constraint extractor using provider=%s model=%s", provider, model)

    try:
        messages = [
            *_FEW_SHOT_EXAMPLES,
            {"role": "user", "content": f'Task: "{task}"'},
        ]
        result = await asyncio.wait_for(
            gateway.call(
                system=_SYSTEM_PROMPT,
                messages=messages,
                model=model,
                temperature=0.0,
                max_tokens=1024,
                tools=None,
            ),
            timeout=15.0,
        )
        markdown = _normalize_markdown(result.content or "")
        if markdown and markdown != _EMPTY_MARKDOWN:
            log.info("Extracted task-rules markdown (%d chars)", len(markdown))
        return markdown
    except asyncio.TimeoutError:
        log.debug("Constraint extractor timed out")
        return _EMPTY_MARKDOWN
    except Exception as exc:  # noqa: BLE001 — never block session start
        log.debug("Constraint extraction failed (non-fatal): %s", exc)
        return _EMPTY_MARKDOWN


async def build_task_constraints(
    *,
    task: str,
    workspace: Any,
    session_bg_gateway: Any,
    session_provider: str,
) -> tuple[str, list[str]]:
    """Build the full task-constraints bundle for a session.

    Returns ``(markdown_text, read_only_patterns)``.

    - ``markdown_text``: the ``<maike-constraints>`` block content.  Prepends
      MAIKE.md ``## Protected Files`` (if any) to the extractor output.
    - ``read_only_patterns``: glob list from MAIKE.md, used by the post-hoc
      check in ``agents/core.py`` to detect violations after they happen.

    Never raises.
    """
    from pathlib import Path
    import re as _re

    # 1. MAIKE.md Protected Files — deterministic file parse.
    read_only_patterns: list[str] = []
    maike_md_block = ""
    try:
        if workspace is not None:
            maike_md_path = Path(workspace) / "MAIKE.md"
            if maike_md_path.exists():
                md_text = maike_md_path.read_text(encoding="utf-8")
                m = _re.search(
                    r"##\s*Protected\s+Files\s*\n((?:[-*]\s+.+\n?)+)",
                    md_text, _re.IGNORECASE,
                )
                if m:
                    for line in m.group(1).strip().splitlines():
                        # Strip ONE leading "- " or "* " bullet marker, not
                        # a set of chars (otherwise "* **/vendor/**" loses
                        # the glob prefix).
                        stripped = line.strip()
                        if stripped.startswith("- "):
                            pattern = stripped[2:].strip()
                        elif stripped.startswith("* "):
                            pattern = stripped[2:].strip()
                        else:
                            pattern = stripped
                        if pattern:
                            read_only_patterns.append(pattern)
                    if read_only_patterns:
                        files_md = "\n".join(f"- `{p}`" for p in read_only_patterns)
                        maike_md_block = (
                            "## Protected files (from MAIKE.md)\n\n"
                            "Do not modify any file matching these patterns:\n\n"
                            f"{files_md}\n\n"
                        )
    except Exception as exc:  # noqa: BLE001
        log.debug("MAIKE.md parse failed (non-fatal): %s", exc)

    # 2. LLM-extracted task-rules markdown.
    try:
        extracted_md = await extract_task_constraints(
            task=task,
            session_bg_gateway=session_bg_gateway,
            session_provider=session_provider,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("Task constraint extraction failed (non-fatal): %s", exc)
        extracted_md = _EMPTY_MARKDOWN

    combined = (maike_md_block + extracted_md).strip()
    return combined, read_only_patterns


# ---------------------------------------------------------------------------
# Deterministic post-hoc read-only check (unchanged)
# ---------------------------------------------------------------------------


def check_path_against_constraints(
    file_path: str,
    read_only_patterns: list[str],
) -> str | None:
    """Check if a file path matches any read-only pattern.

    Returns the matching pattern, or None.  Used only by the post-hoc
    "you mutated a protected file" nag in ``agents/core.py``, which operates
    on the deterministic ``read_only_patterns`` list from MAIKE.md.
    """
    if not read_only_patterns:
        return None
    import fnmatch

    for pattern in read_only_patterns:
        if file_path == pattern:
            return pattern
        if fnmatch.fnmatch(file_path, pattern):
            return pattern
        if "/" not in pattern and file_path.endswith("/" + pattern):
            return pattern
        if file_path.endswith(pattern):
            return pattern
    return None


def merge_constraint_sources(
    *,
    extracted: list[str],
    maike_md_patterns: list[str],
    cli_patterns: list[str],
) -> list[str]:
    """Merge read-only path constraints from all sources, deduplicating.

    Kept for back-compat with existing callers and tests.  Order:
    CLI → MAIKE.md → extracted.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for pattern in cli_patterns + maike_md_patterns + extracted:
        if pattern not in seen:
            seen.add(pattern)
            merged.append(pattern)
    return merged
