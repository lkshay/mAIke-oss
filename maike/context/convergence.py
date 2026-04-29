"""Convergence detection and nudge generation for long-running agents."""

from __future__ import annotations

from typing import Any


# Tool names that indicate file mutation.
_MUTATION_TOOLS = {"Write", "Edit", "write_file", "edit_file", "delete_file"}

# Tool names that indicate file reading.
_READ_TOOLS = {"Read", "Grep", "read_file", "search_files"}

# Tool names that indicate validation.
_VALIDATION_TOOLS = {"Bash", "syntax_check", "run_tests", "run_lint", "run_typecheck"}


def detect_spinning(conversation: list[dict[str, Any]], window: int = 6) -> bool:
    """Return True if the last *window* tool calls show a repetitive pattern.

    A spinning agent repeatedly calls the same tool on the same primary
    argument (typically a file path).  We consider the agent to be spinning
    when more than 50% of the recent tool calls share the same
    ``(tool_name, primary_arg)`` pair.
    """
    recent_calls = _extract_recent_tool_calls(conversation, window)
    if len(recent_calls) < 3:
        return False

    # Count occurrences of each (tool, primary_arg) pair.
    counts: dict[tuple[str, str], int] = {}
    for tool_name, primary_arg in recent_calls:
        key = (tool_name, primary_arg)
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return False

    max_count = max(counts.values())
    return max_count > len(recent_calls) / 2


def _outcome_signal(conversation: list[dict[str, Any]], window: int = 10) -> bool:
    """Detect spinning via high failure rate in recent tool calls."""
    results: list[bool] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(block.get("is_error", False))
    recent = results[-window:]
    if len(recent) < 4:
        return False
    failure_count = sum(1 for is_err in recent if is_err)
    return failure_count > len(recent) * 0.6


def _file_churn_signal(conversation: list[dict[str, Any]], window: int = 12) -> bool:
    """Detect spinning via repeated read-then-edit cycling on the same files."""
    calls = _extract_recent_tool_calls(conversation, window)
    file_reads: dict[str, int] = {}
    file_edits: dict[str, int] = {}
    for tool_name, path in calls:
        if not path:
            continue
        if tool_name in _READ_TOOLS:
            file_reads[path] = file_reads.get(path, 0) + 1
        elif tool_name in _MUTATION_TOOLS:
            file_edits[path] = file_edits.get(path, 0) + 1
    return any(
        file_reads.get(f, 0) >= 3 and file_edits.get(f, 0) >= 2
        for f in set(file_reads) | set(file_edits)
    )


def _ls_loop_signal(conversation: list[dict[str, Any]], window: int = 8) -> bool:
    """Detect ls/find loop: agent running directory listings repeatedly.

    Fires when 50%+ of recent tool calls are Bash commands that are just
    ``ls``, ``find``, or ``tree``.  Threshold lowered from 70% to 50% after
    the 76-turn transcript showed that approval-rejected ``ls`` retries
    interleaved with the occasional Read kept the ratio under the original
    cap, so the loop was never flagged despite 35+ ``ls`` variants.

    The cost of a false positive (a legitimate exploration session firing
    the nudge) is one advisory message; the cost of a false negative is
    a 35-turn approval-purgatory loop.
    """
    calls = _extract_recent_tool_calls(conversation, window)
    if len(calls) < 4:
        return False

    _LS_PREFIXES = ("ls ", "ls\n", "ls\t", "find ", "tree ", "tree\n")
    ls_count = 0
    for tool_name, arg in calls:
        if tool_name not in ("Bash", "execute_bash"):
            continue
        arg_stripped = arg.strip()
        if any(arg_stripped.startswith(p) for p in _LS_PREFIXES) or arg_stripped == "ls":
            ls_count += 1

    return ls_count >= len(calls) * 0.5


def detect_spinning_v2(conversation: list[dict[str, Any]]) -> bool:
    """Multi-signal spinning detection.

    Fires if ANY signal triggers:
    1. Outcome-based: >60% of last 10 tool calls failed
    2. File churn: same file read 3+ and edited 2+ times in last 12 calls
    3. Tool-call pattern: >40% identical (tool, arg) pairs in last 12 calls
    4. ls-loop: 50%+ of last 8 calls are directory listings
    """
    return (
        _outcome_signal(conversation)
        or _file_churn_signal(conversation)
        or detect_spinning(conversation, window=12)
        or _ls_loop_signal(conversation)
    )


_TEST_COMMAND_KEYWORDS = frozenset({
    "pytest", "unittest", "npm test", "pnpm test", "yarn test",
    "jest", "vitest", "mocha", "cargo test", "go test",
    "mvn test", "gradle test",
})


def _detect_test_loop(
    conversation: list[dict[str, Any]], window: int = 10,
) -> tuple[bool, str | None]:
    """Detect a test-fixing death spiral in recent tool calls.

    Returns ``(True, test_target)`` when 60%+ of recent Bash calls are
    test runs AND 60%+ of those failed.  ``test_target`` is the test
    file or command if extractable, else None.
    """
    import re

    bash_calls: list[tuple[str, bool]] = []  # (cmd, is_error)
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") in ("Bash", "execute_bash"):
                cmd = block.get("input", {}).get("cmd", "")
                bash_calls.append((cmd, False))  # placeholder
            elif block.get("type") == "tool_result":
                name = block.get("tool_name") or block.get("name") or ""
                if name in ("Bash", "execute_bash") and bash_calls:
                    is_error = block.get("is_error", False)
                    bash_calls[-1] = (bash_calls[-1][0], is_error)

    recent = bash_calls[-window:]
    if len(recent) < 4:
        return False, None

    # Count test-related calls
    test_calls: list[tuple[str, bool]] = []
    for cmd, is_error in recent:
        cmd_lower = cmd.lower()
        if any(kw in cmd_lower for kw in _TEST_COMMAND_KEYWORDS):
            test_calls.append((cmd, is_error))

    if len(test_calls) < len(recent) * 0.6:
        return False, None

    failing = sum(1 for _, err in test_calls if err)
    if failing < len(test_calls) * 0.6:
        return False, None

    # Extract test target from the most recent test command
    last_cmd = test_calls[-1][0]
    m = re.search(r"([\w/.\-]+(?:test[\w/.\-]*\.(?:py|js|ts|rs|go)))", last_cmd, re.IGNORECASE)
    test_target = m.group(1) if m else None

    return True, test_target


def build_escalated_nudge(
    conversation: list[dict[str, Any]],
    iteration_count: int,
    max_iterations: int,
    level: int,
) -> str:
    """Build an escalating convergence nudge.

    Level 1: Progress summary with suggested focus (same as build_convergence_nudge).
    Level 2: Strategy reset — force rethink, suggest delegation.
    Level 3: Final warning — wrap up with what you have.
    """
    if level <= 1:
        nudge = build_convergence_nudge(conversation, iteration_count, max_iterations)
        # Add escalation transparency so agent knows where it is in the curve.
        nudge += (
            "\n\n*This is escalation level 1 of 3. Level 2 forces a strategy "
            "reset and suggests delegation. Level 3 asks you to wrap up.*"
        )
        return nudge

    remaining = max_iterations - iteration_count if max_iterations > 0 else "unknown"
    files = _extract_file_mutations(conversation)
    validations = _extract_validation_results(conversation)

    if level == 2:
        sections = [
            f"## Strategy Reset — Level 2 of 3 (iteration {iteration_count}/{max_iterations})",
            "",
            "You have been working on this task for a while and appear to be stuck.",
            "*This is escalation level 2 of 3. Level 3 will ask you to stop and wrap up.*",
            "",
            "### What You've Done",
        ]
        if files:
            for path, iterations in sorted(files.items()):
                sections.append(f"- Modified `{path}` ({len(iterations)} times)")
        else:
            sections.append("- No files modified yet.")
        if validations:
            failing = [v for v in validations if not v[1]]
            passing = [v for v in validations if v[1]]
            sections.append(f"\n- {len(passing)} validations passed, {len(failing)} failed")
        # Detect if the agent has been reading framework/library internals.
        _framework_paths = ("site-packages/", "node_modules/", "vendor/", ".venv/lib/")
        _read_framework = any(
            any(fp in str(msg.get("content", "")) for fp in _framework_paths)
            for msg in conversation
            if msg.get("role") == "user"
        )
        if _read_framework:
            sections.extend([
                "",
                "### ⚠ Framework Modification Detected",
                "You appear to be exploring framework/library internals (site-packages, "
                "node_modules, etc.). **Stop and reconsider**: can you solve this at the "
                "application level instead? Create wrappers, adapters, or decorators "
                "rather than monkey-patching framework code.",
            ])

        sections.extend([
            "",
            "### Required Actions",
            "1. **Stop** and list what you have tried so far and why each attempt failed.",
            "2. **Propose a fundamentally different approach** — do not retry the same strategy.",
            "3. **Delegate now** — hand this to a fresh agent with clean context:",
            f"4. You have ~{remaining} iterations remaining. Use them wisely.",
        ])
        # Build a concrete delegation example from the most recent failure.
        failing_outputs = [v[2] for v in validations if not v[1]]
        if failing_outputs:
            _fail_snippet = failing_outputs[-1][:120].replace("\n", " ").replace('"', "'")
            _fail_files = [path for path in files] if files else []
            _fail_file = _fail_files[-1] if _fail_files else "the failing module"
            sections.append(
                f'\n   ```\n'
                f'   Delegate(\n'
                f'     task="Fix the issue in {_fail_file}. Error: {_fail_snippet}",\n'
                f'     context="I tried modifying {", ".join(_fail_files[:3]) if _fail_files else "several files"} '
                f'but the tests still fail. Read the file fresh and try a different approach.",\n'
                f'     background=false\n'
                f'   )\n'
                f'   ```'
            )
        # Detect specific failure patterns for targeted guidance.
        all_fail_text = " ".join(failing_outputs).lower()
        if "recursionerror" in all_fail_text or "maximum recursion" in all_fail_text:
            sections.append(
                "\n### Detected Issue: Infinite Recursion\n"
                "Your code has an infinite recursion bug. Before making more changes:\n"
                "- Add a print statement at the top of the recursive function to trace its arguments.\n"
                "- Check that every recursive call reduces the problem toward a base case.\n"
                "- If you modified an existing recursive function, consider reverting and starting with a minimal change."
            )
        elif "timeout" in all_fail_text or "timed out" in all_fail_text:
            sections.append(
                "\n### Detected Issue: Timeout / Infinite Loop\n"
                "Your code appears to hang or time out. Add progress output or reduce the input size to find the bottleneck."
            )
        # Check for directory-listing loop (flash model behavior).
        if _ls_loop_signal(conversation):
            sections.append(
                "\n### Detected Issue: Directory Listing Loop\n"
                "You are repeatedly running `ls` or `find` without making progress. "
                "Stop listing directories. Instead:\n"
                "1. If you are waiting for delegates, end your turn.\n"
                "2. If you need specific information, use Grep to search for it directly.\n"
                "3. If you have already delegated exploration, summarize what you "
                "delegated and wait for those results."
            )
        # Check for test-fixing death spiral
        is_test_loop, test_target = _detect_test_loop(conversation)
        if is_test_loop:
            target_note = f" (`{test_target}`)" if test_target else ""
            sections.append(
                f"\n### Detected Issue: Test-Fixing Loop\n"
                f"You have been running and failing the same tests{target_note} "
                f"repeatedly. Small edits are not converging.\n\n"
                f"**Concrete steps to break out:**\n"
                f"1. **Write a standalone debug script** — create a small script that "
                f"imports the function under test and prints its actual behavior with "
                f"specific inputs. Run this script instead of the full test suite.\n"
                f"2. **Fix the function based on debug output**, not based on test "
                f"assertions.\n"
                f"3. **Only then re-run the tests** to confirm.\n"
                f"4. If you cannot fix it in 3 more iterations, use **Delegate** to "
                f"hand the specific failing test to a fresh agent."
            )
        return "\n".join(sections)

    # Level 3: Final warning
    sections = [
        f"## Final Warning — Level 3 of 3 (iteration {iteration_count}/{max_iterations})",
        "",
        f"You have ~{remaining} iterations remaining. **Stop iterating and wrap up.**",
        "*This is the final escalation level. No more nudges — finish now.*",
        "",
        "### Instructions",
        "1. Finish with what you have — do not start new approaches.",
        "2. Run a final validation if you haven't recently.",
        "3. Write a brief summary of what works and what remains incomplete.",
        "4. If tests are failing, leave them as-is rather than making more changes.",
        "5. Do NOT re-read files you have already read 3+ times.",
    ]
    return "\n".join(sections)


def build_convergence_nudge(
    conversation: list[dict[str, Any]],
    iteration_count: int,
    max_iterations: int,
) -> str:
    """Build a structured progress summary for injection into the conversation.

    Extracts from the conversation history:
    1. Files created/modified and which iterations they were touched.
    2. Validation results (pass/fail) from tool results.
    3. Contract items from the initial message (if acceptance-contract present).
    """
    files = _extract_file_mutations(conversation)
    validations = _extract_validation_results(conversation)
    contract_items = _extract_contract_items(conversation)
    covered_files = set(files.keys())

    sections: list[str] = [
        f"## Convergence Check (iteration {iteration_count}/{max_iterations})",
    ]

    # Files modified.
    if files:
        sections.append("\n### Files Modified")
        for path, iterations in sorted(files.items()):
            iter_labels = ", ".join(f"i{i}" for i in iterations[-5:])  # last 5
            sections.append(f"- `{path}` (touched at {iter_labels})")
    else:
        sections.append("\n### Files Modified\nNone yet.")

    # Validation results.
    if validations:
        sections.append("\n### Validation Results")
        for tool, passed, snippet in validations[-6:]:  # last 6
            status = "PASS" if passed else "FAIL"
            sections.append(f"- {tool}: {status} — {snippet[:80]}")
    else:
        sections.append("\n### Validation Results\nNo validation run yet.")

    # Contract items.
    if contract_items:
        sections.append("\n### Contract Coverage")
        for item in contract_items:
            # Simple heuristic: if any word from the item appears in covered files, mark covered.
            item_lower = item.lower()
            covered = any(
                f.lower().rstrip(".py").rstrip(".js").rstrip(".ts") in item_lower
                or item_lower in f.lower()
                for f in covered_files
            )
            marker = "[COVERED]" if covered else "[UNCOVERED]"
            sections.append(f"- {marker} {item}")

    # Suggested focus.
    failing_validations = [v for v in validations if not v[1]]
    sections.append("\n### Suggested Focus")
    if failing_validations:
        last_fail = failing_validations[-1]
        sections.append(
            f"Fix the failing {last_fail[0]} check before adding new functionality. "
            "Stop modifying files that already pass validation."
        )
    elif contract_items:
        uncovered = [item for item in contract_items if not any(
            f.lower().rstrip(".py").rstrip(".js").rstrip(".ts") in item.lower()
            for f in covered_files
        )]
        if uncovered:
            sections.append(
                f"Address uncovered contract items: {uncovered[0]}. "
                "Focus on completing one item at a time."
            )
        else:
            sections.append(
                "All contract items appear covered. Run final validation and finish."
            )
    else:
        sections.append(
            "Review what remains incomplete and focus on finishing rather than iterating."
        )

    # Narrow-focus warning: agent is fixated on 1-2 files.
    if max_iterations > 0 and iteration_count >= max_iterations * 0.3:
        total_mutations = sum(len(iters) for iters in files.values())
        if len(files) <= 2 and total_mutations >= 3:
            sections.append(
                "\n### Warning: Narrow Focus\n"
                "You have spent significant time modifying only 1-2 files. If the task "
                "requires multiple components, build a minimal skeleton of ALL required "
                "files first, then iterate on each. Perfecting one component before "
                "others exist leads to rework when integration reveals design "
                "mismatches.\n\n"
                "**Action:** List all files the task requires. Create stubs for any "
                "that don't exist yet. Then return to fixing the current issue."
            )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_recent_tool_calls(
    conversation: list[dict[str, Any]], window: int,
) -> list[tuple[str, str]]:
    """Extract the last *window* ``(tool_name, primary_arg)`` pairs."""
    calls: list[tuple[str, str]] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            inp = block.get("input", {})
            primary = str(inp.get("path") or inp.get("file_path") or inp.get("command", "") or "")
            calls.append((name, primary))
    return calls[-window:]


def _extract_file_mutations(
    conversation: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Map file paths to the iteration indices where they were mutated."""
    files: dict[str, list[int]] = {}
    iteration = 0
    for message in conversation:
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                has_tool_use = any(b.get("type") == "tool_use" for b in content)
                if has_tool_use:
                    iteration += 1
                for block in content:
                    if block.get("type") == "tool_use" and block.get("name") in _MUTATION_TOOLS:
                        inp = block.get("input", {})
                        path = str(inp.get("path") or inp.get("file_path") or "")
                        if path:
                            files.setdefault(path, []).append(iteration)
    return files


def _extract_validation_results(
    conversation: list[dict[str, Any]],
) -> list[tuple[str, bool, str]]:
    """Extract ``(tool_name, passed, snippet)`` from validation tool results."""
    results: list[tuple[str, bool, str]] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            name = block.get("tool_name") or block.get("name") or ""
            if name not in _VALIDATION_TOOLS:
                continue
            is_error = block.get("is_error", False)
            output = str(block.get("content", ""))
            passed = not is_error and "fail" not in output.lower()[:200]
            snippet = output.strip().split("\n")[-1] if output.strip() else "(empty)"
            results.append((name, passed, snippet))
    return results


def _extract_contract_items(
    conversation: list[dict[str, Any]],
) -> list[str]:
    """Parse contract items from the acceptance-contract in the initial message."""
    if not conversation:
        return []
    first_content = conversation[0].get("content", "")
    if not isinstance(first_content, str):
        return []

    # Look for "Required Files and Docs" or "Executable Checks" sections.
    items: list[str] = []
    in_section = False
    for line in first_content.split("\n"):
        heading = line.strip().lower()
        if "required files" in heading or "executable checks" in heading:
            in_section = True
            continue
        if in_section:
            if line.startswith("#") or line.startswith("---"):
                in_section = False
                continue
            stripped = line.strip().lstrip("-").lstrip("*").strip()
            if stripped and not stripped.startswith("```"):
                items.append(stripped)
            if len(items) >= 20:
                break
    return items
