#!/usr/bin/env sh

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

MAIKE_BIN="${MAIKE_BIN:-${ROOT_DIR}/.venv/bin/maike}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SMOKE_TASK="${SMOKE_TASK:-Create a README with a one-file hello world app}"
SMOKE_LANGUAGE="${SMOKE_LANGUAGE:-python}"
SMOKE_BUDGET_USD="${SMOKE_BUDGET_USD:-0.0001}"
KEEP_WORKSPACES="${KEEP_WORKSPACES:-0}"
MAIKE_SMOKE_ALL_PROVIDERS="${MAIKE_SMOKE_ALL_PROVIDERS:-0}"
MAIKE_SMOKE_SKIP_ANTHROPIC="${MAIKE_SMOKE_SKIP_ANTHROPIC:-1}"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
WORKSPACES=""
FAILED_TESTS=""

append_line() {
  local current="$1"
  local value="$2"
  if [[ -z "${current}" ]]; then
    printf '%s' "${value}"
    return
  fi
  printf '%s\n%s' "${current}" "${value}"
}

print_multiline_entries() {
  local entries="$1"
  local prefix="$2"
  local line

  if [[ -z "${entries}" ]]; then
    return
  fi

  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    echo "${prefix}${line}"
  done <<EOF
${entries}
EOF
}

remove_workspace_dirs() {
  local line

  if [[ -z "${WORKSPACES}" ]]; then
    return
  fi

  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    rm -rf "${line}"
  done <<EOF
${WORKSPACES}
EOF
}

cleanup() {
  if [[ "${KEEP_WORKSPACES}" == "1" || "${FAIL_COUNT}" -gt 0 ]]; then
    echo
    echo "Workspace retention enabled."
    print_multiline_entries "${WORKSPACES}" "  "
    return
  fi

  remove_workspace_dirs
}

trap cleanup EXIT

fail() {
  local name="$1"
  local reason="$2"
  echo "[FAIL] ${name}: ${reason}"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  FAILED_TESTS="$(append_line "${FAILED_TESTS}" "${name}")"
}

pass() {
  local name="$1"
  local workspace="$2"
  echo "[PASS] ${name}: ${workspace}"
  PASS_COUNT=$((PASS_COUNT + 1))
}

skip() {
  local name="$1"
  local reason="$2"
  echo "[SKIP] ${name}: ${reason}"
  SKIP_COUNT=$((SKIP_COUNT + 1))
}

make_workspace() {
  local workspace
  workspace="$(mktemp -d /tmp/maike-smoke.XXXXXX)"
  WORKSPACES="$(append_line "${WORKSPACES}" "${workspace}")"
  echo "${workspace}"
}

provider_has_key() {
  case "$1" in
    anthropic)
      [[ -n "${ANTHROPIC_API_KEY:-}" ]]
      ;;
    gemini)
      [[ -n "${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}" ]]
      ;;
    openai)
      [[ -n "${OPENAI_API_KEY:-}" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

show_log_tail() {
  local logfile="$1"
  if [[ -f "${logfile}" ]]; then
    echo "--- log tail: ${logfile} ---"
    tail -n 40 "${logfile}"
    echo "--- end log tail ---"
  fi
}

verify_generated_workspace() {
  local workspace="$1"
  local source_file_count

  [[ -s "${workspace}/README.md" ]] || return 1
  [[ -f "${workspace}/.maike/session.db" ]] || return 1

  source_file_count="$(
    find "${workspace}" -maxdepth 1 -type f \
      \( -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.tsx' -o -name '*.jsx' -o -name '*.go' -o -name '*.rs' \) \
      | wc -l | tr -d ' '
  )"
  [[ "${source_file_count}" -ge 1 ]] || return 1

  grep -Fq "Hello World" "${workspace}/README.md" || return 1
}

verify_history_and_cost_commands() {
  local workspace="$1"
  "${PYTHON_BIN}" - "${workspace}" <<'PY'
import sys
from pathlib import Path

from maike.cli import cost_command, history_command

workspace = Path(sys.argv[1])
history = history_command(workspace=workspace, limit=5)
cost = cost_command(workspace=workspace, last=5)

assert history, "history_command returned no sessions"
assert cost, "cost_command returned no sessions"
assert any(item["workspace"] == str(workspace) for item in history), history
assert any(item["workspace"] == str(workspace) for item in cost), cost
PY
}

verify_phase2_db_invariants() {
  local workspace="$1"
  "${PYTHON_BIN}" - "${workspace}" <<'PY'
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
db_path = workspace / ".maike" / "session.db"
required_stage_outputs = [
    "spec.md",
    "acceptance-contract.md",
    "plan.md",
    "architecture.md",
    "code-summary.md",
    "test-results.md",
    "review.md",
    "acceptance-results.md",
]
required_stages = [
    "requirements",
    "planning",
    "architecture",
    "coding",
    "testing",
    "review",
    "acceptance",
]
stage_outputs_with_deps = [
    "plan.md",
    "architecture.md",
    "code-summary.md",
    "test-results.md",
    "review.md",
    "acceptance-results.md",
]
source_suffixes = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs"}

conn = sqlite3.connect(db_path)
try:
    git_check = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    assert git_check.stdout.strip() == "true", git_check.stdout

    row = conn.execute(
        """
        SELECT id, status
        FROM sessions
        ORDER BY updated_at DESC
        LIMIT 1
        """
    ).fetchone()
    assert row is not None, "no session found for workspace"
    session_id, status = row
    assert status == "completed", status

    stage_rows = conn.execute(
        """
        SELECT logical_name, content_hash, depends_on
        FROM artifacts
        WHERE session_id = ? AND kind = 'stage' AND invalidated = 0
        """,
        (session_id,),
    ).fetchall()
    stage_map = {logical_name: (content_hash, depends_on) for logical_name, content_hash, depends_on in stage_rows}
    for logical_name in required_stage_outputs:
        assert logical_name in stage_map, f"missing stage artifact: {logical_name}"
        assert stage_map[logical_name][0], f"empty content hash for {logical_name}"
    for logical_name in stage_outputs_with_deps:
        depends_on = json.loads(stage_map[logical_name][1])
        assert depends_on, f"missing dependency edges for {logical_name}"

    file_rows = conn.execute(
        """
        SELECT logical_name, path, content_hash, depends_on
        FROM artifacts
        WHERE session_id = ? AND kind = 'file' AND invalidated = 0
        """,
        (session_id,),
    ).fetchall()
    source_files = [
        row
        for row in file_rows
        if row[1]
        and "/" not in row[1]
        and Path(row[1]).suffix.lower() in source_suffixes
    ]
    assert source_files, "missing generated top-level source file artifact"
    for logical_name, path, content_hash, depends_on_text in source_files:
        assert logical_name, "missing logical_name on file artifact"
        assert path, "missing path on file artifact"
        assert content_hash, f"missing content hash on {path}"
        depends_on = json.loads(depends_on_text)
        assert depends_on, f"missing dependency edges on {path}"

    checkpoint_rows = conn.execute(
        """
        SELECT label, step
        FROM checkpoints
        WHERE session_id = ?
        ORDER BY created_at ASC
        """,
        (session_id,),
    ).fetchall()
    checkpoint_pairs = {(label, step) for label, step in checkpoint_rows}
    for stage_name in required_stages:
        assert (f"pre-{stage_name}", stage_name) in checkpoint_pairs, f"missing checkpoint for {stage_name}"

    agent_run_stages = {
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT stage_name
            FROM agent_runs
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()
    }
    step_result_stages = {
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT stage_name
            FROM step_results
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()
    }
    assert set(required_stages).issubset(agent_run_stages), agent_run_stages
    assert set(required_stages).issubset(step_result_stages), step_result_stages
finally:
    conn.close()
PY
}

verify_phase3_runtime_and_safety_probes() {
  local workspace="$1"
  "${PYTHON_BIN}" - "${workspace}" <<'PY'
import asyncio
import shutil
import sys
from pathlib import Path

from maike.agents.core import AgentCore
from maike.atoms.context import AgentContext
from maike.atoms.llm import LLMResult, StopReason, TokenUsage
from maike.atoms.tool import RiskLevel
from maike.constants import DEFAULT_MODEL
from maike.memory.working import WorkingMemory
from maike.observability.tracer import Tracer
from maike.runtime.local import LocalRuntime
from maike.safety.approval import ApprovalGate
from maike.safety.hooks import SafetyLayer
from maike.safety.rules import Decision
from maike.tools import register_default_tools
from maike.tools.registry import ToolRegistry


class FakeGateway:
    def __init__(self, responses):
        self.responses = list(responses)

    async def call(self, **kwargs):
        del kwargs
        return self.responses.pop(0)


def make_text_result(text: str) -> LLMResult:
    return LLMResult(
        provider="anthropic",
        content=text,
        tool_calls=[],
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        latency_ms=1,
        model=DEFAULT_MODEL,
    )


def make_tool_result(tool_name: str, tool_input: dict) -> LLMResult:
    return LLMResult(
        provider="anthropic",
        content=None,
        tool_calls=[{"id": "tool-1", "name": tool_name, "input": tool_input}],
        stop_reason=StopReason.TOOL_USE,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        latency_ms=1,
        model=DEFAULT_MODEL,
    )


async def main(workspace: Path) -> None:
    runtime_probe = workspace / ".phase3_runtime_probe"
    agent_probe = workspace / ".phase3_agent_probe"
    shutil.rmtree(runtime_probe, ignore_errors=True)
    shutil.rmtree(agent_probe, ignore_errors=True)

    runtime = LocalRuntime(runtime_probe)
    init_result = await runtime.init_git_repo()
    assert init_result.success is True, init_result
    await runtime.write_file("tracked.txt", "one\n")
    checkpoint = await runtime.checkpoint("pre-coding", "coding")
    await runtime.write_file("tracked.txt", "two\n")
    await runtime.write_file("scratch/new.txt", "temp\n")
    await runtime.restore(checkpoint)
    assert (runtime_probe / "tracked.txt").read_text(encoding="utf-8") == "one\n"
    assert not (runtime_probe / "scratch").exists()

    safety = SafetyLayer(workspace)
    ctx_without_checkpoint = AgentContext(
        role="coder",
        task="x",
        stage_name="coding",
        tool_profile="coding",
        metadata={"session_id": "session-1", "mutated_paths": ["first.py"]},
        model=DEFAULT_MODEL,
    )
    ctx_with_checkpoint = AgentContext(
        role="coder",
        task="x",
        stage_name="coding",
        tool_profile="coding",
        metadata={
            "session_id": "session-1",
            "stage_checkpoint_sha": "abcdef123456",
            "mutated_paths": ["first.py"],
        },
        model=DEFAULT_MODEL,
    )

    delete_block = safety.assess(
        "delete_file",
        {"path": "probe.txt"},
        risk_level=RiskLevel.DESTRUCTIVE,
        ctx=ctx_without_checkpoint,
    )
    assert delete_block.decision == Decision.BLOCK
    multi_write_block = safety.assess(
        "write_file",
        {"path": "second.py", "content": "x"},
        risk_level=RiskLevel.WRITE,
        ctx=ctx_without_checkpoint,
    )
    assert multi_write_block.decision == Decision.BLOCK
    mutating_bash_block = safety.assess(
        "execute_bash",
        {"cmd": "git add -A"},
        risk_level=RiskLevel.EXECUTE,
        ctx=ctx_without_checkpoint,
    )
    assert mutating_bash_block.decision == Decision.BLOCK
    read_only_bash = safety.assess(
        "execute_bash",
        {"cmd": "pytest -q"},
        risk_level=RiskLevel.EXECUTE,
        ctx=ctx_without_checkpoint,
    )
    assert read_only_bash.decision == Decision.REQUIRE_APPROVAL
    delete_allowed = safety.assess(
        "delete_file",
        {"path": "probe.txt"},
        risk_level=RiskLevel.DESTRUCTIVE,
        ctx=ctx_with_checkpoint,
    )
    assert delete_allowed.decision == Decision.REQUIRE_APPROVAL
    assert "checkpoint=abcdef1" in delete_allowed.approval_prompt
    multi_write_allowed = safety.assess(
        "write_file",
        {"path": "second.py", "content": "x"},
        risk_level=RiskLevel.WRITE,
        ctx=ctx_with_checkpoint,
    )
    assert multi_write_allowed.decision == Decision.ALLOW

    agent_probe.mkdir(parents=True, exist_ok=True)
    runtime = LocalRuntime(agent_probe)
    registry = ToolRegistry()
    register_default_tools(registry, runtime)

    blocked_agent = AgentCore(
        llm_gateway=FakeGateway(
            [
                make_tool_result("write_file", {"path": "blocked.py", "content": "x"}),
                make_text_result("done"),
            ]
        ),
        tool_registry=registry,
        runtime=runtime,
        safety_layer=SafetyLayer(agent_probe),
        working_memory=WorkingMemory(),
        tracer=Tracer(),
        approval_gate=ApprovalGate(auto_approve=True),
    )
    blocked_result = await blocked_agent.run(
        AgentContext(
            role="reflection",
            task="review",
            stage_name="review",
            tool_profile="reflection_readonly",
            metadata={"session_id": "session-1", "mutated_paths": []},
            model=DEFAULT_MODEL,
        ),
        [{"role": "user", "content": "review"}],
    )
    blocked_tool = blocked_result.messages[-2]["content"][0]
    assert blocked_tool["is_error"] is True
    assert "Tool not allowed for tool profile 'reflection_readonly'" in blocked_tool["content"]

    (agent_probe / "app.py").write_text("print('hello')\n", encoding="utf-8")
    grep_agent = AgentCore(
        llm_gateway=FakeGateway(
            [
                make_tool_result("grep_codebase", {"pattern": "nomatch", "path": "."}),
                make_text_result("done"),
            ]
        ),
        tool_registry=registry,
        runtime=runtime,
        safety_layer=SafetyLayer(agent_probe),
        working_memory=WorkingMemory(),
        tracer=Tracer(),
        approval_gate=ApprovalGate(auto_approve=True),
    )
    grep_result = await grep_agent.run(
        AgentContext(
            role="coder",
            task="search",
            stage_name="coding",
            tool_profile="coding",
            metadata={"session_id": "session-1", "mutated_paths": []},
            model=DEFAULT_MODEL,
        ),
        [{"role": "user", "content": "search"}],
    )
    grep_tool = grep_result.messages[-2]["content"][0]
    assert grep_tool["is_error"] is False
    assert grep_tool["content"] == ""


asyncio.run(main(Path(sys.argv[1])))
PY
}

run_success_smoke() {
  local name="$1"
  local provider="$2"
  local workspace logfile
  workspace="$(make_workspace)"
  logfile="${workspace}/$(echo "${name}" | tr ' ' '_' | tr '[:upper:]' '[:lower:]').log"

  local -a cmd=("${MAIKE_BIN}" run "${SMOKE_TASK}" -w "${workspace}" --language "${SMOKE_LANGUAGE}" --yes)
  if [[ "${provider}" != "anthropic-default" ]]; then
    cmd+=(--provider "${provider}")
  fi

  if "${cmd[@]}" >"${logfile}" 2>&1; then
    if ! verify_generated_workspace "${workspace}"; then
      fail "${name}" "expected README.md, a generated source file, and session.db in ${workspace}"
      show_log_tail "${logfile}"
      return
    fi
    if ! verify_history_and_cost_commands "${workspace}" >>"${logfile}" 2>&1; then
      fail "${name}" "history/cost checks failed"
      show_log_tail "${logfile}"
      return
    fi
    if ! verify_phase2_db_invariants "${workspace}" >>"${logfile}" 2>&1; then
      fail "${name}" "phase 2 DB invariant checks failed"
      show_log_tail "${logfile}"
      return
    fi
    if ! verify_phase3_runtime_and_safety_probes "${workspace}" >>"${logfile}" 2>&1; then
      fail "${name}" "phase 3 runtime/safety probes failed"
      show_log_tail "${logfile}"
      return
    fi
    pass "${name}" "${workspace}"
    return
  fi

  fail "${name}" "command exited non-zero"
  show_log_tail "${logfile}"
}

run_budget_failure_smoke() {
  local provider="$1"
  local workspace logfile name
  name="Budget enforcement (${provider})"
  workspace="$(make_workspace)"
  logfile="${workspace}/budget_enforcement_${provider}.log"

  local -a cmd=(
    "${MAIKE_BIN}" run "${SMOKE_TASK}"
    -w "${workspace}"
    --language "${SMOKE_LANGUAGE}"
    --budget "${SMOKE_BUDGET_USD}"
    --yes
  )
  if [[ "${provider}" != "anthropic-default" ]]; then
    cmd+=(--provider "${provider}")
  fi

  if "${cmd[@]}" >"${logfile}" 2>&1; then
    fail "${name}" "expected a budget failure but command succeeded"
    show_log_tail "${logfile}"
    return
  fi

  if grep -Fq "Session cost budget exceeded" "${logfile}"; then
    pass "${name}" "${workspace}"
    return
  fi

  fail "${name}" "expected budget error message"
  show_log_tail "${logfile}"
}

if [[ ! -x "${MAIKE_BIN}" ]]; then
  echo "maike binary not found or not executable: ${MAIKE_BIN}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python binary not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Running mAIke smoke suite from ${ROOT_DIR}"

if [[ "${MAIKE_SMOKE_SKIP_ANTHROPIC}" == "1" ]]; then
  skip "Anthropic default model path" "MAIKE_SMOKE_SKIP_ANTHROPIC=1"
elif provider_has_key anthropic; then
  run_success_smoke "Anthropic default model path" "anthropic-default"
else
  skip "Anthropic default model path" "ANTHROPIC_API_KEY is not set"
fi

ALT_PROVIDER_COUNT=0
if provider_has_key gemini; then
  ALT_PROVIDER_COUNT=$((ALT_PROVIDER_COUNT + 1))
  run_success_smoke "Provider default model path (gemini)" "gemini"
fi
if provider_has_key openai; then
  ALT_PROVIDER_COUNT=$((ALT_PROVIDER_COUNT + 1))
  if [[ "${MAIKE_SMOKE_ALL_PROVIDERS}" == "1" || "${ALT_PROVIDER_COUNT}" -eq 1 ]]; then
    run_success_smoke "Provider default model path (openai)" "openai"
  fi
fi

if [[ "${ALT_PROVIDER_COUNT}" -eq 0 ]]; then
  skip "Alternate provider default-model path" "No Gemini or OpenAI API key is set"
fi

BUDGET_PROVIDER=""
if provider_has_key gemini; then
  BUDGET_PROVIDER="gemini"
elif provider_has_key openai; then
  BUDGET_PROVIDER="openai"
elif [[ "${MAIKE_SMOKE_SKIP_ANTHROPIC}" != "1" ]] && provider_has_key anthropic; then
  BUDGET_PROVIDER="anthropic-default"
fi

if [[ -n "${BUDGET_PROVIDER}" ]]; then
  run_budget_failure_smoke "${BUDGET_PROVIDER}"
else
  skip "Budget enforcement" "No provider API key is set"
fi

echo
echo "Smoke summary: ${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${SKIP_COUNT} skipped"

if [[ "${PASS_COUNT}" -eq 0 ]]; then
  echo "No smoke tests ran successfully."
  exit 1
fi

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "Failures:"
  print_multiline_entries "${FAILED_TESTS}" "  "
  exit 1
fi
