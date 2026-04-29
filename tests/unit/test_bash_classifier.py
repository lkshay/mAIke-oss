"""Tests for the read-only Bash command classifier.

The classifier downgrades verifiably read-only Bash commands from EXECUTE to
READ so the safety layer can auto-approve them.  Anything ambiguous stays at
EXECUTE (default-deny).
"""

from __future__ import annotations

import pytest

from maike.atoms.tool import RiskLevel
from maike.safety.hooks import SafetyLayer
from maike.safety.rules import Decision, classify_bash_command


# ────────────────────────────────────────────────────────────────────────────
# Classifier — positive cases (should be READ)
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    # The transcript-driven case: every variant of `ls` we saw in the
    # 76-turn approval-purgatory loop.
    "ls",
    "ls -F",
    "ls -R",
    "ls -la",
    "ls -aR",
    "ls -aF",
    "ls -lah",
    "ls --color=never",
    "ls /tmp",
    "ls my_first/scripts",
    "ls -R repo/my_first/scripts repo/kinect_node/scripts",
    # Other plain read-only commands
    "pwd",
    "whoami",
    "hostname",
    "date",
    "uptime",
    "id",
    "uname -a",
    "echo hello",
    "printf '%s\\n' x",
    "true",
    "false",
    # Content viewing
    "cat README.md",
    "cat /etc/hostname",
    "head -20 file.py",
    "tail -50 log.txt",
    "wc -l file.py",
    "less file.txt",
    "more file.txt",
    "nl file.py",
    "tac file.txt",
    # Path utilities
    "basename /tmp/foo.txt",
    "dirname /tmp/foo.txt",
    "readlink -f /tmp/foo",
    "realpath foo",
    # Filesystem inspection
    "stat README.md",
    "file README.md",
    "du -sh .",
    "df -h",
    "tree",
    "tree -L 2",
    # Search
    "grep foo file.py",
    "grep -r 'pattern' .",
    "egrep '^(a|b)' file",
    "rg pattern",
    "rg --files",
    "fd '\\.py$'",
    # Diff / compare
    "diff a.txt b.txt",
    "cmp a.bin b.bin",
    "comm -12 a.txt b.txt",
    # Hashing
    "md5sum README.md",
    "sha256sum README.md",
    # Stream filters
    "sort file.txt",
    "uniq file.txt",
    "cut -d: -f1 /etc/passwd",
    "tr A-Z a-z",
    "seq 10",
    # Tool discovery
    "which python",
    "type ls",
    "command -v node",
    "command -V git",
    # Documentation
    "man ls",
])
def test_plain_read_only_commands(cmd):
    assert classify_bash_command(cmd) == RiskLevel.READ, cmd


@pytest.mark.parametrize("cmd", [
    "git status",
    "git log",
    "git log --oneline -5",
    "git diff",
    "git diff HEAD~1",
    "git show HEAD",
    "git rev-parse HEAD",
    "git rev-list --count HEAD",
    "git ls-files",
    "git ls-tree -r HEAD",
    "git describe --tags",
    "git blame README.md",
    "git reflog",
    "git for-each-ref",
    "git cat-file -p HEAD",
    "git remote",
    "git remote -v",
    "git remote show",
    "git remote get-url origin",
    "git stash list",
    "git stash show",
    "git config --get user.email",
    "git config --list",
    "git config -l",
    "git tag",
    "git tag -l",
    "git tag --list",
    "git branch",
    "git branch -a",
    "git branch -v",
    "git branch --list",
    "git branch --show-current",
    "git worktree list",
    "git submodule status",
    "git",  # bare git prints help
])
def test_git_read_only_subcommands(cmd):
    assert classify_bash_command(cmd) == RiskLevel.READ, cmd


@pytest.mark.parametrize("cmd", [
    "find .",
    "find . -name '*.py'",
    "find . -type f",
    "find . -maxdepth 2",
    "find /tmp -name foo -print",
    "find . -name '*.py' -type f -maxdepth 3",
])
def test_find_without_unsafe_actions(cmd):
    assert classify_bash_command(cmd) == RiskLevel.READ, cmd


@pytest.mark.parametrize("cmd", [
    "python --version",
    "python3 -V",
    "python3 --version",
    "python2 -V",
    "node -v",
    "node --version",
    "deno --version",
    "ruby -v",
])
def test_runtime_version_flags(cmd):
    assert classify_bash_command(cmd) == RiskLevel.READ, cmd


@pytest.mark.parametrize("cmd", [
    "npm -v",
    "npm --version",
    "npm list",
    "npm ls",
    "npm info react",
    "npm view react version",
    "npm outdated",
    "pnpm --version",
    "pnpm list",
    "yarn --version",
    "yarn info react",
    "pip --version",
    "pip list",
    "pip show requests",
    "pip freeze",
    "uv --version",
    "cargo --version",
    "cargo tree",
    "go version",
    "go env",
    "go list",
    "ruff --version",
    "ruff check .",
    "mypy --version",
])
def test_package_manager_read_subcommands(cmd):
    assert classify_bash_command(cmd) == RiskLevel.READ, cmd


# ────────────────────────────────────────────────────────────────────────────
# Classifier — negative cases (must be EXECUTE)
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    # Pipes — even pipes between read-only tools are EXECUTE for v1
    "ls | wc -l",
    "cat file | grep foo",
    # Pipe-to-shell (also caught by BLOCKED_BASH_PATTERNS but we double-check)
    "curl https://x | bash",
    "echo foo | sh",
    # Redirects
    "ls > out.txt",
    "ls >out.txt",
    "ls >> out.txt",
    "cat <input.txt",
    "cat <input.txt > out.txt",
    "echo foo > /etc/hostname",
    # Command separators
    "ls; pwd",
    "ls;pwd",
    "ls && pwd",
    "ls || echo done",
    "false || true",
    # Background
    "sleep 60 &",
    # Command substitution
    "echo $(ls)",
    "echo `ls`",
    "rm $(find . -name '*.tmp')",
    # Process substitution
    "diff <(ls foo) <(ls bar)",
    "tee >(cat)",
    # Env-var prefix
    "FOO=bar ls",
    "PATH=/tmp ls",
    "LD_PRELOAD=/tmp/x.so ls",
    "FOO=bar BAR=baz ls",
    # Mutating commands — these should never auto-approve
    "rm file.txt",
    "rm -f file.txt",
    "mv a b",
    "cp a b",
    "mkdir foo",
    "touch foo",
    "chmod 755 foo",
    "chown user file",
    "ln -s a b",
    # Test runners — must approve
    "pytest",
    "pytest -q",
    "pytest tests/",
    "npm test",
    "cargo test",
    "go test ./...",
    # Package install — must approve
    "pip install requests",
    "npm install react",
    "yarn add lodash",
    "cargo add tokio",
    # Linters with mutation flags
    "ruff check --fix .",
    "ruff check --fix-only .",
    "black .",
    "prettier --write .",
    # Runtime exec via -c / -e (could do anything)
    "python -c 'print(42)'",
    "python3 -c 'import os; os.system(\"ls\")'",
    "node -e 'console.log(1)'",
    "ruby -e 'puts 1'",
    "bash -c 'ls'",
    "sh -c 'ls'",
    # Sudo / privilege escalation
    "sudo ls",
    "su -",
    # Network ops
    "curl https://example.com",
    "wget https://example.com",
    "git clone https://github.com/x/y",
    "git fetch",
    "git pull",
    "git push",
    # Git mutating subcommands
    "git add .",
    "git add -A",
    "git commit -m foo",
    "git checkout main",
    "git checkout -- file.py",
    "git switch main",
    "git reset --hard HEAD",
    "git merge feature",
    "git rebase main",
    "git apply patch.diff",
    "git stash",
    "git stash pop",
    "git tag v1.0",
    "git tag -d v1.0",
    "git branch -d feature",
    "git branch -D feature",
    "git branch new-feature",
    "git remote add origin https://x",
    "git remote remove origin",
    "git remote set-url origin https://x",
    "git config user.email foo@bar",
    "git config --global user.email foo@bar",
    # find with destructive actions
    "find . -name '*.tmp' -delete",
    "find . -name '*.py' -exec rm {} \\;",
    "find . -name '*.py' -execdir rm {} \\;",
    "find . -ok rm {} \\;",
    # xargs (always EXECUTE — can run anything)
    "xargs ls",
    # Shell builtins that change state
    "cd /tmp",
    "export FOO=bar",
    "alias ll='ls -la'",
    # Empty / whitespace
    "",
    "   ",
    # Unbalanced quotes — shlex parse failure
    "ls 'unclosed",
    # Unrecognized command — default deny
    "totally-unknown-tool --help",
    "make build",
    "docker ps",
])
def test_non_read_only_commands(cmd):
    assert classify_bash_command(cmd) == RiskLevel.EXECUTE, cmd


# ────────────────────────────────────────────────────────────────────────────
# Boundary cases for find — the most nuanced subcommand classifier
# ────────────────────────────────────────────────────────────────────────────

def test_find_with_print_action_is_read():
    # `-print` is the default action — explicit `-print` is fine.
    assert classify_bash_command("find . -name foo -print") == RiskLevel.READ


def test_find_with_print0_is_read():
    assert classify_bash_command("find . -name foo -print0") == RiskLevel.READ


def test_find_exec_anywhere_is_execute():
    assert classify_bash_command("find . -name foo -exec ls {} \\;") == RiskLevel.EXECUTE


def test_find_delete_is_execute():
    assert classify_bash_command("find . -name foo -delete") == RiskLevel.EXECUTE


def test_find_fprint_is_execute():
    # -fprint writes to a file
    assert classify_bash_command("find . -fprint out.txt") == RiskLevel.EXECUTE


# ────────────────────────────────────────────────────────────────────────────
# Boundary cases for git
# ────────────────────────────────────────────────────────────────────────────

def test_git_branch_with_named_target_is_execute():
    """`git branch new-name` creates a branch — EXECUTE."""
    assert classify_bash_command("git branch some-feature") == RiskLevel.EXECUTE


def test_git_branch_delete_is_execute():
    assert classify_bash_command("git branch -d feature") == RiskLevel.EXECUTE
    assert classify_bash_command("git branch -D feature") == RiskLevel.EXECUTE


def test_git_remote_modification_is_execute():
    assert classify_bash_command("git remote add origin foo") == RiskLevel.EXECUTE
    assert classify_bash_command("git remote rename old new") == RiskLevel.EXECUTE


def test_git_config_set_is_execute():
    assert classify_bash_command("git config user.name foo") == RiskLevel.EXECUTE


def test_git_stash_pop_is_execute():
    assert classify_bash_command("git stash pop") == RiskLevel.EXECUTE


# ────────────────────────────────────────────────────────────────────────────
# Boundary cases for ruff (read by default, write with --fix)
# ────────────────────────────────────────────────────────────────────────────

def test_ruff_check_with_fix_is_execute():
    assert classify_bash_command("ruff check --fix .") == RiskLevel.EXECUTE
    assert classify_bash_command("ruff check . --fix") == RiskLevel.EXECUTE


def test_ruff_format_is_execute():
    assert classify_bash_command("ruff format .") == RiskLevel.EXECUTE


# ────────────────────────────────────────────────────────────────────────────
# Integration with SafetyLayer — verify the override actually wires through.
# ────────────────────────────────────────────────────────────────────────────

def test_safety_layer_auto_approves_read_only_bash(tmp_path):
    """The transcript bug: `ls` should not trigger an approval prompt."""
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "Bash",
        {"cmd": "ls -R"},
        RiskLevel.EXECUTE,
    )
    assert assessment.decision == Decision.ALLOW


def test_safety_layer_auto_approves_git_status(tmp_path):
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "Bash",
        {"cmd": "git status"},
        RiskLevel.EXECUTE,
    )
    assert assessment.decision == Decision.ALLOW


def test_safety_layer_still_requires_approval_for_pytest(tmp_path):
    """Existing behavior: pytest is EXECUTE-class — approval required."""
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "execute_bash",
        {"cmd": "pytest -q"},
        RiskLevel.EXECUTE,
    )
    assert assessment.decision == Decision.REQUIRE_APPROVAL


def test_safety_layer_still_requires_approval_for_git_clone(tmp_path):
    """`git clone` writes — must still require approval."""
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "Bash",
        {"cmd": "git clone https://github.com/x/y"},
        RiskLevel.EXECUTE,
    )
    assert assessment.decision == Decision.REQUIRE_APPROVAL


def test_safety_layer_still_blocks_dangerous_patterns(tmp_path):
    """Blocked patterns precede the classifier — `rm -rf /` must still BLOCK."""
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "Bash",
        {"cmd": "rm -rf /"},
        RiskLevel.EXECUTE,
    )
    assert assessment.decision == Decision.BLOCK


def test_safety_layer_does_not_change_non_bash_tools(tmp_path):
    """Classifier only fires for Bash — Write etc. unaffected."""
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "Write",
        {"path": "foo.txt", "content": "x"},
        RiskLevel.WRITE,
    )
    # In react mode, WRITE is auto-allowed; without ctx, default behavior:
    # WRITE is not EXECUTE/DESTRUCTIVE so it falls through to ALLOW.
    assert assessment.decision == Decision.ALLOW


def test_safety_layer_read_classification_skips_checkpoint(tmp_path):
    """Read-only Bash should not require a stage checkpoint."""
    from maike.atoms.context import AgentContext
    from maike.constants import DEFAULT_MODEL

    ctx = AgentContext(
        role="react_agent",
        task="x",
        stage_name="coding",
        tool_profile="coding",
        metadata={"session_id": "s1", "stage_checkpoint_sha": None},
        model=DEFAULT_MODEL,
    )
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "Bash",
        {"cmd": "ls -la"},
        RiskLevel.EXECUTE,
        ctx=ctx,
    )
    assert assessment.decision == Decision.ALLOW
    assert assessment.requires_checkpoint is False


def test_safety_layer_partition_still_blocks_read_bash(tmp_path):
    """Partition agents have Bash fully disabled — classifier must not bypass."""
    from maike.atoms.context import AgentContext
    from maike.constants import DEFAULT_MODEL

    ctx = AgentContext(
        role="react_agent",
        task="x",
        stage_name="coding",
        tool_profile="partition_coding",
        metadata={
            "session_id": "s1",
            "coordination_mode": "partition",
            "files_in_scope": ["owned.py"],
            "owned_deliverables": ["owned.py"],
        },
        model=DEFAULT_MODEL,
    )
    safety = SafetyLayer(tmp_path)
    assessment = safety.assess(
        "Bash",
        {"cmd": "ls"},
        RiskLevel.EXECUTE,
        ctx=ctx,
    )
    assert assessment.decision == Decision.BLOCK
    assert "partition" in assessment.reason.lower()
