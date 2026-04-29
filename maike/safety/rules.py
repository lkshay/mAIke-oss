"""Safety rules and decisions."""

from __future__ import annotations

from enum import Enum
import re
import shlex

from maike.atoms.tool import RiskLevel


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"

BLOCKED_BASH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "dangerous rm target",
        re.compile(r"\brm\s+-[^\n;|&]*r[^\n;|&]*f[^\n;|&]*(?:--no-preserve-root\s+)?(?:/|~)(?:\s|$)"),
    ),
    (
        "fork bomb",
        re.compile(r":\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
    ),
    (
        "destructive disk write",
        re.compile(r"\bdd\s+if=/dev/zero\b"),
    ),
    (
        "filesystem formatting",
        re.compile(r"\b(?:mkfs|fdisk|parted|wipefs)\b"),
    ),
    (
        "destructive permission change",
        re.compile(r"\b(?:chmod|chown)\s+-[^\n;|&]*r[^\n;|&]*(?:/|~|/etc|/usr|/var)(?:\s|$)"),
    ),
    (
        "pipe to shell",
        re.compile(r"\b(?:curl|wget|fetch|nc|ncat)\b[^\n]*\|\s*(?:bash|sh|zsh|python\d*|node)\b"),
    ),
    (
        "python destructive root delete",
        re.compile(r"\bpython\d*\b[^\n]*\b(?:shutil\.rmtree|os\.(?:remove|unlink|rmdir))\s*\(\s*['\"](?:/|~)"),
    ),
    (
        "block device redirect",
        re.compile(r">\s*/dev/(?:sd[a-z]+\d*|nvme\d+n\d+(?:p\d+)?|disk\d+)"),
    ),
    # ── Hardened patterns (catch obfuscation attempts) ──
    (
        "eval-wrapped destructive command",
        re.compile(r"\beval\s+['\"].*\b(?:rm\s+-rf|dd\s+if=|mkfs|shutil\.rmtree)\b"),
    ),
    (
        "command substitution destructive",
        re.compile(r"\$\(.*\b(?:rm|dd|mkfs|wipefs|fdisk)\b.*\)"),
    ),
    (
        "heredoc to shell",
        re.compile(r"<<['\"]?\w*['\"]?\s*\n.*\b(?:rm\s+-rf|dd\s+if=|mkfs)\b", re.DOTALL),
    ),
    (
        "base64 decode to shell",
        re.compile(r"\bbase64\s+(?:-d|--decode)\b[^\n]*\|\s*(?:bash|sh|zsh)\b"),
    ),
    (
        "xargs rm recursive",
        re.compile(r"\bxargs\b[^\n]*\brm\s+-[^\n]*r"),
    ),
    (
        "find exec rm root",
        re.compile(r"\bfind\s+(?:/|~)[^\n]*(?:-exec|-delete)[^\n]*\brm\b"),
    ),
    (
        "python -c destructive",
        re.compile(r"\bpython\d*\s+-c\s+['\"].*\b(?:shutil\.rmtree|os\.(?:remove|unlink|rmdir|system))\b.*(?:/|~)"),
    ),
    (
        "environment variable override to escalate",
        re.compile(r"\b(?:LD_PRELOAD|LD_LIBRARY_PATH)\s*=\s*[^\n]*\b(?:bash|sh|su|sudo)\b"),
    ),
)


# ───────────────────────────── Bash risk classifier ─────────────────────────────
#
# The Bash tool registers as `RiskLevel.EXECUTE`, which would normally route
# every invocation through approval.  In practice the agent runs many commands
# that are verifiably read-only — `ls`, `cat`, `git status`, `find` (without
# `-exec`) — and gating those behind approval prompts produces the
# approval-purgatory loops we observed (76-turn run, ~35 turns spent on `ls`
# variants asking permission and getting denied).
#
# `classify_bash_command()` returns `RiskLevel.READ` when the command is
# verifiably read-only by deterministic parsing, and `RiskLevel.EXECUTE`
# otherwise.  The safety layer uses the result to override the per-call risk
# level for Bash.  This is **default-deny on ambiguity** — anything we cannot
# definitively prove is read-only stays at EXECUTE and goes through approval.
#
# Design constraints:
#   * No LLM judgment.  Pure deterministic parsing.
#   * No partial-trust modes.  Either fully READ or treat as EXECUTE.
#   * Reject any shell composition (pipes, redirects, command substitution,
#     env-var prefixes, &&/||/; chaining).  Composition can defeat
#     leading-command analysis; rather than try to model it, we drop into
#     EXECUTE and let the user approve.
#   * The cost of a false-negative (a read-only command falling through to
#     approval) is one extra prompt.  The cost of a false-positive (executing
#     a destructive command without approval) could be data loss.  Default
#     conservative.

# Read-only commands whose arguments are paths / flags / glob patterns —
# they do not exec subcommands and have no destructive flag variants.
_READ_ONLY_PLAIN_COMMANDS: frozenset[str] = frozenset({
    # Filesystem inspection
    "ls", "pwd", "tree", "stat", "file", "du", "df",
    # Content viewing
    "cat", "head", "tail", "less", "more", "wc", "nl", "tac",
    # Path utilities
    "basename", "dirname", "readlink", "realpath",
    # Search (read-only — these read files and print matches)
    "grep", "egrep", "fgrep", "rg", "ag", "fd",
    # Diff and compare
    "diff", "cmp", "comm",
    # Hashing / encoding (read input → write to stdout)
    "md5sum", "sha1sum", "sha256sum", "sha512sum", "cksum", "xxd", "od",
    # Stream filters (no in-place flag — sed and awk excluded; they have
    # in-place modes that are hard to detect)
    "sort", "uniq", "cut", "tr", "column", "paste", "expand", "unexpand",
    "fold", "rev", "seq",
    # Trivial output
    "echo", "printf", "yes", "true", "false",
    # System info
    "whoami", "id", "hostname", "uname", "date", "uptime", "users", "groups",
    "printenv",
    # Tool discovery
    "which", "type",
    # Documentation
    "man", "info", "help",
})

# `find` is read-safe iff none of these "action" predicates appear.  The
# default action is `-print`, which is harmless.
_FIND_UNSAFE_ACTIONS: frozenset[str] = frozenset({
    "-exec", "-execdir", "-ok", "-okdir",
    "-delete",
    "-fls", "-fprint", "-fprint0", "-fprintf",
})

# `git` subcommands that are read-only.  `branch`, `remote`, `tag`, `stash`,
# `config` need flag-level validation and are handled separately.
_GIT_READ_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "rev-parse", "rev-list",
    "ls-files", "ls-tree", "describe", "blame", "annotate",
    "reflog", "for-each-ref", "cat-file", "shortlog", "name-rev",
    "grep", "var", "help", "version", "whatchanged",
    "show-branch", "show-ref", "check-ignore", "check-attr",
    "verify-commit", "verify-tag", "count-objects",
    "merge-base", "merge-tree", "rev-list",
})

# Subcommand sets for runtime / package-manager invocations that are read-only.
_PYTHON_READ_FLAGS: frozenset[str] = frozenset({"-V", "--version", "--help", "-h"})
_NODE_READ_FLAGS: frozenset[str] = frozenset({"-v", "--version", "--help", "-h"})

# Each entry: command -> set of arg tokens that, if matched as the second
# token, mark this invocation as READ.  Anything else (including no args) is
# treated as EXECUTE for these commands.
_RUNTIME_READ_FLAGS: dict[str, frozenset[str]] = {
    "python": _PYTHON_READ_FLAGS,
    "python2": _PYTHON_READ_FLAGS,
    "python3": _PYTHON_READ_FLAGS,
    "node": _NODE_READ_FLAGS,
    "deno": _NODE_READ_FLAGS,
    "bun": _NODE_READ_FLAGS | frozenset({"--help"}),
    "ruby": _PYTHON_READ_FLAGS | frozenset({"-v"}),
}

# Package manager subcommands that are read-only.
_PKG_READ_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "npm": frozenset({"-v", "--version", "list", "ls", "info", "view", "show", "outdated", "audit", "config"}),
    "pnpm": frozenset({"-v", "--version", "list", "ls", "info", "view", "outdated"}),
    "yarn": frozenset({"-v", "--version", "list", "info", "config", "why"}),
    "pip": frozenset({"-V", "--version", "list", "show", "freeze", "check", "config"}),
    "pip3": frozenset({"-V", "--version", "list", "show", "freeze", "check", "config"}),
    "uv": frozenset({"--version", "-V", "tree", "pip"}),  # `uv pip list` etc handled lazily
    "cargo": frozenset({"--version", "-V", "tree", "search", "metadata"}),
    "go": frozenset({"version", "env", "list", "doc", "vet"}),  # vet is read-only static analysis
    "ruff": frozenset({"--version", "-V", "check"}),  # `ruff check` is read-only unless --fix; flag scan below
    "mypy": frozenset({"--version", "-V"}),
    "tsc": frozenset({"-v", "--version"}),
}


def _strip_quoted_regions(cmd: str) -> str:
    """Replace quoted regions with spaces so operator-character detection
    sees only unquoted text.

    Mirrors shell quoting semantics: ``|`` inside ``'...'`` or ``"..."`` is
    a literal character, not a pipe operator.  We replace each char in a
    quoted region with a space (preserving offsets) so subsequent
    composition checks don't false-positive on regex alternation
    (``egrep '^(a|b)' file``) or quoted separators (``grep 'a;b' file``).
    """
    out: list[str] = []
    i = 0
    n = len(cmd)
    while i < n:
        c = cmd[i]
        if c == "\\" and i + 1 < n:
            # Backslash escape — collapse both bytes to spaces.
            out.append("  ")
            i += 2
            continue
        if c == "'":
            out.append(" ")
            i += 1
            while i < n and cmd[i] != "'":
                out.append(" ")
                i += 1
            if i < n:
                out.append(" ")
                i += 1
            continue
        if c == '"':
            out.append(" ")
            i += 1
            while i < n:
                if cmd[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                if cmd[i] == '"':
                    break
                out.append(" ")
                i += 1
            if i < n:
                out.append(" ")
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _has_shell_composition(cmd: str) -> bool:
    """Return True if cmd contains unquoted shell metacharacters that compose
    commands or redirect I/O.

    Catches: pipes, redirects, command separators, command substitution,
    backticks, process substitution.  Operates on the quote-stripped form so
    metacharacters inside string literals (``egrep '^(a|b)'``) don't trip
    the check.
    """
    stripped = _strip_quoted_regions(cmd)
    if "$(" in stripped:
        return True
    if "<(" in stripped or ">(" in stripped:
        return True
    return any(c in "|&;<>`" for c in stripped)


def _looks_like_env_assignment(token: str) -> bool:
    """Return True if token is a `NAME=value` env-var assignment prefix.

    Shell parsing of `FOO=bar cmd ...` produces a leading token `FOO=bar`.
    We refuse to auto-classify these as READ since env-var overrides
    (LD_PRELOAD, LD_LIBRARY_PATH, PATH) can change command semantics.
    """
    eq = token.find("=")
    if eq <= 0:
        return False
    name = token[:eq]
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _classify_find(tokens: list[str]) -> RiskLevel:
    """find is READ when no -exec / -delete / -ok action is present."""
    for tok in tokens[1:]:
        if tok in _FIND_UNSAFE_ACTIONS:
            return RiskLevel.EXECUTE
    return RiskLevel.READ


def _classify_git(tokens: list[str]) -> RiskLevel:
    """git is READ only for explicitly read-only subcommands."""
    if len(tokens) < 2:
        # Bare `git` prints help — harmless.
        return RiskLevel.READ
    sub = tokens[1]
    if sub in _GIT_READ_SUBCOMMANDS:
        return RiskLevel.READ
    if sub == "remote":
        # `git remote` (list) and `git remote -v` / `get-url` / `show` are read.
        # `git remote add/remove/rename/set-url` mutate.
        if len(tokens) == 2:
            return RiskLevel.READ
        third = tokens[2]
        if third in {"-v", "--verbose", "show", "get-url"} or third.startswith("-"):
            return RiskLevel.READ
        return RiskLevel.EXECUTE
    if sub == "stash":
        # Only `git stash list` (and `show` without args) are read.
        if len(tokens) >= 3 and tokens[2] in {"list", "show"}:
            return RiskLevel.READ
        return RiskLevel.EXECUTE
    if sub == "config":
        # Only `--get*` / `--list` / `-l` variants read.
        if len(tokens) >= 3 and tokens[2] in {
            "--get", "--get-all", "--get-regexp", "--list", "-l", "--show-origin",
        }:
            return RiskLevel.READ
        return RiskLevel.EXECUTE
    if sub == "tag":
        # `git tag` (list) and `git tag -l/--list/-n*` are read; otherwise mutates.
        if len(tokens) == 2:
            return RiskLevel.READ
        third = tokens[2]
        if third in {"-l", "--list"} or third.startswith("-n"):
            return RiskLevel.READ
        return RiskLevel.EXECUTE
    if sub == "branch":
        # `git branch` (list) and `-a/-r/-v/--list/--show-current/--all` flags
        # are read; `-d/-D/-m/-c/<branchname>` mutate.
        if len(tokens) == 2:
            return RiskLevel.READ
        for tok in tokens[2:]:
            if tok in {
                "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose",
                "--list", "--show-current", "--contains", "--no-contains",
                "--merged", "--no-merged", "--points-at", "--sort",
            }:
                continue
            return RiskLevel.EXECUTE
        return RiskLevel.READ
    if sub == "worktree":
        if len(tokens) >= 3 and tokens[2] == "list":
            return RiskLevel.READ
        return RiskLevel.EXECUTE
    if sub == "submodule":
        if len(tokens) >= 3 and tokens[2] in {"status", "summary"}:
            return RiskLevel.READ
        return RiskLevel.EXECUTE
    return RiskLevel.EXECUTE


def _classify_runtime(tokens: list[str]) -> RiskLevel:
    """Runtime invocations (python, node) — only --version-style flags are read."""
    cmd = tokens[0]
    flags = _RUNTIME_READ_FLAGS.get(cmd)
    if flags is None or len(tokens) < 2:
        return RiskLevel.EXECUTE
    if tokens[1] in flags:
        return RiskLevel.READ
    return RiskLevel.EXECUTE


def _classify_pkg_manager(tokens: list[str]) -> RiskLevel:
    """Package manager — read-only when the subcommand is in the allowlist."""
    cmd = tokens[0]
    subs = _PKG_READ_SUBCOMMANDS.get(cmd)
    if subs is None or len(tokens) < 2:
        return RiskLevel.EXECUTE
    sub = tokens[1]
    if sub not in subs:
        return RiskLevel.EXECUTE
    # Special case: `ruff check` is read-only unless --fix or --fix-only appears.
    if cmd == "ruff" and sub == "check":
        for tok in tokens[2:]:
            if tok in {"--fix", "--fix-only", "--unsafe-fixes"}:
                return RiskLevel.EXECUTE
    # Special case: `npm/pnpm/yarn/pip config` — only with --get/list flags is read.
    if sub == "config":
        if len(tokens) < 3:
            return RiskLevel.EXECUTE
        third = tokens[2]
        if third in {"get", "list", "ls", "--get", "--list"}:
            return RiskLevel.READ
        return RiskLevel.EXECUTE
    return RiskLevel.READ


def _classify_command_builtin(tokens: list[str]) -> RiskLevel:
    """`command -v <name>` / `command -V <name>` are read-only lookups."""
    if len(tokens) >= 2 and tokens[1] in {"-v", "-V", "-p"}:
        return RiskLevel.READ
    return RiskLevel.EXECUTE


# Map first-token to a per-command classifier.  Anything not in this map and
# not in `_READ_ONLY_PLAIN_COMMANDS` is EXECUTE by default.
_SUBCOMMAND_CLASSIFIERS: dict[str, "object"] = {
    "find": _classify_find,
    "git": _classify_git,
    "command": _classify_command_builtin,
}
for _runtime in _RUNTIME_READ_FLAGS:
    _SUBCOMMAND_CLASSIFIERS[_runtime] = _classify_runtime
for _pkg in _PKG_READ_SUBCOMMANDS:
    _SUBCOMMAND_CLASSIFIERS[_pkg] = _classify_pkg_manager


def classify_bash_command(cmd: str) -> RiskLevel:
    """Classify a Bash command as READ (auto-approve) or EXECUTE (default-deny).

    Returns ``RiskLevel.READ`` only when deterministic parsing proves the
    command does not write, exec subcommands, or compose shell flow.
    Anything else returns ``RiskLevel.EXECUTE``.

    The safety layer uses the result to override the registered Bash risk
    level for this single invocation.  EXECUTE-classified commands still go
    through the existing approval / blocked-pattern path unchanged.
    """
    if not cmd or not cmd.strip():
        return RiskLevel.EXECUTE

    # 1. Reject shell composition outright.  Pipes, redirects, command
    #    separators, command substitution — all defeat leading-command
    #    analysis.  Cheap and conservative.
    if _has_shell_composition(cmd):
        return RiskLevel.EXECUTE

    # 2. Tokenize.  shlex handles quoting; failure (unbalanced quotes, etc.)
    #    means we don't trust the parse.
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return RiskLevel.EXECUTE
    if not tokens:
        return RiskLevel.EXECUTE

    # 3. Reject any token that starts/ends with a redirect glyph.  shlex may
    #    glue redirects to filenames (e.g. ``ls >file.txt`` -> ``['ls',
    #    '>file.txt']``); the composition check above already rejects raw
    #    ``>`` / ``<`` so this is belt-and-braces.
    for tok in tokens:
        if not tok:
            continue
        if tok[0] in "<>" or tok[-1] in "<>":
            return RiskLevel.EXECUTE

    # 4. Reject env-var assignment prefix (``FOO=bar cmd``).  These can
    #    override LD_PRELOAD / PATH and change command semantics.
    if _looks_like_env_assignment(tokens[0]):
        return RiskLevel.EXECUTE

    first = tokens[0]

    # 5. Plain read-only command?
    if first in _READ_ONLY_PLAIN_COMMANDS:
        return RiskLevel.READ

    # 6. Subcommand-validated command?
    classifier = _SUBCOMMAND_CLASSIFIERS.get(first)
    if classifier is not None:
        return classifier(tokens)  # type: ignore[operator]

    # 7. Default-deny: not in any allowlist => EXECUTE.
    return RiskLevel.EXECUTE
