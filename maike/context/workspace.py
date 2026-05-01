"""Workspace snapshot builder — concise orientation context for agents."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from maike.tools.repomap import _build_repo_map

# Caps for individual sections within the snapshot.
_TREE_CAP = 3000
_GIT_CAP = 800
_SNAPSHOT_CAP = 4000


# ────────────────────────── Remote repo URL detection ──────────────────────────
#
# When the workspace is empty AND the user's task references a remote git
# repository, the snapshot should orient the agent toward cloning rather than
# the generic "do not run directory listings" greenfield message.  Without
# this, an agent given "summarize https://github.com/foo/bar" sees only
# "empty workspace, do not list files" and has to figure out from scratch
# that the answer is to clone the URL.  In the 76-turn transcript that drove
# this fix, this exact misorientation produced ~10 wasted turns of `ls`
# variants before the agent landed on `git clone`.

# Pattern 1: well-known git providers.  Catches the bare https URL form even
# when it has no .git suffix (e.g. https://github.com/user/repo).
_KNOWN_PROVIDERS_URL_RE = re.compile(
    r"https?://(?:www\.)?"
    r"(?:github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|gitea\.com|"
    r"git\.sr\.ht|sr\.ht)"
    r"/[\w./~%-]+",
    re.IGNORECASE,
)

# Pattern 2: any HTTPS URL ending in .git (covers self-hosted git instances).
_GENERIC_DOT_GIT_URL_RE = re.compile(
    r"https?://[\w.-]+(?::\d+)?/[\w./~%-]+\.git\b",
    re.IGNORECASE,
)

# Pattern 3: SSH-style git URLs (git@host:user/repo[.git]).
_SSH_GIT_URL_RE = re.compile(
    r"git@[\w.-]+:[\w./~%-]+(?:\.git)?",
)

# Trailing punctuation we strip from URL matches — these are almost always
# sentence punctuation, not part of the URL.
_URL_TRAILING_TRIM = ".,;:)]>}'\"!?"


def _detect_remote_repo_url(text: str) -> str | None:
    """Return the first plausible git clone URL in *text*, or None.

    Tries known providers first, then any ``.git`` URL, then SSH-style.
    Trailing sentence punctuation is stripped.  Returns None on empty input
    or no match.
    """
    if not text:
        return None
    for pattern in (_KNOWN_PROVIDERS_URL_RE, _GENERIC_DOT_GIT_URL_RE, _SSH_GIT_URL_RE):
        match = pattern.search(text)
        if match:
            return match.group(0).rstrip(_URL_TRAILING_TRIM)
    return None


def _git_recent_changes(workspace: Path, depth: int = 3) -> str:
    """Return ``git diff --stat HEAD~N..HEAD`` output, or empty string.

    Only uses git if ``.git`` exists in the workspace root itself —
    does not pick up inherited parent repositories.
    """
    # Guard: only use git if .git is in THIS directory, not inherited.
    if not (workspace / ".git").exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", f"HEAD~{depth}..HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            output = result.stdout.strip()
            if len(output) > _GIT_CAP:
                output = output[:_GIT_CAP] + "\n... (truncated)"
            return output
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


def _detect_config_files(workspace: Path) -> list[str]:
    """Return names of well-known config files present at the workspace root."""
    candidates = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "tsconfig.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env",
        "requirements.txt",
        "Pipfile",
    ]
    return [name for name in candidates if (workspace / name).exists()]


def _test_directories(workspace: Path) -> list[str]:
    """Return names of test-related directories present at the workspace root."""
    candidates = ["tests", "test", "__tests__", "spec", "specs", "test_data"]
    return [name for name in candidates if (workspace / name).is_dir()]


class WorkspaceSnapshotBuilder:
    """Produces concise workspace orientation context capped at ~4000 chars."""

    def build_snapshot(
        self,
        workspace: Path,
        *,
        recent_changes: list[str] | None = None,
        include_git: bool = True,
        task: str = "",
    ) -> str:
        """Build a generic workspace snapshot.

        When the workspace is empty AND *task* references a remote git
        repository URL, the snapshot points the agent toward cloning instead
        of the default "no files to explore" greenfield message.  This
        prevents the agent from looping on `ls` when the task is "summarize
        repo X" against an empty workspace.
        """
        parts: list[str] = []

        # File tree (no signatures, shallow).
        tree = _build_repo_map(str(workspace), max_depth=2, include_signatures=False)
        if tree:
            if len(tree) > _TREE_CAP:
                tree = tree[:_TREE_CAP] + "\n... (truncated)"
            parts.append(f"### File Structure\n```\n{tree}\n```")

        # Git changes.
        if include_git:
            git_stat = _git_recent_changes(workspace)
            if git_stat:
                parts.append(f"### Recent Changes (last 3 commits)\n```\n{git_stat}\n```")

        # Config files.
        configs = _detect_config_files(workspace)
        if configs:
            parts.append("### Config Files Present\n" + ", ".join(configs))

        if not parts:
            return self._empty_workspace_snapshot(task)

        snapshot = "## Workspace Snapshot\n\n" + "\n\n".join(parts)
        if len(snapshot) > _SNAPSHOT_CAP:
            snapshot = snapshot[:_SNAPSHOT_CAP] + "\n... (snapshot truncated)"
        return snapshot

    @staticmethod
    def _empty_workspace_snapshot(task: str) -> str:
        """Return the empty-workspace snapshot, URL-aware when possible.

        If *task* contains a remote git URL, orient the agent toward cloning
        (and away from the generic "do not run ls" message that contradicts
        the obvious next step).  Otherwise, return the original greenfield
        message verbatim — same wording, same anti-`ls`-loop guidance.
        """
        repo_url = _detect_remote_repo_url(task or "")
        if repo_url:
            return (
                "## Workspace Snapshot\n\n"
                "**Empty workspace** + **remote repository URL detected** "
                f"in the task: `{repo_url}`\n\n"
                "If the task requires fetching that code, clone it into a "
                "subdirectory first:\n\n"
                f"```\ngit clone --depth=1 {repo_url} repo\n```\n\n"
                "Then read files via paths like `repo/README.md`. Do NOT "
                "clone into `.` — mAIke creates a `.maike/` session "
                "directory at the workspace root and `git clone <url> .` "
                "will fail with \"destination path '.' already exists "
                "and is not an empty directory\".\n\n"
                "If the task does not require fetching the remote code, "
                "ignore this hint."
            )
        return (
            "## Workspace Snapshot\n\n"
            "**Empty workspace** — no source files found. This is a "
            "greenfield project. There are no files to explore — do not "
            "run directory listings."
        )

    def build_snapshot_for_role(
        self,
        workspace: Path,
        role: str,
        *,
        files_in_scope: list[str] | None = None,
        task: str = "",
    ) -> str:
        """Build a role-specific workspace snapshot variant.

        *task* is used by the empty-workspace branch of ``build_snapshot``
        to surface a clone-first hint when the task references a remote
        git URL.  Role-specific variants don't currently consume *task*
        beyond the greenfield fallback.
        """
        if role in ("coder", "debugger"):
            return self._coder_snapshot(workspace, files_in_scope)
        if role == "tester":
            return self._tester_snapshot(workspace, task=task)
        if role in ("reviewer", "acceptance"):
            return self._reviewer_snapshot(workspace)
        return self.build_snapshot(workspace, task=task)

    def _coder_snapshot(self, workspace: Path, files_in_scope: list[str] | None = None) -> str:
        parts: list[str] = []

        # File tree.
        tree = _build_repo_map(str(workspace), max_depth=2, include_signatures=False)
        if tree:
            if len(tree) > _TREE_CAP:
                tree = tree[:_TREE_CAP] + "\n... (truncated)"
            parts.append(f"### File Structure\n```\n{tree}\n```")

        # Git changes.
        git_stat = _git_recent_changes(workspace)
        if git_stat:
            parts.append(f"### Recent Changes\n```\n{git_stat}\n```")

        # Scoped files.
        if files_in_scope:
            parts.append("### Files In Scope\n" + "\n".join(f"- {f}" for f in files_in_scope))

        configs = _detect_config_files(workspace)
        if configs:
            parts.append("### Config Files\n" + ", ".join(configs))

        snapshot = "## Workspace Snapshot\n\n" + "\n\n".join(parts)
        if len(snapshot) > _SNAPSHOT_CAP:
            snapshot = snapshot[:_SNAPSHOT_CAP] + "\n... (snapshot truncated)"
        return snapshot

    def _tester_snapshot(self, workspace: Path, *, task: str = "") -> str:
        parts: list[str] = []

        # Test directories.
        test_dirs = _test_directories(workspace)
        if test_dirs:
            trees: list[str] = []
            for td in test_dirs:
                t = _build_repo_map(str(workspace / td), max_depth=2, include_signatures=False)
                if t:
                    trees.append(f"{td}/\n{t}")
            if trees:
                combined = "\n".join(trees)
                if len(combined) > _TREE_CAP:
                    combined = combined[:_TREE_CAP] + "\n... (truncated)"
                parts.append(f"### Test Directory Structure\n```\n{combined}\n```")

        # Config files relevant to testing.
        configs = _detect_config_files(workspace)
        test_configs = [c for c in configs if c in ("pyproject.toml", "setup.cfg", "package.json", "Cargo.toml", "go.mod")]
        if test_configs:
            parts.append("### Test Config Files\n" + ", ".join(test_configs))

        if not parts:
            return self.build_snapshot(workspace, include_git=False, task=task)

        snapshot = "## Workspace Snapshot (Test Focus)\n\n" + "\n\n".join(parts)
        if len(snapshot) > _SNAPSHOT_CAP:
            snapshot = snapshot[:_SNAPSHOT_CAP] + "\n... (snapshot truncated)"
        return snapshot

    def _reviewer_snapshot(self, workspace: Path) -> str:
        parts: list[str] = []

        # Shallow tree.
        tree = _build_repo_map(str(workspace), max_depth=1, include_signatures=False)
        if tree:
            if len(tree) > 2000:
                tree = tree[:2000] + "\n... (truncated)"
            parts.append(f"### Project Structure\n```\n{tree}\n```")

        configs = _detect_config_files(workspace)
        if configs:
            parts.append("### Config Files\n" + ", ".join(configs))

        snapshot = "## Workspace Snapshot\n\n" + "\n\n".join(parts)
        if len(snapshot) > _SNAPSHOT_CAP:
            snapshot = snapshot[:_SNAPSHOT_CAP] + "\n... (snapshot truncated)"
        return snapshot
