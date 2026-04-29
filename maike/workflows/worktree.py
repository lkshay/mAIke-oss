"""Git worktree management for mAIke.

Exposes user-facing operations to create, list, and remove worktrees
that mAIke manages under ``{workspace}/.maike/worktrees/{name}/``.

The primary use case is "try this approach in a separate branch" —
users can spin up a worktree, run mAIke in it, then either merge the
branch or discard the worktree without disturbing their main working
tree.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# All mAIke-managed worktrees live under this subdirectory.
WORKTREES_SUBDIR = ".maike/worktrees"


@dataclass
class WorktreeInfo:
    """Metadata about an active worktree."""

    name: str           # the branch/directory name
    path: Path          # absolute path to the worktree
    branch: str         # current branch in the worktree
    head: str           # current commit SHA (short)


class WorktreeError(RuntimeError):
    """Raised when a worktree operation fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_git_repo(workspace: Path) -> bool:
    """Check whether *workspace* is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _repo_root(workspace: Path) -> Path:
    """Return the git repository root for *workspace*."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise WorktreeError(
            f"Not a git repository: {workspace}. "
            f"cd into a repo or initialise one with `git init`."
        )
    return Path(result.stdout.strip())


def _sanitize_name(name: str) -> str:
    """Sanitise a branch/worktree name to a filesystem-safe slug."""
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)
    safe = safe.strip("-").strip(".")
    if not safe:
        raise WorktreeError("Worktree name cannot be empty after sanitisation.")
    return safe


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def create_worktree(
    workspace: Path,
    name: str,
    base: str = "HEAD",
) -> Path:
    """Create a worktree at ``{repo_root}/.maike/worktrees/{name}/``.

    Creates a new branch named *name* from *base* and adds a worktree
    for it.  Returns the absolute path to the worktree.

    Raises ``WorktreeError`` on any failure (not a git repo, branch
    exists, worktree path exists, etc.).
    """
    workspace = Path(workspace).resolve()
    if not _is_git_repo(workspace):
        raise WorktreeError(
            f"Not a git repository: {workspace}. "
            f"cd into a repo first."
        )

    repo_root = _repo_root(workspace)
    safe_name = _sanitize_name(name)
    worktree_path = repo_root / WORKTREES_SUBDIR / safe_name

    if worktree_path.exists():
        raise WorktreeError(
            f"Worktree path already exists: {worktree_path}. "
            f"Use `maike worktree remove {safe_name}` first."
        )

    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    # git worktree add -b <branch> <path> <base>
    # The -b flag creates a new branch.  If the branch already exists,
    # git will fail — we surface that clearly.
    try:
        result = subprocess.run(
            ["git", "worktree", "add", "-b", safe_name, str(worktree_path), base],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise WorktreeError("git worktree add timed out after 60s.") from None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Detect common failure: branch already exists.
        if "already exists" in stderr.lower():
            raise WorktreeError(
                f"Branch '{safe_name}' already exists. "
                f"Pick a different name or delete the existing branch first."
            )
        raise WorktreeError(f"git worktree add failed: {stderr}")

    return worktree_path


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_worktrees(workspace: Path) -> list[WorktreeInfo]:
    """List all mAIke-managed worktrees in the current repository.

    Filters ``git worktree list`` to only include worktrees rooted under
    ``.maike/worktrees/`` — external worktrees created by other tools
    are ignored.
    """
    workspace = Path(workspace).resolve()
    if not _is_git_repo(workspace):
        return []

    repo_root = _repo_root(workspace)
    worktrees_dir = repo_root / WORKTREES_SUBDIR

    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return []

    infos: list[WorktreeInfo] = []
    current: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                info = _parse_worktree_block(current, worktrees_dir)
                if info is not None:
                    infos.append(info)
                current = {}
            continue
        if " " in line:
            key, _, value = line.partition(" ")
            current[key] = value
        else:
            current[line] = ""
    # Catch trailing block (no final blank line).
    if current:
        info = _parse_worktree_block(current, worktrees_dir)
        if info is not None:
            infos.append(info)

    return sorted(infos, key=lambda w: w.name)


def _parse_worktree_block(block: dict, worktrees_dir: Path) -> WorktreeInfo | None:
    """Parse one porcelain block into a WorktreeInfo, filtering to mAIke-managed."""
    path_str = block.get("worktree", "")
    if not path_str:
        return None
    path = Path(path_str)
    # Filter: only worktrees under .maike/worktrees/
    try:
        path.relative_to(worktrees_dir)
    except ValueError:
        return None

    branch = block.get("branch", "").replace("refs/heads/", "")
    head = block.get("HEAD", "")[:12]  # short SHA
    name = path.name
    return WorktreeInfo(name=name, path=path, branch=branch or "(detached)", head=head)


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

def remove_worktree(
    workspace: Path,
    name: str,
    delete_branch: bool = False,
) -> None:
    """Remove a mAIke-managed worktree by name.

    If *delete_branch* is True, the associated branch is also deleted.
    Force-removes the worktree (overrides any local changes).

    Raises ``WorktreeError`` if the worktree doesn't exist.
    """
    workspace = Path(workspace).resolve()
    if not _is_git_repo(workspace):
        raise WorktreeError(f"Not a git repository: {workspace}")

    repo_root = _repo_root(workspace)
    safe_name = _sanitize_name(name)
    worktree_path = repo_root / WORKTREES_SUBDIR / safe_name

    if not worktree_path.exists():
        raise WorktreeError(
            f"Worktree not found: {worktree_path}. "
            f"Use `maike worktree list` to see available worktrees."
        )

    # Force-remove the worktree registration.
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=repo_root,
        capture_output=True,
        timeout=30,
    )

    # Ensure directory is gone.
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)

    # Optionally delete the branch.
    if delete_branch:
        subprocess.run(
            ["git", "branch", "-D", safe_name],
            cwd=repo_root,
            capture_output=True,
            timeout=10,
        )
