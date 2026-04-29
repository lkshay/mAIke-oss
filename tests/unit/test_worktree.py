"""Tests for maike.workflows.worktree — user-facing worktree operations."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from maike.workflows.worktree import (
    WorktreeError,
    create_worktree,
    list_worktrees,
    remove_worktree,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    """Initialise a git repo with one commit at *path*."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path, check=True, capture_output=True,
    )
    # Seed with a commit so HEAD is valid.
    (path / "README.md").write_text("# test\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init", "-q"],
        cwd=path, check=True, capture_output=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A fresh git repository with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    return repo


# ---------------------------------------------------------------------------
# create_worktree
# ---------------------------------------------------------------------------


def test_create_worktree_happy_path(git_repo: Path):
    """Create a worktree on a new branch at .maike/worktrees/<name>/."""
    path = create_worktree(git_repo, "feature-x")
    assert path.exists()
    assert path.parent == git_repo / ".maike" / "worktrees"
    assert path.name == "feature-x"

    # Verify the worktree has the seeded commit.
    assert (path / "README.md").exists()


def test_create_worktree_non_git_workspace_fails(tmp_path: Path):
    """Creating a worktree outside a git repo raises WorktreeError."""
    not_a_repo = tmp_path / "plain-dir"
    not_a_repo.mkdir()

    with pytest.raises(WorktreeError, match="Not a git repository"):
        create_worktree(not_a_repo, "feature-x")


def test_create_worktree_existing_path_fails(git_repo: Path):
    """Creating a worktree where one already exists raises."""
    create_worktree(git_repo, "feature-x")
    with pytest.raises(WorktreeError, match="already exists"):
        create_worktree(git_repo, "feature-x")


def test_create_worktree_sanitises_name(git_repo: Path):
    """Names with illegal chars are sanitised."""
    path = create_worktree(git_repo, "feat/abc 123")
    # The directory name should be the sanitised form.
    assert "/" not in path.name
    assert " " not in path.name


# ---------------------------------------------------------------------------
# list_worktrees
# ---------------------------------------------------------------------------


def test_list_worktrees_empty(git_repo: Path):
    """Fresh repo with no worktrees returns empty list."""
    assert list_worktrees(git_repo) == []


def test_list_worktrees_shows_maike_managed(git_repo: Path):
    """Only mAIke-managed worktrees are listed."""
    create_worktree(git_repo, "branch-a")
    create_worktree(git_repo, "branch-b")

    worktrees = list_worktrees(git_repo)
    names = sorted(w.name for w in worktrees)
    assert names == ["branch-a", "branch-b"]


def test_list_worktrees_filters_external(git_repo: Path, tmp_path: Path):
    """Worktrees created outside .maike/worktrees/ are not listed."""
    external_wt = tmp_path / "external-wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", "external", str(external_wt), "HEAD"],
        cwd=git_repo, check=True, capture_output=True,
    )
    # A mAIke-managed worktree for comparison.
    create_worktree(git_repo, "managed")

    worktrees = list_worktrees(git_repo)
    names = [w.name for w in worktrees]
    assert "managed" in names
    assert "external-wt" not in names


# ---------------------------------------------------------------------------
# remove_worktree
# ---------------------------------------------------------------------------


def test_remove_worktree_cleans_up(git_repo: Path):
    """remove_worktree deletes the directory and deregisters."""
    path = create_worktree(git_repo, "temp-branch")
    assert path.exists()

    remove_worktree(git_repo, "temp-branch")
    assert not path.exists()
    # Branch should still exist (delete_branch=False default).
    result = subprocess.run(
        ["git", "branch", "--list", "temp-branch"],
        cwd=git_repo, capture_output=True, text=True,
    )
    assert "temp-branch" in result.stdout


def test_remove_worktree_with_delete_branch(git_repo: Path):
    """remove_worktree(delete_branch=True) also deletes the branch."""
    path = create_worktree(git_repo, "temp-branch")
    assert path.exists()

    remove_worktree(git_repo, "temp-branch", delete_branch=True)
    assert not path.exists()
    result = subprocess.run(
        ["git", "branch", "--list", "temp-branch"],
        cwd=git_repo, capture_output=True, text=True,
    )
    assert "temp-branch" not in result.stdout


def test_remove_worktree_missing_raises(git_repo: Path):
    """Removing a nonexistent worktree raises WorktreeError."""
    with pytest.raises(WorktreeError, match="not found"):
        remove_worktree(git_repo, "does-not-exist")
