"""Tests for Phase 4: Workspace Snapshot + Progressive Context Loading."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from maike.context.workspace import (
    WorkspaceSnapshotBuilder,
    _detect_config_files,
    _detect_remote_repo_url,
    _git_recent_changes,
    _test_directories,
)
from maike.tools.artifact_detail import _extract_section


# ======================================================================
# WorkspaceSnapshotBuilder
# ======================================================================


class TestWorkspaceSnapshotBuilder:
    def test_build_snapshot_returns_string(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Project")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(tmp_path, include_git=False)
        assert isinstance(snapshot, str)
        assert "## Workspace Snapshot" in snapshot

    def test_build_snapshot_includes_file_structure(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(tmp_path, include_git=False)
        assert "main.py" in snapshot
        assert "utils.py" in snapshot

    def test_build_snapshot_respects_cap(self, tmp_path: Path):
        # Create many files to exceed the cap.
        for i in range(200):
            (tmp_path / f"file_{i:04d}.py").write_text(f"x = {i}")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(tmp_path, include_git=False)
        assert len(snapshot) <= 4200  # 4000 + small header overhead

    def test_build_snapshot_detects_config_files(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]")
        (tmp_path / "Dockerfile").write_text("FROM python:3.11")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(tmp_path, include_git=False)
        assert "pyproject.toml" in snapshot
        assert "Dockerfile" in snapshot

    def test_build_snapshot_for_role_coder(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "pyproject.toml").write_text("[project]")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(tmp_path, "coder")
        assert "## Workspace Snapshot" in snapshot
        assert "main.py" in snapshot

    def test_build_snapshot_for_role_tester(self, tmp_path: Path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test_it(): pass")
        (tmp_path / "pyproject.toml").write_text("[project]")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(tmp_path, "tester")
        assert "Test" in snapshot
        assert "test_main.py" in snapshot

    def test_build_snapshot_for_role_tester_fallback(self, tmp_path: Path):
        # No test directories — should fall back to generic snapshot.
        (tmp_path / "main.py").write_text("x = 1")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(tmp_path, "tester")
        assert "## Workspace Snapshot" in snapshot

    def test_build_snapshot_for_role_reviewer(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("x = 1")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(tmp_path, "reviewer")
        assert "## Workspace Snapshot" in snapshot

    def test_build_snapshot_for_role_debugger(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def main(): pass")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(tmp_path, "debugger")
        assert "## Workspace Snapshot" in snapshot

    def test_coder_snapshot_with_files_in_scope(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(
            tmp_path, "coder", files_in_scope=["a.py"]
        )
        assert "a.py" in snapshot
        assert "Files In Scope" in snapshot

    def test_build_snapshot_for_unknown_role(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("x = 1")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(tmp_path, "unknown_role")
        assert "## Workspace Snapshot" in snapshot


# ======================================================================
# Helper functions
# ======================================================================


class TestDetectConfigFiles:
    def test_detects_pyproject(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]")
        assert "pyproject.toml" in _detect_config_files(tmp_path)

    def test_detects_package_json(self, tmp_path: Path):
        (tmp_path / "package.json").write_text("{}")
        assert "package.json" in _detect_config_files(tmp_path)

    def test_no_configs(self, tmp_path: Path):
        assert _detect_config_files(tmp_path) == []


class TestTestDirectories:
    def test_detects_tests_dir(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        assert "tests" in _test_directories(tmp_path)

    def test_detects_test_dir(self, tmp_path: Path):
        (tmp_path / "test").mkdir()
        assert "test" in _test_directories(tmp_path)

    def test_no_test_dirs(self, tmp_path: Path):
        assert _test_directories(tmp_path) == []


class TestGitRecentChanges:
    def test_returns_empty_for_non_git_dir(self, tmp_path: Path):
        assert _git_recent_changes(tmp_path) == ""

    @patch("maike.context.workspace.subprocess.run")
    def test_returns_git_stat(self, mock_run, tmp_path: Path):
        (tmp_path / ".git").mkdir()  # Must have local .git for the function to proceed
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" main.py | 5 +++++\n 1 file changed\n",
        )
        result = _git_recent_changes(tmp_path)
        assert "main.py" in result

    @patch("maike.context.workspace.subprocess.run")
    def test_truncates_long_output(self, mock_run, tmp_path: Path):
        (tmp_path / ".git").mkdir()  # Must have local .git
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="x" * 1000,
        )
        result = _git_recent_changes(tmp_path)
        assert len(result) <= 850  # 800 + truncation marker

    def test_ignores_inherited_parent_git(self, tmp_path: Path):
        """Workspace without its own .git should return empty even if parent has one."""
        # Don't create .git in tmp_path — simulate inherited parent repo
        assert _git_recent_changes(tmp_path) == ""


# ======================================================================
# _extract_section (from artifact_detail.py)
# ======================================================================


class TestExtractSection:
    def test_extracts_single_section(self):
        content = textwrap.dedent("""\
            # Introduction
            Some intro text.

            # Requirements
            - Req 1
            - Req 2

            # Design
            Design stuff.
        """)
        result = _extract_section(content, "Requirements")
        assert result is not None
        assert "Req 1" in result
        assert "Req 2" in result
        assert "Design stuff" not in result

    def test_extracts_last_section(self):
        content = textwrap.dedent("""\
            # First
            first content

            # Last
            last content
        """)
        result = _extract_section(content, "Last")
        assert result is not None
        assert "last content" in result

    def test_returns_none_for_missing_section(self):
        content = "# Existing\nSome content."
        assert _extract_section(content, "Nonexistent") is None

    def test_case_insensitive(self):
        content = "# Requirements\n- item"
        assert _extract_section(content, "requirements") is not None
        assert _extract_section(content, "REQUIREMENTS") is not None

    def test_respects_heading_level(self):
        content = textwrap.dedent("""\
            ## Parent
            Parent content.

            ### Child
            Child content.

            ## Sibling
            Sibling content.
        """)
        result = _extract_section(content, "Parent")
        assert result is not None
        assert "Parent content" in result
        assert "Child content" in result
        assert "Sibling content" not in result

    def test_nested_headings_included(self):
        content = textwrap.dedent("""\
            # Main
            intro

            ## Sub1
            sub1 content

            ## Sub2
            sub2 content

            # Other
            other
        """)
        result = _extract_section(content, "Main")
        assert result is not None
        assert "sub1 content" in result
        assert "sub2 content" in result
        assert "other" not in result


# ======================================================================
# Progressive loading hint in helpers.py
# ======================================================================


class TestProgressiveLoadingHint:
    def test_hint_added_for_summarized_artifacts(self):
        from maike.agents.helpers import build_messages
        from maike.atoms.artifact import Artifact, ArtifactKind, ArtifactStatus, ArtifactType

        artifact = Artifact(
            id="test-1",
            kind=ArtifactKind.STAGE,
            type=ArtifactType.SPEC,
            logical_name="spec.md",
            content="[SUMMARIZED]\n# Spec\n- requirement 1",
            content_hash="abc123",
            produced_by="test",
            stage_name="requirements",
            status=ArtifactStatus.APPROVED,
        )
        messages = build_messages("do something", [artifact])
        content = messages[0]["content"]
        assert "fetch_artifact_detail" in content
        assert "`spec.md`" in content

    def test_no_hint_when_no_summarized_artifacts(self):
        from maike.agents.helpers import build_messages
        from maike.atoms.artifact import Artifact, ArtifactKind, ArtifactStatus, ArtifactType

        artifact = Artifact(
            id="test-1",
            kind=ArtifactKind.STAGE,
            type=ArtifactType.SPEC,
            logical_name="spec.md",
            content="# Full Spec\n- requirement 1",
            content_hash="abc123",
            produced_by="test",
            stage_name="requirements",
            status=ArtifactStatus.APPROVED,
        )
        messages = build_messages("do something", [artifact])
        content = messages[0]["content"]
        assert "fetch_artifact_detail" not in content


# ======================================================================
# build_workspace_snapshot helper
# ======================================================================


class TestBuildWorkspaceSnapshotHelper:
    def test_returns_empty_when_no_workspace(self):
        from maike.agents.helpers import build_workspace_snapshot

        session = MagicMock(spec=[])  # No workspace attribute
        assert build_workspace_snapshot(session, "coder") == ""

    def test_returns_snapshot_with_workspace(self, tmp_path: Path):
        from maike.agents.helpers import build_workspace_snapshot

        (tmp_path / "app.py").write_text("x = 1")
        session = MagicMock()
        session.workspace = tmp_path
        snapshot = build_workspace_snapshot(session, "coder")
        assert "## Workspace Snapshot" in snapshot
        assert "app.py" in snapshot


# ======================================================================
# fetch_artifact_detail tool (unit test with mock session)
# ======================================================================


class TestFetchArtifactDetailTool:
    def test_extract_section_returns_content(self):
        content = "# Spec\n\n## Requirements\n- req1\n- req2\n\n## Design\ndesign stuff"
        result = _extract_section(content, "Requirements")
        assert result is not None
        assert "req1" in result
        assert "design stuff" not in result

    def test_extract_section_not_found(self):
        content = "# Only Section\nsome text"
        assert _extract_section(content, "Missing") is None


# ======================================================================
# Remote repo URL detection — Fix 4 (URL-aware workspace orientation)
# ======================================================================


class TestDetectRemoteRepoUrl:
    """The detector underpins the empty-workspace clone-first hint.

    A URL match flips the orientation message from "do not run ls" to
    "clone first" — preventing the ls-loop documented in the 76-turn
    transcript.
    """

    def test_returns_none_for_empty(self):
        assert _detect_remote_repo_url("") is None
        assert _detect_remote_repo_url(None) is None  # type: ignore[arg-type]

    def test_returns_none_for_no_url(self):
        assert _detect_remote_repo_url("Fix the bug in foo.py") is None
        assert _detect_remote_repo_url("Run pytest and report failures") is None

    def test_returns_none_for_non_git_url(self):
        # Bare https URL on a non-git provider — not detected.
        assert _detect_remote_repo_url("Read https://example.com/docs") is None
        assert _detect_remote_repo_url("Fetch https://news.ycombinator.com") is None

    def test_detects_github_https(self):
        url = _detect_remote_repo_url("Summarize https://github.com/foo/bar")
        assert url == "https://github.com/foo/bar"

    def test_detects_github_https_with_dot_git(self):
        url = _detect_remote_repo_url("Clone https://github.com/foo/bar.git please")
        assert url == "https://github.com/foo/bar.git"

    def test_detects_gitlab(self):
        url = _detect_remote_repo_url("https://gitlab.com/owner/proj")
        assert url == "https://gitlab.com/owner/proj"

    def test_detects_bitbucket(self):
        url = _detect_remote_repo_url("https://bitbucket.org/owner/repo")
        assert url == "https://bitbucket.org/owner/repo"

    def test_detects_codeberg(self):
        url = _detect_remote_repo_url("https://codeberg.org/owner/repo")
        assert url == "https://codeberg.org/owner/repo"

    def test_detects_self_hosted_dot_git(self):
        url = _detect_remote_repo_url("clone https://git.example.com/team/repo.git")
        assert url == "https://git.example.com/team/repo.git"

    def test_detects_ssh_style(self):
        url = _detect_remote_repo_url("git@github.com:foo/bar.git")
        assert url == "git@github.com:foo/bar.git"

    def test_detects_ssh_style_without_dot_git(self):
        url = _detect_remote_repo_url("git@github.com:foo/bar")
        assert url == "git@github.com:foo/bar"

    def test_strips_trailing_punctuation(self):
        # Trailing period from a sentence shouldn't be part of the URL.
        url = _detect_remote_repo_url("See https://github.com/foo/bar.")
        assert url == "https://github.com/foo/bar"
        url2 = _detect_remote_repo_url("(https://github.com/foo/bar)")
        assert url2 == "https://github.com/foo/bar"

    def test_returns_first_match(self):
        url = _detect_remote_repo_url(
            "Compare https://github.com/a/b and https://github.com/c/d"
        )
        assert url == "https://github.com/a/b"

    def test_case_insensitive_host(self):
        url = _detect_remote_repo_url("https://GitHub.com/foo/bar")
        assert url is not None
        # Don't assert exact case — just that the URL was detected.
        assert "foo/bar" in url

    def test_handles_blob_path(self):
        # User mentions a specific file in a repo — we still detect the URL.
        # The hint is advisory; the agent can extract the relevant subpath.
        url = _detect_remote_repo_url(
            "Look at https://github.com/foo/bar/blob/main/src/x.py"
        )
        assert url is not None
        assert url.startswith("https://github.com/foo/bar")


# ======================================================================
# Empty-workspace snapshot — URL-aware orientation
# ======================================================================


class TestEmptyWorkspaceSnapshot:
    """Verify the empty-workspace branch of build_snapshot picks the right
    orientation message based on whether the task contains a remote git URL.
    """

    def test_greenfield_message_when_no_task(self, tmp_path: Path):
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(tmp_path, include_git=False)
        assert "greenfield project" in snapshot
        assert "do not" in snapshot.lower()
        assert "git clone" not in snapshot

    def test_greenfield_message_when_task_has_no_url(self, tmp_path: Path):
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(
            tmp_path, include_git=False, task="Build a CLI tool"
        )
        assert "greenfield project" in snapshot
        assert "git clone" not in snapshot

    def test_clone_hint_when_task_has_github_url(self, tmp_path: Path):
        """The transcript bug: empty workspace + GitHub URL must point at clone."""
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(
            tmp_path,
            include_git=False,
            task="Summarize https://github.com/foo/bar",
        )
        assert "git clone" in snapshot
        assert "https://github.com/foo/bar" in snapshot
        # Must explicitly warn against cloning into `.` — that's the trap that
        # produced the `git init && git remote add origin` thrashing in the
        # transcript (turns 30-33: exit code 3 "remote already exists").
        assert "." in snapshot  # the warning text contains the dot
        assert "destination path" in snapshot.lower()
        # And must NOT carry over the "do not run directory listings" message
        # that contradicts the obvious "clone, then explore" workflow.
        assert "do not run directory listings" not in snapshot

    def test_clone_hint_for_self_hosted_dot_git(self, tmp_path: Path):
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(
            tmp_path,
            include_git=False,
            task="clone https://git.example.com/team/repo.git",
        )
        assert "git clone" in snapshot
        assert "https://git.example.com/team/repo.git" in snapshot

    def test_clone_hint_advisory_phrasing(self, tmp_path: Path):
        """Hint is advisory — false positives must not force bad behavior."""
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(
            tmp_path,
            include_git=False,
            task="Read this code https://github.com/foo/bar",
        )
        # Must contain an "ignore this hint" escape hatch.
        assert "ignore" in snapshot.lower()

    def test_url_hint_only_fires_for_empty_workspace(self, tmp_path: Path):
        """Workspace with files: no clone hint, even if URL in task."""
        (tmp_path / "main.py").write_text("x = 1")
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot(
            tmp_path,
            include_git=False,
            task="Update https://github.com/foo/bar based on main.py",
        )
        # File listing should appear, NOT the clone hint.
        assert "main.py" in snapshot
        assert "git clone" not in snapshot

    def test_build_snapshot_for_role_threads_task(self, tmp_path: Path):
        """build_snapshot_for_role must propagate task to the empty branch."""
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(
            tmp_path,
            "react_agent",
            task="Summarize https://github.com/foo/bar",
        )
        assert "git clone" in snapshot
        assert "https://github.com/foo/bar" in snapshot

    def test_tester_role_empty_workspace_with_url(self, tmp_path: Path):
        """Tester falls back to build_snapshot when no test dirs — task threads."""
        builder = WorkspaceSnapshotBuilder()
        snapshot = builder.build_snapshot_for_role(
            tmp_path,
            "tester",
            task="Test https://github.com/foo/bar",
        )
        assert "git clone" in snapshot

    def test_helper_threads_task(self, tmp_path: Path):
        """The agents.helpers.build_workspace_snapshot wrapper threads task."""
        from maike.agents.helpers import build_workspace_snapshot

        session = MagicMock()
        session.workspace = tmp_path
        snapshot = build_workspace_snapshot(
            session,
            "react_agent",
            task="Summarize https://github.com/foo/bar",
        )
        assert "git clone" in snapshot
        assert "https://github.com/foo/bar" in snapshot

    def test_helper_no_task_falls_back_to_greenfield(self, tmp_path: Path):
        from maike.agents.helpers import build_workspace_snapshot

        session = MagicMock()
        session.workspace = tmp_path
        snapshot = build_workspace_snapshot(session, "react_agent")
        assert "greenfield" in snapshot
        assert "git clone" not in snapshot
