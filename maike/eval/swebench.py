"""SWE-bench dataset adapter for mAIke.

Loads SWE-bench instances from HuggingFace, sets up workspaces by cloning
repos at the right commit, and captures git diffs after the agent runs.
The predictions JSONL file can be evaluated by the SWE-bench Docker harness.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# HuggingFace dataset IDs for each variant.
_DATASET_IDS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


@dataclass(frozen=True)
class SWEBenchInstance:
    """A single SWE-bench task instance."""

    instance_id: str
    repo: str  # e.g. "django/django"
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    # Evaluation-only fields — never shown to the agent.
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    version: str = ""

    @classmethod
    def from_hf_row(cls, row: dict[str, Any]) -> SWEBenchInstance:
        """Parse a HuggingFace dataset row into an instance."""
        fail_to_pass = row.get("FAIL_TO_PASS", "[]")
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = []
        pass_to_pass = row.get("PASS_TO_PASS", "[]")
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = []
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            hints_text=row.get("hints_text", ""),
            test_patch=row.get("test_patch", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            version=row.get("version", ""),
        )


def load_dataset(
    variant: str = "lite",
    split: str = "test",
    instance_ids: list[str] | None = None,
) -> list[SWEBenchInstance]:
    """Load SWE-bench instances from HuggingFace.

    Args:
        variant: ``"lite"`` (300), ``"verified"`` (500), or ``"full"`` (2294).
        split: Dataset split (``"test"`` or ``"dev"``).
        instance_ids: Optional filter — only return these instance IDs.

    Returns:
        List of ``SWEBenchInstance`` objects.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required for SWE-bench. "
            "Install it: pip install datasets"
        )

    dataset_id = _DATASET_IDS.get(variant)
    if dataset_id is None:
        raise ValueError(f"Unknown SWE-bench variant: {variant!r}. Use: {list(_DATASET_IDS)}")

    logger.info("Loading SWE-bench %s (%s split) from %s", variant, split, dataset_id)
    ds = hf_load(dataset_id, split=split)

    instances = [SWEBenchInstance.from_hf_row(row) for row in ds]

    if instance_ids:
        id_set = set(instance_ids)
        instances = [i for i in instances if i.instance_id in id_set]
        logger.info("Filtered to %d instances", len(instances))

    logger.info("Loaded %d SWE-bench instances", len(instances))
    return instances


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------

_CLONE_CACHE_DIR_NAME = ".swebench-repos"


def setup_workspace(
    instance: SWEBenchInstance,
    workspace_root: Path,
    clone_cache: Path | None = None,
) -> Path:
    """Clone the repo at the correct commit for a SWE-bench instance.

    Uses a bare clone cache to avoid re-downloading large repos. Each
    instance gets a worktree (or a fresh clone if worktrees fail).

    Returns the workspace path.
    """
    cache_dir = clone_cache or (workspace_root / _CLONE_CACHE_DIR_NAME)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Repo cache key: owner__repo (e.g. "django__django")
    repo_slug = instance.repo.replace("/", "__")
    bare_path = cache_dir / f"{repo_slug}.git"

    # Clone bare repo if not cached.
    if not bare_path.exists():
        logger.info("Cloning %s (bare) ...", instance.repo)
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{instance.repo}.git", str(bare_path)],
            check=True,
            capture_output=True,
            timeout=300,
        )
    else:
        # Fetch latest refs in case the commit isn't available.
        subprocess.run(
            ["git", "fetch", "--all"],
            cwd=bare_path,
            capture_output=True,
            timeout=120,
        )

    # Create workspace via worktree.
    workspace = workspace_root / instance.instance_id
    # Clean stale state from prior runs. With --keep-workspaces, the
    # cleanup_workspace step is skipped between runs, leaving both the
    # directory and the bare repo's worktree registration in place. Plain
    # shutil.rmtree clears the dir but leaves the registration, so the next
    # `git worktree add` fails with "fatal: '<path>' already exists" or
    # "is a missing but already" — putting every subsequent instance into
    # fallback-clone mode (~30s/instance overhead). Sequence:
    #   1. `worktree remove --force` for the real-worktree case
    #   2. `rmtree` for the fallback-clone case (or whatever's left)
    #   3. `worktree prune` to drop any stale registration whose dir is gone
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(workspace)],
        cwd=bare_path,
        capture_output=True,
        timeout=30,
    )
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=bare_path,
        capture_output=True,
        timeout=30,
    )

    try:
        subprocess.run(
            ["git", "worktree", "add", str(workspace), instance.base_commit],
            cwd=bare_path,
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # Worktree failed — fall back to full clone. Surface the actual
        # error so future debugging doesn't require a manual reproduction.
        stderr_tail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr_tail = exc.stderr.decode(errors="replace").strip().splitlines()[-1] if exc.stderr else ""
        logger.warning(
            "Worktree failed for %s (%s), falling back to clone",
            instance.instance_id,
            stderr_tail or type(exc).__name__,
        )
        subprocess.run(
            ["git", "clone", f"https://github.com/{instance.repo}.git", str(workspace)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "checkout", instance.base_commit],
            cwd=workspace,
            check=True,
            capture_output=True,
            timeout=30,
        )

    return workspace


def cleanup_workspace(
    instance: SWEBenchInstance,
    workspace: Path,
    clone_cache: Path | None = None,
) -> None:
    """Remove a workspace (and its worktree registration if applicable)."""
    cache_dir = clone_cache or (workspace.parent / _CLONE_CACHE_DIR_NAME)
    repo_slug = instance.repo.replace("/", "__")
    bare_path = cache_dir / f"{repo_slug}.git"

    # Remove worktree registration.
    if bare_path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace)],
            cwd=bare_path,
            capture_output=True,
        )

    # Remove directory if it still exists.
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Patch capture and predictions
# ---------------------------------------------------------------------------


def capture_patch(workspace: Path, base_commit: str | None = None) -> str:
    """Capture the agent's net changes against ``base_commit``.

    mAIke's react checkpoint callback commits agent edits to HEAD after every
    file write, so plain ``git diff`` (working tree vs HEAD) returns empty
    even when the agent made real changes. We diff ``base_commit..HEAD`` to
    capture everything since the SWE-bench starting point — including all
    checkpoint commits — plus any uncommitted working-tree changes.

    When ``base_commit`` is omitted, falls back to plain ``git diff`` (the
    legacy behavior, kept for non-SWE-bench callers).

    No filename filtering is applied. If the agent produces scratch files or
    a malformed diff, that is its own failure to surface honestly — not
    something the harness should paper over.
    """
    try:
        if base_commit:
            committed = subprocess.run(
                ["git", "diff", f"{base_commit}..HEAD"],
                cwd=workspace, capture_output=True, text=True, timeout=30,
            ).stdout
            # Append any unstaged working-tree changes (rare — react checkpoint
            # usually commits them, but covers the case where the run aborted
            # between an Edit and the next checkpoint).
            uncommitted = subprocess.run(
                ["git", "diff"],
                cwd=workspace, capture_output=True, text=True, timeout=30,
            ).stdout
            return (committed + uncommitted).strip()
        result = subprocess.run(
            ["git", "diff"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


@dataclass
class Prediction:
    """A single SWE-bench prediction."""

    instance_id: str
    model_name_or_path: str
    model_patch: str


def write_predictions(predictions: list[Prediction], output_path: Path) -> None:
    """Write predictions in SWE-bench JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps({
                "instance_id": pred.instance_id,
                "model_name_or_path": pred.model_name_or_path,
                "model_patch": pred.model_patch,
            }) + "\n")
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)


def load_completed_ids(predictions_path: Path) -> set[str]:
    """Load instance IDs from an existing predictions file (for resume)."""
    if not predictions_path.exists():
        return set()
    ids: set[str] = set()
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ids.add(data["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_prediction(pred: Prediction, output_path: Path) -> None:
    """Append a single prediction to the JSONL file (for incremental writes)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps({
            "instance_id": pred.instance_id,
            "model_name_or_path": pred.model_name_or_path,
            "model_patch": pred.model_patch,
        }) + "\n")
