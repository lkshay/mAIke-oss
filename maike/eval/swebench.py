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
    # ``errors="replace"`` — git diff occasionally surfaces non-UTF-8 bytes
    # (binary files added by the agent, files with mixed encodings, mojibake
    # in test fixtures).  Without this, capture_patch crashes with
    # ``UnicodeDecodeError`` and torpedoes the entire eval loop.  Bug
    # observed during SWE-bench v2 Pro run, 24 Apr 2026 — instance 9 of 12
    # crashed on byte 0x8f at position 79651 in subprocess output.
    try:
        if base_commit:
            committed = subprocess.run(
                ["git", "diff", f"{base_commit}..HEAD"],
                cwd=workspace, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30,
            ).stdout
            uncommitted = subprocess.run(
                ["git", "diff"],
                cwd=workspace, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30,
            ).stdout
            combined = committed + uncommitted

            # Stash recovery — if the captured diff is empty but a stash
            # exists, the agent likely ran ``git stash`` without popping
            # (often during confused "clean up" attempts).  Surface the
            # stashed diff so the work isn't silently lost.  See
            # django__django-11400 (24 Apr 2026): agent did real work per
            # session memory but the patch was empty.  We log loudly so
            # the failure mode is debuggable, and append the stash diff
            # to the captured patch so harness scoring at least sees the
            # work.  If the user intentionally stashed (rare), they'll
            # see the warning and can investigate.
            if not _normalize_patch(combined):
                stash_diff = _capture_stash_diff(workspace)
                if stash_diff:
                    logger.warning(
                        "capture_patch: HEAD diff against base_commit is empty "
                        "but git stash contains %d chars of agent work; "
                        "recovering stashed changes into the patch. "
                        "(workspace=%s)",
                        len(stash_diff), workspace,
                    )
                    return _normalize_patch(stash_diff)
            return _normalize_patch(combined)
        result = subprocess.run(
            ["git", "diff"],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return _normalize_patch(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _capture_stash_diff(workspace: Path) -> str:
    """Return the diff of all entries in ``git stash list``, concatenated.

    Used as a recovery path in ``capture_patch`` when HEAD is unchanged.
    Returns empty string when there are no stashes or git fails.
    """
    try:
        listing = subprocess.run(
            ["git", "stash", "list"],
            cwd=workspace, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    stash_refs = [
        line.split(":", 1)[0].strip()
        for line in listing.splitlines()
        if line.startswith("stash@")
    ]
    if not stash_refs:
        return ""
    diffs: list[str] = []
    for ref in stash_refs:
        try:
            d = subprocess.run(
                ["git", "stash", "show", "-p", ref],
                cwd=workspace, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=15,
            ).stdout
            if d:
                diffs.append(d)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    return "\n".join(diffs)


def _normalize_patch(text: str) -> str:
    """Make captured-diff text harness-safe.

    Three transforms:

    1. Strip leading whitespace (cosmetic).
    2. Strip no-op hunks — hunks where ``-`` and ``+`` lines are
       byte-identical and in the same order.  These appear when the
       agent edits a file then partially reverts, leaving phantom
       hunks that ``patch(1)`` interprets as "Reversed (or previously
       applied)" — sometimes reverse-applying them and undoing real
       work.  See SWE-bench v2 astropy__astropy-13398 (24 Apr 2026).
    3. Guarantee exactly one trailing newline.  ``patch(1)`` rejects
       diffs ending mid-line with "patch unexpectedly ends in middle
       of line".

    Returns empty when there's no meaningful content.
    """
    text = text.lstrip()
    if not text.strip():
        return ""
    text = _strip_noop_hunks(text)
    if not text.strip():
        return ""
    return text if text.endswith("\n") else text + "\n"


def _strip_noop_hunks(diff: str) -> str:
    """Drop hunks whose ``-`` lines exactly match ``+`` lines (no-ops).

    Also drops file sections whose every hunk is a no-op.  Keeps file
    headers (``diff --git``, ``---``, ``+++``, ``index``, mode lines)
    only for files with at least one surviving hunk.
    """
    if not diff:
        return ""

    output_files: list[str] = []
    current_file_header: list[str] = []
    current_hunks: list[tuple[str, list[str]]] = []
    current_hunk_header: str | None = None
    current_hunk_body: list[str] = []
    in_file = False

    def _flush_hunk() -> None:
        nonlocal current_hunk_header, current_hunk_body
        if current_hunk_header is not None:
            current_hunks.append((current_hunk_header, current_hunk_body))
        current_hunk_header = None
        current_hunk_body = []

    def _flush_file() -> None:
        nonlocal current_file_header, current_hunks
        _flush_hunk()
        kept = [(h, body) for h, body in current_hunks if not _is_noop_hunk(body)]
        if kept and current_file_header:
            output_files.extend(current_file_header)
            for h, body in kept:
                output_files.append(h)
                output_files.extend(body)
        current_file_header = []
        current_hunks = []

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            _flush_file()
            current_file_header = [line]
            in_file = True
        elif in_file and line.startswith("@@"):
            _flush_hunk()
            current_hunk_header = line
            current_hunk_body = []
        elif current_hunk_header is not None:
            current_hunk_body.append(line)
        elif in_file:
            current_file_header.append(line)
    _flush_file()
    return "".join(output_files)


def _is_noop_hunk(body: list[str]) -> bool:
    """A hunk whose removed lines exactly match its added lines does nothing."""
    removed = [
        line[1:]
        for line in body
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line[1:]
        for line in body
        if line.startswith("+") and not line.startswith("+++")
    ]
    # Empty + empty would be a hunk with only context (malformed).  Treating
    # it as a no-op is the safe choice — patch(1) would reject it anyway.
    return removed == added


def validate_patch_applies(patch: str, base_commit: str, repo_workspace: Path) -> tuple[bool, str]:
    """Pre-flight check: does ``patch`` apply cleanly against ``base_commit``?

    Uses ``git apply --check --reverse`` against the repo's current HEAD
    (assumed to contain the patch's effects, since capture_patch was run
    on this same workspace).  ``--reverse --check`` succeeds iff the
    patch can be cleanly *un*-applied — which is equivalent to the patch
    being internally consistent with the workspace's net change since
    ``base_commit``.

    Returns ``(ok, reason)``.  ``reason`` is an empty string on success,
    or a brief diagnostic on failure (e.g. "git apply --check --reverse
    failed: …").  Failure here doesn't mean the patch is necessarily
    bad in the harness — it just means the local workspace state can't
    self-validate it.
    """
    if not patch:
        return False, "empty patch"
    try:
        proc = subprocess.run(
            ["git", "apply", "--check", "--reverse", "-"],
            cwd=repo_workspace,
            input=patch,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"validation subprocess failed: {exc!r}"
    if proc.returncode == 0:
        return True, ""
    msg = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = " | ".join(msg[-3:]) if msg else "unknown"
    return False, f"git apply --check --reverse failed: {tail}"


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
