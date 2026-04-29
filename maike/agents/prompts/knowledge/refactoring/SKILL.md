---
name: refactoring
description: "Safe refactoring patterns — preserve behavior, make incremental changes, verify at each step. Use when restructuring, reorganizing, or migrating code."
compatibility: "mAIke coding agent"
metadata:
  triggers: "refactor, restructure, reorganize, migrate, rename across, extract, move to"
  auto_inject: "false"
---

## Safe Refactoring

When restructuring, renaming, or reorganizing existing code:

1. **Run existing tests first** — verify the baseline is green before touching anything.
   If tests are already failing, fix them first or note the pre-existing failures.

2. **One refactoring at a time** — do not combine a rename with a logic change.
   Rename first, verify tests pass, then change logic.

3. **Preserve the public interface** — when moving or renaming:
   - Keep old names as aliases/re-exports until all callers are updated.
   - Use Grep to find ALL references before changing a name.
   - Update imports in dependency order (leaves first, roots last).

4. **Verify after each step** — run the test suite after every structural change.
   If a test breaks, fix it immediately before continuing.

5. **For large renames** — use Grep to find all occurrences, then Edit each file.
   Don't rely on memory — always search for both the old and new name after.
