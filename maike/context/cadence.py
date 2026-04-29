"""Per-turn cadence-based reminders for long-running agents.

Complements the reactive nudge systems (SessionToolTracker, RepeatedFailureTracker,
convergence detection) with proactive, periodic reminders that prevent agents from
losing sight of their plan and TODOs on long multi-step tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Cadence configuration constants
# ---------------------------------------------------------------------------

# Milestone / progress reminder: fires when the agent hasn't written a
# milestone note for this many assistant turns.
MILESTONE_REMINDER_INTERVAL = 15

# Plan reminder: re-inject the agent's stated plan every N turns. Wider
# than the old 5-turn cadence because the earlier setting nudged the
# agent to defensively restate the plan on every reminder ("I am still
# following the plan…"). 15 gives real breathing room between reminders.
PLAN_REMINDER_INTERVAL = 15

# Full plan text every Nth plan reminder; sparse (one-liner) otherwise.
PLAN_FULL_EVERY_N = 3

# Minimum number of numbered steps required for a message block to be
# treated as a "plan" worth re-injecting. Low counts (1-2 items) are
# usually ad-hoc to-do lists, not architectural plans, and re-injecting
# them trains the model to restate trivial lists.
MIN_PLAN_STEPS = 3

# Context usage warning threshold (fraction of context window).
CONTEXT_WARNING_THRESHOLD = 0.50

# Maximum cadence reminders per turn to avoid flooding.
MAX_REMINDERS_PER_TURN = 2


@dataclass
class CadenceTracker:
    """Track turn counts and emit proactive reminders at configured intervals.

    Usage::

        tracker = CadenceTracker()
        # After each assistant turn completes:
        tracker.tick(wrote_milestone=True/False)
        # Before the next LLM call:
        reminders = tracker.get_reminders(
            convergence_level=0,
            context_usage_fraction=0.4,
            plan_text="...",
        )
    """

    # --- Internal counters ---------------------------------------------------
    _turns_since_milestone: int = 0
    _turns_total: int = 0
    _plan_reminder_count: int = 0
    _milestone_reminders_emitted: int = 0
    _context_warning_emitted: bool = False

    # Stored plan text extracted from conversation (set externally).
    _plan_text: str = ""

    def tick(self, *, wrote_milestone: bool = False) -> None:
        """Advance all cadence counters by one assistant turn.

        Args:
            wrote_milestone: True if the assistant wrote a ``## Milestone:``
                note in the turn that just completed.  Resets the milestone
                reminder counter.
        """
        self._turns_total += 1
        if wrote_milestone:
            self._turns_since_milestone = 0
        else:
            self._turns_since_milestone += 1

    def set_plan(self, plan_text: str) -> None:
        """Store the agent's stated plan for periodic re-injection."""
        self._plan_text = plan_text.strip()

    def get_reminders(
        self,
        *,
        convergence_level: int = 0,
        context_usage_fraction: float = 0.0,
    ) -> list[str]:
        """Generate cadence-based reminders for this turn.

        Args:
            convergence_level: Current convergence escalation level (0-3).
                When >= 1, cadence reminders are suppressed — convergence
                nudges are higher priority and should not be diluted.
            context_usage_fraction: Fraction of context window used (0.0-1.0).

        Returns:
            List of reminder strings to inject as ``<maike-nudge>`` messages.
            At most ``MAX_REMINDERS_PER_TURN`` reminders per call.
        """
        if convergence_level >= 1:
            return []

        reminders: list[str] = []

        # Cadence reminders are framed as <system-reminder> blocks and written
        # imperatively (not as questions) so the model treats them as passive
        # nudges rather than user turns that need a reply. Questions like
        # "Are you still following this plan?" trained the agent to
        # defensively restate the plan every cycle.

        def _wrap(text: str) -> str:
            return (
                "<system-reminder>\n"
                f"{text}\n"
                "This is a passive nudge — do not acknowledge it in your "
                "response. Continue working if no course correction is needed."
                "\n</system-reminder>"
            )

        # --- Milestone / progress reminder ---
        if (
            self._turns_since_milestone >= MILESTONE_REMINDER_INTERVAL
            and self._turns_since_milestone % MILESTONE_REMINDER_INTERVAL == 0
        ):
            reminders.append(_wrap(
                "No `## Milestone:` note has been written in "
                f"{self._turns_since_milestone} turns. Add one at the next "
                "natural break (after a sub-task completes) to preserve "
                "progress across context compression."
            ))
            self._milestone_reminders_emitted += 1

        # --- Plan re-injection ---
        # Only fires when a plan was actually extracted; the sparse "you
        # stated a plan earlier…" variant is gone because it produced
        # defensive replies even when no real plan existed.
        if (
            self._plan_text
            and self._turns_total > 0
            and self._turns_total % PLAN_REMINDER_INTERVAL == 0
        ):
            self._plan_reminder_count += 1
            is_full = (self._plan_reminder_count % PLAN_FULL_EVERY_N) == 1
            if is_full:
                reminders.append(_wrap(
                    "Your stated plan:\n\n"
                    f"{self._plan_text}\n\n"
                    "Keep executing. Only restate the plan if you are "
                    "changing direction."
                ))

        # --- Context usage warning ---
        if (
            not self._context_warning_emitted
            and context_usage_fraction >= CONTEXT_WARNING_THRESHOLD
        ):
            pct = int(context_usage_fraction * 100)
            reminders.append(_wrap(
                f"Context window is at ~{pct}% capacity. Wrap up the current "
                "sub-task and write a `## Milestone:` note. Avoid loading "
                "large files or starting new exploration."
            ))
            self._context_warning_emitted = True

        return reminders[:MAX_REMINDERS_PER_TURN]


def extract_plan_from_conversation(conversation: list[dict]) -> str:
    """Extract a stated plan from the conversation if one exists.

    Looks for numbered plans (lines starting with digits) in assistant
    messages, particularly early messages where the agent typically
    states its approach. Requires at least ``MIN_PLAN_STEPS`` numbered
    items so trivial to-do lists ("1. read file 2. edit") don't get
    treated as architectural plans worth re-injecting.
    """
    for msg in conversation[:10]:  # Only check early messages
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract text blocks
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if not isinstance(content, str):
            continue
        # Look for numbered plan patterns
        lines = content.split("\n")
        plan_lines: list[str] = []
        numbered_count = 0
        in_plan = False
        for line in lines:
            stripped = line.strip()
            is_numbered = bool(stripped) and (
                stripped[0].isdigit()
                and any(stripped[1:3].startswith(c) for c in [".", ")", ":"])
            )
            if is_numbered:
                in_plan = True
                plan_lines.append(stripped)
                numbered_count += 1
            elif in_plan and stripped.startswith("-"):
                plan_lines.append(stripped)
            elif in_plan and not stripped:
                # Blank line ends the plan block
                if numbered_count >= MIN_PLAN_STEPS:
                    break
                in_plan = False
                plan_lines.clear()
                numbered_count = 0
            else:
                if in_plan and numbered_count >= MIN_PLAN_STEPS:
                    break
                in_plan = False
                plan_lines.clear()
                numbered_count = 0
        if numbered_count >= MIN_PLAN_STEPS:
            return "\n".join(plan_lines)
    return ""


def detect_milestone_in_text(text: str) -> bool:
    """Check if the given text contains a milestone note."""
    if not isinstance(text, str):
        return False
    return "## Milestone:" in text or "## milestone:" in text
