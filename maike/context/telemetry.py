"""Context telemetry — non-intrusive per-agent context metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallTelemetry:
    """Per-tool-call size tracking."""
    tool_name: str
    input_chars: int
    output_chars: int
    compressed_chars: int


@dataclass
class ContextTelemetry:
    """Tracks context budget usage and compression events for a single agent run.

    All mutation methods are O(1) with no I/O so they impose negligible overhead
    on the agent loop.
    """

    # Set once at agent start.
    initial_context_tokens: int = 0
    workspace_snapshot_tokens: int = 0

    # Per-artifact token counts: {"spec.md": 1234, ...}
    artifact_tokens: dict[str, int] = field(default_factory=dict)

    # Which artifacts were summarized vs kept full.
    summarized_artifacts: list[str] = field(default_factory=list)

    # Compression tracking.
    compression_applied: bool = False
    compression_levels_used: list[str] = field(default_factory=list)

    # Conversation lifecycle.
    peak_conversation_tokens: int = 0
    prune_events: int = 0
    tokens_pruned: int = 0

    # Progressive loading.
    on_demand_fetches: int = 0

    # Convergence.
    convergence_nudge_injected: bool = False

    # Per-tool-call tracking.
    tool_calls: list[ToolCallTelemetry] = field(default_factory=list)
    total_tool_output_chars: int = 0
    total_compressed_chars: int = 0

    # Inline waste-detection nudges (Track 3).
    inline_nudge_count: int = 0
    triple_reads_detected: int = 0
    zero_result_greps_detected: int = 0

    def record_prune(self, tokens_before: int, tokens_after: int) -> None:
        """Record a pruning event with before/after token counts."""
        self.prune_events += 1
        self.tokens_pruned += max(0, tokens_before - tokens_after)

    def record_compression(self, levels: list[str]) -> None:
        """Record that context budget compression was applied."""
        self.compression_applied = True
        self.compression_levels_used.extend(levels)

    def update_peak(self, current_tokens: int) -> None:
        """Track the maximum conversation size seen during the agent run."""
        if current_tokens > self.peak_conversation_tokens:
            self.peak_conversation_tokens = current_tokens

    def record_fetch(self) -> None:
        """Record an on-demand fetch_artifact_detail call."""
        self.on_demand_fetches += 1

    def record_tool_call(
        self,
        tool_name: str,
        input_chars: int,
        output_chars: int,
        compressed_chars: int,
    ) -> None:
        """Record the size of a single tool call's input/output."""
        self.tool_calls.append(ToolCallTelemetry(
            tool_name=tool_name,
            input_chars=input_chars,
            output_chars=output_chars,
            compressed_chars=compressed_chars,
        ))
        self.total_tool_output_chars += output_chars
        self.total_compressed_chars += compressed_chars

    def _tool_call_summary(self) -> dict[str, Any]:
        """Aggregate tool call stats by tool name."""
        by_tool: dict[str, dict[str, int]] = {}
        for tc in self.tool_calls:
            entry = by_tool.setdefault(tc.tool_name, {"count": 0, "output_chars": 0, "compressed_chars": 0})
            entry["count"] += 1
            entry["output_chars"] += tc.output_chars
            entry["compressed_chars"] += tc.compressed_chars
        return by_tool

    def report(self) -> dict[str, Any]:
        """Return a plain dict suitable for ``AgentResult.metadata``."""
        return {
            "initial_context_tokens": self.initial_context_tokens,
            "workspace_snapshot_tokens": self.workspace_snapshot_tokens,
            "artifact_tokens": dict(self.artifact_tokens),
            "summarized_artifacts": list(self.summarized_artifacts),
            "compression_applied": self.compression_applied,
            "compression_levels_used": list(self.compression_levels_used),
            "peak_conversation_tokens": self.peak_conversation_tokens,
            "prune_events": self.prune_events,
            "tokens_pruned": self.tokens_pruned,
            "on_demand_fetches": self.on_demand_fetches,
            "convergence_nudge_injected": self.convergence_nudge_injected,
            "tool_call_summary": {
                "total_calls": len(self.tool_calls),
                "total_output_chars": self.total_tool_output_chars,
                "total_compressed_chars": self.total_compressed_chars,
                "by_tool": self._tool_call_summary(),
            },
        }
