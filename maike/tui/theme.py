"""Color palette and constants for the mAIke TUI."""

from __future__ import annotations

import platform

# Platform-aware symbols
IS_MAC = platform.system() == "Darwin"

# ---------------------------------------------------------------------------
# Color palette — coherent 6-color system
# ---------------------------------------------------------------------------
MAIKE_ACCENT = "#3dbfa0"     # teal-green (primary, brand, tools)
MAIKE_SECONDARY = "#8b5cf6"  # purple (delegates, thinking)
MAIKE_WARNING = "#f5a623"    # amber (approvals, slow states)
MAIKE_SUCCESS = "#22c55e"    # green (completed tools)
MAIKE_ERROR = "#ef4444"      # red (errors)
MAIKE_DIM = "#6b7280"        # gray (secondary text)

# Status dot — single glyph on every platform.  U+25CF (BLACK CIRCLE) is a
# text-presentation codepoint so it renders in the terminal's monospace
# font instead of the emoji fallback.  macOS Terminal used to get U+23FA
# (RECORD BUTTON) which some fonts render as a colored emoji, producing
# inconsistent width and a "weird" glyph mixed in with plain text.
DOT = "\u25cf"  # ●
DOT_SUCCESS = f"[{MAIKE_SUCCESS}]{DOT}[/{MAIKE_SUCCESS}]"
DOT_ERROR = f"[{MAIKE_ERROR}]{DOT}[/{MAIKE_ERROR}]"
DOT_PENDING = f"[dim]{DOT}[/dim]"
DOT_ACTIVE = f"[{MAIKE_ACCENT}]{DOT}[/{MAIKE_ACCENT}]"
DOT_DELEGATE = f"[{MAIKE_SECONDARY}]{DOT}[/{MAIKE_SECONDARY}]"

# Gutter connector (box-drawing char)
GUTTER = "[dim]  \u231f  [/dim]"  # ⎿

# ---------------------------------------------------------------------------
# Spinner configuration — simple dot-cycle for a steady, monospace-safe
# animation.  The previous diamond bounce (◇◈◆◈◇·) mixed glyphs with
# different visual weights, so each frame shifted perceived width.  A
# three-dot cycle reads as a smooth "thinking" pulse without jitter.
# ---------------------------------------------------------------------------
SPINNER_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
# Braille spinner — same cell width every frame, renders cleanly in any
# monospace font.  (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏)
SPINNER_COLOR = MAIKE_ACCENT
SPINNER_INTERVAL_S = 0.1  # 100ms per frame — smooth without burning CPU
# Stall color thresholds (seconds)
SPINNER_STALL_WARN_S = 10   # shift toward amber
SPINNER_STALL_ALERT_S = 30  # shift toward soft red

# Tool output truncation limits
TOOL_OUTPUT_MAX_LINES = 30
TOOL_OUTPUT_MAX_CHARS = 3000

# Tool rollup threshold — collapse after this many consecutive tool calls
TOOL_ROLLUP_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Agent states
# ---------------------------------------------------------------------------
STATE_THINKING = "thinking"
STATE_EXECUTING = "executing"
STATE_WAITING = "waiting"

STATE_COLORS = {
    STATE_THINKING: MAIKE_ACCENT,
    STATE_EXECUTING: MAIKE_WARNING,
    STATE_WAITING: MAIKE_DIM,
}

STATE_LABELS = {
    STATE_THINKING: "Reasoning\u2026",
    STATE_EXECUTING: "Running tool\u2026",
    STATE_WAITING: "Ready",
}

STATE_SYMBOLS = {
    STATE_THINKING: "\u25c6",    # ◆
    STATE_EXECUTING: "\u25c6",   # ◆
    STATE_WAITING: "\u25c7",     # ◇
}

# Turn separator
TURN_SEPARATOR_CHAR = "\u2500"   # ─
