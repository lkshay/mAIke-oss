"""mAIke Terminal User Interface — Textual-based TUI."""

from __future__ import annotations

from pathlib import Path


def launch_tui(
    workspace: Path,
    provider: str = "",
    model: str = "",
    budget: float = 5.0,
    verbose: bool = False,
    yes: bool = False,
    language: str = "python",
    adaptive_model: bool = True,
    advisor_enabled: bool = False,
    advisor_provider: str | None = None,
    advisor_model: str | None = None,
    advisor_budget_pct: float = 0.20,
) -> None:
    """Launch the mAIke TUI application."""
    from maike.tui.app import MaikeTUIApp

    kwargs = dict(
        workspace=workspace,
        provider=provider,
        model=model,
        budget=budget,
        verbose=verbose,
        yes=yes,
        language=language,
        adaptive_model=adaptive_model,
    )
    # Pass advisor kwargs only if MaikeTUIApp accepts them — keeps this
    # backwards-compatible while the TUI app constructor catches up.
    try:
        app = MaikeTUIApp(
            **kwargs,
            advisor_enabled=advisor_enabled,
            advisor_provider=advisor_provider,
            advisor_model=advisor_model,
            advisor_budget_pct=advisor_budget_pct,
        )
    except TypeError:
        app = MaikeTUIApp(**kwargs)
    app.run()
