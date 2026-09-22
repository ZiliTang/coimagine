"""Micro oracle diagnostics: run the suite and write the gap report."""

from .micro import (
    DiagnosticConfig,
    OracleRecord,
    format_gap_markdown,
    run_micro_diagnostics,
    write_report,
)
from .sweeps import (
    communication_budget_curve,
    feedback_strength_sweep,
    probe_cost_sweep,
    run_all as run_sweeps,
)

__all__ = [
    "DiagnosticConfig",
    "OracleRecord",
    "communication_budget_curve",
    "feedback_strength_sweep",
    "format_gap_markdown",
    "probe_cost_sweep",
    "run_micro_diagnostics",
    "run_sweeps",
    "write_report",
]
