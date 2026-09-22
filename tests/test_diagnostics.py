"""The diagnostics runner and its outputs."""

from __future__ import annotations

import json

from coimagine.diagnostics import (
    DiagnosticConfig,
    format_gap_markdown,
    run_micro_diagnostics,
    write_report,
)


def test_diagnostics_cover_all_instances_and_report_gaps() -> None:
    config = DiagnosticConfig(horizons=(1,), bits=(0, 1), limited_search_depth=1)
    reports, summary = run_micro_diagnostics(config)

    assert [report.instance for report in reports] == [
        "M0",
        "M1",
        "M2",
        "M2-uninformative",
        "M2-expensive",
        "M3",
    ]
    assert summary["config"]["horizons"] == [1]
    assert summary["git_revision"]

    m1 = next(report for report in reports if report.instance == "M1")
    # Single-boundary ordering is asserted here as well as in the oracle tests.
    assert m1.gaps["H1.J_full"] <= m1.gaps["H1.J_central"] + 1e-9
    assert m1.gaps["H1.J_central"] <= m1.gaps["H1.J_B(bits=1)"] + 1e-9
    assert "J_central" in format_gap_markdown(reports)


def test_write_report_creates_artefacts(tmp_path) -> None:
    config = DiagnosticConfig(horizons=(1,), bits=(0, 1))
    reports, summary = run_micro_diagnostics(config)
    paths = write_report(reports, summary, str(tmp_path))

    for key in ("oracle_results_csv", "oracle_results_json", "oracle_results_md"):
        assert key in paths
    payload = json.loads((tmp_path / "oracle_results.json").read_text())
    assert payload["summary"]["package_version"]
    assert len(payload["instances"]) == 6
    assert (tmp_path / "oracle_gap_decomposition.json").exists()
