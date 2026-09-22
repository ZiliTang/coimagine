"""Stage 2 design gate: C1 room and C2 value on the grid abstraction.

The tests below do two things: they check the gate machinery and the oracle
ordering, and they **lock in the measured gate findings** so that any change to the
abstraction, the codecs or the oracles is flagged explicitly rather than silently
altering the conclusion.
"""

from __future__ import annotations

import json

import pytest

from coimagine.diagnostics import (
    GATE_CONFIGS,
    format_gate_markdown,
    run_grid_gate,
    write_gate_report,
)
from coimagine.toy.grid_abstraction import (
    ACTIONS,
    THETA_VALUES,
    X_VALUES,
    build_grid_abstraction,
    build_grid_split_abstraction,
)
from coimagine.toy.model import validate_model

TOL = 1e-9


def test_gate_models_validate_and_differ_from_the_micro_instances() -> None:
    same_face = build_grid_abstraction(horizon=3)
    split = build_grid_split_abstraction(horizon=3)
    for model in (same_face, split):
        validate_model(model)
        assert len(model.theta_values) == 6, "the gate needs more than two hidden types"
        assert model.probe_actions, "the gate needs an explicit probe action"
    # Four local feedback symbols: one bit is genuinely lossy.
    symbols = {o for row in same_face.observations.values() for (pair, _y) in row for o in pair}
    assert len(symbols) == 4


def test_gate_reports_and_writes_files(tmp_path) -> None:
    reports, summary = run_grid_gate()
    assert len(reports) == len(GATE_CONFIGS)
    assert summary["git_revision"]
    paths = write_gate_report(reports, summary, str(tmp_path))
    payload = json.loads((tmp_path / "grid_gate.json").read_text())
    assert len(payload["configs"]) == len(GATE_CONFIGS)
    assert "C1 room" in format_gate_markdown(reports)


def test_gate_oracle_ordering_at_one_boundary() -> None:
    reports, _summary = run_grid_gate()
    for report in reports:
        gaps = report.gaps
        assert gaps["H1.J_full"] <= gaps["H1.J_central"] + TOL, report.config
        assert gaps["H1.J_central"] <= gaps["H1.J_B(bits=1)"] + TOL, report.config
        assert gaps["H1.J_B(bits=1)"] <= gaps["H1.J_B(bits=0)"] + TOL, report.config


def test_gate_finding_c1_has_room_at_a_multi_step_horizon() -> None:
    """Locked finding: unlike the micro instances, the message representation matters.

    In the same-face configuration the naive compact posterior is strictly worse
    than a better one-bit representation once the horizon is longer than one
    boundary.
    """

    reports, _summary = run_grid_gate()
    by_config = {report.config: report for report in reports}
    assert by_config["same_face_blind"].gaps["H3.c1_room"] > TOL
    assert by_config["same_face_informed"].gaps["H3.c1_room"] > TOL
    # At a single boundary there is no room, exactly as in Stage 1.
    for report in reports:
        assert abs(report.gaps["H1.message_representation_loss"]) < TOL


def test_gate_finding_c2_has_no_value_in_this_structure() -> None:
    """Locked finding: anticipating the next message never changes the first decision.

    Mechanism: pushing and repositioning are simultaneously the physically right and
    the informative actions, and the re-planning control uses the real messages after
    execution. C2 would need a decision whose only justification is the information
    it will produce, or a costly wrong attempt.
    """

    reports, _summary = run_grid_gate()
    for report in reports:
        for codec in ("null", "observation", "posterior-mode", "consequence"):
            key = f"H3.c2_value[{codec}]"
            assert abs(report.gaps[key]) < TOL, (report.config, codec)
