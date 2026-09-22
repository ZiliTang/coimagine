"""Tests for the error analysis.

The analysis is pure aggregation over decision records, so it is tested on hand-built
records: that keeps the test fast and makes the expected rates checkable by hand.
"""

from __future__ import annotations

import pytest

from coimagine.diagnostics.decision_probe import DecisionRecord
from coimagine.diagnostics.error_analysis import (
    ErrorAnalysis,
    _short_mechanism,
    analyse_errors,
    format_error_markdown,
    render_error_figure,
)


def _record(
    reference: str,
    learned: str,
    *,
    constrained: int = 0,
    heavy: int = 0,
    slips: int = 0,
    distance: int = 2,
    scene_id: int = 0,
) -> DecisionRecord:
    return DecisionRecord(
        scene_id=scene_id,
        true_physics="test",
        oracle_action=reference,
        oracle_value=1.0,
        chosen={"value_optimal": reference, "learned_value": learned},
        regret={"value_optimal": 0.0, "learned_value": 1.0},
        teammate_slips=slips,
        constrained=constrained,
        heavy=heavy,
        distance_to_goal=distance,
    )


def test_mechanism_label_is_built_from_the_flags() -> None:
    assert _record("wait", "wait").mechanism == "plain"
    assert _record("wait", "wait", constrained=1).mechanism == "latched"
    assert (
        _record("wait", "wait", constrained=1, heavy=1, slips=1).mechanism
        == "latched+heavy+teammate_slips"
    )


def test_short_mechanism_is_compact() -> None:
    assert _short_mechanism("latched+heavy+teammate_slips") == "L+H+S"
    assert _short_mechanism("plain") == "plain"


def test_error_rates_are_hand_checkable() -> None:
    records = [
        _record("push:0:E", "push:0:E"),  # agrees, plain
        _record("push:0:E", "wait"),  # disagrees, plain, wait fallback
        _record("co_push:E", "push:1:E", slips=1),  # disagrees
        _record("co_push:E", "co_push:E", slips=1),  # agrees
        _record("release:0:E", "co_push:E", constrained=1),  # disagrees
    ]
    analysis = analyse_errors(records)
    assert isinstance(analysis, ErrorAnalysis)
    assert analysis.n == 5
    assert analysis.overall_disagreement == pytest.approx(3 / 5)
    assert analysis.by_mechanism["plain"] == pytest.approx(1 / 2)
    assert analysis.by_mechanism["teammate_slips"] == pytest.approx(1 / 2)
    assert analysis.by_mechanism["latched"] == pytest.approx(1.0)
    assert sum(analysis.by_mechanism_n.values()) == 5
    # the model picked the free "wait" once while the reference acted
    assert analysis.wait_fallbacks == 1
    assert analysis.learned_action_when_wrong["push:1:E"] == 1
    assert "push:0:E -> wait" in analysis.confusion


def test_distance_breakdown_partitions_the_records() -> None:
    records = [
        _record("push:0:E", "wait", distance=1),
        _record("push:0:E", "push:0:E", distance=1),
        _record("push:0:E", "wait", distance=3),
    ]
    analysis = analyse_errors(records)
    assert analysis.by_distance_n == {1: 2, 3: 1}
    assert analysis.by_distance[1] == pytest.approx(0.5)
    assert analysis.by_distance[3] == pytest.approx(1.0)


def test_markdown_and_figure_are_produced(tmp_path) -> None:
    records = [
        _record("push:0:E", "wait", distance=1),
        _record("co_push:E", "co_push:E", slips=1, distance=2),
    ]
    analysis = analyse_errors(records)
    text = format_error_markdown(analysis)
    assert "By contact mechanism" in text
    assert "distance" in text
    paths = render_error_figure(analysis, str(tmp_path / "figure"))
    assert paths and paths[0].endswith(".png")
