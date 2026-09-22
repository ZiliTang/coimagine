"""Tests for the message-necessity testbed.

The point of this testbed is that the teammate is *necessary*: this robot slips, so a solo
push is impossible at any force, and the outcome depends entirely on a private fact held by
the teammate. These tests pin down that construction, because if it silently degrades back
to "the teammate is merely cheaper" the whole measurement becomes vacuous.
"""

from __future__ import annotations

import pytest

from coimagine.diagnostics.message_necessity import (
    FAIL_PENALTY,
    PUSH_BASE_COST,
    format_necessity_markdown,
    necessity_scene,
    run_necessity_comparison,
)
from coimagine.grid.continuous import FORCE_MAX, push_outcome


def test_solo_push_is_impossible_but_co_push_works_when_the_teammate_grips() -> None:
    scene = necessity_scene(teammate_grip=True)
    # this robot slips, so it contributes nothing at any force
    assert not push_outcome(scene, {0: FORCE_MAX}, 0.0).moved
    assert push_outcome(scene, {0: 1.0, 1: 1.0}, 0.0).moved


def test_co_push_fails_when_the_teammate_slips() -> None:
    scene = necessity_scene(teammate_grip=False)
    assert not push_outcome(scene, {0: FORCE_MAX}, 0.0).moved
    assert not push_outcome(scene, {0: FORCE_MAX, 1: FORCE_MAX}, 0.0).moved


def test_failure_penalty_exceeds_the_cost_of_success() -> None:
    """Otherwise declining a winnable task breaks even with winning it.

    An earlier value of 2.0 was exactly the cost of a successful co-push, which made
    "do nothing" as good as "succeed" and hid the value of the message entirely.
    """

    successful_cost = PUSH_BASE_COST + 1.0
    assert FAIL_PENALTY > successful_cost


def test_message_is_exactly_optimal_and_silence_is_not() -> None:
    result = run_necessity_comparison(n_scenes=120, seed=0)
    by_name = {s.name: s for s in result.summaries}
    assert by_name["oracle"].mean_regret == pytest.approx(0.0)
    # one bit reproduces the oracle exactly
    assert by_name["one_bit"].mean_regret == pytest.approx(0.0)
    assert by_name["one_bit"].success_rate == pytest.approx(
        by_name["oracle"].success_rate
    )
    # silence is strictly worse, and it loses by declining winnable tasks
    assert by_name["silent"].mean_regret > 0.0
    assert by_name["silent"].success_rate == pytest.approx(0.0)
    assert result.message_value > 0.0
    assert result.success_gain > 0.0


def test_the_two_teammate_cases_split_the_loss() -> None:
    """The silent policy's loss is concentrated where the teammate would have gripped."""

    result = run_necessity_comparison(n_scenes=120, seed=0)
    grips = {s.name: s for s in result.by_teammate["teammate_grips"]}
    slips = {s.name: s for s in result.by_teammate["teammate_slips"]}
    assert grips["silent"].mean_regret > 0.0
    assert grips["one_bit"].mean_regret == pytest.approx(0.0)
    assert slips["silent"].mean_regret == pytest.approx(0.0)
    assert slips["one_bit"].mean_regret == pytest.approx(0.0)
    # and always co-pushing is the worst policy where the teammate slips
    assert slips["always_co"].mean_regret > grips["always_co"].mean_regret


def test_report_mentions_the_message_value() -> None:
    result = run_necessity_comparison(n_scenes=40, seed=0)
    text = format_necessity_markdown(result)
    assert "Value of one bit" in text
    assert "necessary" in text
