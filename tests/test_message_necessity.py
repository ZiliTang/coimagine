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


def test_push_threshold_is_derived_from_the_declared_costs() -> None:
    """Attempting beats declining when p >= (1 + f + penalty) / (2 * penalty)."""

    from coimagine.diagnostics.message_necessity import _optimal_push_threshold

    assert _optimal_push_threshold(1.0) == pytest.approx(
        (PUSH_BASE_COST + 1.0 + FAIL_PENALTY) / (2.0 * FAIL_PENALTY)
    )
    # with the declared numbers the threshold sits between a coin flip and certainty
    assert 0.5 < _optimal_push_threshold(1.0) < 1.0


def test_necessity_samples_have_the_declared_shapes() -> None:
    from coimagine.diagnostics.message_necessity import necessity_samples
    from coimagine.learning.features import OBSERVATION_DIM
    from coimagine.learning.messages import message_dim

    for scheme in ("none", "traction"):
        samples = necessity_samples(n_scenes=8, message_scheme=scheme, seed=0)
        assert len(samples) == 8
        expected_obs = OBSERVATION_DIM + message_dim(scheme)
        for observation, action, label, grips in samples:
            assert len(observation) == expected_obs
            assert len(action) == 3
            assert label in (0, 1) and grips in (0, 1)
        # the label is exactly the teammate's traction in this construction
        assert all(label == grips for _o, _a, label, grips in samples)


@pytest.mark.slow
def test_learned_model_reaches_the_oracle_only_with_the_message() -> None:
    """The decisive finding: with one bit the learned model is exactly optimal; without it
    it is at chance, because the deciding fact is not observable through any other channel."""

    from coimagine.diagnostics.message_necessity import (
        format_learned_necessity_markdown,
        run_learned_necessity,
    )

    summaries = {
        s.scheme: s
        for s in run_learned_necessity(
            n_train_scenes=80, n_test_scenes=40, epochs=120, seed=0
        )
    }
    assert summaries["none"].balanced_accuracy == pytest.approx(0.5, abs=0.15)
    assert summaries["none"].mean_regret > 1.0
    assert summaries["traction"].balanced_accuracy > 0.9
    assert summaries["traction"].mean_regret == pytest.approx(0.0, abs=1e-9)
    assert summaries["traction"].success_rate > 0.3
    assert "implied by the declared costs" in format_learned_necessity_markdown(
        list(summaries.values())
    )


def test_the_message_equals_knowing_over_a_whole_episode() -> None:
    """A message policy reproduces the oracle exactly, at every horizon."""

    from coimagine.diagnostics.message_necessity import run_episode_message_value
    from collections import defaultdict

    summaries = run_episode_message_value(horizons=(1, 4), n_episodes=300, seed=0)
    groups = defaultdict(dict)
    for summary in summaries:
        groups[summary.horizon][summary.policy] = summary
    for horizon, group in groups.items():
        assert group["message"].mean_cost == pytest.approx(group["oracle"].mean_cost)
        assert group["message"].success_rate == pytest.approx(
            group["oracle"].success_rate
        )
        assert group["silent"].mean_cost > group["message"].mean_cost
        # silence recovers after one attempt, so its success is not lower, only costlier
        assert group["silent"].success_rate == pytest.approx(
            group["message"].success_rate
        ), horizon


def test_message_value_is_one_experiment_and_does_not_grow_with_the_horizon() -> None:
    """The value of the message is the cost of the single experiment it replaces.

    A silent team that experiments once learns the teammate's traction from the public
    outcome, so its loss is one wasted attempt no matter how long the task is. The value
    is therefore flat in the horizon, and only its *relative* share shrinks.
    """

    from coimagine.diagnostics.message_necessity import run_episode_message_value
    from collections import defaultdict

    summaries = run_episode_message_value(
        horizons=(1, 2, 4, 8), n_episodes=600, seed=0
    )
    groups = defaultdict(dict)
    for summary in summaries:
        groups[summary.horizon][summary.policy] = summary

    values = {}
    for horizon, group in groups.items():
        values[horizon] = group["silent"].mean_cost - group["message"].mean_cost
        # one attempt costs 2 and is wasted half the time
        assert values[horizon] == pytest.approx(1.0, abs=0.4), horizon
    # flat in the horizon rather than growing
    assert max(values.values()) - min(values.values()) < 0.6
    # while the relative share falls
    relative = {
        h: values[h] / groups[h]["message"].mean_cost for h in groups
    }
    assert relative[1] > relative[8]


def test_episode_markdown_reports_the_value() -> None:
    from coimagine.diagnostics.message_necessity import (
        format_episode_markdown,
        run_episode_message_value,
    )

    text = format_episode_markdown(
        run_episode_message_value(horizons=(1,), n_episodes=50, seed=0)
    )
    assert "Message value" in text
    assert "horizon 1" in text
