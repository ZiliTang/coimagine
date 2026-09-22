"""Tests for the episode-level mode ablation.

The guard is that removing a contact mode from the *training data* measurably costs
closed-loop success on episodes containing that mode -- the episode-level version of the
question the learning probes asked per sample.
"""

from __future__ import annotations

import pytest

from coimagine.learning.data import generate_situations  # noqa: F401

from coimagine.diagnostics.rollout_probe import (
    format_mode_ablation_markdown,
    run_mode_ablation,
)


@pytest.mark.slow
def test_training_without_the_heavy_mode_costs_episode_success() -> None:
    """Locked finding: with the outcome-greedy policy (which has headroom, unlike the value
    policy that fails outright), removing heavy objects from training cuts success on heavy
    episodes several-fold."""

    summaries = run_mode_ablation(
        n_train_situations=100,
        n_episodes=100,
        epochs=250,
        seed=0,
        policy="learned_moved",
    )
    by = {(s.mode, s.model.split(" [")[0]): s for s in summaries}
    with_mode = by[("heavy", "trained_with")]
    without_mode = by[("heavy", "trained_without")]
    assert with_mode.success_rate > without_mode.success_rate
    assert "heavy" in format_mode_ablation_markdown(summaries)


@pytest.mark.slow
def test_latched_episodes_are_a_floor_for_the_greedy_policy() -> None:
    """The outcome-greedy policy never releases, so it cannot solve a latched object at all
    and the ablation has no headroom there. Recording the floor prevents the zero from being
    read as an ablation result."""

    summaries = run_mode_ablation(
        n_train_situations=60,
        n_episodes=40,
        epochs=150,
        seed=0,
        policy="learned_moved",
    )
    latched = [s for s in summaries if s.mode == "latched"]
    assert latched
    for summary in latched:
        assert summary.success_rate == pytest.approx(0.0)


@pytest.mark.slow
def test_training_without_the_constraint_mode_eliminates_latched_success() -> None:
    """Locked finding, and the sharpest ablation in the project.

    With a policy that *can* prepare -- progress plus enablement, the only learned rule that
    ever chooses release -- removing the latched mode from the training data takes success on
    latched episodes to zero, against roughly ten percent when the mode is present. The greedy
    motion policy cannot show this because it never releases, so it has a floor of zero either
    way (see the floor test above).
    """

    summaries = run_mode_ablation(
        n_train_situations=100,
        n_episodes=150,
        epochs=250,
        seed=0,
        policy="learned_progress_enable",
    )
    by = {(s.mode, s.model.split(" [")[0]): s for s in summaries}
    with_mode = by[("latched", "trained_with")]
    without_mode = by[("latched", "trained_without")]
    assert with_mode.success_rate > 0.0
    assert without_mode.success_rate == pytest.approx(0.0)
