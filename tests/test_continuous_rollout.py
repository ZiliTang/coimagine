"""Tests for the continuous closed-loop rollout.

The finding is that a cold-start decision rule declines and therefore never gathers the
evidence that would change the decision -- the same structure as the necessity testbed.
"""

from __future__ import annotations

import pytest

from coimagine.diagnostics.threshold_probe import (
    format_continuous_rollout_markdown,
    run_continuous_episode,
    run_continuous_rollout,
)
from coimagine.grid.state import HiddenPhysics
from coimagine.learning.mlp import MultiHeadMLP


def test_episode_advances_the_object_one_cell_per_successful_push() -> None:
    physics = HiddenPhysics(1, "free", ("grip", "grip"))
    model = MultiHeadMLP(seed=0)
    success, cost, steps = run_continuous_episode(physics, "oracle", model, horizon=6)
    # the object starts two pushes from the goal, so the oracle needs two
    assert success == 1
    assert steps == 2
    # each push costs 1 + force, and the threshold force is the resistance
    assert cost == pytest.approx(2 * (1.0 + 1.0))


def test_a_slipping_robot_cannot_push_at_any_force() -> None:
    physics = HiddenPhysics(1, "free", ("slip", "grip"))
    model = MultiHeadMLP(seed=0)
    success, cost, _steps = run_continuous_episode(
        physics, "always_max", model, horizon=4
    )
    assert success == 0
    # it still paid for the attempt
    assert cost > 0


@pytest.mark.slow
def test_cold_start_declines_and_never_learns() -> None:
    """Locked finding: from a fresh scene the physics is unknown, the cost-optimal decision
    is to decline, and declining produces no evidence -- so the learned policy never attempts,
    while blind maximum force matches the oracle's success at twice the cost."""

    rows = {
        r["policy"]: r
        for r in run_continuous_rollout(
            n_train_situations=60, n_episodes=32, epochs=120, seed=0
        )
    }
    assert rows["oracle"]["success_rate"] > 0.0
    assert rows["learned"]["success_rate"] == pytest.approx(0.0)
    assert rows["learned"]["mean_cost"] == pytest.approx(0.0)
    # blind maximum force reaches the same success as the oracle, at a higher cost
    assert rows["always_max"]["success_rate"] == pytest.approx(
        rows["oracle"]["success_rate"]
    )
    assert rows["always_max"]["cost_per_success"] > rows["oracle"]["cost_per_success"]
    assert "cost per success" in format_continuous_rollout_markdown(
        list(rows.values())
    )
