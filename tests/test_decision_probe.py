"""Decision-level probe tests.

The probe's whole value is that it separates two different failures:

* a **decision rule** that is wrong for the task (greedy feasibility instead of
  value-optimal planning), measured by the analytic-greedy baseline;
* a **model** that is wrong, measured as the extra regret the learned model incurs
  under the identical rule.

These tests guard that separation and the honesty of the value computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.diagnostics.decision_probe import (
    posterior_over_physics,
    run_decision_probe,
)
from coimagine.grid.dynamics import FeedbackNoise
from coimagine.learning.data import generate_situations


def test_posterior_keeps_the_truth_and_is_informative() -> None:
    """The belief must never exclude the true physics, and must beat the prior.

    Note what is *not* asserted: a point mass. With a two-step noiseless lead-in the
    posterior is usually diffuse over about three hypotheses (measured median max
    mass 1/3), because plenty of contact types produce identical feedback for the
    actions that happened to be taken. Demanding certainty here would be wrong.
    """

    from coimagine.grid.planner import PHYSICS

    situations = generate_situations(n_scenes=16, lead_in=2, seed=0)
    masses = []
    for situation in situations:
        posterior = posterior_over_physics(situation, noise=FeedbackNoise())
        assert sum(posterior.values()) == pytest.approx(1.0)
        true_index = PHYSICS.index(situation.physics)
        assert posterior[true_index] == pytest.approx(max(posterior.values()))
        masses.append(posterior[true_index])
    assert sum(masses) / len(masses) > 1.0 / 16


def test_value_of_the_oracle_action_is_the_minimum() -> None:
    """Regret is measured against the true-model argmin, so it cannot be negative."""

    result = run_decision_probe(
        n_train_situations=12, n_test_situations=8, epochs=40, seed=0
    )
    for record in result.records:
        assert record.analytic_regret >= -1e-9
        assert record.learned_regret >= -1e-9
        assert record.analytic_greedy_regret >= -1e-9
        assert record.oracle_value <= record.analytic_value + 1e-9
        assert record.oracle_value <= record.learned_value + 1e-9


@pytest.mark.slow
def test_analytic_policy_beats_the_learned_greedy_rule() -> None:
    """Locked finding: at equal information the analytic rule is optimal here, and
    the learned feasibility head used greedily is far behind it."""

    result = run_decision_probe(
        n_train_situations=60, n_test_situations=30, epochs=120, seed=0
    )
    assert result.overall.mean_analytic_regret == pytest.approx(0.0, abs=1e-9)
    assert result.overall.analytic_agreement == pytest.approx(1.0)
    # The learned greedy rule must be worse than the matched analytic greedy rule.
    assert result.overall.mean_learned_regret > result.overall.analytic_greedy_regret
