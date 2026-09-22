"""Decision-level probe tests.

The probe's value is that it separates two different failures:

* a **decision rule** that is wrong for the task, measured by the *matched analytic*
  rule that uses the same rule with the correct model;
* a **model** that is wrong, measured as the extra regret the learned model incurs
  under the identical rule.

These tests guard that separation, the honesty of the value computation, and the
negative finding of round 10: making the rule more task-shaped (progress, enablement)
did not rescue it.
"""

from __future__ import annotations

import pytest

from coimagine.diagnostics.decision_probe import (
    POLICIES,
    posterior_over_physics,
    run_decision_probe,
)
from coimagine.grid.dynamics import FeedbackNoise
from coimagine.grid.planner import PHYSICS
from coimagine.learning.data import generate_situations


def test_posterior_keeps_the_truth_and_is_informative() -> None:
    """The belief must never exclude the true physics, and must beat the prior.

    Note what is *not* asserted: a point mass. With a two-step noiseless lead-in the
    posterior is usually diffuse over about three hypotheses (measured median maximum
    mass 1/3), because plenty of contact types produce identical feedback for the
    actions that happened to be taken. Demanding certainty here would be wrong.
    """

    situations = generate_situations(n_scenes=16, lead_in=2, seed=0)
    masses = []
    for situation in situations:
        posterior = posterior_over_physics(situation, noise=FeedbackNoise())
        assert sum(posterior.values()) == pytest.approx(1.0)
        true_index = PHYSICS.index(situation.physics)
        assert posterior[true_index] == pytest.approx(max(posterior.values()))
        masses.append(posterior[true_index])
    assert sum(masses) / len(masses) > 1.0 / 16


def test_no_policy_beats_the_oracle_it_is_measured_against() -> None:
    """Regret is measured against the true-model argmin, so it cannot be negative."""

    result = run_decision_probe(
        n_train_situations=12, n_test_situations=8, epochs=40, seed=0
    )
    for record in result.records:
        for policy in POLICIES:
            assert record.regret[policy] >= -1e-9, (policy, record.scene_id)
        # Every policy's action is one of the candidates, and the oracle's value is the
        # minimum over them by construction.
        assert record.oracle_action in record.chosen.values()


@pytest.mark.slow
def test_the_value_optimal_rule_is_exact_and_the_greedy_rules_are_not() -> None:
    """Locked finding: at equal information the belief-weighted value rule is exact
    here, and no greedy rule over per-action outcome scores comes close -- including
    the ones reshaped around progress and enablement."""

    result = run_decision_probe(
        n_train_situations=60, n_test_situations=30, epochs=120, seed=0
    )
    by_name = {s.name: s for s in result.summaries}
    assert by_name["value_optimal"].mean_regret == pytest.approx(0.0, abs=1e-9)
    assert by_name["value_optimal"].agreement == pytest.approx(1.0)
    for policy in POLICIES:
        if policy == "value_optimal":
            continue
        assert by_name[policy].mean_regret > 0.0, policy
