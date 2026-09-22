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
        # Every policy picks from the same candidate template, and regret is measured
        # against the true-model argmin, so it cannot be negative. (An earlier version of
        # this test also asserted that some policy reproduced the oracle's action; that is
        # not an invariant -- the oracle may pick an action no policy chooses.)
        from coimagine.learning.features import ACTION_IDS

        assert record.oracle_action in ACTION_IDS
        assert set(record.chosen.values()) <= set(ACTION_IDS)


@pytest.mark.slow
def test_the_value_target_gives_the_best_agreement_but_not_the_best_regret() -> None:
    """Corrected finding, after removing the private-feedback leak.

    Round 11 reported that learning the value beat learning every per-action outcome score
    on both metrics. That held only because the encoder was fed the *teammate's private
    feedback*, which the protocol forbids. With the leak removed and only the public event
    history available, the value target still gives the best agreement with the reference
    among the learned policies, but it no longer gives the best regret: its errors are
    rarer but larger. The test asserts exactly what reproduces.
    """

    result = run_decision_probe(
        n_train_situations=150, n_test_situations=80, epochs=350, seed=0, depth=2
    )
    by_name = {s.name: s for s in result.summaries}
    outcome_policies = ("learned_moved", "learned_progress", "learned_progress_enable")
    reference = by_name["value_optimal"]
    assert reference.mean_regret > 0.0
    assert reference.agreement > 0.4
    assert reference.mean_regret < min(
        by_name[p].mean_regret for p in outcome_policies + ("learned_value",)
    )
    # the robust part: the value target agrees with the reference most often
    assert by_name["learned_value"].agreement > max(
        by_name[p].agreement for p in outcome_policies
    )
