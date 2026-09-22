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
def test_learning_the_value_beats_learning_the_outcome_score() -> None:
    """Locked finding of round 11, at the configuration where it was measured.

    Small configurations are too noisy for a strict ordering (at 60 training
    situations the value and outcome policies tie), so this deliberately uses the
    measured size rather than a token smoke configuration.
    """

    result = run_decision_probe(
        n_train_situations=120, n_test_situations=60, epochs=300, seed=0, depth=2
    )
    by_name = {s.name: s for s in result.summaries}
    outcome_policies = ("learned_moved", "learned_progress", "learned_progress_enable")
    reference = by_name["value_optimal"]
    # The equal-information reference is exact given its belief, so its regret is the
    # value of the missing contact physics, not zero.
    assert reference.mean_regret > 0.0
    assert reference.agreement > 0.4
    assert reference.mean_regret <= max(by_name[p].mean_regret for p in outcome_policies)
    # Learning the value must beat learning any per-action outcome score.
    assert by_name["learned_value"].agreement > max(
        by_name[p].agreement for p in outcome_policies
    )
    assert by_name["learned_value"].mean_regret < max(
        by_name[p].mean_regret for p in outcome_policies
    )


@pytest.mark.slow
def test_closed_loop_oracle_solves_solvable_episodes() -> None:
    """Guard for the released-flag bug: without it the oracle could not solve a latched
    episode at all, because it kept believing the object was still latched."""

    import numpy as np

    from coimagine.diagnostics.rollout_probe import run_episode
    from coimagine.learning.data import generate_situations
    from coimagine.learning.mlp import MultiHeadMLP
    from coimagine.grid.state import HiddenPhysics

    scene = None
    for situation in generate_situations(n_scenes=40, lead_in=0, seed=500000):
        if situation.physics == HiddenPhysics(1, "constrained", ("grip", "slip")):
            scene = situation.scene
            break
    assert scene is not None, "the generator must produce a releasable latched scene"
    model = MultiHeadMLP(seed=0)
    success, cost, steps = run_episode(
        scene, "oracle_true_state", model, horizon=8, depth=2,
        rng=np.random.default_rng(0),
    )
    assert success == 1
    assert cost == pytest.approx(3.0)
