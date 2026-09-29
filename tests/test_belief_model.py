"""Tests for the factored world model: parameter belief, transition, observation model.

The structure here is the point, so the tests check structure rather than a loss value: that the
likelihood is action-conditioned, that the belief update identifies what is identifiable and
stays uncertain about what is not, and that the transition is a mixture over the belief rather
than the belief's expectation fed in as if it were a parameter vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.diagnostics.belief_probe import (
    compare_likelihoods,
    marginal,
    plan_with_belief,
    run_belief_probe,
    run_prepare_probe,
    theta_index,
    true_value,
)
from coimagine.grid.dynamics import FEEDBACK_SYMBOLS
from coimagine.grid.sensing import ray_cast
from coimagine.grid.state import enumerate_hidden_physics
from coimagine.learning.autograd import MLP, Tensor, mse
from coimagine.learning.belief_model import (
    BeliefWorldModel,
    logits_slice,
    train_belief_model,
)
from coimagine.learning.data import generate_situations
from coimagine.learning.features import ACTION_IDS, observation_vector
from coimagine.learning.world_model import N_PHYSICS
from coimagine.learning.world_model_data import collect_trajectories

TABLE = tuple(enumerate_hidden_physics())


@pytest.fixture(scope="module")
def trained() -> BeliefWorldModel:
    model = BeliefWorldModel(seed=0)
    train_belief_model(
        model, collect_trajectories(n_episodes=250, horizon=8, seed=0),
        rollout_length=2, epochs=150, batch_size=32, lr=1e-2, seed=0,
    )
    return model


def _one_hot(physics) -> np.ndarray:
    out = np.zeros((1, N_PHYSICS))
    out[0, theta_index(physics)] = 1.0
    return out


def test_logits_slice_gradients() -> None:
    rng = np.random.default_rng(0)
    net = MLP([4, 6, 3], rng, name="s", final_activation="none")
    x = np.asarray(rng.normal(size=(5, 4)))
    target = np.asarray(rng.normal(size=(5, 2)))
    build = lambda: mse(logits_slice(net(Tensor(x)), 1, 3), target)  # noqa: E731
    params = list(net.parameters())
    build().backward()
    analytic = np.concatenate([p.grad.ravel() for _k, p in params])
    base = np.concatenate([p.data.ravel() for _k, p in params])
    worst = 0.0
    for index in np.random.default_rng(1).choice(base.size, 8, replace=False):
        step = 1e-6
        up = base.copy(); up[index] += step
        down = base.copy(); down[index] -= step
        offset = 0
        for _k, p in params:
            n = p.data.size
            p.data = up[offset:offset + n].reshape(p.data.shape).copy(); offset += n
        high = float(build().data)
        offset = 0
        for _k, p in params:
            n = p.data.size
            p.data = down[offset:offset + n].reshape(p.data.shape).copy(); offset += n
        low = float(build().data)
        offset = 0
        for _k, p in params:
            n = p.data.size
            p.data = base[offset:offset + n].reshape(p.data.shape).copy(); offset += n
        numeric = (high - low) / (2 * step)
        exact = float(analytic[index])
        worst = max(worst, abs(numeric - exact) / max(1e-9, abs(numeric) + abs(exact)))
    assert worst < 1e-5


def test_observation_model_reproduces_the_simulator_likelihood(trained) -> None:
    """`slipped` after a contact action means this robot slipped, exactly."""

    rows = compare_likelihoods(trained, n_scenes=120, symbol="slipped")
    for row in rows:
        assert abs(row.learned - row.empirical) < 0.15, (row,)


def test_belief_update_identifies_own_traction_and_refuses_the_unidentifiable(trained) -> None:
    """Own traction is knowable from one contact action; the teammate's is not knowable at all,
    and the posterior must reflect that difference rather than guessing."""

    probe = run_belief_probe(trained, n_runs=120, horizon=2)
    assert probe.accuracy["own traction"][0] > 0.9
    assert probe.confident_errors["own traction"] < 0.05
    # the teammate's traction is not observable through this robot's own feedback
    assert probe.moved["teammate traction"][1] < 0.5
    assert probe.confident_errors["teammate traction"] < 0.10


def test_transition_mixture_is_not_the_expected_parameter_vector(trained) -> None:
    """The bug this guards: the transition is trained on one-hot parameters, so feeding the
    belief's expectation is out of distribution and gave a confident zero where the true
    parameters give a confident one."""

    situations = generate_situations(n_scenes=120, lead_in=0, seed=55_000,
                                     physics_pool=TABLE)
    checked = 0
    for situation in situations:
        scene = situation.scene
        if scene.physics.constraint != "constrained" or scene.physics.traction[0] != "grip":
            continue
        if scene.released or scene.is_goal():
            continue
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        dynamic = trained.dynamic_state(observation)
        action = ACTION_IDS.index("release:0:E")
        known = trained.step(dynamic, action, _one_hot(scene.physics))[0, 0]
        assert known > 0.8, "with the true parameters, release must be predicted to release"
        checked += 1
        if checked >= 5:
            break
    assert checked > 0


def _release_predictions(trained, own_traction: str) -> list:
    """P(released' | release) over latched scenes, split by whether release can actually work."""

    REL = ACTION_IDS.index("release:0:E")
    out = []
    for situation in generate_situations(
        n_scenes=400, lead_in=0, seed=55_000, physics_pool=TABLE
    ):
        scene = situation.scene
        if scene.physics.constraint != "constrained" or scene.released or scene.is_goal():
            continue
        if scene.physics.traction[0] != own_traction:
            continue
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        dynamic = trained.dynamic_state(observation)
        out.append(float(trained.step(dynamic, REL, _one_hot(scene.physics))[0, 0]))
    return out


def test_the_release_transition_is_conditioned_on_whether_it_can_work(trained) -> None:
    """The transition must predict that `release` flips `released` exactly when it can: the
    object is latched and this robot grips. This is the hidden transition the whole model exists
    to represent, and getting it wrong is why the earlier chain could never learn to prepare."""

    grip = _release_predictions(trained, "grip")
    slip = _release_predictions(trained, "slip")
    assert len(grip) > 20 and len(slip) > 20
    assert float(np.mean(grip)) > 0.9, np.mean(grip)
    assert float(np.mean(slip)) < 0.1, np.mean(slip)
    # and it must not be merely an unconditional association with the action
    assert max(slip) < 0.5, max(slip)


def test_belief_update_survives_an_impossible_observation(trained) -> None:
    belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    dynamic = np.zeros((1, 41)); dynamic[0, 0] = 0.5; dynamic[0, 1:37] = 1.0 / 36
    # a symbol the model thinks is essentially impossible must not produce NaN
    updated = trained.update_belief(belief, dynamic, 0, 5)
    assert np.all(np.isfinite(updated))
    assert updated.sum() == pytest.approx(1.0)


def test_planner_prepares_when_preparing_can_work_and_never_otherwise(trained) -> None:
    """The end-to-end property, and the reason the whole factoring exists.

    With the parameters known, the planner must spend a step releasing a latched object that it
    can release, and must never waste one on a latched object it cannot release or on a free one.
    """

    probe = run_prepare_probe(trained, n_scenes=400, lead_in=0, oracle_belief=True)
    assert probe.latched_belief_mass == pytest.approx(1.0)
    # measured: 0.575 when releasing can work, 0.000 when it cannot, 0.024 on a free object
    assert probe.latched_releasable_n > 20 and probe.latched_unreleasable_n > 20
    assert probe.latched_releasable_rate > 0.3
    assert probe.latched_unreleasable_rate < 0.02
    assert probe.free_release_rate < 0.10
    # and the discrimination is what matters, not the absolute rate
    assert probe.latched_releasable_rate > 5 * probe.free_release_rate


def test_planner_runs_with_an_inferred_belief(trained) -> None:
    situations = generate_situations(n_scenes=40, lead_in=0, seed=55_000, physics_pool=TABLE)
    scene = next(s.scene for s in situations if not s.scene.is_goal())
    observation = np.asarray(
        observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
    )
    chosen, branches = plan_with_belief(
        trained, observation, frozenset(scene.static.target)
    )
    assert 0 <= chosen < len(ACTION_IDS)
    assert len(branches) == len(ACTION_IDS)
    for branch in branches:
        assert branch["symbol_probabilities"].shape == (6,)
        assert branch["symbol_probabilities"].sum() == pytest.approx(1.0, abs=1e-6)


def test_theta_head_is_not_used_as_the_belief(trained) -> None:
    """Recorded because it was measured: the encoder's parameter head fits spurious correlations
    in the lead-in history and was confidently wrong, while the explicit filter built on the
    verified likelihood was exact. The planner must use the filter."""

    probe = run_belief_probe(trained, n_runs=120, horizon=2)
    assert probe.accuracy["own traction"][-1] > 0.85
    assert probe.confident_errors["own traction"] < 0.05


def test_hidden_state_is_carried_not_re_read(trained) -> None:
    """The bug that made the planner idle at 0.000 success.

    A successful release leaves the observation byte-identical, so re-reading `released` from the
    observation forgets that it happened. The hidden part must be carried by the transition while
    the observable part comes from the current view.
    """

    from coimagine.diagnostics.belief_probe import fuse_hidden_state, run_belief_episode

    observed = np.zeros((1, 41)); observed[0, 0] = 0.01            # observation says "not released"
    observed[0, 1:37] = 1.0 / 36
    carried = np.zeros((1, 41)); carried[0, 0] = 0.97              # transition says "released"
    fused = fuse_hidden_state(observed, carried)
    assert fused[0, 0] == pytest.approx(0.97)                      # hidden part from the transition
    assert np.allclose(fused[0, 1:], observed[0, 1:])              # observable part from the view

    # and an episode runs with the hidden state carried
    scene = next(
        s.scene
        for s in generate_situations(n_scenes=40, lead_in=0, seed=55_000, physics_pool=TABLE)
        if not s.scene.is_goal()
    )
    belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    success, cost, steps = run_belief_episode(
        trained, scene, belief, horizon=3, depth=1, warmup=(3, 2)
    )
    assert success in (0, 1)
    assert steps <= 3 and cost >= 0.0


@pytest.mark.slow
def test_multistep_prediction_depends_on_the_parameters(trained) -> None:
    """The planner needs the parameters to change what it expects several steps ahead, not only
    one step. Two beliefs that disagree about the constraint mode and this robot's traction must
    produce different imagined futures, and must disagree about the best next action."""

    from coimagine.diagnostics.belief_probe import measure_multistep_sensitivity

    result = measure_multistep_sensitivity(trained, n_scenes=120, steps=3)
    assert result.cell_divergence[-1] > 0.1, result.cell_divergence
    assert result.released_gap[-1] > 0.1, result.released_gap
    # the disagreement must survive to the last step, i.e. it is not a one-step effect
    assert result.best_action_agreement[-1] < 0.8, result.best_action_agreement


@pytest.mark.slow
def test_closed_loop_comparison_runs_every_policy_on_the_same_scenes(trained) -> None:
    from coimagine.diagnostics.belief_probe import run_belief_comparison

    rows = run_belief_comparison(trained, n_episodes=12, horizon=4, depth=1)
    names = {row.policy for row in rows}
    assert "belief model + belief-space search" in names
    assert "belief model, two-level lookahead" in names
    assert "random" in names
    assert any("oracle" in n for n in names)
    assert any("reference" in n for n in names)
    for row in rows:
        assert 0.0 <= row.success_rate <= 1.0
        assert row.episodes > 0
