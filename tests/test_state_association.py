"""Tests for the state-association verification.

The mechanism tests use small synthetic models so they do not depend on a checkpoint; the
real-checkpoint run lives in ``scripts/state_association_probe.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.diagnostics.state_association import (
    DYNAMIC_DIM,
    ExactWorldChain,
    LearnedChain,
    _initial_worlds,
    default_chains,
    dynamic_vector_from_state,
    run_chain_probe,
)
from coimagine.grid.planner import PHYSICS
from coimagine.grid.state import HiddenPhysics, RobotState
from coimagine.learning.belief_model import N_PHYSICS
from coimagine.learning.data import pose_to_index, probe_scene


def _scene():
    physics = HiddenPhysics(resistance=1, constraint="constrained", traction=("grip", "grip"))
    return probe_scene((2, 2), physics)


def test_dynamic_vector_layout():
    vector = dynamic_vector_from_state((3, 2), True, (RobotState(0, (2, 2)), RobotState(1, (2, 3))))
    assert vector.shape == (1, DYNAMIC_DIM)
    assert vector[0, 0] == 1.0
    assert vector[0, 1 + pose_to_index((3, 2))] == 1.0
    assert vector[0, 1:37].sum() == 1.0
    assert vector[0, 37:41].tolist() == pytest.approx([0.4, 0.4, 0.4, 0.6])


def test_exact_chain_conserves_weight_and_matches_manual_enumeration():
    scene = _scene()
    chain = ExactWorldChain(scene.static, _initial_worlds(scene))
    total_before = sum(chain.worlds.values())
    assert total_before == pytest.approx(1.0)
    chain.step_exact("co_push:E")
    assert sum(chain.worlds.values()) == pytest.approx(1.0)
    # every world in the joint still carries its own parameter: 16 distinct parameters survive
    assert len({key[2] for key in chain.worlds}) == N_PHYSICS
    # and the released flag is still tied to the parameter, not flattened
    released = {}
    for (origin, robots, theta, rel), weight in chain.worlds.items():
        released.setdefault(theta, set()).add(rel)
    assert all(len(values) == 1 for values in released.values())


def test_mixed_propagation_is_exact_while_the_joint_is_a_product_and_loses_it_afterwards():
    """The release chain is the user's example: the association appears at step 1."""

    scene = _scene()
    exact = ExactWorldChain(scene.static, _initial_worlds(scene))
    mixed = ExactWorldChain(scene.static, _initial_worlds(scene))

    exact.step_exact("release:0:E")
    mixed.step_mixed("release:0:E")
    position_exact = _position(exact.worlds)
    position_mixed = _position(mixed.mixed_worlds)
    # the prior is a product of b(theta) and one world, so factorising changes nothing yet
    assert np.abs(position_exact - position_mixed).max() == pytest.approx(0.0)
    assert _released_by_theta(exact.worlds) == pytest.approx(_released_by_theta(mixed.mixed_worlds))

    exact.step_exact("co_push:E")
    mixed.step_mixed("co_push:E")
    # released now depends on theta; factorising recombines combinations that did not occur
    assert _released_by_theta(exact.worlds) != pytest.approx(_released_by_theta(mixed.mixed_worlds))


def test_exact_reference_keeps_released_conditioned_on_theta():
    scene = _scene()
    chain = ExactWorldChain(scene.static, _initial_worlds(scene))
    for action in ("release:0:E", "co_push:E"):
        chain.step_exact(action)
    theta = np.zeros(N_PHYSICS)
    released_joint = np.zeros(N_PHYSICS)
    for (origin, robots, index, rel), weight in chain.worlds.items():
        theta[index] += weight
        if rel:
            released_joint[index] += weight
    conditional = np.array(
        [released_joint[i] / theta[i] if theta[i] > 0 else 0.0 for i in range(N_PHYSICS)]
    )
    # every parameter either always released or never: the association is exact
    assert all(value in (0.0, 1.0) for value in conditional)
    assert conditional.max() == 1.0 and conditional.min() == 0.0


class _StubModel:
    """A transition whose released flag follows the parameter one-hot.

    ``step`` keeps the parameter association only when it is handed one: the released dim
    becomes the parameter's own bit and the mass moves to a per-parameter cell.  This is the
    smallest model that can distinguish the two propagation schemes.
    """

    def __init__(self) -> None:
        self.seen: list = []

    def dynamic_state(self, observation):
        return dynamic_vector_from_state((2, 2), False, (RobotState(0, (1, 2)), RobotState(1, (1, 3))))

    def step(self, dynamic, action, belief):
        dynamic = np.atleast_2d(dynamic)
        belief = np.atleast_2d(belief)
        rows = []
        for index in range(N_PHYSICS):
            row = np.zeros(DYNAMIC_DIM)
            row[0] = 1.0 if index % 2 == 0 else 0.0
            row[1 + (index % 36)] = 1.0
            rows.append(row)
        states = np.stack(rows)[None, :, :]
        self.seen.append(np.array(belief))
        return np.einsum("nt,ntd->nd", belief, states)


def test_per_theta_propagation_keeps_the_association_that_the_shipped_scheme_flattens():
    model = _StubModel()
    initial = model.dynamic_state(None)
    belief = np.full(N_PHYSICS, 1.0 / N_PHYSICS)
    shipped = LearnedChain(model, initial, belief)
    per_theta = LearnedChain(model, initial, belief)

    shipped.step_current(0)
    per_theta.step_per_theta(0)

    shipped_values = {round(float(v), 6) for v in shipped.current[0, 0:1]}
    # the shipped scheme has one released number, forced equal across parameters
    assert len(shipped_values) == 1
    per_theta_values = {round(float(row[0]), 6) for row in per_theta.per_theta}
    assert per_theta_values == {0.0, 1.0}


def test_chain_probe_reports_a_first_deviation_marker():
    class _IdentityModel(_StubModel):
        def step(self, dynamic, action, belief):
            return np.atleast_2d(dynamic)

    chain = default_chains()[1]
    result = run_chain_probe(_IdentityModel(), chain, tolerance=0.02)
    assert result["chain"] == "release_then_push"
    assert set(result["first_deviation"]) == {
        "exact_mixed",
        "learned_current",
        "learned_per_theta",
        "learned_single_step_true_theta",
        "learned_recursion_true_theta",
    }
    for step in result["steps"]:
        for key in ("reference", "exact_mixed", "learned_current", "learned_per_theta"):
            assert key in step
        assert 0.0 <= step["reference"]["released_probability"] <= 1.0
        assert abs(sum(step["reference"]["released_by_theta"]) / N_PHYSICS) <= 1.0


def _position(weights) -> np.ndarray:
    out = np.zeros(36)
    for (origin, robots, theta, released), weight in weights.items():
        out[pose_to_index(origin)] += weight
    return out


def _released_by_theta(weights) -> np.ndarray:
    out = np.zeros(N_PHYSICS)
    for (origin, robots, theta, released), weight in weights.items():
        if released:
            out[theta] += weight
    return out
