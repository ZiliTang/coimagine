"""Model tables and exact Bayes beliefs."""

from __future__ import annotations

import dataclasses

import pytest

from coimagine.exact import (
    PublicEntry,
    initial_public_beliefs,
    robot_posterior,
    theta_posterior,
)
from coimagine.exact.codec import NullCodec
from coimagine.toy import (
    O_LOADED,
    O_SLIPPING,
    TH_HEAVY,
    TH_SLIP,
    all_instances,
    build_m1,
    validate_model,
)
from coimagine.toy.instances import A_CO_PUSH, _joint_local
from coimagine.toy.model import ModelValidationError


def test_all_instances_validate() -> None:
    for model in all_instances(horizon=3):
        validate_model(model)


def test_observation_rows_are_normalised_joint_tables() -> None:
    for model in all_instances(horizon=3):
        for key, row in model.observations.items():
            total = sum(row.values())
            assert abs(total - 1.0) < 1e-9, (model.name, key, total)
            assert all(p >= 0.0 for p in row.values())


def test_local_observations_are_correlated_not_independent() -> None:
    """The published joint table must not silently factor."""

    for strength in ("weak", "strong"):
        table = _joint_local(TH_HEAVY, strength)
        p_both = table[(O_LOADED, O_LOADED)]
        p_a = table[(O_LOADED, O_LOADED)] + table[(O_LOADED, O_SLIPPING)]
        p_b = table[(O_LOADED, O_LOADED)] + table[(O_SLIPPING, O_LOADED)]
        assert abs(p_both - p_a * p_b) > 1e-6, (
            f"{strength} table is accidentally independent; the instance would "
            "not test joint observations"
        )


def test_negative_probability_is_rejected() -> None:
    model = build_m1(horizon=1)
    transitions = dict(model.transitions)
    transitions[(model.initial_x, TH_HEAVY, A_CO_PUSH)] = {model.goal_x: 1.1}
    broken = dataclasses.replace(model, transitions=transitions)
    with pytest.raises(ModelValidationError):
        validate_model(broken)


def test_bayes_matches_hand_calculation() -> None:
    """Robot A observes ``loaded`` in M1: P(HEAVY | loaded) = 0.65 exactly."""

    model = build_m1(horizon=1)
    posterior = robot_posterior(model, 0, (O_LOADED,), ("stalled",), ())
    assert posterior[(model.initial_x, TH_HEAVY)] == pytest.approx(0.65, abs=1e-12)
    assert posterior[(model.initial_x, TH_SLIP)] == pytest.approx(0.35, abs=1e-12)

    # And the mirror image.
    posterior_slip = robot_posterior(model, 0, (O_SLIPPING,), ("stalled",), ())
    assert posterior_slip[(model.initial_x, TH_HEAVY)] == pytest.approx(0.35, abs=1e-12)


def test_impossible_observation_is_handled_without_nan() -> None:
    model = build_m1(horizon=1)
    posterior = robot_posterior(model, 0, ("not-a-real-observation",), ("stalled",), ())
    assert posterior == {}


def test_private_posteriors_differ_from_joint_and_are_not_multiplied() -> None:
    """A single robot's belief is not the joint belief; the filter keeps them apart."""

    model = build_m1(horizon=1)
    joint_a = robot_posterior(model, 0, (O_LOADED,), ("stalled",), ())
    joint_b = robot_posterior(model, 1, (O_LOADED,), ("stalled",), ())
    # Robot A's marginal and robot B's marginal are different objects; their
    # product is never used. The public belief is built from the joint table.
    assert joint_a[(model.initial_x, TH_HEAVY)] == pytest.approx(0.65, abs=1e-12)
    assert joint_b[(model.initial_x, TH_HEAVY)] == pytest.approx(0.65, abs=1e-12)

    beliefs = initial_public_beliefs(model, "stalled", NullCodec(), model.actions)
    total = sum(prob for _belief, prob in beliefs.values())
    assert total == pytest.approx(1.0, abs=1e-12)
    entries: list[PublicEntry] = []
    for belief, _prob in beliefs.values():
        entries.extend(belief.keys())
    # Every entry carries both private histories, i.e. the public state tracks
    # more than P(x, theta | C).
    assert all(len(entry[2]) == 1 and len(entry[3]) == 1 for entry in entries)


def test_theta_posterior_normalises() -> None:
    model = build_m1(horizon=1)
    posterior = robot_posterior(model, 0, (O_LOADED,), ("stalled",), ())
    theta = theta_posterior(posterior)
    assert sum(theta.values()) == pytest.approx(1.0, abs=1e-12)
