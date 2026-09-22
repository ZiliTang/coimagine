"""Oracle hierarchy, ordering, message value and probe-choice properties."""

from __future__ import annotations

import pytest

from coimagine.exact import (
    ConsequenceCodec,
    NullCodec,
    ObservationCodec,
    PosteriorModeCodec,
    PosteriorThresholdCodec,
    o_b_exact,
    o_central,
    o_central_root_actions,
    o_codec,
    o_full,
    true_model_limited_search,
)
from coimagine.toy import all_instances, build_m0, build_m1, build_m2, build_m3

TOL = 1e-9


def test_single_boundary_oracle_ordering() -> None:
    """J_full* <= J_central* <= J_B* <= J_codec* at a matched horizon of 1."""

    for model in all_instances(horizon=1):
        full = o_full(model).expected_cost
        central = o_central(model).expected_cost
        o_b = o_b_exact(model, 1).expected_cost
        assert full <= central + TOL, model.name
        assert central <= o_b + TOL, model.name
        for codec in (
            NullCodec(),
            ObservationCodec(bits=1),
            PosteriorModeCodec(bits=1),
            ConsequenceCodec(bits=1),
            PosteriorThresholdCodec(threshold=0.8),
        ):
            value = o_codec(model, codec).expected_cost
            assert o_b <= value + TOL, (model.name, codec.name)


def test_increasing_bits_does_not_increase_exact_o_b() -> None:
    """Nested message alphabets: more bits can never hurt the exact O_B."""

    for model in all_instances(horizon=1):
        values = [o_b_exact(model, bits).expected_cost for bits in (0, 1, 2)]
        assert values[0] >= values[1] - TOL, model.name
        assert values[1] >= values[2] - TOL, model.name


def test_multi_step_exact_o_b_is_reported_incomplete() -> None:
    """The first delivery is honest: multi-step O_B is a Dec-POMDP."""

    result = o_b_exact(build_m1(horizon=3), 1)
    assert not result.exact
    assert result.incomplete
    assert "Dec-POMDP" in result.note


def test_full_information_is_never_worse_than_central_multi_step() -> None:
    for model in all_instances(horizon=3):
        assert (
            o_full(model).expected_cost <= o_central(model).expected_cost + TOL
        ), model.name


def test_message_value_appears_on_m1() -> None:
    model = build_m1(horizon=3)
    without = o_codec(model, NullCodec()).expected_cost
    with_mode = o_codec(model, PosteriorModeCodec(bits=1)).expected_cost
    with_observation = o_codec(model, ObservationCodec(bits=1)).expected_cost
    assert without > with_mode + TOL
    assert with_mode == pytest.approx(o_central(model).expected_cost, abs=TOL)
    assert with_observation == pytest.approx(with_mode, abs=TOL)


def test_no_message_value_when_public_information_is_sufficient() -> None:
    model = build_m0(horizon=3)
    central = o_central(model).expected_cost
    assert o_codec(model, NullCodec()).expected_cost == pytest.approx(central, abs=TOL)
    assert o_codec(model, PosteriorModeCodec(bits=1)).expected_cost == pytest.approx(
        central, abs=TOL
    )
    assert o_codec(model, ConsequenceCodec(bits=1)).expected_cost == pytest.approx(
        central, abs=TOL
    )


def test_a_lossy_codec_at_the_same_budget_costs_more_than_o_b() -> None:
    model = build_m1(horizon=1)
    o_b = o_b_exact(model, 1).expected_cost
    lossy = o_codec(model, PosteriorThresholdCodec(threshold=0.8)).expected_cost
    assert lossy >= o_b - TOL
    assert lossy > o_b + TOL


def test_probe_is_chosen_only_when_it_is_useful_and_affordable() -> None:
    useful = build_m2(horizon=3, probe_cost=0.2, probe_strength="strong")
    uninformative = build_m2(
        horizon=3, probe_cost=0.2, probe_strength="none", name="M2-uninformative"
    )
    expensive = build_m2(
        horizon=3, probe_cost=100.0, probe_strength="strong", name="M2-expensive"
    )

    assert set(o_central_root_actions(useful).values()) == {"PROBE"}
    assert "PROBE" not in set(o_central_root_actions(uninformative).values())
    assert "PROBE" not in set(o_central_root_actions(expensive).values())

    # A useful probe lowers the exact cost; an uninformative or unaffordable one
    # does not, so the solver never uses it.
    assert o_central(useful).expected_cost < o_central(uninformative).expected_cost
    assert o_central(expensive).expected_cost == pytest.approx(
        o_central(uninformative).expected_cost, abs=TOL
    )


def test_public_same_private_different_changes_the_decision() -> None:
    """Identical public input, different private evidence, different action."""

    model = build_m1(horizon=1)
    actions = o_central_root_actions(model)
    both_loaded = actions[("loaded", "loaded", "stalled")]
    both_slipping = actions[("slipping", "slipping", "stalled")]
    assert both_loaded != both_slipping
    assert all(key[2] == "stalled" for key in actions)


def test_limited_search_is_not_exact_and_much_worse_on_m1() -> None:
    model = build_m1(horizon=3)
    limited = true_model_limited_search(model, NullCodec(), depth_limit=1)
    assert not limited.exact
    assert limited.expected_cost > o_central(model).expected_cost + TOL
    assert "not optimal" in limited.note or "NOT exact" in limited.note


def test_cost_and_termination_are_configured_correctly() -> None:
    m3 = build_m3(horizon=3)
    assert o_full(m3).expected_cost == pytest.approx(2.0)
    m0 = build_m0(horizon=3)
    assert o_full(m0).expected_cost == pytest.approx(1.0)
    for model in (m0, m3):
        assert model.fail_cost > 0.0
        for theta in model.theta_values:
            for action in model.actions:
                assert model.transition(model.goal_x, theta, action) == {
                    model.goal_x: 1.0
                }
                assert model.cost(model.goal_x, action) == 0.0


def test_m3_preparation_has_no_informational_value() -> None:
    """Preparation changes the contact mode; it is not a probe."""

    model = build_m3(horizon=3)
    central = o_central(model).expected_cost
    for codec in (
        NullCodec(),
        ObservationCodec(bits=1),
        PosteriorModeCodec(bits=1),
        ConsequenceCodec(bits=1),
    ):
        assert o_codec(model, codec).expected_cost == pytest.approx(central, abs=TOL)
