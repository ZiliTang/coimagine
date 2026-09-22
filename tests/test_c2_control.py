"""C2: does modelling the next round's message change the decision?

The C2 claim is tested by comparing two arms that share the world model, the codec,
the bit budget, the horizon, the action set and the cost:

* **A** ``O_codec`` -- lookahead branches over the next joint observation, runs the
  *same* codec to produce the next messages, and filters the public belief;
* **B** ``O_replan_no_message_branch`` -- lookahead branches only on the public
  observation and never predicts the next message, then executes and replans when
  the real messages arrive.

Three conditions must hold together: a positive separation, zero separation on
information-free tasks, and collapse at ``B = 0``.
"""

from __future__ import annotations

import pytest

from coimagine.exact import (
    NullCodec,
    PosteriorModeCodec,
    o_central,
    o_central_root_actions,
    o_codec,
    o_replan_no_message_branch,
    o_replan_root_actions,
)
from coimagine.toy import all_instances, build_m0, build_m1, build_m2, build_m3

TOL = 1e-9


def test_c2_has_a_positive_separation_on_m2() -> None:
    """M2 is the C2 instance: the probe's value only exists through a future message."""

    model = build_m2(horizon=3, probe_cost=0.2, probe_strength="strong")
    codec = PosteriorModeCodec(bits=1)

    method = o_codec(model, codec).expected_cost
    control = o_replan_no_message_branch(model, codec).expected_cost
    central = o_central(model).expected_cost
    public_only = o_codec(model, NullCodec()).expected_cost

    # The method reaches the full-observation-sharing optimum ...
    assert method == pytest.approx(central, abs=TOL)
    # ... while the same codec without a future-message branch is no better than no
    # communication at all.
    assert control == pytest.approx(public_only, abs=TOL)
    assert control > method + TOL


def test_c2_has_only_a_small_separation_on_m1() -> None:
    """M1 is the C1 instance: its evidence is used in the current round, not later."""

    model = build_m1(horizon=3)
    codec = PosteriorModeCodec(bits=1)
    method = o_codec(model, codec).expected_cost
    control = o_replan_no_message_branch(model, codec).expected_cost
    assert control >= method - TOL
    # Small but non-zero: the second round's message still matters a little.
    assert control - method < 0.1


def test_c2_negative_controls_show_no_separation() -> None:
    """Where information cannot change the decision, the two arms must agree."""

    cases = (
        build_m0(horizon=3),
        build_m2(horizon=3, probe_cost=0.2, probe_strength="none", name="M2-uninformative"),
        build_m2(horizon=3, probe_cost=100.0, probe_strength="strong", name="M2-expensive"),
        build_m3(horizon=3),
    )
    for model in cases:
        for codec in (NullCodec(), PosteriorModeCodec(bits=1)):
            method = o_codec(model, codec).expected_cost
            control = o_replan_no_message_branch(model, codec).expected_cost
            assert method == pytest.approx(control, abs=TOL), (model.name, codec.name)


def test_c2_collapses_when_there_is_no_message_to_predict() -> None:
    """At B = 0 the control and the method must be identical everywhere."""

    for model in all_instances(horizon=3):
        method = o_codec(model, NullCodec()).expected_cost
        control = o_replan_no_message_branch(model, NullCodec()).expected_cost
        assert method == pytest.approx(control, abs=TOL), model.name


def test_message_blind_control_does_not_probe_for_information() -> None:
    """The probe's only value is informational, so the message-blind arm skips it."""

    model = build_m2(horizon=3, probe_cost=0.2, probe_strength="strong")
    codec = PosteriorModeCodec(bits=1)
    method_roots = set(o_central_root_actions(model).values())
    control_roots = set(o_replan_root_actions(model, codec).values())
    assert "PROBE" in method_roots
    assert "PROBE" not in control_roots


def test_c2_gap_is_not_caused_by_a_shorter_horizon() -> None:
    """Both arms use the full remaining horizon; only the branching differs."""

    model = build_m2(horizon=3, probe_cost=0.2, probe_strength="strong")
    codec = PosteriorModeCodec(bits=1)
    method = o_codec(model, codec)
    control = o_replan_no_message_branch(model, codec)
    assert method.horizon == control.horizon == model.horizon
    assert method.bits == control.bits == 1
    assert method.codec == control.codec == "posterior-mode"
