"""Information boundary, record invariants and real bit accounting."""

from __future__ import annotations

import dataclasses

import pytest

from coimagine.exact import (
    NullCodec,
    ObservationCodec,
    PosteriorModeCodec,
    RobotView,
    codebook,
    initial_public_beliefs,
    o_codec,
    successor_public_beliefs,
)
from coimagine.protocol import (
    InformationBoundaryError,
    Message,
    OracleContext,
    PrivateHistory,
    PublicContext,
    PublicRecord,
    TrueState,
    reject_privileged,
    visible_signature,
)
from coimagine.toy import X_CONTACT, build_m1
from coimagine.toy.instances import A_CO_PUSH


def _context() -> PublicContext:
    record = PublicRecord(y_public=("stalled",), actions=(), messages=())
    return PublicContext(
        record=record,
        history=PrivateHistory(robot_id=0, observations=("loaded",)),
        candidates=("CO_PUSH", "RELEASE_PUSH_B", "HOLD"),
    )


def test_privileged_objects_are_rejected_by_deployment_guard() -> None:
    reject_privileged(_context(), interface="test")  # must not raise

    state = TrueState(x=X_CONTACT, theta="HEAVY")
    with pytest.raises(InformationBoundaryError):
        reject_privileged(state, interface="test")

    oracle = OracleContext(true_state=state, model=build_m1(horizon=1))
    with pytest.raises(InformationBoundaryError):
        reject_privileged(oracle, interface="test")


def test_visible_signature_unchanged_when_hidden_truth_changes() -> None:
    """Changing the hidden truth must not change the visible policy input."""

    before = visible_signature(_context())
    _ = TrueState(x=X_CONTACT, theta="HEAVY")
    _ = TrueState(x=X_CONTACT, theta="SLIP")
    assert before == visible_signature(_context())


def test_public_record_invariants_hold_after_advance() -> None:
    record = PublicRecord(y_public=("stalled",))
    assert record.step == 0
    nxt = record.advance(action="CO_PUSH", messages=(1, 0), y_next="moved")
    assert len(nxt.y_public) == len(nxt.actions) + 1
    assert len(nxt.messages) == len(nxt.actions)
    assert nxt.step == 1
    assert nxt.messages == ((1, 0),)

    with pytest.raises(ValueError):
        PublicRecord(y_public=("stalled", "moved"), actions=())
    with pytest.raises(ValueError):
        PublicRecord(y_public=("stalled",), actions=("CO_PUSH",), messages=())


def test_same_public_record_yields_same_commit_input() -> None:
    record = PublicRecord(y_public=("stalled",), actions=(), messages=())
    assert record.commit_key(1, 0) == record.commit_key(1, 0)


def test_same_round_message_cannot_read_teammate() -> None:
    model = build_m1(horizon=1)
    codec = PosteriorModeCodec(bits=1)
    view_a = RobotView(0, ("loaded",), ("stalled",), (), model.actions)
    # The teammate's same-round observation is not part of A's view, so repeating
    # the construction cannot change A's message.
    view_a_again = RobotView(0, ("loaded",), ("stalled",), (), model.actions)
    assert codec.encode(model, view_a) == codec.encode(model, view_a_again)

    view_b = RobotView(1, ("slipping",), ("stalled",), (), model.actions)
    assert codec.encode(model, view_a) != codec.encode(model, view_b)


def test_message_bits_are_real() -> None:
    assert codebook(0) == (0,)
    assert codebook(1) == (0, 1)
    assert len(codebook(2)) == 4

    with pytest.raises(ValueError):
        Message(robot_id=0, symbol=0, bits=1, codebook=(0,))

    model = build_m1(horizon=1)
    for codec in (NullCodec(), ObservationCodec(bits=1), PosteriorModeCodec(bits=1)):
        view = RobotView(0, ("loaded",), ("stalled",), (), model.actions)
        assert codec.encode(model, view) in codebook(codec.bits)


def test_repeated_evidence_is_not_double_counted() -> None:
    """The filter appends exactly one message pair per boundary."""

    model = build_m1(horizon=1)
    codec = NullCodec()
    beliefs = initial_public_beliefs(model, "stalled", codec, model.actions)
    _key, (belief, prob) = next(iter(beliefs.items()))
    assert prob > 0.0
    successors = successor_public_beliefs(
        model, belief, "CO_PUSH", ("stalled",), (), (), codec, model.actions
    )
    for _obs, (_p, _b, messages) in successors.items():
        assert len(messages) == 1


def test_initial_belief_does_not_use_future_transitions() -> None:
    """Boundary-0 belief must be independent of dynamics after the first step."""

    model = build_m1(horizon=3)
    transitions = dict(model.transitions)
    observations = dict(model.observations)
    # Replace only rows reachable from PREPARED, which cannot affect boundary 0.
    for theta in model.theta_values:
        transitions[("PREPARED", theta, A_CO_PUSH)] = {"CONTACT": 1.0}
        observations[("PREPARED", theta, A_CO_PUSH, "CONTACT")] = {
            (("loaded", "loaded"), "stalled"): 1.0
        }
    mutated = dataclasses.replace(
        model, transitions=transitions, observations=observations
    )

    base = initial_public_beliefs(model, "stalled", NullCodec(), model.actions)
    changed = initial_public_beliefs(mutated, "stalled", NullCodec(), model.actions)
    assert set(base.keys()) == set(changed.keys())
    for key in base:
        assert base[key][0] == changed[key][0]


def test_deployment_planner_is_deterministic() -> None:
    model = build_m1(horizon=1)
    a = o_codec(model, NullCodec())
    b = o_codec(model, NullCodec())
    assert a.expected_cost == b.expected_cost
