from copy import deepcopy

import pytest

from experiments.planar_tasks.authoring_v2.guarded_options import compile_policy


def fixture():
    profile = {
        "units": {"clearance": "mm", "advance": "mm", "contact": "flag"},
        "cells": {"start": {"clearance": [280, 320], "contact": [1, 1]}},
        "options": {
            "push": {
                "requires": {"contact": [1, 1]},
                "quality": "model",
                "outcomes": [{
                    "observe": "seen",
                    "change": {"advance": [40, 80]},
                    "post": {"clearance": ["sub", "clearance", "advance"]},
                }],
            },
            "reposition": {"requires": {"clearance": [150, None]}},
        },
    }
    rule = {
        "name": "push-with-contact", "when": {"contact": [1, 1]}, "option": "push",
        "expect": {"advance": [40, 80]}, "then": {"seen": "reposition"},
    }
    return {"rules": [rule]}, profile


def only_case(policy):
    (case,) = policy.check()["cases"]
    return case


def test_forward_and_backward_checks_use_host_model_and_selected_commands():
    program, profile = fixture()
    policy = compile_policy(program, profile)
    assert policy.check()["status"] == "proved"
    assert only_case(policy)["branches"][0]["post"] == {"clearance": [200, 280]}
    assert policy.decide("start") == "push"
    assert policy.resume("start", "seen") == "reposition"
    assert policy.decide("outside") is None
    assert policy.resume("start", "unrecognized") is None

    outcome = profile["options"]["push"]["outcomes"][0]
    outcome["change"]["advance"] = [150, 200]
    program["rules"][0]["expect"]["advance"] = [150, 200]
    long_policy = compile_policy(program, profile)
    case = only_case(long_policy)
    assert case["status"] == "counterexample"
    assert case["issues"] == [
        {"property": "continuation", "metric": "clearance", "status": "counterexample",
         "actual": [80, 170], "required": [150, None], "endpoint": 80,
         "branch": 0, "observation": "seen"}
    ]
    assert long_policy.decide("start", allow_empirical=True) is None
    assert long_policy.resume("start", "seen", allow_empirical=True) is None


@pytest.mark.parametrize(("expected_effect", "status"), [
    ({"advance": [0, 0]}, "counterexample"),
    ({}, "unknown"),
    ({"advance": [None, None]}, "unknown"),
    ({"advance": [0, 100]}, "proved"),
])
def test_claim_cannot_tighten_or_omit_modeled_effect(expected_effect, status):
    program, profile = fixture()
    program["rules"][0]["expect"] = expected_effect
    assert compile_policy(program, profile).check()["status"] == status


def test_rules_cannot_read_future_or_add_arbitrary_actuation():
    program, profile = fixture()
    program["rules"][0]["when"]["advance"] = [0, 100]
    with pytest.raises(ValueError, match="public initial metrics"):
        compile_policy(program, profile)
    del program["rules"][0]["when"]["advance"]
    program["rules"][0]["python"] = "def decide(h): ..."
    with pytest.raises(ValueError, match="expects exactly"):
        compile_policy(program, profile)


def test_branch_selection_respects_public_observation_and_priority():
    program, profile = fixture()
    profile["options"]["hold_back"] = {"requires": {"clearance": [None, 100]}}
    outcomes = profile["options"]["push"]["outcomes"]
    outcomes.append(dict(observe="blocked", change={"advance": [50, 60]},
                         post={"clearance": [80, 100]}))
    program["rules"][0]["then"]["blocked"] = "hold_back"
    policy = compile_policy(program, profile)
    assert policy.resume("start", "seen") == "reposition"
    assert policy.resume("start", "blocked") == "hold_back"
    outcomes[1]["observe"] = "seen"
    del program["rules"][0]["then"]["blocked"]
    policy = compile_policy(program, profile)
    assert only_case(policy)["issues"][0]["branch"] == 1
    assert policy.decide("start") is None  # same public symbol cannot distinguish events

    bad = deepcopy(program["rules"][0])
    bad.update(name="unsafe-first", then={"seen": "STOP"})
    program["rules"].insert(0, bad)
    policy = compile_policy(program, profile)
    case = only_case(policy)
    assert case["enabled"] == ["unsafe-first", "push-with-contact"]
    assert policy.check()["unselected_rules"] == ["push-with-contact"]
    assert case["issues"][0]["property"] == "no_productive_continuation"


def test_incomplete_public_evidence_does_not_activate_a_guard():
    program, profile = fixture()
    profile["cells"]["uncertain"] = {"clearance": [280, 320], "contact": [0, 1]}
    policy = compile_policy(program, profile)
    proved, unselected = policy.check()["cases"]
    assert proved["status"] == "proved"
    assert unselected["status"] == "no_action"
    assert policy.decide("uncertain") is None
    program["rules"][0]["when"] = {}
    uncertain = compile_policy(program, profile).check()["cases"][1]
    assert uncertain["issues"][0]["property"] == "precondition"
    assert uncertain["issues"][0]["endpoint"] == 0


def test_unknown_peer_evidence_fails_closed_empirical_evidence_needs_opt_in():
    program, profile = fixture()
    profile["options"]["push"]["quality"] = "empirical"
    policy = compile_policy(program, profile)
    assert only_case(policy)["status"] == "conditional"
    assert policy.decide("start") is None
    assert policy.resume("start", "seen") is None
    assert policy.decide("start", allow_empirical=True) == "push"
    assert policy.resume("start", "seen", allow_empirical=True) == "reposition"
    outcome = profile["options"]["push"]["outcomes"][0]
    del outcome["post"]["clearance"]
    unknown = compile_policy(program, profile)
    assert only_case(unknown)["status"] == "unknown"
    assert unknown.decide("start", allow_empirical=True) is None


@pytest.mark.parametrize("bad_value", [[80, 20], [0, 1.0], [False, 1], [1]])
def test_malformed_intervals_are_rejected(bad_value):
    program, profile = fixture()
    program["rules"][0]["expect"]["advance"] = bad_value
    with pytest.raises(ValueError):
        compile_policy(program, profile)


def test_units_and_public_dispatch_token_schema():
    program, profile = fixture()
    profile["options"]["push"]["outcomes"][0]["post"]["clearance"] = ["pre", "contact"]
    with pytest.raises(ValueError, match="incompatible units"):
        compile_policy(program, profile)
    program, profile = fixture()
    program["rules"][0]["then"]["seen"] = "arbitrary-expert-command"
    with pytest.raises(ValueError, match="continuation dispatch"):
        compile_policy(program, profile)


def test_explicit_stop_is_preserved_and_counted_as_unproductive():
    program, profile = fixture()
    program["rules"][0].update(option="STOP", expect={}, then={})
    policy = compile_policy(program, profile)
    assert only_case(policy)["status"] == "no_action"
    assert policy.decide("start") is None


@pytest.mark.parametrize("op", ["add", "sub"])
def test_static_continuation_matches_exhaustive_integer_successors(op):
    program, profile = fixture()
    command = profile["options"]["push"]
    profile["cells"]["start"]["clearance"] = [-2, 4]
    command["outcomes"][0].update(change={"advance": [-3, 1]}, post={
        "clearance": [op, "clearance", "advance"]})
    program["rules"][0]["expect"] = {"advance": [-3, 1]}
    all_successors = {pre + delta if op == "add" else pre - delta
                      for pre in range(-2, 5) for delta in range(-3, 2)}
    for limit in range(-7, 9):
        profile["options"]["reposition"]["requires"] = {"clearance": [limit, None]}
        policy = compile_policy(program, profile)
        assert (policy.decide("start") == "push") == all(x >= limit for x in all_successors)


def test_model_and_program_inputs_cannot_change_a_compiled_decision():
    program, profile = fixture()
    policy = compile_policy(program, profile)
    program["rules"][0]["then"]["seen"] = "STOP"
    profile["options"]["push"]["outcomes"][0]["change"]["advance"] = [150, 200]
    policy.check()["cases"][0]["branches"][0]["next_option"] = "STOP"
    assert policy.resume("start", "seen") == "reposition"


@pytest.mark.parametrize(("before", "required", "expected", "after"), [
    ([200, None], [100, None], "proved", [120, None]),
    ([200, None], [100, 300], "unknown", [120, None]),
    ([None, 320], [100, None], "unknown", [None, 280]),
    ([None, 100], [100, None], "counterexample", [None, 60]),
    ([None, None], [0, 0], "unknown", [None, None]),
])
def test_half_bounded_transfer_uses_only_known_endpoints(before, required, expected, after):
    program, profile = fixture()
    profile["cells"]["start"]["clearance"] = before
    profile["options"]["reposition"]["requires"] = {"clearance": required}
    policy = compile_policy(program, profile)
    case = only_case(policy)
    assert case["branches"][0]["post"]["clearance"] == after
    assert case["status"] == expected
    assert (policy.decide("start") is not None) == (expected == "proved")


@pytest.mark.parametrize(("field", "value"), [
    ("option", []), ("then", {"seen": {}}), ("then", {"oracle-only": "reposition"}),
])
def test_malformed_or_undeclared_option_tokens_are_schema_errors(field, value):
    program, profile = fixture()
    program["rules"][0][field] = value
    with pytest.raises(ValueError):
        compile_policy(program, profile)
