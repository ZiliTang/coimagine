"""Mixed-option metric semantics and independent signed-path replay checks."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from coimagine.random_start.effect_push import ACTIONS
from experiments.option_testbench.effect_push_eval import (
    comparison_config, reposition_cases, replay_frames, replay_robot_options, run_one)
from experiments.option_testbench.reposition_combinations_audit import audit


def execute(name):
    c = next(c for c in reposition_cases() if c["name"]==name)
    cfg = asdict(comparison_config("option_combinations", True, {}));feedback=cfg.pop("feedback")
    return run_one(c,feedback,cfg,"option_combinations")


@pytest.fixture(scope="module",params=["rp_CW_solo_0.50", "rp_CCW_corner_1.00",
    "rp_CCW_push_opposite_0.50_0.30_swapped", "rp_CW_follow_0.50_2s", "rp_pair_CW_CCW_0.50"])
def record(request):
    return execute(request.param)


def test_panel_covers_both_reposition_directions_and_other_option_types():
    panel=reposition_cases()
    assert len(panel)==len({c["name"] for c in panel})==24
    partners=set();directions=set()
    for c in panel:
        actions=[ACTIONS[i] for i in c["actions"]]
        assert any(a.kind=="reposition" for a in actions)
        partners.update(a.kind for a in actions)
        directions.update(a.direction for a in actions if a.kind=="reposition")
    assert partners=={"reposition","push","follow","wait"}
    assert directions=={"CW","CCW"}


def test_mixed_metrics_do_not_assume_robot_zero_is_push(record):
    options=record["metric"]["option_outcomes"]
    assert len(options)==2
    assert [o["action"] for o in options]==[ACTIONS[i].label for i in record["case"]["actions"]]
    if options[0]["kind"]!="push":
        assert record["metric"]["actual_effect"] is None
        assert record["metric"]["fulfilled"] is None
    assert record["boundary"]["periods"]==len(record["rows"])


def test_replay_shows_each_side_own_contract_and_logged_progress(record):
    frames,options=replay_frames(record),replay_robot_options(record)
    for i,option in enumerate(options):
        if option["kind"]=="reposition":
            assert option["unit"]=="m"
            assert frames[-1]["robots"][i]["progress"]==pytest.approx(record["metric"]["option_outcomes"][i]["recorded_progress"])
            assert frames[-1]["robots"][i]["actual"] is None
            assert any(f["robots"][i]["surface_points"] for f in frames)
        elif option["kind"]=="follow":
            assert option["unit"]=="s" and option["target"]==2.
            logged=max(r[f"r{i}"]["follow_time"] or 0 for r in record["rows"])
            assert frames[-1]["robots"][i]["progress"]==pytest.approx(logged)


def test_raw_pose_audit_recomputes_requested_boundary_path(record):
    result=audit(record)
    assert result["max_reconstruction_error_m"]<1e-8
    for o in result["options"]:
        if o["kind"]=="reposition" and o["exit"]=="reposition_complete":
            assert o["target_reached"]


def test_path_audit_detects_incorrect_logged_credit():
    r=deepcopy(execute("rp_CW_solo_0.50"))
    next(v for v in r["rows"] if v["r0"]["stage"]=="move")["r0"]["signed_progress"]+=.01
    with pytest.raises(AssertionError):audit(r)
