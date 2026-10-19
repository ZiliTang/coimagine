"""Directed Round-000 testbench checks (bounded: one branch per run)."""
import importlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/option_testbench"
for path in (str(EXPERIMENT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import panel as P  # noqa: E402
import runner  # noqa: E402
from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import spec, skills  # noqa: E402


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("option_panel")
    P.prepare(out)
    return out


def test_same_snapshot_repeat_and_branch_isolation(prepared, tmp_path):
    first = runner.run_branch(prepared, 0, tmp_path / "a")
    second = runner.run_branch(prepared, 0, tmp_path / "b")
    assert first["status"] == second["status"] == "completed"
    assert first["skill_boundaries"] == second["skill_boundaries"]
    assert first["final_state"]["object"] == pytest.approx(
        second["final_state"]["object"], abs=1e-9)
    other = runner.run_branch(prepared, 1, tmp_path / "a")
    assert other["branch_id"] != first["branch_id"]
    again = runner.run_branch(prepared, 0, tmp_path / "c")
    assert again["final_state"]["object"] == pytest.approx(
        first["final_state"]["object"], abs=1e-9)


def test_one_period_advances_once_and_velocity_is_not_zeroed(prepared, tmp_path):
    record = runner.run_branch(prepared, 0, tmp_path)
    periods = [row["period"] for row in record["rows"]]
    assert periods == list(range(1, len(periods) + 1))
    assert record["total_periods"] == len(record["rows"])
    metadata = json.loads((prepared / "panel.json").read_text())
    world, instance, payload = runner.build_world(metadata, "moving_recovery", prepared)
    state = runner._state_tuple(world)
    assert abs(float(state["object"][4])) > 0.05        # real non-zero object velocity
    assert abs(float(state["robots"][0][4])) > 0.05     # real non-zero robot velocity
    assert payload is not None and payload["branch_state"]["total_periods"] > 0


def test_information_boundary_hidden_truth_is_not_used(prepared, tmp_path):
    metadata = json.loads((prepared / "panel.json").read_text())
    world, _instance, _payload = runner.build_world(metadata, "face", prepared)
    actions = [4, 4]
    original = runner.eligibility(world, actions)
    observations = world.boundary_observations()
    world.boundary_observations = lambda: observations  # hold legal inputs fixed
    after = runner.eligibility(world, actions)
    assert original == after
    from coimagine.random_start.skills import SkillObservation
    probe = observations[0]
    assert probe.forbidden_fields() == ()
    assert not hasattr(probe, "object_pose")
    for field in ("object", "true_state", "env", "reward", "success"):
        assert not hasattr(probe, field)


def test_independent_recompute_of_push_and_reposition(prepared, tmp_path):
    import metrics
    # v4.1: the push promise is *robot travel* after contact, so use the same-side teammate
    # branch where the object actually moves; the symmetric face push is contact_only by design.
    push = runner.run_branch(prepared, 18, tmp_path)         # teammate__push_push
    reposition = runner.run_branch(prepared, 1, tmp_path)    # face__push_reposition_ccw
    contract = json.loads((prepared / "contracts" / "contracts.json").read_text())
    push_option = metrics.recompute_option(push, push["boundaries"][0], 0, 17, contract)
    assert push_option["status"] == "fulfilled"
    assert push_option["contact_seconds"] >= contract["push"]["thresholds"]["contact_established_seconds"]
    # v4.1: the push promise is the signed travel, not a 2 s contact accumulation
    assert push_option["signed_progress"] >= (
        contract["push"]["thresholds"]["progress_fraction_min"] * push_option["requested_distance"])
    reposition_option = metrics.recompute_option(
        reposition, reposition["boundaries"][0], 1, 36, contract)
    assert reposition_option["status"] == "fulfilled"
    assert abs(reposition_option["signed_progress"]) >= (
        contract["reposition"]["thresholds"]["progress_fraction_min"]
        * reposition_option["requested_distance"] - 1e-9)


def test_coverage_denominator_handles_zero_accepted(prepared, tmp_path):
    import metrics
    out = tmp_path / "synthetic"
    (out / "contracts").mkdir(parents=True)
    (out / "logs").mkdir(parents=True)
    (out / "contracts" / "contracts.json").write_text(
        (prepared / "contracts" / "contracts.json").read_text())
    panel = json.loads((prepared / "panel.json").read_text())
    (out / "panel.json").write_text(json.dumps(panel))
    request = {"branch_id": "x__rejected", "snapshot": "face", "actions": [[4, 4]],
               "teammate_protocol": "shared_clock_joint", "focus": "rejected_case",
               "recovery": False}
    (out / "requests.json").write_text(json.dumps({"requests": [request]}))
    P.write_json(out / "logs" / "x__rejected.json", {
        "branch_id": "x__rejected", "snapshot": "face", "status": "completed",
        "skill_boundaries": [], "rows": [], "boundaries": [],
        "eligibility": [{"step": 0, "action": [4, 4], "rows": [
            {"robot": 0, "action": 4, "label": "PUSH", "kind": "push", "accepted": False,
             "reasons": ["no_object_patch"]},
            {"robot": 1, "action": 4, "label": "PUSH", "kind": "push", "accepted": False,
             "reasons": ["no_object_patch"]}]}]})
    summary = metrics.summarize(out)
    assert summary["accepted"] == 0
    assert summary["conditional_fulfillment"] is None
    assert summary["conservative_interval"] is None
    assert summary["effective_fulfillment_coverage"] == 0.0


def test_timeout_kills_process_group_and_preserves_log(tmp_path):
    import cli
    marker = "option_testbench_timeout_marker"
    command = [sys.executable, "-c",
               "import subprocess,sys,time;"
               f"subprocess.Popen([sys.executable,'-c',\"import time;time.sleep(30)\"]);"
               f"open(r'{tmp_path / 'marker.txt'}','w').write('{marker}');"
               "time.sleep(30)"]
    code, out, err, timed_out = cli.run_bounded(command, timeout=1.0, cwd=tmp_path)
    assert timed_out is True
    assert (tmp_path / "marker.txt").exists()   # already-written results are preserved
    time.sleep(0.5)
    survivors = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode(errors="ignore")
        except OSError:
            continue
        if marker in cmdline or "time.sleep(30)" in cmdline:
            survivors.append(entry.name)
    assert survivors == []


def test_baseline_menu_and_no_implicit_monkeypatch(prepared, tmp_path):
    assert spec.ENV_VERSION == "random-start-oriented-v4.1"
    assert spec.ROBOT_ACTION_COUNT == 48 and spec.JOINT_ACTION_COUNT == 2304
    assert spec.ROBOT_ACTIONS[16].label.startswith("PUSH(a=+0deg,v=0.16)")
    assert spec.ROBOT_ACTIONS[36].label.startswith("REPOSITION_CCW")
    assert spec.ROBOT_ACTIONS[40].label.startswith("REPOSITION_CW")
    assert spec.ROBOT_ACTIONS[44].label.startswith("WAIT_BRAKE")
    # v4.1: PUSH/REPOSITION scales are requested distances (m); WAIT stays a duration (s)
    assert spec.ROBOT_ACTIONS[16].scale == spec.PUSH_DISTANCES[0]
    assert spec.ROBOT_ACTIONS[19].scale == spec.PUSH_DISTANCES[-1]
    assert spec.ROBOT_ACTIONS[16].scale_kind == "distance"
    assert spec.ROBOT_ACTIONS[44].scale_kind == "duration"
    timeout = spec.option_timeout(spec.ROBOT_ACTIONS[19])
    assert spec.OPTION_PREPARE_SECONDS < timeout <= spec.JOINT_MAX_SECONDS
    metadata = json.loads((prepared / "panel.json").read_text())
    before = E.make_skill
    runner.build_world(metadata, "face", prepared)
    assert E.make_skill is before
