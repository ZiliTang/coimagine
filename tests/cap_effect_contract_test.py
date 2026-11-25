import copy

import pytest

from experiments.planar_tasks.authoring_v2.policy_effect_contract import audit_contract
from tools.cap_effect_contract import CABINET_CLAIM, PLANAR_CLAIM, audit_planar_record, cabinet_cases


def test_cabinet_same_core_and_grasp_outcome_across_two_pairs():
    rows = cabinet_cases()
    assert [row["audit"]["outcome"]["verdict"] for row in rows] == ["supported"] * 4
    assert [row["audit"]["precondition"]["verdict"] for row in rows] == ["supported"] * 4
    assert [row["audit"]["continuation"]["verdict"] for row in rows] == [
        "supported", "contradicted", "supported", "unknown"]
    assert rows[-1]["opening_interval_m"] == rows[-2]["opening_interval_m"]


def test_evidence_cannot_move_between_stages_or_run_undeclared_metrics():
    nested = {"precondition": {}, "outcome": {"empty_gripper_entry_margin_m": [1, 1]},
              "continuation": {}}
    with pytest.raises(ValueError, match="at this stage"):
        audit_contract(CABINET_CLAIM, nested)


@pytest.mark.parametrize("bad", [[], [0, float("nan")], [1, 0], [None, None]])
def test_malformed_or_vacuous_claim(bad):
    claim = dict(CABINET_CLAIM, precondition={"a": bad})
    with pytest.raises(ValueError):
        audit_contract(claim, {"precondition": {"a": [0, 1]}, "outcome": {}, "continuation": {}})


def test_planar_terminal_branch_and_unknown_unexecuted_command():
    step = {"tick": 0, "end_tick": 1, "pending": False, "target_id": "box", "seen": True,
            "action": {"a0": 2, "a1": 44, "target_id": "box"}, "goal": [1, 0],
            "truth": {"object": [0, 0, 0]}, "end_truth": {"object": [0.05, 0, 0]}}
    trace = {"steps": [step], "terminal": {"tick": 1, "truth": step["end_truth"]},
             "result": {"success": True, "ticks": 1}}
    audit = audit_planar_record(trace, PLANAR_CLAIM)[0]["audit"]
    assert audit["outcome"]["verdict"] == audit["continuation"]["verdict"] == "supported"
    assert audit["continuation"]["witnesses"]["next_actual_command"]["verdict"] == "unknown"
    bad = copy.deepcopy(trace)
    bad["steps"][0]["end_tick"] += 2
    with pytest.raises(ValueError, match="complete room"):
        audit_planar_record(bad, PLANAR_CLAIM)
