"""Inspect empirical option expectations, using the same checker in two tasks."""
import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from experiments.planar_tasks.authoring_v2 import policy_effect_contract as engine


PLANAR_CLAIM = {
    "precondition": {"target_visible_at_issue": [1, 1]},
    "outcome": {"goal_gain_m": [0.02, None]},
    "continuation": {
        "next_actual_command": {"next_goal_gain_m": [0.02, None]},
        "finished": {"goal_reached_at_end": [1, 1]},
    },
}
CABINET_CLAIM = {
    "precondition": {"empty_gripper_entry_margin_m": [0, None]},
    "outcome": {"grasp_held": [1, 1]},
    "continuation": {"straight_withdrawal": {"loaded_exit_margin_m": [0, None]}},
}


def point(x):
    return [x, x]


def margin(width_bounds, span):
    return [None if bound is None else bound - span for bound in width_bounds]


def cabinet_cases():
    """One straight planar bottleneck with no other obstacles or rotations.

    All numbers, including the grasp sensor, are synthetic fixture inputs.
    Two grasp poses share one public starting observation per pair. They have
    different transverse carried spans. The partial view establishes only a
    lower bound on the opening. The hidden true opening is never supplied.
    """
    rows = []
    for view, opening in (("visible", [0.11, 0.11]), ("partly_occluded", [0.09, None])):
        for pose, width in (("narrow", 0.06), ("wide", 0.105)):
            empty_span = 0.05
            loaded_span = width + 0.02  # total swept width incl. jaw projection
            evidence = {
                "precondition": {"empty_gripper_entry_margin_m": margin(opening, empty_span)},
                "outcome": {"grasp_held": point(1)},
                "continuation": {"loaded_exit_margin_m": margin(opening, loaded_span)},
            }
            rows.append(dict(view=view, pose=pose, opening_interval_m=opening,
                             empty_span_m=empty_span, loaded_span_m=loaded_span,
                             audit=engine.audit_contract(CABINET_CLAIM, evidence)))
    return rows


def goal_gain(row):
    return (math.dist(row["truth"]["object"][:2], row["goal"])
            - math.dist(row["end_truth"]["object"][:2], row["goal"]))


def audit_planar_record(data, claim):
    """Audit ordinary target PUSH windows. Only the actual next command is measured."""
    steps = data["steps"]
    if not steps or steps[0]["tick"] != 0 or steps[-1]["end_tick"] != data["result"]["ticks"]:
        raise ValueError("expected a complete room from tick zero")
    if any(s["tick"] >= s["end_tick"] for s in steps):
        raise ValueError("nonpositive command duration")
    for step, next_step in zip(steps, steps[1:]):
        if (step["end_tick"] != next_step["tick"] or step["end_truth"] != next_step["truth"]
                or step["target_id"] != next_step["target_id"] or step["goal"] != next_step["goal"]):
            raise ValueError("command endpoints, targets or ticks do not match")
    if data["terminal"] != {"tick": steps[-1]["end_tick"], "truth": steps[-1]["end_truth"]}:
        raise ValueError("terminal endpoint mismatch")
    rows = []
    for i, step in enumerate(steps):
        action = step["action"]
        if step["pending"] or action["target_id"] != step["target_id"] or min(action["a0"], action["a1"]) >= 32:
            continue
        next_step = steps[i+1] if i+1 < len(steps) else None
        finished = next_step is None and data["result"]["success"]
        evidence = {
            "precondition": {"target_visible_at_issue": point(int(step["seen"]))},
            "outcome": {"goal_gain_m": point(goal_gain(step))},
            "continuation": {"goal_reached_at_end": point(int(finished)),
                             "next_goal_gain_m": point(goal_gain(next_step)) if next_step else [None, None]},
        }
        rows.append(dict(tick=step["tick"], end_tick=step["end_tick"], action=action,
                         next_action=next_step["action"] if next_step else None,
                         audit=engine.audit_contract(claim, evidence)))
    return rows


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planar-record", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, help="Full report, created without overwriting")
    args = p.parse_args()
    core_sha = file_sha(engine.__file__)
    result = {
        "core_sha256": core_sha,
        "adapter_sha256": file_sha(__file__),
        "cabinet": {"kind": "synthetic aligned bottleneck fixtures, no simulated arm or control trials",
                    "claim": CABINET_CLAIM, "rows": cabinet_cases()},
        "planar": {"kind": "retrospective host-defined checks on actual ordinary target PUSH commands",
                   "claim": PLANAR_CLAIM, "rooms": []},
        "limits": [
            "A contradicted continuation refers to the stated witnesses only. Untested paths may exist.",
            "The planar continuation measures actual next-command progress, not reachability or all choices.",
            "Goal gain uses offline object centres. No outcomes or private states are policy inputs.",
            "The two-centimetre thresholds are illustrative host tests, not declarations of the historical policy.",
        ],
    }
    for path in args.planar_record:
        payload = path.read_bytes()
        data = json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)
        rows = audit_planar_record(data, PLANAR_CLAIM)
        result["planar"]["rooms"].append(dict(
            path=str(path), sha256=file_sha(path), commands=len(data["steps"]),
            audited_target_pushes=len(rows), rows=rows))
    print("cabinet continuation", [(r["view"], r["pose"], r["audit"]["continuation"]["verdict"])
                                    for r in result["cabinet"]["rows"]])
    for room in result["planar"]["rooms"]:
        print(room["path"], room["audited_target_pushes"], {
            key: dict(Counter(r["audit"][key]["verdict"] for r in room["rows"]))
            for key in ("precondition", "outcome", "continuation")})
    if args.out:
        with args.out.open("x") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
