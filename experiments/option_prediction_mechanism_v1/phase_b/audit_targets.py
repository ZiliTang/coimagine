"""Audit what the phase-B training targets actually are, and whether they are observable.

The round-3 report called the terminal-geometry targets "the local geometry the robot will observe
at the end of the skill". The implementation does not do that: the labels are derived from the
**true** terminal object pose and from a **face index** chosen with the **true** initial object
pose. This script documents that by reading the collector's own fields and measuring how far the
truth-face geometry is from the observed scan segment that the robot actually receives.

No labels are changed and nothing is retrained.

    python experiments/option_prediction_mechanism_v1/phase_b/audit_targets.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / ("outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/"
                       "phase_b/executions_train.jsonl")


def _percentiles(values: List[float]) -> Dict[str, object]:
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {"n": len(values), "p50": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "max": values[-1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_DATA.parent / "target_audit.json"))
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.data).open()]

    truth_distance_vs_observed: List[float] = []
    normal_angle_truth_vs_observed: List[float] = []
    mask_disagreements = {"truth_present_observed_missing": 0,
                          "observed_present_truth_masked": 0,
                          "both_masked": 0, "both_present": 0}
    per_robot: Dict[str, Dict[str, int]] = {}
    for record in records:
        for robot_id in (0, 1):
            robot = record["label"][f"robot{robot_id}"]
            truth_present = robot["mask_terminal_geometry"] == 1
            observed_present = robot["mask_terminal_patch_valid"] == 1
            key = (f"truth_{'present' if truth_present else 'masked'}_"
                   f"observed_{'present' if observed_present else 'masked'}")
            per_robot[key] = per_robot.get(key, 0) + 1
            if truth_present and not observed_present:
                mask_disagreements["truth_present_observed_missing"] += 1
            if observed_present and not truth_present:
                mask_disagreements["observed_present_truth_masked"] += 1
            if truth_present and observed_present:
                mask_disagreements["both_present"] += 1
            if not truth_present and not observed_present:
                mask_disagreements["both_masked"] += 1
            if not (truth_present and observed_present):
                continue
            truth_distance_vs_observed.append(
                abs(float(robot["terminal_signed_distance"])
                    - float(robot["obs_terminal_patch_distance"])))
            observed_normal = robot.get("obs_terminal_patch_normal_own_start_frame")
            truth_normal = robot.get("terminal_normal_own_start_frame")
            if observed_normal is not None and truth_normal is not None:
                dot = max(-1.0, min(1.0, float(observed_normal[0]) * float(truth_normal[0])
                                    + float(observed_normal[1]) * float(truth_normal[1])))
                normal_angle_truth_vs_observed.append(math.degrees(math.acos(dot)))

    payload = {
        "records": len(records),
        "provenance": {
            "terminal_signed_distance":
                "collect_b.segment_geometry(terminal_object, face, terminal_pose) -- TRUE terminal "
                "object pose and the face line of the TRUE face",
            "terminal_tangential_offset": "same call: coordinate along that TRUE face line",
            "terminal_normal_own_start_frame / terminal_normal_sin_cos":
                "collect_b.face_frame(terminal_object, face) -- the TRUE face outward normal",
            "face": "collect_b.reference_face(initial_object, pose_start) -- chosen with the TRUE "
                    "initial object pose, not from the observed patch",
            "mask_terminal_geometry":
                "mask_terminal_patch_valid AND point_box_distance(terminal_pose, terminal_object) "
                "<= SCAN_RANGE -- the second term again uses the TRUE terminal object pose",
            "observable_analogue_that_was_collected_but_not_trained":
                ["obs_terminal_patch_distance", "obs_terminal_patch_normal_own_start_frame",
                 "obs_terminal_patch_centroid_own_start_frame", "mask_terminal_patch_valid"],
        },
        "mask_cross_tab": mask_disagreements,
        "per_robot_mask_cross_tab": per_robot,
        "truth_face_versus_observed_patch": {
            "signed_distance_vs_observed_patch_distance_m":
                _percentiles(truth_distance_vs_observed),
            "normal_angle_between_truth_face_and_observed_patch_deg":
                _percentiles(normal_angle_truth_vs_observed),
            "tangential_offset": {
                "comparable": False,
                "why": "the label is the coordinate along the TRUE face line measured from the "
                       "TRUE face centre; the nearest observable (the patch centroid in the "
                       "robot's start frame) has a different origin and tangent, so the two are "
                       "different quantities by construction and no like-for-like error exists. "
                       "The label as trained is not obtainable from allowed observations."}},
        "conclusion": (
            "The trained terminal-geometry targets are PRIVILEGED object-face quantities, not the "
            "scan segment the robot observes. They are therefore (a) not obtainable from allowed "
            "observations, and (b) not a valid target for predicting what the robot will observe. "
            "The 'deployable' arm is predicting a partly unobservable label, so its small error "
            "cannot be read as 'the robot can predict its own terminal observation'."),
    }
    target = Path(args.out)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    print(json.dumps({key: payload[key] for key in
                      ("records", "mask_cross_tab", "truth_face_versus_observed_patch")},
                     indent=1, default=float))
    print(f"wrote {target}")
    print("AUDIT_TARGETS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
