"""Independent verification of a phase-B collection directory.

Checks the frozen requirements that are cheap and load-bearing:

* planned vs executed counts per split, and that every sampled root produced exactly the
  expected number of executions with distinct commands;
* that all executions of one root share one initial state (same ``snapshot_sha256`` and same
  start poses), i.e. the eight commands are action branches on one physical root;
* split disjointness: no root id / seed crosses splits, and the minimum cross-split
  distance (object pose + robot poses) recomputed *independently* from ``roots.json``;
* exact replay: re-run one stored execution per split from the artifact alone and compare the
  snapshot hash, exit class, duration and terminal state to the stored record;
* field completeness and mask consistency.

Usage::

    python experiments/.../phase_b/verify_b.py \
        --data-dir outputs/.../phase_b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _path in (str(ROOT / "src"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import collect_b as CB  # noqa: E402
import roots_b as RB  # noqa: E402

from coimagine.local_contact_option.roots import VariantSpec  # noqa: E402
from coimagine.local_contact_option.runner import (  # noqa: E402
    build_snapshot,
    root_from_metadata,
    run_execution,
)

SPLITS = ("train", "val", "test")


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def independent_distance(a: Dict[str, object], b: Dict[str, object]) -> Tuple[float, float]:
    """Same family metric as ``roots_b.root_distance``, written independently."""

    oa, ob = a["object_pose"], b["object_pose"]
    position = math.hypot(oa[0] - ob[0], oa[1] - ob[1])
    angle = RB.ANG_SCALE * abs(_wrap(oa[2] - ob[2]))
    for robot_id in (0, 1):
        ra, rb = a["robot_poses"][robot_id], b["robot_poses"][robot_id]
        position = max(position, math.hypot(ra[0] - rb[0], ra[1] - rb[1]))
        angle = max(angle, RB.ANG_SCALE * abs(_wrap(ra[2] - rb[2])))
    return max(position, angle), position


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify(data_dir: Path, *, sample_roots: int = 2) -> Dict[str, object]:
    roots_artifact = json.loads((data_dir / "roots.json").read_text())
    commands_artifact = json.loads((data_dir / "commands.json").read_text())
    ledger = json.loads((data_dir / "ledger.json").read_text())
    commands_by_root = {item["root_id"]: item for item in commands_artifact["roots"]}

    roots: Dict[str, List[Dict[str, object]]] = {
        name: roots_artifact["roots"].get(name, []) for name in SPLITS
    }
    executions: Dict[str, List[Dict[str, object]]] = {}
    for name in SPLITS:
        path = data_dir / f"executions_{name}.jsonl"
        executions[name] = load_jsonl(path) if path.exists() else []

    result: Dict[str, object] = {"data_dir": str(data_dir), "checks": {}}

    # ---- counts ---------------------------------------------------------
    counts = {}
    for name in SPLITS:
        by_root: Dict[str, List[Dict[str, object]]] = {}
        for record in executions[name]:
            by_root.setdefault(record["root_id"], []).append(record)
        expected = int(ledger["split_roots_requested"].get(name, 0))
        commands_per_root = int(ledger["splits"][name]["commands_per_root"])
        bad = {root_id: len(rows) for root_id, rows in by_root.items()
               if len(rows) != commands_per_root}
        counts[name] = {
            "roots_planned": expected,
            "roots_sampled": len(roots[name]),
            "executions": len(executions[name]),
            "roots_with_wrong_command_count": bad,
            "all_roots_have_full_command_set": not bad,
        }
    result["checks"]["counts"] = counts

    # ---- commands distinct, snapshot shared, start state shared ---------
    root_problems: List[str] = []
    snapshot_problems: List[str] = []
    for name in SPLITS:
        by_root = {}
        for record in executions[name]:
            by_root.setdefault(record["root_id"], []).append(record)
        for root_id, rows in by_root.items():
            signatures = {
                json.dumps({
                    "alpha": record["input"]["command"]["alpha_rad"],
                    "speed": record["input"]["command"]["speed"],
                }, sort_keys=True)
                for record in rows
            }
            if len(signatures) != len(rows):
                root_problems.append(f"{root_id}: duplicate commands")
            hashes = {record["snapshot_sha256"] for record in rows}
            starts = {
                json.dumps([record["truth"]["object_start_world"],
                            record["truth"]["robots_start_world"]], sort_keys=True)
                for record in rows
            }
            if len(hashes) != 1 or len(starts) != 1:
                snapshot_problems.append(f"{root_id}: {len(hashes)} snapshot hashes, "
                                         f"{len(starts)} start states")
    result["checks"]["command_and_snapshot_identity"] = {
        "distinct_command_violations": root_problems,
        "shared_snapshot_violations": snapshot_problems,
        "ok": not root_problems and not snapshot_problems,
    }

    # ---- split disjointness --------------------------------------------
    ids = {name: {root["root_id"] for root in roots[name]} for name in SPLITS}
    seeds = {name: {int(root["seed"]) for root in roots[name]} for name in SPLITS}
    id_overlaps = {f"{a}|{b}": sorted(ids[a] & ids[b])
                   for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    seed_overlaps = {f"{a}|{b}": len(seeds[a] & seeds[b])
                     for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    pair_min = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            best = None
            for root_a in roots[a]:
                for root_b in roots[b]:
                    value, position = independent_distance(root_a, root_b)
                    if best is None or value < best[0]:
                        best = (value, position, root_a["root_id"], root_b["root_id"])
            pair_min[f"{a}|{b}"] = {
                "applicable": bool(roots[a] and roots[b]),
                "min_distance": best[0] if best else None,
                "min_position_distance": best[1] if best else None,
                "pair": [best[2], best[3]] if best else None,
            }
    result["checks"]["split_disjointness"] = {
        "root_id_overlaps": id_overlaps,
        "seed_overlaps": seed_overlaps,
        "min_cross_split_distance": pair_min,
        "threshold": RB.MIN_ROOT_DISTANCE,
        "ok": (all(not value for value in id_overlaps.values())
               and all(value == 0 for value in seed_overlaps.values())
               and all((not value["applicable"])
                       or (value["min_distance"] is not None
                           and value["min_distance"] >= RB.MIN_ROOT_DISTANCE)
                       for value in pair_min.values())),
    }

    # ---- class coverage and command-range coverage ----------------------
    classes_seen = {name: sorted({root["interaction_class"] for root in roots[name]})
                    for name in SPLITS}
    alphas = set()
    speeds = set()
    for name in SPLITS:
        for record in executions[name]:
            for value in record["input"]["command"]["alpha_rad"]:
                if value is not None:
                    alphas.add(round(value, 9))
            for value in record["input"]["command"]["speed"]:
                if value is not None:
                    speeds.add(round(value, 9))
    result["checks"]["coverage"] = {
        "classes_per_split": classes_seen,
        "all_classes_covered": all(len(value) == len(RB.CLASSES) for value in classes_seen.values()),
        "alpha_values_rad": sorted(alphas),
        "speed_values": sorted(speeds),
        "alpha_in_frozen_set": alphas <= {round(v, 9) for v in RB.ALPHA_CHOICES},
        "speed_in_frozen_set": speeds <= {round(v, 9) for v in RB.SPEED_CHOICES},
    }

    # ---- field completeness and mask consistency ------------------------
    required_input = {"scene", "command", "robot0", "robot1"}
    required_label = {
        "duration_seconds", "duration_steps", "exit_class", "primary_exit_reason",
        "effective_push_completed", "locally_settled", "settled_observed",
        "settled_after_abort", "exit_truth_low_speed", "window_truth_low_speed",
        "observation_validity", "robot0", "robot1",
    }
    required_robot_label = {
        "terminal_pose_change_own_start", "terminal_residual_velocity_own_start",
        "terminal_signed_distance", "terminal_tangential_offset",
        "segment_tangential_offset_change", "terminal_normal_sin_cos",
        "mask_terminal_geometry", "mask_terminal_patch_valid",
    }
    missing: List[str] = []
    mask_bad: List[str] = []
    nonfinite: List[str] = []
    for name in SPLITS:
        for record in executions[name]:
            if set(record["input"]) < required_input:
                missing.append(f"{record['record_id']}: input keys")
            if set(record["label"]) < required_label:
                missing.append(f"{record['record_id']}: label keys")
            for robot_id in (0, 1):
                label = record["label"][f"robot{robot_id}"]
                if set(label) < required_robot_label:
                    missing.append(f"{record['record_id']}: label.robot{robot_id} keys")
                expected = int(label["mask_terminal_patch_valid"] == 1)
                if int(label["mask_terminal_geometry"]) > expected:
                    mask_bad.append(record["record_id"])
                for key, value in label.items():
                    if isinstance(value, float) and not math.isfinite(value):
                        nonfinite.append(f"{record['record_id']}.{key}")
    result["checks"]["fields"] = {
        "missing_required": missing[:20],
        "mask_terminal_geometry_implies_patch": mask_bad[:20],
        "nonfinite_floats": nonfinite[:20],
        "ok": not missing and not mask_bad and not nonfinite,
    }

    # ---- exact replay from the artifact alone ---------------------------
    base_config = CB.default_config()
    option_config = CB.frozen_option_config()
    replay = []
    for name in SPLITS:
        for root in roots[name][:sample_roots]:
            stored = [record for record in executions[name]
                      if record["root_id"] == root["root_id"]]
            if not stored:
                continue
            stored = sorted(stored, key=lambda record: record["command_index"])[0]
            command_payload = commands_by_root[root["root_id"]]["commands"][stored["command_index"]]
            spec = root_from_metadata(root["meta"])
            command = RB.JointCommand(
                index=int(command_payload["command_index"]),
                alpha=(command_payload["alpha_rad"][0], command_payload["alpha_rad"][1]),
                speed=(command_payload["speed"][0], command_payload["speed"][1]),
                active_robots=tuple(int(v) for v in command_payload["active_robots"]),
            )
            snapshot, _config, _io, _ir, _report, _vel = build_snapshot(
                spec, VariantSpec("baseline"), base_config
            )
            record, _rows, _report = run_execution(
                RB.apply_command(spec, command), VariantSpec("baseline"), "V",
                option_config, base_config,
            )
            same = {
                "snapshot_sha256": CB.snapshot_sha256(snapshot) == stored["snapshot_sha256"],
                "exit_class": record["arm_outcome"]["exit_reason"] == stored["label"]["exit_class"],
                "duration_steps": int(record["metrics"]["wall_clock_steps"])
                == int(stored["label"]["duration_steps"]),
                "terminal_object": [float(v) for v in record["terminal_object"]]
                == [float(v) for v in stored["truth"]["object_terminal_world"]],
                "terminal_robots": [[float(v) for v in row] for row in record["terminal_robots"]]
                == [[float(v) for v in row] for row in stored["truth"]["robots_terminal_world"]],
            }
            replay.append({
                "record_id": stored["record_id"], "command_index": stored["command_index"],
                **same, "ok": all(same.values()),
            })
    result["checks"]["exact_replay"] = {
        "records": replay, "all_ok": all(item["ok"] for item in replay),
    }

    result["checks"]["sha256_recorded"] = ledger.get("sha256")
    result["checks"]["git_commit"] = ledger.get("git_commit")
    result["ok"] = all(
        value.get("ok", True) for key, value in result["checks"].items() if isinstance(value, dict)
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "outputs/option_prediction_mechanism_v1/"
                    "option-prediction-mechanism-v1/phase_b"),
    )
    parser.add_argument("--sample-roots", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    result = verify(Path(args.data_dir), sample_roots=args.sample_roots)
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print("VERIFY_B_OK" if result["ok"] else "VERIFY_B_FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
