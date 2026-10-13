"""Phase A3: put every terminal label on one timestamp, the option's exit instant T.

The phase-B records labelled the terminal observation with `_exit_block`'s "last available
observation", which is the last **logged** period. The stage machine breaks before appending that
period's row, so the last logged row is one control period before the exit; and the fallback could
reach further back when the patch was not visible. This script therefore:

1. replays each existing execution exactly (see ``replay.py``) and reads the observation captured
   **at T**, the same instant the recorded terminal state was taken;
2. checks every replay against the original record -- exit reason, duration in steps, terminal
   object and both terminal robot states -- and reports mismatches instead of replacing them;
3. quantifies how far the last logged row is from T, which is the size of the error the old labelling
   would have introduced;
4. classifies every record: (1) a valid patch at T, (2) genuinely no visible patch at T,
   (3) unrecoverable (replay mismatch or missing terminal observation).

No new initial state and no new command is created: the roots come from the frozen salt and the
commands come from the records themselves.

    python experiments/observation_terminal_v1/audit_labels.py --verify 16
    python experiments/observation_terminal_v1/audit_labels.py --all --workers 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / ("outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/"
                 "phase_b_tier2")
OUT_ROOT = ROOT / "outputs/observation_terminal_v1"
SCALE = {"train": 512, "val": 64, "test": 64}

_WORKER: Dict[str, object] = {}


def _init_worker() -> None:
    from experiments.observation_terminal_v1 import replay

    roots, _reports = replay.load_roots(SCALE)
    _WORKER["roots"] = roots
    _WORKER["replay"] = replay


def _work(record: dict) -> dict:
    replay = _WORKER["replay"]
    roots = _WORKER["roots"]
    outcome = replay.replay(record, roots)
    summary = {}
    for robot_id in (0, 1):
        observation = outcome["terminal_observations"].get(str(robot_id))
        summary[str(robot_id)] = replay.observation_summary(observation)
    # the last logged row, for the T-versus-last-logged comparison
    last = outcome.get("last_logged_period") or {}
    last_observed = last.get("observed") or {}
    comparison = {}
    for robot_id in (0, 1):
        entry = last_observed.get(str(robot_id)) or {}
        observed = summary[str(robot_id)]
        comparison[str(robot_id)] = {
            "last_logged_patch_points": entry.get("patch_points"),
            "t_patch_points": observed.get("patch_points") if observed.get("present") else None,
            "last_logged_distance": entry.get("patch_distance"),
            "t_distance": observed.get("patch_distance") if observed.get("present") else None,
            "last_logged_stage": last.get("stage"),
        }
        if entry.get("patch_normal") and observed.get("patch_normal_robot") is not None:
            # both normals must be compared in the SAME frame: the logged one is world-frame, the
            # T one is robot-frame, so rotate the logged one by the robot's angle that period
            last_own = entry.get("own") or [0.0, 0.0, 0.0]
            world_last = entry["patch_normal"]
            cos, sin = math.cos(-last_own[2]), math.sin(-last_own[2])
            last_robot_xy = (world_last[0] * cos - world_last[1] * sin,
                             world_last[0] * sin + world_last[1] * cos)
            angle_last = math.atan2(last_robot_xy[1], last_robot_xy[0])
            angle_t = math.atan2(observed["patch_normal_robot"][1],
                                 observed["patch_normal_robot"][0])
            difference = (angle_t - angle_last + math.pi) % (2 * math.pi) - math.pi
            comparison[str(robot_id)]["normal_angle_difference_deg"] = math.degrees(difference)
    # classification
    valid = [summary[str(robot_id)].get("patch_valid", False) for robot_id in (0, 1)]
    if not outcome["ok"]:
        klass = 3
    elif any(valid):
        klass = 1
    else:
        klass = 2
    return {
        "record_id": record["record_id"], "root_id": record["root_id"],
        "split": record["split"], "command_index": record["command_index"],
        "interaction_class": record["interaction_class"],
        "exit_class": record["label"]["exit_class"],
        "primary_exit_reason": record["label"]["primary_exit_reason"],
        "replay_ok": outcome["ok"], "replay_check": outcome["check"],
        "category": klass,
        "exit_reason_replayed": outcome["exit_reason"],
        "wall_clock_steps": outcome["wall_clock_steps"],
        "duration_seconds": outcome["duration_seconds"],
        "locally_settled": outcome["locally_settled"],
        "settled_observed": outcome["settled_observed"],
        "settled_after_abort": outcome["settled_after_abort"],
        "terminal_object": outcome["terminal_object"],
        "terminal_robots": outcome["terminal_robots"],
        "terminal_observation": summary,
        "last_logged_comparison": comparison,
    }


def load_records(split: Optional[str] = None) -> List[dict]:
    records: List[dict] = []
    for name in (("train", "val", "test") if split is None else (split,)):
        path = SOURCE / f"executions_{name}.jsonl"
        records.extend(json.loads(line) for line in path.open())
    return records


def _select_verification_sample(records: List[dict], count: int) -> List[dict]:
    """Cover the categories that actually exist: exit class x terminal visibility."""

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        visible = any(record["label"][f"robot{robot_id}"]["mask_terminal_patch_valid"]
                      for robot_id in (0, 1))
        buckets[f"{record['label']['exit_class']}|{'visible' if visible else 'invisible'}"].append(record)
    sample: List[dict] = []
    keys = sorted(buckets)
    index = 0
    while len(sample) < count and any(buckets[key] for key in keys):
        key = keys[index % len(keys)]
        if buckets[key]:
            sample.append(buckets[key].pop(0))
        index += 1
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    records = load_records()
    print(f"records available: {len(records)}")

    if args.verify:
        sample = _select_verification_sample(records, args.verify)
        _init_worker()
        started = time.time()
        results = [_work(record) for record in sample]
        payload = {
            "sample": len(sample), "replay_ok": sum(r["replay_ok"] for r in results),
            "categories": dict(Counter(r["category"] for r in results)),
            "classes": dict(Counter(r["exit_class"] for r in results)),
            "seconds": time.time() - started,
            "records": [{key: r[key] for key in
                         ("record_id", "replay_ok", "replay_check", "category", "exit_class",
                          "last_logged_comparison")} for r in results],
        }
        (out_root / "replay_verification.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=float))
        print(json.dumps({key: payload[key] for key in
                          ("sample", "replay_ok", "categories", "classes", "seconds")}, indent=1))
        print("VERIFY_REPLAY_OK" if payload["replay_ok"] == len(sample) else "VERIFY_REPLAY_FAILED")
        return 0

    if not args.all:
        parser.error("pass --verify N or --all")
    import multiprocessing as mp

    started = time.time()
    with mp.get_context("fork").Pool(processes=args.workers, initializer=_init_worker) as pool:
        results = []
        for index, item in enumerate(pool.imap(_work, records, chunksize=8)):
            results.append(item)
            if (index + 1) % 512 == 0:
                print(f"  replayed {index + 1}/{len(records)}", flush=True)
    by_split: Dict[str, List[dict]] = defaultdict(list)
    for item in results:
        by_split[item["split"]].append(item)
    for split, items in by_split.items():
        with (out_root / f"terminal_observations_{split}.jsonl").open("w") as handle:
            for item in sorted(items, key=lambda row: row["record_id"]):
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    ledger = {
        "records": len(results), "seconds": time.time() - started, "workers": args.workers,
        "replay_ok": sum(item["replay_ok"] for item in results),
        "category_counts": dict(Counter(item["category"] for item in results)),
        "exit_classes": dict(Counter(item["exit_class"] for item in results)),
        "per_split": {split: len(items) for split, items in by_split.items()},
        "category_by_exit_class": {
            f"{exit_class}|cat{category}": count
            for (exit_class, category), count in sorted(Counter(
                (item["exit_class"], item["category"]) for item in results).items())},
    }
    (out_root / "label_audit_ledger.json").write_text(
        json.dumps(ledger, indent=1, sort_keys=True, default=float))
    print(json.dumps(ledger, indent=1, default=float))
    print("AUDIT_LABELS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
