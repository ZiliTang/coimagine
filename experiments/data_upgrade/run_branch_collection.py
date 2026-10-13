"""Phase C2-C5: collect guided trajectories, anchors, branch data and the random control.

    python experiments/data_upgrade/run_branch_collection.py --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.records import CollectionConfig, SPLIT_SCENES
from coimagine.data_upgrade.branch_data import (
    BRANCH_CONTINUATION,
    BRANCHES_PER_ANCHOR,
    MAX_ANCHORS,
    TEST_SCENES,
    TRAIN_SCENES,
    VAL_SCENES,
    build_anchors,
    collect_branches,
    collect_guides,
    collect_random_add,
)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_LIMITS = {"train": 12, "val": 4, "test": 4}


def _order_key(record) -> str:
    return hashlib.sha256(
        json.dumps(
            [record["scene_id"], record["state"], record["joint_skill_specs"], record["next_state"]],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _transition_key(record) -> str:
    """Full-input / action / output / duration / termination-label identity."""

    return hashlib.sha256(
        json.dumps(
            [
                record["state"], record["joint_skill_specs"], record["next_state"],
                round(float(record["duration"]), 12), record["termination_class"],
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _conflict_key(record) -> str:
    """Same visible input, action and output: any label disagreement is a conflict."""

    return hashlib.sha256(
        json.dumps(
            [record["state"], record["joint_skill_specs"], record["next_state"]], sort_keys=True
        ).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    parser.add_argument("--max-anchors", type=int, default=MAX_ANCHORS)
    parser.add_argument("--branches-per-anchor", type=int, default=BRANCHES_PER_ANCHOR)
    parser.add_argument("--skip-guides", action="store_true", help="reuse a previous guides.json")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "branch_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.source_run
    config = default_config()
    started = time.time()

    split_of = {scene: split for split, scenes in SPLIT_SCENES.items() for scene in scenes}
    all_scenes = list(TRAIN_SCENES) + list(VAL_SCENES) + list(TEST_SCENES)
    guide_report_path = out_dir / "guides.json"
    if args.skip_guides and guide_report_path.exists():
        report = json.loads(guide_report_path.read_text())
        runs = None
    else:
        collected = collect_guides(
            all_scenes, config=config, split_for_scene=lambda scene: split_of[scene]
        )
        report = {
            "attempts": collected["attempts"],
            "episodes_spent": collected["episodes_spent"],
            "budget": collected["budget"],
            "runs": [
                {
                    "teacher_episode_id": run["teacher_episode_id"],
                    "scene_id": run["scene_id"],
                    "split": run["split"],
                    "episode_seed": run["episode_seed"],
                    "success": run["success"],
                    "termination": run["termination"],
                    "windows": len(run["windows"]),
                    "wall_clock": run["wall_clock"],
                }
                for run in collected["runs"]
            ],
        }
        guide_report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        # persist the in-memory run payloads for the next steps in this same process
        import pickle

        with (out_dir / "guides.pkl").open("wb") as handle:
            pickle.dump(collected["runs"], handle)
        runs = collected["runs"]

    if runs is None:
        import pickle

        with (out_dir / "guides.pkl").open("rb") as handle:
            runs = pickle.load(handle)

    # split budget: at most 12 train / 4 val / 4 test guided trajectories
    used = defaultdict(int)
    kept_runs = []
    dropped = []
    for run in runs:
        split = str(run["split"])
        if used[split] < SPLIT_LIMITS[split]:
            used[split] += 1
            kept_runs.append(run)
        else:
            dropped.append(run["teacher_episode_id"])

    planner = None
    from coimagine.continuous2d.environment import Continuous2DEnv

    planner = Continuous2DEnv(config)
    live = Continuous2DEnv(config)
    anchors = []
    branch_windows = []
    guide_windows = []
    for run in kept_runs:
        guide_windows.extend(run["windows"])
        run_anchors = build_anchors(run, max_anchors=args.max_anchors)
        for anchor in run_anchors:
            branch_records = collect_branches(anchor, config=config, planner=planner, live=live)
            branch_windows.extend(branch_records)
            serialisable = {
                key: value for key, value in anchor.items() if key not in ("snapshot", "observation")
            }
            from coimagine.continuous2d.skills import joint_spec_to_dicts

            serialisable["expert_joint_spec"] = joint_spec_to_dicts(anchor["expert_joint_spec"])
            anchors.append(serialisable)
        print(
            f"guide {run['teacher_episode_id']}: windows={len(run['windows'])} "
            f"anchors={len(run_anchors)} branch_windows={len(branch_windows)}",
            flush=True,
        )

    # C5: equal-count random control for the train scenes
    branch_by_scene = defaultdict(int)
    for record in branch_windows:
        branch_by_scene[str(record["scene_id"])] += 1
    branch_total = sum(branch_by_scene.values())
    attempt_cap = max(1, branch_total * 3)
    random_windows = []
    random_reports = []
    for scene in TRAIN_SCENES:
        target = branch_by_scene.get(scene, 0)
        records, row = collect_random_add(
            scene, target, config=config, attempt_cap=attempt_cap
        )
        random_windows.extend(records)
        random_reports.append(row)
        print(f"random_add {scene}: target={target} accepted={row['accepted_windows']} shortfall={row['shortfall']}", flush=True)

    # dedupe inside each new source and against the original training data
    original = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
    original_train_keys = {_transition_key(record) for record in original if record["split"] == "train"}
    original_all_keys = {_transition_key(record) for record in original}

    def dedupe(records, label):
        seen = set()
        kept = []
        conflicts = defaultdict(set)
        identical_to_original = 0
        for record in sorted(records, key=_order_key):
            key = _transition_key(record)
            if key in original_all_keys:
                identical_to_original += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            kept.append(record)
            conflicts[_conflict_key(record)].add(
                (round(float(record["duration"]), 6), str(record["termination_class"]))
            )
        conflicting = {key: sorted(values) for key, values in conflicts.items() if len(values) > 1}
        return kept, {
            "label": label,
            "input_windows": len(records),
            "kept": len(kept),
            "identical_to_original": identical_to_original,
            "duplicate_within_group": len(records) - identical_to_original - len(kept),
            "conflicting_inputs": len(conflicting),
            "conflict_examples": list(conflicting.items())[:5],
        }

    branch_kept, branch_dedupe = dedupe(branch_windows, "branch")
    guide_kept, guide_dedupe = dedupe(guide_windows, "guide")
    random_kept, random_dedupe = dedupe(random_windows, "random_add")

    branch_train = [r for r in branch_kept if r["split"] == "train"]
    random_kept = [r for r in random_kept if r["split"] == "train"]

    # equal-count matching: per train scene take the smaller of the two, by fixed hash order
    per_scene = {}
    matched_branch, matched_random = [], []
    for scene in TRAIN_SCENES:
        b = sorted([r for r in branch_train if r["scene_id"] == scene], key=_order_key)
        rnd = sorted([r for r in random_kept if r["scene_id"] == scene], key=_order_key)
        take = min(len(b), len(rnd))
        matched_branch.extend(b[:take])
        matched_random.extend(rnd[:take])
        per_scene[scene] = {"branch": len(b), "random": len(rnd), "matched": take}

    def write_jsonl(path, records):
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    write_jsonl(out_dir / "guide_windows.jsonl", guide_kept)
    write_jsonl(out_dir / "branch_windows.jsonl", branch_kept)
    write_jsonl(out_dir / "random_add_windows.jsonl", random_kept)
    write_jsonl(out_dir / "branch_train_matched.jsonl", matched_branch)
    write_jsonl(out_dir / "random_train_matched.jsonl", matched_random)
    write_jsonl(out_dir / "anchors.jsonl", anchors)

    summary = {
        "guides": report,
        "guide_split_usage": dict(used),
        "guides_dropped_over_budget": dropped,
        "branch_windows": len(branch_kept),
        "guide_windows": len(guide_kept),
        "random_windows": len(random_kept),
        "dedupe": [guide_dedupe, branch_dedupe, random_dedupe],
        "per_scene": per_scene,
        "matched_total": {"branch": len(matched_branch), "random": len(matched_random)},
        "random_reports": random_reports,
        "budget": {
            "branch_per_guide_max": args.max_anchors * args.branches_per_anchor * (1 + BRANCH_CONTINUATION),
            "train_guides_max": SPLIT_LIMITS["train"],
        },
        "wall_clock": time.time() - started,
    }
    (out_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: summary[k] for k in (
        "guide_split_usage", "branch_windows", "guide_windows", "random_windows",
        "dedupe", "matched_total", "wall_clock",
    )}, indent=2))
    print("BRANCH_COLLECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
