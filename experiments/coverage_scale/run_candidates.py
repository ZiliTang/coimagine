"""Phase D3: fixed-candidate decision validation on the new test roots.

    python experiments/coverage_scale/run_candidates.py --run-id coverage-scale-20261013 \
        --tiers 8000 --widths 256,512 --seeds 0,1,2 --narrow

Computes the exact reference once (cached), then scores the main selection point of the first
tier, the final completed tier and the equal-size narrow-coverage control. Pairing is on
``(start_id, candidate_seed)`` and only rows where both sides have a feasible candidate enter
the paired means; the discarded counts are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.coverage_scale import candidate_eval as ce  # noqa: E402
from coimagine.coverage_scale.plan import load_registries  # noqa: E402

PAIRED_KEYS = (
    "chosen_true_return", "candidate_best_true_return", "uniform_candidate_true_return",
    "selected_bias", "candidate_mean_bias", "selection_excess", "unfiltered_regret",
    "rejection_rate", "feasible_best_true_return", "feasible_chosen_true_return",
    "misrejection_loss", "ranking_loss", "execution_regret",
)


def paired_compare(rows, label_a: str, label_b: str):
    def index(label):
        out = {}
        for row in rows:
            if str(row["label"]) != label:
                continue
            out[(row["start_id"], row["candidate_seed"], row["H"])] = row
        return out

    a, b = index(label_a), index(label_b)
    common = sorted(set(a) & set(b))
    entry = {
        "pair": f"{label_a}->{label_b}", "common_rows": len(common),
        "feasible_both": 0, "solved_to_unsolved": 0, "unsolved_to_solved": 0,
    }
    diffs = defaultdict(list)
    for key in common:
        left, right = a[key], b[key]
        if left["feasible_chosen_true_return"] is None or right["feasible_chosen_true_return"] is None:
            if left["feasible_chosen_true_return"] is None and right["feasible_chosen_true_return"] is not None:
                entry["unsolved_to_solved"] += 1
            elif left["feasible_chosen_true_return"] is not None and right["feasible_chosen_true_return"] is None:
                entry["solved_to_unsolved"] += 1
            continue
        entry["feasible_both"] += 1
        for name in PAIRED_KEYS:
            if left[name] is None or right[name] is None:
                continue
            diffs[name].append(float(right[name]) - float(left[name]))
    for name in PAIRED_KEYS:
        values = diffs.get(name, [])
        entry[f"delta_{name}"] = (sum(values) / len(values)) if values else None
        entry[f"delta_{name}_n"] = len(values)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--tiers", default="8000")
    parser.add_argument("--widths", default="256,512")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--narrow-tier", default="")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    registry = load_registries(run_dir / "roots", ("train", "val", "test", "narrow"))
    registry.activate()

    starts = [ce.make_start(root, config) for root in ce.select_test_starts(run_dir)]
    reference_path = out_dir / "reference.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())["reference"]
        reused = True
    else:
        started = time.time()
        reference = ce.build_reference(starts, config)
        reference_path.write_text(json.dumps({
            "reference": reference,
            "starts": [{"start_id": s["start_id"], "root_id": s["root_id"],
                        "scene_id": s["scene_id"], "family": s["family"], "mode": s["mode"]}
                       for s in starts],
            "horizons": list(ce.HORIZONS), "pool_size": ce.POOL_SIZE,
            "candidate_seeds": list(ce.CANDIDATE_SEEDS),
            "wall_clock": time.time() - started,
        }, indent=1, sort_keys=True))
        reused = False
    print(f"reference: {len(reference)} pools reused={reused}", flush=True)
    if args.reference_only:
        print("REFERENCE_OK")
        return 0

    tiers = [value for value in args.tiers.split(",") if value]
    widths = [int(value) for value in args.widths.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    groups = []
    for tier in tiers:
        for width in widths:
            for seed in seeds:
                cell = run_dir / "runs" / f"{tier}_{width}_seed{seed}"
                checkpoint = cell / "best_new_val_total.pt"
                if checkpoint.exists():
                    groups.append((f"{tier}/{width}/{seed}", checkpoint))
                else:
                    print(f"missing checkpoint {checkpoint}", flush=True)
    if args.narrow_tier:
        for width in widths:
            for seed in seeds:
                cell = run_dir / "narrow" / f"{args.narrow_tier}_{width}_seed{seed}"
                checkpoint = cell / "best_new_val_total.pt"
                if checkpoint.exists():
                    groups.append((f"narrow_{args.narrow_tier}/{width}/{seed}", checkpoint))
                else:
                    print(f"missing narrow checkpoint {checkpoint}", flush=True)

    all_rows = []
    for label, checkpoint in groups:
        started = time.time()
        model = ce.load_checkpoint(checkpoint)
        rows = ce.evaluate_model_on_starts(model, starts, reference, config, label=label)
        all_rows.extend(rows)
        print(f"{label}: {len(rows)} rows in {time.time() - started:.1f}s", flush=True)

    (out_dir / "rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    aggregate = ce.aggregate(all_rows)
    labels = sorted({str(row["label"]) for row in all_rows})
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append(paired_compare(all_rows, labels[i], labels[j]))
    payload = {
        "labels": labels, "rows": len(all_rows), "aggregate": aggregate, "paired": pairs,
        "reference_reused": reused,
        "horizons": list(ce.HORIZONS), "pool_size": ce.POOL_SIZE,
        "candidate_seeds": list(ce.CANDIDATE_SEEDS),
        "roots": [
            {"start_id": s["start_id"], "family": s["family"], "mode": s["mode"]}
            for s in starts
        ],
    }
    (out_dir / "aggregate.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({"labels": labels, "aggregate": aggregate}, indent=1))
    print("CANDIDATES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
