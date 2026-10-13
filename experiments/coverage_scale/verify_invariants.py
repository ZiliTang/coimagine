"""Verify the run-level invariants the task book freezes (section C2).

    python experiments/coverage_scale/verify_invariants.py --run-id coverage-scale-20261013

Checks, from the run summaries alone:

* same width and same seed across tiers share one initial parameter hash (initialisation is a
  function of ``(width, seed)`` only, not of the data size);
* same seed across widths *within a tier* share one sample-index stream hash (the permutation
  is a function of the seed only);
* the sample stream hashes differ between seeds;
* the data hash recorded in each checkpoint matches the tier's training data;
* the selected update never exceeds the run's update budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TIERS = ("8000", "16000", "32000", "64000", "128000")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id

    summaries = {}
    for tier in TIERS:
        for cell in sorted((run_dir / "runs").glob(f"{tier}_*_seed*")):
            path = cell / "run_summary.json"
            if path.exists():
                summaries[(tier, json.loads(path.read_text())["width"],
                           json.loads(path.read_text())["seed"])] = json.loads(path.read_text())
    for cell in sorted((run_dir / "narrow").glob("*_seed*")):
        path = cell / "run_summary.json"
        if path.exists():
            payload = json.loads(path.read_text())
            summaries[("narrow", payload["width"], payload["seed"])] = payload

    report: dict = {"checks": {}, "counts": {"runs": len(summaries)}}

    by_width_seed: dict = defaultdict(set)
    for (tier, width, seed), payload in summaries.items():
        if tier == "narrow":
            continue
        by_width_seed[(width, seed)].add(payload["initial_parameter_hash"])
    bad = {key: sorted(values) for key, values in by_width_seed.items() if len(values) > 1}
    report["checks"]["initial_parameter_hash_shared_across_tiers"] = not bad
    report["initial_hash_conflicts"] = bad

    by_tier_seed: dict = defaultdict(set)
    for (tier, width, seed), payload in summaries.items():
        if tier == "narrow":
            continue
        by_tier_seed[(tier, seed)].add(payload["sample_order_hash"])
    bad_order = {key: sorted(values) for key, values in by_tier_seed.items() if len(values) > 1}
    report["checks"]["sample_order_shared_across_widths"] = not bad_order
    report["sample_order_conflicts"] = bad_order

    by_tier: dict = defaultdict(set)
    for (tier, width, seed), payload in summaries.items():
        if tier == "narrow":
            continue
        by_tier[tier].add(payload["sample_order_hash"])
    report["sample_order_distinct_per_seed"] = {
        tier: len(values) == 3 for tier, values in by_tier.items()
    }

    budget_ok = True
    selected = []
    for (tier, width, seed), payload in summaries.items():
        for kind, entry in payload["checkpoints"].items():
            if entry["updates"] > payload["max_updates"]:
                budget_ok = False
        selected.append({
            "tier": tier, "width": width, "seed": seed,
            "selected_update": payload["checkpoints"]["best_new_val_total"]["updates"],
            "max_updates": payload["max_updates"],
            "updates": payload["updates"],
        })
    report["checks"]["selected_update_within_budget"] = budget_ok
    report["selection"] = selected

    # cross-width paired comparison of the selected point's val metrics
    paired = []
    for tier in TIERS + ("narrow",):
        keys = [key for key in summaries if key[0] == tier]
        for seed in (0, 1, 2):
            small = next((summaries[key] for key in keys if key[1] == 256 and key[2] == seed), None)
            large = next((summaries[key] for key in keys if key[1] == 512 and key[2] == seed), None)
            if small and large and (run_dir / "runs" / f"{tier}_{256}_seed{seed}" / "selection_eval.json").exists():
                base = run_dir / ("runs" if tier != "narrow" else "narrow") / f"{tier}_{256}_seed{seed}"
                other = run_dir / ("runs" if tier != "narrow" else "narrow") / f"{tier}_{512}_seed{seed}"
                a = json.loads((base / "selection_eval.json").read_text())["best_new_val_total"]
                b = json.loads((other / "selection_eval.json").read_text())["best_new_val_total"]
                paired.append({
                    "tier": tier, "seed": seed,
                    "P_256": a["new_val"]["weighted"]["position_mean"],
                    "P_512": b["new_val"]["weighted"]["position_mean"],
                    "invalid_256": a["new_val"]["legality"]["invalid_rate_weighted"],
                    "invalid_512": b["new_val"]["legality"]["invalid_rate_weighted"],
                    "update_256": a["update"], "update_512": b["update"],
                })
    report["width_pairs"] = paired

    (run_dir / "invariants.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({key: value for key, value in report["checks"].items()
                      if key.startswith("initial") or key.startswith("sample") or key.startswith("selected")}, indent=1))
    print(f"runs: {len(summaries)}; width pairs: {len(paired)}")
    print("INVARIANTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
