"""Comparison cases for phase D3: Branch vs Old, improved and not improved.

    python experiments/data_upgrade/run_cases.py --run-id <id> --tag origval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--tag", default="origval")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (run_dir / f"closed_loop_groups_{args.tag}" / "episodes.jsonl").open()]
    old = {
        (r["scene_id"], r["episode_seed"], r["H"]): r
        for r in rows if r["data_group"] == f"Old@{args.tag}"
    }
    branch = [r for r in rows if r["data_group"] == f"Old+Branch@{args.tag}"]
    pairs = []
    for row in branch:
        key = (row["scene_id"], row["episode_seed"], row["H"])
        if key not in old:
            continue
        baseline = old[key]
        pairs.append(
            {
                "scene_id": row["scene_id"],
                "episode_seed": row["episode_seed"],
                "H": row["H"],
                "model_seed": row["model_seed"],
                "old_J": baseline["J"],
                "branch_J": row["J"],
                "delta_J": row["J"] - baseline["J"],
                "old_success": baseline["success"],
                "branch_success": row["success"],
                "old_windows": baseline["windows"],
                "branch_windows": row["windows"],
                "old_terminal_distance": baseline["terminal_distance"],
                "branch_terminal_distance": row["terminal_distance"],
                "old_no_feasible": baseline["no_feasible_plan"],
                "branch_no_feasible": row["no_feasible_plan"],
            }
        )
    pairs.sort(key=lambda row: -row["delta_J"])
    improved = [row for row in pairs if row["delta_J"] > 0][:2]
    degraded = [row for row in pairs if row["delta_J"] <= 0][-2:]
    summary = {
        "tag": args.tag,
        "pairs": len(pairs),
        "improved_pairs": sum(1 for row in pairs if row["delta_J"] > 0),
        "degraded_pairs": sum(1 for row in pairs if row["delta_J"] < 0),
        "equal_pairs": sum(1 for row in pairs if row["delta_J"] == 0),
        "mean_delta_J": sum(row["delta_J"] for row in pairs) / max(1, len(pairs)),
        "improved_cases": improved,
        "degraded_cases": degraded,
    }
    (out_dir / f"cases_{args.tag}.json").write_text(json.dumps({"summary": {k: v for k, v in summary.items() if not k.endswith("cases")}, "improved": improved, "degraded": degraded}, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("cases")}, indent=2))
    for label, cases in (("improved", improved), ("degraded", degraded)):
        for row in cases:
            print(label, {k: row[k] for k in ("scene_id", "episode_seed", "model_seed", "old_J", "branch_J", "delta_J", "old_terminal_distance", "branch_terminal_distance")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
