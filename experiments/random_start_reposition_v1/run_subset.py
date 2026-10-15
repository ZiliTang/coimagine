"""Find the frozen S4 diagnostic subset: instances with **evidence** that the task is solvable.

Rule, declared before the scan: run the scripted demo policy (the same one used for the S3
clips) over the whole frozen L0 train panel, then the L0 validation panel, and keep the
instances it succeeds on, in panel order. If fewer than 8 exist, the scan is extended to the
test panel with its own report; if still fewer than 8, the subset is padded with the
shortest-goal-distance remaining instances and the padding is recorded explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_episodes", ROOT / "experiments/random_start_reposition_v1/run_episodes.py")
episodes_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(episodes_module)

from coimagine.random_start import layout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs/random_start_reposition_v1"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--target", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    results = []
    for split in ("train", "validation"):
        panel, _ = layout.build_split("L0", split, count=args.count)
        for instance in panel:
            row = episodes_module.run_episode(instance, "scripted", record=False)
            row.pop("environment", None)
            row.pop("decisions", None)
            results.append(row)
            print(f"{instance.name} success={row['success']} skills={row['skills_used']} "
                  f"distance={row['final_goal_distance']:.3f}", flush=True)
    successes = [row["instance"] for row in results if row["success"]]
    selected = successes[: args.target]
    padded: list = []
    if len(selected) < args.target:
        remaining = [row for row in results if row["instance"] not in selected]
        remaining.sort(key=lambda row: row["start_goal_distance"])
        for row in remaining:
            if len(selected) >= args.target:
                break
            selected.append(row["instance"])
            padded.append(row["instance"])
    report = {"attempted": len(results), "successes": successes,
              "selected": selected, "padded_without_success_evidence": padded,
              "rows": results}
    (out / "results" / "diagnostic_subset.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"attempted": len(results), "successes": len(successes),
                      "selected": selected, "padded": padded}, indent=1))
    print("SUBSET_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
