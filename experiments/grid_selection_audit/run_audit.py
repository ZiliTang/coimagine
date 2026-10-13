"""Run the Phase B legacy-grid candidate-selection audit.

    python experiments/grid_selection_audit/run_audit.py            # full frozen protocol
    python experiments/grid_selection_audit/run_audit.py --smoke    # 2-scene smoke run

Writes ``outputs/migration_v1/<run_id>/grid_selection_audit/`` with candidates.jsonl,
summary.csv, checks.json, two figures and conclusion.md. It never modifies a checkpoint
or any legacy production file; the checkpoint hash is recorded before and after.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from coimagine.diagnostics.grid_selection_audit import run_audit, write_outputs

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/grid/world_model/belief.npz")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--lead-in", type=int, default=2)
    parser.add_argument("--scene-seed", type=int, default=55_000)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out-root", default="outputs/migration_v1")
    parser.add_argument("--smoke", action="store_true", help="2 scenes, depth 1, K=16")
    args = parser.parse_args()

    if args.smoke:
        n_scenes, depths, sizes = 2, (1,), (16,)
    else:
        n_scenes, depths, sizes = args.scenes, (1, 2, 3), (16, 64, 128)

    started = time.time()
    result = run_audit(
        checkpoint=args.checkpoint,
        n_scenes=n_scenes,
        lead_in=args.lead_in,
        depths=depths,
        sizes=sizes,
        scene_seed=args.scene_seed,
    )
    out_dir = ROOT / args.out_root / args.run_id / "grid_selection_audit"
    written = write_outputs(result, str(out_dir))
    elapsed = time.time() - started

    aggregate = [row for row in result["summary"] if row["scope"] == "all"]
    print(f"run_id={args.run_id}  elapsed={elapsed:.1f}s  files={len(written)}")
    for row in aggregate:
        print(
            f"  H={row['H']} K={row['K']}  n={row['n_candidates']}  "
            f"selected_bias={row['selected_bias']:+.4f}  "
            f"candidate_mean_bias={row['candidate_mean_bias']:+.4f}  "
            f"selection_excess={row['selection_excess']:+.4f}  "
            f"regret={row['candidate_regret']:.4f}"
        )
    checks = result["checks"]
    print("checks:", json.dumps({k: v for k, v in checks.items() if k != "nesting"}, ensure_ascii=False))
    print(f"outputs: {out_dir.relative_to(ROOT)}")
    ok = (
        checks["mock_ok"]
        and checks["order_invariant"]
        and checks["checkpoint_unchanged"]
        and all(checks["nesting"].values())
    )
    print("PHASE_B_CHECKS_OK" if ok else "PHASE_B_CHECKS_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
