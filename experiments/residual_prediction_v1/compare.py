"""Paired root-level bootstrap between configurations, from the saved per-window predictions.

The unit of resampling is the **root**: windows and actions of one root are not independent samples.
Each comparison is paired per window because both configurations are scored on the same panel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/residual_prediction_v1"
MOTION_THRESHOLD = 0.05


#: the midpoint columns inside the 8-column geometry vector
GEOMETRY_MIDPOINTS = [0, 1, 4, 5]


def midpoint_error(geometry: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                   robot: int) -> np.ndarray:
    """``geometry`` and ``reference`` are 8-column vectors; the error tables use 4 midpoints."""

    geometry = np.asarray(geometry)[:, GEOMETRY_MIDPOINTS]
    columns = [0, 1] if robot == 0 else [2, 3]
    keep = (mask[:, columns[0]] > 0) & (mask[:, columns[1]] > 0)
    error = np.hypot(geometry[:, columns[0]] - truth[:, columns[0]],
                     geometry[:, columns[1]] - truth[:, columns[1]])
    return np.where(keep, error, np.nan)


def midpoint_error4(midpoints: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                    robot: int) -> np.ndarray:
    """The same error for an already-4-column midpoint vector (the hold baseline)."""

    columns = [0, 1] if robot == 0 else [2, 3]
    keep = (mask[:, columns[0]] > 0) & (mask[:, columns[1]] > 0)
    error = np.hypot(midpoints[:, columns[0]] - truth[:, columns[0]],
                     midpoints[:, columns[1]] - truth[:, columns[1]])
    return np.where(keep, error, np.nan)


def bootstrap(roots: np.ndarray, difference: np.ndarray, *, draws: int = 2000,
              seed: int = 0) -> Dict[str, object]:
    keep = np.isfinite(difference)
    roots, difference = roots[keep], difference[keep]
    unique = np.unique(roots)
    by_root = {root: np.where(roots == root)[0] for root in unique.tolist()}
    generator = np.random.default_rng(seed)
    values = np.empty(draws)
    for index in range(draws):
        picked = generator.choice(len(unique), size=len(unique), replace=True)
        selection = np.concatenate([by_root[unique[p]] for p in picked])
        values[index] = float(np.median(difference[selection]))
    return {"roots": int(len(unique)), "windows": int(keep.sum()), "draws": draws,
            "observed_median_difference_cm": float(np.median(difference) * 100),
            "ci95_low_cm": float(np.quantile(values, 0.025) * 100),
            "ci95_high_cm": float(np.quantile(values, 0.975) * 100),
            "fraction_draws_below_zero": float((values < 0).mean()),
            "interpretation": "negative favours the first configuration"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    out_dir = Path(args.out)
    files = sorted(out_dir.glob(f"predictions_*_{args.split}.npz"))
    if not files:
        print("no saved predictions; run evaluate.py first")
        return 1
    loaded = {}
    for path in files:
        name = path.name.replace("predictions_", "").replace(f"_{args.split}.npz", "")
        loaded[name] = dict(np.load(path, allow_pickle=True))
    result: Dict[str, object] = {"split": args.split, "comparisons": {}}
    pairs: List[Tuple[str, str]] = []
    # residual against the corrected direct control, per seed
    for seed in (0, 1, 2):
        residual = f"mlp_history_residual_per_group_seed{seed}"
        direct = f"mlp_history_direct_per_group_seed{seed}"
        if residual in loaded and direct in loaded:
            pairs.append((residual, direct))
    # residual against the hold baseline of the same checkpoint (same reference rule)
    for seed in (0, 1, 2):
        residual = f"mlp_history_residual_per_group_seed{seed}"
        if residual in loaded:
            pairs.append((residual, "hold"))
    for first, second in pairs:
        if first not in loaded:
            continue
        reference = loaded[first]
        roots = reference["root"].astype(str)
        end = reference["end"]
        for robot in (0, 1):
            error_first = midpoint_error(reference["next"], reference["next_truth"],
                                         reference["next_mask"], robot)
            if second == "hold":
                # the stored reference is already a 4-column midpoint vector
                error_second = midpoint_error4(reference["reference"], reference["next_truth"],
                                               reference["next_mask"], robot)
            else:
                other = loaded[second]
                error_second = midpoint_error(other["next"], other["next_truth"],
                                              other["next_mask"], robot)
            key = f"{first}_vs_{second}|r{robot}|t>0"
            result["comparisons"][key] = bootstrap(roots, error_first - error_second,
                                                   draws=args.draws)
            zero = end == 0
            result["comparisons"][f"{first}_vs_{second}|r{robot}|t=0"] = bootstrap(
                roots[zero], (error_first - error_second)[zero], draws=args.draws)
    (out_dir / f"comparisons_{args.split}.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=float))
    for key, value in sorted(result["comparisons"].items()):
        print(f"  {key:56s} median {value['observed_median_difference_cm']:+.4f} cm "
              f"CI [{value['ci95_low_cm']:+.4f}, {value['ci95_high_cm']:+.4f}] "
              f"roots={value['roots']}")
    print("COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
