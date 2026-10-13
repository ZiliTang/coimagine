"""B follow-up: does the model get the *action differences* right, or only produce variation?

The round-3 report said "same-root true spread 55.6 mm vs predicted 72-82 mm". A reviewer correctly
pointed out that this only shows the output *varies* with the command: permuting the eight command
labels within a root leaves the spread exactly unchanged while destroying any decision value. This
script measures what actually matters:

* **action-difference error** -- per root and per target, the median absolute error of the
  *predicted change* between two commands against the true change;
* **ranking consistency** -- per root, the Spearman correlation between the true ordering of the
  root's commands and the predicted ordering, for each terminal-geometry column;
* **permutation control** -- the same two numbers after shuffling the predicted command labels
  inside each root. Spread is unchanged by construction; if ranking consistency collapses to ~0 the
  metric is doing its job and the spread number was not evidence.

    python experiments/option_prediction_mechanism_v1/phase_b/action_sensitivity.py \
        --data <records_tier2.jsonl> --checkpoints <tier2>/checkpoints --out <tier2>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.option_prediction_mechanism_v1.phase_b.dataset_b import (  # noqa: E402
    build_supervised,
    load_records,
)
from experiments.option_prediction_mechanism_v1.phase_b.evaluate_b import _percentiles  # noqa: E402
from experiments.option_prediction_mechanism_v1.phase_b.model import (  # noqa: E402
    GroupSpec,
    MultiHeadMLP,
    _standardise,
    predict,
)

COLUMNS = ["r0_signed_distance_m", "r0_tangential_offset_m", "r0_normal_sin", "r0_normal_cos",
           "r1_signed_distance_m", "r1_tangential_offset_m", "r1_normal_sin", "r1_normal_cos"]


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks with **average ranks for ties** (1-based), as Spearman requires.

    ``np.argsort(np.argsort(x))`` gives *ordinal* ranks, which break ties by array order and
    therefore manufacture correlation out of ties. With average ranks a tied block contributes no
    ordering information, which is the honest behaviour.
    """

    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=float)
    index = 0
    while index < len(array):
        end = index
        while end + 1 < len(array) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index:end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation with average-rank ties; ``nan`` when it is undefined.

    Undefined means fewer than three points, or one of the two sequences is constant (all ranks
    equal), in which case there is no ordering to correlate and no value is invented.

    This number is a **rank correlation**, not a "ranking accuracy": rho = 0.5 does not mean half
    the action comparisons are right.
    """

    if len(a) < 3:
        return float("nan")
    ranks_a = average_ranks(a)
    ranks_b = average_ranks(b)
    if np.std(ranks_a) < 1e-12 or np.std(ranks_b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def per_root_metrics(roots: Sequence[str], truth: np.ndarray, predicted: np.ndarray,
                     valid: np.ndarray, permutation: np.random.Generator | None = None):
    """Median pairwise action-difference error and Spearman rank correlation, per root."""

    diff_errors: Dict[str, List[float]] = {name: [] for name in COLUMNS}
    rank_correlations: Dict[str, List[float]] = {name: [] for name in COLUMNS}
    attempted: Dict[str, int] = {name: 0 for name in COLUMNS}
    constant_true: Dict[str, int] = {name: 0 for name in COLUMNS}
    roots = np.asarray(roots)
    for root in sorted(set(roots.tolist())):
        index = np.where(roots == root)[0]
        if len(index) < 3:
            continue
        predicted_block = predicted[index].copy()
        if permutation is not None:
            order = permutation.permutation(len(index))
            predicted_block = predicted_block[order]
        for column, name in enumerate(COLUMNS):
            keep = valid[index, column] > 0
            if keep.sum() < 3:
                continue
            true_values = truth[index][keep, column]
            predicted_values = predicted_block[keep, column]
            differences = []
            for i, j in itertools.combinations(range(len(true_values)), 2):
                differences.append(abs((predicted_values[i] - predicted_values[j])
                                       - (true_values[i] - true_values[j])))
            if differences:
                diff_errors[name].append(float(np.median(differences)))
            attempted[name] += 1
            correlation = spearman(true_values, predicted_values)
            if math.isfinite(correlation):
                rank_correlations[name].append(correlation)
            elif float(np.std(true_values)) < 1e-12:
                constant_true[name] += 1
    return {
        "action_difference_error": {name: _percentiles(values)
                                    for name, values in diff_errors.items()},
        "rank_correlation": {
            name: {"mean": float(np.mean(values)) if values else None,
                   "median": float(np.median(values)) if values else None,
                   "fraction_positive": float(np.mean(np.asarray(values) > 0)) if values else None,
                   # undefined roots are reported, never silently replaced by 0
                   "roots_defined": len(values),
                   "roots_attempted": attempted[name],
                   "roots_undefined_constant_truth": constant_true[name],
                   "roots_undefined_other": attempted[name] - len(values) - constant_true[name]}
            for name, values in rank_correlations.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--permutations", type=int, default=5)
    args = parser.parse_args()

    records = load_records(Path(args.data))
    data = build_supervised(records)
    _standardise(data)
    truth = data.targets["terminal_geometry"]["y_raw"]
    mask = data.targets["terminal_geometry"]["mask"]
    test = np.where(data.split == "test")[0]

    payload: Dict[str, object] = {"columns": COLUMNS, "permutations": args.permutations,
                                 "runs": {}}
    generator = np.random.default_rng(20261013)
    for seed in (int(value) for value in args.seeds.split(",")):
        checkpoint = Path(args.checkpoints) / f"deployable_seed{seed}.pt"
        if not checkpoint.exists():
            continue
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        specs = [GroupSpec(**spec) for spec in blob["specs"]]
        model = MultiHeadMLP(blob["input_dim"], specs)
        model.load_state_dict(blob["state_dict"])
        # the deployable arm keeps the privileged slots but pins them to zero, exactly as the
        # trainer does; feeding only ``data.x`` would change the input dimension
        model_input = np.hstack([data.x, np.zeros_like(data.x_priv)])
        predictions = predict(model, model_input, specs)
        raw = predictions["terminal_geometry"] * data.targets["terminal_geometry"]["std"] \
            + data.targets["terminal_geometry"]["mean"]
        entry = {
            "seed": seed,
            "true": per_root_metrics([data.roots[i] for i in test], truth[test], raw[test],
                                     mask[test]),
            "permuted": [],
        }
        for _ in range(args.permutations):
            entry["permuted"].append(
                per_root_metrics([data.roots[i] for i in test], truth[test], raw[test],
                                 mask[test], permutation=generator))
        payload["runs"][f"deployable_seed{seed}"] = entry
        core = entry["true"]
        print(f"seed {seed}: r0 signed-distance action-difference MAE p50 "
              f"{core['action_difference_error']['r0_signed_distance_m']['p50']:.4f} m, "
              f"rank rho mean {core['rank_correlation']['r0_signed_distance_m']['mean']:.3f} | "
              f"permuted rank rho mean "
              f"{np.mean([p['rank_correlation']['r0_signed_distance_m']['mean'] for p in entry['permuted']]):.3f}",
              flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "action_sensitivity.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    print("ACTION_SENSITIVITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
