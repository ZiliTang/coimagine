"""Windows and fixed evaluation panels for the residual round.

Reuses the round-6 window builder (which already had the left-padding readout fixed) and adds the
pieces this round needs:

* the current midpoint of each robot, taken from the last **real** token via ``last_valid_index`` --
  the same helper the production models use, never a copy of ``valid.sum() - 1``;
* the visibility masks of the next-step and terminal midpoints, so no invisible placeholder can enter
  a loss, a statistic or a reported error;
* the residual reference, which is the current midpoint when that robot's patch is visible and the
  fixed origin of R_i0 otherwise.

Panels are drawn once and saved: the training panel and the validation panel are built from
**different records**, so "same protocol" never means "same data".
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example  # noqa: E402
from experiments.night_sequence_v1.models import last_valid_index  # noqa: E402
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    MIDPOINT_COLUMNS,
    visible_reference,
)

DATA = ROOT / "outputs/night_sequence_v1"
OUT = ROOT / "outputs/residual_prediction_v1"
TIER_SPLITS = {"d0": ["d0", "val", "test"], "d1": ["d0", "d1ext", "val", "test"]}
KIND = {name: kind for name, _w, kind in GROUPS}


@dataclass
class Batch:
    """Everything one window needs: inputs, both midpoint targets, and their masks."""

    window: torch.Tensor
    token_valid: torch.Tensor
    current_mid: np.ndarray          # (n, 4) metres, in R_i0
    current_visible: np.ndarray      # (n, 2) bool per robot
    current_speed: np.ndarray        # (n, 2) robot translation speed at the window end, m/s
    next_mid: np.ndarray             # (n, 4)
    next_mask: np.ndarray            # (n, 4)
    term_mid: np.ndarray             # (n, 4)
    term_mask: np.ndarray            # (n, 4)
    targets: Dict[str, torch.Tensor]  # the full multi-task label set
    masks: Dict[str, torch.Tensor]
    reference: np.ndarray            # (n, 4) the residual reference
    index: np.ndarray
    ends: np.ndarray

    def __len__(self) -> int:
        return int(self.window.shape[0])


def build_batch(tokens: Tokens, index: np.ndarray, ends: np.ndarray) -> Batch:
    examples = [build_example(tokens, int(i), int(e)) for i, e in zip(index, ends)]
    window = torch.tensor(np.stack([e["window"] for e in examples]), dtype=torch.float32)
    valid = torch.tensor(np.stack([e["token_valid"] for e in examples]), dtype=torch.float32)
    targets: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}
    for name, width, kind in GROUPS:
        targets[name] = torch.tensor(np.stack([e["targets"][name] for e in examples]),
                                     dtype=torch.long if kind == "multiclass" else torch.float32)
        masks[name] = torch.tensor(np.stack([e["masks"][name] for e in examples]),
                                   dtype=torch.float32)
    columns = list(MIDPOINT_COLUMNS)
    next_mid = targets["process_geometry"][:, columns].numpy().astype(np.float64)
    next_mask = masks["process_geometry"][:, columns].numpy().astype(np.float64)
    term_mid = targets["terminal_geometry"][:, columns].numpy().astype(np.float64)
    term_mask = masks["terminal_geometry"][:, columns].numpy().astype(np.float64)
    # current midpoint from the last real token of the window, using the shared index helper
    rows = torch.arange(window.shape[0])
    last = last_valid_index(valid)
    last_tokens = window[rows, last, :].numpy()
    current_mid = np.stack([last_tokens[:, 7], last_tokens[:, 8],
                            last_tokens[:, 25], last_tokens[:, 26]], axis=1).astype(np.float64)
    current_visible = np.stack([last_tokens[:, 6] > 0.5, last_tokens[:, 24] > 0.5], axis=1)
    # robot translation speed in R_i0, used for the motion subgroups (never the object's speed)
    current_speed = np.stack([np.hypot(last_tokens[:, 3], last_tokens[:, 4]),
                              np.hypot(last_tokens[:, 21], last_tokens[:, 22])], axis=1)
    reference = visible_reference(current_mid, current_visible)
    # a mask of zero means "not supervised": keep the placeholder out of every statistic
    next_mask = np.where(np.isfinite(next_mid), next_mask, 0.0)
    term_mask = np.where(np.isfinite(term_mid), term_mask, 0.0)
    return Batch(window=window, token_valid=valid, current_mid=current_mid,
                 current_visible=current_visible, current_speed=current_speed,
                 next_mid=next_mid, next_mask=next_mask,
                 term_mid=term_mid, term_mask=term_mask, targets=targets, masks=masks,
                 reference=reference, index=np.asarray(index), ends=np.asarray(ends))


def balanced_panel(tokens: Tokens, split: str, count: int, *, seed: int,
                   exclude_roots: Sequence[str] = (),
                   t0_fraction: float = 0.0) -> Dict[str, np.ndarray]:
    """A fixed panel balanced over roots, with the window end drawn uniformly in the sequence.

    Balanced means: first pick ``count`` windows by cycling through the split's roots in a fixed
    shuffled order, so no single long trajectory dominates. The panel is identified by its content
    hash, and the caller is expected to persist it.
    """

    rows = np.where(tokens.split_of == split)[0]
    if exclude_roots:
        forbidden = set(exclude_roots)
        rows = np.asarray([row for row in rows if tokens.roots[row] not in forbidden])
    generator = np.random.default_rng(seed)
    roots = np.asarray([tokens.roots[row] for row in rows])
    unique = np.unique(roots)
    order = generator.permutation(len(unique))
    by_root: Dict[str, List[int]] = {}
    for row, root in zip(rows, roots):
        by_root.setdefault(str(root), []).append(int(row))
    index: List[int] = []
    ends: List[int] = []
    positions = {str(root): 0 for root in unique}
    while len(index) < count:
        progressed = False
        for position in order:
            if len(index) >= count:
                break
            root = str(unique[position])
            pool = by_root[root]
            if positions[root] >= len(pool):
                continue
            row = pool[positions[root]]
            positions[root] += 1
            steps = int(tokens.steps[row])
            # A quota of pre-execution windows keeps the t=0 regime reportable on its own; drawing
            # ends uniformly leaves only a handful of zeros and makes that table meaningless.
            # With no quota the original draw is used *exactly*, so an already-trained checkpoint
            # keeps matching its optimisation panel instead of being silently invalidated.
            if t0_fraction > 0.0:
                if generator.random() < t0_fraction:
                    ends.append(0)
                else:
                    ends.append(int(generator.integers(1, max(2, steps - 1))))
            else:
                ends.append(int(generator.integers(0, max(1, steps - 1))))
            index.append(row)
            progressed = True
        if not progressed:
            break
    index_array = np.asarray(index, dtype=np.int64)
    end_array = np.asarray(ends, dtype=np.int64)
    return {"index": index_array, "end": end_array,
            "root_id": tokens.roots[index_array], "record_id": tokens.records[index_array],
            "split": np.array([split])}


def merge_panels(panels: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate several panels into one, keeping the same key set."""

    keys = set(panels[0])
    merged: Dict[str, np.ndarray] = {}
    for key in keys:
        if key == "split":
            merged[key] = np.array([str(panels[0][key][0])])
            continue
        merged[key] = np.concatenate([np.asarray(panel[key]) for panel in panels])
    return merged


def panel_hash(panel: Dict[str, np.ndarray]) -> str:
    blob = json.dumps({"index": [int(v) for v in panel["index"]],
                       "end": [int(v) for v in panel["end"]],
                       "root_id": [str(v) for v in panel["root_id"]],
                       "record_id": [str(v) for v in panel["record_id"]]},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_panels(tier: str, *, train_windows: int = 2048, val_windows: int = 2048,
                 train_diagnostic_windows: int = 2048, eval_windows: int = 2048,
                 t0_fraction: float = 0.25, seed: int = 20261014,
                 out_dir: Path = OUT) -> dict:
    """The fixed panels: optimised windows, held-out-from-optimisation windows, and validation.

    The validation panel is drawn from the ``val`` split, so its records are disjoint from every
    training record by construction; the training diagnostic panel is drawn from training roots but
    is **never** used by the optimiser, which separates "memorised" from "training distribution".
    """

    tokens = Tokens(TIER_SPLITS[tier])
    # the optimisation panel may span the tier's whole train pool
    train_pool_splits = ["d0"] if tier == "d0" else ["d0", "d1ext"]
    per_split = train_windows // len(train_pool_splits)
    pool = merge_panels([balanced_panel(tokens, name, per_split, seed=seed + offset)
                         for offset, name in enumerate(train_pool_splits)])
    diagnostic = balanced_panel(tokens, train_pool_splits[0], train_diagnostic_windows,
                                seed=seed + 1)
    used_roots = sorted(set(str(r) for r in pool["root_id"]))
    validation = balanced_panel(tokens, "val", val_windows, seed=seed + 2,
                                exclude_roots=used_roots)
    # reporting panels carry an explicit t=0 quota; the selection panel above stays untouched so the
    # checkpoint choice remains the one that was actually made during training
    eval_val = balanced_panel(tokens, "val", eval_windows, seed=seed + 3,
                              exclude_roots=used_roots, t0_fraction=t0_fraction)
    eval_test = balanced_panel(tokens, "test", eval_windows, seed=seed + 4,
                               t0_fraction=t0_fraction)
    payload = {"tier": tier, "seed": seed,
               "panels": {"train": pool, "train_diagnostic": diagnostic, "val": validation,
                          "eval_val": eval_val, "eval_test": eval_test}}
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, panel in payload["panels"].items():
        summary[name] = {"windows": int(len(panel["index"])),
                         "records": int(len(set(str(r) for r in panel["record_id"]))),
                         "roots": int(len(set(str(r) for r in panel["root_id"]))),
                         "hash": panel_hash(panel)}
        np.savez_compressed(out_dir / f"panel_{name}_{tier}.npz", **panel)
    summary["disjoint_check"] = {
        "train_vs_val_records_shared": len(set(str(r) for r in pool["record_id"])
                                           & set(str(r) for r in validation["record_id"])),
        "train_vs_val_roots_shared": len(set(str(r) for r in pool["root_id"])
                                         & set(str(r) for r in validation["root_id"])),
        "diagnostic_vs_val_roots_shared": len(set(str(r) for r in diagnostic["root_id"])
                                              & set(str(r) for r in validation["root_id"])),
        "eval_val_vs_train_records_shared": len(set(str(r) for r in eval_val["record_id"])
                                                & set(str(r) for r in pool["record_id"])),
        "eval_test_vs_train_records_shared": len(set(str(r) for r in eval_test["record_id"])
                                                 & set(str(r) for r in pool["record_id"])),
        "eval_val_t0_windows": int((eval_val["end"] == 0).sum()),
        "eval_test_t0_windows": int((eval_test["end"] == 0).sum()),
    }
    (out_dir / f"panels_{tier}.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary


def load_panel(tier: str, name: str, directory: Path = OUT) -> Dict[str, np.ndarray]:
    return dict(np.load(directory / f"panel_{name}_{tier}.npz", allow_pickle=True))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--train-windows", type=int, default=2048)
    parser.add_argument("--val-windows", type=int, default=2048)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    summary = build_panels(args.tier, train_windows=args.train_windows,
                           val_windows=args.val_windows, out_dir=Path(args.out))
    print(json.dumps(summary, indent=1, sort_keys=True))
    print("PANELS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
