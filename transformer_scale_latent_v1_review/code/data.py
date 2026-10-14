"""Multi-step windows: a real prefix, a fixed residual anchor, and real future frames.

Supervision comes only from frames that actually exist in the same sequence: for a prefix ending at
``end`` the step-k label is the token at ``end + k`` and it is valid only while ``end + k < steps``.
Nothing after the option's exit is used, no frame is fabricated, and a step with no real frame is
masked out of the loss instead of being filled with a padded or repeated terminal state.

The residual anchor is the **prediction start** midpoint (the visible midpoint at ``end``; the fixed
origin of R_i0 when that patch is invisible), frozen once and shared by the direct multi-step model
and the latent autoregressive model. Step k never uses the midpoint at ``end + k`` as a reference.
"""

from __future__ import annotations

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

from experiments.night_sequence_v1.dataset import Tokens, window_indices  # noqa: E402
from experiments.night_sequence_v1.models import last_valid_index  # noqa: E402
from experiments.night_sequence_v1.windows import FEATURES, WINDOW  # noqa: E402

OUT = ROOT / "outputs/transformer_scale_latent_v1"
TIER_SPLITS = {"d0": ["d0", "val", "test"], "d1": ["d0", "d1ext", "val", "test"]}
DT = 0.05

#: the observation vector a model predicts at each future step, exactly the round-6 group layout
OBSERVATION_SLICES = {
    "geometry": [("r0", 7, 11), ("r1", 25, 29)],
    "pose": [("r0", 0, 3), ("r1", 18, 21)],
    "velocity": [("r0", 3, 6), ("r1", 21, 24)],
    "validity": [("r0", 6, 7), ("r0d", 14, 15), ("r1", 24, 25), ("r1d", 32, 33)],
}
OBSERVATION_WIDTH = {"geometry": 8, "pose": 6, "velocity": 6, "validity": 4}
#: which token columns carry each robot's patch visibility at a future step
VISIBILITY_COLUMNS = {"r0": 6, "r1": 24}


def observation_vector(token: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, pieces in OBSERVATION_SLICES.items():
        out[name] = np.concatenate([np.asarray(token)[a:b] for _robot, a, b in pieces])
    return out


@dataclass
class MultiStepBatch:
    window: torch.Tensor            # (n, WINDOW, FEATURES)
    token_valid: torch.Tensor       # (n, WINDOW)
    command: np.ndarray             # (n, 6) alpha0, alpha1, speed0, speed1, active0, active1
    elapsed: np.ndarray             # (n,) clock at the prefix end, seconds
    anchor: np.ndarray              # (n, 4) residual anchor midpoint, R_i0
    future: Dict[str, np.ndarray]   # name -> (n, H, width)
    step_mask: np.ndarray           # (n, H) true frames only
    geometry_mask: np.ndarray       # (n, H, 8) per-robot visibility at that future step
    horizon: int
    index: np.ndarray
    ends: np.ndarray
    root: np.ndarray
    record: np.ndarray
    stage: np.ndarray               # controller stage at the prefix end, for grouping only

    def __len__(self) -> int:
        return int(self.window.shape[0])


def build_multistep(tokens: Tokens, index: np.ndarray, ends: np.ndarray,
                    horizon: int) -> MultiStepBatch:
    index = np.asarray(index)
    ends = np.asarray(ends)
    window = np.zeros((len(index), WINDOW, FEATURES), dtype=np.float32)
    valid = np.zeros((len(index), WINDOW), dtype=np.float32)
    command = np.zeros((len(index), 6), dtype=np.float32)
    elapsed = np.zeros(len(index), dtype=np.float32)
    anchor = np.zeros((len(index), 4), dtype=np.float64)
    stage = np.zeros(len(index), dtype=np.float32)
    future = {name: np.zeros((len(index), horizon, width), dtype=np.float64)
              for name, width in OBSERVATION_WIDTH.items()}
    step_mask = np.zeros((len(index), horizon), dtype=np.float32)
    geometry_mask = np.zeros((len(index), horizon, 8), dtype=np.float32)
    for row, (i, end) in enumerate(zip(index, ends)):
        split = tokens.split_of[i]
        local = int(i) - tokens.offsets[split]
        block = tokens.token_arrays[split][local]
        steps = int(tokens.steps[i])
        end = int(min(max(end, 0), max(0, steps - 2)))
        indices, _dropped = window_indices(steps, end)
        offset = WINDOW - len(indices)
        for position, source in enumerate(indices):
            window[row, offset + position] = block[source]
            valid[row, offset + position] = 1.0
        current = block[end]
        command[row] = current[51:57]
        elapsed[row] = float(end) * DT
        stage[row] = float(current[49]) * 8.0
        visible = np.array([current[VISIBILITY_COLUMNS["r0"]] > 0.5,
                            current[VISIBILITY_COLUMNS["r1"]] > 0.5])
        midpoint = np.array([current[7], current[8], current[25], current[26]])
        anchor[row] = np.where(np.repeat(visible, 2), midpoint, 0.0)
        for step in range(horizon):
            target = end + 1 + step
            if target >= steps:
                continue                     # no real frame: masked out, never fabricated
            token = block[target]
            step_mask[row, step] = 1.0
            for name, pieces in OBSERVATION_SLICES.items():
                future[name][row, step] = np.concatenate([token[a:b] for _r, a, b in pieces])
            # the slice bounds must be built explicitly: `0:4 if cond else 4:8` parses as a slice
            # *step* of 8, which silently writes only the first column and leaves the rest zero
            for first, last, column in ((0, 4, 6), (4, 8, 24)):
                geometry_mask[row, step, first:last] = 1.0 if token[column] > 0.5 else 0.0
    return MultiStepBatch(window=torch.tensor(window), token_valid=torch.tensor(valid),
                          command=command, elapsed=elapsed, anchor=anchor, future=future,
                          step_mask=step_mask, geometry_mask=geometry_mask, horizon=horizon,
                          index=index, ends=ends,
                          root=np.asarray([str(tokens.roots[i]) for i in index]),
                          record=np.asarray([str(tokens.records[i]) for i in index]),
                          stage=stage)


def sample_windows(tokens: Tokens, splits: Sequence[str], count: int, *, seed: int,
                   horizon: int, min_future: int = 1,
                   t0_fraction: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Prefix windows with at least ``min_future`` real frames left, balanced over roots.

    A pre-execution quota is kept deliberately: drawing the end uniformly leaves almost no t=0
    windows, and the two regimes must be reported separately rather than merged into one number.
    """

    rows = np.where(np.isin(tokens.split_of, list(splits)))[0]
    steps = tokens.steps[rows]
    usable = rows[steps - 1 >= min_future]
    generator = np.random.default_rng(seed)
    by_root: Dict[str, List[int]] = {}
    for row in usable:
        by_root.setdefault(str(tokens.roots[row]), []).append(int(row))
    roots = sorted(by_root)
    order = generator.permutation(len(roots))
    index: List[int] = []
    ends: List[int] = []
    positions = {root: 0 for root in roots}
    while len(index) < count:
        progressed = False
        for position in order:
            if len(index) >= count:
                break
            root = roots[position]
            pool = by_root[root]
            if positions[root] >= len(pool):
                continue
            row = pool[positions[root]]
            positions[root] += 1
            steps_here = int(tokens.steps[row])
            # keep enough real frames for the horizon whenever the sequence allows it
            high = max(1, steps_here - 1 - min(min_future, steps_here - 2))
            if t0_fraction > 0.0 and generator.random() < t0_fraction:
                ends.append(0)
            else:
                ends.append(int(generator.integers(0, max(1, high))))
            index.append(row)
            progressed = True
        if not progressed:
            break
    return np.asarray(index, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    tokens = Tokens(TIER_SPLITS[args.tier])
    index, ends = sample_windows(tokens, ["d0"], args.count, seed=1, horizon=args.horizon)
    batch = build_multistep(tokens, index, ends, args.horizon)
    report = {"tier": args.tier, "horizon": args.horizon, "windows": len(index),
              "t0_windows": int((ends == 0).sum()),
              "t_gt_0_windows": int((ends > 0).sum()),
              "real_frames": int(batch.step_mask.sum()),
              "max_possible_frames": int(len(index) * args.horizon),
              "step_fill_ratio": [float(batch.step_mask[:, k].mean())
                                  for k in range(args.horizon)],
              "geometry_visible_ratio": [float(batch.geometry_mask[:, k].mean())
                                         for k in range(args.horizon)]}
    print(json.dumps(report, indent=1))
    print("MULTISTEP_DATA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
