"""Precomputed window tensors.

Both trainers originally rebuilt every window from the token arrays on every update, which cost far
more than the forward and backward passes: ``build_example`` loops over the window in Python. The
windows are deterministic given (record, end), so they are materialised once per panel and indexed
thereafter. The cache changes no value, only *when* it is computed; a hashing check asserts the
cached tensor equals a freshly built one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example  # noqa: E402
from experiments.night_sequence_v1.models import last_valid_index  # noqa: E402
from experiments.residual_prediction_v1.targets import visible_reference  # noqa: E402
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402


@dataclass
class CachedPanel:
    window: torch.Tensor
    token_valid: torch.Tensor
    targets: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor]
    index: np.ndarray
    ends: np.ndarray
    root: np.ndarray
    record: np.ndarray
    reference: np.ndarray          # the residual reference, same rule as the residual round
    current_speed: np.ndarray

    def __len__(self) -> int:
        return int(self.window.shape[0])

    def take(self, rows: np.ndarray) -> "CachedPanel":
        selection = torch.tensor(np.asarray(rows, dtype=np.int64))
        return CachedPanel(window=self.window[selection], token_valid=self.token_valid[selection],
                           targets={k: v[selection] for k, v in self.targets.items()},
                           masks={k: v[selection] for k, v in self.masks.items()},
                           index=self.index[rows], ends=self.ends[rows], root=self.root[rows],
                           record=self.record[rows], reference=self.reference[rows],
                           current_speed=self.current_speed[rows])


def build_cached(tokens: Tokens, index: np.ndarray, ends: np.ndarray,
                 chunk: int = 1024) -> CachedPanel:
    windows, valids = [], []
    targets: Dict[str, List[np.ndarray]] = {}
    masks: Dict[str, List[np.ndarray]] = {}
    for start in range(0, len(index), chunk):
        stop = min(start + chunk, len(index))
        examples = [build_example(tokens, int(i), int(e))
                    for i, e in zip(index[start:stop], ends[start:stop])]
        windows.append(np.stack([e["window"] for e in examples]))
        valids.append(np.stack([e["token_valid"] for e in examples]))
        for name, _width, kind in GROUPS:
            targets.setdefault(name, []).append(np.stack([e["targets"][name] for e in examples]))
            masks.setdefault(name, []).append(np.stack([e["masks"][name] for e in examples]))
    # the residual reference is derived exactly as in the residual round: the newest real token's
    # midpoint when that robot's patch is visible, the fixed origin otherwise
    rows_tensor = torch.tensor(np.concatenate(valids), dtype=torch.float32)
    last = last_valid_index(rows_tensor)
    flat = torch.tensor(np.concatenate(windows), dtype=torch.float32)
    arange = torch.arange(flat.shape[0])
    newest = flat[arange, last, :].numpy()
    current_mid = np.stack([newest[:, 7], newest[:, 8], newest[:, 25], newest[:, 26]],
                           axis=1).astype(np.float64)
    current_visible = np.stack([newest[:, 6] > 0.5, newest[:, 24] > 0.5], axis=1)
    reference = visible_reference(current_mid, current_visible)
    current_speed = np.stack([np.hypot(newest[:, 3], newest[:, 4]),
                              np.hypot(newest[:, 21], newest[:, 22])], axis=1)
    return CachedPanel(
        reference=reference, current_speed=current_speed,
        window=torch.tensor(np.concatenate(windows), dtype=torch.float32),
        token_valid=torch.tensor(np.concatenate(valids), dtype=torch.float32),
        targets={name: torch.tensor(np.concatenate(values),
                                    dtype=torch.long if dict(
                                        (n, k) for n, _w, k in GROUPS)[name] == "multiclass"
                                    else torch.float32)
                 for name, values in targets.items()},
        masks={name: torch.tensor(np.concatenate(values), dtype=torch.float32)
               for name, values in masks.items()},
        index=np.asarray(index), ends=np.asarray(ends),
        root=np.asarray([str(tokens.roots[i]) for i in index]),
        record=np.asarray([str(tokens.records[i]) for i in index]))


@dataclass
class CachedMultiStep:
    window: torch.Tensor
    token_valid: torch.Tensor
    command: torch.Tensor
    elapsed: torch.Tensor
    anchor: np.ndarray
    future: Dict[str, np.ndarray]
    step_mask: np.ndarray
    geometry_mask: np.ndarray
    horizon: int
    index: np.ndarray
    ends: np.ndarray
    root: np.ndarray

    def __len__(self) -> int:
        return int(self.window.shape[0])


def build_multistep_cache(tokens: Tokens, index: np.ndarray, ends: np.ndarray,
                          horizon: int) -> CachedMultiStep:
    batch = D.build_multistep(tokens, index, ends, horizon)
    return CachedMultiStep(window=batch.window, token_valid=batch.token_valid,
                           command=torch.tensor(batch.command, dtype=torch.float32),
                           elapsed=torch.tensor(batch.elapsed, dtype=torch.float32),
                           anchor=batch.anchor, future=batch.future, step_mask=batch.step_mask,
                           geometry_mask=batch.geometry_mask, horizon=horizon,
                           index=batch.index, ends=batch.ends, root=batch.root)
