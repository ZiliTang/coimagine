"""Verify the review package **without the repository**, by recomputing the numbers.

Run from a clean directory:

    python verify_package.py

It loads the packaged window tensors, reconstructs each target transform from the packaged
descriptor (the arithmetic is re-implemented here on purpose, so a mistake shared with the repository
would show up as a mismatch), runs the packaged checkpoint, decodes, and recomputes the hold baseline
and the model p50/p90 with the real visibility masks. Only then does it compare against
``reference/expected_metrics.json``.

A success line means "these packaged numbers reproduce on this subset". It is **not** a rerun of the
experiments and it does not validate the full test panel.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

MIDPOINTS = [0, 1, 4, 5]


def decode(output: np.ndarray, reference: np.ndarray, spec: dict) -> np.ndarray:
    """Re-implementation of the packaged decode rule for the four midpoint columns."""

    mean = np.asarray(spec["mean"], dtype=np.float64)
    std = np.asarray(spec["std"], dtype=np.float64)
    columns = [["r0_mid_x", "r0_mid_y", "r0_normal_sin", "r0_normal_cos",
                "r1_mid_x", "r1_mid_y", "r1_normal_sin", "r1_normal_cos"].index(name)
               for name in spec["columns"]]
    if spec["normalisation"] == "per_group":
        delta = float(mean[0]) + float(std[0]) * output[:, columns]
    else:
        delta = mean + std * output[:, columns]
    if spec["parameterisation"] == "direct":
        return delta
    return reference + delta


def metrics(prediction, samples, robot):
    columns = [0, 1] if robot == 0 else [2, 3]
    mask = samples["next_mask"]
    keep = (mask[:, columns[0]] > 0) & (mask[:, columns[1]] > 0) & (samples["end"] > 0)
    truth = samples["next_truth"]
    error = np.hypot(prediction[:, columns[0]] - truth[:, columns[0]],
                     prediction[:, columns[1]] - truth[:, columns[1]])
    return {"p50_cm": float(np.median(error[keep]) * 100),
            "p90_cm": float(np.quantile(error[keep], 0.9) * 100),
            "windows": int(keep.sum())}


def main() -> int:
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "reference" / "expected_metrics.json").read_text())
    samples = dict(np.load(here / "samples" / "windows_eval_test.npz", allow_pickle=True))
    checkpoint = torch.load(here / "checkpoints" / expected["frozen_checkpoint"],
                            map_location="cpu", weights_only=False)
    sys.path.insert(0, str(here))
    from torch import nn

    state = checkpoint["state_dict"]
    window = torch.tensor(samples["window"], dtype=torch.float32)
    valid = torch.tensor(samples["token_valid"], dtype=torch.float32)
    flat = torch.cat([window.reshape(window.shape[0], -1), valid], dim=-1)
    # rebuild the trunk and the geometry head from whatever Linear layers the checkpoint contains,
    # so the verifier does not depend on a hard-coded layer index from the training code
    def sequential(prefix: str) -> nn.Sequential:
        indices = sorted({int(key[len(prefix):].split(".")[0]) for key in state
                          if key.startswith(prefix) and key.endswith(".weight")})
        modules = []
        for position, index in enumerate(indices):
            weight = state[f"{prefix}{index}.weight"]
            bias = state[f"{prefix}{index}.bias"]
            modules.append(nn.Linear(int(weight.shape[1]), int(weight.shape[0])))
            if position + 1 < len(indices):
                modules.append(nn.ReLU())
        network = nn.Sequential(*modules)
        for position, index in enumerate(indices):
            layer = network[2 * position]
            layer.weight.requires_grad_(False)
            layer.bias.requires_grad_(False)
            layer.weight.copy_(state[f"{prefix}{index}.weight"])
            layer.bias.copy_(state[f"{prefix}{index}.bias"])
        return network

    trunk = sequential("trunk.")
    # the flattened-history trunk ends with a parameterless ReLU, so it is absent from the state
    # dict and has to be re-attached explicitly; the geometry head has no trailing activation
    trunk = nn.Sequential(trunk, nn.ReLU())
    head = sequential("heads.process_geometry.")
    with torch.no_grad():
        features = trunk(flat)
        raw = head(features).numpy().astype(np.float64)
    prediction = decode(raw, samples["reference"], expected["transform"]["midpoint"])
    hold = samples["reference"]
    result = {"model": {f"r{r}": metrics(prediction, samples, r) for r in (0, 1)},
              "hold": {f"r{r}": metrics(hold, samples, r) for r in (0, 1)}}
    ok = True
    for name in ("model", "hold"):
        for robot in ("r0", "r1"):
            for key in ("p50_cm", "p90_cm"):
                want = expected[name][f"{robot}_t_gt_0"][key]
                got = result[name][robot][key]
                if abs(want - got) > 1e-6:
                    ok = False
                    print(f"MISMATCH {name}.{robot}.{key}: packaged {want} vs recomputed {got}")
    print(json.dumps({"samples": int(len(window)), "recomputed": result,
                      "full_panel": expected.get("full_panel_values_from_main_results", {}),
                      "subset_reproduction": ok}, indent=1, default=float))
    print("VERIFY_PACKAGE_OK" if ok else "VERIFY_PACKAGE_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
