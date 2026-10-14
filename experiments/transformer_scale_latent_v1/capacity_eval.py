"""Score the direct-prediction capacity curve on the same frozen panel and against the same floors.

The capacity models predict one step ahead (plus a direct terminal head), so they enter the step-1
comparison only. They are scored with the residual round's own decoder so the geometry units, the
reference rule and the visibility masks match everything else in this round.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.night_sequence_v1.models import build_model, parameter_count  # noqa: E402
from experiments.residual_prediction_v1 import data as RD  # noqa: E402
from experiments.residual_prediction_v1.evaluate import (  # noqa: E402
    geometry_transform_from_descriptor,
)
from experiments.residual_prediction_v1.targets import decode_geometry  # noqa: E402
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402
from experiments.transformer_scale_latent_v1 import panels as P  # noqa: E402

OUT = ROOT / "outputs/transformer_scale_latent_v1"
MIDPOINTS = [0, 1, 4, 5]


def score(checkpoint: Path, tokens: Tokens, panel: Dict[str, np.ndarray],
          chunk: int = 256) -> Dict[str, object]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(saved["model_kind"], **saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    transform = geometry_transform_from_descriptor(saved["spec"])
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(panel["index"]), chunk):
            stop = min(start + chunk, len(panel["index"]))
            batch = RD.build_batch(tokens, panel["index"][start:stop], panel["end"][start:stop])
            raw = model(batch.window, batch.token_valid)["process_geometry"].numpy().astype(np.float64)
            predictions.append(decode_geometry(raw, batch.reference, transform)[:, MIDPOINTS])
    prediction = np.concatenate(predictions)
    batch = RD.build_batch(tokens, panel["index"], panel["end"])
    truth, mask, reference = batch.next_mid, batch.next_mask, batch.reference
    end = panel["end"]
    out: Dict[str, object] = {"checkpoint": str(checkpoint.relative_to(ROOT)),
                              "parameters": parameter_count(model),
                              "model_kwargs": saved["model_kwargs"],
                              "best_val": saved.get("best_val")}
    for regime, selector in (("t=0", end == 0), ("t>0", end > 0)):
        for robot in (0, 1):
            columns = [0, 1] if robot == 0 else [2, 3]
            keep = selector & (mask[:, columns[0]] > 0) & (mask[:, columns[1]] > 0)
            error = np.hypot(prediction[:, columns[0]] - truth[:, columns[0]],
                             prediction[:, columns[1]] - truth[:, columns[1]])
            hold = np.hypot(reference[:, columns[0]] - truth[:, columns[0]],
                            reference[:, columns[1]] - truth[:, columns[1]])
            out[f"{regime}_r{robot}"] = {
                "p50_cm": float(np.median(error[keep]) * 100),
                "p90_cm": float(np.quantile(error[keep], 0.9) * 100),
                "rmse_cm": float(np.sqrt(np.mean(error[keep] ** 2)) * 100),
                "hold_p50_cm": float(np.median(hold[keep]) * 100),
                "windows": int(keep.sum()),
                "roots": int(len(set(panel["root"][keep].tolist())))}
    # inference cost at batch 1 and 128, single thread
    torch.set_num_threads(1)
    window = torch.zeros(1, 33, 57)
    valid = torch.ones(1, 33)
    with torch.no_grad():
        for _ in range(3):
            model(window, valid)
        started = time.perf_counter()
        for _ in range(20):
            model(window, valid)
        out["batch1_latency_ms"] = (time.perf_counter() - started) / 20 * 1000
    big_window = torch.zeros(128, 33, 57)
    big_valid = torch.ones(128, 33)
    with torch.no_grad():
        for _ in range(2):
            model(big_window, big_valid)
        started = time.perf_counter()
        for _ in range(5):
            model(big_window, big_valid)
        out["batch128_latency_ms"] = (time.perf_counter() - started) / 5 * 1000
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="test_multistep")
    parser.add_argument("--pattern", default="capacity_*.pt")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    tokens = Tokens(D.TIER_SPLITS["d1"])
    panel = P.load(args.panel)
    report: Dict[str, object] = {"panel": args.panel,
                                 "windows": int(len(panel["index"])),
                                 "models": {}}
    for checkpoint in sorted((out_dir / "checkpoints").glob(args.pattern)):
        entry = score(checkpoint, tokens, panel)
        report["models"][checkpoint.stem] = entry
        print(f"{checkpoint.stem:28s} params={entry['parameters']:>9,d} "
              f"t>0 r0 p50={entry['t>0_r0']['p50_cm']:.3f} cm "
              f"(hold {entry['t>0_r0']['hold_p50_cm']:.3f})", flush=True)
    (out_dir / f"capacity_results_{args.panel}.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=float))
    print("CAPACITY_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
