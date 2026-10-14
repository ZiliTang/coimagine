"""Online prediction API for the residual round.

The single rule this module exists to enforce: **the online path and the offline evaluation path
must decode a checkpoint to the same original-unit numbers.** Both call
:func:`targets.decode_prediction` with the same stored :class:`TransformSpec`; neither re-implements
the arithmetic, so a change to the reference rule cannot silently apply to only one of them.

    python experiments/residual_prediction_v1/predict.py --checkpoint <path> --panel val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.residual_prediction_v1 import data as D  # noqa: E402
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    GEOMETRY_COLUMNS,
    TransformSpec,
    decode_prediction,
)

OUT = ROOT / "outputs/residual_prediction_v1"


def spec_from_descriptor(descriptor: Dict[str, object]) -> TransformSpec:
    """Rebuild a spec from what a checkpoint carries (the descriptor is the frozen contract)."""

    columns = descriptor.get("columns")
    indices = tuple(GEOMETRY_COLUMNS.index(name) for name in columns) if columns else (0, 1, 4, 5)
    return TransformSpec(name=str(descriptor["name"]),
                         parameterisation=str(descriptor["parameterisation"]),
                         normalisation=str(descriptor["normalisation"]),
                         reference=str(descriptor["reference"]),
                         columns=indices,
                         mean=np.asarray(descriptor["mean"], dtype=np.float64),
                         std=np.asarray(descriptor["std"], dtype=np.float64),
                         version=str(descriptor.get("version", "unknown")))


class OnlinePredictor:
    """Load one checkpoint and expose prediction plus decoding in original units."""

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        saved = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        self.checkpoint = str(checkpoint)
        self.metadata = {k: v for k, v in saved.items() if k not in ("state_dict", "spec")}
        self.spec_descriptor = saved["spec"]
        self.spec = spec_from_descriptor(saved["spec"])
        self.model_kind = saved.get("model_kind", "midpoint_head")
        self.model = _build(saved)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def predict(self, window: torch.Tensor, token_valid: torch.Tensor) -> np.ndarray:
        """Standardised network output. No reference is added here."""

        with torch.no_grad():
            return self.model(window, token_valid).numpy().astype(np.float64)

    def decode(self, output: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return decode_prediction(output, reference, self.spec)

    def predict_original_units(self, window: torch.Tensor, token_valid: torch.Tensor,
                               reference: np.ndarray) -> np.ndarray:
        return self.decode(self.predict(window, token_valid), reference)


def _build(saved: Dict[str, object]):
    kind = saved.get("model_kind", "midpoint_head")
    if kind == "midpoint_head":
        from experiments.residual_prediction_v1.single_head import MidpointHead
        model = MidpointHead(**saved.get("model_kwargs", {}))
        return model
    from experiments.residual_prediction_v1.multitask import MultiTaskPredictor
    return MultiTaskPredictor(**saved.get("model_kwargs", {}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--panel", default="val")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    predictor = OnlinePredictor(Path(args.checkpoint))
    from experiments.night_sequence_v1.dataset import Tokens
    tokens = Tokens(D.TIER_SPLITS[args.tier])
    panel = D.load_panel(args.tier, args.panel, Path(args.out))
    batch = D.build_batch(tokens, panel["index"][:256], panel["end"][:256])
    values = predictor.predict_original_units(batch.window, batch.token_valid, batch.reference)
    payload = {
        "checkpoint": predictor.checkpoint,
        "spec": predictor.spec.descriptor(),
        "windows": int(values.shape[0]),
        "columns": list(GEOMETRY_COLUMNS),
        "example_r0_midpoint_m": values[0][:2].tolist(),
        "file": None,
        "note": "online decode uses the same decode_prediction as the offline evaluation",
    }
    np.savez_compressed(Path(args.out) / "online_predictions.npz", prediction=values,
                        reference=batch.reference, mask=batch.next_mask, truth=batch.next_mid)
    print(json.dumps(payload, indent=1))
    print("PREDICT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
