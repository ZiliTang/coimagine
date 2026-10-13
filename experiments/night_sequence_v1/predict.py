"""Unified inference entry: one sequence prefix in, the same-semantics outputs out.

    # from a stored sequence: give a record id and how many real periods have been observed
    python experiments/night_sequence_v1/predict.py \
        --record phase_b/train/solo_central/0000#cmd0 --observed 20 \
        --models mlp_current mlp_history transformer_history

Every architecture returns **original physical units** with the same field names, so downstream code
does not branch on the model kind:

    terminal_geometry   r0/r1 midpoint (x, y) in that robot's own R_i0 [m] and the surface normal
                        (sin, cos) in the same frame
    terminal_pose       r0/r1 displacement and wrapped heading change in R_i0 [m, rad]
    terminal_velocity   r0/r1 terminal linear velocity in R_i0 [m/s] and angular velocity [rad/s]
    terminal_validity   r0/r1 patch_valid and contacting, as probabilities
    terminal_remaining  seconds left until the option's exit
    terminal_exit_class index into ``EXIT_CODES``
    process_*           the same quantities one fixed control period ahead, plus ``process_exit``,
                        the probability that the option exits within that period

Two honest limits, printed by the tool itself:

* this is a **real-observation-prefix** predictor. It does not feed its own predictions back, so it
  cannot roll a trajectory forward on its own; the missing pieces are listed in
  ``docs/night_sequence_v1/CONTRACT.md`` and ``presentation_assets/INDEX.md`` (F08).
* it returns numbers for the current prefix only. A caller that wants a closed-loop rollout must
  supply the next real observation, or implement the feedback path itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import (  # noqa: E402
    GROUPS,
    Tokens,
    build_example,
)
from experiments.night_sequence_v1.models import build_model  # noqa: E402
from experiments.night_sequence_v1.sequences import EXIT_CODES  # noqa: E402
from experiments.night_sequence_v1.windows import (  # noqa: E402
    GEOMETRY_COLUMNS,
    POSE_COLUMNS,
    VALIDITY_COLUMNS,
    VELOCITY_COLUMNS,
)

DATA = ROOT / "outputs/night_sequence_v1"
RUNS = DATA / "runs"
TIER_SPLITS = {"d0": ["d0", "val", "test"], "d1": ["d0", "d1ext", "val", "test"],
               "d2": ["d0", "d1ext", "d2ext", "val", "test"]}
KIND = {name: kind for name, _w, kind in GROUPS}

SEMANTICS = {
    "terminal_geometry": {"columns": GEOMETRY_COLUMNS, "units": "m, unit vector",
                          "frame": "each robot's own R_i0"},
    "terminal_pose": {"columns": POSE_COLUMNS, "units": "m, rad", "frame": "R_i0"},
    "terminal_velocity": {"columns": VELOCITY_COLUMNS, "units": "m/s, rad/s", "frame": "R_i0"},
    "terminal_validity": {"columns": VALIDITY_COLUMNS, "units": "probability", "frame": "-"},
    "terminal_remaining": {"columns": ["seconds_remaining"], "units": "s", "frame": "-"},
    "terminal_exit_class": {"columns": ["exit_class_index"], "units": "index into EXIT_CODES",
                            "frame": "-"},
}


class Predictor:
    """Loads one checkpoint and converts its raw head outputs into the shared semantics."""

    def __init__(self, checkpoint: Path, tokens: Tokens, device: str = "cpu"):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model_kind = saved["model_kind"]
        self.model = build_model(self.model_kind, **saved["model_kwargs"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.normalisation = saved["normalisation"]
        self.tokens = tokens
        self.parameters = saved["parameters"]
        self.checkpoint = str(checkpoint.relative_to(ROOT))

    def predict(self, index: int, observed_periods: int) -> Dict[str, object]:
        end = max(0, min(int(observed_periods) - 1, int(self.tokens.steps[index]) - 2))
        example = build_example(self.tokens, index, end)
        window = torch.tensor(example["window"][None], dtype=torch.float32)
        valid = torch.tensor(example["token_valid"][None], dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(window, valid)
        out: Dict[str, object] = {
            "model": self.model_kind, "checkpoint": self.checkpoint,
            "parameters": self.parameters, "window_end_period": end,
            "observed_seconds": round(end * 0.05, 4),
        }
        values: Dict[str, np.ndarray] = {}
        for name, _width, kind in GROUPS:
            raw = outputs[name].numpy()[0]
            if kind == "continuous":
                stats = self.normalisation[name]
                raw = raw * stats["std"] + stats["mean"]
            elif kind == "binary" or name == "process_exit":
                raw = 1.0 / (1.0 + np.exp(-raw))
            elif kind == "multiclass":
                raw = np.asarray([int(np.argmax(raw))])
            values[name] = np.asarray(raw, dtype=float)
        for name, value in values.items():
            if name in SEMANTICS:
                spec = SEMANTICS[name]
                if name == "terminal_exit_class":
                    index_value = int(value[0])
                    out[name] = {"index": index_value,
                                 "name": EXIT_CODES[index_value] if index_value < len(EXIT_CODES)
                                 else "unknown", "units": spec["units"]}
                else:
                    out[name] = {column: float(v) for column, v in
                                 zip(spec["columns"], value.tolist())}
                    out[name]["units"] = spec["units"]
                    out[name]["frame"] = spec["frame"]
            else:
                out[name] = {f"c{i}": float(v) for i, v in enumerate(value.tolist())}
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--record", default="", help="record_id; default is the first test sequence")
    parser.add_argument("--observed", type=int, default=0,
                        help="number of real periods observed (0 means only O0)")
    parser.add_argument("--models", nargs="+",
                        default=["mlp_current", "mlp_history", "transformer_history"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tokens = Tokens(TIER_SPLITS[args.tier])
    if args.record:
        matches = np.where(tokens.records == args.record)[0]
        if len(matches) == 0:
            parser.error(f"record {args.record!r} not found in tier {args.tier}")
        index = int(matches[0])
    else:
        index = int(np.where(tokens.split_of == "test")[0][0])

    results: List[dict] = []
    for kind in args.models:
        checkpoint = RUNS / f"{args.tier}_{kind}_seed{args.seed}.pt"
        if not checkpoint.exists():
            results.append({"model": kind, "error": f"checkpoint missing: {checkpoint.name}"})
            continue
        results.append(Predictor(checkpoint, tokens).predict(index, args.observed))

    payload = {
        "record_id": str(tokens.records[index]), "root_id": str(tokens.roots[index]),
        "split": str(tokens.split_of[index]),
        "sequence_periods": int(tokens.steps[index]),
        "true_exit_seconds": float(tokens.T_exit[index]),
        "note": "real-observation-prefix prediction in original units; no recursive rollout",
        "predictions": results,
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    print("\nNOTE: this entry consumes a real observed prefix. It does not feed predictions back, so "
          "it cannot roll the world forward by itself.", file=sys.stderr)
    print("PREDICT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
