"""One direct-prediction Transformer at a chosen size, under the inherited round-7 protocol.

This deliberately reuses ``residual_prediction_v1.multitask`` instead of re-implementing the
protocol: the capacity curve must differ from the earlier runs only in the encoder's size, so the
loss, the residual geometry transform, the panels and the checkpoint selection stay identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.residual_prediction_v1 import data as RD  # noqa: E402
from experiments.residual_prediction_v1.multitask import train_multitask  # noqa: E402

OUT = ROOT / "outputs/transformer_scale_latent_v1"
PANELS = ROOT / "outputs/residual_prediction_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--kwargs", required=True, help="JSON model kwargs")
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    # without this the process grabs every core, and three of them starve the rest of the matrix
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    kwargs = json.loads(args.kwargs)
    result = train_multitask(tokens=Tokens(RD.TIER_SPLITS["d1"]), tier="d1",
                             kind="transformer_history", seed=args.seed, updates=args.updates,
                             lr=3e-4, parameterisation="residual", normalisation="per_group",
                             out_dir=OUT, model_kwargs=kwargs, tag=args.tag,
                             panel_dir=PANELS)
    with open(OUT / "RUN_LEDGER.jsonl", "a") as handle:
        handle.write(json.dumps({**{k: v for k, v in result.items() if k != "curves"},
                                 "experiment": "capacity", "model_kwargs": kwargs}) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=1))
    print("CAPACITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
