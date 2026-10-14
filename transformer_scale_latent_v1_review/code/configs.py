"""Model-size selection: match the two multi-step families and build the capacity curve.

The task book asks for models near 0.9M, 1.6M and 4.0M **total trainable** parameters, and for the
direct-multi-step / latent pair to differ by at most about 5% in total parameters. Rather than
quoting a hand-picked width, every candidate is counted, the closest is kept, and the encoder /
transition / decoder / head split is reported next to the total.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.transformer_scale_latent_v1.models import (  # noqa: E402
    DirectMultiStep,
    LatentAutoregressive,
    parameter_breakdown,
)

#: (d_model, heads, layers, ffn, decoder_hidden)
SIZE_GRID = [
    (128, 4, 4, 512, 256),
    (144, 4, 4, 576, 288),
    (160, 4, 4, 640, 320),
    (176, 4, 4, 704, 352),
    (176, 4, 5, 704, 352),
    (192, 4, 4, 768, 384),
    (192, 4, 5, 768, 384),
    (208, 4, 5, 832, 416),
    (224, 4, 5, 896, 448),
    (240, 4, 6, 960, 480),
    (256, 4, 5, 1024, 512),
    (256, 4, 6, 1024, 512),
    (288, 4, 6, 1152, 576),
    (320, 4, 6, 1280, 640),
    (352, 4, 7, 1408, 704),
    (384, 4, 8, 1536, 768),
]
TRANSITION_HIDDEN = {128: 256, 144: 288, 160: 320, 176: 352, 192: 384, 208: 416, 224: 448,
                     240: 480, 256: 512, 288: 576, 320: 640, 352: 704, 384: 768}


def count(size: Tuple[int, int, int, int, int], family: str,
          decoder_hidden: int = None) -> Dict[str, int]:
    d_model, heads, layers, ffn, default_decoder = size
    decoder = decoder_hidden or default_decoder
    if family == "direct_multistep":
        model = DirectMultiStep(d_model=d_model, heads=heads, layers=layers, ffn=ffn,
                                decoder_hidden=decoder)
    else:
        model = LatentAutoregressive(d_model=d_model, heads=heads, layers=layers, ffn=ffn,
                                     transition_hidden=TRANSITION_HIDDEN[d_model],
                                     decoder_hidden=decoder)
    return parameter_breakdown(model)


def pick_pair(target: int = 1_600_000, tolerance: float = 0.05) -> Dict[str, object]:
    """The closest parameter-matched pair to ``target``, with the direct model counted first."""

    best = None
    for size in SIZE_GRID:
        direct = count(size, "direct_multistep")
        # the latent model carries an extra transition; shrink its decoder to close the gap
        d_model, heads, layers, ffn, decoder = size
        for trim in range(0, decoder, 32):
            latent = count(size, "latent_autoregressive", decoder_hidden=decoder - trim)
            gap = abs(latent["total"] - direct["total"]) / max(direct["total"], latent["total"])
            off = abs(direct["total"] - target) / target
            score = off + gap * 4.0
            if gap <= tolerance and (best is None or score < best["score"]):
                best = {"size": list(size), "decoder_hidden": decoder,
                        "latent_decoder_hidden": decoder - trim,
                        "direct": direct, "latent": latent, "gap": gap, "score": score,
                        "offset_from_target": off}
    if best is None:
        raise RuntimeError("no parameter-matched pair found in the grid")
    return best


def capacity_curve() -> List[Dict[str, object]]:
    """Closest configurations to each anchor size, for the direct-prediction capacity curve."""

    out = []
    for target in (900_000, 1_600_000, 4_000_000):
        best = None
        for size in SIZE_GRID:
            total = count(size, "direct_multistep")["total"]
            off = abs(total - target) / target
            if best is None or off < best["offset"]:
                best = {"size": list(size), "total": total, "offset": off}
        out.append({"target": target, **best})
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1_600_000)
    parser.add_argument("--out", default=str(ROOT / "outputs/transformer_scale_latent_v1/configs"))
    args = parser.parse_args()
    pair = pick_pair(args.target)
    curve = capacity_curve()
    payload = {"pair": pair, "capacity_curve": curve,
               "note": "totals are counted from the constructed modules, encoder/transition/"
                       "decoder/heads reported separately"}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "model_sizes.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    print("CONFIGS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
