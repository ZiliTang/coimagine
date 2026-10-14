"""The frozen multi-step test panel.

Reuses the identity of the earlier frozen panels (same split, same balanced-over-roots rule) and adds
what a multi-step evaluation needs: the raw root / record / time index of every window, the scene
class taken from the root identifier, and a content hash. The panel is written once and never
re-drawn, so every model in this round is scored on exactly the same windows.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402

OUT = ROOT / "outputs/transformer_scale_latent_v1"
#: the four scene classes named in the task book, plus the two solo classes for completeness
SCENE_CLASSES = ("constrained", "dual_rotate", "dual_same_side", "incompatible",
                 "solo_central", "solo_offset")


def scene_of(root: str) -> str:
    """The generator's class, read from the root identifier rather than inferred from the name."""

    parts = str(root).split("/")
    for part in parts:
        if part in SCENE_CLASSES:
            return part
    return "unknown"


def build(tier: str = "d1", split: str = "test", count: int = 2048, *, seed: int = 20261014,
          t0_fraction: float = 0.25, min_future: int = 1, name: str = "test_multistep") -> dict:
    tokens = Tokens(D.TIER_SPLITS[tier])
    index, ends = D.sample_windows(tokens, [split], count, seed=seed, horizon=16,
                                   min_future=min_future, t0_fraction=t0_fraction)
    roots = np.asarray([str(tokens.roots[i]) for i in index])
    records = np.asarray([str(tokens.records[i]) for i in index])
    scenes = np.asarray([scene_of(r) for r in roots])
    steps = np.asarray([int(tokens.steps[i]) for i in index])
    future_available = steps - 1 - ends
    payload = {"index": index, "end": ends, "root": roots, "record": records,
               "scene": scenes, "steps": steps, "future_available": future_available}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"panel_{name}.npz"
    np.savez_compressed(path, **payload)
    blob = json.dumps({"index": [int(v) for v in index], "end": [int(v) for v in ends],
                       "root": roots.tolist()}, sort_keys=True, separators=(",", ":")).encode()
    summary = {"name": name, "tier": tier, "split": split, "windows": int(len(index)),
               "roots": int(len(set(roots.tolist()))),
               "records": int(len(set(records.tolist()))),
               "hash": hashlib.sha256(blob).hexdigest(),
               "t0_windows": int((ends == 0).sum()),
               "t_gt_0_windows": int((ends > 0).sum()),
               "future_ge_8": int((future_available >= 8).sum()),
               "future_ge_16": int((future_available >= 16).sum()),
               "scene_counts": {c: int((scenes == c).sum()) for c in SCENE_CLASSES},
               "path": str(path.relative_to(ROOT))}
    (OUT / f"panel_{name}.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary


def load(name: str = "test_multistep") -> Dict[str, np.ndarray]:
    return dict(np.load(OUT / f"panel_{name}.npz", allow_pickle=True))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summary = build(args.tier, "test", args.count)
    print(json.dumps(summary, indent=1))
    summary = build(args.tier, "val", args.count, name="val_multistep")
    print(json.dumps(summary, indent=1))
    print("PANELS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
