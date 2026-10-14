"""Build the review ZIP for the multi-step round.

The package carries, for the frozen test panel: the raw window tensors, the anchor, the future
labels and their masks, the panel indices and root/scene identifiers, the checkpoints, the per-sample
predictions, the result tables, the figures and the code. The verification script in the archive
recomputes the first-step errors and the hold baseline **from the packaged tensors** and compares
them with the stored reference values; it states explicitly that it reproduces the packaged sample,
not the full panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.transformer_scale_latent_v1 import cache as C  # noqa: E402
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402
from experiments.transformer_scale_latent_v1 import panels as P  # noqa: E402
from experiments.transformer_scale_latent_v1.evaluate import (  # noqa: E402
    load_model,
    predict_steps,
    step_error,
)

OUT = ROOT / "outputs/transformer_scale_latent_v1"
DOCS = ROOT / "docs/transformer_scale_latent_v1"
PACKAGE = ROOT / "transformer_scale_latent_v1_review"
FROZEN = "latent_autoregressive_h8_lr0.0003_seed0"
DIRECT = "direct_multistep_h8_lr0.0003_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--out", default=str(PACKAGE))
    args = parser.parse_args()
    package = Path(args.out)
    if package.exists():
        shutil.rmtree(package)
    for name in ("samples", "checkpoints", "reference", "docs", "code", "figures", "results"):
        (package / name).mkdir(parents=True)
    tokens = Tokens(D.TIER_SPLITS["d1"])
    panel = P.load("test_multistep")
    generator = np.random.default_rng(11)
    picked = np.sort(generator.choice(len(panel["index"]), size=min(args.samples,
                                                                    len(panel["index"])),
                                      replace=False))
    index, ends = panel["index"][picked], panel["end"][picked]
    batch = C.build_multistep_cache(tokens, index, ends, args.horizon)
    np.savez_compressed(package / "samples" / "windows_test_multistep.npz",
                        window=batch.window.numpy(), token_valid=batch.token_valid.numpy(),
                        command=batch.command.numpy(), elapsed=batch.elapsed.numpy(),
                        anchor=batch.anchor, step_mask=batch.step_mask,
                        geometry_mask=batch.geometry_mask,
                        truth_geometry=batch.future["geometry"],
                        truth_pose=batch.future["pose"],
                        truth_velocity=batch.future["velocity"],
                        truth_validity=batch.future["validity"],
                        end=panel["end"][picked], root=panel["root"][picked],
                        scene=panel["scene"][picked], index=index,
                        panel_root=panel["root"], panel_end=panel["end"],
                        panel_index=panel["index"])
    expected: Dict[str, object] = {"samples": int(len(picked)), "horizon": args.horizon,
                                   "panel": "test_multistep",
                                   "panel_hash": json.loads(
                                       (OUT / "panel_test_multistep.json").read_text())["hash"],
                                   "models": {}}
    stored = json.loads((OUT / "multi_step_results_test_multistep.json").read_text())
    for name in (DIRECT, FROZEN):
        source = OUT / "checkpoints" / f"{name}.pt"
        if not source.exists():
            continue
        shutil.copy2(source, package / "checkpoints" / source.name)
        saved, model = load_model(source)
        predicted = predict_steps(saved, model, tokens, index, ends, args.horizon)
        entry = {"kind": saved["kind"], "stats": saved["stats"],
                 "parameters": saved["parameters"]}
        for step in (1, 4, 8, 16):
            error = step_error(predicted["geometry"], batch.future["geometry"],
                               batch.geometry_mask, batch.step_mask, step - 1, 0)
            keep = np.isfinite(error)
            entry[f"step{step}_r0_p50_cm"] = float(np.median(error[keep]) * 100)
            entry[f"step{step}_r0_windows"] = int(keep.sum())
        anchor_error = np.where(
            (batch.geometry_mask[:, 0, 0] > 0) & (batch.step_mask[:, 0] > 0),
            np.hypot(batch.anchor[:, 0] - batch.future["geometry"][:, 0, 0],
                     batch.anchor[:, 1] - batch.future["geometry"][:, 0, 1]), np.nan)
        entry["hold_step1_r0_p50_cm"] = float(np.nanmedian(anchor_error) * 100)
        entry["full_panel"] = stored["models"].get(name, {}).get("step_p50_cm")
        expected["models"][name] = entry
    (package / "reference" / "expected_metrics.json").write_text(
        json.dumps(expected, indent=1, sort_keys=True, default=float))
    for name in ("multi_step_results_test_multistep.json",
                 "pair_comparison_test_multistep.json", "panel_test_multistep.json",
                 "audit.json", "RUN_LEDGER.jsonl"):
        source = OUT / name
        if source.exists():
            shutil.copy2(source, package / "results" / name)
    for name in sorted(DOCS.glob("*.png")):
        shutil.copy2(name, package / "figures" / name.name)
    for name in ("RESULTS.md", "CONTRACT.md", "AUDIT.md", "REVIEW_FIXES.md"):
        source = DOCS / name
        if source.exists():
            shutil.copy2(source, package / "docs" / name)
    for name in ("data.py", "models.py", "train.py", "evaluate.py", "cache.py", "panels.py",
                 "configs.py", "audit.py"):
        shutil.copy2(ROOT / "experiments/transformer_scale_latent_v1" / name,
                     package / "code" / name)
    shutil.copy2(ROOT / "experiments/transformer_scale_latent_v1/verify_package.py",
                 package / "verify_package.py")
    manifest = [{"path": str(path.relative_to(package)), "bytes": path.stat().st_size,
                 "sha256": sha256(path)}
                for path in sorted(package.rglob("*")) if path.is_file()]
    (package / "MANIFEST.json").write_text(json.dumps(
        {"files": manifest, "count": len(manifest),
         "note": "sizes and hashes of the packaged files only; this validates the archive, not any "
                 "experimental conclusion"}, indent=1, sort_keys=True))
    archive = ROOT / "transformer_scale_latent_v1_review.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(package))
    names = zipfile.ZipFile(archive).namelist()
    print(json.dumps({"archive": archive.name, "bytes": archive.stat().st_size,
                      "entries": len(names), "duplicates": len(names) - len(set(names)),
                      "sha256": sha256(archive)}, indent=1))
    print("PACKAGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
