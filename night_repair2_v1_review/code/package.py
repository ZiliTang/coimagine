"""Build the repair-round review archive."""

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

BASE = ROOT / "outputs/night_repair2_v1"
DOCS = ROOT / "docs/night_repair2_v1"
PACKAGE = ROOT / "night_repair2_v1_review"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PACKAGE))
    parser.add_argument("--samples", type=int, default=192)
    args = parser.parse_args()
    package = Path(args.out)
    if package.exists():
        shutil.rmtree(package)
    for name in ("docs", "code", "checkpoints", "samples", "results", "data"):
        (package / name).mkdir(parents=True)
    for name in sorted(DOCS.glob("*.md")):
        shutil.copy2(name, package / "docs" / name.name)
    shutil.copy2(ROOT / "experiments/night_repair2_v1/verify_sample.py",
                 package / "verify_package.py")
    shutil.copy2(ROOT / "experiments/night_repair2_v1/verify.py", package / "code" / "verify.py")
    for name in ("rl.py", "ppo.py", "collect.py", "verify.py", "package.py", "datacard.py",
                 "compare_policies.py", "world_train_eval.py"):
        shutil.copy2(ROOT / "experiments/night_repair2_v1" / name, package / "code" / name)
    # the world-model trainer change is part of the delivered evidence, so ship the file too
    shutil.copy2(ROOT / "experiments/transformer_scale_latent_v1/train.py",
                 package / "code" / "world_train.py")
    for name in ("option_env.py", "policy.py"):
        shutil.copy2(ROOT / "experiments/night_repair_optionrl_v1" / name,
                     package / "code" / f"legacy_{name}")
    # the world-model checkpoints live in the previous round's directory; this round did not
    # retrain them, it only added the fixed train-panel evaluation
    world_models = ROOT / "outputs/night_repair_optionrl_v1/world_models/checkpoints"
    for name in ("repair_direct_multistep_seed0.pt", "repair_latent_autoregressive_seed0.pt"):
        source = world_models / name
        if source.exists():
            shutil.copy2(source, package / "checkpoints" / name)
    for name in ("policy_best_seed0.pt", "policy_resumable_seed0.pt"):
        source = BASE / "option_rl" / name
        if source.exists():
            shutil.copy2(source, package / "checkpoints" / name)
    for name in ("verification.json",):
        source = BASE / "repair" / name
        if source.exists():
            shutil.copy2(source, package / "results" / name)
    for name in ("train_result_seed0.json", "checkpoint_comparison.json",
                 "policy_comparison.json"):
        source = BASE / "option_rl" / name
        if source.exists():
            shutil.copy2(source, package / "results" / name)
    for name in ("world_train_eval.json",):
        source = ROOT / "outputs/night_repair2_v1/world_models" / name
        if source.exists():
            shutil.copy2(source, package / "results" / name)
    for name in ("TRIAL_SUMMARY.json", "DATA_CARD.md", "DATACARD.json"):
        source = BASE / "collection" / name
        if source.exists():
            shutil.copy2(source, package / "data" / name)
    for name in ():

        source = BASE / "option_rl" / name
        if source.exists():
            shutil.copy2(source, package / "results" / name)
    for name in ("TRIAL_SUMMARY.json", "DATA_CARD.md", "DATACARD.json"):
        source = BASE / "collection" / name
        if source.exists():
            shutil.copy2(source, package / "data" / name)
    shards = []
    for directory in sorted((BASE / "collection").iterdir()):
        if directory.is_dir():
            shards.extend(sorted(directory.glob("episode_*.npz"))[:8])
    for shard in shards[:24]:
        shutil.copy2(shard, package / "data" / shard.name)
    # the packaged sample the verifier recomputes
    from experiments.night_sequence_v1.dataset import Tokens
    from experiments.transformer_scale_latent_v1 import cache as C
    from experiments.transformer_scale_latent_v1 import data as D
    from experiments.transformer_scale_latent_v1 import panels as P
    from experiments.transformer_scale_latent_v1.evaluate import load_model, predict_steps, step_error
    tokens = Tokens(D.TIER_SPLITS["d1"])
    panel = P.load("test_multistep")
    generator = np.random.default_rng(5)
    picked = np.sort(generator.choice(len(panel["index"]), size=args.samples, replace=False))
    index, ends = panel["index"][picked], panel["end"][picked]
    batch = C.build_multistep_cache(tokens, index, ends, 16)
    expected: Dict[str, object] = {"samples": int(len(picked)), "models": {}}
    for name in ("repair_direct_multistep_seed0", "repair_latent_autoregressive_seed0"):
        checkpoint = world_models / f"{name}.pt"
        if not checkpoint.exists():
            continue
        saved, model = load_model(checkpoint)
        predicted = predict_steps(saved, model, tokens, index, ends, 16)
        entry = {"kind": saved["kind"], "stats": saved["stats"]}
        for step in (1, 4, 8, 16):
            error = step_error(predicted["geometry"], batch.future["geometry"],
                               batch.geometry_mask, batch.step_mask, step - 1, 0)
            keep = np.isfinite(error)
            entry[f"step{step}_r0_p50_cm"] = float(np.median(error[keep]) * 100)
        anchor_error = np.where((batch.geometry_mask[:, 0, 0] > 0) & (batch.step_mask[:, 0] > 0),
                                np.hypot(batch.anchor[:, 0] - batch.future["geometry"][:, 0, 0],
                                         batch.anchor[:, 1] - batch.future["geometry"][:, 0, 1]),
                                np.nan)
        entry["hold_step1_r0_p50_cm"] = float(np.nanmedian(anchor_error) * 100)
        expected["models"][name] = entry
    np.savez_compressed(package / "samples" / "windows.npz",
                        window=batch.window.numpy(), token_valid=batch.token_valid.numpy(),
                        command=batch.command.numpy(), elapsed=batch.elapsed.numpy(),
                        anchor=batch.anchor, step_mask=batch.step_mask,
                        geometry_mask=batch.geometry_mask, truth=batch.future["geometry"],
                        end=panel["end"][picked], root=panel["root"][picked])
    (package / "results" / "expected_metrics.json").write_text(
        json.dumps(expected, indent=1, sort_keys=True, default=float))
    manifest = [{"path": str(path.relative_to(package)), "bytes": path.stat().st_size,
                 "sha256": sha256(path)} for path in sorted(package.rglob("*")) if path.is_file()]
    (package / "MANIFEST.json").write_text(json.dumps(
        {"files": manifest, "count": len(manifest),
         "scope": "verify_package.py runs the 11 repair checks and reproduces the packaged world "
                  "model sample; the RL training and the full test panel are not re-run",
         "note": "hashes validate the archive, not the conclusions"}, indent=1))
    archive = ROOT / "night_repair2_v1_review.zip"
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
