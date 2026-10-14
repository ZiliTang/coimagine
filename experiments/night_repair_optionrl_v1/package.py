"""Build the morning review archive.

The archive carries the reports, the source of the repaired and new modules, the frozen panels'
hashes, the core checkpoints, representative data shards, the figures and the behaviour clips, plus a
manifest with SHA256. The verification script in the archive recomputes the first-step error and the
hold baseline **from the packaged tensors** and states that it reproduces the packaged sample rather
than the full panel.
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

BASE = ROOT / "outputs/night_repair_optionrl_v1"
DOCS = ROOT / "docs/night_repair_optionrl_v1"
PACKAGE = ROOT / "night_repair_optionrl_v1_review"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PACKAGE))
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    package = Path(args.out)
    if package.exists():
        shutil.rmtree(package)
    for name in ("docs", "code", "checkpoints", "samples", "results", "figures", "clips", "data"):
        (package / name).mkdir(parents=True)
    for name in sorted(DOCS.glob("*.md")) + sorted(DOCS.glob("*.png")):
        shutil.copy2(name, package / ("figures" if name.suffix == ".png" else "docs") / name.name)
    # the verifier sits at the archive root so the clean-directory instruction is one command
    shutil.copy2(ROOT / "experiments/night_repair_optionrl_v1/verify_package.py",
                 package / "verify_package.py")
    for name in ("option_env.py", "policy.py", "ppo.py", "collect.py", "datacard.py", "render.py",
                 "report.py", "verify_invariants.py", "ablation.py", "verify_package.py"):
        source = ROOT / "experiments/night_repair_optionrl_v1" / name
        if source.exists():
            shutil.copy2(source, package / "code" / name)
    for name in ("data.py", "models.py", "train.py", "evaluate.py", "cache.py", "panels.py"):
        source = ROOT / "experiments/transformer_scale_latent_v1" / name
        if source.exists():
            shutil.copy2(source, package / "code" / f"world_{name}")
    checkpoints = list((BASE / "world_models/checkpoints").glob("*.pt"))
    for checkpoint in checkpoints:
        shutil.copy2(checkpoint, package / "checkpoints" / checkpoint.name)
    policy = BASE / "option_rl/policy_seed0.pt"
    if policy.exists():
        shutil.copy2(policy, package / "checkpoints" / policy.name)
    for name in ("multi_step_results_test_multistep.json", "ablation_test_multistep.json",
                 "ablation_per_sample_test_multistep.npz", "TABLES.md"):
        source = BASE / "world_models" / name
        if source.exists():
            shutil.copy2(source, package / "results" / name)
    for name in ("random_baseline.json", "train_result_seed0.json"):
        source = BASE / "option_rl" / name
        if source.exists():
            shutil.copy2(source, package / "results" / name)
    for name in ("TRIAL_SUMMARY.json", "DATA_CARD.md", "DATACARD.json"):
        source = BASE / "collection" / name
        if source.exists():
            shutil.copy2(source, package / "data" / name)
    for name in ("TRIAL_SUMMARY.json", "DATA_CARD.md", "DATACARD.json"):
        source = BASE / "collection_expanded" / name
        if source.exists():
            shutil.copy2(source, package / "data" / f"expanded_{name}")
    # a small representative slice of each source, kept out of the archive's size budget
    shards = []
    for source_name in ("collection", "collection_expanded"):
        for directory in sorted((BASE / source_name).iterdir()):
            if directory.is_dir():
                shards.extend(sorted(directory.glob("episode_*.npz"))[:6])
    for shard in shards[:24]:
        shutil.copy2(shard, package / "data" / shard.name)
    clips = sorted((BASE / "report/clips").glob("*.*")) if (BASE / "report/clips").exists() else []
    for clip in clips[:8]:
        shutil.copy2(clip, package / "clips" / clip.name)
    # the packaged sample the verifier recomputes
    sys.path.insert(0, str(ROOT))
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
    expected: Dict[str, object] = {"samples": int(len(picked)), "panel_hash":
                                   json.loads((BASE / "world_models/panel_test_multistep.json")
                                              .read_text())["hash"]
                                   if (BASE / "world_models/panel_test_multistep.json").exists()
                                   else None, "models": {}}
    for name in ("repair_direct_multistep_seed0", "repair_latent_autoregressive_seed0"):
        checkpoint = BASE / "world_models/checkpoints" / f"{name}.pt"
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
                 "sha256": sha256(path)}
                for path in sorted(package.rglob("*")) if path.is_file()]
    (package / "MANIFEST.json").write_text(json.dumps(
        {"files": manifest, "count": len(manifest),
         "scope": "the packaged sample is reproduced by verify_package.py; the full panel is not "
                  "re-run here",
         "note": "hashes validate the archive, not the experimental conclusions"}, indent=1))
    archive = ROOT / "night_repair_optionrl_v1_review.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(package))
    names = zipfile.ZipFile(archive).namelist()
    print(json.dumps({"archive": archive.name, "bytes": archive.stat().st_size,
                      "entries": len(names), "duplicates": len(names) - len(set(names)),
                      "sha256": sha256(archive), "checkpoints": len(checkpoints),
                      "shards": len(shards[:24]), "clips": len(clips)}, indent=1))
    print("PACKAGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
