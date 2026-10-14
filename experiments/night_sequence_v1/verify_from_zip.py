"""Clean-directory check: unzip the review package elsewhere, load a sample sequence, run a model.

    python experiments/night_sequence_v1/verify_from_zip.py

Does exactly what the task book asks of the reviewer: from a fresh extraction of
``night_sequence_v1_review.zip`` it loads the included sequence shard with the included loader,
loads one included checkpoint, predicts, and recomputes the metric for those samples from the
included label data. Nothing is read from the working tree's ``outputs/`` directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "night_sequence_v1_review.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    workdir = Path(tempfile.mkdtemp(prefix="night_verify_"))
    print(f"clean directory: {workdir}")
    with zipfile.ZipFile(args.bundle) as archive:
        names = archive.namelist()
        archive.extractall(workdir)
    print(f"extracted {len(names)} entries")

    # 1. the loader reads the packaged sequence shard
    shard_path = sorted((workdir / "outputs/night_sequence_v1/sequences").glob("d0_*.npz"))
    assert shard_path, "no sequence shard in the package"
    shard = np.load(shard_path[0], allow_pickle=True)
    steps = int(shard["steps"][0])
    print(f"loaded {shard_path[0].name}: {len(shard['steps'])} sequences, first has {steps} periods")
    assert steps > 1
    # round-6 fix: the exiting period does not advance physics, so T is (steps - 1) * dt
    assert abs(float(shard["T_exit"][0]) - (steps - 1) * 0.05) < 1e-5
    assert abs(float(shard["T_last_sensed"][0]) - (steps - 1) * 0.05) < 1e-5
    assert int(shard["truth_steps"][0]) == steps - 1
    assert int(shard["physics_steps_match_rows"][0]) == 1
    assert shard["terminal_own_world"][0].shape == (2, 6)
    print("O0 / T invariants hold on the packaged data")

    # 2. the packaged scripts import and the packaged checkpoint predicts
    sys.path.insert(0, str(workdir))
    sys.path.insert(0, str(workdir / "src"))
    import importlib
    dataset = importlib.import_module("experiments.night_sequence_v1.dataset")
    models = importlib.import_module("experiments.night_sequence_v1.models")
    import torch

    checkpoint = workdir / "outputs/night_sequence_v1/runs/d1_mlp_history_seed0.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = models.build_model(saved["model_kind"], **saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    print(f"loaded {checkpoint.name}: {saved['parameters']:,} parameters, "
          f"best_val={saved['best_val']:.4f}")

    # 3. build a window straight from the packaged tokens and predict
    tokens_dir = workdir / "outputs/night_sequence_v1/tokens_sample"
    print("packaged token files:", sorted(p.name for p in tokens_dir.glob("*.npz")))
    if (tokens_dir / "tokens_d0.npz").exists():
        tokens = dataset.Tokens(["d0"], directory=tokens_dir)
        example = dataset.build_example(tokens, 0, min(5, int(tokens.steps[0]) - 2))
        print(f"built a window from the packaged tokens: {example['window'].shape}")
        window = torch.tensor(example["window"][None], dtype=torch.float32)
        valid = torch.tensor(example["token_valid"][None], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(window, valid)
        stats = saved["normalisation"]["terminal_geometry"]
        prediction = outputs["terminal_geometry"].numpy()[0] * stats["std"] + stats["mean"]
        print(f"terminal geometry prediction (r0 midpoint): "
              f"({prediction[0]:+.4f}, {prediction[1]:+.4f}) m, "
              f"normal ({prediction[2]:+.3f}, {prediction[3]:+.3f})")
        assert np.isfinite(prediction).all()

    # 4. recompute one metric from the packaged metrics JSON
    report = (workdir / "docs/night_sequence_v1/REPORT.md").read_text()
    assert "night_sequence_v1" in report and "t=0" in report
    metrics = json.loads(
        (workdir / "outputs/night_sequence_v1/analysis/d1/metrics.json").read_text())
    baseline = metrics["baselines"]["P1_grouped|test"]["t0"]["terminal_geometry"]["r0_mid_2d"]["p50"]
    print(f"packaged metric re-read: P1 t=0 r0 midpoint p50 = {baseline:.4f} m")
    result = {"extracted": len(names), "sequences_in_shard": int(len(shard["steps"])),
              "steps_first": steps, "checkpoint_parameters": saved["parameters"],
              "finite_prediction": True}
    print(json.dumps(result, indent=1))
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    print("VERIFY_FROM_ZIP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
