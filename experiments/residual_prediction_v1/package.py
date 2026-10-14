"""Build the review ZIP for the residual round.

The package is meant to be verified **without this repository**: it carries raw window tensors, the
token masks, the residual references, the target-visible masks and the truths, together with the
frozen checkpoint and the transform descriptor. The verification script recomputes the hold baseline
and the model p50/p90 from those tensors and compares against the values stored here -- it does not
merely re-read a metrics file.
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
from experiments.residual_prediction_v1 import data as D  # noqa: E402
from experiments.residual_prediction_v1.evaluate import (  # noqa: E402
    decode_checkpoint_on,
    error_table,
    hold_baseline,
)

OUT = ROOT / "outputs/residual_prediction_v1"
DOCS = ROOT / "docs/residual_prediction_v1"
PACKAGE = ROOT / "residual_prediction_v1_review"
FROZEN = "mlp_history_residual_per_group_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_panel(tier: str, split: str, count: int, *, seed: int = 7) -> Dict[str, np.ndarray]:
    tokens = Tokens(D.TIER_SPLITS[tier])
    panel = D.load_panel(tier, split, OUT)
    generator = np.random.default_rng(seed)
    picked = np.sort(generator.choice(len(panel["index"]), size=min(count, len(panel["index"])),
                                      replace=False))
    index = panel["index"][picked]
    ends = panel["end"][picked]
    batch = D.build_batch(tokens, index, ends)
    return {"window": batch.window.numpy(), "token_valid": batch.token_valid.numpy(),
            "reference": batch.reference, "next_truth": batch.next_mid,
            "term_truth": batch.term_mid, "next_mask": batch.next_mask,
            "term_mask": batch.term_mask, "end": batch.ends,
            "speed": batch.current_speed,
            "visible_now": batch.current_visible.astype(np.float32),
            "root": np.asarray([str(r) for r in tokens.roots[index]]),
            "index": index, "panel_end": ends}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--out", default=str(PACKAGE))
    args = parser.parse_args()
    package = Path(args.out)
    if package.exists():
        shutil.rmtree(package)
    (package / "samples").mkdir(parents=True)
    (package / "checkpoints").mkdir(parents=True)
    (package / "reference").mkdir(parents=True)
    (package / "docs").mkdir(parents=True)

    samples = sample_panel(args.tier, "eval_test", args.samples)
    np.savez_compressed(package / "samples" / "windows_eval_test.npz", **samples)

    frozen = OUT / "checkpoints" / f"{FROZEN}.pt"
    shutil.copy2(frozen, package / "checkpoints" / frozen.name)
    for extra in ("mlp_history_direct_per_group_seed0", "transformer_history_residual_per_group_seed0",
                  "mlp_current_residual_per_group_seed0"):
        source = OUT / "checkpoints" / f"{extra}.pt"
        if source.exists():
            shutil.copy2(source, package / "checkpoints" / source.name)
    diagnostic = OUT / "checkpoints" / "single_head_residual_per_group_seed0.pt"
    if diagnostic.exists():
        shutil.copy2(diagnostic, package / "checkpoints" / "diagnostic_single_head.pt")

    # reference metric values, recomputed here from the packaged sample
    saved = torch.load(frozen, map_location="cpu", weights_only=False)
    model = _build(saved)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    transform = _transform(saved["spec"])
    window = torch.tensor(samples["window"], dtype=torch.float32)
    valid = torch.tensor(samples["token_valid"], dtype=torch.float32)
    with torch.no_grad():
        output = model(window, valid)["process_geometry"].numpy().astype(np.float64)
    prediction = _decode(output, samples["reference"], transform)
    hold = samples["reference"]
    reference = {
        "frozen_checkpoint": frozen.name,
        "samples": int(len(window)),
        "transform": saved["spec"],
        "hold": _metrics(hold, samples),
        "model": _metrics(prediction, samples),
        "panel": D.panel_hash(D.load_panel(args.tier, "eval_test", OUT)),
        "full_panel_values_from_main_results": _full_panel_values(),
    }
    (package / "reference" / "expected_metrics.json").write_text(
        json.dumps(reference, indent=1, sort_keys=True, default=float))

    for name in ("F1_single_head_diagnostic.png", "F2_tiny_set_fit.png", "F3_main_migration.png"):
        source = DOCS / name
        if source.exists():
            shutil.copy2(source, package / "docs" / name)
    for name in ("RESULTS.md", "CONTRACT.md", "REVIEW_FIXES.md", "CHECKPOINT_INTERFACE.md"):
        source = DOCS / name
        if source.exists():
            shutil.copy2(source, package / "docs" / name)
    for source in (OUT / "single_head_diagnostic_d1.json", OUT / "tiny_fit_d1.json",
                   OUT / "main_results_eval_test.json", OUT / "main_results_eval_val.json",
                   OUT / "comparisons_eval_test.json", OUT / "panels_d1.json"):
        if source.exists():
            shutil.copy2(source, package / "reference" / source.name)
    ledger = OUT / "RUN_LEDGER.jsonl"
    if ledger.exists():
        shutil.copy2(ledger, package / "reference" / ledger.name)
    shutil.copy2(ROOT / "experiments/residual_prediction_v1/verify_package.py",
                 package / "verify_package.py")
    # the transform module itself, for inspection: the verifier re-implements the arithmetic on
    # purpose, so shipping the original lets a reviewer compare the two side by side
    code = package / "code"
    code.mkdir(exist_ok=True)
    for name in ("targets.py", "data.py", "single_head.py", "evaluate.py"):
        shutil.copy2(ROOT / "experiments/residual_prediction_v1" / name, code / name)

    manifest = []
    for path in sorted(package.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(package)),
                             "bytes": path.stat().st_size, "sha256": sha256(path)})
    (package / "MANIFEST.json").write_text(json.dumps(
        {"files": manifest, "count": len(manifest),
         "note": "sizes and hashes of the packaged files; this does not by itself validate any "
                 "experimental conclusion"}, indent=1, sort_keys=True))

    archive = ROOT / "residual_prediction_v1_review.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(package))
    names = zipfile.ZipFile(archive).namelist()
    print(json.dumps({"archive": archive.name, "bytes": archive.stat().st_size,
                      "entries": len(names), "duplicates": len(names) - len(set(names)),
                      "sha256": sha256(archive),
                      "manifest_count": len(manifest)}, indent=1))
    print("PACKAGE_OK")
    return 0


def _build(saved):
    from experiments.night_sequence_v1.models import build_model
    return build_model(saved["model_kind"], **saved["model_kwargs"])


def _transform(descriptor):
    from experiments.residual_prediction_v1.evaluate import geometry_transform_from_descriptor
    return geometry_transform_from_descriptor(descriptor)


def _decode(output, reference, transform):
    from experiments.residual_prediction_v1.targets import decode_geometry
    return decode_geometry(output, reference, transform)[:, [0, 1, 4, 5]]


def _metrics(prediction: np.ndarray, samples: Dict[str, np.ndarray]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    truth = samples["next_truth"]
    mask = samples["next_mask"]
    for robot in (0, 1):
        columns = [0, 1] if robot == 0 else [2, 3]
        keep = (mask[:, columns[0]] > 0) & (mask[:, columns[1]] > 0) & (samples["end"] > 0)
        error = np.hypot(prediction[:, columns[0]] - truth[:, columns[0]],
                         prediction[:, columns[1]] - truth[:, columns[1]])
        out[f"r{robot}_t_gt_0"] = {
            "p50_cm": float(np.median(error[keep]) * 100),
            "p90_cm": float(np.quantile(error[keep], 0.9) * 100),
            "windows": int(keep.sum())}
    return out


def _full_panel_values() -> Dict[str, object]:
    path = OUT / "main_results_eval_test.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    entry = payload["models"].get(FROZEN, {})
    return {"source": "main_results_eval_test.json (full 2048-window panel)",
            "next_t_gt_0_r0": entry.get("next", {}).get("t>0", {}).get("r0"),
            "hold_next_t_gt_0_r0": entry.get("hold_next", {}).get("t>0", {}).get("r0")}


if __name__ == "__main__":
    raise SystemExit(main())
