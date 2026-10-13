"""Build ``coverage_scale_v1_review.zip`` (task book section 8).

    python experiments/coverage_scale/make_bundle.py --run-id coverage-scale-20261013

The archive carries the modified source, the tests, the frozen contract, the generator
configuration, root-scene samples and figures, coverage tables, learning-curve source tables,
every run summary, the per-start candidate evaluation and the first-tier / final-tier main
models. The raw windows and the remaining checkpoints stay in the working tree; MISSING.md
inside the archive names their exact paths, sizes, hashes and the recovery command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SOURCE_FILES = [
    "src/coimagine/continuous2d/scenes.py",
    "src/coimagine/coverage_scale/__init__.py",
    "src/coimagine/coverage_scale/scene_registry.py",
    "src/coimagine/coverage_scale/sampler.py",
    "src/coimagine/coverage_scale/plan.py",
    "src/coimagine/coverage_scale/recipe.py",
    "src/coimagine/coverage_scale/shards.py",
    "src/coimagine/coverage_scale/evaluation.py",
    "src/coimagine/coverage_scale/trainer.py",
    "src/coimagine/coverage_scale/candidate_eval.py",
    "experiments/coverage_scale/collect.py",
    "experiments/coverage_scale/collect_narrow.py",
    "experiments/coverage_scale/run_calibration.py",
    "experiments/coverage_scale/run_matrix.py",
    "experiments/coverage_scale/run_decision.py",
    "experiments/coverage_scale/run_candidates.py",
    "experiments/coverage_scale/make_report.py",
    "experiments/coverage_scale/make_coverage.py",
    "experiments/coverage_scale/run_holdout.py",
    "experiments/coverage_scale/verify_invariants.py",
    "experiments/coverage_scale/update_status.py",
    "experiments/coverage_scale/write_manifest.py",
    "experiments/coverage_scale/make_bundle.py",
    "tests/test_coverage_scale.py",
]

DOC_FILES = [
    "docs/coverage_scale_v1/CONTRACT.md",
    "docs/coverage_scale_v1/STATUS.md",
    "docs/coverage_scale_v1/report.md",
    "docs/coverage_scale_v1/errata.md",
    "docs/coverage_scale_v1/manifest.json",
    "docs/coverage_scale_v1/tables.md",
    "docs/coverage_scale_v1/coverage_table.md",
]

RUN_FILES = [
    "root_plan_summary.json",
    "calibration.json",
    "collection_summary.json",
    "narrow_summary.json",
    "decision.json",
    "learning_curve_rows.json",
    "tier_table.json",
    "narrow_table.json",
    "tables.md",
    "candidates/aggregate.json",
    "candidates/reference.json",
    "candidates/rows.jsonl",
    "coverage_table.json",
    "holdout.json",
    "invariants.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--zip", default="coverage_scale_v1_review.zip")
    parser.add_argument("--final-tier", default="")
    parser.add_argument("--first-tier", default="8000")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    staging = Path(tempfile.mkdtemp(prefix="coverage_scale_bundle_"))
    missing = []

    for name in SOURCE_FILES + DOC_FILES:
        source = ROOT / name
        if source.exists():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            missing.append(name)
    for name in RUN_FILES:
        source = run_dir / name
        if source.exists():
            target = staging / "outputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    figures = run_dir / "figures"
    if figures.exists():
        for source in sorted(figures.glob("*.png")):
            target = staging / "figures" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    # The complete scene registries (task book section 8) plus the small split root metadata.
    registries = staging / "scene_registries"
    registries.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        source = run_dir / "roots" / f"scenes_{split}.json"
        if source.exists():
            shutil.copy2(source, registries / f"scenes_{split}.json")
    for split in ("val", "test", "narrow"):
        source = run_dir / "roots" / f"roots_{split}.jsonl"
        if source.exists():
            shutil.copy2(source, registries / f"roots_{split}.jsonl")

    samples = staging / "scene_samples"
    samples.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "dev"):
        path = run_dir / "roots" / f"scenes_{split}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        subset = dict(payload)
        subset["scenes"] = payload["scenes"][:6]
        subset["count"] = len(subset["scenes"])
        subset["note"] = f"first 6 of {payload['count']} scenes; the full registry stays in the working tree"
        (samples / f"scenes_{split}_sample.json").write_text(json.dumps(subset, indent=1, sort_keys=True))

    # run summaries and the whole calibration / tier bookkeeping
    runs_out = staging / "runs"
    for cell in sorted((run_dir / "runs").glob("*")):
        if not cell.is_dir():
            continue
        target = runs_out / cell.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("run_summary.json", "cell_meta.json", "selection_eval.json",
                     "full_train_eval.json"):
            if (cell / name).exists():
                shutil.copy2(cell / name, target / name)
        schedule = cell / "eval_metrics.jsonl"
        if schedule.exists():
            shutil.copy2(schedule, target / "eval_metrics.jsonl")

    # the first-tier and final-tier main selection points, both widths, seed 0
    models = staging / "models"
    model_manifest = []
    for tier in [args.first_tier, args.final_tier]:
        if not tier:
            continue
        for width in (256, 512):
            for seed in (0, 1, 2):
                source = run_dir / "runs" / f"{tier}_{width}_seed{seed}" / "best_new_val_total.pt"
                if not source.exists():
                    continue
                target = models / f"{tier}_{width}_seed{seed}_best_new_val_total.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                model_manifest.append({
                    "tier": tier, "width": width, "seed": seed,
                    "file": target.name, "bytes": target.stat().st_size, "sha256": sha256(target),
                })
    (staging / "models" / "models.json").write_text(json.dumps(model_manifest, indent=1, sort_keys=True))

    # MISSING: exact paths, sizes and hashes of the large artifacts left in place
    large = []
    for split in ("train", "val", "test", "narrow"):
        directory = run_dir / "shards" / split
        if not directory.exists():
            continue
        files = sorted(directory.glob("*.jsonl"))
        total = sum(path.stat().st_size for path in files)
        large.append({
            "path": str(directory.relative_to(ROOT)), "files": len(files), "bytes": total,
            "recovery": f"python experiments/coverage_scale/collect.py --run-id {args.run_id}  # skips finished shards",
            "first_file_sha256": sha256(files[0]) if files else None,
            "last_file": files[-1].name if files else None,
            "last_file_sha256": sha256(files[-1]) if files else None,
        })
    for split in ("train", "val", "test", "narrow"):
        path = run_dir / "roots" / f"scenes_{split}.json"
        if path.exists():
            large.append({
                "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "recovery": f"RootPlan('{run_dir.relative_to(ROOT)}/roots').ensure('{split}', n)",
            })
    (staging / "MISSING.md").write_text(
        "# Large artifacts left in the working tree\n\n"
        "They are omitted from the archive because of size; every one of them is regenerated\n"
        "deterministically by the commands below and verified by the listed hash.\n\n"
        "```json\n" + json.dumps(large, indent=1) + "\n```\n"
    )
    if missing:
        (staging / "MISSING.md").write_text(
            (staging / "MISSING.md").read_text() + "\n\n## Files expected but absent at bundle time\n\n"
            + "\n".join(f"- {name}" for name in missing) + "\n"
        )

    archive = ROOT / args.zip
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    print(json.dumps({"archive": str(archive), "files": len(names),
                      "bytes": archive.stat().st_size, "missing": missing}, indent=1))
    print("BUNDLE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
