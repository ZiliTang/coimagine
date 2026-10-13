"""Build ``coverage_scale_v1_256k_review.zip``.

    python experiments/coverage_scale/make_bundle_256k.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RUN = ROOT / "outputs/coverage_scale_v1/coverage-scale-20261013"
OUT_RUN = ROOT / "outputs/coverage_scale_v1/coverage-scale-256k"

SOURCE = [
    "src/coimagine/coverage_scale/evaluation.py",
    "src/coimagine/coverage_scale/trainer_ext.py",
    "src/coimagine/coverage_scale/candidate_eval.py",
    "src/coimagine/coverage_scale/trainer.py",
    "src/coimagine/coverage_scale/shards.py",
    "src/coimagine/coverage_scale/sampler.py",
    "src/coimagine/coverage_scale/recipe.py",
    "experiments/coverage_scale/run_matrix_256k.py",
    "experiments/coverage_scale/run_groups.py",
    "experiments/coverage_scale/run_eval_256k.py",
    "experiments/coverage_scale/make_report_256k.py",
    "experiments/coverage_scale/make_bundle_256k.py",
    "tests/test_coverage_scale.py",
    "tests/test_coverage_scale_256k.py",
]
DOCS = [
    "docs/coverage_scale_v1/report_256k.md",
    "docs/coverage_scale_v1/CONTRACT_256k_addendum.md",
    "docs/coverage_scale_v1/STATUS_256k.md",
    "docs/coverage_scale_v1/errata_256k.md",
]
RUN_FILES = [
    "model_list.json", "groups.json", "groups_table.json", "holdout_256k.json",
    "comparison.json", "comparison.md", "tier_256000_summary.json",
    "candidates/rows.jsonl", "candidates/aggregate_by_horizon.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="coverage_scale_v1_256k_review.zip")
    args = parser.parse_args()

    staging = Path(tempfile.mkdtemp(prefix="cov_scale_256k_"))
    missing = []
    for name in SOURCE + DOCS:
        source = ROOT / name
        if source.exists():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            missing.append(name)
    for name in RUN_FILES:
        source = OUT_RUN / name
        if source.exists():
            target = staging / "outputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for figure in sorted((OUT_RUN / "figures").glob("*.png")) if (OUT_RUN / "figures").exists() else []:
        target = staging / "figures" / figure.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure, target)
    for cell in sorted((OUT_RUN / "runs").glob("*")):
        if not cell.is_dir():
            continue
        target = staging / "runs" / cell.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("run_summary.json", "cell_meta.json", "selection_eval.json"):
            if (cell / name).exists():
                shutil.copy2(cell / name, target / name)

    models = staging / "models"
    models.mkdir(parents=True, exist_ok=True)
    entries = []
    for width in (256, 512):
        for seed in (0, 1, 2):
            for kind in ("best_new_val_total", "update60000"):
                source = OUT_RUN / "runs" / f"256000_{width}_seed{seed}" / f"{kind}.pt"
                if not source.exists():
                    continue
                target = models / f"256000_{width}_seed{seed}_{kind}.pt"
                shutil.copy2(source, target)
                entries.append({"file": target.name, "bytes": target.stat().st_size,
                                "sha256": sha256(target)})
    (models / "models.json").write_text(json.dumps(entries, indent=1, sort_keys=True))

    large = []
    for split in ("train", "val", "test"):
        directory = DATA_RUN / "shards" / split
        if not directory.exists():
            continue
        files = sorted(directory.glob("*.jsonl"))
        large.append({
            "path": str(directory.relative_to(ROOT)), "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "recovery": "python experiments/coverage_scale/collect.py --run-id coverage-scale-20261013",
            "last_file": files[-1].name if files else None,
            "last_file_sha256": sha256(files[-1]) if files else None,
        })
    for split in ("train",):
        path = DATA_RUN / "roots" / f"{split}.jsonl"
        if path.exists():
            large.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    (staging / "MISSING.md").write_text(
        "# Large artifacts left in the working tree\n\n"
        "The raw window shards and the root registries are not in this archive. Each entry gives\n"
        "the exact path, size, a hash and the command that regenerates or reads it deterministically.\n\n"
        "```json\n" + json.dumps(large, indent=1) + "\n```\n"
        + ("\n## Expected but absent\n\n" + "\n".join(f"- {name}" for name in missing) + "\n"
           if missing else "")
    )

    archive = ROOT / args.zip
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as handle:
        count = len(handle.namelist())
    print(json.dumps({"archive": str(archive), "files": count,
                      "bytes": archive.stat().st_size, "missing": missing}, indent=1))
    print("BUNDLE_256K_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
