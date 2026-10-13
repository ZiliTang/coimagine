"""Build ``fit_capacity_v1_review.zip`` (read-only; nothing is re-run)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "fit_capacity_v1_review.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_fit_capacity_bundle"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

DIRECTORIES = ("src/coimagine/fit_capacity", "experiments/fit_capacity")
FILES = (
    "src/coimagine/data_upgrade/training.py",
    "src/coimagine/continuous2d_learning/model.py",
    "src/coimagine/continuous2d_learning/diagnostic.py",
    "src/coimagine/continuous2d_learning/rollout.py",
    "tests/test_fit_capacity.py",
    "docs/fit_capacity_v1/CONTRACT.md",
    "docs/fit_capacity_v1/STATUS.md",
    "docs/fit_capacity_v1/report.md",
    "docs/fit_capacity_v1/errata.md",
    "docs/fit_capacity_v1/manifest.json",
    "pyproject.toml",
)
RUN_FILES = (
    "a1_audit.json", "existing_checkpoints.csv", "existing_fit_metrics.csv",
    "training_matrix.json",
    "tables/progress_table.json", "tables/capacity_table.json", "tables/selection_table.json", "tables/tables.md",
    "holdout_metrics.csv",
    "candidates/reference_verification.json", "candidates/aggregate.json",
    "candidates/metrics_table.md", "candidates/paired_256_512.json",
)
MAIN_CHECKPOINTS = tuple(
    f"runs/{group}_{width}_seed{seed}/best_mixed_val.pt"
    for group in ("Old_Random", "Old_Branch")
    for width in (256, 512)
    for seed in (0, 1, 2)
)
LARGE_FILES = tuple(
    f"runs/{group}_{width}_seed{seed}/{name}"
    for group in ("Old_Random", "Old_Branch")
    for width in (256, 512)
    for seed in (0, 1, 2)
    for name in ("batch_loss.jsonl", "eval_metrics.jsonl", "update1000.pt", "update3000.pt", "update6000.pt",
                 "best_original_val.pt", "resume_state.pt")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(relative: str, stage: Path, copied: List[str]) -> None:
    source = ROOT / relative
    if not source.exists():
        return
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.append(relative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-capacity-20261012")
    parser.add_argument("--out-root", default="outputs/fit_capacity_v1")
    args = parser.parse_args()
    run = ROOT / args.out_root / args.run_id
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    copied: List[str] = []
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            _copy(str(path.relative_to(ROOT)), STAGE, copied)
    for relative in FILES:
        _copy(relative, STAGE, copied)
    for relative in RUN_FILES:
        _copy(str((run / relative).relative_to(ROOT)), STAGE, copied)
    for relative in MAIN_CHECKPOINTS:
        _copy(str((run / relative).relative_to(ROOT)), STAGE, copied)

    missing = {
        "not_copied_large_files": [
            {"path": str((run / name).relative_to(ROOT)),
             "sha256": _sha256(run / name) if (run / name).exists() else "MISSING",
             "restore": "python experiments/fit_capacity/run_matrix.py --run-id fit-capacity-20261012 "
                        "(deterministic: same seeds, same sample order, same 6000 updates)"}
            for name in LARGE_FILES
        ],
        "note": "The 12 best_mixed_val checkpoints are included. Every number in docs/fit_capacity_v1/ "
                "comes from the copied run files or the listed large files.",
    }
    (STAGE / "MISSING.md").write_text("# Not copied into this bundle\n\n" + json.dumps(missing, indent=2) + "\n")
    lines = [
        "# fit_capacity_v1 review bundle", "", "Reproduction commands:", "", "```bash",
        "python experiments/fit_capacity/run_existing.py --run-id fit-capacity-20261012",
        "python experiments/fit_capacity/run_matrix.py --run-id fit-capacity-20261012",
        "python experiments/fit_capacity/run_tables.py --run-id fit-capacity-20261012",
        "python experiments/fit_capacity/run_holdout.py --run-id fit-capacity-20261012",
        "python experiments/fit_capacity/run_candidates.py --run-id fit-capacity-20261012",
        "pytest tests/test_fit_capacity.py -q",
        "```", "", "## Files", "",
    ]
    hashes = {}
    for relative in sorted(copied):
        hashes[relative] = _sha256(STAGE / relative)
        lines.append(f"- `{relative}`")
    (STAGE / "REVIEW_INDEX.md").write_text("\n".join(lines) + "\n")
    (STAGE / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))
    bundle = ROOT / BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE))
    with zipfile.ZipFile(bundle) as archive:
        count = len(archive.namelist())
    print(json.dumps({"bundle": str(bundle), "files": count, "size_bytes": bundle.stat().st_size,
                      "sha256": _sha256(bundle)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
