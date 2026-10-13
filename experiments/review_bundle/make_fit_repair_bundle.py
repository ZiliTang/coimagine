"""Build ``fit_repair_v1_review.zip`` (read-only).

Thirty-six logical checkpoints map onto unique parameter hashes; duplicate weights are stored once
and the mapping is recorded in ``CHECKPOINT_MAP.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "fit_repair_v1_review.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_fit_repair_bundle"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

DIRECTORIES = ("src/coimagine/fit_capacity", "experiments/fit_repair")
FILES = (
    "experiments/fit_capacity/run_tables.py",
    "src/coimagine/continuous2d_learning/model.py",
    "src/coimagine/data_upgrade/training.py",
    "tests/test_fit_repair.py",
    "docs/fit_repair_v1/CONTRACT.md",
    "docs/fit_repair_v1/STATUS.md",
    "docs/fit_repair_v1/report.md",
    "docs/fit_repair_v1/errata.md",
    "docs/fit_repair_v1/manifest.json",
    "pyproject.toml",
)
RUN_FILES = (
    "data_hashes.json", "training_matrix.json",
    "tables/progress_table.json", "tables/capacity_table.json", "tables/selection_table.json", "tables/tables.md",
    "c1_termination/termination_attribution.json", "c2_geometry/geometry_attribution.json",
    "holdout_metrics.csv",
    "candidates/reference_verification.json", "candidates/aggregate.json", "candidates/paired.json",
    "candidates/identity_check.json", "candidates/summary.json", "candidates/metrics_table.md",
    "figures/fig_progress.png", "figures/fig_losses.png", "figures/fig_selection.png",
)
LARGE_FILES = tuple(
    f"runs/{group}_{width}_seed{seed}/{name}"
    for group in ("Old_Random", "Old_Branch")
    for width in (256, 512)
    for seed in (0, 1, 2)
    for name in ("batch_loss.jsonl", "eval_metrics.jsonl", "sample_order.bin", "resume_state.pt")
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
    parser.add_argument("--run-id", default="fit-repair-20261013")
    parser.add_argument("--out-root", default="outputs/fit_repair_v1")
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

    # store each unique parameter hash once, with the map from logical checkpoint -> stored file
    mapping: Dict[str, Dict[str, str]] = {}
    stored: Dict[str, str] = {}
    (STAGE / "checkpoints").mkdir(parents=True, exist_ok=True)
    for cell in sorted((run / "runs").iterdir()):
        summary = json.loads((cell / "run_summary.json").read_text())
        for kind, entry in sorted(summary["checkpoints"].items()):
            logical = f"{cell.name}/{kind}"
            digest = entry["parameter_hash"]
            if digest not in stored:
                target = f"checkpoints/{digest}.pt"
                _copy(str(Path(entry["path"]).relative_to(ROOT)), STAGE, [])
                shutil.move(str(STAGE / Path(entry["path"]).relative_to(ROOT)), str(STAGE / target))
                stored[digest] = target
                copied.append(target)
            mapping[logical] = {"parameter_hash": digest, "stored_as": stored[digest],
                                "updates": entry["updates"], "file_sha256": entry["file_sha256"]}
    (STAGE / "CHECKPOINT_MAP.json").write_text(json.dumps(mapping, indent=2, sort_keys=True))

    missing = {
        "not_copied_large_files": [
            {"path": str((run / name).relative_to(ROOT)),
             "sha256": _sha256(run / name) if (run / name).exists() else "MISSING",
             "restore": "python experiments/fit_repair/run_matrix.py --run-id fit-repair-20261013 "
                        "(deterministic: same seeds, same sample stream, same 6000 updates; resume supported)"}
            for name in LARGE_FILES
        ],
        "checkpoint_policy": "each unique parameter content hash is stored once; logical checkpoints "
                             "map onto them via CHECKPOINT_MAP.json",
        "note": "Every number in docs/fit_repair_v1/ comes from the copied run files or the listed large files.",
    }
    (STAGE / "MISSING.md").write_text("# Not copied into this bundle\n\n" + json.dumps(missing, indent=2) + "\n")
    lines = [
        "# fit_repair_v1 review bundle", "", "Reproduction commands:", "", "```bash",
        "python experiments/fit_repair/run_matrix.py --run-id fit-repair-20261013",
        "python experiments/fit_capacity/run_tables.py --run-id fit-repair-20261013 --out-root outputs/fit_repair_v1",
        "python experiments/fit_repair/run_c1_termination.py --run-id fit-repair-20261013",
        "python experiments/fit_repair/run_c2_geometry.py --run-id fit-repair-20261013",
        "python experiments/fit_repair/run_holdout.py --run-id fit-repair-20261013",
        "python experiments/fit_repair/run_candidates.py --run-id fit-repair-20261013",
        "python experiments/fit_repair/make_figures.py --run-id fit-repair-20261013",
        "pytest tests/test_fit_repair.py -q",
        "```", "", "## Files", "",
    ]
    hashes = {}
    for relative in sorted(set(copied)):
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
                      "unique_checkpoints": len(stored), "sha256": _sha256(bundle)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
