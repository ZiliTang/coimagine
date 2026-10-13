"""Write ``docs/fit_capacity_v1/manifest.json`` from the finished run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILES = (
    "src/coimagine/fit_capacity/__init__.py",
    "src/coimagine/fit_capacity/candidates.py",
    "src/coimagine/fit_capacity/existing.py",
    "src/coimagine/fit_capacity/metrics.py",
    "src/coimagine/fit_capacity/trainer.py",
    "src/coimagine/continuous2d_learning/model.py",
    "src/coimagine/data_upgrade/training.py",
    "experiments/fit_capacity/run_candidates.py",
    "experiments/fit_capacity/run_existing.py",
    "experiments/fit_capacity/run_holdout.py",
    "experiments/fit_capacity/run_matrix.py",
    "experiments/fit_capacity/run_tables.py",
    "tests/test_fit_capacity.py",
    "docs/fit_capacity_v1/CONTRACT.md",
    "docs/fit_capacity_v1/STATUS.md",
    "docs/fit_capacity_v1/report.md",
    "docs/fit_capacity_v1/errata.md",
)
RUN_FILES = (
    "a1_audit.json", "existing_checkpoints.csv", "existing_fit_metrics.csv",
    "existing_fit_details.json", "training_matrix.json",
    "tables/progress_table.json", "tables/capacity_table.json", "tables/selection_table.json", "tables/tables.md",
    "holdout_metrics.csv", "holdout_details.json",
    "candidates/reference_verification.json", "candidates/aggregate.json", "candidates/metrics_table.md",
    "candidates/paired_256_512.json", "candidates/labels.json",
)
CHECKPOINT_KINDS = ("update1000.pt", "update3000.pt", "update6000.pt", "best_mixed_val.pt", "best_original_val.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-capacity-20261012")
    parser.add_argument("--out-root", default="outputs/fit_capacity_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id

    from coimagine.continuous2d_learning.model import ModelConfig

    manifest = {
        "run_id": args.run_id,
        "contract": "docs/fit_capacity_v1/CONTRACT.md",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(),
        "source_sha256": {},
        "matrix": {"data_groups": ["Old+Random", "Old+Branch"], "widths": [256, 512], "seeds": [0, 1, 2],
                   "total_updates": 6000, "batch_size": 128, "lr": 1e-3,
                   "eval_updates": list(__import__("coimagine.fit_capacity.trainer", fromlist=["EVAL_UPDATES"]).EVAL_UPDATES)},
        "parameter_counts": {str(width): sum(p.numel() for p in __import__(
            "coimagine.continuous2d_learning.model", fromlist=["JointWindowMLP"]
        ).JointWindowMLP(ModelConfig(hidden=width)).parameters()) for width in (256, 512)},
        "threads": 4,
    }
    try:
        import numpy, torch
        manifest["dependency_versions"] = {"numpy": numpy.__version__, "torch": torch.__version__}
    except Exception as error:
        manifest["dependency_versions"] = {"error": str(error)}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        manifest["source_sha256"][relative] = _sha256(path) if path.exists() else "MISSING"
    manifest["run_file_sha256"] = {
        name: (_sha256(run_dir / name) if (run_dir / name).exists() else "NOT_RUN") for name in RUN_FILES
    }
    manifest["checkpoints"] = {}
    for cell in sorted((run_dir / "runs").iterdir()):
        for kind in CHECKPOINT_KINDS:
            path = cell / kind
            if path.exists():
                manifest["checkpoints"][f"{cell.name}/{kind}"] = _sha256(path)
    manifest["input_hashes"] = {
        "data_upgrade/windows.jsonl": _sha256(ROOT / "outputs/data_upgrade_v1/data-upgrade-20261012/branch_data/branch_windows.jsonl"),
        "forecast_baseline/dataset/windows.jsonl": _sha256(ROOT / "outputs/forecast_baseline_v1/forecast-20261012/dataset/windows.jsonl"),
    }
    destination = ROOT / "docs/fit_capacity_v1/manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("wrote", destination)
    print("missing sources:", [k for k, v in manifest["source_sha256"].items() if v == "MISSING"])
    print("NOT_RUN:", [k for k, v in manifest["run_file_sha256"].items() if v == "NOT_RUN"])
    print("checkpoints recorded:", len(manifest["checkpoints"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
