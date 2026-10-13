"""Write ``docs/fit_repair_v1/manifest.json`` from the finished run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILES = (
    "src/coimagine/fit_capacity/candidates.py",
    "src/coimagine/fit_capacity/checkpoints.py",
    "src/coimagine/fit_capacity/existing.py",
    "src/coimagine/fit_capacity/geometry_attribution.py",
    "src/coimagine/fit_capacity/metrics.py",
    "src/coimagine/fit_capacity/paired.py",
    "src/coimagine/fit_capacity/termination.py",
    "src/coimagine/fit_capacity/trainer.py",
    "experiments/fit_repair/make_figures.py",
    "experiments/fit_repair/run_c1_termination.py",
    "experiments/fit_repair/run_c2_geometry.py",
    "experiments/fit_repair/run_candidates.py",
    "experiments/fit_repair/run_holdout.py",
    "experiments/fit_repair/run_matrix.py",
    "experiments/fit_repair/write_manifest.py",
    "experiments/fit_capacity/run_tables.py",
    "tests/test_fit_repair.py",
    "docs/fit_repair_v1/CONTRACT.md",
    "docs/fit_repair_v1/STATUS.md",
    "docs/fit_repair_v1/report.md",
    "docs/fit_repair_v1/errata.md",
)
RUN_FILES = (
    "data_hashes.json", "training_matrix.json",
    "tables/progress_table.json", "tables/capacity_table.json", "tables/selection_table.json", "tables/tables.md",
    "c1_termination/termination_attribution.json", "c2_geometry/geometry_attribution.json",
    "holdout_metrics.csv", "holdout_details.json",
    "candidates/reference_verification.json", "candidates/aggregate.json", "candidates/paired.json",
    "candidates/identity_check.json", "candidates/summary.json", "candidates/metrics_table.md",
    "figures/fig_progress.png", "figures/fig_losses.png", "figures/fig_selection.png",
)
CHECKPOINTS = ("update1000.pt", "update3000.pt", "update6000.pt", "best_mixed_val.pt", "best_original_val.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-repair-20261013")
    parser.add_argument("--out-root", default="outputs/fit_repair_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id

    from coimagine.fit_capacity.trainer import EVAL_UPDATES
    from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig

    manifest = {
        "run_id": args.run_id,
        "contract": "docs/fit_repair_v1/CONTRACT.md",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(),
        "source_sha256": {},
        "matrix": {"data_groups": ["Old+Random", "Old+Branch"], "widths": [256, 512], "seeds": [0, 1, 2],
                   "total_updates": 6000, "batch_size": 128, "lr": 1e-3,
                   "eval_updates": list(EVAL_UPDATES), "threads": 4},
        "parameter_counts": {str(w): sum(p.numel() for p in JointWindowMLP(ModelConfig(hidden=w)).parameters()) for w in (256, 512)},
        "loss_reduction": "per_sample = torch.stack(components, dim=0).sum(dim=0); loss = per_sample.mean()  # A1 fix",
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
        summary = json.loads((cell / "run_summary.json").read_text())
        for kind, entry in sorted(summary["checkpoints"].items()):
            manifest["checkpoints"][f"{cell.name}/{kind}"] = {
                "file_sha256": entry["file_sha256"], "parameter_hash": entry["parameter_hash"],
                "updates": entry["updates"],
            }
    destination = ROOT / "docs/fit_repair_v1/manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("wrote", destination)
    print("missing sources:", [k for k, v in manifest["source_sha256"].items() if v == "MISSING"])
    print("NOT_RUN:", [k for k, v in manifest["run_file_sha256"].items() if v == "NOT_RUN"])
    print("checkpoints:", len(manifest["checkpoints"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
