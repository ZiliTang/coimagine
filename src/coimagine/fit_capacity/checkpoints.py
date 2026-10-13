"""Iterate the logical checkpoints of a training matrix run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

SELECTIONS = ("best_mixed_val", "best_original_val", "fixed_6000")


def iter_checkpoints(
    run_dir: Path, selections: Sequence[str] = SELECTIONS
) -> Iterator[Dict[str, object]]:
    for cell in sorted((run_dir / "runs").iterdir()):
        summary_path = cell / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        for selection in selections:
            checkpoint = summary["checkpoints"].get(selection)
            if checkpoint is None:
                continue
            yield {
                "cell": cell.name,
                "group": summary["group"],
                "width": int(summary["width"]),
                "model_seed": int(summary["seed"]),
                "selection": selection,
                "path": checkpoint["path"],
                "updates": int(checkpoint["updates"]),
                "checkpoint_hash": checkpoint["parameter_hash"],
                "file_sha256": checkpoint["file_sha256"],
                "label": f"{summary['group']}|w{summary['width']}|seed{summary['seed']}|{selection}",
            }


def load_data(run_root: Path, upgrade_run: Path, source_run: Path):
    original = [json.loads(line) for line in (source_run / "dataset" / "windows.jsonl").open()]
    branch = [json.loads(line) for line in (upgrade_run / "branch_data" / "branch_windows.jsonl").open()]
    return original, branch
