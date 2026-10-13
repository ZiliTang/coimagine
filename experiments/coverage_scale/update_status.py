"""Regenerate the live part of ``docs/coverage_scale_v1/STATUS.md``.

    python experiments/coverage_scale/update_status.py --run-id coverage-scale-20261013
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RUN_START = 1791829800.0  # 2026-10-13 02:30:00 CST, recorded when the round started
HARD_LIMIT_HOURS = 8.0
SLOTS = ("8000", "16000", "32000", "64000", "128000")


def progress(run_dir: Path):
    shards = Counter()
    windows = Counter()
    for path in (run_dir / "collection_progress.jsonl", run_dir / "narrow_progress.jsonl"):
        if not path.exists():
            continue
        for line in path.open():
            row = json.loads(line)
            shards[row["split"]] += 1
            windows[row["split"]] += row["windows"]
    return shards, windows


def training_status(run_dir: Path):
    out = {}
    for tier in SLOTS:
        cells = []
        for cell in sorted((run_dir / "runs").glob(f"{tier}_*_seed*")):
            summary = cell / "run_summary.json"
            if summary.exists():
                cells.append(cell.name)
        if cells:
            out[tier] = cells
    narrow = sorted(path.name for path in (run_dir / "narrow").glob("*_seed*")
                    if (path / "run_summary.json").exists())
    return out, narrow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    shards, windows = progress(run_dir)
    trained, narrow = training_status(run_dir)
    now = time.time()
    elapsed = now - RUN_START
    lines = [
        "# STATUS - coverage_scale_v1",
        "",
        f"run_id `{args.run_id}`; wall clock since the round started: **{elapsed/3600:.2f} h** of an "
        f"8 h hard limit; reported at {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(now))}.",
        "",
        "## Collection",
        "",
        "| split | finished shards | windows |",
        "|---|---:|---:|",
    ]
    for split in ("train", "val", "test", "narrow"):
        lines.append(f"| {split} | {shards.get(split, 0)} | {windows.get(split, 0)} |")
    lines += ["", "## Training (finished cells)", ""]
    if trained:
        for tier, cells in trained.items():
            lines.append(f"- tier {tier}: {len(cells)}/6 cells - {', '.join(cells)}")
    else:
        lines.append("- none yet")
    if narrow:
        lines.append(f"- narrow control: {len(narrow)}/6 cells - {', '.join(narrow)}")
    running = subprocess.run(
        ["bash", "-c", "ps -eo comm,args | awk '$1 ~ /python/ && /coverage_scale/ {print $2}' | wc -l"],
        capture_output=True, text=True,
    ).stdout.strip()
    lines += [
        "",
        f"active coverage_scale processes: {running}",
        "",
        "## Deliverables",
        "",
        "- contract `docs/coverage_scale_v1/CONTRACT.md`; errata `docs/coverage_scale_v1/errata.md`",
        "- report `docs/coverage_scale_v1/report.md`; tables `docs/coverage_scale_v1/tables.md`;",
        "  coverage `docs/coverage_scale_v1/coverage_table.md`; manifest `docs/coverage_scale_v1/manifest.json`",
        "- raw shards, root registries and every checkpoint under this run directory",
        "- generator/collector/trainer/evaluator entries under `experiments/coverage_scale/`, library",
        "  under `src/coimagine/coverage_scale/`, acceptance tests `tests/test_coverage_scale.py`",
        "- review bundle `coverage_scale_v1_review.zip`",
        "",
        "## Recovery",
        "",
        "```bash",
        "python experiments/coverage_scale/collect.py --run-id coverage-scale-20261013   # skips finished shards",
        "python experiments/coverage_scale/collect_narrow.py --run-id coverage-scale-20261013 --target-windows 16017 --shards 48",
        "python experiments/coverage_scale/run_matrix.py --run-id coverage-scale-20261013 --tier 128000 --parallel 3",
        "python experiments/coverage_scale/run_matrix.py --run-id coverage-scale-20261013 --tier 16017 --split narrow --parallel 3",
        "python experiments/coverage_scale/run_decision.py --tiers 8000,16000,32000,64000,128000 --narrow-tier 16017",
        "python experiments/coverage_scale/run_holdout.py --tiers 8000,16000,32000,64000,128000 --narrow-tier 16017",
        "python experiments/coverage_scale/run_candidates.py --tiers 8000,16000,128000 --narrow-tier 16017",
        "```",
        "",
        "## Next step (one only)",
        "",
        "Keep every frozen condition and run the next tier at a 256k cap with the same six runs and the",
        "same new val/test. P is still falling at 128k and the C3 rule never asked to stop, so the round",
        "ended on its frozen scope limit, not on a measured plateau.",
        "",
    ]
    (ROOT / "docs" / "coverage_scale_v1" / "STATUS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
