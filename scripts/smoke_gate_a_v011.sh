#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUTPUT="${1:-outputs/gate_a/v011_smoke}"
JOBS="${JOBS:-2}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

rm -rf "${OUTPUT}"

python scripts/run_gate_a_horizon_study.py \
  --skip-controlled \
  --random-channel-per-difficulty 1 \
  --random-channel-candidate-budget 4 \
  --random-channel-steps-per-object 4 \
  --random-channel-extra-steps 1 \
  --random-channel-profile main \
  --jobs "${JOBS}" \
  --skip-story-figures \
  --skip-debug-gallery \
  --output "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
required = (
    "random_channel_episode_results.csv",
    "random_channel_failure_diagnostics.csv",
    "random_channel_failure_summary.json",
    "random_channel_failures.log",
    "manifest.json",
)
for name in required:
    path = output / name
    if not path.exists():
        raise SystemExit(f"missing smoke artifact: {path}")

with (output / "random_channel_episode_results.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 6:
    raise SystemExit(f"expected 6 episode rows, got {len(rows)}")
for row in rows:
    m = int(row["entity_count"])
    expected = 4 * m + 1
    actual = int(row["episode_budget"])
    if actual != expected:
        raise SystemExit(
            f"episode budget propagation failed for seed={row['seed']}: "
            f"expected {expected}, got {actual}"
        )
    if row["termination_reason"] not in {
        "completed",
        "budget_exhausted",
        "execution_failure",
        "planner_no_admitted_root_option",
    }:
        raise SystemExit(f"unexpected termination reason: {row['termination_reason']}")

manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
assert manifest["benchmark_version"] == "v0.11"
assert manifest["random_channel_steps_per_object"] == 4
assert manifest["random_channel_extra_steps"] == 1
PY

python scripts/run_gate_a_horizon_study.py \
  --render-only \
  --skip-debug-gallery \
  --output "${OUTPUT}"

echo "Gate-A v0.11 smoke test passed: ${OUTPUT}"
