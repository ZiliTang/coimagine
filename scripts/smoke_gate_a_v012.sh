#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUTPUT="${1:-outputs/gate_a/v012_smoke}"
JOBS="${JOBS:-2}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

rm -rf "${OUTPUT}"

python scripts/check_cst_template_support.py

python scripts/run_gate_a_horizon_study.py \
  --skip-controlled \
  --random-channel-per-difficulty 1 \
  --random-channel-candidate-budget 2 \
  --random-channel-steps-per-object 3 \
  --random-channel-extra-steps 0 \
  --random-channel-profile template-diagnostic \
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
if len(rows) != 9:
    raise SystemExit(f"expected 9 episode rows, got {len(rows)}")

expected_methods = {"CST-Frozen-K H=3", "CST H=3", "CoImagine-IET H=3"}
if {row["method"] for row in rows} != expected_methods:
    raise SystemExit(f"unexpected methods: {sorted({row['method'] for row in rows})}")

for row in rows:
    m = int(row["entity_count"])
    expected_budget = 3 * m
    actual_budget = int(row["episode_budget"])
    if actual_budget != expected_budget:
        raise SystemExit(
            f"episode budget propagation failed for seed={row['seed']}: "
            f"expected {expected_budget}, got {actual_budget}"
        )
    support = float(row["root_support_count_mean"])
    selected = float(row["root_candidate_count_mean"])
    truncated = float(row["root_truncated_template_count_mean"])
    if support + 1e-12 < selected:
        raise SystemExit(f"support smaller than selected: {row}")
    if abs((support - selected) - truncated) > 1e-6:
        raise SystemExit(f"template truncation accounting mismatch: {row}")
    if row["method"] == "CST-Frozen-K H=3" and int(row["predicted_recovered_root_template_count"]) != 0:
        raise SystemExit(f"Frozen-K recovered a truncated template: {row}")
    if row["method"] == "CST H=3" and int(row["predicted_new_template_count"]) != 0:
        raise SystemExit(f"CST introduced a non-root template: {row}")

manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
assert manifest["benchmark_version"] == "v0.12"
assert manifest["random_channel_steps_per_object"] == 3
assert manifest["random_channel_extra_steps"] == 0
assert manifest["random_channel_profile"] == "template-diagnostic"
PY

python scripts/run_gate_a_horizon_study.py \
  --render-only \
  --skip-debug-gallery \
  --output "${OUTPUT}"

echo "Gate-A v0.12 smoke test passed: ${OUTPUT}"
