#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-outputs/gate_a/v010_smoke}"
JOBS="${JOBS:-2}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/run_gate_a_horizon_study.py \
  --skip-controlled \
  --random-channel-per-difficulty 1 \
  --random-channel-candidate-budget 4 \
  --random-channel-max-steps-cap 6 \
  --random-channel-profile main \
  --jobs "${JOBS}" \
  --skip-story-figures \
  --output "${OUTPUT}"

python scripts/run_gate_a_horizon_study.py \
  --render-only \
  --output "${OUTPUT}"

echo "Gate-A v0.10 smoke test passed: ${OUTPUT}"
