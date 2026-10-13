#!/usr/bin/env bash
# Rebuild every figure, table and video in this package.
# Reads only saved sequences, token arrays, checkpoints and analysis JSON.
# It does NOT retrain, does NOT re-collect data and does NOT run new physics.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
$PY experiments/night_sequence_v1/assets.py
echo "assets rebuilt -> presentation_assets/"
