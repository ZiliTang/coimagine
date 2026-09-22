#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/check_cst_template_support.py
python scripts/check_safe_search_optimizations.py
python scripts/check_randomized_channel_generator.py --per-difficulty 10
python - <<'PY'
from dataclasses import replace

from coimagine.core.types import PlannerMode
from coimagine.envs.grid.generation.procedural import (
    generate_randomized_channel_benchmark_case,
)
from coimagine.evaluation.episodes import rollout_receding_horizon

instance = generate_randomized_channel_benchmark_case(
    30_000_001, difficulty="easy"
)
metadata = dict(instance.state.task.metadata)
metadata["candidate_budget"] = 3
metadata["episode_max_steps"] = 1
instance = replace(
    instance,
    state=replace(
        instance.state,
        task=replace(instance.state.task, metadata=metadata),
    ),
)
for mode in (PlannerMode.COST, PlannerMode.IET):
    trace = rollout_receding_horizon(
        instance,
        mode=mode,
        horizon=3,
        candidate_budget=3,
        max_steps=1,
        observation_radius=int(metadata["observation_radius"]),
    )
    assert trace.option_count == 1
print("Gate-A v0.14 generator/planner smoke passed.")
PY
