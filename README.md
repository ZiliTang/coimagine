# CoImagine — 2D protocol core

Executable 2D validation for the CoImagine v6 research plan
(`docs/CoImagine_v6_研究方案与创新评估.md`), scoped by the first-round execution
prompt (`docs/CoImagine_2D_Agent_执行提示_v1.md`).

The repository asks a different question than the previous Gate A line: **under
hidden contact physics, how much of what message does a team need to plan
preparation, identification and execution?** The old Gate A code is archived under
[`_old/`](_old/README.md) and is never imported.

Current status: **Stage 1 (micro finite-state oracles) complete; Stage 2 (6×6 grid)
in progress** — the grid world model, egocentric ray sensing and the
publication-quality scene renderer are implemented; dynamics, candidate generation,
beliefs, A0–A3 and the baselines are next. See [`results.md`](results.md) and
[`next_steps.md`](next_steps.md).

## What is implemented

- `coimagine.protocol` — strictly isolated `TrueState` / `PublicContext` /
  `PrivateHistory` / `OracleContext`, a `PublicRecord` with checked invariants, and a
  runtime information-boundary guard.
- `coimagine.toy` — an explicit finite POMDP engine with a **complete joint
  observation table** `P(o_A, o_B, y_pub | x, theta, a, x')`, and the instances
  M0, M1, M2 (+ two negative controls) and M3.
- `coimagine.exact` — exact Bayes filters `P(x, theta | h_i, C)` and the public
  coordination belief `P(x, theta, h_A, h_B | C)`; discrete codecs with real bit
  accounting; the oracle hierarchy `O_full`, `O_central`, `O_codec`, exact
  single-boundary `O_B`, and `TrueModelLimitedSearch`.
- `coimagine.diagnostics` — runs the micro oracle suite, writes the gap
  decomposition, and renders the communication-budget, probe-cost and
  feedback-reliability figures.
- `coimagine.grid` — Stage 2: the discrete 6×6 world, occlusion-aware egocentric ray
  sensing with an explicit noise model, and a publication-quality scene renderer
  (ground truth versus each robot's information state) that emits PNG/PDF/SVG.

No learning, no neural model and no external API are used in this round.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

On a host whose site-packages register the ROS `launch_testing` / `launch_ros`
pytest plugins without `lark`, keep the plugin block that is already configured in
`pyproject.toml` (`-p no:launch_testing -p no:launch_ros`).

## Commands

```bash
# key tests
pytest

# validate every micro model table
python -m coimagine.cli check

# run all micro oracle diagnostics (writes CSV/JSON/Markdown under outputs/micro)
python -m coimagine.cli micro --output outputs/micro --print-table

# render the budget / probe-cost / feedback-reliability / C2-control figures
python -m coimagine.cli sweep --output outputs/micro/figures

# Stage 2: render the scene gallery (ground truth vs each robot's information state)
python -m coimagine.cli gallery --output outputs/grid/scenes --noise-comparison

# convenience wrapper around the two steps above
python scripts/run_micro_oracles.py
```

The console script `coimagine-2d` exposes the same subcommands after installation.

## Output layout

```text
outputs/micro/
├── oracle_results.csv / .json / .md
├── oracle_gap_decomposition.json
└── figures/
    ├── fig_communication_budget.png
    ├── fig_probe_cost_sweep.png
    ├── fig_feedback_strength_sweep.png
    └── fig_c2_control.png
```

Stage 2 scene gallery:

```text
outputs/grid/scenes/
├── scene_<name>.{png,pdf}             # ground truth | robot A view | robot B view
└── noise_comparison_<name>.{png,pdf}  # noiseless vs noisy information state
```

## Read next

- protocol contract: [`docs/2d_protocol_spec.md`](docs/2d_protocol_spec.md)
- results and interpretation: [`results.md`](results.md)
- what to do next: [`next_steps.md`](next_steps.md)
- working rules for agents: [`AGENTS.md`](AGENTS.md)
