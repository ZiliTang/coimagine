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

`pymunk==6.9.0` is a declared core dependency (see `pyproject.toml`); the continuous
environment does not run without it. The other dependencies are not pinned to a lock file.

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

## Migration task v1 — legacy-grid audit and continuous 2D prototype

Executed per `docs/CoImagine_Agent_迁移任务书_v1.md`. Status and evidence:
[`STATUS.md`](STATUS.md); short report: [`docs/migration_v1/report.md`](docs/migration_v1/report.md);
frozen facts: [`docs/migration_v1/interface_table.md`](docs/migration_v1/interface_table.md)
and [`docs/migration_v1/manifest.json`](docs/migration_v1/manifest.json).

```bash
# Phase A: one real action and one model prediction on a frozen legacy scene
python experiments/grid_selection_audit/phase_a_check.py

# Phase B: the legacy-grid fixed-candidate selection-bias diagnostic
python experiments/grid_selection_audit/run_audit.py --run-id phaseB-full
#   -> outputs/migration_v1/phaseB-full/grid_selection_audit/
#      candidates.jsonl, summary.csv, checks.json, conclusion.md, figures/

# Phase C: physics acceptance (all checks + the S1 feasibility witness)
python experiments/continuous2d/run_physics_acceptance.py --out outputs/migration_v1/physics_check

# Phase D: random control + depth-1/depth-3 exact-model MPC on the ten frozen scenes
python experiments/continuous2d/run_baseline.py --quick          # S0_dev + S1_dev smoke
python experiments/continuous2d/run_baseline.py --run-id phaseD-full
#   -> outputs/migration_v1/phaseD-full/continuous2d/
#      results.csv, summary.json, windows.jsonl, episodes.jsonl, media/ (frames + GIF)
python experiments/continuous2d/make_report_table.py --run outputs/migration_v1/phaseD-full/continuous2d
```

Trajectory replay is the `media/` GIF plus the per-step frames for the same episode; the
raw per-control-step states and controls are in `windows.jsonl` and can be replayed in
process with `Continuous2DEnv.restore` + `execute_joint_skill`.

### Protocol-fix round

See [`docs/protocol_fix_v1/report.md`](docs/protocol_fix_v1/report.md); results live under
`outputs/protocol_fix_v1/<run_id>/`.

```bash
pytest tests/test_protocol_fix.py tests/test_continuous2d.py -q
python experiments/continuous2d/run_physics_acceptance.py --out outputs/protocol_fix_v1/<run_id>/physics_check
python experiments/continuous2d/run_baseline.py --run-id <run_id> --out-root outputs/protocol_fix_v1
python experiments/continuous2d/compare_baselines.py \
  --after outputs/protocol_fix_v1/<run_id>/continuous2d \
  --out outputs/protocol_fix_v1/<run_id>/comparison
python experiments/grid_selection_audit/write_errata.py
```

### Deterministic grid control experiment

Independent 6x6 fully observed grid; contract in
[`docs/deterministic_grid_v1/CONTRACT.md`](docs/deterministic_grid_v1/CONTRACT.md), report in
[`docs/deterministic_grid_v1/report.md`](docs/deterministic_grid_v1/report.md).

```bash
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel, if not already present

pytest tests/test_deterministic_grid.py -q
python experiments/deterministic_grid/run_env_checks.py --run-id <run_id>
python experiments/deterministic_grid/run_dataset.py   --run-id <run_id>
python experiments/deterministic_grid/run_training.py  --run-id <run_id>
python experiments/deterministic_grid/run_diagnostic.py --run-id <run_id>
```

## Option small-scene testbench (reusable)

A bounded option-level diagnosis harness: 8 frozen development snapshots, a pre-registered
request menu, hard per-branch/per-command wall clocks, and a **local log-driven HTML replay**
(no server, no external service). It exists to make **one attributable controller change per
round** and compare it against the frozen baseline. Full contract:
[`experiments/option_testbench/README.md`](experiments/option_testbench/README.md).

- code: [`experiments/option_testbench/`](experiments/option_testbench/) — `panel.py` (fixtures,
  contracts, request menu), `runner.py` (one branch = one isolated world), `metrics.py`
  (three metric layers, coverage/fulfilment), `renderer.py` + `replay_template.html` (HTML
  replay and keyframes), `cli.py` (`prepare / run / render / summarize`), `candidate_push.py`
  (versioned candidate controller).
- rounds: [`round_000`](outputs/option_testbench_v1/round_000/) (baseline),
  [`round_001`](outputs/option_testbench_v1/round_001/) (`segment-contact-v1`),
  [`round_002`](outputs/option_testbench_v1/round_002/) (authorized multi-step transfer),
  [`round_003`](outputs/option_testbench_v1/round_003/) (metric calibration v1 -> v2; no
  controller change), `round_004`/`round_005` (oriented-model diagnostics). Current metric
  version: `option-metric-v2`.
- RL contract v4 (single maintained model, 2026-10-19): body-fixed lidar clipped at the range
  with a hit flag and category one-hot (756-dim observation), formal heading model
  (skill-declared `desired_heading`, braking/avoidance ungated), and an option scale dimension
  `{2,6,12,24}` s with window = budget (48 actions/robot, 2304 joint). Old 324-dim RL results are
  formally invalidated. See
  [`docs/handoff/20261019_RL_Contract_v4_Design.md`](docs/handoff/20261019_RL_Contract_v4_Design.md).
- oriented-model prototype (A): `experiments/option_testbench/oriented_prototype.py` and
  [`docs/handoff/20261019_Oriented_Model_Prototype_Note.md`](docs/handoff/20261019_Oriented_Model_Prototype_Note.md)
  (kinematic heading + stable pushing side; `EnvConfig(oriented=True)`, default off).
- handoff and design inputs:
  [`docs/handoff/20261019_Option_Testbench_Handoff.md`](docs/handoff/20261019_Option_Testbench_Handoff.md).

```bash
TB=experiments/option_testbench
OUT=outputs/option_testbench_v1/round_000

# 1. freeze the panel, contracts and request menu (first time only)
timeout --signal=TERM --kill-after=2s 60s .venv/bin/python $TB/panel.py --out $OUT
# 2. run the pre-registered branches
timeout --signal=TERM --kill-after=2s 60s .venv/bin/python $TB/cli.py run --out $OUT \
  --branch-wall-seconds 10 --total-wall-seconds 55
# 3. independent metrics (no simulation)
timeout --signal=TERM --kill-after=2s 30s .venv/bin/python $TB/metrics.py --out $OUT
# 4. local HTML replay + keyframes (no simulation)
timeout --signal=TERM --kill-after=2s 30s .venv/bin/python $TB/cli.py render --out $OUT
# 5. directed tests
timeout --signal=TERM --kill-after=2s 30s .venv/bin/python -m pytest tests/test_option_testbench.py -q
```

Open `outputs/option_testbench_v1/round_000/index.html` directly in a browser: all resources are
local and the per-branch frame data is loaded with static `<script src>` tags, so no server is
needed. Use the left column to pick a snapshot and branch, the **Compare** selector for a
baseline/candidate side-by-side on the same physical timeline, and the layer toggles for lidar
(no-return rays are drawn at the 1.20 m range), fitted patches, contacts, normal/tangent,
reference velocity, force, trajectory and truth.

**Budgets (hard).** one branch <= 10 s wall; one batch command <= 60 s; one render/test command
<= 30 s; one round of simulation+test+render <= 180 s. `metrics`/`render`/`summarize` never
re-simulate. The panel and request menu are frozen before running; changing them bumps the panel
version and keeps old results. A user-authorized exception for a specific measurement is recorded
in that round's `manifest.json` (`authorized_exception`), e.g. round_002's <= 8 skills / <= 45 s.

**Adding a round.** Freeze a new `requests.json` (<= 24 branches, <= 3 per snapshot); optionally
inject a candidate controller by `controller_version` (the runner records the version and source
hash). Candidate controllers live in their own module and are injected through
`RandomStartEnv(instance, config, skill_factory=...)`; the default factory is the frozen baseline,
so existing behaviour and checkpoints are untouched. Then run, `metrics.py`, `cli.py render`, and
write `REVIEW.md` with the change, the per-case `baseline -> candidate` comparison, side effects,
and one concrete feedback question.

## Read next

- protocol contract: [`docs/2d_protocol_spec.md`](docs/2d_protocol_spec.md)
- results and interpretation: [`results.md`](results.md)
- what to do next: [`next_steps.md`](next_steps.md)
- working rules for agents: [`AGENTS.md`](AGENTS.md)
