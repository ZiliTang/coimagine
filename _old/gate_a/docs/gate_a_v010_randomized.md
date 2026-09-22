# Gate A v0.10: two-sided randomized channel benchmark

Version 0.10 extends the heterogeneous Gate-A channel suite while retaining the
controlled causal families for mechanism isolation.

## Randomized scene distribution

Each scene samples:

- map width, height, and horizontal channel location;
- 2--4 movable blockers with variable footprint, task weight, and one- or
  two-robot pushing requirement;
- north-only, south-only, or bidirectional clearing actions;
- same-side, split-side, and cross-side-required robot initial configurations;
- robot-private sensing radius;
- two-sided reposition anchors selected by stratified farthest-point sampling;
- static occluders and optional entity-activation dependencies.

The generator is unscreened: no planner result is used to accept or reject a scene.
The debug gallery distinguishes the full anchor bank from the anchors currently
admissible to R0 and R1.

## Episode and candidate budgets

A scene with M movable objects receives at least `3M` joint-option boundaries,
reflecting the worst-case Reposition--Approach--Push chain for every object.
Generated scenes may receive additional slack. `--random-channel-max-steps-cap`
can remove surplus slack, but it cannot reduce the effective budget below `3M`.

The default fixed candidate budget is 5 for matched and tractable exact H=3 tests.
Passing `--random-channel-candidate-budget 0` uses each scene's generated budget,
which is substantially more expensive.

## Recommended full run

```bash
python scripts/run_gate_a_horizon_study.py \
  --skip-controlled \
  --random-channel-per-difficulty 100 \
  --random-channel-candidate-budget 5 \
  --random-channel-profile main \
  --jobs 8 \
  --output outputs/gate_a/v010_randomized_full
```

The old `--random-channel-max-steps-cap 6` argument remains accepted, but the
actual budget is still at least `3M`.

## Smoke test

```bash
bash scripts/smoke_gate_a_v010.sh
```

This runs one scene per difficulty, exercises multiprocessing, statistics, CSV/JSON
export, publication plots, the debug gallery, and the render-only recovery path.

## Recovering from a plotting failure

Simulation tables are written before figures. After updating the source, regenerate
figures without rerunning episodes:

```bash
python scripts/run_gate_a_horizon_study.py \
  --render-only \
  --output outputs/gate_a/v09_randomized_full
```

The v0.10 plotting code clamps machine-scale negative asymmetric error bars before
calling Matplotlib, preventing the `yerr must not contain negative values` failure.
