# Gate A v0.8: Family-Stratified Statistical Benchmark

Version 0.8 separates the causal mechanism study from the robustness study and
assigns an independent sample budget to every open task family.

## Benchmark splits

The default paper-scale run contains:

- 100 open `direct_push` instances;
- 100 open `approach_push` instances;
- 100 open `mixed_approach_push` instances;
- 100 open `reposition_approach_push` instances;
- 100 separate unseen-wall surprise instances.

Map size, blocker translation, horizontal reflection, setup distances, robot
identity assignment, and robot initial positions vary by seed. Instances are
generated before planning and are not screened by method outcome.

The surprise split is reported separately. It is not mixed into the headline
open-task completion rate because its purpose is to measure optimistic false
option/topology support rather than nominal task performance.

## Statistical outputs

The study exports:

- Wilson 95% confidence intervals for completion rates;
- medians and interquartile ranges for planning time, return, duration, and
  option count;
- paired completion comparisons on identical instances;
- exact McNemar tests for paired binary outcomes;
- bootstrap confidence intervals for paired completion-rate differences;
- episode-level CSV/JSON records;
- successor option/topology fidelity diagnostics;
- object-count and robot-count structural scaling curves for factor arities
  `P=2,3,4`.

## Run

Paper-scale run:

```bash
python scripts/run_gate_a_horizon_study.py \
  --open-per-family 100 \
  --surprise-cases 100 \
  --jobs 4 \
  --output outputs/gate_a/horizon_v08_stratified
```

Quick validation:

```bash
python scripts/run_gate_a_horizon_study.py \
  --quick --jobs 4 \
  --output outputs/gate_a/horizon_v08_quick
```

## Interpretation

The family-conditioned 0%/100% values are intentional causal diagnostics. They
show that the three-boundary future-support dependency is solved only when both
IET support expansion and `H=3` are present. The aggregate open benchmark is
non-binary because three current-support families remain solvable by all
methods.

The sandbox timing values are implementation checks rather than final hardware
benchmarks. Paper timing should be rerun on the declared evaluation CPU with
fixed process affinity and repeated trials.
