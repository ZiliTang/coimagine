# CoImagine

Executable research code for the controlled 2-D Gate-A study of
**CoImagine: Distributed Local-Belief Imagination of Evolving Coordination
Topologies**. The code runs on standard Python and does not require ROS, Gazebo,
or Isaac Sim.

## Version 0.9: heterogeneous randomized channel benchmark

Gate A now follows the paper's central information flow:

```text
private local observation
    -> local belief rollout
    -> future observation-supported options
    -> evolving coordination topology
    -> receding-horizon joint decision
```

The global grid state belongs only to the simulator. It is used to execute the
selected root option, render the next real local observations, and construct a
privileged successor-observation oracle. CST and CoImagine-IET never receive a
shared raw map.

### Main comparison

- **CST — Current-Observation-Supported Topology:** predicts successor-centered
  robot poses and known entity states, but every imagined future topology is
  restricted to option templates grounded by the current real observations.
- **CoImagine-IET — Imagined Evolving Topology:** uses an optimistic rule-based
  local model. Known entities and teammate option effects are propagated;
  unobstructed unknown space receives a free-space prior; predicted observations
  regenerate future options, factors, and topology.
- **Privileged IET Oracle:** starts from the same root information and action set,
  but uses the simulator's true successor local observations before regenerating
  the future topology.

Only the root joint option is executed. Every real option boundary discards the
imagined branch and updates each robot from its new real local observation.

## Canonical Gate-A mechanism

Two robots face a heavy two-cell blocker. Initially, both local observations
support only one contact. One robot can approach that contact while the other
uses a currently visible `Reposition` anchor. From the new viewpoint, IET
predicts a second contact and values the sequence

```text
Reposition/Approach -> Approach(new contact) -> coordinated Push
```

This dependency needs three option boundaries. Therefore, **H=3 is the primary
Gate-A configuration** and H=2 is retained as a matched horizon ablation.

The shared option library is

```text
Reposition · Approach · Push · Hold · Observe · Idle
```

`Push` templates may be generated before contact is established; current
executability is handled separately by the factor verifier.

## Setup

```bash
cd coimagine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
pytest
```

## Run the focused H=2/H=3 study

```bash
python scripts/run_gate_a_horizon_study.py \
  --open-per-family 100 --surprise-cases 100 \
  --random-channel-per-difficulty 50 \
  --random-channel-profile main --jobs 4 \
  --output outputs/gate_a/v09_randomized
```

This writes episode-level results, successor-option/topology diagnostics, the
single-blocker mechanism storyboard, a five-boundary multi-blocker channel-clearing
CST-versus-IET main figure with factor topology strips, and an H=2/H=3
completion/runtime figure.

## Run the integrated Gate-A pipeline

```bash
coimagine-gatea --profile smoke
coimagine-gatea --profile pilot
coimagine-gatea --profile pilot_full
coimagine-gatea --profile paper
```

A YAML path may replace the profile name.

```bash
coimagine-gatea --profile configs/gate_a_pilot.yaml \
  --output outputs/gate_a/my_pilot
```

## Main output layout

```text
outputs/gate_a/<run>/
├── data/
│   ├── episode_results.csv/json
│   ├── successor_support.csv/json
│   └── legacy diagnostic/scaling records
├── tables/main/
│   ├── horizon_summary.csv/md
│   └── support_summary.csv/md
├── figures/main/
│   ├── fig_gate_a_local_imagination_story.{pdf,svg,png}
│   ├── fig_gate_a_channel_imagination_story.{pdf,svg,png}
│   └── fig_gate_a_horizon_comparison.{pdf,svg,png}
└── manifest.json
```

Detailed assumptions and interpretation rules are documented in
[`docs/gate_a_v05_local_imagination.md`](docs/gate_a_v05_local_imagination.md).

## Gate A v0.10 randomized generalization study

The controlled causal strata remain available for mechanism isolation. The v0.10
suite adds heterogeneous two-sided channel tasks with north/south clearing,
same-side, split-side, and cross-side-required robot starts, and a spatially
dispersed two-sided reposition-anchor bank. Every M-object scene receives at least
3M option boundaries, so failure is not caused by an episode budget that is too
short by construction.

The debug gallery distinguishes the global anchor bank from anchors currently
admissible to each robot. Exact H=3 search remains expensive; the default matched
candidate budget is 5. Use `bash scripts/smoke_gate_a_v010.sh` before a large run.
Plotting can be recovered without rerunning episodes through `--render-only`.
See [`docs/gate_a_v010_randomized.md`](docs/gate_a_v010_randomized.md).



```python
def generate_randomized_channel_benchmark_case(
    seed: int,
    *,
    difficulty: str | None = None,
) -> ScenarioInstance:
    """Generate a heterogeneous, unscreened multi-blocker channel task.

    Unlike the controlled causal strata, this suite deliberately mixes multiple
    blockers, variable coalition requirements, occluders, sensing radii, anchor
    densities, initial robot poses, and episode budgets.  The generator never
    checks planner outcomes.  It is intended as a generalization/diagnostic suite
    whose completion rates need not collapse to the deterministic 0/100 patterns
    of the mechanism-isolation tasks.
    """
```

### Monitoring long Gate-A runs

`run_gate_a_horizon_study.py` displays live progress by default, including case and
episode counts, throughput, elapsed time, and ETA. Add `--no-progress` for silent
execution. When output is redirected to a file, `--progress-interval <seconds>`
controls the reporting interval.
