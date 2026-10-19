# option_testbench — bounded small-scene testbench (Round 000)

Scope: option-level diagnosis on 8 frozen development snapshots with a pre-registered
baseline menu. No PPO, no world model, no network scaling, no full-suite regression, no
90 s task. This directory only builds the testbench and runs the frozen baseline.

## Layout

```
panel.py      8 fixtures, dynamic recovery snapshot, contracts, request menu, prepare()
runner.py     one branch = one isolated world with an internal monotonic deadline
metrics.py    three metric layers recomputed from the raw logs, coverage and fulfilment
renderer.py   local HTML replay + initial/transition/exit keyframes (no external service)
cli.py        prepare / run / render / summarize
```

Output round directory (`outputs/option_testbench_v1/round_000/`):

```
manifest.json  panel.json  contracts/  requests.json  logs/  metrics.json
METRICS.md  index.html  data/  keyframes/  REVIEW.md  ROUND_PLAN.md  snapshots/
```

## Hard budgets and how they are enforced

| Work | Wall cap | On reaching it |
|---|---:|---|
| one snapshot × one branch | 10 s | stop the branch, mark `wall_timeout`, keep the partial log |
| one batch simulation command | 60 s | keep completed items, mark the rest `batch_budget_exhausted` |
| one render/HTML/keyframe command | 30 s | fall back to log-driven keyframes |
| one directed test command | 30 s | report the unfinished items, do not run the full suite |
| one round of simulation+test+render | 180 s | stop computation and report the budget state |
| first build+delivery | 90 min | save what works and hand back to the user |

Every simulation command is run as `timeout --signal=TERM --kill-after=2s <cap> ...` and the
runner also owns an internal monotonic deadline. The driver kills the process group with
`SIGTERM` and then `SIGKILL`; it never uses `pkill -f` and never touches other processes.

## Commands (copy-paste)

```bash
cd /home/tzl/repo/coimagine
TB=experiments/option_testbench
OUT=outputs/option_testbench_v1/round_000

# 1. freeze panel, contracts and request menu (first time only)
timeout --signal=TERM --kill-after=2s 60s \
  .venv/bin/python $TB/panel.py --out $OUT

# 2. run the 24 pre-registered branches
timeout --signal=TERM --kill-after=2s 60s \
  .venv/bin/python $TB/cli.py run --out $OUT \
  --branch-wall-seconds 10 --total-wall-seconds 55

# 3. independent metrics (no simulation)
timeout --signal=TERM --kill-after=2s 30s \
  .venv/bin/python $TB/metrics.py --out $OUT

# 4. local HTML replay + keyframes (no simulation)
timeout --signal=TERM --kill-after=2s 30s \
  .venv/bin/python $TB/cli.py render --out $OUT

# 5. directed tests
timeout --signal=TERM --kill-after=2s 30s \
  .venv/bin/python -m pytest tests/test_option_testbench.py -q
```

Open `outputs/option_testbench_v1/round_000/index.html` directly in a browser (all resources
are local; the page loads per-branch frame files with `<script src>`, so no server is needed).

## Contracts and metrics

`contracts/contracts.json` freezes the three layers: controller claim, independently
recomputed contract, and offline task effect. A `push_complete`/`reposition_complete` claim
whose recomputed condition fails is a `violated`, not a fulfilment. Safety and blocked exits
are reported separately. Coverage reports the numerator/denominator, the `accepted=0` case is
`N/A` (never 100%), and `unresolved` (wall timeout, missing log, horizon) is a separate count.

## Metric version (v2, Round 003)

The current metric/contract version is **`option-metric-v2` / `option-contract-v2`**. v2 fixes
six calibration issues in v1 and is recomputed from the existing logs (no simulation):
PUSH fulfilment split into `contact_established` (>=0.50 s) and `full_push` (>=2.00 s,
`PUSH_ACTIVE_SECONDS`); per-robot truncation at each robot's own skill-end period; REPOSITION
positive accumulated progress in the requested direction (no `abs`); teammate gap subtracts only
this robot's radius; normal-jump count recomputed from the `normal_world` sequence; conditional
fulfilment numerator restricted to accepted requests. v1 outputs (`metrics.json`, `METRICS.md`)
are preserved; v2 is written as `metrics_v2.json` / `METRICS_v2.md`, and the replay prefers v2.
The calibration comparison is `outputs/option_testbench_v1/round_003/CALIBRATION.md`.

## Oriented-model prototype (A)

`oriented_prototype.py` runs a bounded comparison between the frozen v3.1 model and a
**kinematic heading prototype** (the robot must turn towards the push direction; a misaligned
push is scaled by `cos(error)`). The prototype is enabled by `EnvConfig(oriented=True)`
(default off: v3.1 behaviour is byte-identical) and does not change the object physics, reward,
success rule or safety gate. It exists to answer whether requiring alignment changes the option
conclusions; see `outputs/option_testbench_v1/oriented_prototype/PROTOTYPE.md` and the handoff
note `docs/handoff/20261019_Oriented_Model_Prototype_Note.md`.

```bash
timeout --signal=TERM --kill-after=2s 120s .venv/bin/python \
  experiments/option_testbench/oriented_prototype.py
```

## Later rounds

The taskbook stop point is after Round 000 delivery: review the replay and answer the
feedback question before any controller change. Later rounds write `ROUND_PLAN.md` before
running, use explicit controller versions injected from a factory (no implicit global
monkeypatch), keep the original baseline forever, and re-run at least two unchanged
regression cases. The data schema already supports baseline/candidate side-by-side playback.
