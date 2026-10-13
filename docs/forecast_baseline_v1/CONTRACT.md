# CONTRACT — forecast_baseline_v1 (task book 20261012_2007, phases A–D)

Frozen before any formal run. Written from
`docs/tasks/20261012_2007_CoImagine_Grid诊断收尾与连续世界模型基线任务书.md`. Any deviation
needs a new version of this file; the code may not silently deviate.

Score, candidate and split versions introduced here are **new experiment versions**. They do
not replace, overwrite or re-label the old lexicographic continuous MPC results
(`outputs/protocol_fix_v1/`, `outputs/closeout_v1/`) or the old Grid diagnostic
(`outputs/deterministic_grid_v1/grid-20261012-194632/`). New scalar results may never be
merged into the old lexicographic 9/10 headline.

## 0. Frozen inputs and guards

- Continuous platform: `src/coimagine/continuous2d/`, unmodified. Scene set, physics
  parameters, controllers, the 2 s / 40-control-step window, success criterion
  (`object_in_goal` + `object_slow` for `success_hold_time`), failure criteria and
  `restore_exact` prefix replay are unchanged. The three `closeout_v2` scenes
  (S0_dev, S0_v2, S0_v4) are used as fixed.
- Grid: `src/coimagine/deterministic_grid/`, three saved checkpoints, saved
  `samples.jsonl`/`splits.jsonl`, candidate generator and `J` are unchanged. Phase A only
  re-reads and re-aggregates; it never retrains and never overwrites an old table.
- Every checkpoint and data file hash is recorded at the start and at the end of each phase
  (see `manifest.json`). A difference is a BLOCKED condition, not a footnote.

---

## 1. Phase A — finite Grid close-out

New code: `src/coimagine/deterministic_grid/audit.py`,
`src/coimagine/diagnostics/grid_audit_v2.py`,
`experiments/deterministic_grid/run_audit_v2.py`. Old files stay untouched.

### A1 naming and group statistics

The old grouping names `moved` / `no-op` are retired. New group names:

- `object_moved` — the true object anchor changed this step.
- `object_stationary` — the true object anchor did not change (the robots may still move).
- `full_state_unchanged` — the whole true next state equals the true current state.

`full_state_unchanged` is a separate group and is never merged with `object_stationary`.
Per group, on **both** maps and all three model seeds: object next-anchor accuracy, robot0
and robot1 next-cell accuracy, full-state accuracy, invalid-prediction rate, sample count.
Grouping is fixed by the true transition, never by the model output.

Data are read from the saved `dataset/samples.jsonl` and `dataset/splits.jsonl` of the
existing run directory. The current generator is not used to rebuild old data.

### A2 coordinate-unbiased validation set

New 20 legal non-terminal `test` states, 10 per map:

- from the saved `splits.jsonl`, take the distinct `test` state keys per map;
- drop states that are goals; order by `SHA256("grid-audit-v2|" + state_key)`;
- take the first 10 per map; never order by coordinate, model error, return or success.

Report robot0/robot1 cell coverage of the selected set, full states, split, selection rule
and the SHA256s. Old 20 states and their results are kept unchanged and reported separately.

Model/diagnostic grid for both sets: `H={1,2,4}`, `K={16,64,256}`, candidate seeds `{0,1,2}`,
model seeds `{0,1,2}`. Actual distinct candidate counts are reported; `H=1` caps at 81.

Table columns: `selected_bias`, `candidate_mean_bias`, `selection_excess`, `regret`,
`chosen_true`, `uniform_candidate_true`, `candidate_best_true`, chosen success rate,
candidate invalid-prediction rate. Aggregation is: mean within one true initial state over
candidate seeds, then mean/range over states. Model seeds and candidate seeds are **not**
independent scenes; no significance claim is made.

### A3 return-error decomposition and error tracing

Split `J_hat - J_ref` into success reward, executed steps, terminal distance and invalid
penalty, for the chosen candidate and for the candidate mean. Algebraic check: the four
parts must sum to the total deviation.

On the new validation set at `H=4, K=256`:

- average regret per state over model seeds, take the 3 distinct states with the largest
  mean regret;
- within each, take the run with the largest regret, ties by a stable id;
- add the 1 state with the smallest mean regret as a control;
- print the chosen candidate and the exact-best candidate step by step: true states,
  predicted states, actions, score components.

Record the first full-state mismatch, the first object-anchor mismatch and the first false
success, and state which of them actually changed the candidate ranking. "State differs but
no ranking evidence" is written as **未建立直接联系**. One case may have several errors.

### A4 recursive contrast under the planning protocol

Same invalid-prediction termination and success absorption as the main diagnostic
(`diagnostic.rollout` + `score_trajectory`): once a model step is illegal, or the goal is
reached, the chain stops and no next state is fabricated. Teacher-forced starts from the
true state each step; recursive starts from the model's own last legal predicted state.
Both run over the same fixed true trajectory set. Per step report:

1. cumulative termination rate and invalid rate up to that step;
2. state error on the surviving chains and its denominator;
3. cumulative failure/error ratio over **all** chains, so survivors are not the only
   comparison.

Termination-decision errors and physical-state errors are reported separately. The old
fallback curve stays in the old run directory as a historical attachment.

After A the Grid line is frozen, the main text is under 500 Chinese characters, and B
proceeds regardless of whether the over-estimation survives.

---

## 2. Phase B — single-scale continuous world model

New package: `src/coimagine/continuous2d_learning/`.

### B1 interface and state contract

One transition = one synchronous joint-skill window of at most `window_steps = 40` control
steps / 2.0 s. A robot that finishes its skill early is switched to an explicit `hold` and
the window does **not** end early. Real success or a physical failure may end the window
early.

**Controller reset.** `Continuous2DEnv.execute_joint_skill` sets
`self.controller_state = [{}, {}]` at its start
(`src/coimagine/continuous2d/environment.py:294`). Controller internal state (`path`,
`anchor`, `mode`) therefore does **not** cross a window boundary, and is not a model input.
What does cross the boundary inside the engine is the contact/warm-start cache of the
pymunk space; `restore_exact` rebuilds it by prefix replay and the learned model does not
see it. The learned transition is therefore an **approximate state model**, not a claim of a
perfect Markov window; this is recorded, not silently ignored.

Model input (fixed layout, 57 floats, all deterministic normalisation constants below):

| block | dims |
|---|---:|
| object `x/W, y/H, sin θ, cos θ, vx/Sv, vy/Sv, ω/Sω` | 7 |
| robot0 `x/W, y/H, sin θ, cos θ, vx/Sv, vy/Sv, ω/Sω` | 7 |
| robot1 same | 7 |
| `success_hold / HOLD_SCALE` | 1 |
| goal `cx/W, cy/H, half_w/W, half_h/H` | 4 |
| arena `W, H` | 2 |
| 2 obstacle slots × `cx/W, cy/H, half_w/W, half_h/H` + 2 presence flags | 10 |
| object size `width/W, height/H` | 2 |
| window max duration `/ 2.0` | 1 |
| skill A: kind one-hot(3) + `move_to` target `(x/W, y/H)`(2) / `push_contact` `contact_local`(2, /object half extents), `push_direction_world`(2), `force_limit/max_force`(1) | 8 |
| skill B same | 8 |

Constants: `Sv = 2.0 m/s`, `Sω = 2.0 rad/s`, `HOLD_SCALE = 0.55 s`. The skill kind one-hot
plus its full-precision parameters are used; `candidate_id` and formatted labels are never
inputs.

Model output (21 values + 3 logits):

- three bodies × `Δx/W, Δy/H, wrap(Δθ)/π, Δvx/Sv, Δvy/Sv, Δω/Sω` — 18 values, positions
  added to the current state, `Δθ` wrapped;
- `success_hold` absolute, bounded: `HOLD_SCALE * sigmoid(raw)`;
- `duration` absolute, bounded: `2.0 * sigmoid(raw)`;
- termination logits over `window_complete / success / physical_failure`.

`physical_failure` merges `numeric_failure`, `out_of_bounds`, `penetration_limit`. Raw data
keep their original categories. `success` comes from the real `object_in_goal &&
object_slow` hold criterion, never from "the object centre ended inside the goal".

**Prediction hygiene.** Recursive prediction uses only predicted states. Predicting an
intermediate state may not call exact dynamics, `restore`/`step_control`, real future
contacts or the real future `success_counter`. Geometry (bounds, penetration, obstacle
distance, `passage_cost_to_go`) may be computed from predicted states and known static
geometry. Real future control forces are never inputs: what the skill's internal controller
will produce is part of the transition being predicted.

**Invalid prediction.** Non-finite output, out of the arena by more than
`out_of_bounds_margin = 0.5`, or a geometric overlap deeper than
`penetration_limit = 0.05 m` marks `invalid_prediction` and the chain is absorbed. There is
no projection to the nearest legal state, no fallback to the true state, no oracle repair.
Raw geometric-violation rate (violating transitions / attempted transitions) is reported
separately from the absorbed invalid rate.

### B2 model and training (one configuration, no search)

Plain PyTorch joint-state MLP: 2 hidden layers × 256, ReLU. Loss = normalised-state MSE +
`success_hold` MSE + duration MSE + termination cross-entropy, each coefficient 1. Adam,
`lr = 1e-3`, `batch = 128`, at most 100 epochs, stop when validation loss has not improved
for 12 consecutive epochs. `seeds = {0,1,2}`; the checkpoint is chosen by validation loss
only. No learning-rate, width, loss-weight or data-weighting search; no checkpoint chosen by
test or planning success. Normalisation constants are the fixed ones above (arena-relative
positions, `sin/cos` angles, fixed velocity scales); no statistic is fitted at all, so there
is no train/test leakage.

The termination decision rule is fixed **before** training: `argmax` of the 3-class head.

### B3 data collection and split

Existing repaired ten scenes, no new geometry variants, no perturbed initial states:

| split | scenes | episodes per scene |
|---|---|---:|
| train | S0_dev, S0_v1, S0_v2, S1_dev, S1_v1, S1_v2 | 20 |
| val | S0_v3, S1_v3 | 20 |
| test | S0_v4, S1_v4 | 20 |

200 episodes, at most 12 windows each, hard cap 2400 windows. Split is **by whole scene**;
no episode fragment crosses a split. This is a held-out scene-variant split inside the two
existing task families, not a claim of generalisation to a new task.

Collection behaviour, seeded by `SHA256(scene_name + "|" + episode_id)`:

- window 0: uniform over the current `joint_candidates`;
- each later window: with probability 0.5 uniform over the current `joint_candidates`,
  otherwise the previous joint skill's structured parameters repeated verbatim;
- actions are never filtered by predicted or exact return;
- success or failure ends the episode immediately; no collection continues from a terminal
  state;
- no existing exact-MPC trajectory is mixed in.

Records follow schema v2 (structured full-precision skills, window before/after states,
duration, termination, controller version, scene hash) plus `success_counter_before` and
`success_counter_after`, so the initial and final counters are both present. Full per-step
records are kept for a review subset of episodes; the rest are window-level records that
still carry the reproduction seed and config.

Before training, report split sample counts and coverage of: object translation, object
rotation, single push / double push / move / hold combinations, contact, success and
failure. Grouping thresholds are frozen here: translation `> 0.01 m`, wrapped rotation
magnitude `> 0.02 rad`. They are used only to group evaluation; they do not change any skill
or task event. If a group is too small it is written as **该分组证据不足** and the ordinary
baseline is still trained; sampling and behaviour are not changed.

### B4 model validation

On test: object and both robots' position error, velocity error, object angle error; per
`object_translation` / `object_rotation` / `stationary` group sample count and error;
termination confusion matrix, false-success rate and its denominator; hold-timer error,
duration error, raw invalid rate; comparison against a **state persistence** baseline
(physical state unchanged, unterminated window continues, `duration = 2 s`). Persistence is
a prediction reference only, never presented as exact dynamics.

There is no unified accuracy pass threshold. Bad accuracy is a result. Only interface
misalignment, train/test leakage, NaN, or an output that does not depend on the input counts
as an implementation defect to fix first.

---

## 3. Phase C — fixed-candidate ranking diagnostic

### C1 scalar score (new experiment version)

```
J = 10*S - t/24 - d/L - 20*F
```

- `S`: true trajectory success, or the model's predicted `success` termination; success is
  absorbing.
- `t`: cumulative executed time in seconds from the planning start; no time accrues after
  success or failure.
- `L`: arena diagonal length (`hypot(arena_w, arena_h)`); per scene.
- `d`: the existing `passage_cost_to_go` at the current object pose with object size and
  static geometry; `d = 0` on success. The heuristic content of `passage_cost_to_go` is
  disclosed; plain `corner_distance_to_goal` (`distance_to_box`) is reported separately as
  an independent indicator. This score is **not** described as a model that learned the
  value of preparation.
- `F`: physical failure or `invalid_prediction`; failure is absorbing.
- No control-force, direction or skill-kind reward is added.

Terminal state used for `d` when a chain is absorbed by failure: the state in which the
chain stopped. For a physical failure that is the engine's terminal state; for an
`invalid_prediction` it is the **raw predicted state that violated geometry** (its object
pose is the model's hallucinated progress, which is exactly what is being measured). No
projection, no fallback to the true state, and no fallback to the last legal state is
applied. Only a non-finite predicted pose, for which no geometry can be evaluated, falls
back to the last finite state and is counted separately. Success always sets `d = 0`.

Feasibility under this scalar protocol is deliberately minimal and identical for the exact
and learned sides: a candidate is *scorable* when its terminal state exists and is finite.
`planner_no_feasible_plan` is returned only when the pool is empty or **every** candidate
ends non-finite. A physical failure or an invalid prediction is not a rejection; it is a
`-20` penalty, so a failing candidate is selected only when it is the best of a bad pool.
This keeps the two sides on one function and avoids a second, hidden feasibility rule.

Exact and learned evaluation call the same `score_components → J` function. Ties break on
the smaller stable `candidate_id`. Components, cumulative duration and termination class are
stored raw. Success is never replaced by "the final pose is inside the goal", and logits
probabilities are never used as the success reward.

Hand-worked tests: already-successful initial state, success absorption, failure absorption,
time counted once, `J_hat = J_ref ⇒ regret = 0`.

### C2 evaluation starts and frozen candidates

12 starts, 6 per test scene (S0_v4, S1_v4):

- one is the scene's legal non-terminal initial state;
- the other 5 come from test-split non-terminal window starts ordered by
  `SHA256("continuous-audit-v1|" + episode_id + "|" + window_index)`;
- identical physical state + success timer are excluded; selection never uses error,
  distance or return;
- the recoverable episode prefix, full snapshot, state and hash are saved.

If fewer than 12 are available the actual count is reported; train data is not used to fill.
`H={1,2,3}`, `K={16,64,128}`, candidate seeds `{0,1,2}`. The largest pool is generated once
per (start, candidate seed); smaller `K` are prefixes of that same pool. The full pool is
saved before evaluation.

A candidate is a fixed full-precision joint-skill sequence:

- root library = existing `joint_candidates` from the start's public observation;
- `H=1`: root candidates deduplicated, shuffled with the seed; actual count may be 121;
- `H>1`: sample up to 32 joint skills from the root library without replacement with the
  seed and repeat each `H` times; plus up to 32 from the generic template "both sides
  `move_to` a staging point for one window, then `push_contact`", whose staging points and
  pushes come from the existing root library; then fill with uniform samples from the root
  library up to 128 sequences; deduplicate and stably shuffle. Template shortfall is filled
  the same way, so the pool is not dominated by one repeated skill;
- later-window world coordinates are stored verbatim and never regenerated from model or
  true future states.

Exact-MPC witnesses, exact optimal sequences, model-picked sequences and scene-id-specific
success actions are excluded. The limits of this fixed sequence class are stated in the
report.

### C3 exact reference and metrics

Each fixed sequence is executed from the same true snapshot on an independent
`planner_env` using `restore_exact` prefix replay. Each candidate's truth is computed once
and shared by the three models. Future leakage and live-env pollution are forbidden.

Candidate coverage is reported before the model:

- `candidate_best_true_return`, `uniform_candidate_true_return`;
- fraction of starts with at least one successful candidate;
- fraction of starts with at least one feasible candidate whose `d` is lower than the
  start's `d`;
- candidate true failure rate and actual deduplicated count.

Model metrics: `selected_bias`, `candidate_mean_bias`, `selection_excess`,
`candidate_regret`, `chosen_true_return`, `chosen_true_success`, chosen candidate's
false-success rate, predicted failure / invalid rate. Aggregated by start; the three model
seeds are listed separately with their mean. Repeated measurements are not new scenes. Raw
returns at different `H` include different execution time and are not used alone to claim
"deeper is worse". The component decomposition must contain success reward, time, distance
and failure penalty, listed for both the chosen candidate and the candidate mean.

At least the 3 largest mean-regret starts and 1 low-regret control are shown step by step,
with predicted and true trajectories side by side, not only a final GIF.

Coverage verdict, fixed in advance: if **fewer than 3 of the 12 starts** have a feasible
candidate that lowers `d`, phase C is marked `COVERAGE_LIMITED`; the existing diagnostic is
finished, phase D is not run, and no remedial training is started. A low candidate success
rate still allows progress-ranking statements but not a full task-capability claim.
`COVERAGE_LIMITED` does not mean the over-estimation mechanism is absent, and does not mean
the continuous platform is unusable.

### C4 recursive contrast

The same fixed sequences as C, comparing teacher-forced and recursive per-window state
error, with the A4 rules for termination, invalidity and denominators. Timing, termination
head and state error are reported separately; one error curve is not automatically the
explanation of all regret.

---

## 4. Phase D — closed loop under the same protocol

Only if the C coverage verdict passes. Every method re-plans from the current true
observation at each decision and executes only the first window of the chosen sequence.

1. `exact_fixed_pool` — the same fixed candidate pool, exact `J`.
2. `learned_fixed_pool` — each independent checkpoint recursively evaluates the same pool.
3. `uniform_root` — uniform draw of one window from the same root skill library.

`H={1,3}`, max `K=128`, at most 12 live windows. `H=1` actual candidate count is at most the
root deduplicated count. Each plan builds its pool only from the current true start; pool
generation seeds come from scene, window and episode seed and never from model predictions.

First formal closed loop: test scenes S0_v4 and S1_v4 only.

- exact: episode seeds `{0,1,2}`, 6 episodes per `H`, 12 episodes;
- learned: 3 checkpoints × episode seeds `{0,1,2}` × 2 scenes × 2 `H`, 36 episodes;
- random: 2 scenes × episode seeds `{0,1,2}`, 6 episodes.

Different checkpoints are not an ensemble. Two scenes are a small sample: this is a baseline
re-check, not a statistical generalisation or new-method claim.

Per episode record: true success, completion time, window count, geometric distance,
physical failure, `J`, actual prediction nodes, exact simulation steps, prefix replay steps
and wall clock. If the model rejects every candidate the episode ends as
`planner_no_feasible_plan`; no oracle substitute action is taken. If the model predicts
completion, the first window of the chosen sequence is still executed this round and the
real success test decides the end; a predicted success never marks the live episode as
success. The planner's exact environment is used only by the exact method and the
diagnostic reference; the learned method may not query the future. Live and planner
environments are separate for every method. New scalar exact results are re-run, never
inherited from the old lexicographic score.

---

## 5. Reporting rules

- Every number comes from a run; a missing run is written `NOT_RUN`.
- Each phase is labelled 实现完成 / 测试通过 / 实验已运行.
- Negative results are reported, not hidden: model accuracy may be good, the ranking may
  work, deep planning may not degrade. Nothing is made worse on purpose, and training is
  never reduced to manufacture a phenomenon.
- Out of scope this round: multi-scale prediction, cross-window constraints, new losses,
  model ensembles, distributed prediction fusion, communication, asynchronous skills, a
  third robot, multiple objects, manipulator simulation. Three training seeds are
  independent repeats, not an ensemble.
- Resource limits: A only recomputes diagnostics; B is at most 200 episodes / 2400 windows,
  one training configuration and three seeds; C is fixed at 12 starts × 3 `H` × 3 `K` × 3
  candidate seeds with nested `K` reusing truth; D is only the 54 episodes above.
