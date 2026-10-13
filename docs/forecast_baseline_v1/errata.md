# ERRATA — forecast_baseline_v1

Corrections to statements made in earlier rounds. No old table or old run directory is
overwritten or deleted; every correction below points at the run that supports it. Phase A
details are in [A_grid_audit_v2.md](A_grid_audit_v2.md); continuous numbers are in
`outputs/forecast_baseline_v1/forecast-20261012/`.

## E1 — Grid grouping names and the 0.53 metric

| earlier statement | corrected statement | evidence |
|---|---|---|
| "moved sample accuracy" / "物体推动准确率" | `object_moved` = the object **anchor moved**; it says nothing about a push succeeding | `grid-audit-v2/group_stats.json` |
| "no-op sample" | `object_stationary` = the object anchor did **not** move, the robots may still move; the separate `full_state_unchanged` group means the whole state is unchanged | same |
| "0.53 is the object-push accuracy" | 0.53 is the **full joint-state accuracy on `object_moved` samples** (old field `moved_full_state_accuracy`, recomputed mean 0.5358 over the three seeds) | same |

Corrected test-split numbers (3-seed mean): `object_moved` n=270 full-state 0.536, invalid
0.086; `object_stationary` n=2820 full-state 0.892, invalid 0.047; `full_state_unchanged`
n=1322 full-state 0.858, invalid 0.081; all n=3090 full-state 0.861, invalid 0.050.
`full_state_unchanged ⊆ object_stationary` holds.

## E2 — the old 20 evaluation starts were coordinate-biased

The old starts came from a coordinate-string sort and put **every robot0 at (0,0)**. The new
validation set orders test states by `SHA256("grid-audit-v2|" + state_key)`, 10 per map, and
uses neither coordinates nor model error. Coverage improves from 1 distinct robot0 cell to
16 (robot1: 14 → 18). The old set and its results are kept and reported side by side.

## E3 — `selection_excess = 2.700` is not all "optimism of the selected plan"

`candidate_mean_bias` contains the illegal-prediction penalty term (old data: mean −1.782
over the H=4 rows). The four-term decomposition (`10ΔS, Δsteps, 0.5Δdistance, 10Δinvalid`)
sums to the total with max error 0.0 over 106,740 checks. On the new set at H=4/K=256 the
candidate-mean deviation splits into success −0.002, steps +0.387, distance −0.004,
invalid −2.101 (total −1.719); the **chosen** candidate's invalid penalty is 0 at every
H/K/seed. So `selection_excess` mixes the pool's illegal-prediction penalty with the
chosen plan's own error and must not be read as pure selective optimism.

The old headline "+1.250 / −1.450 / +2.700" is the **legacy row-mean** aggregation. Under the
aggregation fixed in this contract (mean within a state over candidate seeds, then mean over
states) the same old data gives **+0.938 / −1.088 / +2.025**. Both are listed; neither
replaces the other.

## E4 — the old recursive Grid curve used a different protocol

The old recursive diagnostic kept the previous legal state when a prediction was illegal;
the main diagnostic terminates immediately. The two curves are therefore not comparable and
are no longer used to explain the main planning loop. The new A4 contrast uses the main
diagnostic's rule (illegal ⇒ terminate, success ⇒ absorb, never fabricate the next state)
and reports surviving-chain and all-chain denominators separately. The old fallback curve
stays in its old run directory as a historical attachment.

## E5 — the new continuous scalar score is a new version, not a re-scoring of the old runs

`J = 10*S − t/24 − d/L − 20*F` (`forecast-scalar-v1`) replaces neither the old
lexicographic MPC key nor the old headline. **The new exact closed-loop success counts must
not be merged with the old lexicographic `random 1/30 / depth1 5/10 / depth3 9/10` table.**
The two use different objectives, different candidate pools and different coverage, and the
new exact runs were re-executed from scratch.

## E6 — the learned window model is not a perfect Markov window

`execute_joint_skill` resets the controller-internal state each window
(`continuous2d/environment.py:294`), so it is not a model input. What does persist inside
the engine is the contact/warm-start cache of the pymunk space, which the learned model
cannot see. The learned transition is an **approximate state model**; prediction error that
comes from that missing contact state is part of the measurement, not hidden.

## E7 — predicted "success" is a model output, never a task outcome

The model's termination head is scored with the real success criterion only as a label. A
predicted success is absorbed inside the learned evaluation but never marks a live episode
as successful: phase D still executes the chosen first window and lets the true
`object_in_goal && object_slow` hold test decide.

## E8 — the candidate true-failure rate is not evidence of a broken environment

Across the whole phase-C pool the **true** candidate failure rate is 0.000 at every H and
candidate seed; all failure penalties in the model metrics come from the model's own
`invalid_prediction`, not from the simulator. Read every negative `selected_bias` together
with `candidate_invalid_rate` (0.63–0.94).
