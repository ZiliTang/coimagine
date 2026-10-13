# CONTRACT — deterministic grid control experiment (v1)

Independent of the old Grid and of the continuous platform. No hidden parameters, no
beliefs, no learning encoders, no private feedback. Fixed scalar score shared by the exact
environment and the learned model. This file is the frozen contract; code must not deviate
without a new version.

## 1. State and maps

- 6x6 grid, `x, y` in `0..5`. Coordinates are integers.
- Two robots with fixed ids `0, 1`, each one cell.
- One rigid, non-rotatable 2x2 object identified by its anchor (top-left/min cell)
  `(ox, oy)` with `ox, oy` in `0..4` (25 legal anchors). It occupies
  `{(ox,oy), (ox+1,oy), (ox,oy+1), (ox+1,oy+1)}`.
- Static wall cells (fixed per map) and a 2x2 goal region, default anchor `(4,2)`
  (cells `(4,2),(5,2),(4,3),(5,3)`).
- Everything is fully observed. No noise, no hidden mass/slip/release parameters, no
  private feedback, no belief update, no learned encoder.
- `state_is_valid(state)`: object cells inside bounds and not on walls; both robots inside
  bounds, not on walls, not on object cells, and distinct.
- Initial states are valid and **not** already in the goal.
- `is_goal(state)`: all four object cells are inside the goal region.
- Success is **absorbing and immediate**: once `is_goal`, no further step is taken and no
  further cost accrues. There is no hold timer.

Two fixed maps only:

| map | walls | goal anchor |
|---|---|---|
| `empty` | none | (4,2) |
| `obstacle` | one static cell `(3,2)` | (4,2) |

The obstacle layout is fixed before any training and geometry-checked for a deterministic
witness path to the goal. Frozen witness initial state: robots `(0,0)` and `(0,1)`, object
anchor `(1,2)`; the shortest joint-action witness is 9 steps on `obstacle` and 4 steps on
`empty` (`outputs/deterministic_grid_v1/<run>/env_checks.json`). The witness is used only for environment acceptance and is never
injected into evaluation candidates or used to select a model.

## 2. Actions and synchronous dynamics

Per robot, 9 actions in this fixed order:

```
0 move_N  1 move_S  2 move_E  3 move_W
4 push_N  5 push_S  6 push_E  7 push_W
8 wait
```

A joint action is an ordered pair `(a0, a1)` with `a_i` in `0..8`. Joint id is
`a0 * 9 + a1` (robot 0 major), giving 81 ids in a fixed order.

`transition_exact(state, joint_action) -> (next_state, info)` applies one time step using
the start-of-step state throughout:

1. All actions are interpreted simultaneously from the start-of-step state.
2. `move_d`: proposed target is one cell along `d`. It is legal only if it is in bounds,
   not a wall, and not on the **start-of-step** object or the other robot.
3. `push_d`: legal only if `robot_pos + d` is an object cell and `robot_pos` itself is not
   an object cell (the robot is on the side opposite `d`). Otherwise it is an invalid no-op.
4. Only geometrically qualified pushers count. If qualified pushers request **different**
   directions, the object does not move and those pushers do not move; invalid pushers do
   not participate in the direction conflict.
5. If the qualified pushers' direction is unique, propose translating the object one cell
   along it and translating those pushers with it. One pusher suffices; two same-direction
   pushers push together with no extra artificial reward.
6. The object destination must be in bounds and not on a wall; otherwise the object and its
   pushers stay.
7. Legal move proposals and push proposals are merged. If the final occupancy conflicts,
   the step is conservatively rolled back: **all positions stay unchanged**. This is a
   simplification of this control environment, not an exact simultaneous collision solve.
8. Robots may not swap positions, may not pass through the other robot's start-of-step
   cell, and may not "move into place and push" in the same step.
9. Every no-op, `wait` and conflict rollback still consumes one time step. An invalid
   action never ends the episode.
10. Initial states and true transitions are always valid; the true dynamics never branch on
    a scene id.

## 3. Fixed scalar score

Higher is better:

```
J = 10 * success - executed_steps - 0.5 * terminal_distance - 10 * invalid_prediction
```

- `terminal_distance`: over the object's four cells, the **maximum** Manhattan distance to
  the nearest goal cell; 0 when in the goal.
- `success` is 0/1 for whether the evaluated deterministic trajectory actually entered the
  goal.
- After success, absorption: no further step, no further cost, terminal distance 0.
- Every actually attempted non-terminal action costs 1 step, including `wait` and no-ops.
- `invalid_prediction` is used **only** when a model's decoded output is outside the legal
  state space: it records one failure and stops; the distance is taken from the last legal
  state, and the step at which the failure occurred is still charged.
- The exact environment and the learned model call the **same** `score_trajectory` entry.
  The exact environment cannot produce an illegal state, but the learned side may not use a
  separate opaque scoring path.
- Report success, steps, distance and invalid penalty separately, not only total J.
- An initial state that is already in the goal exists only in unit tests and returns
  `J = 10`, 0 executed steps.

`invalid_prediction` is an explicit interface, not a method contribution. The model's
output is never repaired by querying the true successor, and an illegal output is never
silently projected onto a "closest correct" state.

## 4. Candidate protocol (diagnostic)

- Fixed evaluation starts: 20 non-terminal legal test states by stable id, 10 per map when
  available; not selected by model error or return. They must not appear in the training
  state groups (recorded).
- Depths `H = 1, 2, 4`, candidate-set sizes `K = 16, 64, 256`, candidate seeds `0, 1, 2`
  (independent of model seeds). Nested prefixes of one fixed shuffled order.
- H=1 has at most 81 distinct joint actions; H=2/4 at most 256 distinct sequences.
- Half of the sequences are random over all 81 joint actions; half come from fixed
  templates (single push with the other waiting, same-direction double push, move actions,
  wait). Dedup then shuffle; pad with independent random sequences if short.
- Candidate generation never calls the model or the true future return, never injects the
  acceptance witness, and never changes templates based on model results.
- The same fixed sequence is executed without re-selecting actions mid-sequence. No beam,
  no MPC, no feedback branching.
- Truth is computed once per (map, state, H, candidate seed) and reused by all model seeds.
  Same depth and candidate set are used for the exact and learned comparisons.

## 5. Metrics

`chosen = argmax J_hat`, ties by smaller `candidate_id`:

- `selected_bias = J_hat(chosen) - J_ref(chosen)`
- `candidate_mean_bias = mean(J_hat - J_ref)`
- `selection_excess = selected_bias - candidate_mean_bias`
- `candidate_regret = max J_ref(C_K) - J_ref(chosen)`
- `chosen_true_return = J_ref(chosen)`
- `uniform_candidate_return = mean J_ref(C_K)`
- `chosen_true_success`, and the candidate set's best true return/success
- chosen-sequence and all-candidate invalid-prediction rates

`uniform_candidate_return` is a uniform choice of a fixed sequence inside the same
candidate set, not another historical random policy. `regret`'s optimum is only the best
inside the candidate set.

## 6. Interpretation rules (pre-registered)

Three separate questions, three separate answers:
1. Is the model accurate, especially after pushes and repositioning?
2. Is the chosen candidate's error more optimistic than the average candidate's?
3. At fixed H, does increasing K reduce the realised chosen return?

Positive bias with return above random is written as such, not called planning failure.
Increasing return with K is written as such, not called "more search is worse". A model
that learns well and near-zero bias is a valid negative result. If the model is poor, check
single-step key transitions and the invalid rate before attributing anything to search. If
even the exact candidates make no progress, mark candidate coverage as insufficient. Test
metrics are descriptive mechanism evidence from 20 starts, not a universal law.
