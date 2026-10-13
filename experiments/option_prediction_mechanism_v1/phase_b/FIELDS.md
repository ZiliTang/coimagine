# Phase B field table — `option_prediction_mechanism_v1/phase_b`

Frozen by `docs/option_prediction_mechanism_v1/CONTRACT.md` section 2.2 and by the task book
section 3 (B2). One row of `executions_{train,val,test}.jsonl` is **one real joint option
execution**: the pre-execution information plus the joint skill command, and the terminal
information / labels measured at the actual exit. Duration is variable; it is never a fixed
2 s window.

## 0. Conventions

| item | convention |
|---|---|
| units | SI: metres, seconds, kilograms, newtons, radians, m/s, rad/s |
| world frame | arena coordinates, `+x` right, `+y` up, angles CCW from `+x` |
| object frame | origin at the object centre, `+x` along the 1.0 m width, `+y` along the 0.25 m height |
| robot *own* frame | origin at the robot centre, `+x` along the robot's current heading |
| robot *start* frame | origin at the robot's pose **at t = 0 of this execution**, `+x` along its heading at t = 0. A world point `p` maps to `R(-θ0)·(p − c0)`; a world angle to `wrap(θ − θ0)` ∈ (−π, π] |
| face ids | follow the existing generator `roots._face_point`: 0 = local −y, 1 = +x, 2 = +y, 3 = −x |
| normals | a *local patch* normal points **inward**, from the robot towards the surface (the direction the control law and `sensing.LocalPatch.normal_world` use). Label normals use the same inward convention |
| signed distance | along the inward/outward face line: positive = the robot centre is separated from the surface plane; at contact it equals `robot_radius` (0.18 m) |
| normalisation | **none is applied here.** All values are raw SI. The trainer standardises continuous targets with train-split statistics (CONTRACT 2.3) |
| masks | integer `0`/`1`. `1` = the quantity is observable/valid. A masked geometry target must **not** be zero-filled into a geometric loss; the discrete exit class still counts in the denominator (CONTRACT 2.2) |
| availability | `pre` = known before the execution starts (t = 0 / a priori); `exit` = measured at exit; `truth` = ground truth, offline labels/audits only |

### Deployable vs privileged

The `input` block contains only what the two robots can share before the execution: both
robots' start local observations, own kinematics, the static scene description and the joint
skill command, with observation-validity masks. It contains **no** object pose, velocity,
size-derived pose, face identity or physics truth. The `label` and `truth` blocks may use
ground truth and are clearly separated. `roots.json` additionally carries the generator's
oracle contact fields inside each root's `meta` (`contact_local`, `push_direction_world`,
`contact_face`, `contact_lateral`); those belong to the generator / arms O and P and are **not**
part of any model input.

The eight commands of one root are action branches on **one** physical initial state. Every
execution uses `VariantSpec("baseline")` (no added range noise, no observation delay); the old
round's eight perturbation variants are not used. Identity of the initial state is certified by
`snapshot_sha256`, a SHA-256 of the full `Snapshot` (scene, physics, tick, body states and the
replay prefix) for the `(root, baseline)` pair.

## 1. Record identity and provenance

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `record_id` | str | — | — | `root_id#cmd{index}` | pre | — |
| `version` | str | — | — | `option-prediction-mechanism-v1/phase_b-v1` | pre | — |
| `root_id` | str | — | — | generator: `phase_b/{split}/{class}/{index:04d}` | pre | — |
| `split` | str | — | — | `train` / `val` / `test` | pre | — |
| `interaction_class` | str | — | — | one of `roots.CLASSES` | pre | — |
| `template` | str | — | — | `free` / `wall` / `corner` / `passage` | pre | — |
| `command_index` | int | — | — | 0..7 | pre | — |
| `n_commands` | int | — | — | 8 (frozen tier-1 count) | pre | — |
| `arm` | str | — | — | `V` | pre | — |
| `variant` | str | — | — | `baseline` | pre | — |
| `snapshot_sha256` | str | — | — | SHA-256 of `dataclasses.asdict(build_snapshot(root, baseline))` | pre | — |
| `initial_velocities.object` | 2 | world | none | `VariantSpec("baseline")` initial condition | pre | — |
| `initial_velocities.robot0/1` | 2 | world | none | as above | pre | — |

## 2. `input` block — deployable, `pre` availability

### 2.1 `input.command` — the joint skill command

`alpha` is relative to the **currently observed inward local normal**, `t = Rot90(n)`; `speed`
is the reference push speed. Both come from the frozen ranges `ALPHA_CHOICES` / `SPEED_CHOICES`
of `roots.py`; the eight commands per root are distinct joint combinations chosen
deterministically from the root seed (public information only).

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `command_index` | int | — | — | generator | pre | — |
| `active_robots` | list[int] | — | — | root | pre | — |
| `alpha_rad` | 2 | observed inward normal | none | command | pre | `null` for inactive robot |
| `alpha_deg` | 2 | as above | none | command | pre | `null` for inactive robot |
| `speed` | 2 | — | none | command | pre | `null` for inactive robot |
| `active_duration` | 1 | s | none | frozen skill spec: 2.0 | pre | — |
| `force_limit` | 1 | N | none | frozen skill spec: 15.0 | pre | — |

### 2.2 `input.scene` — static, a priori

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `template` | str | — | — | root generator | pre | — |
| `arena` | 2 | world | none | static scene | pre | — |
| `obstacles` | k×5 | world | none | `[cx, cy, half_w, half_h, angle]` per box | pre | — |
| `object_size` | 2 | object frame | none | `PhysicsConfig` (1.0 × 0.25) | pre | — |
| `robot_radius` | 1 | m | none | `PhysicsConfig` (0.18) | pre | — |
| `scan_range` | 1 | m | none | frozen `OptionConfig` (0.6) | pre | — |
| `scan_rays` | 1 | — | — | frozen `OptionConfig` (72) | pre | — |
| `control_dt` | 1 | s | none | `PhysicsConfig` (0.05) | pre | — |

Static scene geometry is public. The object's *pose* is deliberately absent: only its size is
public.

### 2.3 `input.robot0` / `input.robot1` — shareable start local observation and own kinematics

Taken from the first control period's on-board observation (t = 0), so the robot's current
frame is exactly its start frame.

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `active` | 1 | — | — | root's active set | pre | — |
| `start_pose_world` | 3 | world | none | own kinematics | pre | `mask_own_kinematics` |
| `start_twist_world` | 3 | world | none | own kinematics | pre | `mask_own_kinematics` |
| `mask_own_kinematics` | 1 | — | — | always 1 (own state is known) | pre | — |
| `patch_distance` | 1 | m | none | local scan; centre-to-patch-centroid distance (unsigned) | pre | `mask_patch_valid` |
| `patch_points` | 1 | — | — | number of range returns in the fitted segment | pre | — |
| `patch_confidence` | 1 | — | none | `sensing.LocalPatch.confidence` | pre | `mask_patch_valid` |
| `patch_normal_own_frame` | 2 | robot own frame | none | local scan, inward normal | pre | `mask_patch_valid` |
| `patch_normal_sin_cos` | 2 | robot own frame | none | `[n_y, n_x]` of the inward normal (period-safe) | pre | `mask_patch_valid` |
| `patch_segment_own_frame` | 2×2 | robot own frame | none | least-squares segment of the observed run | pre | `mask_patch_valid` |
| `patch_centroid_own_frame` | 2 | robot own frame | none | centroid of `patch_segment_own_frame` | pre | `mask_patch_valid` |
| `contacting` | 1 | — | — | sensed `robot_object` contact flag | pre | `mask_contact` |
| `contact_point_own_frame` | 2 | robot own frame | none | sensed contact point | pre | `mask_contact` |
| `mask_patch_valid` | 1 | — | — | 1 iff ≥ 3 returns and a fitted normal exist | pre | — |
| `mask_contact` | 1 | — | — | equals `contacting` | pre | — |

## 3. `label` block — `exit` availability

### 3.1 Execution-level labels

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `duration_seconds` | 1 | s | none | `wall_clock_steps · control_dt` | exit | — |
| `duration_steps` | 1 | — | — | control periods executed | exit | — |
| `stage_seconds` | dict | s | none | per-stage accumulated control time | exit | — |
| `exit_class` | 1 | — | — | terminal `Exit` value (e.g. `settled_observed`, `contact_lost`, `patch_lost`, `settle_unobserved`) | exit | — |
| `primary_exit_reason` | 1 | — | — | first abort cause, `""` if none | exit | — |
| `aborted_by` | str/null | — | — | `robot0`/`robot1`/`environment:…`/`null` | exit | — |
| `effective_push_completed` | 1 | — | — | `push_completed` (effective push ≥ 2.0 s) | exit | — |
| `effective_push_seconds` | 1 | s | none | min over active robots of accumulated contact-push time | exit | — |
| `locally_settled` | 1 | — | — | the on-board quiet criterion fired at all | exit | — |
| `settled_observed` | 1 | — | — | `locally_settled ∧ ¬abort` | exit | — |
| `settled_after_abort` | 1 | — | — | `locally_settled ∧ abort` (never counted as normal completion) | exit | — |
| `exit_truth_low_speed` | 1 | — | — | truth audit: object **and** every robot below the settle thresholds at the exit instant | truth | — |
| `window_truth_low_speed` | 1 | — | — | truth audit: the same held over the whole hold window the criterion was asked to cover | truth | — |

### 3.2 `label.observation_validity.robot0` / `.robot1`

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `patch_valid` | 1 | — | — | last on-board observation carried a fitted segment | exit | — |
| `contacting` | 1 | — | — | contact flag at the last observation | exit | — |
| `patch_points` | 1 | — | — | returns in that segment | exit | — |
| `patch_valid_fraction` | 1 | — | — | fraction of logged periods with a valid segment | exit | — |
| `unavailable_periods` | 1 | — | — | periods with no delivered observation (0 for this dataset: no delay) | exit | — |

### 3.3 `label.robot0` / `label.robot1` — terminal geometry in that robot's **start** frame

The terminal geometry is the geometry of the **corresponding local segment**: the reference
face is the object face the robot started against (selected from the true initial object pose
and the robot's start centre; it agrees with the root generator's `contact_face` for every
active robot), and it is re-evaluated at the terminal object pose for the same face id. This is
the quantity a locally sensing robot measures, anchored in its own start frame; the mask says
whether the local segment was actually observable.

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `terminal_pose_change_own_start` | 3 | robot start frame | none | `[dx, dy, wrap(θ_T − θ0)]` of the robot | exit | — |
| `terminal_residual_velocity_own_start` | 3 | robot start frame | none | `[vx, vy, ω]` of the robot | exit | — |
| `reference_face` | 1 | — | — | start face id (0..3) | exit | — |
| `start_signed_distance` | 1 | m, face line | none | robot start centre vs. start face line | exit | — |
| `terminal_signed_distance` | 1 | m, face line | none | robot terminal centre vs. terminal face line; positive = separated | exit | `mask_terminal_geometry` |
| `start_tangential_offset` | 1 | m, face tangent | none | tangential foot-of-perpendicular coordinate at start | exit | — |
| `terminal_tangential_offset` | 1 | m, face tangent | none | the same coordinate at exit | exit | `mask_terminal_geometry` |
| `segment_tangential_offset_change` | 1 | m, face tangent | none | `terminal_tangential_offset − start_tangential_offset` | exit | `mask_terminal_geometry` |
| `terminal_normal_own_start_frame` | 2 | robot start frame | none | inward face normal at exit | exit | `mask_terminal_geometry` |
| `terminal_normal_sin_cos` | 2 | robot start frame | none | `[n_y, n_x]` of that normal (period-safe; report angular error) | exit | `mask_terminal_geometry` |
| `mask_terminal_geometry` | 1 | — | — | 1 iff the exit observation had a valid segment **and** the true terminal nearest-point distance to the object box ≤ `scan_range` | exit | — |

> **Naming.** `segment_tangential_offset_change` is the movement of the projection of the
> robot centre onto the **observed/corresponding segment's tangent line**, from the start frame
> to exit. It is a *segment / tangent offset change* of corresponding points. It is **not** the
> displacement of a material point of the object, and it must not be described as one.

### 3.4 `label.robot0` / `.robot1` — observed terminal local geometry (diagnostics)

The last on-board observation at or before exit, re-expressed in the robot's start frame.
Useful to compare the truth-derived labels above with what the robot actually saw. The last
logged observation is at most one control period (0.05 s) before the exact terminal state.

| field | dim | frame | normalisation | source | available | mask |
|---|---|---|---|---|---|---|
| `obs_terminal_patch_points` | 1 | — | — | last delivered observation | exit | — |
| `obs_terminal_patch_distance` | 1 | m | none | centre-to-centroid distance | exit | `mask_terminal_patch_valid` |
| `obs_terminal_patch_normal_own_start_frame` | 2 | robot start frame | none | observed inward normal | exit | `mask_terminal_patch_valid` |
| `obs_terminal_patch_centroid_own_start_frame` | 2 | robot start frame | none | observed segment centroid | exit | `mask_terminal_patch_valid` |
| `obs_terminal_contacting` | 1 | — | — | sensed contact flag | exit | `mask_terminal_contacting` |
| `obs_terminal_contact_point_own_start_frame` | 2 | robot start frame | none | sensed contact point | exit | `mask_terminal_contacting` |
| `mask_terminal_patch_valid` | 1 | — | — | 1 iff the last observation had ≥ 3 returns and a normal | exit | — |
| `mask_terminal_contacting` | 1 | — | — | equals `obs_terminal_contacting` | exit | — |

## 4. `truth` block — ground truth, offline only

Never a deployment input. Regenerate grouped splits, audit bookkeeping and group error
statistics.

| field | dim | frame | source | available |
|---|---|---|---|---|
| `object_start_world` | 6 | world | exact initial snapshot (`x, y, θ, vx, vy, ω`) | truth |
| `object_terminal_world` | 6 | world | executor's exact terminal state at exit | truth |
| `object_relative_terminal_initial_frame` | 3 | initial object frame | terminal object pose relative to the initial object pose | truth |
| `object_displacement` | 1 | m | ‖relative translation‖ | truth |
| `object_rotation` | 1 | rad | \|wrapped relative angle\| | truth |
| `object_final_speed` | 1 | m/s | terminal object ‖v‖ | truth |
| `object_final_omega` | 1 | rad/s | terminal object \|ω\| | truth |
| `robots_start_world` | 2×6 | world | exact initial snapshot | truth |
| `robots_terminal_world` | 2×6 | world | exact terminal states | truth |
| `robot_travel` | 2 | m | per-robot centre travel in the initial object frame | truth |
| `robot_terminal_speed` | 2 | m/s | per-robot terminal ‖v‖ | truth |
| `robot_terminal_omega` | 2 | rad/s | per-robot terminal \|ω\| | truth |
| `micro_motion` | 1 | — | 1 iff object displacement < 0.02 m and rotation < 0.03 rad | truth |
| `reached_effective_push` | 1 | — | 1 iff the execution entered the effective-push accounting | truth |
| `contact_period_ratio` | 1 | — | fraction of logged periods with any sensed contact | truth |
| `push_contact_ratio` | 1 | — | fraction of push periods with sensed contact | truth |
| `settle_evidence_span` | 1/null | s | sample span backing the settle verdict | truth |
| `settle_evidence_lag` | 1/null | s | evidence lag behind the present at the verdict | truth |

## 5. Artifact files

### `roots.json`
`version`, `salt` (the phase-B generator salt), `split_roots_requested`,
`commands_per_root`, `min_root_distance`, `angle_scale`,
`sampling_reports.{split}.{planned, sampled, per_class, sample_none, leakage_rejected, unfilled}`,
`leakage` (minimum cross-split distance and the metric definition),
`leakage_guard_probe` (the same guard re-run with an intentionally large 0.30 m threshold, to
show it really rejects), `command_choice_space`, and
`roots.{split}[].{root_id, split, interaction_class, template, seed, active_robots,
object_pose[3], robot_poses[2][3], meta}`. `meta` is the generator record used to rebuild the
`RootSpec`; it contains privileged oracle contact fields and is **not** a model input.

### `commands.json`
`version`, `commands_per_root`, `selection_rule`, and one entry per root:
`{root_id, split, interaction_class, active_robots, n_commands, commands[]}` where each
command is the `input.command` payload.

### `ledger.json`
`git_commit`, `version`, `phase`, `tag`, `arm`, `variant`, `controller`, `settle_protocol`,
`sensing`, full `option_config`, `split_roots_requested`, `splits.{split}.{roots_planned,
roots_sampled, roots_failed_to_sample, commands_per_root, executions_planned,
executions_executed, executions_dropped, wall_clock_seconds}`, `totals`, 
`executions_dropped_reasons`, `errors`, `sampling_failures`, `leakage_check`,
`leakage_guard_probe`, `sha256` (of `roots.json`, `commands.json` and every
`executions_*.jsonl`), and `artifacts`.

### `spot_check.json`
`root_id`, the eight `commands`, `distinct_command_signatures`,
`snapshot_sha256_first` / `snapshot_sha256_repeat`,
`snapshot_identical_across_commands`, `replay_identical` (two independent executions of
command 0 agree on the terminal state, exit class and duration),
`command_object_displacement`, `distinct_terminal_objects`, `terminal_objects`.

### `verification.json` (written by `verify_b.py`)
`counts`, `command_and_snapshot_identity`, `split_disjointness`, `coverage`, `fields`,
`exact_replay` (stored record vs. a fresh execution rebuilt from `roots.json` +
`commands.json` alone) and `ok`.

## 6. How the train / val / test split is kept disjoint

CONTRACT section 2.1 requires the split to follow the **underlying initial-state family**:
near-duplicate geometry, perturbation versions and same-trajectory derived states must not
cross splits. Phase B enforces this in two independent ways, and both are recorded.

1. **Disjoint seed families.** `roots_b.py` installs a new salt
   (`option-prediction-mechanism-v1/phase_b`) into the unmodified generator. Every seed is
   `sha256(SALT|split|class|index)`, so the train, val and test seed streams are disjoint by
   construction and also differ from every earlier round's roots. No phase-B root is a
   position/velocity perturbation of another: the old `VariantSpec` perturbations are not used
   at all — every execution is `VariantSpec("baseline")`, and the eight branches differ only in
   the joint skill command.
2. **A hard geometric distance guard.** While sampling, a candidate is rejected if
   `root_distance(candidate, other) < 0.02 m` against **any** already-accepted root in another
   split (and within the same split). The metric is the maximum of the object-centre distance,
   both robot-centre distances, and `0.5 · |wrapped angle difference|` for the object and both
   robots — a conservative "near in any channel ⇒ near" test. Rejections are counted in
   `sampling_failures.leakage_rejected`.

The resulting minimum cross-split distances are reported in `roots.json → leakage` and
`ledger.json → leakage_check` and are recomputed independently by `verify_b.py`:
train↔val min 0.0412 m, train↔test min 0.1176 m, val↔test min 0.1796 m — all ≥ 2× the 0.02 m
threshold, with zero root-id and zero seed overlap. `leakage_guard_probe` re-runs the same
guard with a deliberately large 0.30 m threshold and records that it does reject, so the guard
is not dead code.

