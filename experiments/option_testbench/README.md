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

## Effect-directed PUSH (2026-10-27)

`effect_push_eval.py` exercises an opt-in controller on the same `RandomStartEnv`
physics and task costs. The menu has eight world translation directions at 0.1–0.5 m,
two rotation signs at 0.1–0.5 rad, and the original non-push skills. Geometry masks and
feedback use only the robot's own sensing and a declared public rectangle-size prior.
Unavailable requests exit explicitly; there is no automatic REPOSITION fallback.

```bash
# Use a fresh output directory. Existing runs are never overwritten.
.venv/bin/python experiments/option_testbench/effect_push_eval.py \
  --out outputs/option_testbench_v1/effect_push_v1/round_008

# A bounded pair of scenes with an explicit feedback-gain change.
.venv/bin/python experiments/option_testbench/effect_push_eval.py \
  --out outputs/option_testbench_v1/effect_push_v1/round_009 \
  --cases tilt_+8 cooperative_N --heading-gain 0.2 --total-seconds 30

# Rebuild the viewer/plot from saved logs without simulation.
.venv/bin/python experiments/option_testbench/effect_push_eval.py \
  --out outputs/option_testbench_v1/effect_push_v1/round_007 --render-only
```

Default bounds: 10 s wall time per subprocess branch, 120 s total. The development
panel contains 33 scenes, each run with translation contact-point correction on/off.
Both groups retain velocity, gap and effect-endpoint feedback; rotation is identical
between groups. `index.html` embeds its data and opens directly without a server.
Use the selector, correction toggle, timeline and single-step button to inspect fitted
faces, target contact points, estimated effects and offline ground truth. `manifest.json`
freezes inputs/thresholds/source hashes; raw `logs/` and `metrics.json` retain failures.
`fulfilled` is offline option-effect fulfillment, not RL goal success or a controller
completion declaration. See `docs/handoff/20261027_Effect_PUSH_v1_Testbench.md` for
the measured results and remaining rotation limitations. The run9 trainer is unchanged.

## Lidar resolution / boundary jitter probe

`lidar_resolution_probe.py --out <fresh-directory>` compares actual 72/288-ray scans,
4x interpolation strictly within an already fitted face, and a causal endpoint EMA.
It runs three scripted scan sequences (static yaw phase, lateral motion then stop,
teammate occlusion), writes frozen inputs, raw records, timing and a comparison plot.
No controller or sensor defaults change. Synthetic points never become new sensor
returns or inflate support confidence. See
`docs/handoff/20261027_Lidar_Resolution_and_Boundary_Jitter.md` for measured endpoint
jitter, the interpolation negative result and the EMA motion-lag limitation.

## Endpoint-relative PUSH controller

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --compare-tracking \
  --total-seconds 60 --out outputs/option_testbench_v1/endpoint_tracking_v1/round_004
.venv/bin/python experiments/option_testbench/endpoint_tracking_audit.py \
  --out outputs/option_testbench_v1/endpoint_tracking_v1/round_004
```

This compares `endpoint_tracking=False/True`, with contact-point correction enabled in
both groups. It adds two-visible-face and tilted cooperation cases to the original panel
(35 scenes / 70 branches). The paired change includes endpoint-relative target generation
and safe staging, not just a filter. Each robot's own endpoint evidence, bounded physical
edge observations, active anchor, age and contact distance are visible in replay; the
teammate's estimate is shown offline only. The audit aligns estimated endpoints with
pre-step truth for offline errors. Coverage is N/A when tracking is disabled, not zero.
No new training, default sensor change or automatic reposition is performed. See
`docs/handoff/20261027_Endpoint_Tracking_PUSH_v2.md` for measured improvements and
regressions. The controller remains opt-in because task-effect gains were mixed.

## Faster tangential preparation (v3)

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --compare-approach \
  --total-seconds 60 --out outputs/option_testbench_v1/fast_approach_v1/round_005
.venv/bin/python experiments/option_testbench/approach_stability_audit.py \
  --out outputs/option_testbench_v1/fast_approach_v1/round_005
```

Both groups use endpoint tracking and contact-point correction. The paired variable is
only the tangential preparation profile: up to 0.20 m/s away from contact, distance-based
braking, velocity damping and a lateral-speed gate before PUSH. Small initial corrections
keep the previous path; blend strength increases smoothly from 5 to 15 cm initial error.
In-contact control and all physics/budgets/thresholds stay fixed. The panel contains 45
scenes / 90 branches, including +/-1 cm position and +/-1 degree heading perturbations.
The viewer starts at `two_faces_N`; `STABILITY.md`, `stability.json` and `stability.png`
show preparation times and all nearby-start effects. See
`docs/handoff/20261027_PUSH_v3_Faster_Approach_and_Local_Stability.md` for results and
the retained single-robot tilted-face sensitivity. Suggested candidate configuration:
`EffectPushConfig(endpoint_tracking=True, fast_approach=True)`; the legacy path remains
available. The old run9 trainer is not changed and no new RL run is started.

## Independent scales with proportional normal references (v4)

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --compare-joint-execution \
  --total-seconds 60 --out outputs/option_testbench_v1/joint_execution_v1/round_004
.venv/bin/python experiments/option_testbench/joint_execution_audit.py \
  --out outputs/option_testbench_v1/joint_execution_v1/round_004
```

The new translation-pair protocol retains independent directions/scales, proportional
NORMAL propulsion bias and unscaled tangential/contact-following feedback. A common
staging/contact handshake and shared bounded deadline avoid asymmetric waiting failures.
In the archived v4 comparison, scales are contact-conditioned normal reference budgets; neither vector addition nor
a common desired object displacement is imposed. Completion of the reference budget
does not imply object success (opposing pushes may cancel completely). Final object
goal scoring is N/A for new paired exits. Both exit reasons, budgets, phase events and
actual motion remain visible in the standalone replay. Rotation angles remain angle
effects; they are not treated as normal metres. `legacy_v3` is retained for baseline
reproduction; other historical comparison CLI modes explicitly retain that contract.
See `docs/handoff/20261027_PUSH_v4_Independent_Scales_and_Joint_Execution.md` for the
40-branch development result, local sensitivity, failures and test-stage accounting.

## Same-face N / NW diagnosis

`same_face_diagnosis.py --out <fresh-directory>` runs a bounded 2x2 geometry/direction
panel, each paired with contact-point correction off/on. It preserves occlusion and
all production safety logic, separates the original unequal-gap placement from an
explicit equal-gap scene, writes raw logs and standalone replay, and plots requested
contact separation versus robot clearance. This is a diagnostic ablation, not a
controller update or new action mask. See
`docs/handoff/20261027_Same_Face_N_NW_Diagnosis.md` for the target-allocation conflict,
secondary endpoint bias and the recommendation to constrain contact regions.

## Same-face contact regions and working clearance (v5)

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --compare-contact-allocation \
  --total-seconds 60 --out outputs/option_testbench_v1/same_face_allocation_v1/round_003
.venv/bin/python experiments/option_testbench/same_face_allocation_audit.py \
  --out outputs/option_testbench_v1/same_face_allocation_v1/round_003
```

Both variants keep endpoint tracking, direction correction, fast preparation and the
v4 independent reference protocol. The candidate fixes tangential order, reserves a
visible contact region on each side, and projects preferred points into those regions.
Same-face low-speed soft avoidance uses a 4 cm body gap plus a 1 cm target pad; fast
closure/uncertain first observation retain more buffer. Hard geometry and the existing
2 cm safety-abort gap do not change. Narrow/unobservable regions fail explicitly.
The panel includes original/equal-gap N/NW cases, +/-1 cm perturbations, other-face
regressions and a narrow-face rejection (24 scenes / 48 branches). The audit writes
`ALLOCATION.md`, `allocation_audit.json` and `allocation.png`. See
`docs/handoff/20261027_PUSH_v5_Same_Face_Contact_Allocation.md` for the full rules and
measured distances. Reproduce its reference contract with `EffectPushConfig(endpoint_tracking=True,
fast_approach=True, same_face_allocation=True, translation_completion="normal_reference")`; the allocation flag defaults off
pending review. No global mask or RL trainer is changed.

## Measured own normal travel (v6)

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --compare-normal-travel \
  --total-seconds 60 --out outputs/option_testbench_v1/normal_travel_v1/round_003
.venv/bin/python experiments/option_testbench/normal_travel_audit.py \
  --out outputs/option_testbench_v1/normal_travel_v1/round_003
```

Single and dual EffectAction translations default to `translation_completion="normal_travel"`.
Use own global-position differences projected on the preceding sensed inward face normal,
only across continuously observed active contact. Reverse travel subtracts; preparation
and unobserved intervals are excluded. Contact braking contributes to the recorded total.
The reference integral and offline object motion are distinct diagnostics. Both benchmark
variants use the same endpoint tracking, approach and allocation settings; only the
completion/progress measure differs. Single reference-clock baselines are an explicit
counterfactual, not historical v5 single-PUSH results (which used object effects).

`normal_travel_audit.py` reconstructs travel from saved pre-step robot poses and preceding
observed normals, checks completion claims, and writes `travel_audit.json`,
`TRAVEL_AUDIT.md` and `normal_travel.png`. The replay shows measured travel separately
from reference integration and object effects. Different distances can exit at substantially
different times and cause more joint rotation; proportional command biases do not force
actual travel speeds to share their ratio. Opposing pushes with no travel now fail explicitly.
Rotation units and the legacy 28-action trainer are unchanged. Preserve old object-effect
behavior with `translation_completion="object_effect", joint_execution="legacy_v3"`.
The frozen 34-case / 68-branch round and its limitations are documented in
`docs/handoff/20261027_PUSH_v6_Measured_Normal_Travel.md`.

## Post-completion contact following (v7)

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --compare-post-follow \
  --total-seconds 60 --out outputs/option_testbench_v1/post_push_follow_v1/round_003
.venv/bin/python experiments/option_testbench/post_push_follow_audit.py \
  --out outputs/option_testbench_v1/post_push_follow_v1/round_003
```

The 24-case / 48-branch panel compares `post_push="hold_brake"` with
`post_push="contact_follow"`; both use measured own normal travel and all v5 geometry
modules. The candidate switches directly to following when its active distance is
reached. Active travel and command integrals freeze; follow contact-normal travel
and total own follow path are recorded separately. Following uses own observed surface
motion and bounded gap/tangent tracking, preserving its endpoint-relative contact role.
It has no active propulsion bias or continued nominal-direction correction. Commanded
force is capped at 3 N total and 0.75 N normal magnitude; desired speed at 0.16 m/s.
Existing actuator/acceleration and safety limits remain. These are not contact-force
measurement guarantees. Limits: 8 s, 0.75 m own path, 0.25 s continuous loss of contact.

Both active phases resolving causes a next-period joint braking release; failed
followers return to the existing Hold behavior while an active peer may continue.
Only paired measured translations enable follow; solos and rotations remain unchanged.
The default stays `hold_brake`, with the new mode explicitly selectable for review.
The benchmark includes +/-1 cm input perturbations, narrow-face/unavailable requests,
opposing pushes and rotation/solo regressions. The replay shows both stages/contacts,
active completion times, separate follow travel and the two contact targets.
`FOLLOW_AUDIT.md`, `follow_audit.json` and `post_follow.png` separate controller
claims, independently reconstructed travel/limits, physical effects and local sensitivity.
See `docs/handoff/20261027_PUSH_v7_Post_Completion_Contact_Follow.md` for the measured
same-face improvement and explicit adjacent-face contact-loss failures. No RL run.

## Gap-based joint reapproach (v8)

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --compare-joint-recovery \
  --total-seconds 60 --out outputs/option_testbench_v1/joint_contact_recovery_v1/round_003
.venv/bin/python experiments/option_testbench/joint_contact_recovery_audit.py \
  --out outputs/option_testbench_v1/joint_contact_recovery_v1/round_003
```

Both variants follow after their active travel. Compare the frozen v7 binary-contact
abort against `contact_recovery="joint_approach"`. Clearance >10 mm requests a next-period
joint pause. Each side reapproaches its own visible face; both must be within 3 mm with
normal speed <=0.06 m/s for 0.1 s before resuming their prior stages. Normal approach
reference is capped at 0.04 m/s; active push bias is absent. Recovery time (2 s/attempt,
three entries maximum) and geometry/safety failures are explicit. The episode/shared
clock and task costs continue. Active progress excludes recovery; recovery normal
travel and path length have separate ledgers. Small gaps with no contact are labelled
proximity keeping, with physical contact fractions separately visible.

The panel preserves the 24 v7 scenes and adds two declared `outward_pulse` diagnostics:
a 4 N / 0.25 s additive outward motor force, identical across variants, applied through
the existing force/world integration. Pulses are labelled in replay and raw rows;
no teleport, hidden-state control input or physics/reward change is involved. They are
mechanism checks, not ordinary random scenes. All 24 ordinary abort baselines reproduce
v7 states and forces exactly. The audit writes `RECOVERY_AUDIT.md`,
`recovery_audit.json`, and `joint_recovery.png`, checking the active/follow/recovery
split, causal phase release and nominal versus disturbed commands.

Enable with `EffectPushConfig(endpoint_tracking=True, fast_approach=True,
same_face_allocation=True, post_push="contact_follow", contact_recovery="joint_approach")`.
Historical `contact_recovery="abort"` remains the default. See
`docs/handoff/20261027_PUSH_v8_Joint_Contact_Recovery.md` for measured gaps, contact
fractions, recovery times, negative controls and validation scope.

## Compact two-robot replay inspector

The right panel has robot 0/1 buttons. Selecting a robot updates its action, requested
target/unit, measured or estimated progress, actual contact/gap, object-effect
projection, chart and highlighted geometry without restarting the current frame.
Each side uses its own action and logged counters; a held robot retains its last
estimate with an explicit stale label. Non-push durations use their own logged progress.
Core state stays visible; geometry/control diagnostics, comparison notes and the full
results table are collapsed by default. This is a display-only update.

Regenerate with `effect_push_eval.py --render-only --out <existing-round>`; this reads
the saved logs and requires no new simulation. The latest v8 round and portable bundle
include this inspector. The preceding page/bundle are preserved in
`viewer_before_robot_switch/`. Final renderer/template hashes are in `render_manifest.json`;
browser checks and screenshots are in `viewer_inspector_validation.json` and
`inspector_robot1.png`. The frozen input manifest, metrics and raw logs remain unchanged.

## REPOSITION and mixed-option coverage

```bash
timeout --signal=TERM --kill-after=2s 65s .venv/bin/python \
  experiments/option_testbench/effect_push_eval.py --reposition-panel \
  --total-seconds 60 --out outputs/option_testbench_v1/reposition_combinations_v1/round_003
.venv/bin/python experiments/option_testbench/reposition_combinations_audit.py \
  --out outputs/option_testbench_v1/reposition_combinations_v1/round_003
```

The panel has 24 branches and one current configuration: CW/CCW at 0.10/0.25/0.50/1.00 m,
near-corner starts, REPOSITION+PUSH on the same/opposite side, REPOSITION+SUPPORT_FOLLOW,
REPOSITION+WAIT, two REPOSITIONs, swapped robot roles and an unobserved-object rejection.
It is coverage, not an off/on ablation. Existing controllers, physics and task conditions
are unchanged. Mixed actions use the shared clock and Hold after an early exit;
the double-PUSH joint-recovery protocol is not enabled for mixed kinds.

Merge extra frozen results for display only:

```bash
.venv/bin/python experiments/option_testbench/effect_push_eval.py --render-only \
  --out outputs/option_testbench_v1/joint_contact_recovery_v1/round_002 \
  --include-round outputs/option_testbench_v1/reposition_combinations_v1/round_002
```

The latest page contains the original 52 plus 24 new replays and a combination filter.
Single-configuration cases hide the comparison switch; existing paired controls remain.
Each source keeps its own manifest/metrics/logs, with display provenance in
`viewer_sources.json`. REPOSITION displays signed boundary travel, observed face-change
events, the selected robot's path and current tangent. Its progress is not object displacement.
The independent audit writes `COMBINATIONS_AUDIT.md`, `reposition_audit.json` and
`reposition_combinations.png`. Existing SUPPORT_FOLLOW duration overrun and same-face
interference are preserved and reported in
`docs/handoff/20261027_REPOSITION_Mixed_Option_Testbench.md`.

## Rounded boundary following and larger REPOSITION scales

```bash
.venv/bin/python experiments/option_testbench/effect_push_eval.py \
  --compare-reposition-corners --total-seconds 60 \
  --out outputs/option_testbench_v1/reposition_arc_v1/round_003
.venv/bin/python experiments/option_testbench/reposition_arc_audit.py \
  --out outputs/option_testbench_v1/reposition_arc_v1/round_003
.venv/bin/python experiments/option_testbench/effect_push_eval.py \
  --reposition-scales-v2 --total-seconds 60 \
  --out outputs/option_testbench_v1/reposition_scales_v2/round_003
.venv/bin/python experiments/option_testbench/reposition_scales_audit.py \
  --out outputs/option_testbench_v1/reposition_scales_v2/round_003
```

The first panel compares matched old distances with only
`reposition_boundary="legacy_face" / "rounded_scan"` changed. The latter uses nearest
finite own-observed segments and bounded observed endpoints for a continuous radial
normal/tangent around convex corners. Censored endpoints are not invented corners.
Gap, speed, force, distance-credit rules and time bounds retain their original values.

The second panel separately covers the requested larger scales, using the new
`ACTIONS_REPOSITION_V2` / `REPOSITION_V2_VERSION` menu at 0.2/0.4/0.8/1.2/1.8 m.
It has 68 actions; the old 66/28 menus remain unchanged. Use explicit `actions=` with
`action_id`, `action_mask`, and `type_scale_mask` for the new table. Do not reuse old
numeric IDs or assume every type has five scales. A 1.8 m solo request reaches the
opposite face in the tested starts; mixed PUSH cases can still time out.

The latest shared viewer includes both panels and a “新尺度 REPOSITION” filter:

```bash
.venv/bin/python experiments/option_testbench/effect_push_eval.py --render-only \
  --out outputs/option_testbench_v1/joint_contact_recovery_v1/round_002 \
  --include-round outputs/option_testbench_v1/reposition_combinations_v1/round_002 \
  --include-round outputs/option_testbench_v1/reposition_arc_v1/round_002 \
  --include-round outputs/option_testbench_v1/reposition_scales_v2/round_002
```

Each record is decoded against its frozen source menu, validated before displaying
different codebooks together. Original manifests/metrics/logs are preserved. The
control and scale experiments are separate; see
`docs/handoff/20261028_REPOSITION_Rounded_Boundary_and_Scales_v2.md` for evidence,
physical-clearance labels, local perturbation limits and retained coordination failures.


### 同伴外侧绕行与静态窄通道（2026-10-28）

`--compare-reposition-peer` 在冻结 v9 的 68 动作菜单上，比较局部排斥和
`reposition_peer_detour="outer_arc"`（两组均为 rounded_scan）。21 场景覆盖
RP+PUSH、RP+WAIT、双 RP、初态 ±1 cm、无物体观测，以及 0.26/0.78 m 静态间隙。

```bash
.venv/bin/python experiments/option_testbench/effect_push_eval.py \
  --compare-reposition-peer --out outputs/option_testbench_v1/reposition_peer_detour_v1/round_NEW \
  --branch-seconds 6 --total-seconds 60
.venv/bin/python experiments/option_testbench/reposition_peer_audit.py \
  --out outputs/option_testbench_v1/reposition_peer_detour_v1/round_NEW
```

额外绕行计入原有符号行程；不承诺到背面。本侧静态回波阻挡短程路线时
`blocked_static_route`，另一侧维持原独立执行语义。完全丢失物体观测仍停止；
两个同面 WAIT 负例保留 patch_lost。旧配置默认关闭；未启动新 RL。
本轮轨迹、负结果与测试见 `docs/handoff/20261028_REPOSITION_Peer_Detour_and_Static_Clearance.md`。

### Run10 受阻边界与前置安全分离（2026-10-28）

从三条归档训练轨迹精确恢复受阻边界，对照原控制与显式开启的
`peer_clearance_recovery="boundary_v1"`。16 场景包含原下一动作、CW/CCW RP+WAIT、
同一任务时限内的原动作尾段，以及封闭/单侧可退/健康负例。不会启动 RL。

```bash
.venv/bin/python experiments/option_testbench/peer_clearance_eval.py \
  --out outputs/option_testbench_v1/peer_clearance_recovery_v1/round_NEW \
  --branch-seconds 8 --seconds 60
```

需要保留 `overnight_parallel_20261028/run10_001/rollouts/` 的原始源轨迹。
最终实测查看 `outputs/option_testbench_v1/peer_clearance_recovery_v1/round_003/index.html`，
可切换两侧机器人和原控制/前置分离；原始日志、源快照与未完成记录均保留。
6/6 RP+WAIT 分支完成，但三条完整尾段任务仍失败，FOLLOW 再次逼近问题尚未解决。
报告：`docs/handoff/20261028_Run10_Blocked_State_Diagnosis_and_Recovery.md`。

### 独立 FOLLOW 避让与 2 cm 间隙（2026-10-28）

```bash
.venv/bin/python experiments/option_testbench/follow_peer_eval.py \
  --out outputs/option_testbench_v1/follow_peer_guard_v2/round_NEW \
  --branch-seconds 8 --seconds 60
```

需要保留前轮 `peer_clearance_recovery_v1/round_003/parent_*.pkl` 冻结边界。
8 场景 × 原 5 cm/避让 5 cm/避让 2 cm = 24 分支，三组均使用相同的前置分离。
新行为通过 `EffectPushConfig(standalone_follow_gap=.02,
standalone_follow_peer_guard="scan_projection_v1")` 显式选择；历史默认值与 RL 入口保留。
最终测台 `outputs/option_testbench_v1/follow_peer_guard_v2/round_003/index.html`
支持切换对照组和机器人；负例与未完成记录保留。详见
`docs/handoff/20261028_FOLLOW_Peer_Guard_and_2cm_Gap.md`。

### 混合 PUSH 的同伴占据区域投影（2026-10-28）

```bash
.venv/bin/python experiments/option_testbench/mixed_contact_eval.py \
  --out outputs/option_testbench_v1/mixed_contact_projection_v1/round_NEW \
  --branch-seconds 8 --seconds 60
.venv/bin/python -m experiments.option_testbench.mixed_contact_audit \
  --out outputs/option_testbench_v1/mixed_contact_projection_v1/round_NEW
```

两组均使用相同的 2 cm FOLLOW 和同伴避让，仅切换
`mixed_contact_allocation=False/True`。8 场景覆盖原 PUSH+FOLLOW、初态 ±1 cm、
交换机器人、PUSH+WAIT、自由同伴、窄面与原训练 WAIT+PUSH。
结果和可切换接触目标点的测台在
`outputs/option_testbench_v1/mixed_contact_projection_v1/round_001/`。
候选点只从本侧可观测范围选择；不足则退出，不穿过同伴去争抢另一侧区域。
详见 `docs/handoff/20261028_PUSH_Mixed_Contact_Projection.md`。

### FOLLOW 绑定 PUSH 生命周期（2026-10-28）

```bash
.venv/bin/python experiments/option_testbench/push_bound_follow_eval.py \
  --out outputs/option_testbench_v1/push_bound_follow_v1/round_NEW \
  --branch-seconds 8 --seconds 60
.venv/bin/python -m experiments.option_testbench.mixed_contact_audit \
  --out outputs/option_testbench_v1/push_bound_follow_v1/round_NEW
```

两组保持相同的 2 cm FOLLOW、同伴避让和混合接触点选择；只对照原计时退出
与 `follow_completion="peer_push"`。新组显式使用 65 项
`ACTIONS_PUSH_BOUND_FOLLOW` / `PUSH_BOUND_FOLLOW_VERSION`，旧组保留 v9 的 68 项。
旧 FOLLOW 的不同秒数在新组映射为同一个无时长动作，映射前后标签均记录。
默认训练入口没有迁移。测试台在
`outputs/option_testbench_v1/push_bound_follow_v1/round_002/index.html`，
可查看绑定状态、PUSH 阶段、停止通知和末端制动。负例保留失败原因。
详见 `docs/handoff/20261028_PUSH_Bound_FOLLOW_Lifecycle.md`。

### 扫描与执行效率（2026-10-28）

```bash
.venv/bin/python experiments/option_testbench/execution_efficiency_eval.py \
  --out outputs/option_testbench_v1/execution_efficiency_v1/round_NEW \
  --variants eff_scalar eff_batch eff_prepare --seconds 45 --branch-seconds 8
.venv/bin/python -m experiments.option_testbench.mixed_contact_audit \
  --out outputs/option_testbench_v1/execution_efficiency_v1/round_NEW
.venv/bin/python -m experiments.option_testbench.execution_efficiency_benchmark \
  --out outputs/option_testbench_v1/execution_efficiency_v1/round_NEW
```

计时时单独运行 benchmark，不与测试或审计并发。最终比较在
`outputs/option_testbench_v1/execution_efficiency_v1/round_003/index.html`，
完整提速/60 Hz 物理的早期对照保留在 round_001。当前默认仍为标量扫描；
`EnvConfig(lidar_backend="numpy_batch")` 显式选择等语义 CPU 批量实现。
`eff_prepare` 保持 120 Hz 物理/20 Hz 控制/20 Hz 扫描与原接触推动速度，
只加快准备。其他变量、负结果、计时范围与局限见
`docs/handoff/20261028_Option_Execution_Efficiency.md`。

### 执行效率 v2 候选配置（2026-10-28）

`coimagine.random_start.efficient_profile` 提供独立的 v11 / 62 项动作表、
环境构造、联合 mask 和元数据。只保留 WAIT 2 s，FOLLOW 绑定 PUSH；
前 60 个 PUSH/REPOSITION 保持原编号。旧 RL 68 项接口没有迁移。
平移采用平滑准备、接触切向阻尼及分阶段无进展退出；旋转保留原速度参数。
120 Hz 物理、20 Hz 控制/扫描、接触推动速度与安全阈值保留。

```bash
.venv/bin/python experiments/option_testbench/option_efficiency_v2_eval.py \
  --out outputs/option_testbench_v1/option_efficiency_v2/round_NEW \
  --variants baseline efficient --seconds 60 --branch-seconds 8
.venv/bin/python -m experiments.option_testbench.mixed_contact_audit \
  --out outputs/option_testbench_v1/option_efficiency_v2/round_NEW
```

最终回放在 `outputs/option_testbench_v1/option_efficiency_v2/round_002/index.html`。
主测台 16/16 基线完成场景通过相对质量门槛，3 个原失败保留失败；
补充协同为 3/3 完成场景保留、1 个原失败保留。
计算快路径的等物理工作量基准、早期拒绝方案、阻塞重放及局限见
`docs/handoff/20261028_Option_Efficiency_v2.md`。

### 单机器人推动速度与 approach 限速诊断 v2

保持同一单机器人向北0.3 m场景，测试推动/准备速度上限至0.5 m/s，
以及一组独立的物体减半质量对照；生产配置不变。复现命令：

```bash
.venv/bin/python -m experiments.option_testbench.single_push_speed_v2_eval \
  --out outputs/option_testbench_v1/single_push_speed_v2/round_NEW
.venv/bin/python -m experiments.option_testbench.single_push_speed_v2_refine \
  --out outputs/option_testbench_v1/single_push_speed_v2/refine_NEW
```

15组最终结果合并在 `outputs/option_testbench_v1/single_push_speed_v2/review/index.html`。
推荐原型 `prepare_matched_0.30` 为PUSH上限0.32、approach上限0.30、协调切向纠偏及末端减速，
单例6.25→4.35 s；不表示持续实际0.5 m/s已验证或协同回归已完成。
限速链路、物理阻力模型、负结果见 `docs/handoff/20261028_Single_PUSH_Speed_v2.md`。
