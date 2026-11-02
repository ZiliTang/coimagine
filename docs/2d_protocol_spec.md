# CoImagine 2D — protocol specification (first round)

Continuous-2D efficiency candidate (2026-10-28, `execution-efficiency-v1`):
`EnvConfig(lidar_backend="numpy_batch")` batches the same CPU analytic ray/segment,
wall-capsule and circle intersections. Real sensing remains 72 body-fixed rays,
1.2 m, 20 Hz, with the same nearest-hit/category/tie semantics and time stamps;
surface interpolation is not additional ray evidence. The scalar implementation
remains the reference/default. Branches record the backend and reject undeclared
backend changes; ablations explicitly opt in. Current random-start clocks require
20 Hz control, and constructing an EnvConfig with another base control_dt is
rejected rather than silently corrupting reward/time/filter accounting.

The preferred local speed candidate changes preparation only: approach normal cap
0.10→0.15 m/s, free tangential cap 0.20→0.30 m/s, approach tangential gain 3.0→3.3.
Contact propulsion stays 0.12 m/s; feedback point rate, nominal distances, contact
and safety thresholds, force/reference-acceleration bounds, timeouts and costs
retain their values. It is explicitly configured, not a silent checkpoint upgrade.
An aggressive 0.18 m/s translation candidate and 60 Hz physics discretization are
retained as separate negative/limited ablations, not promoted to the maintained
120 Hz model. Local effect gates are offline diagnostics, not task reward changes.
Evidence: `docs/handoff/20261028_Option_Execution_Efficiency.md`.

Continuous-2D candidate amendment (2026-10-28, `push-bound-follow-v1`):
the explicit `effect-options-v10-push-bound-follow` codebook has 65 actions. The
first 64 PUSH/RP/WAIT descriptors and IDs match v9; the four timed FOLLOW slots are
replaced by one `SUPPORT_FOLLOW_PUSH` (`scale_kind="peer_push"`, no duration).
It requires `EffectPushConfig.follow_completion="peer_push"`. Factory mismatches
and FOLLOW without a PUSH partner are rejected before physics. Local, joint and
peer-conditioned mask helpers expose this constraint before sampling; PUSH+RP,
PUSH+PUSH and all non-FOLLOW combination rules are unchanged. The maintained RL
adapter still uses its archived 68-action contract and is not silently migrated.

The follower approaches/tracks during preparation and follows throughout PUSH,
including the pusher's brake. Its observed tracking seconds are diagnostics only,
never an independent completion threshold. Only preceding-period public PUSH
phase/exit reports control handoff; the controller receives no current peer scan,
contact estimate or future feedback. Normal PUSH exit releases bounded follower
braking (`follow_push_complete`); failed PUSH releases with `peer_push_failed`.
If FOLLOW itself fails, PUSH receives `peer_follow_failed` on the next boundary.
Safety/observation exits remain active. Status handoff is part of this centralized
execution experiment, not a claim of free finite-bit deployment communication.

PUSH keeps its original timeout. FOLLOW inherits a bounded joint window of that
timeout plus at most 0.85 s (one 0.05 s status handoff and 0.8 s brake allowance),
subject to existing joint/task caps and control-period resolution. Extra physical
time/motion has the original cost; no new reward or time credit is introduced.
Unfinished skills at a global cap retain explicit window/task exits. FOLLOW normal
exit may lag PUSH by braking time; this is not exact same-instant termination.
The 2 cm target, avoidance and mixed contact projection are unchanged. See
`docs/handoff/20261028_PUSH_Bound_FOLLOW_Lifecycle.md` for local evidence and remaining
gap-tracking limits.

Continuous-2D candidate amendment (2026-10-28, `mixed-peer-occupied-v1`):
`EffectPushConfig(mixed_contact_allocation=True)` requires `same_face_allocation`.
Translation PUSH now applies a reservation when its independently executing peer
occupies the outward side of the locally observed face. It uses only own peer
returns, own fitted geometry and bounded observation history; it does not read the
peer's private target or the true object pose. On the robot's original side of the
peer, project the preferred contact to the nearest remaining visible interval,
excluding peer body separation and the existing clearance/padding. This unilateral
case does not require the dual-PUSH equal separator; dual-PUSH rules stay unchanged.
If no visible lane remains, return `same_face_lane_unavailable`; loss of peer evidence
and a confirmed departure retain explicit age/release handling. Current-face
association and visibility caps remain binding. No global reachability is claimed.

The existing same-face working-clearance rule is now available to this mixed case:
the first/aged peer sample retains 12 cm repulsion, then valid stable allocation
allows the existing 4 cm working margin with dynamic closure padding. The 2 cm
safety exit, static-obstacle rules, actuator limits, request scales, timeouts and
costs do not change. Defaults remain disabled and old checkpoint descriptors omit
the disabled optional field. The 16-branch comparison and retained rotation/short-face
limitations are in `docs/handoff/20261028_PUSH_Mixed_Contact_Projection.md`.

Continuous-2D candidate amendment (2026-10-28, `standalone-follow-peer-v2`):
explicit `EffectPushConfig(standalone_follow_gap=0.02,
standalone_follow_peer_guard="scan_projection_v1")` reduces standalone FOLLOW's
object-edge target to 2 cm and projects closing velocity away from the shared peer
pose/velocity, aiming for a 6 cm peer reserve. Own object/static returns constrain
candidate short swept motions; no feasible candidate means braking. This is local
avoidance, not a collision guarantee against independently moving peers. Original
2 cm peer safety exits and actuator/physics limits remain. The surface reference
still uses the current own visible-point centroid and initial tangential offset.

Projected FOLLOW counts maintenance time only when its observed gap is within 1 cm
of target, relative-pose error is within the existing 6 cm tolerance, and public
peer gap is at least 5 cm. Otherwise yielding consumes the unchanged timeout/cost
without maintenance credit. Unprojected FOLLOW retains its existing clock.
The completed-PUSH contact-follow controller is separate and unchanged. Defaults
remain 5 cm/guard off for historical reproduction; archived run10 descriptors/hashes
omit these disabled optional fields, and enabling either is a distinct descriptor.
The maintained RL adapter has not been switched or restarted. The 24-branch
comparison includes a retained WAIT+PUSH reapproach and a FOLLOW budget failure;
2 cm clearance alone does not establish contact support. See
`docs/handoff/20261028_FOLLOW_Peer_Guard_and_2cm_Gap.md`.

Continuous-2D diagnostic amendment (2026-10-28, `peer-clearance-prefix-v1`):
`EnvConfig.peer_clearance_recovery="boundary_v1"` explicitly enables a joint-option
prefix for close teammates. The maintained RL adapter still uses `off`. At option
entry, shared robot kinematics trigger separation at a 3 cm edge gap, or predicted
entry into that gap. Each robot chooses retreat using only its own scan/contact;
no hidden object state or teammate scan is supplied. Release requires a 6 cm gap,
low relative radial speed and 0.1 s readiness. Recovery is capped at 2 s and reports
`blocked_peer_clearance` if it cannot release. Missing returns do not certify space.

Recovery consumes the original option/task time and ordinary physical costs, but
earns no nominal PUSH/RP travel or FOLLOW maintenance credit. WAIT may move under
this safety override and its clock continues. Original safety guards, action IDs,
physics and reward formulas are unchanged. Branch restoration rejects an undeclared
mode change; diagnostic ablations explicitly set `allow_execution_change=True`.
This prefix does not prevent mid-option FOLLOW from approaching the peer again.
Evidence and remaining failures: `docs/handoff/20261028_Run10_Blocked_State_Diagnosis_and_Recovery.md`.

Continuous-2D training adapter (2026-10-28, `option-rl-effect-v1`, separate
`experiments/option_rl_v10/`): the user requested a new-round readiness check.
The adapter explicitly uses the existing v9 68-ID menu, conditional type/scale
probabilities with the ragged four-scale WAIT/FOLLOW masks, and the current effect
PUSH / rounded-scan / peer-outer controls. The controller configuration and full
codebook are frozen in checkpoints; old 28-ID checkpoints are not migrated.
An opt-in standalone FOLLOW clock `maintenance_duration` fixes completion to the
requested maintenance time; preparation remains in the existing timeout only.
Legacy callers keep `legacy_window` and their historical behavior.

The new `simple-legal-v2` generator rejects actual disc/rotated-box/teammate/wall
intersections and enforces explicit clearances instead of clipping coordinates.
Training uses a new indexed scene each episode; validation/test roots stay fixed
in separate random streams, with content hashes audited. This is a new baseline;
old run9 roots/results are preserved. The adapter uses the frozen gamma=1 cost;
physical anomaly trajectories are recorded and quarantined from PPO, and all
validation roots remain in the evaluation denominator. Neither task reward weights
nor task success are changed. The centrally shared local-observation PPO does not
claim a finite-bit deployment information boundary. Readiness evidence and limits:
`docs/handoff/20261028_New_Option_RL_Readiness.md`.


Continuous-2D amendment (2026-10-28, `peer-outer-v2`): explicitly opt-in
`reposition_peer_detour="outer_arc"` requires `reposition_boundary="rounded_scan"`.
It follows the smooth exterior combination of the finite own-observed object offset
and a 0.50 m reference circle around the CURRENT already-shared teammate pose.
The selected CW/CCW tangent is also the signed-travel accounting direction; the
extra route consumes the original distance/time budgets. Old defaults/menu IDs,
physics, costs and sensing are unchanged. A shadow-censored endpoint is not a corner.
An already-active detour may retain a CURRENT valid observed plane as a conservative
exclusion half-plane when finite support cannot be used; no hidden continuation is
supplied and missing valid object observation still trips the existing guard.

A 0.6-second local swept-disc check of CURRENT own STATIC returns blocks the planned
movement if the predicted body clearance is at most 0.04 m; RP then exits with
`blocked_static_route`. The existing static repulsion and safety abort remain.
This is a local route rejection, not proof of global infeasibility; absent returns
are not confirmed free space. Another robot's independently executing option is
not automatically cancelled. The 42-branch comparison retains two stationary-peer
`patch_lost` failures and separates own route completion from opposite-face arrival.
See `docs/handoff/20261028_REPOSITION_Peer_Detour_and_Static_Clearance.md`.


Continuous-2D amendment (2026-10-28, `rounded-scan-v1` and
`effect-options-v9-reposition-scales`): at the user's explicit request, an opt-in
REPOSITION controller now follows the finite own-observed boundary rather than the
normal of the nearest infinite fitted line. `reposition_boundary="rounded_scan"`
queries the closest point on valid observed segments. A trusted silhouette endpoint
or intersection of two observed faces can supply a convex-corner point; shadow/range
censoring and wide endpoint uncertainty do not create a corner. The radial normal
from that point is filtered, with its CW/CCW tangent providing a continuous local
offset arc. Endpoint uncertainty is subtracted from the estimated clearance.
No object pose, hidden polygon, face ID, rectangle dimensions or peer history enters
this controller. When usable finite-boundary evidence is absent it retains the legacy
local fallback and existing safety/observation guards; no unseen corner is invented.

The reference clearance remains 0.10 m, maximum speed 0.18 m/s, actuator/acceleration
limits and derived option timeout (capped at 24 s) remain unchanged. Signed travel
continues to use the PRECEDING commanded boundary tangent, excluding contact and
missing-observation intervals without later backfill. Rounded tangent and reference
geometry are logged explicitly. `legacy_face` remains the default for old callers.

The separately versioned menu `ACTIONS_REPOSITION_V2` uses REPOSITION distances
`{0.2,0.4,0.8,1.2,1.8} m`. PUSH/rotation parameters and WAIT/FOLLOW durations retain
their prior definitions. The new menu has 68 contiguous IDs and an explicit 14-type /
up-to-5-scale mapping; WAIT/FOLLOW still have four scales. The old 66-action table and
legacy 28-action training menu are unchanged. Because IDs shift, archived IDs must
be decoded from each source's frozen menu, never from the new global table. Masks
and type/scale lookup accept the explicit menu. Branch restore rejects mismatched
versions/codebooks. Replay may display multiple versions after verifying each source
manifest hash and requested parameters; no training/checkpoint migration is implied.

The 1.8 m choice follows the current geometry: a 0.18 m-radius robot maintaining
0.10 m clearance needs about `0.5 + 0.25 + pi*0.28 = 1.63 m` from a long-face midpoint
to reach the opposite long face even with ideal rounded geometry. Distance completion
and opposite-face arrival remain separate offline labels. See
`docs/handoff/20261028_REPOSITION_Rounded_Boundary_and_Scales_v2.md` for the matched
60-branch control comparison, the distinct 26-branch new-scale coverage, and retained
failures. Physics, costs, task success, 90-second horizon and real 72-ray sensing remain
unchanged. No new RL or learned-world-model result is claimed.

Continuous-2D amendment (2026-10-27, `effect-push-v8-joint-recovery`): the user explicitly
requested distance-based tolerance and joint reapproach rather than immediate termination
after a brief physical-contact loss. Opt-in `contact_recovery="joint_approach"` applies to
paired measured translations with `post_push="contact_follow"`. The v7 `"abort"` mode
remains available/default for exact historical comparisons. No physical-contact sensor,
reward, success threshold, body geometry or episode horizon is changed.

After engagement, either local controller reports recovery when its own observed
surface-return clearance exceeds 10 mm. The executor pauses both active propulsion
phases on the following control period, preserving each side's prior push/follow stage.
Both use their own scan and existing contact targets to reapproach at normal reference
speed at most 0.04 m/s, without an active push bias. A side already within the 3 mm
resume threshold commands zero normal approach velocity. Both must report clearance
at most 3 mm and normal body speed at most 0.06 m/s for 0.1 s before the next-period
resume signal. The distinct entry/resume thresholds provide hysteresis.

Binary contact loss within the allowed distance does not alone terminate push/follow
in this mode. This is proximity keeping, not proof of continuous physical contact;
actual contact and within-threshold fractions remain separate diagnostics. Active and
follow contact-travel counters still require actual contact. Reapproach normal travel
(valid observed normals, contact not required) and own path are separately recorded;
none is credited as active pushing. The active no-progress clock pauses during recovery,
but the original shared deadline, episode clock and task costs keep running.

Single recovery times out after 2 s; at most three entries are allowed. Geometry loss,
safety exits and other failures remain explicit; failure during recovery stops the
remaining side rather than leaving it indefinitely waiting. The executor uses only
preceding-period request/ready reports. This is an execution testbench, not a B-bit
distributed-policy result. Normal thresholds are triggers, not hard bounds on transient
separation while bodies decelerate. No physical state is frozen or teleported.

The comparison preserves the 24 v7 development scenes, adding two clearly labelled
diagnostic cases with the same declared 4 N / 0.25 s outward motor disturbance in both
variants. These additive bounded force pulses do not change physics or rewards and
are not counted as ordinary random trials. See `tests/test_joint_contact_recovery.py`
and `docs/handoff/20261027_PUSH_v8_Joint_Contact_Recovery.md` for phase accounting,
physical-contact limitations and the 52-branch result. Legacy/solo/rotation paths and
the old RL trainer remain unchanged.

Continuous-2D amendment (2026-10-27, `effect-push-v7-post-follow`): the user requested
a bounded comparison of early-side braking against contact following. The new
`EffectPushConfig(post_push="contact_follow")` is opt-in and enabled by the executor
only for paired measured-travel translations; the default remains `"hold_brake"`.
At the measured threshold, the candidate freezes the active-drive counter and
reference-clock diagnostic, records drive completion, and switches directly to
following without an intermediate stop. Subsequent contact-normal travel and all
own following path length are separately recorded, including final braking.
Thus the scale is an active-drive distance, not total travel across the joint window.

Following uses the same own-scan face/endpoint association and contact reservation.
It retains the endpoint-relative contact role without continuing nominal-direction
correction. Normal reference follows locally measured surface motion plus a bounded
gap correction, with no active propulsion bias. Tangential tracking retains its
ordinary controller cap; following desired speed is bounded at 0.16 m/s, commanded
force norm at 3 N, and commanded normal component at +/-0.75 N. These are command
limits, not guarantees about physical contact impulses or externally imposed speed.
The existing acceleration ramp, collision/safety checks, and no-face-switch rule remain.
Following has explicit 8 s / 0.75 m own-path limits and a 0.25 s continuous contact-loss
limit, in addition to the unchanged shared deadline. Failures are not called completed
following merely because the active-drive distance was reached earlier.

When both active-drive statuses resolve by completion or failure, the executor sends
a joint stop flag on the next control period and both surviving followers brake.
Only preceding-period phase reports are used. No peer private scan/history, hidden
object pose or future physical feedback enters control. These additional execution
handshake flags are explicit; the testbench makes no B-bit distributed-policy claim.
Solo/rotation actions, legacy action IDs, physics, task horizon and reward definitions
are unchanged. See `tests/test_post_push_follow.py` and
`docs/handoff/20261027_PUSH_v7_Post_Completion_Contact_Follow.md` for the 48-branch
paired result, raw-log audit, local perturbations, and adjacent-face failures.

Continuous-2D amendment (2026-10-27, `effect-push-v6-normal-travel`): at the user's
explicit request, the new EffectAction translation menu now defaults to measured
own-body normal travel for both single and dual PUSH. On each control period,
accumulate `(own_position_now - own_position_previous) dot previous_inward_normal`
only if the preceding interval was active push/braking, both endpoint observations
report own object contact, and both face observations are valid and compatible.
Reverse motion subtracts; tangential preparation, contact acquisition and missing
observation intervals are excluded, with no later backfill. The normals and own
global pose are legal local observations; no object truth enters the counter.
The normal follows the same tracked face, so this is a signed path integral rather
than displacement projected onto one initial world direction.

The 0.1–0.5 m parameter is thus a contacted robot-travel contract, not an object
displacement target. It triggers local braking at the measured threshold and requires
settling without falling below that threshold; contact braking travel remains recorded.
The old reference-speed integral remains a diagnostic. No-progress and all existing
safety/geometry exits remain explicit. Boundary progress features for these translations
carry measured travel and target, including after an early side has switched to Hold.
Rotation remains in radians, with its existing angle-effect contract. The legacy 28-action
menu/trainer, physics, costs, reward weights and task success condition are unchanged.

Proportional normal propulsion biases and unscaled tangential feedback remain, but
different measured distances need not finish at the same time: rigid contact couples
robot velocities. Independent exits can substantially change joint rotation. No common
joint effect target or artificial outcome alignment is introduced. Explicit legacy modes
`translation_completion="normal_reference"` and `"object_effect"` preserve old completion
measures; the new benchmark freezes all other settings. Historical comparison CLIs select
their old measure explicitly. See `tests/test_normal_travel.py` and
`docs/handoff/20261027_PUSH_v6_Measured_Normal_Travel.md` for the bounded counterfactual
run, independent saved-log reconstruction and observed tradeoffs. This supersedes the
earlier amendments' default completion measure only for versioned EffectAction translations.

Continuous-2D amendment (2026-10-27, `random-start-oriented-v4.7`): at the user's
explicit request, REPOSITION no longer terminates on an observed face change. It
follows the new locally sensed tangent, retaining the requested distance and original
timeout. Progress is signed boundary-path travel projected onto the **preceding**
valid movement command, rather than onto the initial world direction. Reverse motion
is subtracted; contact/unobserved intervals are excluded and are not backfilled after
reacquisition. Face switches remain diagnostic events, not privileged face IDs.
The micro/grid tables, rewards, physical task horizon, action IDs, and observations
are unchanged. Counterfactual motion accounting and CW/CCW corner integration tests
are in `tests/test_reposition_face_switch.py`. Historical run9 results remain tied to
their original controller; old weights/branch replays must be labelled explicitly.
The shorter, direction-conditioned PUSH is a separately versioned, opt-in extension
described below; historical action IDs are not reinterpreted.

Continuous-2D amendment (2026-10-27, `effect-push-v1`): at the user's explicit request,
the existing simulator accepts an explicit versioned action table and skill factory.
The new menu contains eight world-frame translation directions at five distances
`{0.1,0.2,0.3,0.4,0.5} m`, two rotation signs at five angles of the same numeric values
in **radians**, and the 16 existing non-push actions. Its 66 IDs are an explicit ragged
14-type / up-to-5-scale codebook, not the legacy `type*4+scale` encoding. Branch restore
checks the action-table parameter hash. The legacy menu remains the default.

Availability and control use only the robot's own scan, own pose/history and a declared
public rectangular-shape prior (default 1.0 by 0.25 m). Fitted face normals are compared
with nominal world directions within 30 degrees; this is a **local geometric mask**,
not a shared public candidate set or a guarantee about hidden friction. Any future
distributed policy must account for communicating private mask information under the
frozen message protocol. No hidden state or teammate private history enters control.

The controller associates one observed face throughout an option, steers a bounded,
rate-limited contact point and brakes against estimated object effect. Its reference
is the first valid option observation, including preparation motion. Unobservable
tangential motion is not credited; ambiguous geometry and expired partial observations
produce explicit exits. An unavailable PUSH does not silently become REPOSITION.
The feedback comparison toggles translation contact-point correction only: both groups
retain velocity, gap and effect-endpoint feedback. Rotation allows accompanying
translation and does not promise a pure rotation. The engine, reward weights, task
success condition and 90-second episode horizon remain unchanged; option time budgets
are explicit controller parameters. Tests: `tests/test_effect_push.py`; frozen development
cases, manifests, raw logs and offline replay: `experiments/option_testbench/effect_push_eval.py`.
This extension has not been integrated into the run9 trainer or evaluated by new RL runs.

Continuous-2D amendment (2026-10-27, `effect-push-v2-endpoints`): the user's requested
endpoint-based contact controller is explicitly enabled by
`EffectPushConfig(endpoint_tracking=True)`. The default remains the v1 control path
pending review of mixed development results. The action parameters/IDs, 72 real lidar
rays, physics, costs and task horizon are unchanged. Each robot tracks its own face
endpoints within an option; no teammate endpoint estimate is shared and no hidden
face ID is used. Tracking resets between options or on an incompatible face/clock.

An endpoint observation is an intersection of two fitted visible faces, a physical
edge interval bounded by adjacent real rays, or explicitly censored by range/shadow.
Actual collinear returns omitted by a fitter split may extend measured support.
Two measured intervals are intersected with the public length constraint before a
causal position/velocity update; a measurement interval is not treated as its midpoint.
One fresh endpoint can anchor an endpoint-relative contact distance using the public
length prior. The other endpoint remains inferred/unobserved. Derived points are
confined to the fitted line and do not increase real support count or confidence.

Tracking has a 0.25 s prediction age limit. A predicted endpoint alone cannot credit
full translational effect or establish completion. Contact targets respect raw shadow
boundaries; raw scans/contact feedback still drive safety. The endpoint controller
also sets an initial rotation destination before motion and approaches a safe staging
line concurrently with tangential alignment; physical velocity/force ramps and original
option budgets still apply. Thus the paired comparison evaluates this controller
change together, not an isolated filter-only ablation. Regression tests are in
`tests/test_endpoint_tracking.py`; bounded branch comparison is
`effect_push_eval.py --compare-tracking`. Historical v1 outputs remain untouched.

Continuous-2D amendment (2026-10-27, `effect-push-v3-approach`): at the user's explicit
request, an opt-in preparation profile (`fast_approach=True`) raises the maximum
tangential approach reference from 0.08 to 0.20 m/s in free clearance. Its strength
blends continuously with the initial sensed tangential error over 0.05–0.15 m;
small initial corrections retain the previous controller. Target-distance braking
and own-velocity damping reduce speed near the destination. Contact/low-clearance
motion retains the 0.08 m/s cap, and a fast approach must settle laterally before
committing PUSH. Existing acceleration ramps, force limits, staging clearance,
physical model, action scales, rewards and option/episode budgets remain unchanged.
This amendment does not retune the in-contact direction or rotation controller.
`--compare-approach` freezes endpoint tracking and contact correction ON in both
variants; its panel includes initial robot-position offsets of +/-1 cm and, for
the two-visible-face case, heading offsets of +/-1 degree. All exits are retained
by `approach_stability_audit.py`. Tests: `tests/test_effect_push_approach.py`.
This is a bounded development check, not evidence of neural fitting or RL learning.

Continuous-2D amendment (2026-10-27, `effect-push-v4-joint-reference`): following
the user's revised instruction, different nominal scales remain legal and are not
merged into an object goal. For two translation EffectActions, the default
`joint_execution="proportional_normal_v1"` computes a shared nominal active duration
`T=max(d_i/v_i)` and normal propulsion biases `d_i/T`. Scale in this paired mode is
an observed-contact NORMAL REFERENCE budget in metres, not a promised object
displacement. Each robot accumulates its own bias times control dt only when it
observes contact during active execution. No physical displacement is credited by
this counter. Tangential contact-point control is not multiplied by this ratio.

The new paired protocol stages both robots, then establishes contact at an unscaled
gentle normal speed before releasing the reference phase. A local surface-motion
estimate adds contact-following feedforward; gap recovery is retained. Total normal
reference is bounded by the existing configured speed, so the propulsion-bias ratio
does not imply the same ratio of actual velocities or forces. Each controller brakes
after its own reference budget and can finish as `normal_reference_complete` using
its own settling feedback; it does not wait for the moving shared object to settle.
Braking has the existing 0.8 s cap with an explicit timeout. A finished side uses the
existing physical HOLD_BRAKE while its peer continues; no pose is frozen.

All new dual EffectActions share the longer bounded option deadline (quantised to
the control grid), so a short nominal scale does not expire merely while its peer
prepares. A pre-release failure signals `peer_prepare_failed` next period; joint
preparation without measurable progress for 3 s signals `joint_prepare_stalled`.
Failure after release does not silently cancel or restart the peer. Readiness and
phase events are recorded execution-kernel signals; no peer scan/history or object
truth enters either local controller. This testbench does not establish deployment
message-bit accounting for these low-level handshakes.

Rotation-containing pairs retain independent angle-effect thresholds and use
`local_effect_reached`; radians are NOT passed through the metre-reference budget
or proportional translation clock. Final joint pose is not scored as fulfillment
of an earlier local exit. Single PUSH and the legacy 28-action controller retain
their prior contracts. `legacy_v3` is an explicit reproducibility flag; historical
comparison CLI modes select it to preserve their contracts. The physics, reward,
task success, actuator limits and 90 s horizon are unchanged. Tests:
`tests/test_joint_execution.py`; paired replay: `--compare-joint-execution`; independent
audit: `joint_execution_audit.py`. New logs include separate nominal budgets, real
object motion, both exit reasons and phase events for later model learning.

Continuous-2D amendment (2026-10-27, `effect-push-v5-contact-regions`): an opt-in
`same_face_allocation=True` controller reserves non-overlapping contact regions
for visible same-face translation pairs. The hypothesis uses only the robot's
own fitted face, own pose and teammate lidar return; there is no true face ID or
access to the peer's private target. Tangential order is fixed for the option.
The preferred contact point is projected onto the intersection of the physical
face prior/end inset, currently observable support, the assigned side of an
estimated robot-pair midpoint, and the observed peer's clearance boundary.
Independent object-centre estimates therefore do not define the role separator.
Direction/lateral feedback stays enabled within the available region; physical
motion remains an outcome, not a prescribed joint vector goal.

Circular robot radii stay 0.18 m, so geometric touching is at 0.36 m centre distance.
The existing 0.02 m measured-gap safety abort is unchanged. Only a locally inferred
same-face pair can reduce its soft teammate buffer to 0.04 m at low closure rate;
contact targets include a further 0.01 m pad (nominal 0.41 m separation). With estimated
closing speed c, the soft buffer is min(0.12, 0.04 + 0.1*c + c*c/(2*A_REF_MAX)).
First/aged peer observations retain the original 0.12 m buffer. Static avoidance,
other-face avoidance, force and acceleration limits, scales, reference budgets,
reward and task horizon are unchanged. Brief peer loss uses bounded local prediction
and increased reservation clearance; loss beyond 0.25 s exits explicitly. Insufficient
face extent/visible interval returns `same_face_no_room`/`same_face_lane_unavailable`.
The v4 lifecycle handles the resulting exits; there is no silent role swap or reposition.

Same-face detection is a local geometric hypothesis, not a formal guarantee for all
occlusion/contact states. The flag defaults off for review; old paths remain reproducible.
Tests: `tests/test_same_face_allocation.py`; paired testbench:
`--compare-contact-allocation`; independent audit: `same_face_allocation_audit.py`.

Status: **frozen for the first round**. Changes require a note in this file and a
matching test update. This is the `spec.md` required by
`docs/CoImagine_2D_Agent_执行提示_v1.md` §二 阶段0.

Scope: 2D only. Stage 1 = micro finite-state diagnostics M0–M3 with exact tabular
models. Stage 2 = 6×6 grid cooperative pushing. No learning, no neural models,
no general-model API in this round.

---

## 1. State and information

### 1.1 True state

```
s = (pose_self, x_object, c, theta)
```

- `pose_self` — each robot's **own pose**. Own global localization is allowed: a
  robot knows where it is. This is the only global quantity a robot is handed.
- `x_object`, `c` — the object's pose and the current contact configuration. These
  are **not public**. A robot may only estimate them from its own egocentric sensing
  `o_i`. There is **no broadcast ground-truth object coordinate**, and no real-time
  global reconstruction of the object is assumed.
- **Static environment (decided).** Walls, obstacles and the goal region are a
  **prior map known to every robot**; they are public and do not have to be
  discovered. Only the *movable object* must be perceived on site. This keeps the
  problem at "private contact belief + finite messages" instead of coupling it to
  SLAM.
- `theta` — **hidden physical type**, stationary within an episode. In the micro
  family `theta ∈ {HEAVY, SLIP}` (or irrelevant for M0). In the grid
  `theta = (r, c_mode, tau)` with resistance `r ∈ {1,2,3}`, constraint mode
  `c_mode ∈ {free, constrained}` and local traction `tau`.

`theta` is never public. The simulator owns `s`.

**Micro-stage caveat.** The Stage 1 micro family deliberately abstracts perception
away: the contact configuration is a single public symbol
`x ∈ {CONTACT, PREPARED, GOAL}` and sensing is reduced to a private binary feedback.
The micro stage therefore tests protocol and oracle logic only and **does not satisfy
the egocentric-sensing constraint**. That constraint binds Stage 2 (grid) and every
later 3D experiment; see section 9.

### 1.2 Public context `C_t`

```
C_t = (task specification, own pose, y_pub[0..t], a[0..t-1], m_A[0..t-1],
       m_B[0..t-1], candidate set + stable IDs, protocol version, timestamps)
```

- `task specification` — goal region, robot/skill model, and the static scene
  *class*. Together with the **prior static map** (walls / obstacles / goal region,
  decided public) it is known a priori. It contains no current object pose and no
  live occupancy map.
- `y_pub` — coarse motion event of the object (`moved` / `stalled`). **Decision: this
  stays public.** Both robots are in contact with the same object and can detect the
  event through their own sensing; it is an **event**, not a coordinate, so keeping
  it public does not leak the object pose.
- `a[t]` — the committed joint action at boundary `t`.
- `m_i[t]` — robot `i`'s message at boundary `t`.
- Candidate IDs are assigned by a deterministic rule from the **frozen candidate
  template space**, not from the current object estimate, so that both sides agree
  on IDs while scoring them with different private beliefs (section 9).

### 1.3 Private history `h_i,t`

```
h_i,t = (o_i[0..t])
```

`o_i` is robot `i`'s **own** local contact/drive feedback (grid: own driving
response, local slip; micro: own binary `loaded` / `slipping` signal). `h_i` is never
shared except through `m_i`.

### 1.4 Joint observation model

The complete joint likelihood is published and queryable:

```
O( (o_A, o_B, y_pub) | s, a, s' )
```

- It is stored as an **explicit joint table**. Independence of `o_A` and `o_B` is
  **not** assumed; the micro instances deliberately use correlated tables.
- `y_pub` is a deterministic function of `(x, x')` in the micro family (moved iff the
  object advanced) and is stored in the same table so that all methods consume one
  object.
- Impossible observations have probability exactly `0`; the Bayes filter must handle
  a zero-likelihood event without producing `NaN` (it returns an explicit
  `ImpossibleObservation` status for that branch).

---

## 2. Timing of one decision boundary

Executed in exactly this order; every step is testable.

1. Environment draws `(o_A,t, o_B,t, y_pub,t) ~ O(· | s_{t-1}, a_{t-1}, s_t)` and gives
   `o_i,t` privately to robot `i`, `y_pub,t` to both.
2. Each robot updates its belief using **only** `h_i,t` and `C_t`.
3. Both sides rebuild the identical candidate set and candidate IDs from public
   information only.
4. Each side sends **one** message `m_i,t = mu_i(h_i,t, C_t)` with at most `B` bits.
   A message may not depend on the teammate's not-yet-received same-round message.
5. Both sides append to the common record; one deterministic public rule
   `pi(C_t, m_A,t, m_B,t)` selects the joint option. Candidate order, RNG seed and
   tie-break are fixed and identical on both sides.
6. Execute `a_t`, charge cost, generate `s_{t+1}` and continue.

`t = 0`: the environment **does** distribute an initial local observation
`(o_A,0, o_B,0, y_pub,0)`. A probe is therefore only valuable through the *additional*
information it produces, never as the first way to see anything.

---

## 3. Skills and actions

A joint action is `a = (a_A, a_B)`; every robot acts at every boundary.

Micro family:

| action | meaning | advance? | notes |
|---|---|---|---|
| `CO_PUSH` | both robots push the object together | depends on `theta`/`x` | cooperation action |
| `RELEASE_B` | B performs a lateral reposition/release | changes `x` | preparation, not information |
| `PROBE` | A performs one short diagnostic push while B holds | no | real short skill; displacement/cost/observation all defined by `T` and `O` |
| `HOLD` | both wait | no | zero cost, used for the too-expensive-probe negative control |

Grid family (stage 2): single push, same-direction cooperative push, one-robot-hold +
other-repositions, short diagnostic push, wait. A robot must occupy a legal contact
cell; the target region must be free; repositioning follows a collision-free grid path
and is charged its path duration (no free teleport). Coalition requirement and the
hidden resistance `r` are read from the transition table; the candidate generator may
only use public geometry.

**Prohibition.** No action may be added whose only role is to reveal truth, no
information-gain reward, no "enabling" reward. Probe and preparation values must fall
out of `T`, `O` and the unified cost.

---

## 4. Cost and termination

- `C(s, a, s')` is the per-transition cost. Units and values are fixed in the instance
  configuration. In the micro family: `CO_PUSH = 1.0`, `RELEASE_B = 1.0`,
  `PROBE = c_probe` (reported per instance), `HOLD = 0.0`.
- Episode cost = cumulative transition cost + `c_fail` if the horizon `H` is exhausted
  without reaching `GOAL`. `c_fail` is explicit and positive; **a failure must not be
  cheap by exiting early**.
- Success = `x_H == GOAL`. Reported separately: success rate, executed steps, invalid
  pushes, probe count, bits sent, search nodes, wall-clock time.
- Comparison across methods uses the same instances, cost and horizon.

---

## 5. Beliefs and the public coordination state

- Robot belief: `b_i,t(s) = P(s_t = s | h_i,t, C_t)` via exact Bayes under the shared
  generative model.
- **Public coordination state**, needed because future messages depend on private
  histories:
  `mu_t(s, h_A, h_B) = P(s_t, h_A,t, h_B,t | C_t)`.
  Micro instances enumerate the private histories exactly. Caching only `P(s | C)` is
  not sufficient and must not be used to predict future messages.
- Two robots' posteriors must **never** be multiplied as if independent. Where a
  joint quantity is needed it comes from `mu_t`, which is built from the joint
  observation table.

---

## 6. Messages and bits

- Message alphabet is the nested set `M_B = {0, ..., 2^B - 1}`; `B = 0, 1, 2` per
  robot per boundary.
- Every message is serialised with an inspectable discrete codebook. The bit count
  reported is the real codebook payload, including candidate IDs and missing masks.
  A Python float array is never reported as "1 bit".
- A micro instance in which one bit fully carries a binary local observation is a
  normal, reportable outcome and is not suppressed.

---

## 7. Oracles

| oracle | information | solution concept | role |
|---|---|---|---|
| `O_full` | true `s`, true model, horizon `H` | exact backward induction | full-information lower bound |
| `O_central` | both robots' observation histories, true model | exact POMDP / expectimax over the joint information state | complete-observation-sharing reference |
| `O_codec` | a fixed message encoding rule `mu`, true model | exact planning over the public policy whose future messages pass through the same `mu` | isolates message-representation loss |
| `O_B` | same `B`-bit budget and same protocol class as deployment | enumerate local message rules and public action rules | fair optimal reference for tiny instances |
| `O_replan_no_message_branch` | same codec and horizon, but lookahead branches only on the public observation | exact closed-loop expectation of a message-blind controller that replans after execution | **C2 control**: isolates the value of anticipating the next round's message |
| `TrueModelLimitedSearch` | true model, node/depth limited | e.g. myopic depth-1 or node-capped expectimax | computation-limit diagnostic, **not** optimal |

Every result reports: actual horizon, strategy class, whether the codec is fixed,
nodes explored, wall-clock time, and an explicit `incomplete` status if a budget was
exhausted. `O_B` is restricted to the strategy class "one synchronous message round,
then public commit"; it is **not** the unrestricted Dec-POMDP optimum.

When instance, cost and horizon match, strategy classes are nested, and the central
method can simulate communication, verify:

```
J_full* <= J_central* <= J_B* <= J_codec* <= J_replan_no_message_branch*
```

The last inequality is the **C2 claim** and only has to hold where private evidence is
produced after the decision and used later. It must be an equality when the task has
no decision-relevant future evidence, and it must be an equality at `B = 0` (there is
no message to anticipate). `O_replan_no_message_branch` is a controller, not an
optimum; its `exact` flag is `False` even though its expectation is computed exactly.

In the first delivery `O_B` is exact for a **single decision boundary** (`H = 1`).
Multi-step exact `O_B` is a Dec-POMDP and is reported as `incomplete` with the
complexity reason; a separately named restricted class may be reported but must not be
labelled `O_B`.

---

## 8. Micro instances M0–M3

Common micro tables: `x ∈ {CONTACT, PREPARED, GOAL}`, `theta ∈ {HEAVY, SLIP}`,
actions `{CO_PUSH, RELEASE_B, PROBE, HOLD}`, horizon `H = 3`, local observables
`o_i ∈ {loaded, slipping}`, public observable `y_pub ∈ {moved, stalled}`.
Exact enumeration size is reported per instance by the runner.

- **M0 — public information sufficient.** The public observation identifies the
  recovery action; private local signals carry no decision-relevant information about
  `theta`. Expectation: communication gives no gain (`J_B = J_codec = J_central`).
- **M1 — stall-cause ambiguity.** Public `stalled` is identical for both causes; the
  correlated joint local table makes `o_A` and `o_B` partial, complementary evidence.
  `HEAVY` succeeds with `CO_PUSH` from `CONTACT`; `SLIP` requires `RELEASE_B` first.
  Expectation: with `B = 0` the team cannot beat the prior-optimal action; a short
  message changes the committed action. **This is the C1 instance**: the evidence is
  used in the current round, so a compact posterior message is already sufficient.
- **M2 — costly identification.** As M1, but the initial local observation is weakly
  informative and `PROBE` produces a sharp observation at cost `c_probe`. The probe's
  value only materialises one boundary later, after its result has been *messaged*.
  **This is the C2 instance**: the exact `O_codec` reaches `O_central`, while the
  same codec without a future-message branch is no better than `B = 0`. Two
  negative controls:
  - *M2-uninformative*: the probe observation is independent of `theta`; the exact
    policy must not probe.
  - *M2-expensive*: `c_probe` is large; the exact policy must not probe.
- **M3 — preparation changes the contact mode** (optional but included). `theta` is
  effectively known; `CO_PUSH` from `CONTACT` cannot succeed and `RELEASE_B` is
  physically required to reach `GOAL`. This separates physical enabling from
  information gathering: the value of `RELEASE_B` is not an information term.

`M0`, `M1`, `M2-uninformative`, `M2-expensive`, `M3` are all constructed from the same
engine; no instance-specific reward term is allowed.

---

## 9. Stage 2 grid contract (preliminary, implemented later)

- 6×6 map, two robots, one 2×2 object, obstacles, target region; translation only
  (a rotation interface is retained but not required).
- **Sensing (binding constraint).** Each robot has its own global localization
  (its own pose is known to itself) and an **egocentric, occlusion-aware ray lidar**
  (or an egocentric occupancy patch) of radius `R`. The **prior static map is
  known** (walls / obstacles / goal region); only the **movable object** is sensed,
  never broadcast, and no real-time global reconstruction of the object is assumed.
  Step 2 must reuse the archived
  `_old/gate_a/src/coimagine/envs/grid/environment.py::visible_cells_from_pose`
  primitive (occlusion-aware Bresenham ray casting, `observation_radius`), with
  attribution, and add sensor noise. The object's planar/global *coordinates* are
  not part of `C_t`.
- Public: task specification, prior static map, own pose, committed actions,
  protocol, exchanged messages, and the coarse `moved`/`stalled` event. Private: the
  raw scan and the robot's own drive/slip feedback.
- Candidate IDs stay public: they are enumerated from the frozen task/candidate
  template space, not instantiated from the current object estimate, so both sides
  agree on IDs while scoring them with different private beliefs. A shared,
  quantized object-pose estimate may be sent, but only through the message budget.
- Hidden resistance `r ∈ {1,2,3}`; a single robot's push capacity is 1, same-direction
  cooperative capacity is 2.
- Hidden `r` must not prune candidates. Geometric legality is evaluated against each
  robot's own estimate, not against simulator truth.
- **Diagnostic separation.** Object-localization error must be reported separately
  from hidden-physics belief error, so a method cannot fail because it cannot see
  rather than because of its message or planning.
- Large grids use `TrueModelLimitedSearch` and are never labelled "exact optimal".

---

## 10. Configuration and reproducibility

- Instances, costs and message budgets are declared in code/config, not tuned on test
  outcomes.
- Seeds are fixed and recorded. Randomness that changes with the action sequence is
  not described as a fully paired observation trajectory.
- Results record the git revision, configuration, seeds and the exact command.

### 2026-10-28 isolated option efficiency candidate

`random_start.efficient_profile` explicitly selects
`effect-options-v11-efficient-options`: the first 60 PUSH/REPOSITION entries
are unchanged, WAIT has a single 2 s entry, and FOLLOW has a single peer-PUSH-bound
entry (62 total). WAIT continues braking/holding while an active peer finishes.
Historical 65/68-entry menus and the run10 RL interface remain separate; old
checkpoints must not be interpreted with the new indices. The joint mask keeps
FOLLOW paired with PUSH while retaining PUSH+REPOSITION and dual PUSH.

The profile retains 120 Hz physics, 20 Hz control and 72-ray body-fixed sensing,
normal robot-travel translation completion, force/acceleration limits, safety
thresholds and task costs. CPU NumPy batch ray intersection and sampling without
full per-period debug materialization preserve these quantities. The native
`step_control(minimal=True)` path omits the full observation and debug contacts;
finite-state, bounds, penetration and success-hold checks still run. The default
full native API remains available.

The explicit profile changes translation preparation to a continuous standoff
adjustment and adds contact tangent damping. Rotation retains the prior motion
parameters. A phase-specific 3 s meaningful-progress guard uses own observed
geometry and feedback, returns named failure reasons on plateaus, and does not
claim task completion. Local quality gates, rejected variants and CPU benchmark
scope are recorded in `handoff/20261028_Option_Efficiency_v2.md`; they are not an
RL generalization result or a replacement for the frozen research protocol.
