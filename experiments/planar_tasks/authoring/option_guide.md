# Public Option behavior guide v1

This is shared interface knowledge for every LLM backend, for both free code-as-policy
and fixed Option search. It describes the frozen `option-v2-yield-signed-push`
controllers used by the planar baseline. It contains no task solutions, test results,
hidden scene state or privileged runtime measurements. `contract.json` is the exact
action codebook; each observation's `legal_pairs` is authoritative for the next command.

## 1. Command, time and information

Return exactly `{"a0": integer, "a1": integer, "ticks": integer}`. The two IDs select
one controller per robot, executed together for 1–40 control ticks (0.05 s each).
The command window includes switching, braking, looking, approaching, aligning,
contact acquisition, teammate waiting and recovery, as well as effective pushing.
It is NOT a guaranteed interval of object motion. The runner advances the requested
window unless the episode terminates; skill failure does not itself end the window.
Afterwards the policy receives new legal observations and can choose again.

PUSH and ROTATE here are persistent controllers, not fixed-distance or fixed-angle
macros. Finishing a command window does not complete or reset a skill. A one-tick
command is a legal 0.05 s interval, not an instantaneous acquisition of contact.
World x points right, y up; angles are radians, positive CCW. Visible patch normals
point INTO the surface; the outward normal used for boundary following is the opposite.

The policy has shared legal robot odometry, scans, observed patches, contact readings,
goal/arena/time and the execution fields below. It does not receive controller objects,
complete contact-allocation state, object pose truth, full maps or hidden dynamics.
Knowing the algorithm does not reveal its current hidden state. Infer outcomes from
legal feedback/history and budgeted experiments; do not assume model truth is reality.

## 2. Menu and parameters

| Family | Canonical IDs | Parameter meaning |
|---|---|---|
| PUSH | 0–31 | Eight world directions E, NE, N, NW, W, SW, S, SE; four commanded speed levels 0.10/0.20/0.30/0.40 m/s per direction. ID = 4 × direction index + speed index. |
| ROTATE_CCW | 32, 33, 34 | Positive rotation intent; normal-speed levels 0.02/0.04/0.06 m/s. |
| ROTATE_CW | 35, 36, 37 | Negative rotation intent; same normal-speed levels. |
| REPOSITION_CCW / CW | 40 / 43 | Boundary-following robot motion at nominal tangent speed 0.30 m/s. |
| BRAKE_HOLD | 44 | Active braking toward zero robot velocity. |
| YIELD_CLEARANCE | 46 | Local threat-aware clearance controller; nominal candidate speeds up to 0.40 m/s. |

IDs 38/39/41/42 are historical reposition aliases, not extra legal choices. ID 45
(SUPPORT_FOLLOW_PUSH) is not exposed in this CAP action language.

PUSH speed controls robot contact-position/velocity drive, not measured object speed.
In the current allocated-contact branch it sets an inward position-reference offset
v/6 and caps the resulting position-feedback velocity at v. Other branches use normal
drive with bounded tangential corrections. ROTATE speed is normal contact drive,
NOT angular velocity, rotation angle or a guaranteed object speed. Approach, contact
engagement, acceleration, force limits, friction and collision constraints also matter.

Legal joint structure: two translational PUSH actions; two REPOSITION actions; or
a PUSH/REPOSITION/BRAKE/YIELD combination containing BRAKE or YIELD. Rotation is
allowed only as two ROTATE actions with the same sign (their speeds can differ).
Translational directions need not agree; legality does not guarantee useful net force.
Always check the actual `legal_pairs`: current sensor-derived availability adds masks.

## 3. PUSH: acquire a contact, then keep driving

The current translation implementation is `UnknownShapePush`, installed by the
environment factory, not merely the older generic `ContinuousPush` implementation.
It selects/tracks contact points from visible finite surface segments. Direction
availability requires a visible inward normal within 35 degrees of the requested
world direction. If no compatible surface remains, it does not silently substitute
an opposite or orthogonal push; it can wait in `find` with `direction_unavailable`.

Contact selection and approach are internal. A selected contact is tracked with the
moving observed segment; it is not a fixed world waypoint. Allocation can use both
robots' visible geometry to separate their contact targets. A PUSH direction change
can invalidate/recompute that allocation even when the skill object is retained.
Changing speed alone is not a request to select a new contact point.

Possible local stages include `find`, `view_insufficient`, `standoff`, `final`,
`approach`, `align`, `touch`, `orient`, `ready`, `wait`, `wait_peer_contact`, `push`,
`retreat`, `shift`. These are conditional branches, NOT one mandatory linear sequence:

- `standoff`/`final`/`approach`: navigate towards an observed contact via a local
  clearance band. `align`/`orient`: align position or heading; `touch`: establish contact.
- With two translational PUSH skills, a contacted robot can wait for its teammate.
  The paired controller uses fitted gap and heading checks and a contact-readiness
  latch, not only the instantaneous binary contact sensor. One-sided preparation
  need not produce object progress; a local `push` label alone does not certify
  that the full pair is driving effectively.
- `view_insufficient`/`find`/`wait`: required geometry or allocation is unavailable,
  or the local contact process is pausing. Zero progress is not automatically a
  terminal skill failure. Single-robot branches may retreat and shift to retry contact.
- Once driving, local feedback maintains contact and bounded lateral correction.
  Neither straight object translation nor eventual goal arrival is guaranteed.

Two contacting robots can still oppose motion, block each other or be constrained
by obstacles. The option does not plan a global object route or promise to clear walls.
`PUSH_E(v, 40)` does not mean “move the object east by 2v metres.”

## 4. ROTATE: offset contacts and coordinated drive

Rotation retains the generic ContinuousPush rotation controller. It estimates a
visible face and selects a contact offset along it according to rotation sign,
clipped to supported visible geometry. It approaches and aligns internally; the
author does not specify contact points directly through this action interface.

Its common local stages are `approach`, `ready`, `push`, `recover`; wrapping waits,
switching and blocked states can also appear. The joint coordinator distinguishes
APPROACH, READY, ENGAGE, PUSH, RECOVER and EXIT internally. It waits for readiness,
establishes contact gently, then enables coordinated drive. A robot already touching
may hold while its peer engages. Contact loss can initiate bounded joint recovery;
peer failure or lost perception can interrupt the pair. These internal joint phases
are explanatory knowledge, not additional fields exposed to the policy.

Rotation has no requested-angle completion in this continuous profile. It can also
translate the object, fail to turn against a constraint, or lose its visible face.
Repeated same-sign commands continue its history. Changing only rotation speed does
not constitute a fresh attempt and will not clear a latched failed skill.

## 5. REPOSITION: move the robot around the observed boundary

CW/CCW means the robot circulates around the object in the world plane, not that it
rotates the object. The controller follows a locally reconstructed rounded boundary,
adjusts separation, handles visible corners, and can take an outer detour around
the teammate. It has `find`, `align`, `move` stages and can become blocked/finished
by visibility loss, safety or progress checks. Nominal speed is not guaranteed travel.
It is intended as non-contact robot repositioning, but incidental contact can occur.
It has no destination point, requested angle, global route or guaranteed reachable side.

## 6. BRAKE and YIELD are different behaviors

BRAKE drives an acceleration-limited reference toward zero robot velocity and applies
feedback force. It does not disable the robot, freeze the scene or move out of the way.
A braking robot may remain an obstacle or resist object motion. It normally reports
`stage=brake`, `done=false` while held.

YIELD uses local scans, shared teammate velocity, estimated surface motion and public
arena boundaries to identify nearby or approaching threats. It samples local velocities
over a short 0.8 s horizon with acceleration limits and a persistent sideways preference.
It has no task-goal route. `yield_idle` means no active clearance response; it commands
zero desired velocity. `yield_escape` means a clearance response is active.
`yield_blocked` means no satisfactory escape was found; the controller may still apply
the least-worsening local motion. “Blocked” here does NOT mean zero force or `done=true`.
YIELD normally reports `done=false` in all three stages. It cannot guarantee global
collision avoidance or relocation to an arbitrary useful position.

Entering YIELD commits immediately while preserving velocity-reference continuity;
it does not first wait for a pushed robot to stop. Exiting YIELD follows ordinary
mode-switch braking. Repeating YIELD preserves its local history.

## 7. Continuation, switching and failure latches

| Request relative to current controller | Actual lifecycle |
|---|---|
| Same action, new ticks | Continue existing controller and history, including failure latches. |
| PUSH speed or direction change while not done | Update intent/speed in the same translation skill; allocation may change with direction. No guaranteed full reset. |
| Same-sign ROTATE speed change | Update speed in the same rotation skill. |
| REPOSITION direction change while not done | Change boundary-following direction in the same skill. |
| Done translation skill + different PUSH direction, or done reposition skill + different direction | Request a fresh skill through the switch/brake path. Speed-only changes do not trigger this retry. |
| Change family, or change ROTATE sign | Request a new skill; normally brake before committing it. |
| Enter YIELD | Immediate commit with reference continuity, as described above. |

Ordinary switching waits for sufficiently low robot speed AND a settled brake
reference. During that delay feedback can show `switching`; a short command may
be spent entirely braking. Another request while a switch is pending replaces the
pending target; it does not create a queue of skills. A committed new skill resets
its local lifecycle, but not physical state or the environment's entire shared history.

When a skill is `done`, repeating it does not automatically restart it; the wrapper
brakes and reports `blocked` with the saved exit reason. This is separate from an
episode ending and from temporary waits with `done=false`. No global planning or
automatic high-level recovery is supplied by these transitions.

## 8. Reading legal execution feedback

Only these execution fields are exported; missing/not-applicable values can be null.

| Field | Interpretation |
|---|---|
| `stage` | Last local skill/wrapper stage; includes `switching`, `perception_wait`, `peer_wait`, `blocked` as applicable. Not a task-success indicator. |
| `done` / `exit_reason` | Local lifecycle termination and reason, not object-at-goal success. Examples include visibility loss, safety abort, no progress and joint recovery limits; exact strings depend on the skill. |
| `pause_reason` | Temporary wrapper wait such as `perception_wait`, `peer_perception_wait`, `peer_push_switching`, `peer_push_failed`. It can coexist with failure; inspect `done`. |
| `direction_valid` / `direction_unavailable` | Current translational direction/visible allocation assessment when implemented. Not global task feasibility; null is not false. |
| `desired_velocity` | Local requested robot velocity before reference filtering/limits; not measured object velocity. |
| `reference_velocity` | Filtered robot velocity reference; not measured motion or force. |
| `force` | Controller-reported force; not net object force or a full contact wrench. Coordination can further gate application. |

Use robot contact readings and successive legal observations alongside these fields.
The public API does not export all internal debug fields such as contact target q,
pair phase, release latch or yield candidate list. Do not read their absence as zero.
Repeated legal commands may produce little motion because of preparation, waiting,
opposing forces, limits or geometry; distinguish those using available evidence.

## 9. Prediction and what still needs physical validation

When enabled, the independent predictor uses the same nominal option implementation
but starts from YOUR supplied synthetic geometry with cold controller/sensor history.
Identical poses do not recreate an executing skill's contact allocation, wait state or
failure latch. Continuing a synthetic handle preserves that synthetic history only.
Predicted state is not privileged access to execution state. Geometry inferred from
partial observations can omit an obstacle or misplace a contact surface.

Interface knowledge tells you what a controller attempts, how time is consumed and
how commands continue. It does not specify actual displacement, contact feasibility,
delay to engagement or success probability in a particular configuration. Those
remain hypotheses to test within the declared physical/prediction budget. Compare
complete commands at their specified durations and distinguish preparation from
effective motion. No task-specific action sequence is prescribed here.
