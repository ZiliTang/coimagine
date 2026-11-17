# Multi-object Option supplement: partial sensing

Read with `../authoring/option_guide.md`. This is experimental interface knowledge,
not a frozen authoring release. It describes the new environment adapter only.

## Observation

The adapter uses centralized sharing of the two robots' local 72-ray scans
(range 5.00 m, from `random_start.spec.LIDAR_RANGE`), own/teammate odometry, goal and execution feedback. There is no
global observation mode. Each ray returns nearest-hit range/category and an
opaque entity ID for a visible movable body. A fully occluded entity is absent;
the client can maintain earlier sightings in its history. IDs do not grant a
pose/shape/mass query. Static shapes can occlude movable bodies as usual.

`entities` contains ray-derived point samples, not complete shapes or centroids.
`entity_legal_pairs[id]` is the sensor-derived joint action menu for that target;
`legal_pairs` is the union plus `global_legal_pairs` (target-free MOVE/BRAKE).
Check the appropriate target-specific or target-free menu.
The menu is computed from the same visible surface patches available to skills,
not from mass, reachability truth or hidden contact mode. Inclusion does not
certify execution feasibility.

The public scan preserves all movable categories and instance IDs. A legacy
skill receives a target-filtered view: target returns produce object patches;
other entities remain at their measured ranges as non-target occupied surfaces.
For that legacy view, `CATEGORY_STATIC` and `contacting_static` mean non-target
occupancy/contact; they do not establish that another body is immovable. The
underlying rigid bodies remain movable and exchange contact forces.

Public `patch`/`patches` and contact flags refer to the currently active target.
The agent knows its requested commands and receives explicit handoff feedback.
The public goal always names the original transport object, independently of
which object is currently manipulated.

`active_target` publicly acknowledges the committed target selected through the
command interface; it does not expose contact allocations, motion-estimator
state or pending controller internals. Nominal prediction may preserve this
ownership while cold-starting the rest of the controller. Synthetic feedback
is marked `source=nominal_prediction`; its observations are hypothetical.

## Commands and handoff

```
{"target_id": "observed-id", "a0": 2, "a1": 2, "ticks": 20}
```

Manipulation uses canonical V2 Option IDs and 1..40 ticks, each 0.05 s. Both robots manipulate
the selected entity; one may brake/yield while the other operates. A visible
surface is required to select an entity. Global BRAKE is always available:

```
{"target_id": null, "a0": 44, "a1": 44, "ticks": 20}
```

Repeating a target/intent preserves persistent controller state. On a changed
target, the adapter emits real double-BRAKE intervals until two decision-boundary
samples have robot speeds <=0.025 m/s and available public reference velocities
<=1e-6 m/s. It then commits the target, recreates target-specific skill slots,
shared motion estimator, contact allocation and coordinator, and begins the
requested intent. No body is teleported and no velocity is reset. All braking
ticks count toward the requested command window and task time. Handoff may take
multiple commands; the reply distinguishes `requested` from `executed`.

Repeated queries do not advance physics. Only command execution does. Mission
success/time/failure accounting remains attached to the original transport
target. Hidden state is used only for sensor generation, physics, termination
and separately labeled offline audit.

## Branches and current limits

`replay_commands()` reconstructs the fixture and replays the entire command
prefix, including braking and target commits. This is real additional physics
work and must be charged by the caller. It is for development reproduction, not
a runtime predictor or a privileged live-state clone. Old snapshot restore is
rejected because it does not contain all bodies and controller histories.

The independent nominal predictor reconstructs geometry from legal history;
it is approximate and requires enough observed geometry of the transport object.
Before that, model construction explicitly fails; real target-free movement is
still available. No hidden pose is supplied to make prediction work. A MOVE
prediction cannot invent an unseen object's future sensor return. MOVE is not
a route planner. There is no automatic semantic author integration or learned
policy evaluation yet. Actual control can stall or return a safety failure even
when an intent is in the public menu.

## Target-free MOVE (new relational protocol v2)

The observation protocol is `relational-option-partial-move-v2`, with the public
action version/hash. Frozen V2 authoring releases and old action IDs are unchanged.
`motion.py` appends 32 velocity Options: eight world-frame directions
`E, NE, N, NW, W, SW, S, SE`, each at `.10, .20, .30, .40` m/s. E is +x, N is +y;
diagonal vectors have unit length, so diagonal speed is not multiplied by sqrt(2).
IDs are `47 + 4 * direction_index + speed_index`. Use `motion.move_id` rather
than inventing IDs. These four discrete reference-speed levels are the complete
current menu, not arbitrary continuous speeds.

Example: R0 east at .30 m/s; R1 north at .20 m/s; one simulated second:

```json
{"target_id": null, "a0": 49, "a1": 56, "ticks": 20}
```

Each side independently chooses MOVE or BRAKE (44); either side can hold while
the other explores. `global_legal_pairs` includes all 33×33 combinations at all
times, independent of object visibility and hidden state. MOVE must have
`target_id=null`; MOVE/PUSH mixed pairs are not supported in this first version.
Target-bound manipulation still requires current visible support. The lazy
search menu interleaves manipulation and navigation candidates and retains all
legal pairs and 1..40 durations; rankers cannot silently exclude movement.

The controller tracks a velocity reference by bounded force, using the inherited
0.5 m/s² reference acceleration limit. Speed is a setpoint, not guaranteed body
velocity or displacement: damping, contact, switching and shared peer-clearance
can change actual motion. Repeated MOVE preserves the servo; direction/speed
changes ramp its reference without teleporting or resetting velocity. A switch
between MOVE and manipulation/brake passes through the existing braking slot.
No automatic end-of-window stop: send BRAKE to stop, or continue MOVE.

Every control tick uses measured ray hits of all object/static/peer categories
and public arena bounds to slow toward zero when the commanded corridor is
occupied. `move_blocked` / `direction_unavailable` report this condition; a
reverse/changed direction can retreat. This local guard is not a collision-free
guarantee or a hidden route planner. The existing physical contacts, anomalies,
peer-clearance recovery, transport success and task time rules still apply.
All movement and braking costs are real; no information/discovery bonus is added.

MOVE leaves the acknowledged `active_target` unchanged and cancels pending
entity handoffs. That field is remembered manipulation ownership, not a claim
that MOVE is targeting the object. A later different-object request still uses
the explicit target-switch brake protocol. Sensor observations after motion
contain only actual nearest ray hits, not a revealed object table.
