# Independent nominal predictor, nominal-v1

Both conditions can use this service in coding via `run_python`. A deployed
`Policy.act` can call it only when `campaign.json.prediction.runtime` is true.
Import `Predictor` from `predictor_client`. The client is supplied automatically
in a submission; submit your own files only. No runtime LLM is supplied.

```python
from predictor_client import Predictor
p = Predictor()
h = {"arena": [4, 4], "goal": [3, 2], "goal_radius": 0.15,
     "object": [2, 2, 0, 0, 0, 0],
     "robots": [[1.2, 2, 0, 0, 0, 0], [2, 1.4, 0, 0, 0, 0]],
     "obstacles": []}
r = p.create(h)
handle = r['handle']
pair = r['observation']['legal_pairs'][0]
r = p.step(handle, {'a0': pair[0], 'a1': pair[1], 'ticks': 10})
fork = p.fork(handle)   # replays predictor history; replay ticks are charged
p.close(fork['handle'])
p.close(handle)
print(p.status())
```

This example demonstrates the API, not a task policy. Infer your own hypothesis
from legal observations/history. There is no execution-state clone, map lookup,
task identifier input, shared RNG, privileged contact state or parameter readback.
The service has no reference to the execution environment. Values returned in
`state` are exact states of YOUR synthetic world, not measurements of the real one.

State arrays: `[x, y, angle, vx, vy, omega]`, world metres, seconds, CCW radians.
Exactly two robots. Obstacles: `[cx, cy, half_width, half_height, angle]` boxes.
All six top-level fields are mandatory; unknown fields rejected. Arena dimensions
1..10 m, centres inside arena, initial velocities/angular velocity bounded by 3,
at most 16 positive-size obstacle boxes. Overlapping hypotheses are not corrected;
the same physics failure rules apply. Object shape is fixed 1.0 by 0.25 m rectangle.
No obstacle shape completion, pose estimation or action selection is supplied.

Nominal known model: same option implementation and fixed dynamics as execution,
including approach, contact acquisition, yielding and controller continuation.
Control tick 0.05 s; physics tick 1/120 s, six substeps. Robot radius 0.18 m,
mass 4 kg; object mass 3 kg; max robot force 15 N; linear drag 8, angular drag 0.6,
contact friction 0.6, restitution 0.05. This is a known-dynamics diagnostic.
Precision is FIXED in this first profile. Changing rollout length or number of
branches does not change physics fidelity. Coarse dt has not been validated.

`create(hypothesis)` starts at synthetic time zero with cold controller/sensor
history and fresh contacts/success hold. Supplying accurate poses and velocities
alone does not reproduce a mid-execution controller state. `step(handle, action)`
continues that branch's option lifecycle, with the same action contract (1..40
ticks, legal pair from its predicted observation). `fork(handle)` reconstructs
only the predictor's own initial hypothesis and complete action history; its cost
includes all replayed ticks. Original branch stays unchanged. `close` releases
memory. `status` reports budgets. Handles are local to a coding session or single
deployment episode; no persistence across deployments. Predictions terminate on
the same success/failure/horizon conditions. No stepping terminated branches.

Replies include `handle`, `source`, `state` (object/robots each x,y,angle,vx,vy,omega),
`observation` (same sensor schema), `ticks`, `success`, `termination`, `usage`.
On a failed step, its handle is discarded, including invalid actions/timeouts;
fork failure leaves its parent untouched. Errors raise RuntimeError in the client.
Every invocation, including status, close and rejected requests, costs a call.

Prediction budgets, separate from physical execution:

| Resource | Shared coding total (scripts + policy tests) | Each deployed episode |
|---|---:|---:|
| Calls | 12000 | 2000 |
| Control ticks (including fork replay) | 120000 | 12000 |
| Constructed worlds (create + fork) | 4096 | 512 |
| Predictor CPU seconds | 300 | 30 |

At most 16 live handles per service. Deployment per `act`: 64 calls, 600 ticks,
1 CPU second; coding direct RPC applies this window to each RPC. A test-policy
run uses deployment per-act and per-episode limits plus the shared coding total. Engine work checks
deadlines between ticks and after construction/observation; maximum overshoot is
one bounded engine operation. Parent policy response and episode wall deadlines
remain in force and are not extended by nested RPCs. Failed partial work is charged.
Logs retain all prediction requests/results and actual CPU/wall usage.

Execution physics pauses during policy computation. This is synchronous planning,
not a real-time robot experiment. Total policy latency includes prediction time.
No-service policies remain free to implement their own mathematical models.
