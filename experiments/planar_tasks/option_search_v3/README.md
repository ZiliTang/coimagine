# Option search v3: local candidates and measured lookahead

Separate version; all v1/v2/old baseline releases remain unchanged. This release
corrects the search/authoring contract, not a claim of task-success improvement.

## Typed local interface

```python
def rank_actions(parent, candidates, memory):
    return {'costs': [...], 'proposals': [
        {'action': {'a0': 3, 'a1': 3, 'ticks': 23}, 'cost': 10.0}
    ]}

def evaluate_state(node, memory):
    return 10.0
```

`costs` has one finite number per candidate: estimated TOTAL remaining task seconds
conditional on this CURRENT command at this CURRENT parent. It includes action time
and anticipated consequences. Lower is better. `proposals` is optional, bounded by
host proposal_limit, and accepts ANY locally legal command, not just default seeds.
`evaluate_state` is a remaining-cost estimate on a concrete estimated/predicted state.
It is never called on unexecuted candidate actions. No admissibility/optimality claim.

Node fields: id, source, state (object/robots x,y,angle,vx,vy,omega or None), observation,
elapsed_ticks, depth, incoming_action, path_context, goal, terminal, success.
At root the observation is real legal shared sensors, state is the frozen legal
estimate. At successors both state/observation are synthetic. `incoming_action` was
already executed in the model; the candidate argument is the action being ranked.
`path_context` is history only. Memory context/history carries real feedback and the
public action contract. No execution truth, task identifier or simulator object enters.

## Fixed planning and evidence semantics

Candidates are lazy: continuation, then each legal pair at40/20/10/1ticks, then all
other durations. Initial batches24, revisits widen by8; at most8 hook proposals per
batch. Duplicate commands coalesce by exact command only, never by state/geometry.
No irreversible heuristic pruning or hidden-state mask. All default proposals use
only public legality/previous command. Current defaults are a finite search allocation,
not a guarantee of sufficient time horizon or task success.

Six-expansion schedule: start at root, follow the latest successful child up to two
times, repeat. Once lookahead_ticks120 is reached, restart at root. This exposes
multiple roots and local continuations; hook-selected short commands are still legal.
Actual time/depth/branch coverage is logged, not inferred from layer count. Optional
max_depth bounds search only, never the executed command language.

Within a path, consume/reuse the synthetic world handle and execute the next command
without cold reconstruction. Revisiting a consumed parent rebuilds its prefix and
charges EVERY replay tick. Handles are closed at the real decision boundary. Calls,
ticks, CPU, depth and wall limits remain explicit; forecasts do not advance real time.
Root initialization remains cold, not an execution-state clone. No physics-dt change.

A predicted edge is scored as actual simulated edge time plus successor value. At a
node with predicted children, back up the best predicted child, not the ancestor's
unexecuted action prior. Leaves use evaluate_state. At the real root, unknown/error
commands retain rank_actions' explicit total-cost prior; they are labeled, eligible
outputs. This permits unverified execution and does not certify safety. Calibrating
priors against rolled-out values remains the author's responsibility; the framework
no longer silently adds elapsed time to an unchanged parent estimate for each duration.
Only the first original command is executed, followed by real feedback and replanning.

## Action support

The lazy stream exhaustively equals legal_pairs × ticks1..40. Under finite candidate
budgets, an author can directly nominate ANY legal command with a low cost and assign
consistent low successor values to that root. Thus any original legal command trace
remains expressible by a corresponding hook program, inductively over real feedback.
Every proposal is host-validated. Predictor rejection cannot delete real-legal roots.
No claim of equal finite-budget coverage, equal programs under CPU limits, or planning
completeness. Immediate access to all commands relies on the explicit proposal channel,
not pretending that a bounded lazy batch has enumerated every command.

## Diagnostics and budgeted authoring

Trace kind option_search_v3 records materialized candidate/state counts, calls/ticks,
actual forecast/selected horizon, suffix durations, candidate ties, distinct first
commands, model errors, prefix reuse, selection evidence and hook failures.
After an executed predicted command, prediction_feedback compares predicted position
with the next LEGAL estimate only if elapsed duration matches; it labels estimate
uncertainty and does not expose truth. No bonus/reward, mask or automatic repair.
Parent/worker CPU and RSS are recorded. The v3 runner records return code/signal,
whether cleanup initiated termination, and waited-process-tree CPU delta. A signal
alone does not identify OOM vs CPU; do not retroactively label the old unexplained exit.

The standard-library hook worker isolates mutations and RPC. Host validates finite
returns/nominations, bounds IPC and disables a failed worker. Failure retains neutral
priors or zero state values, records errors and still chooses an actual legal command.
The enclosing baseline bubblewrap/network and resource limits are unchanged.

Session, drive, test_policy, submit, finalization and evaluation mirror v2, with new
hook signatures and trusted files. Use option_search_v3.session.Session and its
verify_frozen, and option_search_v3.evaluation.evaluate; old verifiers are insufficient.
Deployment includes only heuristic.py plus trusted files. experiments/*.py are for
active validation and never enter the final package. Test/active prediction share
one persistent ledger. Test tick reservations are not refunded. No implicit API call.

The provided heuristic is deliberately neutral (distance leaf value, tied action
costs). Ties are warned about. It is an interface reference, not an optimized baseline.
We do not ship hidden wall-release/detour scripts or port the historical Sol solution.
