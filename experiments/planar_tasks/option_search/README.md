# Fixed Option search v1

This is a separate framework over the unchanged v1.4.0 execution/prediction
protocol. It is not a new learned policy or a claim of improved task success.
Only `heuristic.py` is replaceable in a deployment package; `build.verify_package`
checks the other modules against frozen sources, not just a supplied manifest.
Planner budgets are explicit experimental settings and must match across a comparison.

## Action and trace support

At each **real observation history** h, the original command set is

```
A(h) = {(a0, a1, ticks): [a0,a1] in observation.legal_pairs,
                         ticks in {1,2,...,40}}
```

`search.action_space` materializes exactly A(h). All roots remain output candidates,
including unexpanded nodes and actions rejected by the approximate predictor.
The heuristic may give any desired root a sufficiently lower finite score than the
others; no task-stage mask, minimum progress condition, macro-only interface,
coarse duration grid, mandatory braking or mandatory option completion intervenes.
Repeated options retain the original controller continuation semantics. One original
command is executed, followed by real feedback and a freshly built root set.
The estimated future menu never replaces the next real menu.

Consequently, by induction over real decision boundaries, the framework has the same
**structural support of legal command traces** as free policies under the same
execution interface/horizon. Tests enumerate all 160 commands in a representative
menu and select each one, including under predictor rejection; dynamic-menu tests
cover repeated options and different durations. Real smoke checks reproduce prefixes
of the frozen free policy in all three task families.

This is NOT equivalence of policies computed under finite resources, equal search
coverage, completeness over unknown dynamics, or a safety guarantee. Hook computation,
CPU/memory limits and fixed perception/prediction affect practical performance.
Requiring every selected root to pass approximate simulation would break the support
guarantee, so `selected_status=unexpanded/model_error` is allowed and recorded.
Predicted descendant eligibility restricts the searched hypothetical tree, not the
commands executable after the next real observation.

## Search, prediction and heuristic contract

- Fixed tree search, no pose merging or beam deletion. Distinct controller/action
  histories remain distinct nodes. Roots and each expanded nonterminal node contain
  every legal pair and all 40 durations.
- Selection alternates best-first, child-focused lookahead, child-focused lookahead,
  FIFO, repeating. This allocates finite resources to multi-step paths while retaining
  other branches. It does not prescribe an option sequence or guarantee a useful depth.
- Search priority is `cost_seconds + h`. Cost is simulated elapsed time (requested
  time before prediction); h is unrestricted and need not be admissible. This search
  score does not change environment reward, success or failure accounting.
- `Budget` limits expansions, predictor calls/ticks, wall time and optionally depth.
  Depth bounds hypothetical lookahead only, not executed trace length. A shared wall
  deadline bounds hook IPC and checks between predictor operations; the host's response
  deadline remains the final guard. This is not hard real-time scheduling.
- `ReplayModel` reconstructs each hypothetical prefix in an independent nominal world,
  charging **all** repeated prefix simulation. Synthetic controller history persists
  within that path. Root initialization is cold; actual contacts and controller history
  are NOT cloned. Known cold-start prediction errors remain an explicit limitation.
- Frozen perception uses the earlier runtime program's legal-sensor geometry helper
  (`geometry.py`, byte-identical SHA256
  `f85740f821864b50c5d8b4d35f7ecc658de6afc94ac0323b408b530bece324f0`).
  It estimates a known rectangle, sets object velocity to zero, and approximates static
  returns by a bounding box. It is a controlled initial choice, not general perception.

Replaceable API:

```python
def score(context, nodes, memory):
    # one finite numeric h per node, in supplied order
    return [0.0 for node in nodes]
```

`context` contains the real observation, legal estimated hypothesis, previous action
and estimator error/cold-start label. `memory` persists across decisions and includes
real observation history. Each node contains `id`, `parent_id`, original `plan`,
`cost_seconds`, `status`, `prediction` and `parent_prediction` when available.
Predictions are synthetic state/observation/termination results, not execution truth.
The heuristic can rank candidate actions and evaluate states/action sequences using
these inputs. It runs in a separate process and returns numbers, not commands or RPCs.
Prints are redirected to stderr; invalid returns/timeouts are recorded and fall back
to earlier valid scores or zero. There is no task-specific recovery script.
The enclosing existing bubblewrap sandbox isolates execution-world files/network.
Process separation protects ordinary in-process mutations; it is not a claim of
complete malicious-code or industrial safety certification.

The shipped distance heuristic is intentionally a minimal reference. It can be
short-sighted and often prefers tiny durations. It is not an optimized baseline.
The framework alone does not establish that deeper planning or LLM heuristics help.

## Build and inspect

From the repository root with the existing environment:

```python
from pathlib import Path
from experiments.planar_tasks.option_search.build import build_package, verify_package
from experiments.planar_tasks.option_search.search import Budget
build_package(Path('/tmp/new-search-package'), heuristic=Path('/tmp/heuristic.py'),
              budget=Budget(expansions=6, prediction_ticks=480, prediction_calls=40))
verify_package(Path('/tmp/new-search-package'))
```

Use the existing `package_command` and `runner.run_episode` with the nominal-v1
runtime predictor enabled. Each decision emits an `option_search` JSON record to
`policy.stderr`: all-root count, expansion paths/order, selected path/status/score,
resource reservations/actual requests, model errors and scorer failures. Standard
`decisions.json`, `sensors.jsonl`, `truth.json` and `replay.json` remain available.
Execution truth is written only by the host for audit, never passed to the hook.

`release_v1.json` freezes this overlay separately, including the test-file hash and
the unchanged baseline release hash. `smoke.py` is a bounded one-off acceptance
script with a fresh output path; it refuses to overwrite evidence. Its action-selecting
heuristic uses previously recorded commands solely as a compatibility witness,
not as a deployable or generalizing policy. The reference heuristic is tested separately.

This stage supplies buildable/verified heuristic packages and runtime search.
It does not yet connect a new LLM synthesis campaign, provide an autonomous
heuristic-only authoring session, or add a new console interface. A later campaign
must freeze its authoring contract and count all active physical validation and
prediction work, rather than treating collected experience as free.
