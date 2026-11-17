# History-semantic planning v0

Executable scaffold for the [2026-11-12 research direction](../../../docs/research/History_Semantic_Planning_v0_20261112.md).
This is a systems prototype, not a learned policy or a performance result. It adds
bounded recursive history semantics and declarative value/action trees to the
unchanged Option-search-v3 planner and authoring_v2 interface.

## Build a revision

```bash
.venv/bin/python -m experiments.planar_tasks.semantic_planning build outputs/semantic_planning/reference_v0
.venv/bin/python -m experiments.planar_tasks.semantic_planning verify outputs/semantic_planning/reference_v0
```

Edit a COPY of `reference.py`, then build with `--source FILE`. A subsequent version
uses `--parent REVISION --evidence FILE --reason TEXT`. Sources, embedded runtime,
compiled hooks, evidence copies and parent hash are retained. Existing destinations
are rejected. Evidence is author-supplied, not automatically judged correct. These
commands do not execute authored Python, step physics or call an LLM.

## Author interface

The source contains exactly three functions and a literal `TREES` dictionary, plus
optional math/copy imports. Functions may use local computations and nested helpers.

```python
def initial_memory(): ...                 # JSON object, at most 8192 bytes
def update_memory(memory, event): ...     # new JSON memory
def semantic_features(memory, frame, action): ...  # <=64 scalar/None features
TREES = {'value': ..., 'rank': ...}
```

`event` has `frame`, the previously executed `action` (None at reset), and actual
`executed_ticks` (None at reset). A frame contains whitelisted legal `observation`,
`estimate` (object/robot kinematics, possibly missing, carried or wrong), `goal`,
`source`, and cumulative synthetic `elapsed_ticks`. No task ID, execution truth,
environment object, complete map, hidden dynamics or global controller state enters.
Current observations are centralized shared legal sensors, NOT decentralized data.

Root updates occur once per real decision. Each synthetic path copies its parent's
memory and advances only on that path's observations. Branch updates never reach
real memory. `source` distinguishes real estimates from synthetic observations;
synthetic sensor timestamps are model-local, not the real episode clock. Resetting
an episode requires a new runtime/process. Candidate features are pure queries:
they receive an optional command but never update memory. The root current sample
is included before feature evaluation. Historical events between boundaries are
unobserved; contact counts cannot be described as contact durations.

Trees contain either a leaf `{'cost': expression, 'proposals': [...]}` or a branch
`{'feature': name, 'op': 'eq|lt|le|gt|ge', 'value': scalar,
'yes': subtree, 'no': subtree, 'unknown': subtree}`. Proposals only occur in rank
leaves. Missing/None semantic values follow `unknown`; they are never coerced to
false. Every branch must provide that fallback.

Expressions are finite numeric constants, `{'feature': name}`, `{'action':
'a0|a1|ticks'}` (rank costs only), or `add/mul/min/max` over expression lists.
Booleans may drive branches but cannot serve as numeric costs. Rank costs are
TOTAL remaining task seconds conditional on the current command; value costs are
remaining seconds at a concrete node. No optimality, admissibility or safety claim.

Optional leaf proposals are `{a0: expr, a1: expr, ticks: expr, cost: expr}`. They are
evaluated once with `action=None`; command fields must evaluate to integers and
belong to the current legal menu. At most eight proposals; all candidate costs
are still evaluated and returned. Arbitrary legal pairs and durations remain
expressible via proposal constants or semantic output fields. No tree-based mask
or hard pruning is introduced. The fixed planner can still select unpredicted
roots. Invalid hooks retain the v3 recorded fallback behavior, not a safety guarantee.

The tree interpreter receives scalar features and command fields only. Authored
Python must be pure, use only supplied inputs, and keep persistent state in memory.
These programming rules are not a language sandbox: Python can access globals or
attempt to tamper with its own module. The existing OS sandbox enforces external
information access; formal enforcement of semantic purity is not implemented.

## Use with existing authoring and execution

Use `semantic_planning.session.create_session`, which delegates creation to
`authoring_v2.create_session`, then supply the compiled `heuristic.py`
via `write_file`, `test_policy` and atomic `submit`. `verify_frozen` is still required
before evaluation. No old framework/knowledge release is changed by this module.
For a future model experiment, seal this method guide and runtime as additional
experimental materials; all authors also get the same `authoring/option_guide.md`
and `authoring_v2/experiment_api.md`. A formal method-specific author/repair driver
and tamper-resistant runtime restriction inside authoring are not yet implemented.

This thin session adapter also fixes a known v2 summary-reader defect locally:
`sensors.jsonl` alternates request/response rows, so only request observations may
be summarized. Historical v2 sources/seals remain unchanged. Reopen through this
adapter to retain the fix. It does not add observations or change cost accounting.

`history_semantic_v0` JSONL records in `policy.stderr` contain knowledge hash,
real decision ID, synthetic node ID/source, semantic values, branch choices,
candidate costs and proposals. Join with `option_search_v3` records on decision ID
to inspect expansion and selected action; execution/replay remains in the existing
episode artifacts. Large logs stay in artifacts, not automatically in LLM context.

## Acceptance and scope

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_semantic_planning.py
```

An explicit bounded physical acceptance command (three anchors, at most 240
execution ticks total, no model calls) is:

```bash
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.semantic_planning.smoke outputs/semantic_planning/NEW_RUN
```

It uses the ordinary v2 session test/submission machinery via the local log-reader
adapter. Each episode stops at 80 ticks; truncation is expected and retains the
original failure cost. Passing means wiring/forecast/feedback/summary/freeze works,
not that any transport task was solved. Existing output directories are rejected.

Tests distinguish synthetic mechanism checks from the separate short physical smoke
run. The reference uses a neutral distance estimate and logs contact evidence but
contains no wall-release/detour recovery strategy. No task-success, generalization,
counterexample-learning or physical-evidence benefit follows from passing these
checks. Automatic conflict discovery, experiment selection and LLM joint revision
are next-stage work; this version supports compiling, tracing and versioning those
revisions through existing authoring, not an autonomous new learning algorithm.
