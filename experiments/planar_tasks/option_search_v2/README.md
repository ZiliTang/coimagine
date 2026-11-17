# Heuristic-only CAP authoring / Option search v2

This separately frozen overlay preserves option-search-v1 and baseline v1.4.0.
It adds two editable hooks and a restricted authoring session. No new model synthesis
or success-rate campaign is started by these modules.

## The two hooks

```python
def evaluate_nodes(context, nodes, memory):
    return [0.0 for node in nodes]  # estimated remaining cost, seconds

def score_actions(context, nodes, memory):
    return [0.0 for node in nodes]  # search priority adjustment
```

Both return one finite number per supplied node; lower is better. Search priority
is `elapsed/requested seconds + evaluate_nodes + score_actions`; final path selection
is `elapsed/requested seconds + evaluate_nodes`. Thus action scores allocate search,
not directly select execution or modify reward. Node evaluation can include the full
path, action identity and state/history, and need not be admissible. No optimality claim.
The scheduler retains periodic FIFO and child-focused lookahead, so priority is not
a promise that every expansion follows greedy order. Fixed numeric tie-breaking remains.

`context`: actual observation, estimated hypothesis, previous real command and cold
initialization/error labels. `memory`: persistent dictionary, including actual history.
`nodes`: id, parent_id, plan of original commands, cost_seconds, status, prediction,
parent_prediction. Unexpanded nodes have no own prediction; use their parent prediction
or the root hypothesis. Predictor observations and state are synthetic, never real truth.
Functions run in an isolated child process. Invalid values/timeouts are logged and
fall back to prior valid scores or zero; they cannot change the parent legal menu.

All actual legal_pairs × integer durations 1..40 remain executable roots, including
unpredicted/model-error roots. Every real command is followed by feedback and root
reconstruction. Repeated options retain controller semantics. No beam deletion,
pose merging, task-stage masks or mandatory progress. These preserve structural legal
trace support, NOT equal finite-budget policy expressivity/search coverage or safety.
Cold predictor roots and frozen approximate geometry remain known limitations from v1.

## Authoring tools

`session.Session` retains baseline tools and accounting: list_files, read_file,
write_file, run_python, test_policy, read_log, submit. Author-editable files:

- `heuristic.py`: both hooks in one self-contained deployment module.
- `experiments/*.py`: development-only active execution/prediction scripts.
- `notes/*.txt`: development-only notes.

`run_python` runs only experiments/*.py, under the existing filesystem/network sandbox.
Scripts can write scratch workspace files; none can replace trusted deployed core files.
Use `sys.path.insert(0,'.')` in these scripts to import the provided root-level clients.
The host always assembles the package from frozen trusted sources and the one heuristic.
`test_policy(files=['heuristic.py'],task=...,max_ticks=...)` and
`submit(files=['heuristic.py'],contents={'heuristic.py': 'complete text'})` share that builder.
No policy.py, helper module, budget or framework override is accepted. The workspace
framework.json is informational; the authoritative host profile is outside the sandbox.

Actual physics is charged per step; tests reserve max_ticks before running even when
packaging/runtime fails. Test prediction and active-script prediction share the same
persistent development ledger. The planner additionally has per-decision budgets and
the prediction service has episode/CPU/handle limits. Status exposes both ledgers.
Two-stage finalization is inherited: experiment tools close, only atomic submit remains.
Submission checks syntax and isolated loading/signatures of both functions; no real
simulation or success check. Test logs expose only sensors/decisions/stderr and script
RPC/stdout/stderr, never truth/replay internals or source snapshots. Host replay files
still exist for the human's audit.

## Host entry points

```python
from experiments.planar_tasks.option_search_v2.session import Session, verify_frozen
from experiments.planar_tasks.option_search_v2.search import Budget
from experiments.planar_tasks.option_search_v2.driver import drive
s = Session.create('/absolute/new/session', config,
                   planner_budget=Budget(expansions=6, prediction_ticks=480))
# config is the existing explicitly budgeted development config, with
# prediction={'version':'nominal-v1','runtime':True}.
try:
    result = drive(s, complete)  # caller-supplied transport, no implicit model/API
finally:
    s.close()
verify_frozen('/absolute/new/session')
```

For an externally relayed LLM, `Session.tool(name,arguments)` is the same audited
entry; model calls/tokens must still be reserved and recorded by the relay host.
Do not invoke the old free-policy driver: this overlay's driver has the correct hook
instructions and submission schema. No provider is contacted by creation/import/tests.

Use `option_search_v2.evaluation.evaluate` for a frozen batch evaluation: it retains
the baseline panel preregistration, no-repeat claim and immutable-package checks,
and calls this overlay's `verify_frozen`. For individual host checks, call that
verifier before using `package_command` / `run_episode`. It additionally checks framework/profile,
all package files, trusted core hashes and fixed planner budget. Do not use the old
free-policy evaluation verifier alone. Authoring configuration hashes, model and
prediction ledgers, test packages, submissions and original baseline release are kept.
The final evaluation itself must have a separately declared panel and resource budget.
