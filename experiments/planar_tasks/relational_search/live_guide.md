# Relational seed-repair author contract v1 (experimental)

This is a separately sealed **development** profile under `authoring_v2.relational`.
It does not change historical free-CAP/Option-search releases. Read the supplied
original Option guide and relational MOVE/entity supplement. All model backends
must receive the same knowledge and tools. No model calls happen on creation.

## Task and workflow

Transport the public goal entity using the two robots and existing Options.
Develop from a successful seed: perturb initial geometry within the declared
neighborhood, replay the seed, identify a mismatch using replay and available evidence,
record a short decision brief, choose an intervention boundary, and repair its
suffix. All nearby variants belong to the same mother group/data split.
Do not assume the first detected failure is the best intervention point.

Use `LiveSession(path).tool(name, arguments)` from
`experiments.planar_tasks.authoring_v2.relational_live`. This supersedes the old
public-only author / initial-sequence-generator profile.

Offline author may inspect the provided full simulator/controller/pipeline code,
true state, fixtures, full replay, and write probes. Do not read previous runs'
semantic functions, answers, author workspaces, acceptance/smoke examples or reports.
Use only your clean source bundle and this campaign's artifacts. Native authors
have an instruction boundary, not an OS-enforced filesystem boundary. Deployment
feature workers ARE isolated and only see causal public history and candidates.
Canonical experiment tools meter execution and produce fit evidence. Extra
standalone simulation probes are allowed but must be explicitly reported with
costs; they are not automatically trusted fitting labels or metered tool work.

Tools (Python keyword arguments in parentheses):

- `status`, `knowledge`: budget/state and shared skill documentation.
- `new_round(delta={}, parent=None, expectations=[])`: perturb relative to the
  campaign seed. Deltas are `object.x/y/angle`, `robot0.x/y/angle`, `robot1.*`, or
  the observed entity ID plus field. The host validates geometry. The reply contains local observations; author_context provides full truth separately. A parent must
  have a successful witness; its latest successful sequence is inherited.
- `replay(rid)`: execute the original sequence in this round. Stop on invalid
  commands and preserve all earlier evidence. Returns a compact summary.
- `context(rid, attempt_id=None, boundary=None)`: initial observation, or legal
  history ending at the requested **pre-action** boundary, max 64 events. No
  future evidence is returned in that history. Attempt summaries may report
  already-observed development outcomes. It is not a deployment observation.
- `brief(rid, turn)`: store `{actor, model?, summary, rationale?, evidence?,
  output, usage?}`. Actor is `llm`, `human`, or `engineering`. LLM requires model
  identity/settings. `summary` is one short sentence; `rationale` is a short
  decision explanation, not private chain of thought. `output` holds the
  proposed patch/semantic hypotheses, with evidence IDs. Usage, if attached,
  is externally reported, not host-metered API billing.
- `repair(rid, attempt_id, boundary, suffix, turn_id)`: keep exactly the original
  prefix and replace the suffix. Commands are exactly
  `{target_id, a0, a1, ticks}`. Every command consumes 1..40 ticks of 0.05 seconds.
  Workbench segments are expanded into 4-tick boundaries; preserve these when
  reproducing its witness. A repair evaluates a whole sequence, not a local Q.
- `compare(rid, attempt_id, boundary, candidates, continuation)`: each candidate
  is one legal action; 2..12 candidates. Each is followed by the **same exact
  continuation list** from the same reconstructed prefix. All reset/prefix work
  is charged. Errors/unknowns/ties retained; only comparable completed task
  outcomes create preferences. Preference means better with this continuation,
  not optimal Q or universal action superiority. Duration is part of the action.
- `submit_semantics(rid, source, turn_id)`: submit standard-library Python below.
  Hermetic bubblewrap worker: no network, repository, credentials, environment
  object, fixture or replay geometry. Reject rather than run without isolation.
- `fit(rid, semantic_id, comparison_refs=None)`: fit all completed comparison
  data so far, or an explicit fixed list of `[round_id, comparison_id]`. Same
  verified data for S (semantic), E (fixed geometry/history summary), M (S plus
  regularized residual). Only train split fits/normalizes. `fit_detail(fit_id)`
  returns curves, checkpoints, pair scores/margins and split counts.
- `semantic_source(semantic_id)`: inspect a submitted semantic version.
- `trace(rid, attempt_id, start=0, count=1)`: page 1..5 legal observation rows;
  prefer this to repeatedly loading full histories.
- `author_context(rid)`: true fixture, scene, round and public initial observation.
- `author_replay(rid, attempt_id, start=0, count=5)`: privileged frame pages,
  at most 100. Use full truth to diagnose offline, never as feature inputs.
- `probe(rid, commands)`: execute a manually authored sequence; fully metered.
- `test_closed_loop(rid, fit_id, turn_id, condition='M', max_ticks=400,
  max_decisions=30, prefix=[])`: run the fitted scorer in the fixed planner at
  EVERY actual Option boundary. Each boundary rebuilds an independent nominal
  model from legal actual history, ranks legal candidates, expands predictions,
  executes only the first action, then replans from real feedback. No author
  sequence is followed after that action. Prefix is a metered diagnostic start;
  only empty-prefix tests are full-root deployment. Max episode ticks 1800,
  max decisions 90, wall time 300 s. Record brief before each test.
- `decisions(rid, attempt_id, start=0, count=1)`: full decision evidence including
  candidate batches/scores, expansions, prediction duration, selected path,
  actual chosen action and history hash. Max count 5.

`test_plan` and `apply_ranker` are disabled in this profile. Final submission is
semantic_id, fitted fit_id and S/E/M condition plus evidence, NOT make_sequence.
You can manually repair successful trajectories to collect data without writing
any general sequence generator. Fitted coefficients are host-optimized; authors
submit semantic features rather than editing checkpoints or planner internals.

Fixed search profile (identical for S/E/M): 12 prediction calls, 600 simulated
physics ticks, 120-tick lookahead cap, 72 scored candidates per refill, every
fourth expansion FIFO for coverage, 15 s search wall limit per actual boundary.
Candidate stream first interleaves observed entities and global navigation,
robot roles, directions and durations 40/20/4 ticks; the tail retains ALL legal
pairs and durations 1..40. Finite budget does not guarantee all are considered.
Full requested action duration is simulated unless task terminal; no truncation
to 1 tick. Score determines expansion order; every fourth expansion uses FIFO.
Predicted success wins; otherwise compare longest reached nonterminal leaf
horizons with shared task-seconds plus visible goal-support distance / .3 m/s.
Rank breaks ties. This is bounded heuristic search, not an optimal oracle.
Nominal dynamics reconstruct geometry from rays/history, with cold controller
state, so predictions can be wrong. True replay does not initialize it.
If unavailable, common BRAKE4; three consecutive planner failures stop explicitly.
All prediction prefix replay costs count; total prediction budget 200000 ticks,
10000 calls. Failed/interrupted work retains conservative reservation.


Semantics API:

```python
NAMES = ['example']
def initial_memory():
    return {}
def update_memory(memory, event):
    # event: observation, preceding action, executed_ticks; first action is None
    return memory
def semantic_features(memory, observation, action):
    return {'example': None}  # scalar / bool / None (unknown)
```

At most 16 fixed named outputs; memory is <=8192 JSON bytes; source <=32 KiB.
Training evaluates one history/candidate per isolated process. Inference batches
up to 256 candidates at ONE history, without labels or other histories. Features
must be deterministic functions of their arguments, not batch position or global
call counters; training and deployment must agree.
Both feature inputs and memory use only the retained legal history. Bind entity
IDs to observations/history, never use lexical/numeric ID as a learned feature.
Missing evidence is unknown, not clear space. Avoid future outcome-derived
features. A schema probe is not proof of semantic adequacy.

Expectations are declared before a round runs, e.g.
`{"boundary":80,"kind":"action_legal","action":{...}}` or
`{"boundary":0,"kind":"entity_visible","entity_id":"entity-a"}`.
Automatic mismatch localization reports failed predicates and invalid commands;
it does not claim to infer causal explanation or the earliest optimal repair.

## Budgets, evidence, and interpretation

Defaults: 16000 executed ticks, 50 resets, 50 attempts, 12 rounds, 12 fits, 40
author briefs. Episode cap 1800 ticks/120 wall seconds. Configure **before**
creation. Reset inspection counts; every branch replays and charges its prefix.
Reserve the full attempt before stepping; settle known actual work; interruption
retains its reservation. `audit` marks abandoned running attempts interrupted
without refunds. No automatic retry, hidden state clone or uncharged branch.

Keep original task-cost reporting: successful elapsed seconds, failure twice the
environment time budget (currently 180 seconds); errors are unknown for fitting.
No exploration/clearing/semantic bonus. Sequence exhaustion is a failed bounded
sequence, not a claim of impossibility. S/E/M currently use linear pairwise
logistic rankers; E is an explicit fixed summary, **not a GRU/complete-history
embedding**. Training margins alone prove no generalization or value in search.

After repair, compare legal actions at a common boundary, submit semantics, fit,
inspect residual conflicts, and submit a revised version. For a semantic-only
comparison refit old/new versions on the exact same comparison_refs. Hypotheses
and decision briefs are author claims, not certified causal findings.

The read-only UI includes true geometry, author briefs, comparisons, fit curves,
and per-boundary search logs. Truth is allowed for offline authors but isolated
from runtime features. Freeze and verify_frozen before downstream evaluation.
Native token billing is unavailable: never report it as zero. All these runs are
mother-group development; they do not establish held-out generalization.
