# Relational seed-repair author contract v1 (experimental)

This is a separately sealed **development** profile under `authoring_v2.relational`.
It does not change historical free-CAP/Option-search releases. Read the supplied
original Option guide and relational MOVE/entity supplement. All model backends
must receive the same knowledge and tools. No model calls happen on creation.

## Task and workflow

Transport the public goal entity using the two robots and existing Options.
Develop from a successful seed: perturb initial geometry within the declared
neighborhood, replay the seed, identify a mismatch using legal observations,
record a short decision brief, choose an intervention boundary, and repair its
suffix. All nearby variants belong to the same mother group/data split.
Do not assume the first detected failure is the best intervention point.

Use `RelationalSession(path).tool(name, arguments)` or the JSON CLI. This is a
host adapter for an external author; it is not yet an autonomous provider driver.
An external author must receive only responses/knowledge, not the host artifact
directory. A native agent with unrestricted repository access does **not** gain
an enforced information boundary merely by promising to use these tools.

Tools (Python keyword arguments in parentheses):

- `status`, `knowledge`: budget/state and shared skill documentation.
- `new_round(delta={}, parent=None, expectations=[])`: perturb relative to the
  campaign seed. Deltas are `object.x/y/angle`, `robot0.x/y/angle`, `robot1.*`, or
  the observed entity ID plus field. The host validates geometry. The reply
  contains actual local observation, never hidden geometry/mass. A parent must
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
- `apply_ranker(rid, fit_id, attempt_id, boundary, candidates, continuation,
  condition='M')`: score a supplied variable-sized legal candidate set, execute
  the top-ranked action plus the supplied continuation after replaying the
  prefix. Records all scores and selection. This tests the fit-to-action wire;
  it is not a complete search-based deployment evaluation or held-out success.
- `test_plan(rid, source, turn_id)`: test an isolated standard-library function
  `make_sequence(observation, seed_commands)` returning a list of commands. It
  sees the initial public observation and the campaign seed sequence, never
  fixture/delta/truth or live execution feedback. This is an open-loop sequence
  generator, not an online replanning policy. Keep the exact same source hash
  across variants to test transfer. Source <=32 KiB, <=1800 generated ticks.

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
Each process receives one history/candidate only, with no preference labels or
other examples, so author code cannot infer labels from a batch's ordering.
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

The human pipeline UI is read-only and includes privileged scene geometry for
replay; never expose its API to a policy/author. Freeze artifacts with the profile's
`verify_frozen` before downstream evaluation. This profile does not yet implement
an automatic deployment-policy export, full closed-loop learned planner panel,
or autonomous provider transport. These remain separate acceptance steps.
