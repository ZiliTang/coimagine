# Action semantics + nonlinear ranking v1

Use `authoring_v2.action_semantics.ActionSession(path).tool(name, arguments)`.
This is a separately sealed experimental profile. Read the ORIGINAL option guide
and relational option supplement as well; campaign.json contains actual budgets.

## Objective and access

Starting from a supplied successful seed sequence, perturb initial conditions,
replay, identify a divergence, and manually repair a suffix. Offline authors may
inspect provided simulator/controller implementation, truth, replay, and write
metered probes. Do not read other experiments' semantics/policies/results. New
submitted deployment code sees ONLY causal legal public history (last64 boundaries),
its own reconstructed memory, current observation and candidate actions. No files,
network, simulator object, true geometry or future outcomes are available there.

Submit short public decision briefs before experiments/edits: hypothesis, evidence,
and intended test. Do not submit private chain-of-thought. The seed is intentional
supervision; fitting/replaying it is not evidence of generalization.

## Author program (stdlib Python, <=48 KiB)

```
NAMES = ['feature_name', ...]  # 1..24 unique names
initial_memory() -> dict
update_memory(memory, event) -> dict  # <=8192 JSON bytes
propose_actions(memory, observation) -> list[command]  # 0..16
semantic_features(memory, observation, action, proposals) -> dict[NAMES, scalar|None]
```

An event has observation, action and executed_ticks; first event has action=None.
Memory is reconstructed once from the available history at each query. Thus memory
is NOT an unbounded persistent clock; history older than64 boundaries is unavailable.
Each command is exactly `{target_id, a0, a1, ticks}`; ticks integer1..40, target
visible or None for target-free movement/braking, joint pair legal under public menu.
Use the actual public action-menu schema and the relational guide for legal pairs. All proposals undergo host validation.

Semantics may directly recommend an entity, joint Option pair, and duration, just
like a CAP policy. Descriptive and action-conditioned features may coexist. Features
can describe agreement with a preferred action type/scale or conditional alternatives;
there is no requirement to use linear duration multiples. A short/medium/long action
can have different outcomes. Never use unseen entity truth or label/candidate order.
Host appends proposal-match features and shared action encodings (duration, target
role, Option one-hots); variable numbers of entities use one shared candidate scorer.

## Three distinct execution modes

* `proposal`: execute the FIRST raw proposal, validate it, then observe real feedback
  and query again. Missing/illegal first proposals fail explicitly. This embeds a
  legal CAP policy with the SAME causal memory, action and timing contract exactly.
* `learned`: merge semantic proposals with first48 public generic candidates, then
  execute argmax of the two-hidden-layer MLP; observe feedback and replan. No nominal
  search overrides the learned choice. This is a diagnostic greedy ranking policy.
* `search`: separately labelled inherited budgeted nominal search. Proposals enter
  early expansion; leaf cost/longest-horizon selection may override learned ranking.
  It is NOT claimed to preserve the CAP policy. Optional, lower priority this round.

Semantic proposals can express ANY currently legal command. The generic generator
has a full legal enumeration tail, but the learned arm uses a finite48-candidate
prefix; do not claim every legal action is considered at every finite-budget step.
Proposal mode provides expressive equivalence, NOT a guarantee that retraining or
search improves it. Test proposal and learned choices on matched initial conditions.

## Tools (in addition to status/knowledge/context/new_round/replay/brief/repair/compare)

* `submit_action_semantics(rid, source, turn_id)` -> semantic ID (schema probe only).
* `inspect_action(rid, semantic_id, attempt_id=None, boundary=None, fit_id=None,
  condition='S')` -> proposals, top10 scores/features, invalid proposals.
* `fit_action_semantics(rid, semantic_id, demonstration_refs=None,
  comparison_refs=None)` -> fit ID, training metrics for S/E/M. Refs are [rid,aid]
  and [rid,cid]; defaults use all successful TRAIN attempts and complete TRAIN
  comparisons IN THIS campaign. No prior experiment import. At most600 distinct
  histories per fit; use explicit demonstration_refs when accumulated data exceed
  this cap (about three long trajectories), and report the selected subset.
* `test_action_policy(rid, semantic_id, turn_id, mode='proposal', fit_id=None,
  condition='S', max_ticks=900, max_decisions=225)` -> summary. All modes execute
  online at every boundary. No fixed prefix is silently appended.
* `author_context(rid)` / `author_replay(rid,attempt_id,start=0,count=5)` offline
  truth; `trace` and `decisions` return small legal evidence pages.
* `probe(rid,commands)` executes a metered command sequence. Arbitrary author-written
  analysis is allowed, but ALL simulator execution/prediction must use metered tools.
* `fit_detail(fit_id)`, `semantic_source(semantic_id)` read artifacts.

`brief` takes `{rid, turn:{actor:'llm', model:'gpt-6.1-sol high', summary:'...',
rationale:'brief evidence-based reason', output:{...}, evidence:[...]}}`.
`repair` takes rid,attempt_id,boundary,suffix,turn_id; compare takes rid,attempt_id,
boundary,candidates,continuation. Prefix replay ticks are charged. A successful new
round can become parent of a subsequent round; deltas are relative to original seed.
Check signatures in supplied implementation for details.

## Fixed host fitting; author cannot edit weights

Every successful training trajectory contributes its actual next action at each
legal history. Same-history multiple successful choices form one positive set.
Proposals + generic candidates + teacher actions + same-pair4/10/20/40tick variants
form the training candidate set. Teacher injection has NO label passed to features;
extras are content-hash ordered. Full trajectories include control/target handoffs.

The loss is successful-choice set cross entropy plus independent verified physical
pairwise preference loss. Undemonstrated alternatives are choice contrasts, NOT
labelled physical failures. Timeouts/unknown physical comparisons give no preference.
A training fit alone never establishes closed-loop success or optimal duration.

S: semantic + proposal-match + action features, 32-tanh-32-tanh MLP.
E: fixed geometric/history summary + action features, same MLP. E has the SAME
LLM-generated candidate set; this is an input ablation, not an LLM-free baseline.
M: frozen S score + E residual MLP, centered residual penalty0.05.
Normalization uses training data only. Adam, seed17,400 epochs fixed by host.
Raw E is not a learned full-history encoder; residual strength is diagnostic, not
proof of semantic sufficiency. Loss/metrics/data/checkpoints are saved for the UI.

## Budget and conclusion

Read actual caps via status. Execution, prefix replay, predictions, feature-worker
calls, resets, author briefs and fits are recorded. Author source is immutable per
submission; new source requires new semantic ID and new fit. Use summary first,
request only relevant trace boundaries. Finalize own FINAL.md/FINAL.json, specify
semantic_id, fit_id, condition/mode, successes AND failures with denominators,
remaining limitations. Freeze via ActionCampaign(path).freeze(), verify_frozen().
Do not claim held-out generalization from development perturbations or repeated seeds.
