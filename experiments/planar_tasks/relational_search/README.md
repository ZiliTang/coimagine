# Relational search: first construction increment

2026-11-13 update: [seed/perturb/repair/comparison/fitting pipeline and viewer](PIPELINE.md)
now have a bounded real-physics engineering acceptance. The separate author entry is
`authoring_v2.relational`; no LLM experiment or full learned-planner evaluation has
been run. The component status below describes the earlier construction stages.

Implements the first contracts of the [minimal-system plan](../../../docs/agent_exchange/tasks/Relational_Search_Minimal_System_20261112.md).
This is an experimental component library, not a deployable multi-object policy
or an authoring release. No LLM knowledge discovery has been run.

## Implemented

- `testbed.py`: visible-obstacle clearing/transport development fixture and a
  manual sequence adapter for the existing local workbench. Entity-bound
  commands are checked every 0.2 s; recorded post-step frames contain all bodies
  for privileged human replay. Editable seed `relational_clear_transport_demo_v1`
  has an actual 32.75 s hand-authored task witness, not a learned policy result.
  See [scenario notes](../../../docs/research/Relational_Scenario_Pilot_20261113.md).
- `motion.py`: target-free eight-direction MOVE with four speed levels, two
  independent robot commands, inherited bounded-force/reference-acceleration
  actuation and a measured-surface stopping guard. MOVE/BRAKE remain available
  with no visible entities. Versioned separately from frozen V2; the public
  supplement documents the 32 appended IDs, switching and model limits.
- `estimator.py` / `predictor.py`: visible-hit box hypotheses, explicit depth/
  zero-velocity/material assumptions, independent nominal Option worlds,
  absolute-time synthetic feedback, consumed-prefix reconstruction and full
  replay charges. No execution state/fixture is accepted by the predictor.
- `environment.py`: partial-observation multi-object adapter to the existing
  persistent V2 Option controllers. Target-filtered patches/contacts, per-entity
  legal menus, full-body collision/anomaly checks, command replay and fixed
  transport-goal identity. See [public supplement](option_guide.md).
- `contract.py`: public sensor projection, observed entity points and instance
  labels, entity-bound commands, exhaustive lazy action support, and isolated
  64-boundary histories. Hidden packet extensions are discarded. Target IDs
  require current ray support for manipulation. Target-free MOVE/BRAKE
  (`target_id=null`) remain expressible even when all targets are occluded;
  both candidate families are interleaved without removing legal actions.
- `features.py`: three-function executable semantics, at most 16 named scalar/
  boolean/unknown features and 8192-byte JSON memory. Recomputes within the same
  history window for every query. This trusted-function adapter is not an
  untrusted-Python sandbox; the future author must use an isolated worker.
- `evidence.py` / `dataset.py`: comparable continuation records, explicit
  execution vs prediction sources, unknown/tie preservation, history hashes,
  mother-group split checks, non-overwriting JSONL, and training-only selection.
  The ledger admits all resource charges atomically; durable campaign recovery
  and reservation accounting remain to implement.
- `fit.py`: NumPy weighted pairwise logistic ranker, train-only normalization,
  missing-value masks, checkpoint roundtrip, frozen semantic offsets and
  pairwise-contribution regularization for residual fitting. No history GRU yet.
- `search.py`: model-independent search with separate expansion scores and task
  seconds, one fairness expansion in four, exact requested edge durations,
  conservative charges on prediction failures, branch history isolation, and
  no selection of unpredicted root priors. The model must manage immutable
  prefix handles in an independent predictor. If no edge succeeds, the caller
  receives `needs_common_fallback`; no unsafe action is silently invented.
- `target_binding.py`: observation-only target handoff state machine. Two slow
  boundary samples with stopped public velocity references authorize target
  commit; preceding brake intervals are real commands. The environment resets
  target-specific Option allocation/estimation on commit.
- `physics_probe.py`: two movable bodies in one pinned Pymunk backend, force-only
  robot actuation, ray labels/occlusion, damping on the added body, and explicit
  fixture/prefix replay. This probe is NOT an Option environment. Existing
  single-object snapshot export is deliberately rejected.

Search currently learns only candidate expansion ordering. Candidate generation
and leaf task cost are supplied separately. The added protocol is independent
of old frozen action contracts. The public-projection function assumes its input
is produced by a trusted sensor adapter; it cannot certify the provenance of
arbitrary point lists supplied by a caller.

## Checks and bounded probe

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_relational_search.py
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.relational_search.smoke outputs/relational_search/NEW_DIRECTORY
```

The smoke command refuses to overwrite. It runs a **handwritten synthetic**
history-aliasing/fitting fixture and three low-level physics probes (free,
coupled, heavy), each 160 force-control ticks plus 160 replay ticks: 960 ticks
total, bounded by 4000 execution ticks and 900 seconds. Source copies, raw public
sensor rows, separately labeled offline audit states, configuration, counters,
summary and file hashes are saved. Audit truth must never be fed to a policy.

The synthetic case is an implementation diagnostic: a supplied history feature
resolves a supplied ranking ambiguity. It is not a physical task result, an LLM
discovery or evidence for the research hypothesis. Force probes establish
coupling/replay, not collaborative skill success or task feasibility.

## Option wiring acceptance

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_relational_environment.py
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.relational_search.option_acceptance outputs/relational_search/NEW_OPTION_RUN
```

The bounded command-driven acceptance uses a reachable and a constrained B
fixture, A→B→A handoffs and full command replay. This verifies skill wiring, not
general task success. Existing single-object forced trajectories are compared
against the original OptionSystemEnv. Physics and source artifacts for the
current adapter are in `outputs/relational_search/partial_option_acceptance_20261113_02/`.
The earlier `_01/` is preserved with its original source snapshot.

## Independent nominal prediction

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_relational_predictor.py
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.relational_search.predictor_acceptance outputs/relational_search/NEW_PREDICTION_RUN
```

Create `NominalPredictor(public_history)` and pass its `root` handle and instance
to `search.choose`. The model gets only whitelisted history; it fits a separate
rectangle to each entity's latest adequate observed point set. A single surface
uses a shared 0.25 m depth assumption, not hidden extent. Sparse entities can be
omitted; stale entities retain their last fitted pose and staleness metadata.
Static-hit clusters remain separate thin occupied boxes; unobserved space is
unconstrained in this nominal hypothesis. Mass/materials use common defaults.

Object velocities, success hold, pending handoffs and controller internals are
cold-started at a new real root. Public committed target ownership is preserved,
and measured robot velocities initialize the new world. This is not an exact
execution-state clone. Within that nominal world, repeated branches reproduce
the complete command prefix including controller history. Each consumable live
handle can be continued once; siblings rebuild and pay prefix work.

`cost_ticks` quotes suffix plus reconstruction; the search admits against this
quote and records charged work separately from forward horizon. Errors keep
conservative reservations. Prediction packets carry `source=nominal_prediction`
and root-offset timestamps. The model records geometry assumptions, creations,
replay/forward counts and failures. `close()` releases cached nominal worlds.
The interface is currently in-process; isolated author RPC integration and
durable global campaign quotas remain to implement.

Acceptance `_02` used 140 actual execution ticks and 280 nominal prediction
ticks across cold-A, moving-A, committed-B cases plus one search-selected actual
command. Moving-A showed about 8.5 cm one-second position error; do not treat
the model as a precise oracle. Artifacts and offline-only error diagnostics:
`outputs/relational_search/nominal_predictor_acceptance_20261113_02/`.

Next: decision-relevant prediction error diagnosis, comparable physical
data acquisition, causal history encoder, authoring_v2 method/knowledge seals,
isolated LLM feature execution, one semantic revision and frozen evaluation.
The first source copies and physical evidence are under
`outputs/relational_search/m0_m1_probe_20261113_01/`.

## Movement acceptance

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_relational_motion.py
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.relational_search.motion_acceptance outputs/relational_search/NEW_MOTION_RUN
```

The bounded acceptance checks no-hit→motion→actual object discovery, stopping
at a wall and retreat, independently moving robots, PUSH→MOVE→PUSH, exact
command replay, and nominal movement/search with an observed transport object.
It saves configuration, public observations, offline audit, per-tick rows,
boundaries, source snapshots, hashes and charged execution/prediction work.
It is not a complete clearing/transport benchmark. The predictor still refuses
roots with insufficient observed transport geometry; movement execution does
not require that model. No old authoring knowledge seal is modified.
