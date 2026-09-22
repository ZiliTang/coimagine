# Gate A v0.5: local-belief imagination benchmark

## Purpose

Gate A isolates whether predicting robot-specific future local observations and
using them to regenerate local options improves distributed multi-step decisions.
The evolving coordination topology is the structured decision consequence of the
belief imagination, rather than a separately rewarded objective.

## Information boundary

- The simulator owns the complete grid state.
- Each robot receives an occlusion-aware ray-cast local observation.
- Walls, movable entities, and teammates occlude rays.
- Robots exchange planning packets and option/intent tokens through all-to-all
  communication; raw local maps are not fused.
- Root options must be grounded by current real local observations.
- Only the selected root joint option is executed; real observations replace all
  imagined beliefs at the next option boundary.

Once any part of a controlled benchmark object is observed, its discrete object
template, mass class, required coalition size, and allowed push directions are
known. Only currently visible surfaces ground contact options.

## Option semantics

- `Reposition(anchor)`: move to a sparse anchor that is currently visible and
  reachable through the current observed free component.
- `Approach(entity, contact)`: reach the staging cell associated with a currently
  observation-supported contact.
- `Push(entity, contact, direction)`: a template may exist before the robot is at
  contact; admission checks current contact, direction, coalition, resources, and
  swept-space feasibility.
- `Hold`, `Observe`, and `Idle`: retained for the general controlled suite.

## Compared planners

### CST

Current-Observation-Supported Topology performs successor-centered state and
executability rollout, but future option vertices are projected onto the root
observation-supported template set. Edges among retained vertices may still be
created or destroyed.

### CoImagine-IET

Imagined Evolving Topology propagates known local entities and teammate option
effects. In the analytic Gate-A model, unobserved space that is not occluded by a
known predicted entity receives an optimistic free-space prior. The predicted
robot-centered observation regenerates future options, factors, and executable
hyperedges.

### Privileged IET Oracle

The root information and root action set are identical to CST/IET. After an
imagined root action, the simulator renders the true successor local observations.
These observations regenerate the next option bank and topology. It is an
imagination upper bound, not a deployable planner.

## Horizon choice

The canonical positive-return dependency is

```text
Reposition/Approach -> Approach(new contact) -> coordinated Push.
```

H=2 cannot include the task-positive Push in the same branch and therefore does
not value the zero-immediate-progress preparation. H=3 is the main setting. H=2
remains a matched ablation that uses the same local model, option generator,
verifier, and receding-horizon execution.

## Open and surprise variants

Procedural variants translate and reflect the canonical geometry without
post-selecting by planner outcome. Three quarters are open continuations. One
quarter contain a wall at the imagined second support position that is outside
the root observations. These surprise variants measure the cost of the optimistic
unknown-as-free prior.

## Primary metrics

- closed-loop completion, realized return, option count, and duration;
- planning time per executed option;
- agreement with the privileged oracle's root action;
- newly generated option-template precision and recall;
- newly generated topology-edge precision and recall;
- retained oracle gain when the oracle has a positive advantage over CST;
- unnecessary reposition cost on surprise variants.

CST should create no future option vertices outside current observation support.
IET should match the privileged oracle on open variants and may produce false
future support on surprise variants.

## Evidence allocation

Main paper:

1. CST versus IET action-sequence/FoV/topology storyboard;
2. H=2 versus H=3 completion and planning-time comparison;
3. compact successor-support accuracy table.

Supplementary:

- optimistic surprise cases;
- local-availability, pairwise, recursive-factor, and full-tuple admission checks;
- entity-count, robot-count, and factor-arity scaling;
- legacy edge-creation/destruction, hold-pass, and reveal-act diagnostics.
