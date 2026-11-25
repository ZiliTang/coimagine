# Four-stage CAP development profile

This is an in-progress, versioned extension of authoring_v2. Read the ordinary
option guide, authoring_v2 README/experiment API and checked-policy guide/example.
Historical symbolic_general and symbolic_unknown source seals remain unchanged.
The current physical action version is relational-options-local-push45-compact20-v4;
the old public guide's 35-degree and historical 22.5-degree results are not this
profile. Robot diameter is 20 cm; scans have exactly 72 body-frame rays.

## Four stages

1. PublicFrontend.prepare consumes whitelisted public observations in causal order.
   Geometry, evidence history and explicit motion assumptions are host adapters.
2. register returns detached typed semantic values. schema lists every primitive;
   derived expressions have explicit dependency order and units.
3. The author writes decide(z), returning (option_name, score) candidates. The
   extractor consumes its AST without executing arbitrary source and retains each
   emission's path condition. Concrete native-function parity is checked separately.
4. Gateway considers ranked candidates until one passes. A one-step forecast sees
   the raw public observation/history; only the third-stage function is restricted
   to registered symbols. Future selection checks startup predicates without
   recursively forecasting the successor. Actual future observations trigger a
   fresh decision. Negative scores and preparation actions remain eligible.

Modes: author retains current geometric/startup checks without predictive vetoes;
shadow records predictive checks without enforcing them; enforce applies them.
Unknown evidence is not silently free. No admissible action is an explicit result,
not an automatic unchecked brake certificate.

## Pure candidate function

The current supported subset is local scalar expressions, if/else, range or literal
loops of at most 128 iterations, and candidates.append((name, score)). The entire
unrolled function is capped at 8192 statements. Action names and semantic keys may
use static loop-variable f-strings. Scores are dimensionless; Q(value, unit) declares
typed constants. abs, min, max and conditional expressions are supported. Division
requires a nonzero dimensionless literal denominator.

Start with candidates = [] and end with return candidates. Branch-local variables
cannot escape or reassign an outer variable; use a conditional expression instead.
Early return, arbitrary calls, raw-observation access, hidden globals, mutation of
z and unsupported control flow are rejected, never silently approximated. This is
an explicit implementation subset, not a claim that all Python can be verified.

Candidate generation need not cover all physical actions. The checker covers
possible emissions of the submitted program and preserves branch associations.
Tie breaking follows emission order. Startup screening is reflected in future
selection. Reports must distinguish startup, predictive admission and completion.

## Geometry and evidence

Initial separation, exact public ranges/odometry, static-obstacle component size,
target CAD and motion envelopes are declared assumptions. The free-space map uses
the existing closed-ray-cell construction; cached queries preserve its semantics.
The map concerns static geometry; it is not a certificate about arbitrary movement
of other objects. Peer, target and process-effect audits remain separately visible.
The first profile offers 69 short commands: two PUSH durations, both/single roles,
single-robot MOVE, independent ORBIT and BRAKE.

Physical audits use backend state only in physical.py for offline labels, including
each actual tick and the stopping tail. No environment or audit state crosses the
gateway or prediction interface.

## Development status

The task book is docs/agent_exchange/tasks/CAP_Four_Stage_20261125.md.
Initial integration trajectories are diagnostics, not submitted CAP rounds or
confirmation evidence. Full author sessions, model fitting, deployment rollout,
checked submission and final paired evaluation are still being integrated.
