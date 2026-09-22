# Next steps — triggered by the first-round results

This file is deliberately result-driven. It is not a wish list: each item names the
observation in [`results.md`](results.md) that motivates it.

## The decision gate

The first round returned a clear, allowed split result:

> On the micro instances, a **1-bit compact posterior** already reaches the exact
> observation-sharing optimum (`J_codec(mode) = J_central`), and the action-conditioned
> **consequence summary ties it**. But the value of that bit is only realised when the
> planner **anticipates the next round's message**: in M2 the same codec with a
> message-blind lookahead (`O_replan_no_message_branch`) is exactly as bad as no
> communication at all (1.5000), while the method reaches the optimum (1.2600).

Consequences:

- **C1 (prediction-organised messages) has no evidence at this scale.** Do not invest
  in a learned message encoder to "prove" it on the current instances.
- **C2 (skill-level planning over future messages) has a positive, reproducible
  micro-scale example**, with the control now implemented and tested as
  `o_replan_no_message_branch`.
- Mirrors v6 stop condition 2 ("极短后验在各预算均足够") **and** the v6 §4 branch
  ("若 RQ2 失败但 RQ3 成立，可改为接触信念规划路线").

## Track A (recommended): Stage 2 grid with sensing constraints and a stronger hidden structure

Goal: build the 6×6 grid stage of the execution prompt, satisfying the new sensing
constraint and making both the C1 and C2 questions decidable instead of degenerate.

1. **Sensing constraint (binding, from the project owner).** Each robot has its own
   global localization but perceives the object only through its own egocentric,
   occlusion-aware ray lidar (or an egocentric occupancy patch). No broadcast object
   pose, no global reconstruction of the object or the environment. Port
   `visible_cells_from_pose` from
   `_old/gate_a/src/coimagine/envs/grid/environment.py` with attribution and add
   sensor noise. Consequences to handle explicitly:
   - candidate **IDs** stay public (frozen template space) while each robot scores and
     geometrically filters them against its **own** estimate;
   - if the team wants a shared object-pose estimate it must be quantized into the
     message budget, so object localization now competes with hidden-physics evidence
     for the same bits;
   - report object-localization error separately from hidden-physics belief error, so
     a method cannot fail for "cannot see" reasons and be read as a message failure.
2. **World model.** Two robots, one 2×2 object, obstacles, target region. Hidden
   `theta = (r, c_mode, tau)`: resistance `r ∈ {1,2,3}`, contact/constraint mode,
   per-robot local traction. Single-push capacity 1, same-direction cooperative
   capacity 2. Private: raw scan + own drive/slip feedback.
3. **Make the C1 question non-degenerate** (otherwise "1 bit suffices" repeats). At
   least one of: local observations with >2 values; a hidden mode whose recovery
   differs in two steps; correlated evidence whose joint value cannot be recovered
   from the marginals.
4. **Instances A0–A3** (mirroring M0–M3): A0 public-sufficient, A1 stall-cause
   ambiguity, A2 preparation changes the contact mode, A3 identify→prepare→execute.
   A2/A3 must produce evidence **after** the decision that is used a boundary later —
   that is the C2 structure.
5. **Minimal baselines** (execution prompt §五): public-only (B=0); compact
   posterior/mode message; consequence-summary message; full-history `O_central`;
   **`O_replan_no_message_branch` (the C2 control)**; true-model limited search.
   Fix the two audit defects while porting: `allowed_push_directions` is not checked
   in the optimistic path, and `activation_dependencies` is never enforced.
6. **Large grids** use `TrueModelLimitedSearch` and are never called exact.
7. **Replay artefacts**: 2–3 lightweight PNG/GIF sequences showing the true hidden
   state (evaluation only), each robot's egocentric scan and estimate, both beliefs,
   the messages, the candidate evaluation and the committed action.

Acceptance for C2 at grid scale: `J(O_codec) < J(O_replan_no_message_branch)` at a
matched codec/budget/horizon, with equality on an information-free scenario and
equality at `B = 0`.
Acceptance for C1 at grid scale: a paired instance where the consequence summary and
the compact posterior induce **different partitions of the belief space**; if that
cannot be constructed, record that C1 is unsupported in this task family.

## Track B (only after Track A): multi-step exact O_B

The single boundary `O_B` is exact; `H ≥ 2` is a finite-horizon Dec-POMDP over message
rules and public action rules. Options, in increasing cost:

1. exact over a **declared, named** restricted class (e.g. memoryless message rules),
   clearly labelled as restricted and never as `O_B`;
2. exact `O_B` for `H = 2` with a node budget and an explicit `incomplete` status;
3. no exact multi-step `O_B`; keep it documented as an unfinished item.

Do not silently approximate and call it `O_B`.

## Explicitly not next

- 3D simulation, dual-arm auxiliary task, mobile hardware.
- learned world models / neural message encoders / training loops;
- `option_prior` or any general-model API;
- deleting or rewriting `_old/`.

## Near-term checklist

- [ ] Implement the egocentric ray lidar (port `visible_cells_from_pose`, add noise)
      and remove broadcast object pose from the public context.
- [x] Resolved: the static map (walls/obstacles/goal region) is a **prior known to
      all robots**; only the movable object must be perceived on site. The coarse
      `moved`/`stalled` event stays public.
- [ ] Define the richer hidden `theta` and the calibrated joint observation table.
- [ ] Verify both questions are non-degenerate on a mini grid *before* the full suite:
      a paired instance where the consequence summary and the posterior induce
      different belief partitions (C1), and an A2/A3 instance where
      `O_codec < O_replan_no_message_branch` (C2).
- [ ] Port the remaining geometry primitives, fixing the two audit defects.
- [ ] Implement A0–A3 and the six minimal baselines.
- [ ] Re-run the gap decomposition at grid scale; update `results.md`.
- [ ] Produce the replay sequences.
- [ ] Report object-localization error separately from hidden-physics belief error.

---

## Update after the Stage 2 design gate

The gate (`python -m coimagine.cli gate`) measured, on a finite abstraction of the
grid interaction:

* **C1 has room at a multi-step horizon.** With six hidden types and four correlated
  feedback symbols, a naive compact posterior is strictly worse than a better one-bit
  representation (`H3.c1_room` = 0.167 informed / 0.333 blind). This is the first
  configuration in the project where the message representation matters at all.
* **C2 is zero in every configuration tested.** Explicitly anticipating the next
  round's message never changed the first decision, because pushing and repositioning
  are simultaneously the physically right and the informative actions, and the
  re-planning control consumes the real messages after execution.

### What this changes

1. **Grid line = C1 + mechanism evidence.** Build A0–A3 around message
   representation under the sensing constraint, not around future-message planning.
2. **C2 becomes an explicitly designed branch, not a default claim.** It only has a
   chance if the scenario contains a decision whose *only* justification is the
   information it will produce. Concretely, add one scenario with a **costly or
   irreversible wrong attempt** (e.g. a blind cooperative push that displaces the
   object away from the goal and costs extra recovery steps) or a **costly,
   non-informative preparation**. Re-run the gate on it before featuring C2.
3. **Do not add information-gain or enabling rewards** to manufacture C2. The
   asymmetry must come from the transition and observation model.
4. A negative C2 result at grid scale is a legitimate, reportable outcome and is
   consistent with v6 stop condition 3.

### Revised near-term checklist

- [x] Grid world, egocentric ray sensing, publication-quality scenes.
- [x] Dynamics, hidden physics, candidates, per-robot belief filter.
- [x] Design gate for C1 and C2.
- [ ] A0–A3 mechanism scenes on the real grid, with a grid belief-space planner
      (public-only / compact posterior / consequence summary / full sharing /
      no-future-message branch / limited search).
- [ ] Scenarios with a costly wrong attempt, to give C2 a fair test.
- [ ] Oracle gap table, communication-budget curve, probe-cost curve, 2–3 replays
      (including each robot's scan and belief evolution).
- [ ] Update `results.md` with the grid numbers; list unfinished items.

---

## Update after the grid planner (round 3)

Implemented `coimagine.grid.planner`: a six-way planner panel over the contact
hypothesis (32 hidden contact types), a closed-loop **exact** expectation (Stage 2 has
no process noise), and per-boundary regeneration of the public candidate ID space.

Verified:

- the full-information planner reaches the optimum on the reference scenario
  (2 pushes = cost 2.0; a latched object needs a release = 3.0);
- `public`, `codec`, `observation`, `consequence` and `replan` all solve the
  reference scenario;
- the C2 control coincides with the method here, matching the design gate;
- a latched object is rationally ignored under a uniform prior because it is only
  2 of 32 hypotheses — scenario design for A2/A3 must give that mode prior mass or
  early evidence, it is not a planner bug.

Open correctness item (next round):

- **`central` disagrees with `codec` on the reference scenario** (2.3 versus 2.0,
  where complete observation sharing must be no worse). The trace shows `central`
  choosing `probe` at step 1 on a tie while `codec` chooses `co_push`; the
  full-sharing value is therefore not yet trustworthy and the ordering
  `J_central <= J_codec` is deliberately **not** asserted in the test suite until it
  is fixed.

Also improved in this round: a successful push now advances the pushers with the
object, so multi-step pushing no longer needs a reposition between every cell.


---

## Round 4: planner correctness fixes and what is still open

Fixed three real bugs, all found by testing:

1. **The value functions never recognised "already in the goal"**, so a planner at the
   goal paid an extra probe and `central` looked worse than `codec` (2.3 vs 2.0).
   A goal short-circuit in every value function and in action selection fixes it;
   `J_full <= J_central <= J_codec` is now asserted in the tests.
2. **The initial belief wrongly included `released=True`** hypotheses. An episode
   starts unreleased; the prior is now 16 types.
3. **The closed-loop driver passed an empty action history to the codec** at every
   boundary, so the message computation collapsed and the belief degenerated to
   B = 0. The committed action history is now threaded through.

Still open:

- **Depth-3 lookahead is too slow** (the run did not finish within the budget) and the
  latched (constrained) A2/A3 case needs exactly three actions: release, then two
  cooperative pushes. Required before A2/A3: memoise/prune the depth-3 search
  (canonical belief keys, branch merging, or a bounded-node fallback), and check the
  release branch's message split (the posterior-mode codec collapses released and
  unreleased hypotheses in that branch).
- A0/A1 (public-information-sufficient and stall-cause) work at depth 2 today; the
  grid scenario suite can start there.

---

## Round 5: calibration of the non-privileged predictor

The main line was realigned in `docs/CoImagine_研究主线对齐_v1.md`: the paper's claim
is now **"what must the shared prediction retain so that each robot can correctly
derive what it can do next"**, and the thesis only bites when the state is uncertain.
Before building any learned model, the testbed has to show that a predictor which is
*not* privileged but *does* know the transition rules measurably loses to the ceiling.

`python -m coimagine.diagnostics.run_calibration` (module
`src/coimagine/diagnostics/calibration.py`) measures exactly that: the true-state
planner versus a planner that uses the same known rules but the robot's own
scan-estimated object pose and contact-physics prior, with identical candidates,
lookahead depth and cost.

| radius | belief placements | p(correct origin) | p(correct physics) | ceiling | non-privileged | gap |
|---|---|---|---|---|---|---|
| 2 | 2 | 0.50 | 0.063 | 3.00 | 4.00 | **1.00** |
| 3 | 1 | 1.00 | 0.063 | 3.00 | 7.00 | **4.00** |
| 6 | 1 | 1.00 | 0.063 | 3.00 | 7.00 | **4.00** |
| 2 | 2 | 0.50 | 0.063 | 3.00 | 4.00 | **1.00** |
| 3 | 1 | 1.00 | 0.063 | 3.00 | 4.00 | **1.00** |
| 6 | 1 | 1.00 | 0.063 | 3.00 | 4.00 | **1.00** |

Mean gap 2.00; the gap is strictly positive in every run, which satisfies the
precondition for starting the learned model. Two honest caveats:

1. **Most of the gap is the contact-physics belief, not localisation.** With a wide
   scan the object's pose is known exactly (`p(correct origin) = 1.00`) and the gap is
   still 1.00–4.00, because the physics is still the flat prior. The scan carries no
   physics information by construction, so that part must come from contact feedback.
2. **The numbers are not monotone in sensing quality** (radius 3/6 gives a *larger*
   gap than radius 2 in one configuration). The outcome is dominated by discrete
   first-action tie-breaks, so the magnitudes are not yet a reliable quantitative
   target. Averaging over many placements and seeds is required before using the gap
   as a benchmark.

Real bugs found and fixed while calibrating:

- the true-state planner evaluated every boundary from the **initial** robot poses, so
  it could never follow through a multi-step plan (it looked as if the ceiling could
  not solve a solvable task);
- the terminal value was a flat failure penalty, so no planner could value partial
  progress; a distance-to-goal heuristic was added and is applied identically to all
  planners;
- the planner's candidate generator pruned geometry using the **true** object pose; a
  pose-agnostic public template set is now available and is what the calibrated
  planner uses.

Next: average the calibration over many placements/seeds to get a stable target,
separate the pose component from the physics component explicitly, and only then
start the multi-task latent world model (privileged-pose decoder head plus
multi-step feasibility loss).

---

## Round 6: what the calibration actually shows, and the corrected decision

The calibration was stabilised (multi-frame estimation, 36 runs over placements,
sensing radii and scan seeds) and the two candidate gaps were separated. The result
changed the decision.

### Gap 1 -- value of information: real, small, and irreducible

With the true model on both sides and only the *state* estimated, the non-privileged
planner pays exactly **one extra step** (cost 3.00 versus 2.00) when the object is
free, and **zero** when it is constrained:

| true contact physics | true model + true state | true model + estimated state | gap |
|---|---|---|---|
| r=1 free | 2.00 | 3.00 | 1.00 |
| r=2 free | 2.00 | 3.00 | 1.00 |
| r=1 constrained | 3.00 | 3.00 | 0.00 |

That step is the optimal hedge against not knowing the contact physics -- it is the
price of the missing information, not a defect a learned model could remove. **A
learned world model cannot close this gap**, so it is the wrong target.

Multi-frame estimation works as intended: at a two-cell sensing radius a single scan
leaves two plausible placements, and three viewpoints along a public route resolve it
to one (`p(correct origin)` 0.50 -> 1.00), matching a wide scan.

An earlier, larger number (gap 2.50) was an artefact: with staging left in the task,
the estimator parked a robot on the cell its teammate needed and paid a three-step
repositioning detour. Removing staging from the comparison collapsed it to the clean
one-step identification cost above.

### Gap 2 -- model error at equal information: this is the one that matters

The correct target is a **wrong model given exactly the same information**. Varying
one aspect of the contact physics at a time, with the robots already at contact:

| model used by the planner | cost | success | gap |
|---|---|---|---|
| true model | 2.00 | 1.0 | 0.00 |
| thinks the object is latched | 25.00 | 0.0 | **23.00** |
| thinks the object is heavy | 2.00 | 1.0 | 0.00 |
| thinks the teammate slips | 2.00 | 1.0 | 0.00 |
| thinks heavy + latched + slipping | 20.00 | 0.0 | **18.00** |

**A model that gets the contact/constraint mode wrong fails completely; a model that
gets the resistance or the traction wrong is tolerated in this task.** So the testbed
*can* discriminate models, but only through the discrete contact mode. That is a sharp
requirement for any learned model: it must represent the latch/release state, and it
is not yet required to represent resistance or traction magnitudes.

### Decision

The precondition for a learned world model is met only in the narrow sense above. Do
**not** start the neural model yet, because:

1. the large gap is caused by a single discrete mode, which a much simpler structured
   model (or a correct hand-coded mode variable) already captures;
2. resistance and traction errors currently cost nothing, so the learning problem is
   too narrow to justify a latent world model;
3. the target metric -- model error at equal information -- is now defined and
   implemented, which is the prerequisite for any learned-model comparison.

Next round: widen the testbed so that resistance and traction also change the decision
(tasks where the number of pushers and who is gripping determines success, and where
partial progress must be predicted), re-run the model-error table, and only if a
wrong *continuous* or *multi-factor* model measurably loses, build the multi-task
latent world model on top.

---

## Round 7: widening the testbed -- which physics factors actually matter

Round 6 showed that a wrong *resistance* or *traction* cost nothing, because with the
robots already at contact the robust cooperative push works either way. The testbed was
therefore too narrow to study a model. It is now widened: robot A starts on the
object's west face and robot B starts two cells away, so "A pushes alone" and "bring B
and push together" are genuinely different plans with different costs, and the number
of effective pushers and who is gripping both change the outcome.

### Factor sensitivity -- with a prior that expects the wrong regime

| factor | truth | prior expects it | prior expects the other | gap |
|---|---|---|---|---|
| resistance | r=2, free, grip/grip | 3.00 | 4.00 | **1.00** |
| traction | r=1, free, B slips | 3.00 | 4.00 | **1.00** |
| constraint | r=1, latched | 3.00 | 4.00 | **1.00** |

All three factors now change the decision. The cost of expecting the wrong regime is a
uniform one-step recovery, because the belief updates from the feedback and the planner
switches plans.

### Representability -- when the model class cannot express the truth at all

| factor | truth | class contains it | class excludes it | gap | success |
|---|---|---|---|---|---|
| resistance | r=2, free, grip/grip | 4.00 | 4.00 | 0.00 | 1.0 |
| traction | r=1, free, B slips | 4.00 | 22.00 | **18.00** | **0.0** |
| constraint | r=1, latched | 4.00 | 22.00 | **18.00** | **0.0** |

**A model class that cannot represent the teammate's traction, or cannot represent the
contact/constraint mode, fails outright.** Excluding heavy resistance from the class
costs nothing, because the robust plan brings the teammate anyway.

### What this means for the learned model

The requirement is now precise, and it is a *representation* requirement rather than an
accuracy requirement:

* the model **must** be able to express *who can actually push* (the teammate's local
  traction) and *whether the object is latched* (the constraint mode); getting these
  wrong or omitting them is catastrophic, not merely costly;
* resistance only has to be represented well enough for the planner to recover from a
  wrong prior, which costs one step.

This is exactly the design hypothesis behind the multi-task latent: the latent must
preserve the discrete contact structure, and it is not enough to predict geometry well.

The precondition for the learned model is now both met and *measurable*: the testbed can
distinguish a model that carries these modes from one that does not, by a gap of 18.00
with success falling to zero. Next round: build the multi-task latent world model
(privileged-pose decoder head plus multi-step feasibility loss) and check whether a
model learned from data discovers these modes, or whether the action space must first be
extended to continuous contact parameters.

---

## Round 8: the learning probe -- coverage of a contact mode decides whether it exists

The calibration established *what* a model must represent (round 7). Round 8 asks the
learning question directly, with a hand-written NumPy MLP (`src/coimagine/learning/`,
no heavy dependency, gradients verified numerically against finite differences).

The probe takes samples from short **lead-in episodes**, so the robots have accumulated
genuine contact feedback before each sample is labelled; on a fresh scene no model could
tell a heavy object from a light one. Inputs are the robot's own scan, its own feedback
history and public information only. Labels are the true action outcome, the true object
placement and the true contact physics.

600 scenes / 3600 samples, 70/30 scene split, 500 epochs:

| evaluation set | n | feasibility accuracy | balanced accuracy | BCE |
|---|---|---|---|---|
| in distribution | 1080 | 0.939 | **0.740** | 0.503 |
| trained **with** slipping, on held-out slipping cases | 552 | 0.958 | **0.779** | 0.415 |
| trained **without** slipping, on held-out slipping cases | 552 | 0.911 | **0.735** | 0.543 |

| evaluation set | physics decode | pose decode |
|---|---|---|
| in distribution | 0.306 | 0.944 |
| trained with slipping | 0.185 | 0.978 |
| trained without slipping | **0.000** | 0.978 |

Two findings:

1. **The latent does carry contact information.** Physics decode reaches 0.306 against
   a 1/16 = 0.0625 chance rate, and the pose decoder recovers the object placement at
   0.944 from the raw scan. Feasibility balanced accuracy is 0.740, well above the
   0.915-accuracy majority baseline which is a trivial 0.085-positive classifier.
2. **A mode that never appears in training is not merely mispredicted, it is
   unrepresentable.** The model trained without the teammate-slipping mode predicts
   that mode **0 times out of 552** (physics decode exactly 0.000), while the model that
   saw it reaches 0.185. The per-sample feasibility cost is modest (+0.045 balanced
   accuracy), but round 7 showed the *decision-level* cost of the same omission is
   total failure (gap 18.00, success 0.0). Per-sample classification understates what
   the omission does to a rollout.

Honest limitations:

- feasibility is heavily imbalanced (8.5% positive), so plain accuracy is near the
  majority baseline; balanced accuracy and BCE are the metrics that mean anything;
- in-distribution physics decode is still low in absolute terms (0.306), so the latent
  carries the mode only partially;
- one mode (teammate slipping) was ablated; resistance and the constraint mode should be
  ablated the same way before any general claim;
- the model is not yet wired into the planner, so no decision-level number has been
  produced for a *learned* model yet.

**Next round:** connect the learned feasibility head to the planner and measure the
decision-level gap against the true-model ceiling at equal information -- the comparison
round 6 defined. Then ablate the constraint mode and resistance the same way. Only after
that is the "learned feasibility head versus the known decision rule" contrast meaningful.

---

## Round 9: the learned head at the decision -- and a correction to round 8

Round 8 measured the learned model per sample. Round 9 wires it into the decision and
measures **regret**: the true-model value of the action each policy picks, minus the
value of the best action. Values are computed by the simulator, so a preparation action
such as release is valued by the dynamics rather than asserted by us.

Two baselines of different kinds are needed to read the result correctly:

* **value-optimal** -- argmin of the belief-weighted true-model value. This is what the
  planner does with a correct model at equal information.
* **analytic-greedy** -- the *same greedy rule the network implements* (argmax of the
  probability that the object moves), but with the analytic probability. This isolates
  the model from the rule.

400 training situations, 200 held-out, two-step lead-in, 500 epochs:

| group | n | value-optimal regret | analytic-greedy regret | learned-greedy regret |
|---|---|---|---|---|
| all | 200 | 0.000 | 0.405 | **1.100** |
| constrained | 100 | 0.000 | 0.420 | 1.000 |
| heavy | 105 | 0.000 | 0.362 | 0.981 |
| teammate slips | 95 | 0.000 | 0.326 | 0.958 |

| group | n | value-optimal agreement | analytic-greedy agreement | learned-greedy agreement |
|---|---|---|---|---|
| all | 200 | 1.000 | 0.580 | **0.090** |

Three findings, one of which contradicts what round 8 predicted:

1. **The value-optimal policy is exact here even with a diffuse belief.** Two noiseless
   lead-in actions usually leave about three consistent contact types (median maximum
   posterior mass 1/3, and the true type is never excluded), yet the belief-weighted
   argmin equals the oracle argmin in every one of the 200 situations. The ambiguity in
   this scene family does not change which action is best -- the geometry dominates.
2. **The greedy decision rule is itself wrong for the task**: 0.405 regret and 58%
   agreement, because "will the object move" is not "does this action make progress".
3. **The learned model is much worse than the analytic model under the identical rule**:
   1.100 regret and 9% agreement. The learned feasibility head is not yet usable as a
   decision component.

**Correction to round 8.** Round 8 found that a model trained without the
teammate-slipping mode predicts that mode zero times, and I speculated that this
representational collapse would show up as decision-level damage. It does not, at least
here: the full and the slipping-ablated models give the same decision-level numbers
(agreement 0.090 versus 0.080; the slipping subgroup is identical at 0.042). The reason
is that the learned greedy rule is already failing for a more basic reason, which
swamps the mode-specific effect. The honest statement is therefore narrower than the
round-8 note: **a representational collapse is a real property of the learned latent,
but it has not been shown to cause decision-level damage.** The round-7 planner-level
result (a model *class* that excludes a mode fails outright, gap 18.00) stands; what
does not follow is that a *classifier* trained without a mode inherits that failure.

**Where this leaves the programme.** The learned model as currently built -- a
per-action feasibility classifier used greedily -- cannot serve a rollout. What it
lacks is not accuracy but the two things the alignment document identified: a
prediction of *progress* (not merely motion) and a prediction of *enablement* (whether
an action creates new feasible actions, which is how release earns its place). Next
round: add an enablement head and a progress-oriented target, re-run this exact regret
comparison, and only then re-attempt the mode ablation -- there is no point ablating a
mode while the baseline rule is still broken.

---

## Round 10: reshaping the target did not rescue the greedy rule (negative result)

Round 9 concluded that the learned head lacked a prediction of *progress* and of
*enablement*. Round 10 implemented exactly that:

* the action head now predicts three channels -- **moves**, **progress** (the object
  ends strictly closer to the goal) and **enables** (a push that failed before this
  action now succeeds, which is how release earns its place);
* the labels behave as designed: `enables` fires **only** for release (4 of 120 samples
  in the small dataset), `progress` only for pushes;
* six policies are compared, each with a matched analytic twin so the rule and the
  model can be judged separately.

500 training situations, 250 held-out, 600 epochs:

| group | policy | mean regret | agreement with oracle |
|---|---|---|---|
| all | value_optimal | **0.000** | **1.000** |
| all | analytic_progress | 2.428 | 0.516 |
| all | analytic_progress_enable | 2.652 | 0.292 |
| all | learned_moved | 0.892 | 0.092 |
| all | learned_progress | 0.972 | 0.080 |
| all | learned_progress_enable | 1.132 | 0.080 |

Per mode, the same pattern: on constrained objects `analytic_progress` reaches 0.620
agreement but `analytic_progress_enable` falls to 0.289; on heavy objects 0.617 versus
0.323; on slipping teammate 0.659 versus 0.318. Every learned variant sits at 0.00-0.09.

**The redesign made things worse, not better.** Three things follow:

1. **The reference is sound.** The belief-weighted value rule is still exact (0.000
   regret, 100% agreement) even though the belief is diffuse, so the comparison has a
   meaningful anchor and the greedy policies are genuinely bad rather than badly scored.
2. **Enabling supervision backfired.** `progress + enables` is worse than `progress`
   alone for both the analytic and the learned policies. The enablement label is very
   sparse (release only), so adding it mostly promotes release in situations that do not
   need it.
3. **The honest scope of this negative result.** A greedy argmax over a *sparse binary*
   target is a weak decision rule, and this experiment shows that target engineering does
   not rescue it. It does **not** show that progress or enablement supervision is useless
   for a rollout -- only that it cannot help a rule that scores each action
   independently and ignores consequences. The structure of the decision rule, not the
   choice of target, is what failed.

**Where this leaves the programme.** Two options, in order of preference:

1. stop trying to make a per-action classifier serve the rollout, and instead have the
   learned model predict a **value** (or a rollout-consistent multi-step quantity), which
   is what the exact planner actually consumes -- this is the multi-step feasibility loss
   the alignment document proposed, and it is untested;
2. extend the action space to continuous contact parameters first, so that a per-action
   score is at least a well-posed function to learn.

Unfinished: the action space is still discrete; the learned model still is not inside a
rollout; results.md has not been updated with any of rounds 5-10.
