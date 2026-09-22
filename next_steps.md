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

---

## Round 11: predict the value, not the outcome score (positive result)

Rounds 9 and 10 established that no per-action outcome score -- motion, progress, or
enablement -- can serve the rollout. Round 11 changed the supervised target to the
quantity the planner actually consumes: the **belief-weighted value at equal
information**, produced by the same code path (`situation_quantities`) that the
reference policy minimises.

300 training situations / 150 held out / 500 epochs:

| policy | mean regret | agreement with oracle |
|---|---|---|
| value_optimal (exact reference) | 0.000 | 1.000 |
| analytic_progress | 2.187 | 0.493 |
| analytic_progress_enable | 2.400 | 0.280 |
| learned_moved | 1.573 | 0.040 |
| learned_progress | 1.440 | 0.053 |
| learned_progress_enable | 1.573 | 0.047 |
| **learned_value** | **0.553** | **0.947** |

By mechanism group the learned value reproduces the exact planner almost completely:

| group | n | learned-value regret | learned-value agreement | learned-moved agreement |
|---|---|---|---|---|
| constrained | 71 | **0.000** | **1.000** | 0.000 |
| teammate slips | 77 | **0.000** | **1.000** | 0.039 |
| heavy | 80 | 0.263 | 0.975 | 0.000 |

**Agreement rises from about 4% to 94.7%, reaching 100% on the latched and
teammate-slipping groups.** This is the clearest positive result of the project so far:

> what a prediction must retain in order to serve a rollout is a **value**, not a
> per-action outcome score. However accurate a single-action outcome prediction is, it
> cannot substitute for a sequence value, because that is what the planner consumes.

**The mode ablation still shows nothing at the decision level.** Retraining with
teammate-slipping removed entirely gives overall regret 0.547 (versus 0.553) and
slipping-group regret 0.286 (versus 0.000) -- neither better nor meaningfully worse.
Combined with the round-9 correction, the defensible statement is: the representational
collapse is a real property of the learned latent, but it has **no measurable
decision-level consequence**, with either an outcome head or a value head. Round 7's
result (a model *class* that cannot express a mode fails outright) stands; the inference
from "the training set rarely contained this mode" does not follow.

Also this round: `results.md` finally carries rounds 5-11 (part three onward), including
the realignment, both calibration gaps, the learning probe and both decision probes.

Unfinished, unchanged: continuous contact parameters; the learned value is not yet inside
a closed-loop rollout (only compared at single decision points); one scene family only;
the constraint and resistance modes have not been ablated at the decision level; no
communication; visualisations and replays not yet rebuilt for the new main line.

---

## Round 12: two real value-function bugs, and a correction to rounds 9-11

Building the closed-loop rollout exposed a correctness bug that **invalidates the
decision-level numbers reported in rounds 9-11**.

### Bug 1: the lookahead cut value was a state-independent constant

`full_value` returned a flat `fail_cost` at its depth cut. That is not a heuristic --
it ignores the state entirely. Consequence: `_value_of_action` (one step plus
`full_value(depth 0)`) returned "immediate cost + 20" for every action, so

* the "value-optimal" reference collapsed to *pick the cheapest action* (and `wait` is
  free, so it idled forever);
* the value head was trained to reproduce that same degenerate quantity;
* the two therefore agreed, which is where the reported 94.7% agreement came from.

Fixed: the cut now uses the same distance-to-goal heuristic as the belief-space value
functions, so every planner shares one state-dependent cut value.

### Bug 2: the progress weight made idling optimal

With the cut value fixed, the progress weight still had to exceed the per-step action
cost. At weight 1.0 one cell of progress is worth exactly one step, so a belief-space
planner facing a risky push is indifferent between progress and standing still, and the
free "wait" wins: the equal-information policy measured **zero cost over a whole
horizon**, i.e. it never moved at all. Weight raised to 3.0, applied identically to
every planner.

### Also found: depth 1 cannot value preparation

Even with both fixes, a one-step lookahead cannot see that `release` pays off through
the push that follows it, because that lies beyond the cut. A latched object left the
one-step oracle idling with zero cost. The closed loop therefore uses depth 2; the
depth-1 decision probes could not have measured preparation either.

### Corrected decision-level numbers (120 train / 60 held out / depth 2)

| policy | mean regret | agreement |
|---|---|---|
| value_optimal (equal-information reference) | 0.433 | 0.667 |
| analytic_progress | 0.533 | 0.450 |
| analytic_progress_enable | 0.733 | 0.517 |
| learned_moved | 1.383 | 0.117 |
| learned_progress | 1.383 | 0.117 |
| learned_progress_enable | 1.517 | 0.217 |
| learned_value | 0.967 | 0.450 |

The reference's regret is no longer zero, and that is correct: it is the equal-information
reference measured against the true-state oracle, so its regret *is* the value of the
missing contact physics.

**The qualitative conclusion survives** -- learning the value beats learning a per-action
outcome score (agreement 0.450 versus 0.117, regret 0.967 versus 1.383) and comes close
to the equal-information reference -- but the reported magnitude was an artefact and is
withdrawn. `results.md` carries the correction.

### Closed-loop rollout: built, not yet conclusive

`diagnostics/rollout_probe.py` now runs real episodes (the action is executed, a fresh
scan is taken, feedback accumulates, release is tracked per hypothesis by replaying the
committed actions). It is **not yet conclusive**: the depth-2 reference still fails on
latched objects that require release plus a push, and some generated episodes are
unsolvable (both robots slipping gives zero capacity). No closed-loop claim is made this
round.

Tests: the two assertions that encoded the old expectations ("the reference is exactly
optimal, regret 0.000") were wrong and have been replaced; small test configurations are
too noisy for a strict ordering, so the empirical claim is locked at the configuration
where it was measured.

---

## Round 13: the closed loop works; three bugs found while building it

Building and validating the closed loop exposed three real defects, all now fixed, and
produced the first end-to-end numbers.

### Bug 3: the oracle never updated the released flag

`run_episode` pinned the true hypothesis at `(physics, released=False)` for the whole
episode. After a successful release the oracle still believed the object was latched, so
it released again instead of pushing and never solved a latched episode. Fixed by
tracking `successor.released`.

With bugs 1-3 fixed, the true-state oracle solves **100% of solvable episodes**.

### Bug 4: the value labels were computed at a shallower depth than the planner

The closed loop plans at depth 2, but the training labels were built with
`situation_quantities(situation)` at its default depth 1. Depth 1 cannot see that release
enables a later push, so the *labels themselves* said that release and pushing were worse
than waiting. The learned value therefore learned to idle, and measured **0.000 success
and 0.000 cost over a whole horizon**. Fixed by computing the labels at the closed-loop
depth.

This is a general lesson worth keeping: the value target and the planner that consumes it
must share a lookahead depth, otherwise the model is trained to reproduce a different
objective than the one it is scored on.

### Closed-loop results (200 episodes, 85 of which the oracle can solve)

| policy | success rate | mean cost | cost per success |
|---|---|---|---|
| oracle_true_state (ceiling) | 1.000 | 2.941 | 2.94 |
| true_model_belief (equal information) | 0.424 | 1.153 | **2.72** |
| learned_value | 0.353 | 1.400 | 3.97 |
| learned_moved | 0.588 | 4.765 | 8.10 |

* The equal-information reference is far below the ceiling (0.424 versus 1.000): not
  knowing the contact physics costs a great deal over a full episode, much more than the
  single-step calibration suggested.
* **The learned value reaches 83% of the reference's success rate and its cost per
  success is within 46% (3.97 versus 2.72).** The model-error gap is real but modest.
* **The crude outcome-score policy wins on raw success (0.588) but at three times the
  cost** (8.10 versus 3.97 per success). It succeeds by always pushing; aggression is a
  good heuristic for "push the object east", but it is not efficient.

### Caveat on the reference

The reference is exact *given its value function*, but that value function charges the
full failure cost at every lookahead cut, which makes it over-conservative: it hedges
into cheap actions and burns the horizon, which is why an aggressive policy can beat it
on raw success. Comparisons against the reference should be read with that in mind; a
better reference would separate the true terminal cost from the cut heuristic.

### Still unfinished

Continuous contact parameters; the constraint and resistance modes are still not ablated
at the decision or episode level; one scene family only; no communication; visualisations
and replays not yet rebuilt for the new main line.

---

## Round 14: a fourth heuristic bug, and a design decision that went back

### Bug 5: the distance heuristic used the minimum over object cells

`_cut_value` measured the distance from the object to the goal as the **minimum** over
all object cells of the distance to the nearest goal cell. That returns 0 as soon as the
object merely *overlaps* the goal region, so an object sitting half on the goal looked
finished and the planner stopped pushing. The constant failure-cost term had been hiding
it. Fixed to use the **worst** object cell: every cell must be able to reach the goal.

This is unambiguously a correctness fix and it did not change any closed-loop conclusion
(see below), which is a useful check on how much the earlier numbers depended on it.

### A design decision I tried and reversed

Charging the failure cost at the lookahead cut double-counts the true terminal cost, which
is already charged once when the episode really ends. I removed it, since with the risk
term gone the planner should be less over-conservative.

Measured, it made things worse: a *failed* action and a *neutral preparation* action then
tie, because both leave the object where it is and cost a step, and the fixed tie-break
picks the failing one. The belief-space planner stopped solving a task that the
full-information planner solves in three steps. The risk term is therefore back, and it
is documented as what it is -- a lookahead heuristic whose job is to make an unproductive
action look worse than a productive one, not a terminal cost.

### Closed-loop results are unchanged

| policy | success rate | mean cost | cost per success |
|---|---|---|---|
| oracle_true_state | 1.000 | 2.941 | 2.94 |
| true_model_belief | 0.424 | 1.153 | **2.72** |
| learned_value | 0.353 | 1.812 | 5.13 |
| learned_moved | 0.471 | 5.412 | 11.50 |

The qualitative picture from round 13 holds: the learned value reaches 83% of the
equal-information reference's success rate, the crude outcome policy wins on raw success
at several times the cost, and the reference itself is well below the ceiling.

### Still unfinished

Continuous contact parameters; constraint and resistance modes not ablated at the
decision or episode level; one scene family only; no communication; visualisations and
replays not rebuilt for the new main line; and the reference's over-conservatism is now
documented but not fixed.

---

## Round 15: where the learned value actually errs

Objective item 4 asked for visualisation that shows **where** the learned model gets
things wrong, not just how often. `diagnostics/error_analysis.py` does that: for every
held-out decision point it compares the learned argmin against the belief-weighted
reference argmin and breaks the disagreements down by contact mechanism and by distance
to the goal, then renders it as a figure (PNG + PDF).

250 training / 140 held-out decision points, lookahead depth 2. Overall disagreement
**22.9%**.

### By contact mechanism

| mechanism | n | disagreement |
|---|---|---|
| teammate slips (S) | 17 | **0.588** |
| plain | 17 | 0.353 |
| heavy (H) | 19 | 0.263 |
| latched+heavy (L+H) | 15 | 0.200 |
| latched+heavy+slips (L+H+S) | 18 | 0.167 |
| heavy+slips (H+S) | 23 | 0.130 |
| latched+slips (L+S) | 14 | 0.071 |
| latched (L) | 17 | 0.059 |

### By distance to the goal

| distance | n | disagreement |
|---|---|---|
| 1 | 13 | **0.692** |
| 2 | 69 | 0.261 |
| 3 | 57 | 0.088 |

### What the model does when it is wrong

| confusion (reference -> model) | count |
|---|---|
| co_push:E -> push:1:E | 8 |
| co_push:E -> push:0:E | 8 |
| co_push:E -> wait | 6 |
| release:0:E -> wait | 4 |
| push:0:E -> wait | 3 |
| other | 6 |

Four findings:

1. **The worst mechanism is the one the project cares about**: the model errs on 59% of
   teammate-slipping decisions, several times its error rate on latched objects (6%).
2. **The dominant error is under-coordinating**: in 16 of 31 errors the reference says
   "both robots push" and the model says "one robot pushes". It under-estimates the
   coalition requirement, which is a resistance/coalition representation failure.
3. **Errors concentrate one step from the goal** (69% at distance 1 against 9% at
   distance 3): the model is least reliable exactly where the episode is about to end.
4. **The model idles where the reference acts** in 14 cases (9 `co_push`/`push` -> `wait`,
   4 `release` -> `wait`).

**A confound that must be reported with these numbers**: the reference itself chooses the
free `wait` in **74 of 140** decisions (53%). That is the documented over-conservatism --
with the risk penalty identical for a productive and an unproductive action, and progress
worth 3, waiting wins whenever the belief puts the push below about one third. So a share
of the "disagreements" is the reference being wrong, not the model. The disagreement
metric is only as good as the reference, and the reference is the next thing to fix.

Figure: `outputs/grid/error_analysis/error_by_situation.{png,pdf}`.

---

## Round 16: the reference was not over-conservative after all, and a default-argument bug

Round 15 flagged that the equal-information reference chose the free `wait` in 53% of
decisions and called it over-conservatism. **That diagnosis was wrong and is corrected
here.**

### What the reference actually is now

The reference had been the `public` planner, which branches only on the public
`moved/stalled` event and ignores both robots' private feedback. It therefore cannot
learn from a failed push. The grid planner already contained the correct object: an exact
**belief-space backup** (`central_value`) that branches on the joint observation and
updates the belief *along the lookahead*, which is what `docs/2d_protocol_spec.md` calls
`O_central`. The reference is now that backup (`central_q_values`, newly exposed), at
depth 3.

### Why depth 3

Measured on a latched, solvable scene: depth 2 fails (cost 21.00, success 0) while depth
3 solves it in 4 steps. The plan needs reposition + two pushes, which is three actions,
so a two-step lookahead cannot contain it. The decision probe had been running at depth 1
and the closed loop at depth 2; both were too shallow for this task family.

### The correction

With the belief-space reference at depth 3, on 120 held-out decision points:

* disagreement with the reference is **30.0%**;
* the reference chooses `wait` in 62 decisions, and **in 52 of them (84%) the true-state
  oracle also waits** -- acting genuinely does not help there, because the surviving
  hypotheses are unsolvable (both robots slipping gives zero capacity; a heavy object with
  only one gripping robot cannot move).

So the reference is **not** idling by default; it waits where waiting is optimal. The
round-15 warning that "part of the disagreement is the reference being wrong" is
withdrawn. This also means the round-13 headline (the learned value reaching 83% of the
reference) is measured against a genuinely strong reference and should be read at face
value.

### Bug 6: a default argument that silently ignored a constant

`_cut_value(..., progress_weight: float = DEFAULT_PROGRESS_WEIGHT)` binds the constant at
**function definition**, so assigning `planner.DEFAULT_PROGRESS_WEIGHT = 6.0` afterwards
had no effect. A weight sweep therefore reported an identical wait rate for weights 3, 6,
10 and 20 and appeared to prove the weight was irrelevant. The function now reads the
module constant at call time. The sweep was re-run and the weight genuinely does not
change the reference's wait rate -- but that had to be measured properly to be believed.

### Also fixed

A test asserted that some policy must reproduce the oracle's action. That is not an
invariant (the oracle may pick an action no policy chooses) and it was replaced with the
real invariants: every policy picks from the same candidate template, and no regret is
negative.

---

## Round 17: the bottleneck is the learned model, not the missing information

The closed-loop reference was still the old frozen-belief average. It now uses the same
exact belief-space backup as the decision probe (`central_q_values`) at depth 3, and the
feedback channel is swept for noise. **The headline finding inverts round 13.**

Solvable episodes (those the true-state oracle can solve), cost per success:

| feedback noise | oracle | true-model + belief | learned value | learned moved |
|---|---|---|---|---|
| 0.0 | 1.000 / 2.92 | **1.000 / 3.43** | 0.262 / 16.62 | 0.459 / 12.14 |
| 0.2 | 1.000 / 2.92 | **1.000 / 3.43** | 0.164 / 43.59 | 0.541 / 9.52 |
| 0.4 | 1.000 / 2.92 | **1.000 / 3.43** | 0.049 / 159.67 | 0.475 / 11.48 |

Three findings:

1. **The information gap is essentially zero.** The equal-information reference succeeds on
   **every** solvable episode, matching the full-information oracle exactly, at about 18%
   more cost. Not knowing the contact physics costs extra *steps*, not failures -- provided
   the planner reasons in belief space. This holds at every noise level tested.
2. **The learned value is the bottleneck.** It reaches 26% of the reference at best and
   **collapses under feedback noise** (0.262 -> 0.049 success; 16.62 -> 159.67 cost per
   success) because it does not model the noise correctly.
3. **The simple outcome-greedy policy beats the learned value in the closed loop and is
   robust to noise** (0.459 / 0.541 / 0.475). This reverses the decision-point result of
   round 11, where learning the value clearly beat learning the outcome score. Single-step
   decision quality did not transfer to episodes.

**Round 13's conclusion is withdrawn.** It reported that not knowing the contact physics
"costs a great deal over a full episode" (reference 0.424 versus oracle 1.000). That was an
artefact of the frozen-belief reference, which cannot value an informative action and
therefore hedges. With the correct belief-space reference the gap is ~0.

**Round 13's caveat is also withdrawn**: I had written that the reference was
over-conservative and was beaten on raw success by an aggressive policy. With the correct
reference it is never beaten on success by anything, and it is the aggressive policy that
is now revealed as far behind.

### What this means for the programme

The premise of the learned-model work was "the model must close the gap to the true model".
That premise is now much weaker: at equal information the exact planner is already
essentially optimal, so a learned value model has almost no room to help *in this task*
unless it is cheap or the model is unknown. Meanwhile the learned model as built is far
behind a five-line greedy heuristic.

Two consequences, in order:

1. **The learned value must be fixed or abandoned before more is built on it.** It does
   not model feedback noise, and its per-situation advantage does not survive into
   episodes.
2. **The honest framing for the paper is now**: with exact belief-space planning the
   information asymmetry costs almost nothing in this task; the open problem is whether a
   *learned* model can match that planner when the dynamics are not given. That question
   requires the continuous-parameter action space, where the exact planner is not
   available -- which is objective item 3 and still not started.

---

## Round 18: continuous contact parameters, and the first setting the exact planner cannot be copied

Objective item 3, the last unstarted piece, is now implemented. `grid/continuous.py`
replaces the discrete action template with continuous parameters:

* a **push angle** in degrees (0 = east, 90 = north) and a **force** in ``[0, 2]``;
* the nearest axis is selected by the cosine alignment, and the push succeeds when the
  direction is geometrically legal, the object is not latched, and
  ``sum_i traction_i * force_i >= resistance``.

The feasible set is therefore a **rectangle in (angle, force)**: an angular window of
+-45 degrees around a free axis, crossed with a force threshold -- and that threshold is
**set by the hidden contact physics**. Measured:

| physics | force threshold (A alone) | feasible at f=1.0? |
|---|---|---|
| resistance 1, free, A grips | 1.0 | yes, within +-45 degrees |
| resistance 2, free, A grips | 2.0 | no (needs f=2.0) |
| resistance 1, free, A slips | infinite | no, at any force |
| resistance 1 latched | 1.0 | no, until released |

So "what can this robot do next" is now a region whose boundary encodes exactly the
quantity the robot cannot observe.

### The learned model does not recover the region

`diagnostics/continuous_probe.py` evaluates on held-out scenes with region-level metrics
(region IoU, force-threshold error, decision success:

| configuration | region IoU | threshold error | decision success | balanced accuracy |
|---|---|---|---|---|
| unweighted BCE | 0.725 | +2.000 | 0.275 | 0.756 |
| positive-weighted BCE | 0.134 | +0.023 | 0.275 | 0.943 |

Read carefully, neither is good:

* **unweighted**: the labels are 3.7% positive, so the model minimises loss by predicting
  "never feasible". Per-sample accuracy is 0.966 and the IoU looks respectable (0.725)
  only because the latched scenes are correctly empty. The force threshold is 2.0 too
  high, so it finds almost no feasible push, and its decisions succeed 27.5% of the time.
* **positive-weighted** (weight 12.5, set from the training rate): the threshold error
  collapses to +0.023, i.e. the boundary is finally in the right place, but the region IoU
  **falls to 0.134** because the model now over-predicts feasibility, and the decision
  success does not move (0.275). A correct average threshold with a poor region is not
  useful for choosing a direction and a force.

**This is the first setting where the exact planner cannot be trivially copied**: with
continuous parameters there is no small candidate list to enumerate and no hand-written
decision rule to compare against. And it is where the learned model fails.

### Also fixed

`1 / (1 + exp(-logit))` overflowed once the positive-class weight drove logits to -30;
replaced with a numerically stable logistic that branches on the sign.

### Unfinished

The continuous probe needs a better formulation before it means anything: the binary
per-(angle, force) target under heavy imbalance is the wrong object. A threshold
regression, or a region-valued output, is the natural next attempt. Also still open:
mode ablations at the decision and episode level, communication, and replays rebuilt for
the new main line.

---

## Round 19: the continuous formulation was wrong, not the approach

Round 18 concluded that the learned model "does not recover the feasible region" on the
continuous action space. That conclusion was about the **formulation**, and the formulation
has been fixed.

The binary-per-(angle, force) target under heavy imbalance was the wrong object. The physics
says the right object is a **scalar**: inside the angular window a push works exactly when
``force >= resistance / sum(traction)``. `diagnostics/threshold_probe.py` therefore
regresses two targets directly:

* **can push at all** -- whether a push at maximum force works (false when the robot slips,
  the object is latched, or the direction is blocked);
* **force threshold** -- the smallest working force, regressed only where a push is possible
  (else the target is infinite and is masked rather than given an arbitrary value).

Crucially the threshold is only decision-relevant if **force costs something**, which it now
does: a push at force ``f`` costs ``1 + f``, and failing, or declining a possible push, costs
a fixed penalty of 2. Soft pushes fail, hard pushes waste energy, and the optimum is exactly
the threshold. The metric is therefore a regret in the task's own units.

250 training / 120 held-out decision points, 500 epochs:

| policy | mean regret | success rate |
|---|---|---|
| oracle | 0.000 | 1.000 |
| **learned threshold** | **0.881** | **0.792** |
| always maximum force | 3.192 | 0.692 |
| always minimum force | 2.250 | 0.467 |

Can-push balanced accuracy 0.732; threshold MAE **0.341 force units** on the 27 pushable
points. **The learned model beats both fixed-force baselines substantially**, which the
binary-grid formulation never did.

### Where it is still weak

By mechanism, the model is perfect on `plain` and `self_slips` (regret 0.000, success 1.000)
but **loses to always-maximum-force on `teammate_slips`** (regret 1.897 versus 1.000, success
0.333 versus 1.000). That is the third independent measurement pointing at the same weakness:
the teammate's slipping mode is what this model does not capture. It is now the single most
reproducible finding in the project, and it is a concrete target.

### Caveats

Only 27 of 120 decision points are pushable, so the threshold MAE rests on a small subset;
the can-push balanced accuracy (0.732) is moderate; and the probe still uses robot 0's own
scan with the object's placement taken from the simulator, so it is a *physics*-inference
probe, not yet a full perception-plus-physics probe.

---

## Round 20: the learner was reading the teammate's private feedback

Chasing the teammate-slipping weakness (three independent measurements) turned up a
**protocol violation in the feature encoder**: the observation vector included
``feedback_counts(history_b)`` -- the teammate's **private** feedback symbols.
`AGENTS.md` forbids exactly this ("pass ... the teammate's private history to a deployment
policy"), and it is not observable without a message.

### The fix

The observation half is now: the robot's own scan, its own pose, its own feedback history,
and the **public** event history (``moved``/``stalled``). The teammate's private history is
kept in ``Situation`` because the centralized reference (``O_central``) legitimately sees
both sides' available histories -- but it is never an input to the learned policy.

A test guards the boundary directly: holding everything else fixed and changing only
``history_b`` must leave the feature vector **identical**, while changing the public record
must change it.

### What the leak was doing

It was inflating the round-11 headline. Corrected decision-level numbers (150 training /
80 held-out / depth 2):

| policy | mean regret | agreement |
|---|---|---|
| value_optimal (equal-information reference) | 0.425 | 0.688 |
| analytic_progress | 0.512 | 0.450 |
| analytic_progress_enable | 0.688 | 0.512 |
| learned_moved | 1.050 | 0.125 |
| learned_progress | 1.038 | 0.150 |
| learned_progress_enable | 1.363 | 0.263 |
| learned_value | **1.438** | **0.412** |

**The clean round-11 claim no longer holds.** With the leak present the learned value beat
every outcome policy on both metrics. With the leak removed it still has the **best agreement**
among the learned policies (0.412 against 0.125-0.263) but the **worst regret** (1.438 against
1.038-1.363): its errors are rarer but larger in magnitude. The claim is now split, and the
test asserts only the part that reproduces.

### Why this matters for the teammate-slipping weakness

The weakness has a clean explanation once the leak is gone: **without a message, robot A
cannot observe whether robot B is slipping.** It sees only the public ``moved``/``stalled``
event, which does not say *who* failed to contribute. So the mode that the model cannot
learn is precisely the mode that is information-theoretically unavailable to it -- which is
the original motivation for communicating, and the first result in this project that points
at messaging from evidence rather than from assumption.

This strengthens the case for objective item (2)'s untouched half (limited messages) rather
than more model work.

### Unfinished

Re-run the closed loop and the threshold probe under the corrected features (the numbers in
rounds 17 and 19 were produced with the leak and must be re-measured); message channel; mode
ablations; replays rebuilt for the new main line.

---

## Round 21: a limited message channel, and it is still worth nothing here

Round 20 argued that the teammate-slipping weakness is an information limit, so a message
should help. This round built the channel and tested it. **The channel exists; the value
does not, yet, and the reason is instructive.**

### The channel

`learning/messages.py` declares three inspectable codebooks, each a function of the
sender's **own** private state or history:

| scheme | bits | codebook |
|---|---|---|
| `none` | 0 | silent |
| `traction` | 1 | grip / slip |
| `feedback` | 2 | the four feedback symbols |

The receiver appends the decoded message to its own observation. A test asserts the channel
actually changes the receiver's input (otherwise it is not a channel) and that the payload
can distinguish every symbol in its codebook.

### Result 1: the single-robot decision cannot benefit by construction

In the threshold probe the action is *this robot* pushing alone, so the teammate's traction
is irrelevant. Measured, the message does not help -- and the 1-bit traction message
actively **hurts** the teammate-slipping subgroup (regret 0.823 -> 1.230, success 0.667 ->
0.562) by adding an input that carries no information about the decision.

That is a design error of mine caught by measurement: I had been treating "the
teammate-slips subgroup" in that probe as if the teammate mattered there. It does not.

### Result 2: in the joint decision a fair baseline removes the apparent gain

The joint probe compares **solo push** against **co-push**, where the teammate's traction
genuinely decides the outcome.

| silent baseline | bits | mean regret | success |
|---|---|---|---|
| "always solo" (straw man) | 0 | 0.676 | 0.294 |
| 1-bit message | 1 | 0.353 | 0.294 |
| **expected-cost under a prior** (fair) | 0 | **0.000** | 0.412 |
| 1-bit / 2-bit message against it | 1 / 2 | **0.000** | 0.412 |

Against the straw man the message appeared to halve the regret. Against a **rational silent
policy** that reasons about the chance its teammate grips, the message adds **exactly
nothing**. The earlier gain was an artifact of the baseline, and the fair comparison is the
one that counts.

The deeper reason is that the message is not needed for *this* decision: a silent policy
already accounts for the teammate in expectation, and the co-push option is only
*cheaper*, never *necessary*. A message only pays when the teammate is **necessary and not
substitutable** -- for example a heavy object where a solo push is impossible at any force,
rather than merely more expensive.

### What the testbed needs before messaging can be judged

A decision where the teammate's contribution is **required**, not merely cheaper: resistance
above what one robot can deliver at any force, so that "solo" is not an option. The
perception and learning machinery can stay as it is; the *scenario* has to change. This is
the third time this project has found messaging to be worth zero in a testbed that did not
need it (the micro design gate, the grid gate, and now here), and the pattern is now clear
enough to state as a rule: **message value shows up only where the silent policy's
expectation is genuinely wrong, and its errors are not recoverable by hedging.**

---

## Round 22: the first clean positive message value

Round 21 stated the rule: a message pays only where the teammate is **necessary**, not
merely cheaper. This round builds exactly that decision and finally measures a positive
value.

### The construction

`diagnostics/message_necessity.py`, using only physics values the model already has:

* robot A **slips**, which A knows about itself, so a solo push is impossible at any force
  -- the teammate is genuinely required;
* robot B's traction is **private to B**;
* the object is light, so a co-push works **exactly when B grips**.

Costs follow the project's convention: a push at force ``f`` costs ``1 + f``, and a failed
push or a **declined winnable task** costs a penalty of 10.

| policy | mean regret | success rate |
|---|---|---|
| oracle | 0.000 | 0.555 |
| **one bit** | **0.000** | **0.555** |
| silent (hedges) | **4.440** | **0.000** |
| always co-push | 5.340 | 0.555 |
| always decline | 4.440 | 0.000 |

**One bit is exactly optimal, and it is worth 4.440 regret plus 55.5 percentage points of
success.** Split by the teammate's private state: where the teammate grips, silence loses
8.000 by declining a winnable task while one bit wins; where the teammate slips, silence
happens to be right and one bit is equally right.

### A degenerate cost model found on the way

The first version set the penalty to 2.0 -- exactly the cost of a successful co-push. Under
it, declining a winnable task broke even with winning it, so "do nothing" looked free, the
silent policy scored regret 0.000, and the message appeared to be worth nothing. The price
of a missed task must clearly exceed the cost of achieving it, or the objective disappears.
This is the same convention the planner already uses (failure cost 20 against action costs
of order 1); the module now states it explicitly and a test guards it.

### Why this is the first positive one

This is the fourth testbed in which messaging has been measured, and the only one where it
pays:

1. the micro design gate -- zero, because trying is physically correct anyway;
2. the grid gate -- zero, for the same reason;
3. the threshold and joint probes (round 21) -- zero, because the teammate was only
   *cheaper*, never *necessary*, so a silent policy could hedge correctly;
4. here -- large and exactly optimal, because the teammate is necessary and the deciding
   fact is private.

The rule from round 21 now has a positive instance to go with its three negative ones.

---

## Round 23: the learned model reaches the oracle with one bit, and chance without it

Round 22 showed the information *has* value. This round asks the learning question: can a
trained model actually extract it? `run_learned_necessity` trains the identical model twice,
once with the message channel and once without.

The dataset deliberately uses **no lead-in**. Any real lead-in that produced public evidence
-- a co-push that moved the object, say -- would reveal the teammate's traction through the
public record and make the message redundant. On a fresh scene the message is the only
channel carrying that fact, which is the comparison worth measuring.

A co-push is attempted when the predicted success probability reaches **0.60**, which is the
threshold *implied by the declared costs* (`p >= (1 + f + penalty) / (2 * penalty)`), not a
hand-picked number.

| scheme | bits | accuracy | balanced accuracy | mean regret | success |
|---|---|---|---|---|---|
| none | 0 | 0.453 | **0.500** | 4.373 | 0.000 |
| traction | 1 | 1.000 | **1.000** | **0.000** | 0.547 |

**With one bit the learned model is exactly optimal** -- perfect classification, zero regret,
the oracle's success rate. **Without it the same model is at chance** (balanced accuracy
0.500) and scores exactly the silent analytic baseline's regret (4.373).

This is the cleanest result in the project so far because it is decisive in both directions:

* it is not a capacity problem -- the same architecture reaches the optimum;
* it is not an artifact of a hand-written rule -- the policy is the trained model's own
  probability plus a threshold derived from the declared costs;
* and the failure without the message is exactly chance, which is what an
  information-theoretic limit looks like.

The framing the project can now defend: **a learned model is limited by what it can observe,
not by what it can represent -- and a one-bit message about the one private fact that decides
the outcome is enough to take it from chance to optimal.**

Figure/detail: the necessity testbed uses a slipping robot A (so the teammate is required),
a private teammate traction, and a light object (so a co-push works exactly when the
teammate grips).
