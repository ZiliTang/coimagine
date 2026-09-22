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
- [ ] Decide whether the static environment is a prior map or must be discovered
      (see the open question raised with the project owner).
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
