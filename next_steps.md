# Next steps — triggered by the first-round results

This file is deliberately result-driven. It is not a wish list: each item names the
observation in [`results.md`](results.md) that motivates it.

## The decision gate

The first round returned a clear, allowed negative result:

> On the micro instances, a **1-bit compact posterior** already reaches the exact
> observation-sharing optimum (`J_codec(mode) = J_central`), and the action-conditioned
> **consequence summary ties it**. The largest gap is not the message at all: it is
> look-ahead (`computation` 4.5–8.0 vs `best_message_value` 0.21–0.24).

Consequences:

- **C1 (prediction-organised messages) has no evidence at this scale.** Do not invest
  in a learned message encoder to "prove" it on the current instances.
- **C2 (skill-level planning over future messages) is the better bet**, because the
  true-model look-ahead value is large and measurable.
- Mirrors v6 stop condition 2 ("极短后验在各预算均足够").

## Track A (recommended): Stage 2 grid with a stronger hidden structure

Goal: build the 6×6 grid stage of the execution prompt, but with a hidden structure
that can actually distinguish message representations. Otherwise the "1 bit suffices"
result will simply repeat.

1. **World model.** Two robots, one 2×2 object, obstacles, target region. Hidden
   `theta = (r, c, tau)`: resistance `r ∈ {1,2,3}`, contact/constraint mode
   `c ∈ {free, constrained}`, local traction `tau` per robot. Single-push capacity 1,
   same-direction cooperative capacity 2. Public: geometry, committed actions,
   low-frequency object pose. Private: own drive/slip feedback, with a calibrated
   likelihood table (not hand-set to be monotone in `theta`).
2. **Make the message question non-degenerate.** At least one of:
   - local observations with >2 values (so 1 bit is genuinely lossy);
   - a hidden mode whose recovery differs in *two* steps (so the consequence
     *sequence* matters, not just the current best action);
   - correlated private evidence whose joint value cannot be recovered from the
     marginals.
3. **Instances A0–A3** (mirroring M0–M3): A0 public-sufficient, A1 stall-cause
   ambiguity, A2 preparation changes the contact mode, A3 identify→prepare→execute.
   Start with 4 mechanisms, not 20 maps.
4. **Candidate generation from public geometry only.** Port the useful geometry
   helpers from `_old/gate_a/src/coimagine/envs/grid/environment.py` (ray casting,
   swept-region checks, shortest path, coordinated-push validation) with attribution,
   and fix the two known defects found in the audit: `allowed_push_directions` is not
   checked in the optimistic path, and `activation_dependencies` is never enforced.
   Hidden `r` must never prune candidates.
5. **Minimal baselines** (execution prompt §五): public-only (B=0); compact
   posterior/mode message; consequence-summary message; full-history `O_central`;
   same codec without future-branch planning + replanning; true-model limited search.
6. **Large grids** use `TrueModelLimitedSearch` and are never called exact.
7. **Replay artefacts**: 2–3 lightweight PNG/GIF sequences showing the true hidden
   state (evaluation only), both beliefs, the messages, the candidate evaluation and
   the committed action.

Acceptance: A0 shows no message gain; A1/A3 show a message gain that survives
`O_codec` vs `O_B`; the grid is replayable; the figures regenerate from saved JSON.

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

- [ ] Port grid geometry primitives and fix the two audit defects.
- [ ] Define the richer hidden `theta` and the calibrated joint observation table.
- [ ] Verify the message question is non-degenerate *before* running the full suite
      (a quick exact check on a mini grid instance, analogous to M1).
- [ ] Implement A0–A3 and the six minimal baselines.
- [ ] Re-run the gap decomposition at grid scale; update `results.md`.
- [ ] Produce the replay sequences.
