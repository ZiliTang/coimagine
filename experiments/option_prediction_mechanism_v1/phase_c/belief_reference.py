"""Phase C, C4: the finite-candidate belief-planning reference.

CONTRACT section 3.4 / task book C4 forbid handing the planner the true world after skill 1.
The rule implemented here is exactly the stated one:

1. run the first skill ``a1`` in both worlds;
2. compare the **allowed observations** of the two resulting states, with the same declared
   sensing model and the same comparison as at the decision time;
3. if the worlds are still indistinguishable, the second skill must be the *same* for both
   worlds (one common decision rule);
4. only if the post-``a1`` observation actually distinguishes them may the plan branch on the
   real observation.

The reference is enumerated over the finite candidate set (all ``a1``, all ``a2``); no general
POMDP solver is built.  Two values are reported for every ``a1``:

* ``common`` = ``max_{a2} 0.5 (Q_A(a1,a2) + Q_B(a1,a2))`` -- the same second rule in both
  worlds, which is the only legal option when the states remain indistinguishable;
* ``branch`` = ``0.5 max_{a2} Q_A(a1,a2) + 0.5 max_{a2} Q_B(a1,a2)`` -- the value if the
  planner were allowed to pick the best ``a2`` separately in each world.

``branch - common`` is the value of being allowed to branch; it is legitimate only when the
post-``a1`` observation is distinguishable, and it is reported separately for the
distinguishable and indistinguishable cases.  The benchmark "common action" reference of
CONTRACT 3.3 is the single-skill analogue, ``max_a 0.5 (Q_A(a) + Q_B(a))``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
for extra in (ROOT / "src", ROOT / "experiments" / "option_prediction_mechanism_v1" / "phase_c"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import actions as ac  # noqa: E402
import perception as pc  # noqa: E402
import run_phase_c as R  # noqa: E402
from coimagine.continuous2d import Continuous2DEnv  # noqa: E402


def _post_skill1_diff(pair: R.CandidatePair, first: ac.NamedJointSkill) -> Dict[str, float]:
    """Observation difference after executing ``first`` in both worlds (allowed inputs only)."""

    params = pair.params
    scene_a, scene_b = params.scenes()
    observations = {}
    for world, scene in (("A", scene_a), ("B", scene_b)):
        env = Continuous2DEnv(R.mechanism_config(params))
        env.reset(scene)
        result = env.execute_joint_skill(first.joint)
        observations[world] = pc.observe(
            scene,
            (env.backend.object_state().x, env.backend.object_state().y,
             env.backend.object_state().angle),
            env.backend.robot_states(),
            tick=env.tick,
            time=env.tick * R.mechanism_config(params).physics.control_dt,
            contact_state=env.backend.contacts(),
            history=(first.skill_id,),
        )
    return pc.observation_differences(observations["A"], observations["B"])


def belief_reference(pair: R.CandidatePair, *, settle: bool = True) -> Dict[str, object]:
    """Full finite-candidate belief-planning reference for one pair."""

    params = pair.params
    skills = list(pair.skills)
    by_id = pair.skill_by_id()
    # Post-first-skill distinguishability, at the exact decision boundary after skill 1.
    post = {}
    for first in skills:
        diff = _post_skill1_diff(pair, first)
        # Distinguishability is derived from FINITE quantities plus explicit mismatch flags;
        # a quantity present in one world and absent in the other is a real difference (flag)
        # and is never reported as an infinite number.
        mismatches = {name: value for name, value in diff.items()
                      if name.endswith("_mismatch") and value != 0.0}
        max_finite = max((value for name, value in diff.items()
                          if not name.endswith("_mismatch")), default=0.0)
        identical = pc.observation_is_identical(diff, tolerance=1e-12)
        post[first.skill_id] = {
            "max_abs_difference": max_finite,
            "mismatch_counts": mismatches,
            "any_mismatch": bool(mismatches),
            "distinguishable": bool(not identical),
            "per_quantity": diff,
        }

    # Two-skill Q matrices for both worlds (true simulation).  Sequences that diverged
    # numerically in either world are removed from the finite candidate set; they are not
    # task outcomes and must not enter a belief-planning reference.
    q2 = {"A": {}, "B": {}}
    diverged = {"A": set(), "B": set()}
    for world in ("A", "B"):
        env = Continuous2DEnv(R.mechanism_config(params))
        for first in skills:
            for second in skills:
                key = f"{first.skill_id}/{second.skill_id}"
                outcome = R.run_sequence(env, world, params, [first, second], settle=settle)
                if outcome.diverged:
                    diverged[world].add(key)
                    continue
                q2[world][key] = outcome.q
    valid_keys = sorted(set(q2["A"]) & set(q2["B"]))

    per_first: Dict[str, object] = {}
    for first in skills:
        keys = [key for key in valid_keys if key.startswith(first.skill_id + "/")]
        if not keys:
            per_first[first.skill_id] = {
                "post_skill1": post[first.skill_id],
                "common_value": None,
                "branch_value": None,
                "branch_minus_common": None,
                "reason": "no valid (non-diverged) second skill in both worlds",
            }
            continue
        common = max(0.5 * (q2["A"][key] + q2["B"][key]) for key in keys)
        branch = 0.5 * max(q2["A"][key] for key in keys) + 0.5 * max(q2["B"][key] for key in keys)
        best_common_key = max(keys, key=lambda key: 0.5 * (q2["A"][key] + q2["B"][key]))
        best_a = max(keys, key=lambda key: q2["A"][key])
        best_b = max(keys, key=lambda key: q2["B"][key])
        per_first[first.skill_id] = {
            "post_skill1": post[first.skill_id],
            "common_value": common,
            "branch_value": branch,
            "branch_minus_common": branch - common,
            "best_common_sequence": best_common_key,
            "best_A_sequence": best_a,
            "best_B_sequence": best_b,
            "Q_A_of_best_common": q2["A"][best_common_key],
            "Q_B_of_best_common": q2["B"][best_common_key],
            "q2": {key: {"A": q2["A"][key], "B": q2["B"][key]} for key in keys},
        }
    # Overall plan values, all computed from the SAME finite candidate set ``valid_keys``.
    #
    # A policy is (a1, second-skill rule).  ``a1`` is chosen before any world-specific
    # observation, so it is the same in both worlds.  After ``a1`` the planner may choose a
    # different ``a2`` in the two worlds only when the post-``a1`` allowed observation really
    # distinguishes them; otherwise it must reuse one rule.  The best such policy is a **max**
    # over ``a1`` of the per-``a1`` feasible value (it is not an average over ``a1``: the
    # planner commits to one ``a1``).  Because the non-branching plan is always feasible, the
    # legal observation-conditioned value is >= the common-plan value by construction.
    keys = list(valid_keys)
    common_plan = max(0.5 * (q2["A"][key] + q2["B"][key]) for key in keys) if keys else None
    mean_state_best = max(keys, key=lambda key: 0.5 * (q2["A"][key] + q2["B"][key])) if keys else None
    mean_state_value = (0.5 * (q2["A"][mean_state_best] + q2["B"][mean_state_best])
                        if mean_state_best else None)
    all_knowing = (0.5 * max(q2["A"][key] for key in keys)
                   + 0.5 * max(q2["B"][key] for key in keys)) if keys else None

    feasible: List[Tuple[float, str, bool]] = []
    for first in skills:
        entry = per_first[first.skill_id]
        if entry.get("common_value") is None:
            continue
        distinguishable = bool(entry["post_skill1"]["distinguishable"])
        value = entry["branch_value"] if distinguishable else entry["common_value"]
        entry["legal_value_for_this_a1"] = value
        feasible.append((value, first.skill_id, distinguishable))
    if feasible:
        best_value, best_first, best_dist = max(feasible, key=lambda row: (row[0], row[1]))
    else:
        best_value, best_first, best_dist = None, None, None
    return {
        "pair_id": pair.pair_id,
        "skills": [entry.as_dict() for entry in skills],
        "diverged_sequences": {world: sorted(values) for world, values in diverged.items()},
        "valid_sequence_count": len(valid_keys),
        "post_skill1": post,
        "per_first_skill": per_first,
        "common_plan_value": common_plan,
        "legal_observation_conditioned_plan_value": best_value,
        "legal_plan_first_skill": best_first,
        "legal_plan_post_skill1_distinguishable": best_dist,
        "mean_state_planner_best_sequence": mean_state_best,
        "mean_state_planner_value": mean_state_value,
        "all_knowing_two_skill_value": all_knowing,
        "information_gap_two_skill": (all_knowing - mean_state_value)
        if (all_knowing is not None and mean_state_value is not None) else None,
        "settle_appended": settle,
        "note": "All three references use the same finite candidate set (non-diverged "
                "two-skill sequences). common_plan_value = max over sequences of the 50/50 "
                "average (the same rule in both worlds); all_knowing = 0.5 max_A + 0.5 max_B; "
                "legal_observation_conditioned = max over a1 of (branch where the post-a1 "
                "observation distinguishes the worlds, else the common rule). The legal value "
                "is therefore guaranteed >= the common value.",
    }


def main() -> int:
    out_root = ROOT / "outputs" / "option_prediction_mechanism_v1" / "phase_c"
    records = json.loads((out_root / "phase_c_records.json").read_text())
    positives = [r for r in records["records"] if r.get("analysis", {}).get("positive")]
    pairs = {pair.pair_id: pair for pair in R.build_candidates()}
    only = sys.argv[1:] or [r["pair_id"] for r in positives]
    out = {"pair_ids": list(only), "references": {}}
    for pair_id in only:
        pair = pairs[pair_id]
        reference = belief_reference(pair)
        out["references"][pair_id] = reference
        print(f"{pair_id}: common_plan={reference['common_plan_value']:.4f} "
              f"legal_observed={reference['legal_observation_conditioned_plan_value']:.4f}",
              flush=True)
        for first in pair.skills:
            entry = reference["per_first_skill"][first.skill_id]
            if entry.get("common_value") is None:
                print(f"   a1={first.skill_id:16s} post_diff="
                      f"{entry['post_skill1']['max_abs_difference']:.3e} "
                      f"mismatches={len(entry['post_skill1']['mismatch_counts'])} "
                      f"dist={entry['post_skill1']['distinguishable']} no valid a2",
                      flush=True)
            else:
                print(f"   a1={first.skill_id:16s} post_diff="
                      f"{entry['post_skill1']['max_abs_difference']:.3e} "
                      f"mismatches={len(entry['post_skill1']['mismatch_counts'])} "
                      f"dist={entry['post_skill1']['distinguishable']} "
                      f"common={entry['common_value']:+.4f} branch={entry['branch_value']:+.4f} "
                      f"legal={entry['legal_value_for_this_a1']:+.4f}", flush=True)
    (out_root / "phase_c_belief_reference.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str))
    print("wrote", out_root / "phase_c_belief_reference.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
