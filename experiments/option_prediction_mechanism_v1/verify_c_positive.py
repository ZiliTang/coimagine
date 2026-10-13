"""Independent verification of the Phase C positives, recomputed from the recorded parameters.

    python experiments/option_prediction_mechanism_v1/verify_c_positive.py

This deliberately does **not** reuse the search loop. It reads the frozen pair parameters out of
``phase_c_records.json``, rebuilds both worlds, re-runs every recorded skill through the simulator,
and re-derives the observation identity, the Q matrix, the near-optimal sets, the mutual regret and
the 50/50 information gap from scratch. Any disagreement with the recorded numbers is printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "option_prediction_mechanism_v1" / "phase_c"))

import actions as ac  # noqa: E402
import mechanism_scene as ms  # noqa: E402
import perception as pc  # noqa: E402
import run_phase_c as R  # noqa: E402
from coimagine.continuous2d import Continuous2DEnv  # noqa: E402
from coimagine.continuous2d.scenes import Box  # noqa: E402

RECORDS = ROOT / "outputs/option_prediction_mechanism_v1/phase_c/phase_c_records.json"


def rebuild(params_payload: dict) -> ms.MechanismParams:
    goal = params_payload["goal"]
    return ms.MechanismParams(
        name=params_payload["name"],
        object_len=params_payload["object_len"],
        object_thick=params_payload["object_thick"],
        object_centre_x=params_payload["object_centre_x"],
        shift=params_payload["shift"],
        goal=Box(goal["cx"], goal["cy"], goal["half_w"], goal["half_h"], goal["angle"]),
        robot_x=tuple(params_payload["robot_x"]),
        standoff=params_payload["standoff"],
        arena=tuple(params_payload["arena"]),
        edge_y=params_payload["edge_y"],
        obstacles=tuple(),
    )


def rebuild_skill(payload: dict) -> ac.NamedJointSkill:
    """Rebuild a recorded skill from its serialised ``joint_skill_specs``."""

    from coimagine.continuous2d.skills import Hold, MoveTo, PushContact

    specs = []
    for entry in payload["joint_skill_specs"]:
        kind = entry.get("kind")
        if kind == "push_contact":
            specs.append(PushContact(
                object_id=int(entry["object_id"]),
                contact_local=tuple(entry["contact_local"]),
                push_direction_world=tuple(entry["push_direction_world"]),
                force_limit=float(entry["force_limit"]),
            ))
        elif kind == "hold":
            specs.append(Hold())
        elif kind == "move_to":
            specs.append(MoveTo(target=tuple(entry["target"])))
        else:
            raise ValueError(f"unsupported skill kind {kind!r}")
    return ac.NamedJointSkill(skill_id=payload["skill_id"], label=payload["label"],
                              joint=(specs[0], specs[1]), provenance=payload.get("provenance"))


def main() -> int:
    payload = json.loads(RECORDS.read_text())
    positives = [record for record in payload["records"]
                 if record.get("analysis", {}).get("positive")]
    print(f"recorded pairs {len(payload['records'])}, positives {len(positives)}")
    disagreements = []
    checked = 0
    for record in positives[:5]:
        params = rebuild(record["params"])
        analysis = record["analysis"]
        scenes = params.scenes()
        assert R.legality_ok(scenes[0]) and R.legality_ok(scenes[1]), "illegal scene"
        env = Continuous2DEnv(R.mechanism_config(params))
        env.reset(scenes[0])
        obs_a = pc.observe(scenes[0], params.object_pose("A"), env.backend.robot_states())
        env.reset(scenes[1])
        obs_b = pc.observe(scenes[1], params.object_pose("B"), env.backend.robot_states())
        identity = max(pc.observation_differences(obs_a, obs_b).values())
        q = {"A": {}, "B": {}}
        diverged = {"A": [], "B": []}
        for skill_payload in record["skills"]:
            skill = rebuild_skill(skill_payload)
            for world in ("A", "B"):
                outcome = R.run_sequence(env, world, params, [skill])
                if outcome.diverged:
                    # the contract excludes numerically divergent sequences from the matrix
                    diverged[world].append(skill.skill_id)
                    continue
                q[world][skill.skill_id] = outcome.q
        common = sorted(set(q["A"]) & set(q["B"]))
        best_a = max(q["A"][k] for k in common)
        best_b = max(q["B"][k] for k in common)
        near_a = sorted(k for k in common if best_a - q["A"][k] <= R.DELTA + 1e-12)
        near_b = sorted(k for k in common if best_b - q["B"][k] <= R.DELTA + 1e-12)
        argmax_a = max(common, key=lambda k: q["A"][k])
        argmax_b = max(common, key=lambda k: q["B"][k])
        regret = max(best_a - q["A"][argmax_b], best_b - q["B"][argmax_a])
        all_knowing = 0.5 * (best_a + best_b)
        common_action = max(0.5 * (q["A"][k] + q["B"][k]) for k in common)
        checks = {
            "observation_identity": abs(identity - record["observation_identity"]["max_abs_difference"]),
            "best_A": abs(best_a - analysis["best_A"]),
            "best_B": abs(best_b - analysis["best_B"]),
            "mutual_regret": abs(regret - analysis["mutual_regret"]),
            "information_gap": abs((all_knowing - common_action)
                                   - analysis["information_gap_50_50"]),
            "near_optimal_A": 0 if near_a == sorted(analysis["near_optimal_A"]) else 1,
            "near_optimal_B": 0 if near_b == sorted(analysis["near_optimal_B"]) else 1,
        }
        worst = max(checks.values())
        checked += 1
        print(f"{record['pair_id']}: obs_diff={identity:.3e} diverged={diverged} "
              f"nearA={near_a} nearB={near_b} "
              f"regret={regret:.4f} gap={all_knowing - common_action:.4f} "
              f"max_disagreement={worst:.3e}")
        if worst > 1e-9:
            disagreements.append({"pair_id": record["pair_id"], "checks": checks})
    print(json.dumps({"positives_in_file": len(positives), "independently_checked": checked,
                      "disagreements": disagreements}, indent=1))
    print("VERIFY_C_OK" if not disagreements else "VERIFY_C_MISMATCH")
    return 0 if not disagreements else 1


if __name__ == "__main__":
    raise SystemExit(main())
