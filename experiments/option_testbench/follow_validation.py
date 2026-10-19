"""Round 009 — SUPPORT_FOLLOW v1 comparison: double push vs push+brake vs push+follow.

The question is whether the follower keeps up, does not block, and does not add main-push
resistance. Measurement only (no RL).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (HERE, ROOT / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import panel as P  # noqa: E402
from coimagine.random_start import env as E, spec  # noqa: E402

DT = spec.CONTROL_DT
PUSH = 17       # PUSH(a=+0deg,v=0.16)[0.10m]
WAIT6 = 45      # WAIT_BRAKE[6s]
FOLLOW6 = 49    # SUPPORT_FOLLOW[6s]
CASES = {"double_push": (PUSH, PUSH), "push_brake": (PUSH, WAIT6), "push_follow": (PUSH, FOLLOW6)}


def run(pair):
    world = E.RandomStartEnv(P.build_panel()[0]["teammate"])
    initial = world.object_state().as_tuple()
    boundary = world.joint_step(*pair)
    final = world.object_state().as_tuple()
    goal = world.instance.goal_centre
    to_goal = (goal[0] - initial[0], goal[1] - initial[1])
    norm = math.hypot(*to_goal)
    displacement = (final[0] - initial[0], final[1] - initial[1])
    rows = world.rows
    follower = [row["r1"] for row in rows]
    main = [row["r0"] for row in rows]
    return {
        "physical_seconds": boundary["periods"] * DT,
        "exits": {str(k): v["exit_reason"] for k, v in boundary["skill_ends"].items()},
        "object_displacement": list(displacement),
        "object_rotation_deg": math.degrees(math.atan2(math.sin(final[2] - initial[2]),
                                                       math.cos(final[2] - initial[2]))),
        "object_goal_progress": (displacement[0] * to_goal[0]
                                 + displacement[1] * to_goal[1]) / norm if norm else None,
        "main_force_mean": sum(math.hypot(*row["force"]) for row in main) / len(main),
        "follower_force_mean": sum(math.hypot(*row["force"]) for row in follower) / len(follower),
        "follower_force_peak": max(math.hypot(*row["force"]) for row in follower),
        "follower_contact_fraction": sum(1 for row in follower if row["contacting_object"]) / len(follower),
        "follower_pose_error_mean": (sum(row.get("follow_pose_error") or 0.0 for row in follower)
                                     / len(follower)),
        "follower_pose_error_max": max((row.get("follow_pose_error") or 0.0) for row in follower),
        "follower_gap_mean": (sum(row["gap"] for row in follower if row.get("gap") is not None)
                              / max(1, sum(1 for row in follower if row.get("gap") is not None))),
        "follower_move": math.dist(
            (rows[-1]["true"]["robots"][1][0], rows[-1]["true"]["robots"][1][1]),
            (rows[0]["true"]["robots"][1][0], rows[0]["true"]["robots"][1][1])),
        "teammate_gap_min": min((row["r1"]["teammate_nearest_distance"] or 99.0)
                                for row in rows),
    }


def main():
    out = ROOT / "outputs/option_testbench_v1/round_009"; out.mkdir(parents=True, exist_ok=True)
    report = {name: run(pair) for name, pair in CASES.items()}
    (out / "follow_validation.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["# Round 009 — SUPPORT_FOLLOW v1 comparison (teammate snapshot)", "",
             "| case | phys s | exits | object d | goal progress | main force | follower force mean/peak | follower contact | pose err mean/max | follower move |",
             "|---|---:|---|---:|---:|---:|---|---:|---|---:|"]
    for name, r in report.items():
        lines.append(f"| {name} | {r['physical_seconds']:.2f} | {r['exits']} | "
                     f"({r['object_displacement'][0]:+.3f},{r['object_displacement'][1]:+.3f}) | "
                     f"{r['object_goal_progress']:+.3f} | {r['main_force_mean']:.2f} | "
                     f"{r['follower_force_mean']:.2f}/{r['follower_force_peak']:.2f} | "
                     f"{r['follower_contact_fraction']:.2f} | "
                     f"{r['follower_pose_error_mean']:.3f}/{r['follower_pose_error_max']:.3f} | "
                     f"{r['follower_move']:.3f} |")
    (out / "FOLLOW.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
