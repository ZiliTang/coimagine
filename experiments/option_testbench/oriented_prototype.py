"""Oriented-model prototype (A): kinematic heading DOF + stable pushing side.

Bounded comparison on a few key branches, v3.1 (``oriented=False``) vs the prototype
(``oriented=True``). It answers whether requiring the robot to turn towards the push direction
changes the option-level conclusions from rounds 000-002. This is a prototype: the heading is
kinematic and the misaligned push is scaled by cos(error); it is not a torque-coupled contact
model, and it does not change the object physics, reward, success rule or safety gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import panel as P  # noqa: E402
from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import spec  # noqa: E402

DT = spec.CONTROL_DT
BRANCHES = {
    "face__push_push": ("face", [[4, 4]]),
    "corner_minus__push_push": ("corner_minus", [[4, 4]]),
    "corner_plus__push_push": ("corner_plus", [[4, 4]]),
    "teammate__push_push": ("teammate", [[4, 4]]),
    "edge_turn__transfer_cw6_then_push": ("edge_turn", [[13, 11]] * 6 + [[4, 4]]),
}


def run_branch(instance, actions, oriented):
    config = E.EnvConfig(oriented=oriented)
    world = E.RandomStartEnv(instance, config)
    initial = world.object_state().as_tuple()
    started = time.monotonic()
    skill_exits = []
    for pair in actions:
        if world.termination is not None:
            break
        boundary = world.joint_step(*pair)
        skill_exits.append({str(robot): end["exit_reason"]
                            for robot, end in boundary["skill_ends"].items()})
    final = world.object_state().as_tuple()
    goal = instance.goal_centre
    to_goal = (goal[0] - initial[0], goal[1] - initial[1])
    norm = math.hypot(*to_goal)
    displacement = (final[0] - initial[0], final[1] - initial[1])
    rows = world.rows
    contact = [sum(DT for row in rows if row[f"r{robot}"]["contacting_object"])
               for robot in (0, 1)]
    align_errors = [abs(float(row[f"r{robot}"]["align_error_deg"]))
                    for row in rows for robot in (0, 1)
                    if row[f"r{robot}"].get("align_error_deg") is not None]
    force_scales = [float(row[f"r{robot}"]["force_scale"])
                    for row in rows for robot in (0, 1)
                    if row[f"r{robot}"].get("force_scale") is not None]
    return {
        "oriented": oriented,
        "skill_exits": skill_exits,
        "physical_seconds": float(world.total_periods * DT),
        "wall_seconds": time.monotonic() - started,
        "object_displacement": list(displacement),
        "object_rotation_deg": math.degrees(math.atan2(
            math.sin(final[2] - initial[2]), math.cos(final[2] - initial[2]))),
        "goal_progress": ((displacement[0] * to_goal[0] + displacement[1] * to_goal[1]) / norm
                          if norm > 1e-9 else None),
        "final_headings_deg": [math.degrees(value) for value in world.heading],
        "mean_abs_align_error_deg": (sum(align_errors) / len(align_errors)
                                     if align_errors else None),
        "mean_force_scale": (sum(force_scales) / len(force_scales)
                             if force_scales else None),
        "contact_seconds": contact,
        "termination": world.termination,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/option_testbench_v1/oriented_prototype")
    args = parser.parse_args()
    panel, _report = P.build_panel()
    report = {"env_version": spec.ENV_VERSION, "prototype": "oriented-kinematic-v1",
              "note": "kinematic heading + cos(error) force gating; object physics, reward, "
                      "success and safety unchanged; v3.1 is the frozen reference",
              "branches": []}
    for branch_id, (snapshot, actions) in BRANCHES.items():
        instance = panel[snapshot]
        entry = {"branch_id": branch_id, "snapshot": snapshot,
                 "actions": actions, "skills": len(actions)}
        entry["v31"] = run_branch(instance, actions, oriented=False)
        entry["oriented"] = run_branch(instance, actions, oriented=True)
        report["branches"].append(entry)
        print(branch_id, "v31 goal", round(entry["v31"]["goal_progress"], 3),
              "| oriented goal", round(entry["oriented"]["goal_progress"], 3),
              "| oriented mean align err",
              None if entry["oriented"]["mean_abs_align_error_deg"] is None
              else round(entry["oriented"]["mean_abs_align_error_deg"], 1),
              flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    P.write_json(args.out / "prototype.json", report)
    lines = ["# Oriented-model prototype (A) — v3.1 vs kinematic heading", "",
             "Prototype: the robot must turn its heading towards the push direction; a misaligned",
             "push is scaled by `cos(error)`. Object physics, reward, success rule and safety gate",
             "are unchanged. This is a kinematic prototype, not a torque-coupled contact model.", "",
             "| branch | model | skills | physical s | object d | rotation | goal progress | "
             "contact s (r0/r1) | mean align err | mean force scale |",
             "|---|---|---:|---:|---|---:|---:|---|---:|---:|"]
    for entry in report["branches"]:
        for key in ("v31", "oriented"):
            row = entry[key]
            align = row["mean_abs_align_error_deg"]
            scale = row["mean_force_scale"]
            lines.append(
                f"| {entry['branch_id']} | {key} | {entry['skills']} | "
                f"{row['physical_seconds']:.2f} | "
                f"({row['object_displacement'][0]:+.3f},{row['object_displacement'][1]:+.3f}) | "
                f"{row['object_rotation_deg']:+.1f} | {row['goal_progress']:+.3f} | "
                f"{row['contact_seconds'][0]:.2f}/{row['contact_seconds'][1]:.2f} | "
                f"{'—' if align is None else f'{align:.1f}'} | "
                f"{'—' if scale is None else f'{scale:.2f}'} |")
    (args.out / "PROTOTYPE.md").write_text("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
