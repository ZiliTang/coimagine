"""Cooperative push: does both-robots-push actually beat one-robot-push?

Only adjacent-side or same-side start geometries are included (opposite-face squeeze is not the
target). For each scene the same start runs PUSH/PUSH, PUSH/WAIT and WAIT/PUSH; the cooperation
gain is both_push_goal_progress - max(two single_push_goal_progress).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (HERE, ROOT / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import panel as P  # noqa: E402
from coimagine.random_start import env as E  # noqa: E402

PUSH, WAIT = 6, 21
SCENES = {"teammate": "同侧 same side", "corner_plus": "相邻侧 (corner+)",
          "corner_minus": "相邻侧 (corner-)", "edge_turn": "相邻侧 (edge)"}
CASES = {"both_push": (PUSH, PUSH), "push_only": (PUSH, WAIT), "wait_then_push": (WAIT, PUSH)}


def run(instance, pair):
    world = E.RandomStartEnv(instance)
    initial = world.object_state().as_tuple()
    boundary = world.joint_step(*pair)
    final = world.object_state().as_tuple()
    goal = world.instance.goal_centre
    to_goal = (goal[0] - initial[0], goal[1] - initial[1])
    norm = math.hypot(*to_goal)
    displacement = (final[0] - initial[0], final[1] - initial[1])
    return {"goal_progress": (displacement[0] * to_goal[0] + displacement[1] * to_goal[1]) / norm
            if norm else None,
            "displacement": list(displacement),
            "rotation_deg": math.degrees(math.atan2(math.sin(final[2] - initial[2]),
                                                    math.cos(final[2] - initial[2]))),
            "exits": {str(k): v["exit_reason"] for k, v in boundary["skill_ends"].items()},
            "physical_seconds": boundary["periods"] * 0.05}


def main():
    out = ROOT / "outputs/option_testbench_v1/round_019"; out.mkdir(parents=True, exist_ok=True)
    panel, _ = P.build_panel()
    report = {}
    for name, label in SCENES.items():
        entry = {"label": label, "cases": {}}
        for case, pair in CASES.items():
            entry["cases"][case] = run(panel[name], pair)
        best_single = max(entry["cases"]["push_only"]["goal_progress"] or 0,
                          entry["cases"]["wait_then_push"]["goal_progress"] or 0)
        both = entry["cases"]["both_push"]["goal_progress"] or 0
        entry["cooperation_gain"] = both - best_single
        entry["cooperative"] = bool(entry["cooperation_gain"] > 0.02)
        report[name] = entry
    (out / "cooperation.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["# Round 019 — 同侧/相邻侧双推协同评测", "",
             "| scene | type | both push | push only | wait+push | gain | cooperative |",
             "|---|---|---:|---:|---:|---:|---|"]
    for name, entry in report.items():
        c = entry["cases"]
        lines.append(f"| {name} | {entry['label']} | {c['both_push']['goal_progress']:+.3f} | "
                     f"{c['push_only']['goal_progress']:+.3f} | "
                     f"{c['wait_then_push']['goal_progress']:+.3f} | "
                     f"{entry['cooperation_gain']:+.3f} | {entry['cooperative']} |")
    (out / "COOPERATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
