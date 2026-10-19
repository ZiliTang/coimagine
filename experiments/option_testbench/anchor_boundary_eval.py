"""Round 011 — anchor boundary counterexamples and false-confidence evaluation.

It evaluates the *claimed observed* corners (not only the best-of-K candidate) against the true
face corners, and reports the best-of-K value explicitly as candidate coverage only.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (HERE, ROOT / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from coimagine.continuous2d.scenes import Box, Pose  # noqa: E402
from coimagine.random_start import anchor, env as E, layout  # noqa: E402

TOL = 0.05
CORNERS = ((1.10, 0.975), (2.10, 0.975))


def build(name, object_pose, robots, obstacles=()):
    return layout.InstanceSpec(
        name=name, map="L0", split="test", index=0, seed=0, arena=(3.2, 2.2),
        obstacles=tuple(obstacles), object_pose=object_pose,
        robot_poses=tuple(robots), goal_centre=(1.60, 0.30), goal_radius=0.20)


def measure(instance, corners=CORNERS):
    world = E.RandomStartEnv(instance)
    estimate = anchor.AnchorTracker().update(world.observe(0, 0, 0.0).scan, 0.0, 0.05, tick=0)
    claimed = [min(math.dist(tuple(point), corner) for corner in corners)
               for point in estimate.observed_corners]
    candidates = [min(math.dist(tuple(point), corner) for corner in corners)
                  for point in estimate.candidates]
    return {"status": estimate.status, "reason": estimate.reason,
            "claimed_errors_m": claimed, "candidate_errors_m": candidates,
            "best_of_k_m": min(candidates) if candidates else None,
            # a false confidence is an *observable* claim whose claimed corners are all far;
            # uncertain/unobservable explicitly decline to claim, so they are not false claims
            "false_confidence": bool(estimate.status == anchor.STATUS_OBSERVABLE
                                     and claimed and min(claimed) > TOL),
            "legal": estimate.legal_candidates, "shadow": estimate.shadow_boundaries}


def main():
    out = ROOT / "outputs/option_testbench_v1/round_011"; out.mkdir(parents=True, exist_ok=True)
    far = Pose(3.0, 2.0, 0.0)
    cases = {
        "reference": measure(build("reference", Pose(1.60, 1.10, 0.0),
                                   [Pose(1.60, 0.40, math.pi / 2), far])),
        # the object beyond the 1.2 m range: the run ends at the range limit, not a true corner
        "range_truncation": measure(build("range_truncation", Pose(1.60, 1.80, 0.0),
                                          [Pose(1.60, 0.50, math.pi / 2), far])),
        # a static box hides one end of the face
        "static_occlusion": measure(build("static_occlusion", Pose(1.60, 1.10, 0.0),
                                          [Pose(1.60, 0.40, math.pi / 2), far],
                                          [Box(2.00, 0.62, 0.06, 0.06, 0.0)])),
        # the observer is beside a corner and sees two faces in one run
        "cross_two_faces": measure(build("cross_two_faces", Pose(1.60, 1.10, 0.0),
                                         [Pose(0.95, 0.80, 0.0), far])),
    }
    claimed = [value for case in cases.values() for value in case["claimed_errors_m"]]
    summary = {
        "tolerance_m": TOL, "cases": cases,
        "claimed_corner_count": len(claimed),
        "claimed_corner_error_max_m": max(claimed) if claimed else None,
        "false_confidence_cases": [name for name, case in cases.items() if case["false_confidence"]],
        "note": "best_of_k is candidate coverage only; claimed_errors evaluate the corners the "
                "tracker says it observed",
    }
    (out / "anchor_boundary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = ["# Round 011 — anchor boundary counterexamples / false confidence", "",
             "| case | status | reason | claimed errors | best-of-K | false confidence |",
             "|---|---|---|---:|---:|---|"]
    for name, case in cases.items():
        lines.append(f"| {name} | {case['status']} | {case['reason']} | "
                     f"{[round(v, 3) for v in case['claimed_errors_m']]} | "
                     f"{'—' if case['best_of_k_m'] is None else round(case['best_of_k_m'], 3)} | "
                     f"{case['false_confidence']} |")
    (out / "ANCHOR_BOUNDARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
