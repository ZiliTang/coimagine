"""Round 010 — FOLLOW contract acceptance (gap semantics fixed; no zero-filled errors).

Phases are separated: prepare (find/align), FOLLOW, HOLD. A missing error is *excluded*, never
filled with zero. The follower's control gap is the edge clearance; the offline true edge gap is
reported separately. Force is decomposed by the surface normal, and the actual contact time is
reported because a smaller drive-force norm does not prove "no added resistance".
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
PUSH = 6; WAIT6 = 21; FOLLOW6 = 25
FOLLOW_DURATION = spec.ROBOT_ACTIONS[FOLLOW6].duration
TOLERANCE = spec.FOLLOW_POSE_TOLERANCE
CASES = {"double_push": (PUSH, PUSH), "push_brake": (PUSH, WAIT6), "push_follow": (PUSH, FOLLOW6)}


def progression(rows):
    out = []
    for row in rows:
        robot = row["r1"]
        out.append({"stage": robot["stage"], "done": bool(robot["done"]),
                    "gap": robot.get("gap"), "centre_gap": robot.get("centre_gap"),
                    "true_gap": row["true_edge_gap"]["1"],
                    "offset": robot.get("follow_offset"),
                    "reference": robot.get("follow_reference_offset"),
                    "pose_error": robot.get("follow_pose_error"),
                    "force": robot.get("force"), "normal": robot.get("normal_world"),
                    "tangent": robot.get("tangent_world"),
                    "contact": bool(robot.get("contacting_object")),
                    "object": row["true"]["object"]})
    return out


def summarize(rows):
    prepare = [r for r in rows if r["stage"] in ("find", "align")]
    follow = [r for r in rows if r["stage"] == "follow" and not r["done"]]
    hold = [r for r in rows if r["done"]]
    errors = [r["pose_error"] for r in follow if r["pose_error"] is not None]
    gap_errors = [r["gap"] - spec.FOLLOW_GAP for r in follow if r["gap"] is not None]
    tangential = [r["offset"] - r["reference"] for r in follow
                  if r["offset"] is not None and r["reference"] is not None]
    within = sum(1 for value in errors if value <= TOLERANCE)
    longest = current = 0
    for value in errors:
        current = current + 1 if value > TOLERANCE else 0
        longest = max(longest, current)
    normal_force, tangent_force = [], []
    for r in follow:
        if r["force"] is None or r["normal"] is None or r["tangent"] is None:
            continue
        n, t = r["normal"], r["tangent"]
        normal_force.append(r["force"][0] * n[0] + r["force"][1] * n[1])
        tangent_force.append(r["force"][0] * t[0] + r["force"][1] * t[1])
    effective = len(follow) * DT
    return {
        "prepare_seconds": len(prepare) * DT, "follow_seconds": effective,
        "hold_seconds": len(hold) * DT,
        "pose_error_n": len(errors),
        "pose_error_mean": (sum(errors) / len(errors) if errors else None),
        "pose_error_max": (max(errors) if errors else None),
        "gap_error_mean": (sum(gap_errors) / len(gap_errors) if gap_errors else None),
        "tangential_error_mean": (sum(tangential) / len(tangential) if tangential else None),
        "within_tolerance_fraction": (within / len(errors) if errors else None),
        "longest_out_of_tolerance_s": longest * DT,
        "unexpected_contact_seconds": sum(1 for r in follow if r["contact"]) * DT,
        "normal_force_mean": (sum(normal_force) / len(normal_force) if normal_force else None),
        "tangential_force_mean": (sum(tangent_force) / len(tangent_force) if tangent_force else None),
        "exit": rows[-1]["stage"],
    }


def verdict(summary):
    if summary["follow_seconds"] < 0.8 * FOLLOW_DURATION:
        return "undetermined_insufficient_follow_time"
    if (summary["within_tolerance_fraction"] is not None
            and summary["within_tolerance_fraction"] >= 0.8):
        return "fulfilled"
    return "not_fulfilled"


def run(name, pair):
    world = E.RandomStartEnv(P.build_panel()[0]["teammate"])
    boundary = world.joint_step(*pair)
    rows = progression(world.rows[-boundary["periods"]:])
    summary = summarize(rows)
    summary.update({"name": name, "physical_seconds": boundary["periods"] * DT,
                    "exits": {str(k): v["exit_reason"] for k, v in boundary["skill_ends"].items()},
                    "verdict": verdict(summary)})
    return summary


def main():
    out = ROOT / "outputs/option_testbench_v1/round_010"; out.mkdir(parents=True, exist_ok=True)
    report = {name: run(name, pair) for name, pair in CASES.items()}
    (out / "follow_validation.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["# Round 010 — FOLLOW contract acceptance (gap semantics fixed)", "",
             f"requested follow duration {FOLLOW_DURATION} s; tolerance {TOLERANCE} m; "
             f"gap target {spec.FOLLOW_GAP} m (edge clearance)", "",
             "| case | phys s | prepare | follow | hold | gap err | tang err | pose err mean/max | within | longest out | contact s | normal/tangent force | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|"]
    for name, r in report.items():
        lines.append(f"| {name} | {r['physical_seconds']:.2f} | {r['prepare_seconds']:.2f} | "
                     f"{r['follow_seconds']:.2f} | {r['hold_seconds']:.2f} | "
                     f"{'—' if r['gap_error_mean'] is None else f'{r[chr(103)+chr(97)+chr(112)+chr(95)+chr(101)+chr(114)+chr(114)+chr(111)+chr(114)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.3f}'} | "
                     f"{'—' if r['tangential_error_mean'] is None else f'{r[chr(116)+chr(97)+chr(110)+chr(103)+chr(101)+chr(110)+chr(116)+chr(105)+chr(97)+chr(108)+chr(95)+chr(101)+chr(114)+chr(114)+chr(111)+chr(114)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.3f}'} | "
                     f"{'—' if r['pose_error_mean'] is None else f'{r[chr(112)+chr(111)+chr(115)+chr(101)+chr(95)+chr(101)+chr(114)+chr(114)+chr(111)+chr(114)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.3f}/{r[chr(112)+chr(111)+chr(115)+chr(101)+chr(95)+chr(101)+chr(114)+chr(114)+chr(111)+chr(114)+chr(95)+chr(109)+chr(97)+chr(120)]:.3f}'} | "
                     f"{'—' if r['within_tolerance_fraction'] is None else f'{r[chr(119)+chr(105)+chr(116)+chr(104)+chr(105)+chr(110)+chr(95)+chr(116)+chr(111)+chr(108)+chr(101)+chr(114)+chr(97)+chr(110)+chr(99)+chr(101)+chr(95)+chr(102)+chr(114)+chr(97)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)]:.2f}'} | "
                     f"{r['longest_out_of_tolerance_s']:.2f} | {r['unexpected_contact_seconds']:.2f} | "
                     f"{'—' if r['normal_force_mean'] is None else f'{r[chr(110)+chr(111)+chr(114)+chr(109)+chr(97)+chr(108)+chr(95)+chr(102)+chr(111)+chr(114)+chr(99)+chr(101)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.2f}'}/"
                     f"{'—' if r['tangential_force_mean'] is None else f'{r[chr(116)+chr(97)+chr(110)+chr(103)+chr(101)+chr(110)+chr(116)+chr(105)+chr(97)+chr(108)+chr(95)+chr(102)+chr(111)+chr(114)+chr(99)+chr(101)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.2f}'} | "
                     f"**{r['verdict']}** |")
    (out / "FOLLOW.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
