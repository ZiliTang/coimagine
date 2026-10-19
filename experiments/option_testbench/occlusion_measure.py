"""Round 006 — same-side occlusion measurement (measurement only, no control change).

Two robots work on the *same* long side of the object; the robot closer to the face occludes
part of the further robot's view. This script measures, for the observer robot A:

- the visible object run(s) and their world endpoints;
- which true bottom-face corners are visible;
- the breakpoint classification (true-corner boundary ``object->none`` vs occlusion boundary
  ``object->teammate``), using only the scan's own category channel;
- the naive endpoint error, the legal-corner-candidate error, and the false-corner-claim count;
- whether the anchor is observable at all (``anchor_unobservable``);
- a history-based recovery that uses a *real* unoccluded observation prefix (never truth).

It does not modify the controller, the contract or the physics. Thresholds below are frozen
before the run; continuous errors are kept, and no single score is synthesised.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import panel as P  # noqa: E402
from coimagine.continuous2d.scenes import Pose  # noqa: E402
from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import layout, spec  # noqa: E402
from coimagine.random_start.lidar import (CATEGORY_NONE, CATEGORY_OBJECT, CATEGORY_TEAMMATE)  # noqa: E402

# -- frozen constants (thresholds fixed before the run) --------------------------------------
OBJECT_CENTRE = (1.60, 1.10)
OBJECT_ANGLE = 0.0
FACE_Y = OBJECT_CENTRE[1] - spec.ROBOT_RADIUS * 0.0 - 0.125   # bottom face y = 0.975
TRUE_CORNERS = ((OBJECT_CENTRE[0] - 0.50, FACE_Y), (OBJECT_CENTRE[0] + 0.50, FACE_Y))
# The observer stands further back (a 0.395 m standoff, inside the 0.50 m random-start range)
# so that a teammate can be *both* legal (>= 0.36 m centre distance) and actually between the
# observer and the face; a teammate glued to the observer cannot occlude the face at all.
OBSERVER_Y = 0.40            # robot A, its own gap to the face is 0.975-0.40-0.18 = 0.395 m
FACE_Y_CLEAR = 0.975
CORNER_TOLERANCE_M = 0.04    # a true corner counts as observed if a hit lands within this
MIN_VISIBLE_SPAN_M = 0.10    # below this the anchor is declared unobservable
GRID_DX = (-0.45, -0.30, -0.15, 0.00, 0.15, 0.30, 0.45)
GRID_TEAMMATE_Y = (0.72, 0.79)   # teammate standoffs: 0.075 m and 0.005 m clear of the face
GRID_DY = tuple(round(team_y - OBSERVER_Y, 3) for team_y in GRID_TEAMMATE_Y)
MIN_CENTRE_DISTANCE = 2.0 * spec.ROBOT_RADIUS     # 0.36 m: robots may not overlap
REFERENCE_TEAMMATE = (3.00, 0.30)                 # beyond the 1.20 m range: no occlusion


def instance(name: str, team_xy: Tuple[float, float]) -> layout.InstanceSpec:
    return layout.InstanceSpec(
        name=name, map="L0", split="test", index=0, seed=0, arena=(3.2, 2.2), obstacles=(),
        object_pose=Pose(OBJECT_CENTRE[0], OBJECT_CENTRE[1], OBJECT_ANGLE),
        robot_poses=(Pose(OBJECT_CENTRE[0], OBSERVER_Y, math.pi / 2.0),
                     Pose(team_xy[0], team_xy[1], math.pi / 2.0)),
        goal_centre=(OBJECT_CENTRE[0], 0.30), goal_radius=0.20,
        env_version=spec.ENV_VERSION)


def analyse(observation, corners=TRUE_CORNERS) -> Dict[str, object]:
    """Everything below uses only the observer's own scan and its own pose.

    The object often straddles ray 0, so runs are found *circularly*; a naive linear scan
    splits one physical face into two runs and halves the measured span.
    """
    scan = observation.scan
    origin = scan.origin
    count = len(scan.valid)
    flags = [bool(scan.valid[index]) and int(scan.category[index]) == CATEGORY_OBJECT
             for index in range(count)]
    if not any(flags):
        runs_index = []
    elif all(flags):
        runs_index = [list(range(count))]
    else:
        starts = [index for index in range(count) if flags[index] and not flags[index - 1]]
        runs_index = []
        for start_index in starts:
            run, index = [], start_index
            while flags[index]:
                run.append(index)
                index = (index + 1) % count
                if index == start_index:
                    break
            runs_index.append(run)

    def point(index):
        angle = scan.ray_angles[index]
        distance = float(scan.distances[index])
        return (origin[0] + distance * math.cos(angle), origin[1] + distance * math.sin(angle))

    run_reports = []
    for run in runs_index:
        first, last = run[0], run[-1]
        start_point, end_point = point(first), point(last)
        before = int(scan.category[(first - 1) % count])
        after = int(scan.category[(last + 1) % count])
        run_reports.append({
            "ray_first": first, "ray_last": last, "rays": len(run),
            "end_a_world": list(start_point), "end_b_world": list(end_point),
            "span_m": math.dist(start_point, end_point),
            "boundary_before": before, "boundary_after": after,
            "before_is_occlusion": before == CATEGORY_TEAMMATE,
            "after_is_occlusion": after == CATEGORY_TEAMMATE,
            "before_is_none": before == CATEGORY_NONE,
            "after_is_none": after == CATEGORY_NONE,
        })
    corners_visible = [any(math.dist(point(index), corner) <= CORNER_TOLERANCE_M
                           for index in range(count) if flags[index])
                       for corner in corners]
    naive_errors, false_claims, legal_errors = [], 0, []
    for report in run_reports:
        for end_key, occlusion_key in (("end_a_world", "before_is_occlusion"),
                                       ("end_b_world", "after_is_occlusion")):
            endpoint = tuple(report[end_key])
            nearest = min(math.dist(endpoint, corner) for corner in corners)
            naive_errors.append(nearest)
            if report[occlusion_key]:
                false_claims += 1          # a shadow boundary is not a true corner
            else:
                legal_errors.append(nearest)
    span = max((report["span_m"] for report in run_reports), default=0.0)
    return {
        "object_rays": int(sum(flags)), "runs": run_reports, "visible_span_m": span,
        "corners_visible": corners_visible,
        "corner_visible_count": int(sum(corners_visible)),
        "naive_endpoint_error_m": naive_errors,
        "naive_endpoint_error_max_m": max(naive_errors) if naive_errors else None,
        "false_corner_claims": false_claims,
        "legal_candidate_error_m": legal_errors,
        "legal_candidate_count": len(legal_errors),
        "legal_candidate_error_max_m": max(legal_errors) if legal_errors else None,
        # observable means: a usable face span *and* at least one endpoint that is a legal
        # corner candidate (boundary to nothing), not merely a corner that happens to be seen
        "anchor_observable": bool(span >= MIN_VISIBLE_SPAN_M and legal_errors),
        "anchor_unobservable": not bool(span >= MIN_VISIBLE_SPAN_M and legal_errors),
    }


def observe(team_xy: Tuple[float, float]) -> Dict[str, object]:
    world = E.RandomStartEnv(instance("occlusion", team_xy))
    return analyse(world.observe(0, 0, 0.0))


SHORT_SIDE_CORNERS = ((1.10, 0.975), (1.10, 1.225))     # the object's left short side


def short_side_case_instance(team_xy: Tuple[float, float]) -> layout.InstanceSpec:
    """Observer faces the 0.25 m short side; a teammate directly in front can hide both ends."""
    return layout.InstanceSpec(
        name="short_side", map="L0", split="test", index=0, seed=0, arena=(3.2, 2.2),
        obstacles=(), object_pose=Pose(OBJECT_CENTRE[0], OBJECT_CENTRE[1], OBJECT_ANGLE),
        robot_poses=(Pose(0.40, 1.10, 0.0), Pose(team_xy[0], team_xy[1], 0.0)),
        goal_centre=(1.60, 0.30), goal_radius=0.20, env_version=spec.ENV_VERSION)


def short_side_case(team_xy: Tuple[float, float]) -> Dict[str, object]:
    world = E.RandomStartEnv(short_side_case_instance(team_xy))
    return analyse(world.observe(0, 0, 0.0), corners=SHORT_SIDE_CORNERS)


def history_recovery(reference: Dict[str, object], occluded: Dict[str, object]) -> Dict[str, object]:
    """Extrapolate the unseen corner from one *observed* corner plus the observed face line.

    The face direction and length come from the reference observation only (a real prefix);
    the answer is allowed to be ambiguous, so both candidates are reported.
    """
    if not reference["runs"] or not occluded["runs"]:
        return {"available": False, "reason": "missing_observation"}
    ref_main = max(reference["runs"], key=lambda report: report["span_m"])
    occ_main = max(occluded["runs"], key=lambda report: report["span_m"])
    ref_left = np.asarray(ref_main["end_a_world"])
    ref_right = np.asarray(ref_main["end_b_world"])
    direction = ref_right - ref_left
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return {"available": False, "reason": "degenerate_reference"}
    direction = direction / length
    candidates = []
    for end_key, occlusion_key in (("end_a_world", "before_is_occlusion"),
                                   ("end_b_world", "after_is_occlusion")):
        if occ_main[occlusion_key]:
            continue                          # shadow boundary: not a legal anchor
        observed = np.asarray(occ_main[end_key])
        for sign in (+1.0, -1.0):
            candidates.append([float(value) for value in observed + sign * length * direction])
    if not candidates:
        return {"available": False, "reason": "no_legal_candidate"}
    errors = [min(math.dist(tuple(candidate), corner) for corner in TRUE_CORNERS)
              for candidate in candidates]
    return {"available": True, "observed_span_m": occ_main["span_m"],
            "face_length_from_reference_m": length, "candidates": candidates,
            "errors_m": errors, "best_error_m": min(errors),
            "ambiguous": len(candidates) > 1,
            "ambiguous_but_one_correct": bool(len(candidates) > 1 and min(errors) <= 0.05),
            "all_candidates_wrong": bool(candidates and min(errors) > 0.05)}


def _scan_view(row) -> object:
    """Rebuild a scan-shaped view from the logged row (origin/heading/rays are all logged)."""
    import types
    scan = row["r0"]["scan"]
    count = len(scan["valid"])
    heading = float(scan["heading"])
    angles = tuple(heading + 2.0 * math.pi * index / count for index in range(count))
    return types.SimpleNamespace(scan=types.SimpleNamespace(
        origin=tuple(scan["origin"]), heading=heading, ray_angles=angles,
        distances=tuple(scan["distance"]), valid=tuple(scan["valid"]),
        category=tuple(scan["category"])))


def dynamic_trace() -> Dict[str, object]:
    """A real run: the observer waits while the teammate sweeps along the same face.

    It starts directly in front of the observer (the face is split into slivers with no legal
    corner candidate) so that the re-acquisition time is actually exercised.
    """
    wait = next(a.index for a in spec.ROBOT_ACTIONS if a.type_index == 11 and a.scale == 2.0)
    runs = {}
    for tag, type_index in (("teammate_sweeps_ccw", 9), ("teammate_sweeps_cw", 10)):
        world = E.RandomStartEnv(instance(tag, (OBJECT_CENTRE[0], 0.79)))
        action = next(a.index for a in spec.ROBOT_ACTIONS
                      if a.type_index == type_index and a.scale == 6.0)
        boundary = world.joint_step(wait, action)
        trace = []
        for row in world.rows[-boundary["periods"]:]:
            report = analyse(_scan_view(row))
            trace.append({"time": float(row["time"]), "object_rays": report["object_rays"],
                          "visible_span_m": report["visible_span_m"],
                          "corners_visible": report["corner_visible_count"],
                          "legal_candidate_count": report["legal_candidate_count"],
                          "anchor_unobservable": report["anchor_unobservable"]})
        lost_at = next((entry["time"] for entry in trace if entry["anchor_unobservable"]), None)
        reacquired = None
        if lost_at is not None:
            reacquired = next((entry["time"] - lost_at for entry in trace
                               if entry["time"] > lost_at and not entry["anchor_unobservable"]),
                              None)
        runs[tag] = {"periods": len(trace), "trace": trace, "lost_at_s": lost_at,
                     "reacquisition_time_s": reacquired,
                     "unobservable_periods": sum(1 for entry in trace
                                                 if entry["anchor_unobservable"]),
                     "final_span_m": trace[-1]["visible_span_m"] if trace else None}
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/option_testbench_v1/round_006")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = observe(REFERENCE_TEAMMATE)
    grid = []
    for team_y in GRID_TEAMMATE_Y:
        dy = round(team_y - OBSERVER_Y, 3)
        for dx in GRID_DX:
            team = (OBJECT_CENTRE[0] + dx, team_y)
            overlap = math.dist((OBJECT_CENTRE[0], OBSERVER_Y), team) < MIN_CENTRE_DISTANCE
            gap = FACE_Y_CLEAR - team[1] - spec.ROBOT_RADIUS
            if overlap or gap < -1e-9:
                grid.append({"dx": dx, "dy": dy, "skipped": "illegal",
                             "reason": "overlap" if overlap else "penetrates_face"})
                continue
            report = observe(team)
            grid.append({"dx": dx, "dy": dy, "team": list(team), "edge_gap_m": gap,
                         **{key: report[key] for key in
                            ("object_rays", "visible_span_m", "corners_visible",
                             "corner_visible_count", "false_corner_claims",
                             "naive_endpoint_error_max_m", "legal_candidate_error_max_m",
                             "anchor_observable")}})
    representatives = {}
    for tag, team in (("reference_no_occluder", REFERENCE_TEAMMATE),
                      ("partial", (OBJECT_CENTRE[0] + 0.45, 0.79)),
                      ("severe", (OBJECT_CENTRE[0] + 0.30, 0.79))):
        report = observe(team)
        report["history"] = history_recovery(reference, report)
        representatives[tag] = report
    dynamic = dynamic_trace()
    short_side = {"no_teammate": short_side_case((3.00, 0.30)),
                  "teammate_in_front": short_side_case((0.92, 1.10))}
    payload = {
        "measurement": "same-side occlusion, observer robot A at (1.60, 0.70)",
        "true_corners": [list(corner) for corner in TRUE_CORNERS],
        "frozen_thresholds": {"corner_tolerance_m": CORNER_TOLERANCE_M,
                              "min_visible_span_m": MIN_VISIBLE_SPAN_M,
                              "min_centre_distance_m": MIN_CENTRE_DISTANCE},
        "reference_no_occluder": reference, "grid": grid,
        "representatives": representatives, "dynamic": dynamic,
        "short_side_corners": [list(corner) for corner in SHORT_SIDE_CORNERS],
        "short_side": short_side,
    }
    P.write_json(args.out / "occlusion.json", payload)
    lines = ["# Round 006 — same-side occlusion measurement (no control change)", "",
             f"Observer A at ({OBJECT_CENTRE[0]:.2f}, {OBSERVER_Y:.2f}), heading +90 deg; teammate B on the same side,",
             "bottom-face corners: (1.10, 0.975) and (2.10, 0.975). Thresholds are frozen",
             "before the run; continuous errors are reported, no single score is synthesised.", "",
             "## Reference (no occluder)",
             f"- object rays {reference['object_rays']}, visible span "
             f"{reference['visible_span_m']:.3f} m, corners visible {reference['corners_visible']}",
             f"- naive endpoint error max {reference['naive_endpoint_error_max_m']}, "
             f"false corner claims {reference['false_corner_claims']}", "",
             "## Grid (dx = teammate offset along the face, dy = teammate offset toward it)", "",
             "| dx | dy | edge gap | object rays | visible span | corners visible | false claims |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in grid:
        if entry.get("skipped"):
            lines.append(f"| {entry['dx']:+.2f} | {entry['dy']:.2f} | — | — | — | — | skipped "
                         f"({entry['reason']}) |")
        else:
            lines.append(f"| {entry['dx']:+.2f} | {entry['dy']:.2f} | {entry['edge_gap_m']:.3f} | "
                         f"{entry['object_rays']} | {entry['visible_span_m']:.3f} | "
                         f"{entry['corner_visible_count']} | {entry['false_corner_claims']} |")
    lines += ["", "## Representatives", ""]
    for tag, report in representatives.items():
        lines.append(f"### {tag}")
        lines.append(f"- visible span {report['visible_span_m']:.3f} m, corners visible "
                     f"{report['corners_visible']}, object rays {report['object_rays']}, "
                     f"false corner claims {report['false_corner_claims']}")
        lines.append(f"- anchor unobservable: {report['anchor_unobservable']}; legal candidate "
                     f"error max {report['legal_candidate_error_max_m']}")
        lines.append(f"- history recovery: {report['history']}")
        lines.append("")
    lines += ["## Short side (0.25 m) — can both corners be hidden?", ""]
    for tag, report in short_side.items():
        lines.append(f"- {tag}: object rays {report['object_rays']}, span "
                     f"{report['visible_span_m']:.3f} m, corners visible "
                     f"{report['corners_visible']}, legal candidates "
                     f"{report['legal_candidate_count']}, false claims "
                     f"{report['false_corner_claims']}, anchor_unobservable "
                     f"{report['anchor_unobservable']}")
    lines += ["", "## Dynamic (teammate sweeps along the same face, observer waits)", ""]
    for tag, run in dynamic.items():
        lines.append(f"- {tag}: periods {run['periods']}, lost at {run['lost_at_s']} s, "
                     f"re-acquisition {run['reacquisition_time_s']} s, unobservable periods "
                     f"{run['unobservable_periods']}, final span {run['final_span_m']}")
    lines += ["", "| time | span | corners | legal candidates | unobservable |",
              "|---:|---:|---:|---:|---|"]
    for tag, run in dynamic.items():
        for entry in run["trace"][::max(1, run["periods"] // 8)]:
            lines.append(f"| {tag} {entry['time']:.1f}s | {entry['visible_span_m']:.3f} | "
                         f"{entry['corners_visible']} | {entry['legal_candidate_count']} | "
                         f"{entry['anchor_unobservable']} |")
    (args.out / "OCCLUSION.md").write_text("\n".join(lines) + "\n")
    _figure(args.out / "occlusion_grid.png", grid)
    print(json.dumps({"grid_cells": len([g for g in grid if not g.get('skipped')]),
                      "representatives": list(representatives),
                      "short_side": {tag: {"corners_visible": report["corners_visible"],
                                           "anchor_unobservable": report["anchor_unobservable"],
                                           "false_corner_claims": report["false_corner_claims"]}
                                     for tag, report in short_side.items()},
                      "dynamic": {tag: {"lost_at_s": run["lost_at_s"],
                                        "reacquisition_time_s": run["reacquisition_time_s"],
                                        "unobservable_periods": run["unobservable_periods"]}
                                  for tag, run in dynamic.items()}}, indent=1))


def _figure(path: Path, grid: Sequence[Dict[str, object]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                    # pragma: no cover
        print(f"figure skipped: {exc}")
        return
    spans = np.full((len(GRID_DY), len(GRID_DX)), np.nan)
    corners = np.full((len(GRID_DY), len(GRID_DX)), np.nan)
    for entry in grid:
        if entry.get("skipped"):
            continue
        row = GRID_DY.index(entry["dy"])
        column = GRID_DX.index(entry["dx"])
        spans[row, column] = entry["visible_span_m"]
        corners[row, column] = entry["corner_visible_count"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for axis, data, title in ((axes[0], spans, "visible span (m)"),
                              (axes[1], corners, "true corners visible (0-2)")):
        image = axis.imshow(data, origin="lower", aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(GRID_DX)), [f"{dx:+.2f}" for dx in GRID_DX])
        axis.set_yticks(range(len(GRID_DY)), [f"{dy:.2f}" for dy in GRID_DY])
        axis.set_xlabel("dx from observer (m)"); axis.set_ylabel("dy toward the face (m)")
        axis.set_title(title); figure.colorbar(image, ax=axis)
    false_claims = np.full((len(GRID_DY), len(GRID_DX)), np.nan)
    for entry in grid:
        if entry.get("skipped"):
            continue
        false_claims[GRID_DY.index(entry["dy"]), GRID_DX.index(entry["dx"])] = \
            entry["false_corner_claims"]
    image = axes[2].imshow(false_claims, origin="lower", aspect="auto", cmap="magma")
    axes[2].set_xticks(range(len(GRID_DX)), [f"{dx:+.2f}" for dx in GRID_DX])
    axes[2].set_yticks(range(len(GRID_DY)), [f"{dy:.2f}" for dy in GRID_DY])
    axes[2].set_xlabel("dx from observer (m)"); axes[2].set_ylabel("dy toward the face (m)")
    axes[2].set_title("false corner claims"); figure.colorbar(image, ax=axes[2])
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


if __name__ == "__main__":
    main()
