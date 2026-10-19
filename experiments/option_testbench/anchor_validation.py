"""Round 008 validation: run the anchor tracker over the round-006 occlusion geometry.

It reports the tracker's status distribution, the candidate error against the true corners,
the false-confidence rate (a non-unobservable status whose candidates are all far from any true
corner), the unobservable-detection rate and the face-switch flag. Measurement only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import occlusion_measure as OM  # noqa: E402
from coimagine.random_start import anchor  # noqa: E402
from coimagine.random_start import env as E  # noqa: E402

TOLERANCE_M = 0.05


def score(estimate, corners):
    errors = [min(math.dist(tuple(candidate), corner) for corner in corners)
              for candidate in estimate.candidates]
    best = min(errors) if errors else None
    false_confidence = bool(estimate.status != anchor.STATUS_UNOBSERVABLE
                            and errors and best is not None and best > TOLERANCE_M)
    return {"status": estimate.status, "candidates": [list(c) for c in estimate.candidates],
            "observed_corners": [list(c) for c in estimate.observed_corners],
            "best_error_m": best, "errors_m": errors, "false_confidence": false_confidence,
            "confidence": estimate.confidence,
            "legal_candidates": estimate.legal_candidates,
            "shadow_boundaries": estimate.shadow_boundaries,
            "face_switch": estimate.face_switch, "reason": estimate.reason}


def run_case(name, team_xy, corners=OM.TRUE_CORNERS, short=False):
    tracker = anchor.AnchorTracker()
    instance = OM.short_side_case_instance(team_xy) if short else OM.instance(name, team_xy)
    world = E.RandomStartEnv(instance)
    observation = world.observe(0, 0, 0.0)
    estimate = tracker.update(observation.scan, 0.0, 0.05)
    result = score(estimate, corners)
    result.update({"name": name, "team": list(team_xy)})
    return result


def main():
    out = ROOT / "outputs/option_testbench_v1/round_008"
    out.mkdir(parents=True, exist_ok=True)
    cases = [run_case("reference_no_occluder", OM.REFERENCE_TEAMMATE)]
    for dy in OM.GRID_DY:
        for dx in OM.GRID_DX:
            team_y = OM.OBSERVER_Y + dy
            team = (OM.OBJECT_CENTRE[0] + dx, team_y)
            if math.dist((OM.OBJECT_CENTRE[0], OM.OBSERVER_Y), team) < OM.MIN_CENTRE_DISTANCE:
                continue
            if OM.FACE_Y_CLEAR - team_y - 0.18 < -1e-9:
                continue
            cases.append(run_case(f"grid_dx{dx:+.2f}_dy{dy:.2f}", team))
    cases.append(run_case("short_side_no_teammate", (3.00, 0.30),
                          corners=OM.SHORT_SIDE_CORNERS, short=True))
    cases.append(run_case("short_side_blocked", (0.92, 1.10),
                          corners=OM.SHORT_SIDE_CORNERS, short=True))
    statuses = {}
    for case in cases:
        statuses[case["status"]] = statuses.get(case["status"], 0) + 1
    observable_cases = [c for c in cases if c["status"] != anchor.STATUS_UNOBSERVABLE]
    errors = [c["best_error_m"] for c in observable_cases if c["best_error_m"] is not None]
    false_rate = (sum(1 for c in observable_cases if c["false_confidence"])
                  / len(observable_cases)) if observable_cases else None
    unobservable_expected = [c for c in cases if c["name"] == "short_side_blocked"]
    payload = {
        "cases": cases, "status_counts": statuses,
        "observable_cases": len(observable_cases),
        "max_best_error_m": max(errors) if errors else None,
        "median_best_error_m": (sorted(errors)[len(errors) // 2] if errors else None),
        "false_confidence_rate": false_rate,
        "unobservable_detected_on_short_side_blocked": bool(
            unobservable_expected
            and unobservable_expected[0]["status"] == anchor.STATUS_UNOBSERVABLE),
        "false_confidence_threshold_m": TOLERANCE_M,
    }
    (out / "anchor_validation.json").write_text(json.dumps(payload, indent=1) + "\n")
    lines = ["# Round 008 — anchor tracker validation over the round-006 geometry", "",
             f"- status counts: {statuses}",
             f"- observable/uncertain cases: {len(observable_cases)}",
             f"- best-candidate error: median {payload['median_best_error_m']}, "
             f"max {payload['max_best_error_m']} m",
             f"- false-confidence rate (non-unobservable with all candidates > "
             f"{TOLERANCE_M} m): {false_rate}",
             f"- short-side blocked correctly reported anchor_unobservable: "
             f"{payload['unobservable_detected_on_short_side_blocked']}", "",
             "| case | status | legal | shadow | best error | false confidence | reason |",
             "|---|---|---:|---:|---:|---|---|"]
    for case in cases:
        error = case["best_error_m"]
        lines.append(f"| {case['name']} | {case['status']} | {case['legal_candidates']} | "
                     f"{case['shadow_boundaries']} | "
                     f"{'—' if error is None else f'{error:.3f}'} | "
                     f"{case['false_confidence']} | {case['reason']} |")
    (out / "ANCHOR.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: payload[key] for key in
                      ("status_counts", "observable_cases", "median_best_error_m",
                       "max_best_error_m", "false_confidence_rate",
                       "unobservable_detected_on_short_side_blocked")}, indent=1))


if __name__ == "__main__":
    main()
