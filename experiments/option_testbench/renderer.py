"""Log-driven local HTML replay. No external service, no browser fetch (script src only)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import panel as P  # noqa: E402
from coimagine.random_start import spec  # noqa: E402

JS_HEAD = "window.__DATA__ = window.__DATA__ || {};\n"


def _round(value, digits=4):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _patch_points(row, robot_id, limit=16):
    points = row[f"r{robot_id}"].get("patch_points_world") or []
    if len(points) > limit:
        step = max(1, len(points) // limit)
        points = points[::step][:limit]
    return [[_round(point[0]), _round(point[1])] for point in points]


def frame(row):
    object_state = row["true"]["object"]
    robots = row["true"]["robots"]
    frames = {
        "t": _round(row["time"]),
        "p": int(row["period"]),
        "o": [_round(object_state[0]), _round(object_state[1]), _round(object_state[2])],
        "r": [[_round(robots[0][0]), _round(robots[0][1]), _round(robots[0][2])],
              [_round(robots[1][0]), _round(robots[1][1]), _round(robots[1][2])]],
        "f": [[_round(row["r0"]["force"][0]), _round(row["r0"]["force"][1])],
              [_round(row["r1"]["force"][0]), _round(row["r1"]["force"][1])]],
        "st": [row["r0"]["stage"], row["r1"]["stage"]],
        "ex": [row["r0"]["exit_reason"], row["r1"]["exit_reason"]],
        "ct": [bool(row["r0"]["contacting_object"]), bool(row["r1"]["contacting_object"])],
        "cs": [bool(row["r0"]["contacting_static"]), bool(row["r1"]["contacting_static"])],
        "cm": [bool(row["r0"]["contacting_teammate"]), bool(row["r1"]["contacting_teammate"])],
        "sf": [bool(row["r0"]["safety_trigger"]), bool(row["r1"]["safety_trigger"])],
        "n": [[_round(row["r0"]["normal_world"][0]) if row["r0"]["normal_world"] else None,
               _round(row["r0"]["normal_world"][1]) if row["r0"]["normal_world"] else None],
              [_round(row["r1"]["normal_world"][0]) if row["r1"]["normal_world"] else None,
               _round(row["r1"]["normal_world"][1]) if row["r1"]["normal_world"] else None]],
        "tg": [[_round(row["r0"]["tangent_world"][0]) if row["r0"]["tangent_world"] else None,
                _round(row["r0"]["tangent_world"][1]) if row["r0"]["tangent_world"] else None],
               [_round(row["r1"]["tangent_world"][0]) if row["r1"]["tangent_world"] else None,
                _round(row["r1"]["tangent_world"][1]) if row["r1"]["tangent_world"] else None]],
        "g": [_round(row["r0"]["gap"]), _round(row["r1"]["gap"])],
        "pr": [_round(row["r0"]["progress"]), _round(row["r1"]["progress"])],
        "rv": [[_round(row["r0"]["reference_velocity"][0]), _round(row["r0"]["reference_velocity"][1])],
               [_round(row["r1"]["reference_velocity"][0]), _round(row["r1"]["reference_velocity"][1])]],
        "sk": [row["r0"]["skill"], row["r1"]["skill"]],
        "pp": [_patch_points(row, 0), _patch_points(row, 1)],
        "fs": [bool(row["r0"].get("anchor_face_switch")),
               bool(row["r1"].get("anchor_face_switch"))],
        "scan": [[_round(value, 3) for value in row["r0"]["scan"]["distance"]],
                 [_round(value, 3) for value in row["r1"]["scan"]["distance"]]],
        "cat": [list(row["r0"]["scan"]["category"]), list(row["r1"]["scan"]["category"])],
        "val": [list(row["r0"]["scan"]["valid"]), list(row["r1"]["scan"]["valid"])],
        "tgd": _round(row["true_goal_distance"]),
        "hold": _round(row["hold_seconds"]),
    }
    return frames


def write_js(path, variable, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = JS_HEAD + f"window.__DATA__[{json.dumps(variable)}] = " \
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + ";\n"
    path.write_text(text)


def build_data(out_dir):
    panel = json.loads((out_dir / "panel.json").read_text())
    requests = json.loads((out_dir / "requests.json").read_text())
    metrics_path = out_dir / "metrics_v2.json"
    if not metrics_path.exists():
        metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    summary = {row["branch_id"]: row for row in metrics["branches"]}
    snapshots = {}
    for name, entry in panel["snapshots"].items():
        if entry["kind"] == "static":
            instance = entry["instance"]
            snapshot = {
                "name": name, "sha256": entry["sha256"], "kind": "static",
                "arena": instance["arena"], "obstacles": instance["obstacles"],
                "object": instance["object"], "robots": instance["robots"],
                "goal": instance["goal_centre"], "goal_radius": instance["goal_radius"]}
            snapshot.update(panel["fixtures"].get(name, {}).get("labels", {}))
        else:
            snapshot = {
                "name": name, "sha256": entry["sha256"], "kind": "dynamic",
                "arena": list(spec.ARENA), "obstacles": [],
                "object": {"x": 1.6, "y": 1.3118, "angle": 0.0},
                "robots": [{"x": 1.6, "y": 1.0069, "angle": -1.5708},
                           {"x": 1.6, "y": 1.655, "angle": 1.5708}],
                "goal": [1.6, 1.95], "goal_radius": spec.GOAL_RADIUS}
            snapshot.update(entry.get("labels", {}))
        snapshots[name] = snapshot
    branches = []
    frames_dir = out_dir / "data" / "frames"
    for request in requests["requests"]:
        log_path = out_dir / "logs" / f"{request['branch_id']}.json"
        if not log_path.exists():
            continue
        log = json.loads(log_path.read_text())
        rows = log.get("rows", [])
        frames = [frame(row) for row in rows]
        write_js(frames_dir / f"{request['branch_id']}.js", request["branch_id"],
                 {"frames": frames})
        entry = summary.get(request["branch_id"], {})
        branches.append({
            "branch_id": request["branch_id"], "snapshot": request["snapshot"],
            "focus": request["focus"], "recovery": request["recovery"],
            "focus_zh": request.get("focus_zh"), "focus_en": request.get("focus_en"),
            "focus_detail_zh": request.get("focus_detail_zh"),
            "focus_detail_en": request.get("focus_detail_en"),
            "action_labels": request.get("action_labels"),
            "controller_version": request.get("controller_version", "baseline"),
            "actions": request["actions"], "teammate_protocol": request["teammate_protocol"],
            "status": log.get("status"), "physical_seconds": log.get("physical_seconds"),
            "wall_seconds": log.get("wall_seconds"),
            "frames": len(frames),
            "contract_status": entry.get("branch_contract_status"),
            "request_accepted": entry.get("request_accepted"),
            "options": entry.get("options", []),
            "skill_boundaries": log.get("skill_boundaries", []),
            "eligibility": log.get("eligibility", []),
            "initial_state": log.get("initial_state"),
            "final_state": log.get("final_state"),
            "problem_zh": snapshots[request["snapshot"]].get("problem_zh", ""),
            "problem_en": snapshots[request["snapshot"]].get("problem_en", ""),
        })
    write_js(out_dir / "data" / "panel.js", "panel", snapshots)
    write_js(out_dir / "data" / "branches.js", "branches", branches)
    write_js(out_dir / "data" / "metrics.js", "metrics", {
        "requests": metrics["requests"], "accepted": metrics["accepted"],
        "request_coverage": metrics["request_coverage"],
        "state_coverage": metrics["state_coverage"],
        "metric_version": metrics.get("metric_version"),
        "fulfilled": metrics.get("fulfilled_accepted", metrics.get("fulfilled")),
        "contact_only": metrics.get("contact_only_accepted"),
        "violated": metrics.get("violated_accepted", metrics.get("violated")),
        "unresolved": metrics.get("unresolved_accepted", metrics.get("unresolved")),
        "partial": metrics.get("partial_accepted", metrics.get("partial")),
        "not_fulfilled_benign": metrics.get("not_fulfilled_benign_accepted",
                                           metrics.get("not_fulfilled_benign")),
        "conservative_interval": metrics["conservative_interval"],
        "effective_fulfillment_coverage": metrics["effective_fulfillment_coverage"],
        "constants": {"lidar_range": spec.LIDAR_RANGE, "control_dt": spec.CONTROL_DT,
                      "robot_radius": spec.ROBOT_RADIUS,
                      "joint_max_seconds": spec.JOINT_MAX_SECONDS}})
    frame_tags = "\n".join(
        f'<script src="data/frames/{entry["branch_id"]}.js"></script>' for entry in branches)
    page = HTML.replace("__PANEL_VERSION__", P.PANEL_VERSION).replace(
        "<!--FRAME_SCRIPTS-->", frame_tags)
    (out_dir / "index.html").write_text(page)
    return {"snapshots": len(snapshots), "branches": len(branches)}


HTML = (Path(__file__).with_name("replay_template.html")).read_text(encoding="utf-8")


def write_keyframes(out_dir, panel, requests):
    """Initial / key transition / exit frame for each snapshot's primary branch."""
    from PIL import Image
    from coimagine.random_start import render as R

    out_dir = Path(out_dir)
    directory = out_dir / "keyframes"
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    primary = {}
    for request in requests["requests"]:
        primary.setdefault(request["snapshot"], request)
    for snapshot, request in primary.items():
        log_path = out_dir / "logs" / f"{request['branch_id']}.json"
        if not log_path.exists():
            continue
        log = json.loads(log_path.read_text())
        rows = log.get("rows", [])
        if not rows:
            continue
        contact = next((i for i, row in enumerate(rows)
                        if row["r0"]["contacting_object"] or row["r1"]["contacting_object"]), None)
        exit_index = next((i for i, row in enumerate(rows)
                           if row["r0"]["exit_reason"] or row["r1"]["exit_reason"]), None)
        picks = [("initial", 0), ("key_transition", contact if contact is not None
                                  else (exit_index if exit_index is not None else len(rows) // 2)),
                 ("exit", len(rows) - 1)]
        info = panel["snapshots"][snapshot]
        geometry = info["instance"] if info["kind"] == "static" else None
        meta = {"name": f"{snapshot}__{request['branch_id']}", "map": "L0",
                "arena": list(geometry["arena"]) if geometry else list(spec.ARENA),
                "obstacles": geometry["obstacles"] if geometry else [],
                "title": f"{snapshot} / {request['branch_id']} / {request['focus']}"}
        written = []
        for label, position in picks:
            position = max(0, min(len(rows) - 1, position))
            image = R.render_frame(rows, position, meta, ("world", "observation", "decision"))
            path = directory / f"{snapshot}__{label}.png"
            Image.fromarray(image).save(path)
            written.append({"label": label, "period": rows[position]["period"],
                            "time": rows[position]["time"], "path": str(path.relative_to(out_dir))})
        index.append({"snapshot": snapshot, "branch_id": request["branch_id"],
                      "focus": request["focus"], "frames": written})
    P.write_json(directory / "INDEX.json", {"keyframes": index})
    return index


def render(out_dir):
    out_dir = Path(out_dir)
    data = build_data(out_dir)
    requests = json.loads((out_dir / "requests.json").read_text())
    panel = json.loads((out_dir / "panel.json").read_text())
    data["keyframes"] = len(write_keyframes(out_dir, panel, requests))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/option_testbench_v1/round_000")
    args = parser.parse_args()
    print(json.dumps(render(args.out), indent=1))


if __name__ == "__main__":
    main()
