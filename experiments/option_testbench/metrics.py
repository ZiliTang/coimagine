"""Independent recomputation of the three metric layers from the raw branch logs.

Round 003 (metric v2) fixes six calibration issues found in v1; the v1 outputs are preserved as
files and are never overwritten:

1. PUSH fulfilment is split into ``contact_established`` (>= 0.50 s) and ``full_push``
   (>= PUSH_ACTIVE_SECONDS = 2.00 s, the skill's own completion);
2. contact/reposition statistics are truncated at each robot's **own** skill-end period, so the
   HOLD phase after a robot finishes is not counted as its skill;
3. REPOSITION uses **positive accumulated progress along the requested direction** (the skill's
   own tangent), not ``abs(net progress)``;
4. the teammate gap subtracts only this robot's radius (the lidar range is to the teammate
   surface), consistent with ``Skill._safety`` and the runner;
5. the normal-jump count is recomputed from the ``normal_world`` sequence, because PUSH reports
   ``corner_event = 0``;
6. conditional fulfilment restricts the numerator to accepted requests; rejected and
   unknown-applicability requests are reported separately.
"""
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

DT = spec.CONTROL_DT
UNRESOLVED_BRANCH = ("wall_timeout", "batch_budget_exhausted", "snapshot_error", "failed")


def load_branch_logs(out_dir):
    logs = []
    for path in sorted((out_dir / "logs").glob("*.json")):
        if path.name == "index.json":
            continue
        logs.append(json.loads(path.read_text()))
    return logs


def _robot_key(mapping, robot_id):
    return str(robot_id) if str(robot_id) in mapping else robot_id


def _boundary_rows(branch, boundary, robot_id):
    """Rows of this skill, truncated at *this robot's* own skill end (fix 2)."""
    start = int(boundary["start_period"])
    ends = boundary.get("skill_ends", {})
    end = int(ends[_robot_key(ends, robot_id)]["period"]) if ends else \
        int(boundary.get("joint_end_period", start))
    return [row for row in branch.get("rows", [])
            if start < int(row["period"]) <= end]


def _robot_speed(row, robot_id):
    state = row["true"]["robots"][robot_id]
    return math.hypot(float(state[3]), float(state[4]))


def _angle_between(a, b):
    na, nb = math.hypot(*a), math.hypot(*b)
    if na < 1e-9 or nb < 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cosine))


def _normal_jumps(rows, key):
    """Fix 5: independent count of normal-direction jumps (PUSH reports corner_event = 0)."""
    jumps = 0
    previous = None
    for row in rows:
        normal = row[key].get("normal_world")
        if normal is None:
            previous = None
            continue
        if previous is not None:
            angle = _angle_between(previous, (float(normal[0]), float(normal[1])))
            if angle is not None and angle > spec.NORMAL_JUMP_LIMIT_DEG:
                jumps += 1
        previous = (float(normal[0]), float(normal[1]))
    return jumps


def recompute_option(branch, boundary, robot_id, action_index, contract):
    rows = _boundary_rows(branch, boundary, robot_id)
    action = spec.ROBOT_ACTIONS[action_index]
    option = "push" if action.kind == "push" else "reposition"
    terms = contract[option]
    key = f"r{robot_id}"
    ends = boundary.get("skill_ends", {})
    exit_reason = ends[_robot_key(ends, robot_id)]["exit_reason"] if ends else None
    metrics = {"robot": robot_id, "option": option, "action": int(action_index),
               "label": action.label, "periods": len(rows), "claim_exit": exit_reason}
    if not rows:
        metrics["status"] = "unresolved"
        metrics["unresolved_reason"] = "missing_log"
        return metrics
    first, last = rows[0], rows[-1]
    contact_seconds = sum(DT for row in rows if row[key]["contacting_object"])
    metrics["contact_seconds"] = contact_seconds
    metrics["contact_established_time"] = next(
        (float(row["time"]) for row in rows if row[key]["contacting_object"]), None)
    first_contact = metrics["contact_established_time"]
    metrics["contact_lost_seconds"] = (
        sum(DT for row in rows if first_contact is not None
            and float(row["time"]) > first_contact and not row[key]["contacting_object"]))
    metrics["normal_jump_count"] = _normal_jumps(rows, key)
    metrics["unexpected_contact_seconds"] = sum(
        DT for row in rows if row[key]["contacting_static"] or row[key]["contacting_teammate"])
    metrics["force_saturated_seconds"] = sum(DT for row in rows if row[key]["force_saturated"])
    gaps = [float(row[key]["gap"]) for row in rows if row[key].get("gap") is not None]
    metrics["gap_min"] = min(gaps) if gaps else None
    metrics["gap_median"] = sorted(gaps)[len(gaps) // 2] if gaps else None
    metrics["gap_max"] = max(gaps) if gaps else None
    teammate_distances = [float(row[key]["teammate_nearest_distance"]) for row in rows
                          if row[key].get("teammate_nearest_distance") is not None]
    # Fix 4: the range is to the teammate surface, so only this robot's radius is subtracted.
    metrics["teammate_min_gap"] = (min(teammate_distances) - spec.ROBOT_RADIUS
                                   if teammate_distances else None)
    static_distances = [float(row[key]["obstacle_nearest_distance"]) for row in rows
                        if row[key].get("obstacle_nearest_distance") is not None]
    metrics["static_min_gap"] = (min(static_distances) - spec.ROBOT_RADIUS
                                 if static_distances else None)
    started = first["true"]["robots"][robot_id]
    ended = last["true"]["robots"][robot_id]
    displacement = (float(ended[0]) - float(started[0]), float(ended[1]) - float(started[1]))
    metrics["robot_displacement"] = list(displacement)
    if option == "push":
        metrics["contact_established"] = contact_seconds >= terms["thresholds"][
            "contact_established_seconds"]
        metrics["full_push_fulfilled"] = contact_seconds >= terms["thresholds"]["full_push_seconds"]
        metrics["completion_met"] = metrics["full_push_fulfilled"]
        direction_errors = []
        for row in rows:
            if _robot_speed(row, robot_id) < terms["thresholds"]["direction_error_speed_min"]:
                continue
            reference = row[key].get("reference_velocity")
            state = row["true"]["robots"][robot_id]
            if reference is None:
                continue
            error = _angle_between((float(reference[0]), float(reference[1])),
                                   (float(state[3]), float(state[4])))
            if error is not None:
                direction_errors.append(error)
        metrics["reference_direction_error_deg_max"] = (max(direction_errors)
                                                        if direction_errors else None)
        metrics["reference_direction_error_samples"] = len(direction_errors)
    else:
        # Fix 3: positive accumulated progress along the skill's own tangent (the row's
        # tangent_world is the skill tangent for REPOSITION), skipping contact/corner periods
        # exactly like the controller. No abs().
        progress = 0.0
        samples = 0
        for previous, current in zip(rows, rows[1:]):
            tangent = previous[key].get("tangent_world")
            if (tangent is None or previous[key]["contacting_object"]
                    or previous[key]["corner_event"]):
                continue
            norm = math.hypot(float(tangent[0]), float(tangent[1]))
            if norm < 1e-9:
                continue
            before = previous["true"]["robots"][robot_id]
            after = current["true"]["robots"][robot_id]
            advance = ((float(after[0]) - float(before[0])) * float(tangent[0])
                       + (float(after[1]) - float(before[1])) * float(tangent[1])) / norm
            if advance > 0.0:
                progress += advance
                samples += 1
        metrics["requested_distance"] = float(action.distance)
        metrics["tangential_progress"] = progress
        metrics["tangential_samples"] = samples
        metrics["completion_met"] = progress >= (
            terms["thresholds"]["progress_fraction_min"] * float(action.distance))
    # layer 3: offline task effect only
    object_before = [float(value) for value in first["true"]["object"]]
    object_after = [float(value) for value in last["true"]["object"]]
    object_displacement = (object_after[0] - object_before[0], object_after[1] - object_before[1])
    metrics["object_displacement"] = list(object_displacement)
    metrics["object_rotation_deg"] = math.degrees(
        math.atan2(math.sin(object_after[2] - object_before[2]),
                   math.cos(object_after[2] - object_before[2])))
    goal_xy = first["goal"]
    to_goal = (float(goal_xy[0]) - object_before[0], float(goal_xy[1]) - object_before[1])
    norm = math.hypot(*to_goal)
    metrics["object_goal_progress"] = (object_displacement[0] * to_goal[0]
                                       + object_displacement[1] * to_goal[1]) / norm \
        if norm > 1e-9 else None
    metrics["final_goal_distance"] = float(last["true_goal_distance"])
    # contract decision
    if exit_reason in contract["unresolved"]:
        status = "unresolved"
    elif metrics["completion_met"]:
        status = "fulfilled"
    elif exit_reason in terms["claim_exits"]:
        status = "violated"          # claimed completion but the recomputed condition failed
    elif exit_reason in terms["violation_exits"]:
        status = "violated"
    elif option == "push" and metrics["contact_established"]:
        status = "contact_only"      # weaker promise met; the full 2 s push was not
    elif exit_reason in terms["benign_exits"]:
        status = "not_fulfilled_benign"
    else:
        status = "unresolved"
    metrics["status"] = status
    metrics["safety_or_rejection"] = exit_reason in ("safety_abort", "blocked_contact",
                                                     "not_observed", "blocked_no_progress")
    return metrics


def evaluate_branch(branch, contract):
    options = []
    for boundary in branch.get("boundaries", []):
        pair = boundary.get("action_index")
        if not pair:
            continue
        for robot_id, action_index in enumerate(pair):
            action = spec.ROBOT_ACTIONS[int(action_index)]
            if action.kind in ("push", "reposition"):
                options.append(recompute_option(branch, boundary, robot_id,
                                                int(action_index), contract))
    statuses = [option["status"] for option in options]
    if branch["status"] in UNRESOLVED_BRANCH:
        branch_status = "unresolved"
    elif any(status == "unresolved" for status in statuses):
        branch_status = "unresolved"
    elif any(status == "violated" for status in statuses):
        branch_status = "violated"
    elif options and all(status == "fulfilled" for status in statuses):
        branch_status = "fulfilled"
    elif statuses and all(status in ("fulfilled", "contact_only") for status in statuses):
        branch_status = "contact_only"
    elif any(status == "fulfilled" for status in statuses):
        branch_status = "partial"
    else:
        branch_status = "not_fulfilled_benign"
    return {"branch_id": branch["branch_id"], "snapshot": branch["snapshot"],
            "status": branch["status"], "branch_contract_status": branch_status,
            "physical_seconds": branch.get("physical_seconds"),
            "wall_seconds": branch.get("wall_seconds"),
            "options": options}


def summarize(out_dir, contract=None, suffix="_v2"):
    out_dir = Path(out_dir)
    contract = contract if contract is not None else json.loads(
        (out_dir / "contracts" / "contracts.json").read_text())
    requests = json.loads((out_dir / "requests.json").read_text())["requests"]
    logs = {log["branch_id"]: log for log in load_branch_logs(out_dir)}
    branch_rows = []
    accepted_requests = 0
    unknown_applicability = 0
    for request in requests:
        log = logs.get(request["branch_id"], {"branch_id": request["branch_id"],
                                              "snapshot": request["snapshot"],
                                              "status": "unresolved",
                                              "boundaries": [], "eligibility": []})
        eligibility = log.get("eligibility", [])
        first = eligibility[0]["rows"] if eligibility else []
        if not first:
            unknown_applicability += 1
        involved = [row for row in first
                    if spec.ROBOT_ACTIONS[row["action"]].kind in ("push", "reposition")]
        accepted = bool(involved) and all(row["accepted"] for row in involved)
        if accepted:
            accepted_requests += 1
        evaluation = evaluate_branch(log, contract)
        evaluation["request_accepted"] = accepted
        evaluation["applicability_known"] = bool(first)
        evaluation["request_rejections"] = [{"robot": row["robot"], "reasons": row["reasons"]}
                                            for row in involved if not row["accepted"]]
        evaluation["focus"] = request["focus"]
        evaluation["recovery"] = request["recovery"]
        branch_rows.append(evaluation)
    accepted_rows = [row for row in branch_rows if row["request_accepted"]]
    fulfilled_accepted = sum(1 for row in accepted_rows
                             if row["branch_contract_status"] == "fulfilled")
    contact_only_accepted = sum(1 for row in accepted_rows
                                if row["branch_contract_status"] == "contact_only")
    violated_accepted = sum(1 for row in accepted_rows
                            if row["branch_contract_status"] == "violated")
    unresolved_accepted = sum(1 for row in accepted_rows
                              if row["branch_contract_status"] == "unresolved")
    partial_accepted = sum(1 for row in accepted_rows
                           if row["branch_contract_status"] == "partial")
    benign_accepted = sum(1 for row in accepted_rows
                          if row["branch_contract_status"] == "not_fulfilled_benign")
    option_rows = [option for row in branch_rows for option in row["options"]]
    statuses = ("fulfilled", "contact_only", "violated", "unresolved", "not_fulfilled_benign")
    snapshots = sorted({request["snapshot"] for request in requests})
    covered_states = sorted({row["snapshot"] for row in accepted_rows})
    summary = {
        "metric_version": P.METRIC_VERSION, "contract_version": contract["version"],
        "requests": len(requests), "accepted": accepted_requests,
        "rejected": len(requests) - accepted_requests,
        "unknown_applicability": unknown_applicability,
        "request_coverage": accepted_requests / len(requests) if requests else None,
        "state_coverage": (len(covered_states) / len(snapshots) if snapshots else None),
        "covered_states": covered_states, "panel_states": snapshots,
        "fulfilled_accepted": fulfilled_accepted,
        "contact_only_accepted": contact_only_accepted,
        "violated_accepted": violated_accepted,
        "unresolved_accepted": unresolved_accepted,
        "partial_accepted": partial_accepted,
        "not_fulfilled_benign_accepted": benign_accepted,
        "fulfilled_all_requests": sum(1 for row in branch_rows
                                      if row["branch_contract_status"] == "fulfilled"),
        "conditional_fulfillment": (fulfilled_accepted / accepted_requests
                                    if accepted_requests else None),
        "conservative_interval": (
            [fulfilled_accepted / accepted_requests,
             (fulfilled_accepted + unresolved_accepted) / accepted_requests]
            if accepted_requests else None),
        "effective_fulfillment_coverage": (fulfilled_accepted / len(requests)
                                           if requests else None),
        "option_evaluations": len(option_rows),
        "option_status_counts": {status: sum(1 for row in option_rows if row["status"] == status)
                                 for status in statuses},
        "safety_or_rejection_exits": sum(1 for row in option_rows
                                         if row.get("safety_or_rejection")),
        "branches": branch_rows,
    }
    P.write_json(out_dir / f"metrics{suffix}.json", summary)
    _write_markdown(out_dir, summary, branch_rows, suffix)
    return summary


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(out_dir, summary, branch_rows, suffix):
    lines = [f"# Independent metrics ({summary['metric_version']})", "",
             "Three layers: controller claim, independently recomputed contract, offline task "
             "effect. PUSH is split into `contact_established` (>=0.50 s) and `full_push` "
             "(>=2.00 s). Safety/rejection exits are listed separately.", "",
             f"- requests: {summary['requests']}  accepted: {summary['accepted']}  "
             f"rejected: {summary['rejected']}  unknown_applicability: "
             f"{summary['unknown_applicability']}",
             f"- request_coverage: {_fmt(summary['request_coverage'])}  "
             f"state_coverage: {_fmt(summary['state_coverage'])}",
             f"- accepted: fulfilled {summary['fulfilled_accepted']} / contact_only "
             f"{summary['contact_only_accepted']} / violated {summary['violated_accepted']} / "
             f"unresolved {summary['unresolved_accepted']} / partial "
             f"{summary['partial_accepted']} / benign {summary['not_fulfilled_benign_accepted']}",
             f"- conditional_fulfillment (accepted only): "
             f"{_fmt(summary['conditional_fulfillment'])}  conservative_interval: "
             f"{summary['conservative_interval']}",
             f"- effective_fulfillment_coverage: "
             f"{_fmt(summary['effective_fulfillment_coverage'])}",
             f"- option status counts: {summary['option_status_counts']}", "",
             "| branch | snapshot | request | branch status | options (claim -> recomputed) | "
             "contact s | object d | goal progress |", "|---|---|---|---|---|---:|---|---:|"]
    for row in branch_rows:
        options = "; ".join(
            f"{option['label']}: {option['claim_exit']} -> {option['status']}"
            for option in row["options"]) or "none"
        contact = row["options"][0].get("contact_seconds") if row["options"] else None
        object_d = row["options"][0].get("object_displacement") if row["options"] else None
        progress = row["options"][0].get("object_goal_progress") if row["options"] else None
        lines.append(f"| {row['branch_id']} | {row['snapshot']} | "
                     f"{'accepted' if row['request_accepted'] else 'rejected'} | "
                     f"{row['branch_contract_status']} | {options} | {_fmt(contact)} | "
                     f"{_fmt(object_d)} | {_fmt(progress)} |")
    (out_dir / f"METRICS{suffix}.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/option_testbench_v1/round_000")
    parser.add_argument("--suffix", default="_v2",
                        help="output suffix; the v1 metrics.json/METRICS.md are never overwritten")
    parser.add_argument("--use-round-contract", action="store_true",
                        help="use the round's frozen contracts/contracts.json instead of v2")
    args = parser.parse_args()
    contract = (json.loads((args.out / "contracts" / "contracts.json").read_text())
                if args.use_round_contract else P.contracts())
    summary = summarize(args.out, contract, args.suffix)
    print(json.dumps({key: summary[key] for key in
                      ("metric_version", "requests", "accepted", "rejected",
                       "unknown_applicability", "fulfilled_accepted", "contact_only_accepted",
                       "violated_accepted", "unresolved_accepted",
                       "conditional_fulfillment", "conservative_interval",
                       "effective_fulfillment_coverage")}, indent=1))


if __name__ == "__main__":
    main()
