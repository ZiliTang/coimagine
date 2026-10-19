"""Bounded branch runner: one pre-registered request, one isolated physical world.

The runner owns an internal monotonic deadline. The CLI driver also wraps it in an external
process-group timeout. A branch never runs a 90 s task: the default skill window is 8 s and a
recovery sequence is at most two skills and 12 s of physical time.
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import panel as P  # noqa: E402
from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import layout, skills, spec  # noqa: E402

CLEANUP_MARGIN = 0.5


def load_request(panel_dir, index):
    metadata = json.loads((panel_dir / "panel.json").read_text())
    requests = json.loads((panel_dir / "requests.json").read_text())["requests"]
    return metadata, requests[index], requests


def build_world(metadata, snapshot_name, panel_dir, skill_factory=None, env_config=None):
    snapshot = metadata["snapshots"][snapshot_name]
    if snapshot["kind"] == "static":
        instance = P.instance_from_dict(snapshot["instance"])
        return E.RandomStartEnv(instance, env_config, skill_factory), instance, None
    payload = P.pickle_load(panel_dir / snapshot["file"])
    instance = P.instance_from_dict(payload["base_instance"])
    world = E.RandomStartEnv(instance, env_config, skill_factory)
    world.restore_branch_state(payload["branch_state"])
    return world, instance, payload


def _safety_reason(observation):
    return skills.make_skill(spec.ROBOT_ACTIONS[13])._safety(observation)


def eligibility(world, action_indices):
    """Accept/reject each requested robot option from legal observations only."""
    observations = world.boundary_observations()
    rows = []
    for robot_id, action_index in enumerate(action_indices):
        action = spec.ROBOT_ACTIONS[action_index]
        observation = observations[robot_id]
        reasons = []
        patch = getattr(observation, "patch", None)
        gap = None if patch is None else float(skills.patch_gap(patch))
        if action.kind in ("push", "reposition"):
            if patch is None:
                reasons.append("no_object_patch")
            if _safety_reason(observation) is not None:
                reasons.append("safety_trigger")
        rows.append({
            "robot": robot_id, "action": int(action_index), "label": action.label,
            "kind": action.kind, "accepted": not reasons, "reasons": reasons,
            "patch_valid": patch is not None, "patch_gap": gap,
            "contacting_object": bool(getattr(observation, "contacting_object", False)),
            "static_gap": None if getattr(observation, "static_nearest", None) is None
            else float(observation.static_nearest[0] - spec.ROBOT_RADIUS),
            "teammate_gap": None if getattr(observation, "teammate_nearest", None) is None
            else float(observation.teammate_nearest[0] - spec.ROBOT_RADIUS)})
    return rows


def _state_tuple(env):
    return {"object": list(env.object_state().as_tuple()),
            "robots": [list(state.as_tuple()) for state in env.robot_states()],
            "total_periods": int(env.total_periods), "hold_seconds": float(env.hold_seconds),
            "termination": env.termination, "success": bool(env.success)}


def run_branch(panel_dir, request_index, out_dir, branch_wall_seconds=10.0,
               recovery_horizon_seconds=12.0):
    started = time.monotonic()
    deadline = started + max(0.1, branch_wall_seconds - CLEANUP_MARGIN)
    metadata, request, _requests = load_request(panel_dir, request_index)
    out_dir = Path(out_dir)
    log_path = out_dir / "logs" / f"{request['branch_id']}.json"
    record = {
        "branch_id": request["branch_id"], "snapshot": request["snapshot"],
        "actions": request["actions"], "teammate_protocol": request["teammate_protocol"],
        "focus": request["focus"], "recovery": bool(request["recovery"]),
        "env_version": spec.ENV_VERSION, "policy": "baseline-option-v1",
        "contract_version": P.CONTRACT_VERSION, "metric_version": P.METRIC_VERSION,
        "branch_wall_seconds": branch_wall_seconds,
        "recovery_horizon_seconds": recovery_horizon_seconds,
    }
    controller_version = request.get("controller_version", "baseline")
    oriented_mode = request.get("oriented_mode", "off")
    env_config = E.EnvConfig(oriented_mode=oriented_mode)
    controller_hash = None
    skill_factory = None
    try:
        if controller_version == "segment":
            import candidate_push
            skill_factory = candidate_push.make_skill_segment
            controller_hash = candidate_push.source_sha256()
        elif controller_version != "baseline":
            raise ValueError(f"unknown controller version: {controller_version}")
        world, instance, payload = build_world(metadata, request["snapshot"], panel_dir,
                                               skill_factory, env_config)
    except Exception as error:  # a broken snapshot must still produce a result record
        record.update(status="snapshot_error", error=repr(error),
                      wall_seconds=time.monotonic() - started)
        P.write_json(log_path, record)
        return record
    record["instance"] = instance.name
    record["controller_version"] = controller_version
    record["oriented_mode"] = oriented_mode
    record["controller_source_sha256"] = controller_hash
    record["initial_state"] = _state_tuple(world)
    record["snapshot_sha256"] = metadata["snapshots"][request["snapshot"]]["sha256"]
    physical_start = world.total_periods * spec.CONTROL_DT
    record["physical_start_seconds"] = physical_start
    eligibility_rows = []
    skill_boundaries = []
    status = "completed"
    for step_index, pair in enumerate(request["actions"]):
        if time.monotonic() > deadline:
            status = "wall_timeout"
            break
        elapsed = world.total_periods * spec.CONTROL_DT - physical_start
        if elapsed >= recovery_horizon_seconds - 1e-9:
            status = "evaluation_horizon"
            break
        if step_index > 0 and elapsed + spec.SKILL_MAX_SECONDS > recovery_horizon_seconds:
            # A whole skill may not cross the cumulative horizon; do not start it.
            status = "evaluation_horizon"
            break
        if world.termination is not None:
            status = "environment_termination"
            break
        eligibility_rows.append({"step": step_index, "action": list(pair),
                                 "rows": eligibility(world, pair)})
        boundary = world.joint_step(*pair)
        skill_boundaries.append({
            "step": step_index, "action": list(pair), "periods": int(boundary["periods"]),
            "start_period": int(boundary["start_period"]),
            "end_period": int(boundary["joint_end_period"]),
            "distance_before": float(boundary["distance_before"]),
            "distance_after": float(boundary["distance_after"]),
            "exit_reasons": {str(robot): end["exit_reason"]
                             for robot, end in boundary["skill_ends"].items()},
            "success": bool(boundary["success"]), "truncated": bool(boundary["truncated"]),
            "anomaly": boundary["anomaly"], "termination": boundary["termination"]})
        if world.termination is not None:
            break
    if world.termination is not None and status == "completed":
        status = world.termination
    record.update(status=status, eligibility=eligibility_rows,
                  skill_boundaries=skill_boundaries,
                  rows=world.rows, boundaries=world.boundaries,
                  termination=world.termination, success=bool(world.success),
                  final_state=_state_tuple(world),
                  physical_seconds=world.total_periods * spec.CONTROL_DT - physical_start,
                  total_periods=int(world.total_periods),
                  wall_seconds=time.monotonic() - started)
    P.write_json(log_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    single = sub.add_parser("single")
    single.add_argument("--panel-dir", type=Path, required=True)
    single.add_argument("--request-index", type=int, required=True)
    single.add_argument("--out", type=Path, required=True)
    single.add_argument("--branch-wall-seconds", type=float, default=10.0)
    single.add_argument("--recovery-horizon-seconds", type=float, default=12.0)
    args = parser.parse_args()
    if args.command == "single":
        record = run_branch(args.panel_dir, args.request_index, args.out,
                            args.branch_wall_seconds, args.recovery_horizon_seconds)
        print(json.dumps({"branch_id": record["branch_id"], "status": record["status"],
                          "physical_seconds": record.get("physical_seconds"),
                          "wall_seconds": record.get("wall_seconds")}, default=str))


if __name__ == "__main__":
    main()
