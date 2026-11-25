"""Isolate three clearance terms at the frozen CCW change-face suffix on development rooms."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import math
from unittest.mock import patch

from coimagine.random_start import boundary_follow, skills
from tools.guarded_calibration_v1 import cap_guarded_calibration_forks as forks

ARMS = ("original", "static_soft_06", "peer_radius_40", "box_gap_05")
OUT = forks.OUT / "change_face_clearance_dev.json"


def compare(room):
    bundle = forks.archive.read(forks.OUT / "forks/dev" / (room + ".json"))
    if bundle["collector_sha256"] != forks.frozen_hash():
        raise ValueError("frozen source mismatch")
    anchors = {a["anchor_id"]: a for a in bundle["anchors"]}
    return [replay(room, anchors[r["anchor_id"]], r, arm) for r in bundle["rows"] for arm in ARMS]


def replay(room, anchor, ref, arm):
    stats, extrema = Counter(), {}
    end = {}
    original_step = skills.RepositionSkill.step
    avoid = skills.RepositionSkill._obstacle_avoidance
    follow = forks.followup
    def measure(self, obs, dt):
        if arm == "peer_radius_40":
            self.peer_router.radius = .40
        result = original_step(self, obs, dt)
        _, log = result
        peer = log.get("peer_detour") or {}
        stats["reposition_controller_calls"] += 1
        stats["peer_outer_active"] += bool(peer.get("active"))
        stats["peer_weight_over_half"] += bool(peer.get("active") and peer["peer_weight"] > .5)
        stats["swept_static_block"] += bool(log.get("static_route_guard"))
        stats["hard_safety_abort"] += bool(log.get("safety_trigger"))
        stats["static_touch"] += obs.contacting_static
        stats["peer_touch"] += obs.contacting_teammate
        if obs.static_nearest:
            gap = obs.static_nearest[0] - skills.ROBOT_RADIUS
            extrema["min_sensed_static_edge_gap"] = min(gap,extrema.get("min_sensed_static_edge_gap",gap))
            stats["within_static_soft_default"] += gap < skills.OBSTACLE_AVOID_MARGIN
        return result
    def soft_static(self, obs, raw, include_teammate=False):
        # The force law uses margin - (range - robot radius). Shifting only its static
        # range by the margin difference is exact; hard checks see the original range.
        if obs.static_nearest:
            d, point = obs.static_nearest
            obs = replace(obs, static_nearest=(d + skills.OBSTACLE_AVOID_MARGIN - .06, point))
        return avoid(self, obs, raw, include_teammate)
    def suffix(world, action, *, execute):
        robot, _ = forks.push_phases.pusher(action)
        with ExitStack() as scoped:
            scoped.enter_context(patch.object(skills.RepositionSkill, "step", measure))
            if arm == "static_soft_06":
                scoped.enter_context(patch.object(skills.RepositionSkill, "_obstacle_avoidance", soft_static))
            if arm == "box_gap_05":
                scoped.enter_context(patch.object(skills, "REPOSITION_GAP", .05))
                scoped.enter_context(patch.object(boundary_follow, "REPOSITION_GAP", .05))
            outcome = follow(world, action, execute=execute)
        robot_xy = forks.xy(forks.archive.S.observe(world)["robots"][robot])
        box = forks.archive.truth(world)["object"][:2]
        end.update(robot_xy=robot_xy, box_xy=box, center_distance_m=math.dist(robot_xy, box))
        return outcome
    with patch.object(forks, "followup", suffix):
        outcome = forks.evaluate(forks.development_rooms()[room], forks.record_for("dev",room), anchor,
                                 ref["progress_m"], execute_followup=True)
    if arm == "original" and outcome != ref["outcome"]:
        raise ValueError("instrumentation changed frozen suffix")
    return dict(room=room, anchor_id=ref["anchor_id"], progress_m=ref["progress_m"], arm=arm,
                controller=dict(stats), **extrema, end=end, outcome=outcome)


def main():
    with ProcessPoolExecutor(4) as jobs:
        rows = [r for group in jobs.map(compare,sorted(forks.development_rooms())) for r in group]
    original = {(r["room"],r["anchor_id"],r["progress_m"]):r for r in rows if r["arm"]=="original"}
    summary = {}
    for arm in ARMS:
        pool = [r for r in rows if r["arm"] == arm]
        changed = sum(r["end"] != original[r["room"],r["anchor_id"],r["progress_m"]]["end"] for r in pool)
        fields = {k for r in pool for k in r["controller"]}
        summary[arm] = dict(candidates=len(pool), actual_change_of_face=sum(r["outcome"]["followup_completed"] for r in pool),
                            attempted_contact=sum(r["outcome"]["approach_ticks"]>0 for r in pool),
                            different_endpoints=changed,
                            cases_with={k:sum(r["controller"].get(k,0)>0 for r in pool) for k in sorted(fields)})
    forks.write_json(OUT, dict(schema="development-clearance-suffix-audit-v1",collector_sha256=forks.frozen_hash(),
                    arms=list(ARMS), stages_measured="REPOSITION controller calls; command switches and PUSH excluded",
                    summary=summary, rows=rows))
    print(summary,flush=True)


if __name__ == "__main__":
    main()
