"""Recheck predicted whole-motion bounds against current public geometry."""
import math

from shapely.geometry import box
from shapely.ops import unary_union

from .frontend import OPTIONS


def forecast_regions(prepared, action, motion):
    if set(motion) != {"r0", "r1", "target_boundary"}:
        raise ValueError("complete whole-motion channels required")
    target_motion = motion["target_boundary"]
    if type(target_motion) not in (int, float) or not math.isfinite(target_motion) or target_motion < 0:
        raise ValueError("finite nonnegative target boundary displacement required")
    if prepared["pose"] is None:
        raise ValueError("target pose unknown")
    robots = []
    radius = prepared["contract"]["robot_radius"]
    for i, r in enumerate(prepared["observation"]["robots"]):
        b = motion[f"r{i}"]
        if (not isinstance(b, (list, tuple)) or len(b) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in b) or
                not b[0] <= 0 <= b[1] or not b[2] <= 0 <= b[3]):
            raise ValueError("robot centre bounds must contain their initial centre")
        own = r["own"]
        # Outward polygonal disk approximation covers the circular body.
        robots.append(box(own["x"] + b[0], own["y"] + b[2],
                          own["x"] + b[1], own["y"] + b[3]).buffer(
                              (radius + .003) / math.cos(math.pi / 128), quad_segs=32))
    target = prepared["pose"].outer.buffer(target_motion + .001)
    return dict(robots=robots, target=target, static=unary_union([*robots, target]))


class ProcessChecker:
    def __init__(self, public_frontend):
        self.frontend = public_frontend

    def __call__(self, prepared, action, branch):
        if "motion" not in branch:
            return prepared["reports"][action]
        try:
            regions = forecast_regions(prepared, action, branch["motion"])
        except ValueError as exc:
            return dict(allowed=False, reason="invalid_motion_forecast", error=str(exc))
        evidence = self.frontend.evidence.query(regions["static"])
        opt = OPTIONS[action]
        dynamic = not regions["robots"][0].intersects(regions["robots"][1])
        if opt["family"] == "move":
            dynamic = dynamic and not regions["robots"][opt["robots"][0]].intersects(regions["target"])
        if opt["family"] == "push":
            dynamic = dynamic and all(not regions["robots"][i].intersects(regions["target"])
                                      for i in range(2) if i not in opt["robots"])
        report = prepared["reports"][action]
        return dict(allowed=bool(report["legal"] and report["bound_valid"] and evidence.certified and dynamic),
                    evidence_status=evidence.status, missing_area=evidence.unknown_area_m2,
                    occupied_hits=evidence.occupied_hits, dynamic=dynamic,
                    quality="conditional_on_joint_motion_forecast_and_current_public_evidence")
