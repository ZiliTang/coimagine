"""Round-001 candidate controller: aim the PUSH contact point inside the visible segment.

The frozen baseline PUSH drives along the fitted normal only. Near an object corner the robot
centre can sit within one robot radius of the visible segment end, so the disc touches the
corner and slips off (`contact_lost`) while the fitted plane still looks closed. This candidate
adds a bounded tangential reference that re-centres the robot inside the visible segment before
and during the push. It does not change the observation, the safety gate, the physics, the
reward or the action menu; it is injected explicitly through ``RandomStartEnv(skill_factory=)``.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from coimagine.random_start import skills, spec  # noqa: E402
from coimagine.random_start.skills import PushSkill  # noqa: E402

CONTROLLER_VERSION = "segment-contact-v1"
#: keep the robot centre at least one robot radius plus a small margin inside the visible
#: segment end, so the disc contacts the face rather than the corner.
END_MARGIN = spec.ROBOT_RADIUS + 0.02
TANGENTIAL_GAIN = 1.0


class PushSkillSegment(PushSkill):
    """PUSH variant that targets a contact point inside the visible segment."""

    def _tangential_reference(self, observation, patch):
        points = getattr(patch, "points_world", None)
        tangent = getattr(patch, "tangent_world", None)
        if not points or tangent is None or len(points) < 2:
            return (0.0, 0.0)
        projections = [point[0] * tangent[0] + point[1] * tangent[1] for point in points]
        low, high = min(projections), max(projections)
        if high - low < 2.0 * END_MARGIN:
            return (0.0, 0.0)          # segment too short to re-centre safely
        robot = observation.own.x * tangent[0] + observation.own.y * tangent[1]
        target = min(max(robot, low + END_MARGIN), high - END_MARGIN)
        error = target - robot
        if abs(error) < 1e-3:
            return (0.0, 0.0)
        speed = max(-spec.APPROACH_MAX_SPEED,
                    min(spec.APPROACH_MAX_SPEED, TANGENTIAL_GAIN * error))
        return (speed * tangent[0], speed * tangent[1])


def make_skill_segment(action, force_limit: float = spec.MAX_FORCE):
    if action.kind == "push":
        return PushSkillSegment(action, force_limit)
    return skills.make_skill(action, force_limit)


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
