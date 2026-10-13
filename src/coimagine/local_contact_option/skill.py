"""The new skill spec: ``LocalSurfacePush(alpha, speed, active_duration, force_limit)``.

``alpha`` is an angle relative to the *currently observed* inward surface normal; its sign is
fixed by ``t = Rot90(n)``. ``local_patch`` is a handle to the fragment observed at the moment
the skill starts; it is not a permanent object coordinate frame, and the skill makes no promise
that the material contact point stays fixed -- sliding along the face and sliding onto a corner
are both measured and reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .contract import DELTA_DEFAULT, DELTA_MAX, FORCE_LIMIT, PUSH_TARGET_ACTIVE


@dataclass(frozen=True)
class LocalSurfacePush:
    alpha: float
    speed: float
    active_duration: float = PUSH_TARGET_ACTIVE
    force_limit: float = FORCE_LIMIT
    #: virtual advance bias of the surface-position controller (arm X), in metres. It is a
    #: length, not an expected object displacement.
    delta: float = DELTA_DEFAULT
    kind: str = "local_surface_push"

    @property
    def label(self) -> str:
        return (f"local_surface_push(alpha={math.degrees(self.alpha):+.1f}deg,"
                f"v={self.speed:.2f},delta={self.delta:.3f},T={self.active_duration:.1f},"
                f"F={self.force_limit:.1f})")

    def as_dict(self) -> Dict[str, float]:
        return {
            "kind": "local_surface_push", "alpha": float(self.alpha), "speed": float(self.speed),
            "active_duration": float(self.active_duration), "force_limit": float(self.force_limit),
            "delta": float(self.delta),
        }
