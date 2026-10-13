"""Frozen configuration for the continuous 2D rigid-body prototype.

All values are SI (metres, kilograms, seconds, newtons, radians). They are engineering
defaults, not tuned paper parameters; the point is that they are written down once, hashed
into every result record, and checked by the physics acceptance tests rather than adjusted
per scene.

The environment is a top-down planar approximation. There is no gravity; table drag is an
explicit viscous force ``-linear_damping * v`` (and ``-angular_damping * omega``) applied to
every dynamic body each physics substep. Contact friction and restitution are set on the
shapes and recorded here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class PhysicsConfig:
    """Backend time steps, materials and success tolerances."""

    physics_dt: float = 1.0 / 120.0
    control_dt: float = 1.0 / 20.0
    robot_radius: float = 0.18
    robot_mass: float = 4.0
    object_width: float = 1.0
    object_height: float = 0.25
    object_mass: float = 3.0
    max_force: float = 15.0
    linear_damping: float = 8.0
    angular_damping: float = 0.6
    friction: float = 0.6
    elasticity: float = 0.05
    #: The object must stay inside the goal and below these speeds for ``success_hold_time``.
    velocity_tolerance: float = 0.03
    angular_tolerance: float = 0.08
    success_hold_time: float = 0.5
    #: The goal rectangle is shrunk by this margin so a single frame at the edge is not success.
    goal_margin: float = 0.02
    #: Length of one synchronous joint-skill window, in control steps (2 s by default).
    window_steps: int = 40
    #: A body further than this outside the arena is a numerical/containment failure.
    out_of_bounds_margin: float = 0.5
    #: Non-finite state, or a penetration deeper than this, aborts the episode.
    penetration_limit: float = 0.05

    @property
    def substeps(self) -> int:
        ratio = self.control_dt / self.physics_dt
        substeps = round(ratio)
        if abs(ratio - substeps) > 1e-12:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        return substeps


@dataclass(frozen=True)
class ControlConfig:
    """Gains of the three feedback skills. Bounds come from ``PhysicsConfig.max_force``."""

    move_kp: float = 55.0
    move_kd: float = 22.0
    move_position_tolerance: float = 0.02
    move_speed_tolerance: float = 0.05
    push_speed: float = 0.26
    push_k_distance: float = 90.0
    push_k_velocity: float = 12.0
    push_lateral_kp: float = 60.0
    push_lateral_kd: float = 14.0
    hold_kd: float = 16.0
    hold_speed_tolerance: float = 0.03
    #: ``hold`` means damped braking to rest; stated once so it cannot be mixed with position hold.
    hold_semantics: str = "damped_brake"


@dataclass(frozen=True)
class CandidateConfig:
    """The finite candidate template (public geometry only)."""

    contact_samples_per_face: int = 2
    faces_per_robot: int = 4
    #: Multipliers of ``max_force``. One level keeps the joint set at 11x11=121 <= 128.
    force_levels: Tuple[float, ...] = (1.0,)
    staging_standoff: float = 0.35
    max_candidates: int = 128


@dataclass(frozen=True)
class MPCConfig:
    depths: Tuple[int, ...] = (1, 3)
    beam: int = 3
    horizon_windows: int = 12


@dataclass(frozen=True)
class Continuous2DConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    seeds: Tuple[int, ...] = (0, 1, 2)
    #: Cached digest. ``config_hash`` is read on every observation, so it must not rebuild
    #: the dictionary each time; recomputing it dominated the planner's runtime.
    _hash: str = field(init=False, default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        payload = json.dumps(
            {
                "physics": asdict(self.physics),
                "control": asdict(self.control),
                "candidates": asdict(self.candidates),
                "mpc": asdict(self.mpc),
                "seeds": list(self.seeds),
            },
            sort_keys=True,
        ).encode()
        object.__setattr__(self, "_hash", hashlib.sha256(payload).hexdigest()[:16])

    def as_dict(self) -> Dict[str, object]:
        return {
            "physics": asdict(self.physics),
            "control": asdict(self.control),
            "candidates": asdict(self.candidates),
            "mpc": asdict(self.mpc),
            "seeds": list(self.seeds),
            "config_hash": self._hash,
        }

    def config_hash(self) -> str:
        return self._hash


def default_config() -> Continuous2DConfig:
    return Continuous2DConfig()


#: Bumped whenever controller semantics change; written into every data record.
CONTROLLER_VERSION = "continuous2d-skill-v1"
