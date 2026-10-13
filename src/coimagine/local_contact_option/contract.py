"""Frozen parameters, stages and exit reasons for the local-contact option round.

Every number in this module is written down **before** any test execution and is copied into
``docs/local_contact_option_v1/CONTRACT.md``. The physics configuration is not touched: object
mass and shape, friction, damping, robot size and mass, force limit and physics step all stay
at their frozen ``continuous2d`` values, so a positive result cannot come from making the
contact easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

LOCAL_CONTACT_VERSION = "local-contact-option-v1"

#: Local scan geometry. Ideal sensing, no noise unless the variant asks for it.
SCAN_RANGE = 0.6           # m
SCAN_RAYS = 72             # uniform in the robot frame
MIN_POINTS_FOR_NORMAL = 3  # a surface needs at least three returns before it has a normal
RANGE_NOISE_SIGMA = 0.002  # m, used only by the observation-noise variant

#: Normal filtering and jump rejection.
NORMAL_FILTER_TAU = 0.10   # s, first-order filter on the unit normal
NORMAL_JUMP_LIMIT = 30.0   # degrees between two consecutive frames -> patch switch
PATCH_GAP_TOLERANCE = 0.03  # m, angular/range continuity for grouping returns into a patch
PATCH_MIN_ARC = 2           # returns; a fitted segment needs MIN_POINTS_FOR_NORMAL points

#: Reference-velocity generation and the bounded velocity feedback.
A_REF_MAX = 0.5            # m/s^2, limit on the reference-velocity change per second
KV_DEFAULT = 40.0          # N s / m
KV_DEV_CANDIDATES: Tuple[float, ...] = (20.0, 40.0, 80.0)
FORCE_LIMIT = 15.0         # N, identical to PhysicsConfig.max_force

#: Stage timing.
APPROACH_TIMEOUT = 5.0     # s
#: APPROACH climbs at ``gain * gap`` clipped into [min, max]; the old 0.08 m/s ceiling made
#: the approach dominate the wall clock (median 1.7 s of a 6.3 s execution) while the push
#: itself only lasted 2 s.
APPROACH_MAX_SPEED = 0.08  # m/s
APPROACH_MIN_SPEED = 0.02  # m/s
APPROACH_GAIN = 2.0        # 1/s
APPROACH_CONTACT_SPEED = 0.02  # m/s once the gap is closed
READY_CONTACT_HOLD = 0.10  # s of continuous sensed contact
READY_PRECONTACT_SPEED = 0.01  # m/s, keep touching without running away
READY_BOTH_HOLD = 0.10     # s both sides must be ready together before PUSH
PUSH_TARGET_ACTIVE = 2.0   # s of accumulated effective contact
PUSH_WALL_LIMIT = 3.0      # s
CONTACT_LOSS_TOLERANCE = 0.15  # s of continuous contact loss -> abort
PATCH_LOSS_TOLERANCE = 0.15    # s without a usable local patch -> abort
BRAKE_SETTLE_LIMIT = 4.0   # s
SETTLE_LINEAR = 0.03       # m/s
SETTLE_ANGULAR = 0.08      # rad/s
SETTLE_SURFACE = 0.02      # m/s, estimated along the local surface normal
SETTLE_HOLD = 0.30         # s
SURFACE_MATCH_TOLERANCE = 0.05  # m, nearest-neighbour association between two scans
SURFACE_MATCH_MIN_POINTS = 3
#: A scan older than this may not contribute to the settle verdict (frozen this round).
SURFACE_FRESHNESS = 0.20  # s
#: Diagnostic margin between the robust median and the RMS. A large gap means a minority of
#: matches disagree with the majority, i.e. the association is unreliable this period.
SURFACE_CONSISTENCY = 0.05  # m/s

# ---------------------------------------------------------------------------
# surface-position-feedback push (arm X). Not a gain search: these are the single fixed
# configuration the task book specifies.
# ---------------------------------------------------------------------------

KP_POSITION = 100.0        # N/m, position-error stiffness
KD_POSITION = 25.0         # N s/m, velocity damping; equals the frozen baseline Kv by design
E_MAX_POSITION = 0.10      # m, cap on the position error fed to the stiffness term
BETA_RAMP = 0.20           # s, linear 0 -> 1 ramp of the virtual advance bias after PUSH entry
DELTA_MAX = 0.08           # m, largest virtual advance bias this round defines
DELTA_DEFAULT = 0.04       # m, default bias for the free-standing skill spec
PATCH_SUPPORT_MARGIN = 0.01  # m, foot-of-perpendicular may leave the observed segment by this
#: delta = Kv * s / Kp, the mapping that makes the two laws algebraically identical at zero gap
DELTA_PER_SPEED = KD_POSITION / KP_POSITION

#: Equal-wall-clock arm F.
F_WALL_CLOCK = 2.0

#: Value-frame alignment tolerances used by the evaluator only.
MIN_MOTION_FOR_TWIST = 0.02   # m
MIN_ROTATION_FOR_TWIST = 0.05  # rad
MICRO_MOTION_LINEAR = 0.02    # m
MICRO_MOTION_ANGULAR = 0.03   # rad

#: Bootstrap and clustering.
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261013


class Stage(str, Enum):
    APPROACH = "approach"
    READY = "ready"
    PUSH = "push"
    BRAKE = "brake"
    SETTLE = "settle"
    DONE = "done"


class Exit(str, Enum):
    PUSH_COMPLETE = "push_complete"
    SETTLED_OBSERVED = "settled_observed"
    PATCH_SUPPORT_LOST = "patch_support_lost"
    SETTLE_TIMEOUT = "settle_timeout"
    SETTLE_UNOBSERVED = "settle_unobserved"
    APPROACH_TIMEOUT = "approach_timeout"
    CONTACT_LOST = "contact_lost"
    PATCH_LOST = "patch_lost"
    SURFACE_UNOBSERVED = "surface_unobserved"
    FACE_SWITCH = "face_switch"
    PUSH_WALL_TIMEOUT = "push_wall_timeout"
    TEAMMATE_ABORT = "teammate_abort"
    WALL_CLOCK_END = "wall_clock_end"
    ENVIRONMENT_TERMINATION = "environment_termination"
    NOT_READY = "not_ready"


TERMINAL_EXITS = tuple(exit_value.value for exit_value in Exit)


@dataclass(frozen=True)
class OptionConfig:
    """All tunables in one place so a run records exactly what it used."""

    scan_range: float = SCAN_RANGE
    scan_rays: int = SCAN_RAYS
    min_points_for_normal: int = MIN_POINTS_FOR_NORMAL
    normal_filter_tau: float = NORMAL_FILTER_TAU
    normal_jump_limit_deg: float = NORMAL_JUMP_LIMIT
    a_ref_max: float = A_REF_MAX
    kv: float = KV_DEFAULT
    #: scale parameters of the reference generator. They are configuration (not constants) so
    #: that the aggressive-first / stable-second comparison is a recorded parameter change and
    #: not an edit smuggled into the middle of the round.
    approach_min_speed: float = APPROACH_MIN_SPEED
    approach_max_speed: float = APPROACH_MAX_SPEED
    approach_gain: float = APPROACH_GAIN
    approach_contact_speed: float = APPROACH_CONTACT_SPEED
    ready_precontact_speed: float = READY_PRECONTACT_SPEED
    #: arm X (surface position feedback)
    kp: float = KP_POSITION
    kd: float = KD_POSITION
    e_max: float = E_MAX_POSITION
    beta_ramp: float = BETA_RAMP
    patch_support_margin: float = PATCH_SUPPORT_MARGIN
    force_limit: float = FORCE_LIMIT
    approach_timeout: float = APPROACH_TIMEOUT
    push_target_active: float = PUSH_TARGET_ACTIVE
    push_wall_limit: float = PUSH_WALL_LIMIT
    contact_loss_tolerance: float = CONTACT_LOSS_TOLERANCE
    patch_loss_tolerance: float = PATCH_LOSS_TOLERANCE
    brake_settle_limit: float = BRAKE_SETTLE_LIMIT
    settle_linear: float = SETTLE_LINEAR
    settle_angular: float = SETTLE_ANGULAR
    settle_surface: float = SETTLE_SURFACE
    settle_hold: float = SETTLE_HOLD
    ready_contact_hold: float = READY_CONTACT_HOLD
    ready_both_hold: float = READY_BOTH_HOLD
    f_wall_clock: float = F_WALL_CLOCK
    range_noise_sigma: float = 0.0
    observation_delay: float = 0.0
    version: str = LOCAL_CONTACT_VERSION

    def as_dict(self) -> Dict[str, object]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


#: Arm labels and their meaning (task book section 3.1).
ARM_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "O": {"push_control": "old", "protocol": "old_window",
          "meaning": "original control and original 2 s window, untouched"},
    "P": {"push_control": "old_oracle", "protocol": "staged",
          "meaning": "old force formula, new approach/handshake/effective-push/brake protocol"},
    "F": {"push_control": "local", "protocol": "equal_wall_clock",
          "meaning": "new local-surface control, 2 s total wall clock, no terminal settling"},
    "N": {"push_control": "local", "protocol": "staged",
          "meaning": "new local-surface control with the full staged protocol"},
    "V": {"push_control": "local_velocity", "protocol": "staged",
          "meaning": "frozen v3 velocity control with the full staged protocol (this round's V)"},
    "X": {"push_control": "local_surface_position", "protocol": "staged",
          "meaning": "surface-position feedback push with the identical staged protocol (arm X)"},
}
