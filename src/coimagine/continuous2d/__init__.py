"""Continuous 2D rigid-body prototype (Phase C/D of the migration task).

Top-down planar approximation of two omnidirectional circular robots pushing a rectangular
object through a static environment, with three feedback skills and a synchronous
joint-skill protocol. The physics backend is Pymunk/Chipmunk 6.9.0; no custom collision
engine is used and no learned model is trained this round.
"""

from .config import (
    CONTROLLER_VERSION,
    CandidateConfig,
    Continuous2DConfig,
    ControlConfig,
    MPCConfig,
    PhysicsConfig,
    default_config,
)
from .environment import (
    Continuous2DEnv,
    SkillResult,
    StepRecord,
    TrajectoryResult,
)
from .scenes import (
    S0_VARIANTS,
    S1_VARIANTS,
    Box,
    Pose,
    SceneSpec,
    default_scenes,
    scene_by_name,
)
from .skills import Hold, JointSpec, MoveTo, Observation, PushContact, Skill

__all__ = [
    "CONTROLLER_VERSION",
    "Box",
    "CandidateConfig",
    "Continuous2DConfig",
    "Continuous2DEnv",
    "ControlConfig",
    "Hold",
    "JointSpec",
    "MPCConfig",
    "MoveTo",
    "Observation",
    "PhysicsConfig",
    "Pose",
    "PushContact",
    "S0_VARIANTS",
    "S1_VARIANTS",
    "SceneSpec",
    "Skill",
    "SkillResult",
    "StepRecord",
    "TrajectoryResult",
    "default_config",
    "default_scenes",
    "scene_by_name",
]
