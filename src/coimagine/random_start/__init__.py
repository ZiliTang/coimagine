"""Random-start, reposition-capable continuous-2D environment.

This package is the round-12 environment: two robots start near an object at random
legal poses, use a 360 degree range scanner (72 rays, 1.2 m, 20 Hz) plus contact sensing,
and choose one of 15 skills each at every joint boundary. The physics, object size, mass,
friction and actuator limit are the frozen ``continuous2d`` values; only the scene
sampling, the sensing interface and the skill set are new.
"""

from .spec import (  # noqa: F401
    CONTROL_DT,
    JOINT_ACTION_COUNT,
    LIDAR_RANGE,
    LIDAR_RAYS,
    ROBOT_ACTION_COUNT,
    action_table,
    joint_action_index,
    joint_action_label,
)
