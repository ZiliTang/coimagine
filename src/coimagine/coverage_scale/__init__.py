"""coverage_scale_v1: random initial-state coverage crossed with data scale and width.

The round keeps the frozen continuous-2D environment, skills, scoring and 57-dimensional
encoding. What changes is the *support* of the data: instead of six fixed training scenes it
uses a deterministic registry of randomly sampled SceneSpecs whose initial object pose, goal
placement and passage geometry vary inside the ranges frozen in the task book
(``docs/tasks/20261013_0225_...``). Every other variable is held fixed.
"""

from .scene_registry import SceneRegistry, scene_from_dict, scene_to_dict
from .sampler import RootSpec, sample_root, sample_roots

__all__ = [
    "SceneRegistry",
    "scene_to_dict",
    "scene_from_dict",
    "RootSpec",
    "sample_root",
    "sample_roots",
]
