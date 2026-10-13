"""The frozen root set and its eight paired variants (task book section 3.2).

A *root* is one legal initial configuration plus a pair of nominal push actions. Roots are
stratified over six interaction classes, 24 per class (4 development + 20 test). Each root
gets eight variants -- baseline, two position-perturbation levels, two residual-velocity
levels, range noise, a 100 ms observation/message delay, and one mass/friction perturbation
combination -- and all four arms run on every (root, variant) pair.

The generator may use ground truth to lay out the initial state and to initialise the oracle
actions of arms O and P. Arms N and F are given only the nominal ``alpha`` and ``speed``; their
direction is read from the actual local scan at run time.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.config import Continuous2DConfig, default_config
from ..continuous2d.scenes import Box, Pose, SceneSpec
from ..continuous2d.skills import PushContact
from ..coverage_scale.sampler import strict_legal
from .contract import DELTA_MAX, DELTA_PER_SPEED

Vec = Tuple[float, float]

CLASSES: Tuple[str, ...] = (
    "solo_central",
    "solo_offset",
    "dual_same_side",
    "dual_rotate",
    "incompatible",
    "constrained",
)
DEV_PER_CLASS = 4
TEST_PER_CLASS = 20
TOTAL_PER_CLASS = DEV_PER_CLASS + TEST_PER_CLASS

ALPHA_CHOICES = (-math.radians(15.0), 0.0, math.radians(15.0))
SPEED_CHOICES = (0.08, 0.16, 0.26)
GAP_CHOICES = (0.02, 0.06, 0.12, 0.18)
FORCE_LIMIT = 15.0
ROOT_SALT = "local-contact-option-v1"

VARIANTS: Tuple[str, ...] = (
    "baseline", "pos_2mm", "pos_10mm", "vel_0.02", "vel_0.05", "noise", "delay", "param",
)


@dataclass(frozen=True)
class SceneTemplate:
    name: str
    arena: Tuple[float, float]
    obstacles: Tuple[Box, ...] = ()


TEMPLATES: Dict[str, SceneTemplate] = {
    "free": SceneTemplate("free", (3.2, 2.2)),
    "wall": SceneTemplate("wall", (2.4, 1.8), (Box(2.30, 0.90, 0.10, 0.90),)),
    "corner": SceneTemplate("corner", (2.0, 1.6), (Box(1.90, 1.50, 0.10, 0.10),)),
    "passage": SceneTemplate(
        "passage", (3.0, 2.0),
        (Box(1.60, 0.325, 0.09, 0.325), Box(1.60, 1.675, 0.09, 0.325)),
    ),
}


@dataclass
class RootSpec:
    root_id: str
    split: str
    interaction_class: str
    template: str
    seed: int
    object_pose: Pose
    robot_poses: Tuple[Pose, Pose]
    #: nominal per-robot action parameters; only alpha/speed reach arms N and F
    alpha: Tuple[Optional[float], Optional[float]]
    speed: Tuple[Optional[float], Optional[float]]
    gap: Tuple[Optional[float], Optional[float]]
    #: oracle contact point in the object frame and the world direction, for arms O and P
    contact_local: Tuple[Optional[Vec], Optional[Vec]]
    push_direction_world: Tuple[Optional[Vec], Optional[Vec]]
    active_robots: Tuple[int, ...]
    contact_face: Tuple[Optional[int], Optional[int]]
    contact_lateral: Tuple[Optional[float], Optional[float]]
    notes: str = ""

    def scene(self) -> SceneSpec:
        template = TEMPLATES[self.template]
        return SceneSpec(
            name=f"lco_{self.template}",
            category="LCO",
            variant=-1,
            arena=template.arena,
            obstacles=template.obstacles,
            goal=Box(0.0, 0.0, 0.05, 0.05),   # unused by this round
            object_pose=self.object_pose,
            robot_poses=self.robot_poses,
            description=f"local_contact_option {self.interaction_class} {self.template}",
            scene_version="local-contact-option-v1",
        )

    def oracle_spec(self, robot_id: int) -> Optional[PushContact]:
        if robot_id not in self.active_robots:
            return None
        return PushContact(
            object_id=2,
            contact_local=self.contact_local[robot_id],  # type: ignore[arg-type]
            push_direction_world=self.push_direction_world[robot_id],  # type: ignore[arg-type]
            force_limit=FORCE_LIMIT,
        )

    def local_parameters(self, robot_id: int) -> Optional[Dict[str, float]]:
        if robot_id not in self.active_robots:
            return None
        speed = float(self.speed[robot_id])
        delta = DELTA_PER_SPEED * speed
        if delta > DELTA_MAX + 1e-12:
            raise ValueError(
                f"{self.root_id}: delta={delta:.4f} m exceeds the {DELTA_MAX} m limit this round "
                "defines; resolve the configuration before running, do not silently clip")
        return {
            "alpha": float(self.alpha[robot_id]),
            "speed": speed,
            "active_duration": 2.0,
            "force_limit": FORCE_LIMIT,
            "delta": delta,
        }

    def as_metadata(self) -> Dict[str, object]:
        return {
            "root_id": self.root_id, "split": self.split,
            "interaction_class": self.interaction_class, "template": self.template,
            "seed": self.seed, "active_robots": list(self.active_robots),
            "alpha_deg": [None if a is None else round(math.degrees(a), 3) for a in self.alpha],
            "speed": list(self.speed), "gap": list(self.gap),
            "contact_face": list(self.contact_face),
            "contact_lateral": list(self.contact_lateral),
            "contact_local": [None if c is None else [c[0], c[1]] for c in self.contact_local],
            "push_direction_world": [None if d is None else [d[0], d[1]]
                                     for d in self.push_direction_world],
            "object_pose": [self.object_pose.x, self.object_pose.y, self.object_pose.angle],
            "robot_poses": [[p.x, p.y, p.angle] for p in self.robot_poses],
            "notes": self.notes,
        }


def _seed_for(split: str, interaction_class: str, index: int) -> int:
    digest = hashlib.sha256(f"{ROOT_SALT}|{split}|{interaction_class}|{index}".encode()).hexdigest()
    return int(digest[:9], 16)


def _face_point(pose: Pose, face: int, lateral: float) -> Tuple[Vec, Vec, Vec]:
    """Contact point, *inward* normal and local coordinate for one face of the object box.

    The inward normal points from the robot towards the object, which is the convention the
    control law uses for ``n`` and the convention the old oracle skill uses for
    ``push_direction_world`` (the robot stands at ``contact - n * radius``).
    """

    half_w, half_h = 0.5, 0.125
    cos, sin = math.cos(pose.angle), math.sin(pose.angle)
    if face == 0:      # local -y face; outward (0, -1), inward (0, +1)
        local, outward = (-half_w + (lateral + 1.0) * half_w, -half_h), (0.0, -1.0)
    elif face == 1:    # local +x face; outward (+1, 0)
        local, outward = (half_w, -half_h + (lateral + 1.0) * half_h), (1.0, 0.0)
    elif face == 2:    # local +y face; outward (0, +1)
        local, outward = (half_w - (lateral + 1.0) * half_w, half_h), (0.0, 1.0)
    else:              # local -x face; outward (-1, 0)
        local, outward = (-half_w, half_h - (lateral + 1.0) * half_h), (-1.0, 0.0)
    inward = (-outward[0], -outward[1])
    world = (
        pose.x + local[0] * cos - local[1] * sin,
        pose.y + local[0] * sin + local[1] * cos,
    )
    normal_world = (inward[0] * cos - inward[1] * sin, inward[0] * sin + inward[1] * cos)
    return world, normal_world, local


def _robot_pose_against(point: Vec, normal: Vec, gap: float, radius: float) -> Pose:
    centre = (point[0] - normal[0] * (radius + gap), point[1] - normal[1] * (radius + gap))
    return Pose(centre[0], centre[1], math.atan2(normal[1], normal[0]))


def _within_arena(pose: Pose, arena: Tuple[float, float], margin: float = 0.25) -> bool:
    return (margin <= pose.x <= arena[0] - margin) and (margin <= pose.y <= arena[1] - margin)


def _legal(scene: SceneSpec) -> bool:
    try:
        return bool(strict_legal(scene))
    except Exception:  # pragma: no cover - defensive
        return False


def sample_root(split: str, interaction_class: str, index: int,
                *, max_attempts: int = 400) -> Optional[RootSpec]:
    """Deterministic stratified root. Returns ``None`` if the cell cannot be filled."""

    seed = _seed_for(split, interaction_class, index)
    rng = random.Random(seed)
    template_name = {
        "solo_central": "free",
        "solo_offset": "free",
        "dual_same_side": "free",
        "dual_rotate": "free",
        "incompatible": "free",
        "constrained": ("passage", "wall", "corner")[index % 3],
    }[interaction_class]
    template = TEMPLATES[template_name]
    radius = default_config().physics.robot_radius
    for _ in range(max_attempts):
        object_x = rng.uniform(0.9, template.arena[0] - 1.1)
        object_y = rng.uniform(0.6, template.arena[1] - 0.6)
        object_angle = rng.uniform(-math.pi, math.pi)
        object_pose = Pose(object_x, object_y, object_angle)
        alpha = [None, None]
        speed = [None, None]
        gap = [None, None]
        contact_local: List[Optional[Vec]] = [None, None]
        direction: List[Optional[Vec]] = [None, None]
        face: List[Optional[int]] = [None, None]
        lateral: List[Optional[float]] = [None, None]
        poses = [Pose(0.3, 0.3, 0.0), Pose(0.3, template.arena[1] - 0.3, 0.0)]

        if interaction_class == "solo_central":
            active = (0,)
            face[0] = rng.randrange(4)
            lateral[0] = rng.choice((-0.05, 0.0, 0.05))
        elif interaction_class == "solo_offset":
            active = (0,)
            face[0] = rng.choice((1, 3))       # the long faces, so lateral offset matters
            lateral[0] = rng.choice((-0.7, -0.5, 0.5, 0.7))
        elif interaction_class == "dual_same_side":
            active = (0, 1)
            shared = rng.randrange(4)
            face[0] = face[1] = shared
            lateral[0] = rng.choice((-0.6, -0.3))
            lateral[1] = rng.choice((0.3, 0.6))
        elif interaction_class == "dual_rotate":
            active = (0, 1)
            face[0], face[1] = rng.choice(((0, 2), (1, 3), (0, 1), (2, 3)))
            lateral[0] = rng.uniform(-0.5, 0.5)
            lateral[1] = rng.uniform(-0.5, 0.5)
        elif interaction_class == "incompatible":
            active = (0, 1)
            if index % 2 == 0:
                face[0], face[1] = 1, 3          # directly opposing faces
                lateral[0] = rng.uniform(-0.3, 0.3)
                lateral[1] = rng.uniform(-0.3, 0.3)
            else:
                face[0] = face[1] = rng.randrange(4)
                lateral[0], lateral[1] = -0.4, 0.4
        else:  # constrained
            active = (0,) if index % 2 == 0 else (0, 1)
            face[0] = rng.randrange(4)
            lateral[0] = rng.choice((-0.4, 0.0, 0.4))
            if len(active) == 2:
                face[1] = (face[0] + 2) % 4
                lateral[1] = rng.choice((-0.4, 0.4))

        for robot_id in active:
            alpha[robot_id] = float(rng.choice(ALPHA_CHOICES))
            speed[robot_id] = float(rng.choice(SPEED_CHOICES))
            gap[robot_id] = float(rng.choice(GAP_CHOICES))
            point, normal, local = _face_point(object_pose, face[robot_id], lateral[robot_id])
            contact_local[robot_id] = local
            direction[robot_id] = normal
            pose = _robot_pose_against(point, normal, gap[robot_id], radius)
            if not _within_arena(pose, template.arena):
                break
            poses[robot_id] = pose
        else:
            if interaction_class == "incompatible" and index % 2 == 0 and speed[0] and speed[1]:
                # mismatched magnitudes on opposing faces: a legal, expected-to-fail input
                speed[0] = 0.26
                speed[1] = 0.08
            if len(active) == 1:
                # the passive robot waits far from the object, braking
                angle = math.atan2(object_pose.y - poses[1 - active[0]].y,
                                   object_pose.x - poses[1 - active[0]].x) + math.pi
                far = (object_pose.x + math.cos(angle) * 0.85, object_pose.y + math.sin(angle) * 0.85)
                far = (min(max(far[0], 0.25), template.arena[0] - 0.25),
                       min(max(far[1], 0.25), template.arena[1] - 0.25))
                poses[1 - active[0]] = Pose(far[0], far[1], angle)
            scene = SceneSpec(
                name=f"lco_{template_name}", category="LCO", variant=-1,
                arena=template.arena, obstacles=template.obstacles,
                goal=Box(0.0, 0.0, 0.05, 0.05), object_pose=object_pose,
                robot_poses=(poses[0], poses[1]),
                description="local_contact_option root", scene_version="local-contact-option-v1",
            )
            if not _legal(scene):
                continue
            return RootSpec(
                root_id=f"{split}/{interaction_class}/{index:03d}",
                split=split, interaction_class=interaction_class, template=template_name,
                seed=seed, object_pose=object_pose, robot_poses=(poses[0], poses[1]),
                alpha=(alpha[0], alpha[1]), speed=(speed[0], speed[1]), gap=(gap[0], gap[1]),
                contact_local=(contact_local[0], contact_local[1]),
                push_direction_world=(direction[0], direction[1]),
                active_robots=tuple(active),
                contact_face=(face[0], face[1]), contact_lateral=(lateral[0], lateral[1]),
                notes="oracle contact point recorded for arms O/P only",
            )
    return None


def build_roots(split: str, per_class: int, *, max_seed_tries: int = 40) -> List[RootSpec]:
    roots: List[RootSpec] = []
    for interaction_class in CLASSES:
        accepted: List[RootSpec] = []
        index = 0
        while len(accepted) < per_class:
            root = sample_root(split, interaction_class, index)
            index += 1
            if root is not None:
                accepted.append(root)
            if index > per_class * max_seed_tries:
                raise RuntimeError(f"could not fill {split}/{interaction_class}")
        roots.extend(accepted)
    return roots


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    name: str
    position_delta: float = 0.0
    velocity: float = 0.0
    noise_sigma: float = 0.0
    observation_delay: float = 0.0
    mass_scale: float = 1.0
    friction_scale: float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name, "position_delta": self.position_delta, "velocity": self.velocity,
            "noise_sigma": self.noise_sigma, "observation_delay": self.observation_delay,
            "mass_scale": self.mass_scale, "friction_scale": self.friction_scale,
        }


VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec("baseline"),
    VariantSpec("pos_2mm", position_delta=0.002),
    VariantSpec("pos_10mm", position_delta=0.010),
    VariantSpec("vel_0.02", velocity=0.02),
    VariantSpec("vel_0.05", velocity=0.05),
    VariantSpec("noise", noise_sigma=0.002),
    VariantSpec("delay", observation_delay=0.100),
    VariantSpec("param", mass_scale=1.10, friction_scale=0.90),
)


def variant_scene(root: RootSpec, variant: VariantSpec) -> Tuple[SceneSpec, Dict[str, object]]:
    """The scene of one variant: position perturbations belong to the *initial state*.

    They are written into the ``SceneSpec`` rather than poked into the engine after a reset,
    because the engine's exact replay can only reproduce a world that was built from its own
    scene. Legal configurations are kept; a perturbed configuration that would overlap is
    rejected with a different deterministic direction, and the rejection count is reported.
    """

    report: Dict[str, object] = {"variant": variant.name, "rejections": 0}
    scene = root.scene()
    if variant.position_delta <= 0.0:
        return scene, report
    rng = random.Random(root.seed ^ 0x5EED)
    for attempt in range(64):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        delta = variant.position_delta
        poses = []
        for robot_id, pose in enumerate(root.robot_poses):
            theta = angle + robot_id * math.pi / 2.0
            poses.append(Pose(pose.x + math.cos(theta) * delta, pose.y + math.sin(theta) * delta,
                              pose.angle))
        candidate = replace(scene, robot_poses=(poses[0], poses[1]))
        if _legal(candidate):
            report["rejections"] = attempt
            report["direction"] = angle
            return candidate, report
    report["position_perturbation_skipped"] = True
    return scene, report


def variant_config(config: Continuous2DConfig, variant: VariantSpec) -> Continuous2DConfig:
    """Mass and friction perturbation. Pressure test only; the nominal defaults never move."""

    if variant.mass_scale == 1.0 and variant.friction_scale == 1.0:
        return config
    physics = replace(
        config.physics,
        object_mass=config.physics.object_mass * variant.mass_scale,
        friction=config.physics.friction * variant.friction_scale,
    )
    return replace(config, physics=physics)


def initial_velocities(root: RootSpec, variant: VariantSpec) -> Dict[str, Tuple[float, float]]:
    """Residual initial velocity, drawn from the root seed. Zero except for the velocity variants."""

    out = {"object": (0.0, 0.0), "robot0": (0.0, 0.0), "robot1": (0.0, 0.0)}
    if variant.velocity <= 0.0:
        return out
    rng = random.Random(root.seed ^ 0x1234)
    object_angle = rng.uniform(0.0, 2.0 * math.pi)
    out["object"] = (math.cos(object_angle) * variant.velocity, math.sin(object_angle) * variant.velocity)
    for robot_id in (0, 1):
        angle = object_angle + math.pi * (robot_id + 1) / 2.0
        out[f"robot{robot_id}"] = (math.cos(angle) * variant.velocity,
                                   math.sin(angle) * variant.velocity)
    return out


def apply_initial_velocities(env, velocities: Dict[str, Tuple[float, float]]) -> None:
    """Write the variant's residual velocity once, as an initial condition.

    This is the same kind of write as the pose inside ``SceneSpec``: it happens before the
    first control period, is identical for all four arms, and is never repeated by a stage
    transition. No other body write exists anywhere in this round.
    """

    env.backend.object_body.velocity = tuple(velocities["object"])
    for robot_id, body in enumerate(env.backend.robot_bodies):
        body.velocity = tuple(velocities[f"robot{robot_id}"])


def manifest(dev: Sequence[RootSpec], test: Sequence[RootSpec]) -> Dict[str, object]:
    def counts(roots):
        out: Dict[str, int] = {}
        for root in roots:
            name = (root["interaction_class"] if isinstance(root, dict) else root.interaction_class)
            out[name] = out.get(name, 0) + 1
        return out

    return {
        "root_salt": ROOT_SALT,
        "classes": list(CLASSES),
        "dev_roots": len(dev), "test_roots": len(test),
        "dev_per_class": counts(dev), "test_per_class": counts(test),
        "variants": [spec.as_dict() for spec in VARIANT_SPECS],
        "alpha_choices_deg": [round(math.degrees(value), 3) for value in ALPHA_CHOICES],
        "speed_choices": list(SPEED_CHOICES),
        "gap_choices": list(GAP_CHOICES),
        "executions_planned": len(test) * len(VARIANT_SPECS) * 4,
    }
