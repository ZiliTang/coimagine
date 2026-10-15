"""Option-level RL environment: one decision per skill boundary, physics never reset.

Information boundary (task book section 4.1). The actor and the critic both read **only**:

* each robot's own body kinematics (position, heading, velocity) and the teammate's shared
  kinematics;
* what each robot's own range scan reports about the object surface (distance, normal, contact);
* the public goal region, the public arena and the public skill menu;
* the clock: executed time and the public remaining task budget.

The true object pose and velocity are read **only** by the reward function, are marked privileged,
and are never assembled into the observation vector. There is no privileged feature in the actor
input, and the candidate mask is the full public menu, never filtered by hidden truth.

One episode is a sequence of options on **one** world: the object, the robots, the physics engine and
the goal are untouched between skills; only the skill's internal controller state is rebuilt.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.continuous2d.config import Continuous2DConfig  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.local_contact_option.contract import (  # noqa: E402
    DELTA_PER_SPEED,
    FORCE_LIMIT,
    OptionConfig,
    PUSH_TARGET_ACTIVE,
)
from coimagine.local_contact_option.option import (  # noqa: E402
    OptionExecutor,
    build_observation,
)
from coimagine.local_contact_option.roots import (  # noqa: E402
    RootSpec,
    VariantSpec,
    apply_initial_velocities,
    build_roots,
)
from coimagine.local_contact_option.runner import build_snapshot  # noqa: E402
from coimagine.local_contact_option.skill import LocalSurfacePush  # noqa: E402

#: the public joint skill menu. Every entry is expressible in each robot's own legal frame, and the
#: menu is fixed and public: no entry is removed using hidden truth.
ALPHA_CHOICES = (-math.radians(15.0), 0.0, math.radians(15.0))
SPEED_CHOICES = (0.08, 0.16, 0.26)
PAIRINGS = ((0, 1), (1, 0), (0, ), (1, ))


@dataclass(frozen=True)
class MenuAction:
    """One joint skill: which robots push, at what heading offset and what commanded speed."""

    active: Tuple[int, ...]
    alpha: Tuple[float, ...]
    speed: Tuple[float, ...]

    def label(self) -> str:
        parts = []
        for index, robot in enumerate(self.active):
            parts.append(f"r{robot}:a={math.degrees(self.alpha[index]):+.0f}deg,"
                         f"v={self.speed[index]:.2f}")
        return "|".join(parts)


def build_menu() -> List[MenuAction]:
    menu: List[MenuAction] = []
    for active in PAIRINGS:
        for alpha_index in range(len(ALPHA_CHOICES) ** len(active)):
            for speed_index in range(len(SPEED_CHOICES) ** len(active)):
                alphas = tuple(ALPHA_CHOICES[(alpha_index // (len(ALPHA_CHOICES) ** k))
                                             % len(ALPHA_CHOICES)] for k in range(len(active)))
                speeds = tuple(SPEED_CHOICES[(speed_index // (len(SPEED_CHOICES) ** k))
                                             % len(SPEED_CHOICES)] for k in range(len(active)))
                menu.append(MenuAction(active=active, alpha=alphas, speed=speeds))
    return menu


MENU = build_menu()
#: the four collaboration modes present in the menu, for per-mode reporting
MODE_OF_ACTION = {(0, 1): "both", (1, 0): "both", (0,): "r0_only", (1,): "r1_only"}


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]
    #: one legal observation per 0.05 s control period of the skill that was just executed. The
    #: policy replays these through its GRU, so memory advances on the real grid while the high-level
    #: action is only sampled at skill boundaries.
    inner_observations: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    skill_steps: int = 0
    #: true state per control period of the executed skill, for rendering and for privileged labels
    #: only. Never concatenated into an observation vector.
    trace: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class ObservationSpec:
    """The layout of the observation vector, recorded so a loader can rebuild it."""

    fields: Tuple[str, ...] = (
        "r0_x", "r0_y", "r0_theta", "r0_vx", "r0_vy", "r0_omega",
        "r0_patch_valid", "r0_patch_distance", "r0_patch_normal_sin", "r0_patch_normal_cos",
        "r0_contacting",
        "r1_x", "r1_y", "r1_theta", "r1_vx", "r1_vy", "r1_omega",
        "r1_patch_valid", "r1_patch_distance", "r1_patch_normal_sin", "r1_patch_normal_cos",
        "r1_contacting",
        "teammate_dx", "teammate_dy", "teammate_dtheta",
        "goal_dx", "goal_dy", "goal_distance",
        "object_sensed_dx", "object_sensed_dy",
        "elapsed_fraction", "skills_used_fraction",
    )

    @property
    def width(self) -> int:
        return len(self.fields)


@dataclass
class OptionRLConfig:
    #: per-0.05 s discount, so a skill of n steps carries gamma_step ** n
    gamma_step: float = 0.997
    #: frozen after the calibration in CALIBRATION.md: 0.45 m is reachable in 2-3 effective skills
    #: (median effective push 0.25 m), and 0.20 m tolerance absorbs the overshoot that the skills
    #: cannot undo. ``hold_steps`` counts **skills**, so 2 means the object must sit inside the
    #: region for two consecutive boundaries.
    max_skills: int = 12
    goal_tolerance: float = 0.20
    hold_steps: int = 2
    progress_scale: float = 1.0
    success_bonus: float = 10.0
    failure_penalty: float = -2.0
    time_cost_per_step: float = 0.002
    potential_weight: float = 1.0
    seed: int = 0

    def discount_for(self, steps: int) -> float:
        return float(self.gamma_step ** steps)


class OptionRLEnv:
    """One world, a public goal, and a sequence of skills that never resets the world."""

    def __init__(self, config: Optional[OptionRLConfig] = None,
                 option_config: Optional[OptionConfig] = None,
                 base_config: Optional[Continuous2DConfig] = None,
                 arm: str = "V"):
        self.config = config or OptionRLConfig()
        self.option_config = option_config or OptionConfig()
        self.base_config = base_config or Continuous2DConfig()
        self.spec = ObservationSpec()
        self.arm = arm
        self.env: Optional[Continuous2DEnv] = None
        self.goal: np.ndarray = np.zeros(2)
        self.root: Optional[RootSpec] = None
        self.variant = VariantSpec(name="baseline")
        self.skills_used = 0
        self.steps = 0
        self.hold = 0
        self.trajectory: List[Dict[str, object]] = []
        self.boundaries: List[Dict[str, object]] = []
        self.initial_object: Optional[np.ndarray] = None
        self._last_potential = 0.0
        self._rng = np.random.default_rng(self.config.seed)

    # -- privileged access, reward only ------------------------------------
    def privileged_object(self) -> np.ndarray:
        """True object pose and velocity. **Never** placed in an observation vector."""

        observation = self.env.observation()
        return np.asarray(observation.object_pose(), dtype=np.float64)

    def reset(self, root: RootSpec, goal: np.ndarray,
              noise_rng=None) -> np.ndarray:
        snapshot, config, initial_object, _initial_robots, _report, velocities = build_snapshot(
            root, self.variant, self.base_config)
        self.env = Continuous2DEnv(config)
        self.env.restore(snapshot, exact=True)
        apply_initial_velocities(self.env, velocities)
        self.root = root
        self.goal = np.asarray(goal, dtype=np.float64)
        self.skills_used = 0
        self.steps = 0
        self.hold = 0
        self.trajectory = []
        self.boundaries = []
        self.initial_object = np.asarray(initial_object, dtype=np.float64)
        # the distance the episode starts at, so a caller never has to re-derive it
        self.start_distance = float(np.hypot(self.initial_object[0] - self.goal[0],
                                             self.initial_object[1] - self.goal[1]))
        self._last_potential = self._potential()
        self._noise_rng = noise_rng
        return self._observation()

    # -- reward ------------------------------------------------------------
    def _potential(self) -> float:
        """Public-configurable potential: the object's distance to the goal, from true state.

        The word "potential" refers to the shaping term; the value itself comes from the privileged
        object pose and is used **only** in the reward, never in an observation.
        """

        pose = self.privileged_object()
        distance = float(np.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1]))
        return -distance

    def _goal_reached(self) -> bool:
        pose = self.privileged_object()
        return float(np.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1])) <= \
            self.config.goal_tolerance

    def _task_budget_expired(self) -> bool:
        """True when the public task budget ran out rather than the sampling buffer being cut."""

        return self.skills_used >= self.config.max_skills

    # -- observation -------------------------------------------------------
    #: nominal physics steps per skill, used only to normalise the clock feature
    NOMINAL_SKILL_STEPS = 80

    def _clock_features(self, elapsed_seconds: float) -> List[float]:
        """The two clock columns, defined once so boundary and inner frames cannot disagree."""

        horizon = max(1.0, self.config.max_skills * self.NOMINAL_SKILL_STEPS * 0.05)
        return [min(1.0, elapsed_seconds / horizon),
                self.skills_used / max(1, self.config.max_skills)]

    def _robot_block(self, robot_id: int) -> List[float]:
        """One robot's legal block: own kinematics plus what its **own** scan reports."""

        observation = self.env.observation()
        pose = observation.robots[robot_id]
        sensed = build_observation(self.env, self.config_option(), robot_id, stage="idle",
                                   teammate=None, skill={"alpha": 0.0, "speed": 0.0})
        patch = sensed.patch
        valid = 1.0 if (patch is not None and patch.arc_rays > 0) else 0.0
        distance = float(patch.distance) if valid else 0.0
        normal = patch.normal_world if valid else (0.0, 0.0)
        return [pose.x, pose.y, pose.angle, pose.vx, pose.vy, pose.omega,
                valid, distance, float(normal[0]), float(normal[1]),
                1.0 if sensed.contacting else 0.0]

    def _object_block(self, robot0_block) -> List[float]:
        """The two ``object_sensed_*`` columns, taken from robot 0's own scan block.

        They are listed separately in ``ObservationSpec`` but must be derived from the same scan,
        so they are read out of the already-built robot block rather than sensed a second time.
        """

        return [float(robot0_block[7]), 0.0]

    def _assemble(self, robot_blocks, teammate_block, goal_block, object_block,
                  clock_block) -> np.ndarray:
        values = list(robot_blocks[0]) + list(robot_blocks[1]) + list(teammate_block) \
            + list(goal_block) + list(object_block) + list(clock_block)
        array = np.asarray(values, dtype=np.float32)
        if array.shape[0] != self.spec.width:
            raise RuntimeError(f"observation width {array.shape[0]} != spec {self.spec.width}")
        return array

    def _observation(self) -> np.ndarray:
        """The boundary observation, built from the shared field assembler."""

        own0 = self.env.observation().robots[0]
        own1 = self.env.observation().robots[1]
        teammate = [own1.x - own0.x, own1.y - own0.y,
                    math.atan2(math.sin(own1.angle - own0.angle),
                               math.cos(own1.angle - own0.angle))]
        goal = [self.goal[0] - own0.x, self.goal[1] - own0.y,
                float(np.hypot(self.goal[0] - own0.x, self.goal[1] - own0.y))]
        blocks = [self._robot_block(0), self._robot_block(1)]
        return self._assemble(blocks, teammate, goal, self._object_block(blocks[0]),
                              self._clock_features(self.steps * 0.05))

    def config_option(self) -> OptionConfig:
        return self.option_config

    # -- step --------------------------------------------------------------
    def step(self, action_index: int) -> StepResult:
        action = MENU[action_index]
        specs: Dict[int, object] = {}
        for index, robot_id in enumerate(action.active):
            speed = float(action.speed[index])
            specs[robot_id] = LocalSurfacePush(
                alpha=float(action.alpha[index]), speed=speed,
                active_duration=PUSH_TARGET_ACTIVE, force_limit=FORCE_LIMIT,
                delta=DELTA_PER_SPEED * speed)
        before = self._potential()
        executor = OptionExecutor(self.option_config, self.arm, log_every=1)
        result = executor.run(self.env, root_id=getattr(self.root, "root_id", "rl"),
                              variant=self.variant.name, joint_specs=specs,
                              active_robots=tuple(action.active),
                              noise_rng=getattr(self, "_noise_rng", None),
                              record_trajectory=False)
        after = self._potential()
        # ``physics_steps`` is how many control periods actually advanced the world (round-6 fix)
        steps = max(1, int(result.physics_steps))
        discount_to_end = self.config.discount_for(steps)
        # semi-Markov shaping: the potential difference is discounted to the end of the skill, so the
        # same progress is not paid twice (once inside, once at the boundary)
        reward = self.config.potential_weight * (discount_to_end * after - before)
        reward -= self.config.time_cost_per_step * steps
        self.skills_used += 1
        self.steps += steps
        self.boundaries.append({
            "skill_index": self.skills_used, "action_index": int(action_index),
            "action": action.label(), "active": list(action.active),
            "alpha_deg": [round(math.degrees(a), 3) for a in action.alpha],
            "speed": [float(v) for v in action.speed],
            "steps": steps, "exit_reason": str(result.exit_reason),
            "settled": bool(result.settled_observed),
            "push_completed": bool(result.push_completed),
            "object_before": self.privileged_object()[:2].tolist()})
        terminated = False
        success = False
        if self._goal_reached():
            self.hold += 1
            if self.hold >= self.config.hold_steps:
                terminated, success = True, True
                reward += self.config.success_bonus
        else:
            self.hold = 0
        truncated = False
        if self.skills_used >= self.config.max_skills:
            truncated = True
            reward += self.config.failure_penalty
        info = {"success": success, "steps": steps, "skills_used": self.skills_used,
                "object_pose": self.privileged_object().tolist(),
                "goal_distance": float(-after), "discount": discount_to_end,
                # the caller must be able to tell "the task's own budget ran out" from "the sampler
                # stopped early"; only the latter bootstraps
                "task_budget_expired": bool(self._task_budget_expired()),
                "goal_reached": bool(self._goal_reached()),
                # ``task_event`` is the only task-reward information the learner needs; the shaping
                # and the time cost are aggregated by the learner from the state trace so that the
                # success bonus can never be silently dropped
                "task_event": ("success" if success else
                               ("budget_exhausted" if (truncated and self._task_budget_expired())
                                else None)),
                "physics_steps": steps,
                "object_pose_after": self.privileged_object()[:2].tolist(),
                "elapsed_steps_before": int(self.steps - steps)}
        return StepResult(observation=self._observation(), reward=float(reward),
                          terminated=terminated, truncated=truncated, info=info,
                          inner_observations=self._inner_observations(
                              result.rows, float(self.steps - steps) * 0.05),
                          skill_steps=steps,
                          trace=self._privileged_trace(result.rows))

    def _privileged_trace(self, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for row in rows:
            true = row.get("true")
            if not true:
                continue
            out.append({"time": float(row.get("time", 0.0)),
                        "object": [float(v) for v in true["object"][:3]],
                        "robots": [[float(v) for v in robot[:3]] for robot in true["robots"]],
                        "skill_index": self.skills_used, "stage": str(row.get("stage", ""))})
        return out

    def _inner_observations(self, rows: Sequence[Dict[str, object]],
                            elapsed_offset_seconds: float = 0.0) -> np.ndarray:
        """The legal joint observation at every logged 0.05 s period of the executed skill.

        Built only from the ``observed`` block (each robot's own whitelist sensing) and the public
        clock; the ``true`` block in the same rows is privileged and is ignored here.
        """

        out: List[np.ndarray] = []
        for row in rows:
            observed = row.get("observed")
            if not observed:
                continue
            values: List[float] = []
            for robot_id in (0, 1):
                block = observed.get(str(robot_id), {})
                own = block.get("own") or [0.0] * 6
                values += [float(v) for v in own[:6]]
                # exactly the same five fields as ``_robot_block``; the previous version put the
                # elapsed time here, so column 28 meant "patch distance" at a boundary and "clock"
                # inside a skill
                values += [1.0 if block.get("patch_points") else 0.0,
                           float(block.get("patch_distance") or 0.0)]
                normal = block.get("patch_normal") or (0.0, 0.0)
                values += [float(normal[0]), float(normal[1])]
                values += [1.0 if block.get("contact") else 0.0]
            robot0 = observed.get("0", {}).get("own") or [0.0] * 6
            robot1 = observed.get("1", {}).get("own") or [0.0] * 6
            values += [robot1[0] - robot0[0], robot1[1] - robot0[1],
                       math.atan2(math.sin(robot1[2] - robot0[2]),
                                  math.cos(robot1[2] - robot0[2]))]
            values += [self.goal[0] - robot0[0], self.goal[1] - robot0[1],
                       float(np.hypot(self.goal[0] - robot0[0], self.goal[1] - robot0[1]))]
            # the inner block has the same width as the boundary block: 11 + 11 + 3 + 3 + 2
            values += [float((observed.get("0", {}) or {}).get("patch_distance") or 0.0), 0.0]
            # ``row['time']`` is the skill-local clock; the offset makes the stored clock the
            # episode's physical time, which is what the task book's time axis means
            values += self._clock_features(elapsed_offset_seconds
                                           + float(row.get("time", 0.0)))
            array = np.asarray(values, dtype=np.float32)
            if array.shape[0] != self.spec.width:
                raise RuntimeError(f"inner observation width {array.shape[0]} != {self.spec.width}")
            out.append(array)
        if not out:
            return np.zeros((0, self.spec.width), dtype=np.float32)
        return np.stack(out)


def default_goal(root: RootSpec, distance: float = 0.45, angle: Optional[float] = None) -> np.ndarray:
    """A public goal region: ``distance`` from the object's initial position.

    The direction defaults to the line from the robots' initial centroid **through** the object, so
    the goal lies on the side the existing skills are already set up to push towards. Both inputs are
    part of the published initial configuration, so the goal is public. Passing ``angle`` overrides
    the direction explicitly, which is how the validation scenarios vary the goal bearing.
    """

    pose = root.object_pose
    if angle is None:
        centroid = np.asarray([[pose.x, pose.y] for pose in root.robot_poses]).mean(axis=0)
        direction = np.asarray([pose.x - centroid[0], pose.y - centroid[1]], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.asarray([1.0, 0.0])
    else:
        direction = np.asarray([math.cos(angle), math.sin(angle)])
    return np.asarray([pose.x + distance * direction[0],
                       pose.y + distance * direction[1]], dtype=np.float64)


def calibration(roots: Sequence[RootSpec], samples: int = 24, seed: int = 0) -> Dict[str, object]:
    """How far does one skill actually move the object? Used to size the task distance."""

    generator = np.random.default_rng(seed)
    displacements = []
    for root in roots[:samples]:
        env = OptionRLEnv(OptionRLConfig(seed=seed))
        env.reset(root, default_goal(root, 0.5))
        for _ in range(6):
            # only actions that are legal for this root's active set: a joint skill cannot be
            # executed by a robot the scene does not place at the object
            candidates = [i for i, a in enumerate(MENU)
                          if set(a.active).issubset(set(root.active_robots))]
            action_index = int(candidates[generator.integers(0, len(candidates))])
            action = MENU[action_index]
            specs = {}
            for index, robot_id in enumerate(action.active):
                speed = float(action.speed[index])
                specs[robot_id] = LocalSurfacePush(
                    alpha=float(action.alpha[index]), speed=speed,
                    active_duration=PUSH_TARGET_ACTIVE, force_limit=FORCE_LIMIT,
                    delta=DELTA_PER_SPEED * speed)
            before = env.privileged_object()
            executor = OptionExecutor(env.option_config, env.arm)
            result = executor.run(env.env, root_id=root.root_id, variant="baseline",
                                  joint_specs=specs, active_robots=tuple(action.active))
            after = env.privileged_object()
            displacements.append(float(np.hypot(after[0] - before[0], after[1] - before[1])))
            env.skills_used += 1
    values = np.asarray(displacements)
    return {"samples": int(values.size), "median_m": float(np.median(values)),
            "p90_m": float(np.quantile(values, 0.9)), "max_m": float(values.max()),
            "mean_m": float(values.mean())}


def main() -> int:
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "outputs/night_repair_optionrl_v1/option_rl"))
    args = parser.parse_args()
    roots = build_roots("dev", 2)
    report = {"menu_size": len(MENU), "observation_width": ObservationSpec().width,
              "roots": len(roots)}
    if args.calibrate:
        report["displacement"] = calibration(roots, args.samples)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "option_env_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("OPTION_ENV_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
