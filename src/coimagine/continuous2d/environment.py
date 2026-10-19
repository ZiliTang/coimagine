"""The continuous-2D environment: the C2 interface and the synchronous joint-skill protocol.

Interface (names kept exactly as the contract asks for)::

    reset(scene_config, seed) -> observation
    export_state() -> replayable_snapshot
    restore(snapshot) -> None
    step_control(joint_control) -> observation, diagnostics
    execute_joint_skill(joint_spec) -> SkillResult
    rollout(snapshot, joint_specs, backend) -> TrajectoryResult

One ``joint_spec`` holds one local skill per robot; both run synchronously for a fixed
window. A robot that finishes early switches to an explicit ``hold`` and the window does
not end early. Real success or a numerical failure may end the window.

The same ``execute_joint_skill`` drives both real execution and the planner's simulated
copies, so prediction and execution cannot drift apart in protocol.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .backend import BodyState, PymunkBackend, Snapshot
from .config import CONTROLLER_VERSION, Continuous2DConfig, default_config
from .scenes import SceneSpec, scene_by_name
from .skills import (
    ControllerState,
    Hold,
    JointSpec,
    MoveTo,
    Observation,
    PushContact,
    Skill,
    control,
    is_done,
)

Vec = Tuple[float, float]

TERMINATION_SUCCESS = "success"
TERMINATION_WINDOW = "window_complete"
TERMINATION_NUMERIC = "numeric_failure"
TERMINATION_OUT_OF_BOUNDS = "out_of_bounds"
TERMINATION_PENETRATION = "penetration_limit"


@dataclass(frozen=True)
class StepRecord:
    tick: int
    time: float
    #: State *before* this control step and the state after it, so the pair is well defined.
    before_robots: Tuple[BodyState, BodyState]
    before_object: BodyState
    robots: Tuple[BodyState, BodyState]
    object: BodyState
    applied_control: Tuple[Vec, Vec]
    skill_labels: Tuple[str, str]
    penetration: float
    penetration_by_pair: Dict[str, float]
    contacts: Tuple[Tuple[object, ...], ...]


@dataclass
class SkillResult:
    joint_spec: JointSpec
    initial: Snapshot
    final: Snapshot
    duration: float
    termination: str
    success: bool
    steps: Tuple[StepRecord, ...]
    controller_version: str = CONTROLLER_VERSION
    diagnostics: Dict[str, object] = field(default_factory=dict)
    cost_components: Dict[str, float] = field(default_factory=dict)

    def to_record(self, episode_id: str, scene_id: str) -> Dict[str, object]:
        return {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "t": self.initial.tick,
            "joint_skill_specs": [self.joint_spec[0].label, self.joint_spec[1].label],
            "initial_state": self.initial.as_key(),
            "final_state": self.final.as_key(),
            "duration": self.duration,
            "termination": self.termination,
            "success": int(self.success),
            "cost_components": dict(self.cost_components),
            "physics_config_hash": self.initial.config_hash,
            "controller_version": self.controller_version,
            "steps": [
                {
                    "tick": step.tick,
                    "time": step.time,
                    "state": {
                        "robots": [s.as_tuple() for s in step.before_robots],
                        "object": step.before_object.as_tuple(),
                    },
                    "next_state": {
                        "robots": [s.as_tuple() for s in step.robots],
                        "object": step.object.as_tuple(),
                    },
                    "applied_control": [list(step.applied_control[0]), list(step.applied_control[1])],
                    "skill_labels": list(step.skill_labels),
                    "penetration": step.penetration,
                    "penetration_by_pair": dict(step.penetration_by_pair),
                    "contacts": [list(point) for point in step.contacts],
                }
                for step in self.steps
            ],
        }


@dataclass
class TrajectoryResult:
    initial: Snapshot
    final: Snapshot
    results: Tuple[SkillResult, ...]
    success: bool
    termination: str
    cost_components: Dict[str, float]


class Continuous2DEnv:
    """One world. ``rollout`` restores into ``self``, so planners use a scratch instance."""

    def __init__(self, config: Optional[Continuous2DConfig] = None):
        self.config = config or default_config()
        self.scene: Optional[SceneSpec] = None
        self.backend: Optional[PymunkBackend] = None
        self.tick = 0
        self.controller_state: List[ControllerState] = [{}, {}]
        self.success_counter = 0.0
        self.termination: Optional[str] = None
        #: Per-quantity deviation of the last exact prefix replay (diagnostic only).
        self.last_restore_deviation: Dict[str, float] = {}

    # -- reset / state -----------------------------------------------------
    def reset(self, scene: Union[str, SceneSpec], seed: int = 0) -> Observation:
        self.scene = scene_by_name(scene) if isinstance(scene, str) else scene
        self.backend = PymunkBackend(
            self.scene, self.config.physics, self.config.config_hash()
        )
        self.tick = 0
        self.controller_state = [{}, {}]
        self.success_counter = 0.0
        self.termination = None
        return self.observation()

    def export_state(self) -> Snapshot:
        return self.backend.export_state(
            controller_state={"controllers": copy.deepcopy(self.controller_state)},
            success_counter=self.success_counter,
            termination=self.termination,
        )

    def restore(self, snapshot: Snapshot, *, exact: bool = True) -> None:
        """Restore the world and the task-level state.

        ``exact=True`` (the default and the only mode the experiment uses) rebuilds the
        initial scene and replays the applied-control prefix, verifying the physics state
        against the snapshot. ``exact=False`` is the development-only approximate clone
        that discards the engine's cached contact state. Neither mode clears the success
        hold timer or the termination state; only :meth:`reset` initialises those.
        """

        if snapshot.physics != self.config.physics:
            raise ValueError(
                "snapshot physics configuration is incompatible with this environment; "
                "refusing to reinterpret an old snapshot under another configuration"
            )
        if snapshot.config_hash != self.config.config_hash():
            raise ValueError(
                "snapshot config hash "
                f"{snapshot.config_hash!r} != environment {self.config.config_hash()!r}"
            )
        self.scene = snapshot.scene_config
        self.backend = PymunkBackend(self.scene, self.config.physics, self.config.config_hash())
        if exact:
            self.last_restore_deviation = self.backend.restore_exact(snapshot)
        else:
            self.last_restore_deviation = {}
            self.backend.approximate_restore(snapshot)
        self.tick = snapshot.tick
        controllers = snapshot.controller_state.get("controllers", [{}, {}])
        self.controller_state = copy.deepcopy([dict(controllers[0]), dict(controllers[1])])
        self.success_counter = float(snapshot.success_counter)
        self.termination = snapshot.termination

    # -- observation -------------------------------------------------------
    def observation(self) -> Observation:
        contacts = tuple(
            (
                str(entry["pair"]),
                int(entry["robot_id"]) if entry["robot_id"] is not None else -1,
                float(entry["point"][0]),
                float(entry["point"][1]),
            )
            for entry in self.backend.contacts()
        )
        return Observation(
            tick=self.tick,
            time=self.tick * self.config.physics.control_dt,
            robots=self.backend.robot_states(),
            object=self.backend.object_state(),
            goal=self.scene.goal,
            arena=self.scene.arena,
            obstacles=self.scene.obstacles,
            window_progress=0.0,
            controller_state=(dict(self.controller_state[0]), dict(self.controller_state[1])),
            config_hash=self.config.config_hash(),
            contact_points=contacts,
            object_size=(self.config.physics.object_width, self.config.physics.object_height),
        )

    # -- stepping ----------------------------------------------------------
    def step_control(self, joint_control: Tuple[Vec, Vec],
                     headings=None) -> Tuple[Observation, Dict[str, object]]:
        if self.termination is not None:
            # A terminated episode never advances: explicit zero-increment terminal result.
            return self.observation(), {
                "penetration": self.backend.max_penetration(),
                "penetration_by_pair": self.backend.penetration_by_pair(),
                "finite": self.backend.finite(),
                "in_bounds": self.backend.in_bounds(),
                "contacts": self.backend.contacts(),
                "success_hold": self.success_counter,
                "already_terminated": True,
                "termination": self.termination,
            }
        self.backend.apply_and_step(joint_control, headings)
        self.tick += 1
        diagnostics: Dict[str, object] = {
            "penetration": self.backend.max_penetration(),
            "penetration_by_pair": self.backend.penetration_by_pair(),
            "finite": self.backend.finite(),
            "in_bounds": self.backend.in_bounds(),
            "contacts": self.backend.contacts(),
            "already_terminated": False,
        }
        if self.object_in_goal() and self.object_slow():
            self.success_counter += self.config.physics.control_dt
        else:
            self.success_counter = 0.0
        diagnostics["success_hold"] = self.success_counter
        return self.observation(), diagnostics

    def object_in_goal(self) -> bool:
        """All four object corners inside the goal rectangle (not a single-frame centre test)."""

        goal = self.scene.goal
        margin = self.config.physics.goal_margin
        for x, y in self.backend.object_corners():
            if abs(x - goal.cx) > goal.half_w - margin:
                return False
            if abs(y - goal.cy) > goal.half_h - margin:
                return False
        return True

    def object_slow(self) -> bool:
        object_state = self.backend.object_state()
        physics = self.config.physics
        return (
            math.hypot(object_state.vx, object_state.vy) <= physics.velocity_tolerance
            and abs(object_state.omega) <= physics.angular_tolerance
        )

    def task_success(self) -> bool:
        return self.success_counter >= self.config.physics.success_hold_time

    # -- joint skill execution --------------------------------------------
    def execute_joint_skill(self, joint_spec: JointSpec) -> SkillResult:
        physics = self.config.physics
        control_config = self.config.control
        if self.termination is not None:
            # Explicit zero-increment terminal result: no physics, no cost, no candidate
            # execution after the episode is over.
            initial = self.export_state()
            return SkillResult(
                joint_spec=joint_spec,
                initial=initial,
                final=initial,
                duration=0.0,
                termination=self.termination,
                success=self.termination == TERMINATION_SUCCESS,
                steps=(),
                diagnostics={"already_terminated": True, "termination": self.termination},
                cost_components={"time": 0.0, "steps": 0.0, "force_usage": 0.0},
            )
        initial = self.export_state()
        active: List[Skill] = list(joint_spec)
        self.controller_state = [{}, {}]
        steps: List[StepRecord] = []
        force_usage = 0.0
        termination = TERMINATION_WINDOW
        success = False

        for _ in range(physics.window_steps):
            observation = self.observation()
            before_robots = observation.robots
            before_object = observation.object
            forces: List[Vec] = []
            for robot_id in (0, 1):
                if is_done(active[robot_id], observation, robot_id, control_config):
                    # Explicit hold; the window is not ended early.
                    active[robot_id] = Hold()
                force, state = control(
                    active[robot_id],
                    observation,
                    robot_id,
                    control_config,
                    physics,
                    state=self.controller_state[robot_id],
                )
                self.controller_state[robot_id] = state
                forces.append(force)
            joint_control = (forces[0], forces[1])
            observation, diagnostics = self.step_control(joint_control)
            force_usage += sum(math.hypot(*force) for force in joint_control) * physics.control_dt
            steps.append(
                StepRecord(
                    tick=self.tick,
                    time=self.tick * physics.control_dt,
                    before_robots=before_robots,
                    before_object=before_object,
                    robots=observation.robots,
                    object=observation.object,
                    applied_control=joint_control,
                    skill_labels=(active[0].label, active[1].label),
                    penetration=float(diagnostics["penetration"]),
                    penetration_by_pair=dict(diagnostics["penetration_by_pair"]),
                    contacts=observation.contact_points,
                )
            )
            if not diagnostics["finite"]:
                termination = TERMINATION_NUMERIC
                break
            if not diagnostics["in_bounds"]:
                termination = TERMINATION_OUT_OF_BOUNDS
                break
            if float(diagnostics["penetration"]) > physics.penetration_limit:
                termination = TERMINATION_PENETRATION
                break
            if self.task_success():
                termination = TERMINATION_SUCCESS
                success = True
                break

        if termination in (
            TERMINATION_SUCCESS,
            TERMINATION_NUMERIC,
            TERMINATION_OUT_OF_BOUNDS,
            TERMINATION_PENETRATION,
        ):
            # Terminal state is part of the task state and is snapshotted.
            self.termination = termination

        final = self.export_state()
        duration = (final.tick - initial.tick) * physics.control_dt
        return SkillResult(
            joint_spec=joint_spec,
            initial=initial,
            final=final,
            duration=duration,
            termination=termination,
            success=success,
            steps=tuple(steps),
            diagnostics={
                "object_in_goal": self.object_in_goal(),
                "object_slow": self.object_slow(),
                "success_hold": self.success_counter,
                "final_penetration": self.backend.max_penetration(),
                "final_penetration_by_pair": self.backend.penetration_by_pair(),
                "already_terminated": False,
            },
            cost_components={
                "time": duration,
                "steps": float(len(steps)),
                "force_usage": force_usage,
            },
        )

    # -- rollout -----------------------------------------------------------
    def rollout(
        self,
        snapshot: Snapshot,
        joint_specs: Sequence[JointSpec],
        backend: str = "simulator",
    ) -> TrajectoryResult:
        if backend != "simulator":
            raise NotImplementedError(
                "This round has the true simulator and its exact planning adapter only; "
                f"backend={backend!r} has no implementation."
            )
        self.restore(snapshot, exact=True)
        results: List[SkillResult] = []
        total_cost = 0.0
        force_usage = 0.0
        success = False
        termination = TERMINATION_WINDOW
        for joint_spec in joint_specs:
            result = self.execute_joint_skill(joint_spec)
            results.append(result)
            total_cost += result.cost_components["time"]
            force_usage += result.cost_components["force_usage"]
            termination = result.termination
            if result.termination in (
                TERMINATION_SUCCESS,
                TERMINATION_NUMERIC,
                TERMINATION_OUT_OF_BOUNDS,
                TERMINATION_PENETRATION,
            ):
                success = result.success
                break
        return TrajectoryResult(
            initial=snapshot,
            final=self.export_state(),
            results=tuple(results),
            success=success,
            termination=termination,
            cost_components={
                "time": total_cost,
                "force_usage": force_usage,
                "windows": float(len(results)),
                "steps": float(sum(len(r.steps) for r in results)),
            },
        )

    # -- geometry helpers --------------------------------------------------
    def terminal_distance(self) -> float:
        """Largest overhang of an object corner outside the goal rectangle (0 if inside).

        Corner-based on purpose: this is the same geometry as the success test, so a state
        the metric calls zero distance is exactly a state ``object_in_goal`` accepts. A
        centre-only distance would score a rotated object as "at the goal" while success
        still failed, and the planner then preferred to hold forever.
        """

        from .scoring import distance_to_box

        return distance_to_box(self.backend.object_corners(), self.scene.goal)

    def passage_distance(self) -> float:
        """Geometric cost-to-go through the passage (planner failure branch only)."""

        from .scoring import passage_cost_to_go

        object_state = self.backend.object_state()
        return passage_cost_to_go(
            (object_state.x, object_state.y, object_state.angle),
            (self.config.physics.object_width, self.config.physics.object_height),
            self.scene.goal,
            self.scene.obstacles,
            self.backend.object_corners(),
        )
