"""The staged option executor and the four experimental arms (O, P, F, N).

The stage machine is decoupled from the control law: ``force_source`` selects the old oracle
formula or the new local law, ``protocol`` selects the old 2 s window, the equal-wall-clock
variant or the full staged protocol. Nothing here writes a body pose or velocity, and no stage
transition resets the engine's contact cache: the only inputs to the world are bounded forces
through :meth:`Continuous2DEnv.step_control`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.skills import Observation, PushContact, control as old_control
from .contract import (
    APPROACH_CONTACT_SPEED,
    DELTA_DEFAULT,
    APPROACH_MAX_SPEED,
    APPROACH_MIN_SPEED,
    APPROACH_TIMEOUT,
    BRAKE_SETTLE_LIMIT,
    CONTACT_LOSS_TOLERANCE,
    Exit,
    OptionConfig,
    PATCH_LOSS_TOLERANCE,
    READY_BOTH_HOLD,
    READY_CONTACT_HOLD,
    PUSH_TARGET_ACTIVE,
    PUSH_WALL_LIMIT,
    SETTLE_HOLD,
    Stage,
)
from .control import (
    LocalControlState,
    ReferenceVelocity,
    brake_force,
    brake_force_with_hold,
    clip_norm,
    local_push_force,
    update_settle_evidence,
    update_surface_estimate,
    wrap_angle,
)
from .delay import DelayFrame, FixedDelayQueue
from .position import PositionPushState, _to_world, position_push_force
from .sensing import (
    BodyKinematics,
    LocalContactObservation,
    TeammateMessage,
    best_target_patch,
    scan_surface,
)

Vec = Tuple[float, float]
ARM_ORDER = ("O", "P", "F", "N", "V", "X")
#: arms whose PUSH is driven by the surface-position law
POSITION_ARMS = ("X",)
#: arms whose PUSH is driven by the local velocity law
VELOCITY_ARMS = ("F", "N", "V")


@dataclass
class RobotRuntime:
    robot_id: int
    stage: Stage = Stage.APPROACH
    ready: bool = False
    contact: bool = False
    contact_hold: float = 0.0
    contact_loss: float = 0.0
    patch_loss: float = 0.0
    active_contact: float = 0.0
    push_wall: float = 0.0
    brake_time: float = 0.0
    settle_hold: float = 0.0
    approach_time: float = 0.0
    force: Vec = (0.0, 0.0)
    reference: Optional[Vec] = None
    patch_distance: Optional[float] = None
    patch_normal: Optional[Vec] = None
    patch_confidence: Optional[float] = None
    patch_points: int = 0
    surface_speed: Optional[float] = None
    #: round-3 settle evidence bookkeeping (see control.SettleEvidence)
    settle_evidence_span: float = 0.0
    settle_evidence_lag: float = 0.0
    settle_new_sample: bool = False
    normal_jump: bool = False
    ready_started: Optional[float] = None
    brake_initialised: bool = False
    push_interval_start: Optional[float] = None
    push_interval_end: Optional[float] = None
    control: LocalControlState = field(default_factory=LocalControlState)
    position: PositionPushState = field(default_factory=PositionPushState)
    position_active: bool = False
    skill_delta: float = 0.0
    skill_alpha: float = 0.0
    old_state: Dict[str, object] = field(default_factory=lambda: {"mode": "approach"})
    flags: List[str] = field(default_factory=list)
    exit: Optional[str] = None

    def note(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)


@dataclass
class ExecutionResult:
    arm: str
    root_id: str
    variant: str
    exit_reason: str
    wall_clock_steps: int
    rows: List[Dict[str, object]] = field(default_factory=list)
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    aborted_by: Optional[str] = None
    #: exact terminal state, captured at the moment of exit, independent of any sampled log
    terminal_object: Optional[Tuple[float, ...]] = None
    terminal_robots: Optional[Tuple[Tuple[float, ...], ...]] = None
    #: the three outcomes are kept apart instead of being folded into one "success" label
    push_completed: bool = False
    #: round-3 bookkeeping. ``primary_exit_reason`` is the FIRST abort cause (empty when the run
    #: ended naturally); ``locally_settled`` is whether the on-board quiet criterion ever fired,
    #: regardless of how the run ended. The two derived flags are then consistent by construction:
    #: ``settled_observed = locally_settled and not primary_exit_reason`` and
    #: ``settled_after_abort = locally_settled and bool(primary_exit_reason)``.
    primary_exit_reason: str = ""
    locally_settled: bool = False
    settled_observed: bool = False
    settled_after_abort: bool = False
    effective_push_seconds: float = 0.0
    #: round-4: the whitelist observation each robot had at the option's exit instant T. It is the
    #: sensing the *same* pipeline produced for that period; nothing is stepped or re-sensed after
    #: the loop exits, so this is the observation at T rather than at the last logged period.
    terminal_observations: Optional[Dict[int, object]] = None
    #: round-5: every period's observation set in execution order (``record_trajectory=True``).
    #: Index 0 is the observation taken **before** the first control force -- the true O0.
    trajectory: Optional[List[Dict[int, object]]] = None
    #: settle-evidence coverage at the moment of the verdict, and its lag behind the present
    settle_evidence_span: Optional[float] = None
    settle_evidence_lag: Optional[float] = None
    delay_stats: Optional[Dict[str, object]] = None
    push_intervals: Optional[Dict[str, object]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "arm": self.arm, "root_id": self.root_id, "variant": self.variant,
            "exit_reason": self.exit_reason, "wall_clock_steps": self.wall_clock_steps,
            "stage_seconds": self.stage_seconds, "flags": self.flags,
            "aborted_by": self.aborted_by, "push_completed": self.push_completed,
            "primary_exit_reason": self.primary_exit_reason,
            "locally_settled": self.locally_settled,
            "settled_observed": self.settled_observed,
            "settled_after_abort": self.settled_after_abort,
            "settle_evidence_span": self.settle_evidence_span,
            "settle_evidence_lag": self.settle_evidence_lag,
            "effective_push_seconds": self.effective_push_seconds,
            "delay_stats": self.delay_stats,
            "push_intervals": self.push_intervals,
            "terminal_object": list(self.terminal_object) if self.terminal_object else None,
            "terminal_robots": [list(row) for row in self.terminal_robots] if self.terminal_robots else None,
        }


def _body_kinematics(state) -> BodyKinematics:
    return BodyKinematics(state.x, state.y, state.angle, state.vx, state.vy, state.omega)


def _contact_bool(env: Continuous2DEnv, robot_id: int) -> Tuple[bool, Optional[Vec], Vec, float]:
    """Sensed contact flag, contact point in the robot frame, world force and impulse proxy."""

    observation = env.observation()
    point_robot: Optional[Vec] = None
    force_world: Vec = (0.0, 0.0)
    impulse = 0.0
    contacting = False
    for entry in env.backend.contacts():
        if entry["pair"] != "robot_object" or int(entry["robot_id"]) != robot_id:
            continue
        contacting = True
        point_world = (float(entry["point"][0]), float(entry["point"][1]))
        robot = observation.robots[robot_id]
        cos, sin = math.cos(-robot.angle), math.sin(-robot.angle)
        dx, dy = point_world[0] - robot.x, point_world[1] - robot.y
        point_robot = (dx * cos - dy * sin, dx * sin + dy * cos)
        impulse = max(impulse, max(0.0, -float(entry["distance"])))
    body = env.backend.robot_bodies[robot_id]
    force_world = (float(body.force.x), float(body.force.y))
    return contacting, point_robot, force_world, impulse


def build_observation(
    env: Continuous2DEnv,
    config: OptionConfig,
    robot_id: int,
    *,
    stage: str,
    teammate: Optional[TeammateMessage],
    skill: Dict[str, float],
    noise_rng=None,
) -> LocalContactObservation:
    """Assemble the whitelist observation from the sensing simulation."""

    observation = env.observation()
    scene = env.scene
    robots = tuple(state.as_tuple() for state in observation.robots)
    object_pose = observation.object_pose()
    patches = scan_surface(
        scene, object_pose, observation.object_size, robots, robot_id,
        scan_range=config.scan_range, scan_rays=config.scan_rays, timestamp=observation.time,
        min_points=config.min_points_for_normal,
        noise_sigma=config.range_noise_sigma if noise_rng is not None else 0.0,
        noise_rng=noise_rng,
    )
    patch = best_target_patch(patches)
    contacting, point_robot, force_world, impulse = _contact_bool(env, robot_id)
    own = observation.robots[robot_id]
    return LocalContactObservation(
        robot_id=robot_id,
        tick=observation.tick,
        time=observation.time,
        sample_time=observation.time,
        own=_body_kinematics(own),
        teammate=teammate,
        patch=patch,
        contacting=contacting,
        contact_point_robot=point_robot,
        contact_force_world=force_world,
        contact_impulse=impulse,
        arena=scene.arena,
        robot_radius=config_force_radius(env),
        force_limit=config.force_limit,
        skill=dict(skill),
        stage=stage,
    )


def config_force_radius(env: Continuous2DEnv) -> float:
    return env.config.physics.robot_radius


class OptionExecutor:
    """Runs one skill execution for one arm on one environment."""

    def __init__(self, config: OptionConfig, arm: str, *, log_every: int = 1):
        if arm not in ARM_ORDER:
            raise ValueError(f"unknown arm {arm!r}")
        self.config = config
        self.arm = arm
        self.log_every = log_every

    # -- public ------------------------------------------------------------
    def run(
        self,
        env: Continuous2DEnv,
        *,
        root_id: str,
        variant: str,
        joint_specs: Dict[int, object],
        active_robots: Sequence[int],
        noise_rng=None,
        record_trajectory: bool = False,
    ) -> ExecutionResult:
        if self.arm == "O":
            return self._run_old_window(env, root_id, variant, joint_specs, active_robots)
        if self.arm in POSITION_ARMS:
            force_source = "position"
        elif self.arm == "P":
            force_source = "old"
        else:
            force_source = "local"
        protocol = "equal_wall_clock" if self.arm == "F" else "staged"
        return self._run_staged(
            env, root_id, variant, joint_specs, active_robots,
            force_source=force_source, protocol=protocol, noise_rng=noise_rng,
            record_trajectory=record_trajectory,
        )

    # -- arm O: the old window, byte for byte -------------------------------
    def _run_old_window(self, env, root_id, variant, joint_specs, active_robots) -> ExecutionResult:
        from ..continuous2d.skills import Hold

        spec = tuple(
            joint_specs.get(robot_id) if joint_specs.get(robot_id) is not None else Hold()
            for robot_id in (0, 1)
        )
        result = env.execute_joint_skill(spec)  # type: ignore[arg-type]
        rows: List[Dict[str, object]] = []
        active_contact = 0.0
        dt = env.config.physics.control_dt
        for index, step in enumerate(result.steps):
            contacts = step.contacts
            touching = {
                str(robot_id): any(
                    str(point[0]) == "robot_object" and int(point[1]) == robot_id
                    for point in contacts
                )
                for robot_id in (0, 1)
            }
            if touching.get("0"):
                active_contact += dt
            rows.append({
                "period": index,
                "time": round(step.time, 6),
                "stage": "old_window",
                "force_source": "old",
                "protocol": "old_window",
                "force": [list(step.applied_control[0]), list(step.applied_control[1])],
                "true": {
                    "object": list(step.object.as_tuple()),
                    "robots": [list(step.robots[0].as_tuple()), list(step.robots[1].as_tuple())],
                },
                "observed": {
                    robot_id: {"contact": bool(touching[robot_id]), "patch_distance": None,
                               "patch_normal": None, "patch_points": 0, "patch_confidence": None,
                               "reference": None, "surface_speed": None,
                               "stage": "old_window"}
                    for robot_id in ("0", "1")
                },
                "penetration": float(step.penetration),
                "active_contact": {str(robot_id): active_contact if robot_id == 0 else 0.0
                                   for robot_id in (0, 1)},
                "push_wall": {str(robot_id): (index + 1) * dt for robot_id in (0, 1)},
                "termination": None,
                "labels": list(step.skill_labels),
            })
        return ExecutionResult(
            arm=self.arm, root_id=root_id, variant=variant,
            exit_reason=result.termination if result.termination != "window_complete" else "wall_clock_end",
            wall_clock_steps=len(result.steps), rows=rows,
            stage_seconds={"old_window": float(result.duration)},
            flags=["old_window"], aborted_by=None,
            # the old path also reports an explicit terminal snapshot: the acceptance check
            # requires a complete terminal state from every arm, including the untouched one
            terminal_object=env.backend.object_state().as_tuple(),
            terminal_robots=tuple(state.as_tuple() for state in env.backend.robot_states()),
            push_completed=False, settled_observed=False, settled_after_abort=False,
            effective_push_seconds=active_contact,
        )

    # -- arms P / F / N ------------------------------------------------------
    def _run_staged(
        self, env, root_id, variant, joint_specs, active_robots,
        *, force_source: str, protocol: str, noise_rng=None,
        record_trajectory: bool = False,
    ) -> ExecutionResult:
        config = self.config
        physics = env.config.physics
        control_config = env.config.control
        dt = physics.control_dt
        runtimes = {robot_id: RobotRuntime(robot_id=robot_id) for robot_id in (0, 1)}
        for robot_id in active_robots:
            runtime = runtimes[robot_id]
            skill = joint_specs[robot_id]
            runtime.control.start(
                alpha=float(getattr(skill, "alpha", 0.0)),
                speed=float(getattr(skill, "speed", 0.0)),
                force_limit=float(getattr(skill, "force_limit", config.force_limit)),
                kv=config.kv, a_ref_max=config.a_ref_max,
                approach_min_speed=config.approach_min_speed,
                approach_max_speed=config.approach_max_speed,
                approach_gain=config.approach_gain,
                approach_contact_speed=config.approach_contact_speed,
                ready_precontact_speed=config.ready_precontact_speed,
            )
            runtime.skill_delta = float(getattr(skill, "delta", DELTA_DEFAULT))
            runtime.skill_alpha = float(getattr(skill, "alpha", 0.0))
            runtime.note("active")
        for robot_id in (0, 1):
            if robot_id not in active_robots:
                runtimes[robot_id].note("passive")

        stage = Stage.APPROACH
        abort_exit: Optional[str] = None
        locally_settled = False
        both_ready_hold = 0.0
        stage_seconds: Dict[str, float] = {value.value: 0.0 for value in Stage}
        rows: List[Dict[str, object]] = []
        step = 0
        #: Round-4: the observation set of the period the loop is currently in. The stage machine
        #: can break inside that same iteration, so after the loop this holds the observations at
        #: the option's **exit instant T** -- the moment the recorded terminal state was captured.
        #: It is a plain reference to a dict that is rebuilt each period and never mutated later.
        terminal_observations: Dict[int, LocalContactObservation] = {}
        #: round-5: the full per-period observation set, in order, when the caller asks for it.
        #: Each entry is the (never-mutated) dict that period's controller actually consumed.
        trajectory: List[Dict[int, LocalContactObservation]] = []
        exit_reason = Exit.WALL_CLOCK_END.value
        aborted_by: Optional[str] = None
        delay_queues = {
            robot_id: FixedDelayQueue(config.observation_delay) for robot_id in (0, 1)
        }

        while True:
            now = step * dt
            if protocol == "equal_wall_clock" and now >= config.f_wall_clock:
                exit_reason = Exit.WALL_CLOCK_END.value
                break
            if now > 20.0:
                break

            # ---- sense ----------------------------------------------------
            fresh: Dict[int, LocalContactObservation] = {}
            messages: Dict[int, TeammateMessage] = {}
            for robot_id in (0, 1):
                other = 1 - robot_id
                teammate = messages.get(other)
                if teammate is None and other in active_robots:
                    other_state = env.backend.robot_states()[other]
                    teammate = TeammateMessage(
                        robot_id=other, kinematics=_body_kinematics(other_state),
                        stage=runtimes[other].stage.value, ready=runtimes[other].ready,
                        contact=runtimes[other].contact, normal_world=runtimes[other].patch_normal,
                        reference_velocity_world=runtimes[other].reference,
                    )
                fresh[robot_id] = build_observation(
                    env, config, robot_id, stage=stage.value, teammate=teammate,
                    skill=skill_parameters(joint_specs.get(robot_id)), noise_rng=noise_rng,
                )
                messages[robot_id] = TeammateMessage(
                    robot_id=robot_id, kinematics=fresh[robot_id].own, stage=stage.value,
                    ready=runtimes[robot_id].ready, contact=fresh[robot_id].contacting,
                    normal_world=fresh[robot_id].unit_normal_world(),
                    reference_velocity_world=runtimes[robot_id].reference,
                )
            observations: Dict[int, LocalContactObservation] = {}
            observation_missing: Dict[int, bool] = {}
            for robot_id in (0, 1):
                queue = delay_queues[robot_id]
                queue.push(DelayFrame(
                    sample_time=fresh[robot_id].sample_time, arrival_time=now,
                    observation=fresh[robot_id], message=messages[robot_id],
                ))
                frame = queue.deliver(now)
                if frame is None:
                    observation_missing[robot_id] = True
                else:
                    observations[robot_id] = frame.observation
            if observation_missing:
                # the stage machine does not advance on information that has not arrived: the
                # world keeps stepping under a zero command and the period is logged as such
                env.step_control(((0.0, 0.0), (0.0, 0.0)))
                rows.append(self._unavailable_row(
                    env, step, stage, sorted(observation_missing), dt, force_source, protocol))
                step += 1
                if env.termination is not None:
                    exit_reason = Exit.ENVIRONMENT_TERMINATION.value
                    aborted_by = f"environment:{env.termination}"
                    break
                continue

            # the exit instant's observation set (see the declaration above)
            terminal_observations = observations
            if record_trajectory:
                trajectory.append(observations)

            # ---- per-robot bookkeeping ------------------------------------
            for robot_id in (0, 1):
                runtime = runtimes[robot_id]
                observation = observations[robot_id]
                runtime.contact = observation.contacting
                runtime.patch_distance = observation.patch.distance if observation.patch else None
                runtime.patch_normal = observation.unit_normal_world()
                runtime.patch_confidence = observation.patch.confidence if observation.patch else None
                runtime.patch_points = len(observation.patch.points_robot) if observation.patch else 0
                if observation.contacting:
                    runtime.contact_hold += dt
                    runtime.contact_loss = 0.0
                else:
                    runtime.contact_hold = 0.0
                    runtime.contact_loss += dt
                if observation.patch is None:
                    runtime.patch_loss += dt
                else:
                    runtime.patch_loss = 0.0

            # ---- stage machine --------------------------------------------
            # Per-robot stages: one robot can already be READY and holding contact at the low
            # pre-contact reference while its teammate is still in APPROACH. PUSH starts only
            # when every active robot has been READY with continuous contact for the frozen
            # both-ready hold.
            if protocol == "equal_wall_clock":
                for robot_id in active_robots:
                    runtime = runtimes[robot_id]
                    runtime.approach_time += dt
                    if runtime.stage == Stage.APPROACH:
                        if runtime.contact_hold >= READY_CONTACT_HOLD:
                            runtime.stage = Stage.PUSH
                            runtime.ready = True
                            runtime.push_interval_start = step * dt
                        elif runtime.approach_time > APPROACH_TIMEOUT:
                            runtime.exit = Exit.APPROACH_TIMEOUT.value
                    elif runtime.stage == Stage.PUSH:
                        if runtime.contact:
                            runtime.active_contact += dt
                        runtime.push_wall += dt
                if any(runtimes[robot_id].exit for robot_id in active_robots):
                    exit_reason = next(runtimes[robot_id].exit for robot_id in active_robots
                                       if runtimes[robot_id].exit)
                    break
                stage = Stage.PUSH if all(runtimes[robot_id].stage == Stage.PUSH
                                          for robot_id in active_robots) else Stage.APPROACH
            elif stage in (Stage.APPROACH, Stage.READY):
                # every active robot walks APPROACH -> READY on its own contact evidence
                for robot_id in active_robots:
                    runtime = runtimes[robot_id]
                    runtime.approach_time += dt
                    if runtime.patch_loss > config.patch_loss_tolerance:
                        runtime.note("surface_unobserved")
                        runtime.exit = Exit.SURFACE_UNOBSERVED.value
                        continue
                    if runtime.stage == Stage.APPROACH:
                        if runtime.contact_hold >= READY_CONTACT_HOLD:
                            runtime.stage = Stage.READY
                            runtime.ready = True
                            runtime.ready_started = step * dt
                    elif runtime.stage == Stage.READY:
                        if not runtime.contact:
                            # losing contact drops READY and re-enters contact establishment
                            runtime.stage = Stage.APPROACH
                            runtime.ready = False
                            runtime.ready_started = None
                            runtime.note("ready_dropped")
                    if (runtime.approach_time > APPROACH_TIMEOUT
                            and not any(runtimes[other].stage == Stage.PUSH
                                        for other in active_robots)):
                        runtime.exit = Exit.APPROACH_TIMEOUT.value
                failed = [robot_id for robot_id in active_robots if runtimes[robot_id].exit]
                if failed:
                    exit_reason = runtimes[failed[0]].exit  # type: ignore[assignment]
                    aborted_by = f"robot{failed[0]}"
                    for robot_id in active_robots:
                        if robot_id != failed[0]:
                            runtimes[robot_id].note("teammate_abort")
                    stage = Stage.BRAKE
                    abort_exit = exit_reason
                    for robot_id in active_robots:
                        self._end_position_push(runtimes[robot_id])
                        self._enter_brake(runtimes[robot_id], observations[robot_id], config)
                        runtimes[robot_id].stage = Stage.BRAKE
                        runtimes[robot_id].push_interval_end = step * dt
                else:
                    all_ready = all(runtimes[robot_id].stage == Stage.READY
                                    and runtimes[robot_id].contact_hold >= READY_CONTACT_HOLD
                                    for robot_id in active_robots)
                    both_ready_hold = both_ready_hold + dt if all_ready else 0.0
                    if both_ready_hold >= READY_BOTH_HOLD:
                        stage = Stage.PUSH
                        for robot_id in active_robots:
                            runtimes[robot_id].stage = Stage.PUSH
                            runtimes[robot_id].push_interval_start = step * dt
                            self._start_position_push(runtimes[robot_id],
                                                      joint_specs.get(robot_id))
            elif stage == Stage.PUSH:
                for robot_id in active_robots:
                    runtime = runtimes[robot_id]
                    runtime.stage = Stage.PUSH
                    runtime.push_wall += dt
                trigger = push_abort_trigger(runtimes, active_robots, config)
                if trigger:
                    aborted_by = f"robot{trigger[0]}"
                    exit_reason = trigger[1]
                    abort_exit = trigger[1]
                    for robot_id in active_robots:
                        if robot_id != trigger[0]:
                            runtimes[robot_id].note("teammate_abort")
                    stage = Stage.BRAKE
                    for robot_id in active_robots:
                        self._end_position_push(runtimes[robot_id])
                        self._enter_brake(runtimes[robot_id], observations[robot_id], config)
                        runtimes[robot_id].stage = Stage.BRAKE
                elif all(runtimes[robot_id].active_contact >= PUSH_TARGET_ACTIVE
                         for robot_id in active_robots):
                    stage = Stage.BRAKE
                    for robot_id in active_robots:
                        self._end_position_push(runtimes[robot_id])
                        self._enter_brake(runtimes[robot_id], observations[robot_id], config)
                        runtimes[robot_id].stage = Stage.BRAKE
                else:
                    for robot_id in active_robots:
                        if not runtimes[robot_id].contact:
                            runtimes[robot_id].note("contact_paused")
            elif stage == Stage.BRAKE:
                finished = True
                for robot_id in active_robots:
                    runtime = runtimes[robot_id]
                    runtime.stage = Stage.BRAKE
                    runtime.brake_time += dt
                    if runtime.control.reference is None:
                        runtime.control.reference = ReferenceVelocity(a_ref_max=config.a_ref_max)
                    # the ramp is applied exactly once per period, inside the force call; the
                    # reference logged here is the value that call produced
                    if runtime.reference is None or math.hypot(*runtime.reference) > 1e-9:
                        finished = False
                if finished:
                    stage = Stage.SETTLE
                    for robot_id in active_robots:
                        runtimes[robot_id].stage = Stage.SETTLE
                        runtimes[robot_id].settle_hold = 0.0
                if any(runtimes[robot_id].brake_time > BRAKE_SETTLE_LIMIT
                       for robot_id in active_robots):
                    exit_reason = abort_exit or Exit.SETTLE_TIMEOUT.value
                    stage = Stage.DONE
                    break
            elif stage == Stage.SETTLE:
                unobserved = False
                for robot_id in active_robots:
                    runtime = runtimes[robot_id]
                    runtime.stage = Stage.SETTLE
                    runtime.brake_time += dt
                    observation = observations[robot_id]
                    own_speed = math.hypot(observation.own.vx, observation.own.vy)
                    omega = abs(observation.own.omega)
                    if config.settle_window > 0.0:
                        evidence = update_settle_evidence(
                            runtime.control, observation.patch,
                            (observation.own.x, observation.own.y, observation.own.angle),
                            observation.sample_time, now,
                            window=config.settle_window,
                            min_baseline=config.settle_window_min_baseline,
                            max_sample_gap=config.settle_max_sample_gap,
                        )
                        speed = evidence.speed
                        signed = evidence.signed
                        rms = evidence.rms
                        matches = evidence.matches
                        reason = evidence.reason
                        runtime.settle_evidence_span = evidence.span
                        runtime.settle_evidence_lag = evidence.lag
                        runtime.settle_new_sample = evidence.new_sample
                    else:
                        speed, signed, rms, matches, reason = update_surface_estimate(
                            runtime.control, observation.patch,
                            (observation.own.x, observation.own.y, observation.own.angle),
                            observation.sample_time, now,
                        )
                        runtime.settle_evidence_span = 0.0
                        runtime.settle_evidence_lag = 0.0
                        runtime.settle_new_sample = True
                    runtime.surface_speed = speed
                    runtime.control.surface_speed = speed
                    runtime.control.surface_speed_signed = signed
                    runtime.control.surface_speed_rms = rms
                    runtime.control.surface_matches = matches
                    runtime.control.surface_unavailable_reason = reason
                    if speed is None:
                        # a genuinely missing/stale/unmatchable observation is not evidence of
                        # stillness under any reading, so it clears the hold
                        unobserved = True
                        runtime.note(f"settle_unobserved:{reason}")
                        runtime.settle_hold = 0.0
                        continue
                    quiet = (own_speed < config.settle_linear and omega < config.settle_angular
                             and speed < config.settle_surface)
                    if not quiet:
                        runtime.settle_hold = 0.0
                    elif not runtime.settle_new_sample:
                        # a repeated frame is one piece of evidence: it neither accrues stillness
                        # time (that would double-count the same observation) nor clears what has
                        # already been accumulated
                        runtime.note("settle_repeated_frame_frozen")
                        continue
                    else:
                        runtime.settle_hold += dt
                if all(runtimes[robot_id].settle_hold >= SETTLE_HOLD for robot_id in active_robots):
                    locally_settled = True
                    exit_reason = abort_exit or Exit.SETTLED_OBSERVED.value
                    if abort_exit:
                        for robot_id in active_robots:
                            runtimes[robot_id].note("settled_after_abort")
                    stage = Stage.DONE
                    break
                if any(runtimes[robot_id].brake_time > BRAKE_SETTLE_LIMIT
                       for robot_id in active_robots):
                    exit_reason = abort_exit or (Exit.SETTLE_UNOBSERVED.value if unobserved
                                                 else Exit.SETTLE_TIMEOUT.value)
                    stage = Stage.DONE
                    break
            stage_seconds[stage.value] = stage_seconds.get(stage.value, 0.0) + dt

            # ---- forces ---------------------------------------------------
            joint_force: List[Vec] = [(0.0, 0.0), (0.0, 0.0)]
            for robot_id in (0, 1):
                runtimes[robot_id]._dt = dt
            for robot_id in (0, 1):
                if robot_id in active_robots:
                    continue
                # a robot outside this skill's active set still has to come to rest; it is not
                # allowed to sit at zero force, which would let it coast into the object
                runtime = runtimes[robot_id]
                runtime.stage = stage
                joint_force[robot_id] = brake_force(observations[robot_id], runtime.control, dt)
                runtime.reference = runtime.control.reference.value
                runtime.force = joint_force[robot_id]
            for robot_id in active_robots:
                runtime = runtimes[robot_id]
                if protocol == "equal_wall_clock":
                    mode = "push" if runtime.stage == Stage.PUSH else "approach"
                else:
                    mode = {
                        Stage.APPROACH: "approach", Stage.READY: "ready",
                        Stage.PUSH: "push", Stage.BRAKE: "brake", Stage.SETTLE: "brake",
                        Stage.DONE: "brake",
                    }.get(stage, "approach")
                if mode == "brake":
                    joint_force[robot_id] = brake_force_with_hold(
                        observations[robot_id], runtime.control, dt, config.brake_hold_force)
                    runtime.reference = runtime.control.reference.value
                elif force_source == "position":
                    joint_force[robot_id] = self._position_force(
                        observations[robot_id], runtime, dt, mode, config_force_radius(env))
                elif force_source == "old":
                    joint_force[robot_id] = self._old_force(
                        env, joint_specs[robot_id], robot_id, runtime, control_config, physics, mode
                    )
                else:
                    joint_force[robot_id] = self._local_force(observations[robot_id], runtime, dt, mode)
                runtime.force = joint_force[robot_id]

            # effective push is credited for exactly the period in which the push force is
            # applied and the required robots are in contact; the interval bounds are logged
            if stage == Stage.PUSH and all(
                runtimes[robot_id].contact for robot_id in active_robots
            ) and not abort_exit:
                for robot_id in active_robots:
                    runtimes[robot_id].active_contact += dt

            env.step_control((joint_force[0], joint_force[1]))
            if env.termination is not None:
                exit_reason = Exit.ENVIRONMENT_TERMINATION.value
                aborted_by = f"environment:{env.termination}"
                break

            if step % self.log_every == 0:
                rows.append(self._row(env, step, stage, runtimes, active_robots, observations,
                                      joint_specs, force_source, dt, protocol))
            step += 1

        terminal_object = env.backend.object_state().as_tuple()
        terminal_robots = tuple(state.as_tuple() for state in env.backend.robot_states())
        for robot_id in active_robots:
            runtimes[robot_id].push_interval_end = step * dt
        effective = min(
            (runtimes[robot_id].active_contact for robot_id in active_robots), default=0.0
        )
        push_completed = bool(effective >= PUSH_TARGET_ACTIVE - 1e-9)
        # round-3 bookkeeping. ``primary_exit_reason`` is the FIRST abort cause and is what the run
        # is recorded under; ``locally_settled`` records whether the quiet criterion fired at all.
        # Keeping both makes the two derived flags reachable and consistent by construction, so
        # "stalled and then genuinely came to rest" can never masquerade as a normal completion.
        primary_exit_reason = abort_exit or ""
        settled_observed = bool(locally_settled and not primary_exit_reason)
        settled_after_abort = bool(locally_settled and primary_exit_reason)
        settle_span = max((runtimes[robot_id].settle_evidence_span for robot_id in active_robots),
                          default=0.0)
        settle_lag = max((runtimes[robot_id].settle_evidence_lag for robot_id in active_robots),
                         default=0.0)
        return ExecutionResult(
            arm=self.arm, root_id=root_id, variant=variant, exit_reason=exit_reason,
            wall_clock_steps=step, rows=rows, stage_seconds=stage_seconds,
            flags=sorted({flag for runtime in runtimes.values() for flag in runtime.flags}),
            aborted_by=aborted_by,
            terminal_object=terminal_object, terminal_robots=terminal_robots,
            push_completed=push_completed,
            primary_exit_reason=primary_exit_reason,
            locally_settled=locally_settled,
            settled_observed=settled_observed,
            settled_after_abort=settled_after_abort,
            effective_push_seconds=effective,
            terminal_observations=dict(terminal_observations) if terminal_observations else None,
            trajectory=trajectory if record_trajectory else None,
            settle_evidence_span=settle_span if locally_settled else None,
            settle_evidence_lag=settle_lag if locally_settled else None,
            delay_stats={str(robot_id): delay_queues[robot_id].stats() for robot_id in (0, 1)},
            push_intervals={
                str(robot_id): {
                    "start": runtimes[robot_id].push_interval_start,
                    "end": runtimes[robot_id].push_interval_end,
                    "contact_source": "sensed robot_object contact flag",
                    "effective_seconds": runtimes[robot_id].active_contact,
                }
                for robot_id in (0, 1)
            },
        )

    # -- force sources -------------------------------------------------------
    def _old_force(self, env, skill, robot_id, runtime, control_config, physics, mode) -> Vec:
        observation: Observation = env.observation()
        if mode == "push":
            direction = skill.push_direction_world
            length = math.hypot(direction[0], direction[1]) or 1.0
            speed = control_config.push_speed
            target = (direction[0] / length * speed, direction[1] / length * speed)
            if runtime.control.reference is None:
                runtime.control.reference = ReferenceVelocity(a_ref_max=self.config.a_ref_max)
            runtime.control.reference.update(target, self.dt_for(runtime))
        runtime.old_state["mode"] = "push" if mode == "push" else "approach"
        force, state = old_control(
            skill, observation, robot_id, control_config, physics, state=runtime.old_state,
        )
        runtime.old_state = state
        runtime.old_state["mode"] = "push" if mode == "push" else "approach"
        if runtime.control.reference is not None:
            runtime.reference = runtime.control.reference.value
        return force

    def dt_for(self, runtime) -> float:
        return getattr(runtime, "_dt", 0.05)

    # -- arm X: surface-position feedback ------------------------------------
    def _start_position_push(self, runtime: RobotRuntime, skill) -> None:
        """Called once, on PUSH entry: freeze the skill's delta/alpha and restart beta."""

        runtime.skill_delta = float(getattr(skill, "delta", DELTA_DEFAULT))
        runtime.skill_alpha = float(getattr(skill, "alpha", 0.0))
        runtime.position.start_push(delta=runtime.skill_delta, alpha=runtime.skill_alpha)
        runtime.position_active = True

    def _end_position_push(self, runtime: RobotRuntime) -> None:
        """Called when PUSH is left: withdraw the virtual advance so SETTLE does not keep
        pressing the object through a stale, invasive reference."""

        if runtime.position_active:
            runtime.position.end_push()
        runtime.position_active = False

    def _position_force(self, observation, runtime: RobotRuntime, dt: float, mode: str,
                        robot_radius: float) -> Vec:
        runtime._dt = dt
        if mode == "brake":
            # BRAKE/SETTLE are the same fixed baseline implementation for both arms
            force = brake_force(observation, runtime.control, dt)
            runtime.reference = runtime.control.reference.value
            return force
        if mode != "push":
            # APPROACH and READY are the same fixed baseline implementation for both arms:
            # only the PUSH force differs, which is what isolates the comparison
            force, diagnostics = local_push_force(
                observation, runtime.control, dt, mode=mode)
            runtime.reference = runtime.control.reference.value
            runtime.normal_jump = bool(diagnostics.get("normal_jump"))
            if diagnostics.get("reason") == "surface_unobserved":
                runtime.note("surface_unobserved")
            return force
        if not runtime.position_active:
            runtime.position.start_push(delta=runtime.skill_delta, alpha=runtime.skill_alpha)
            runtime.position_active = True
        force = position_push_force(
            observation, runtime.position, dt,
            robot_radius=robot_radius,
            margin=self.config.patch_support_margin,
            kp=self.config.kp, kd=self.config.kd, e_max=self.config.e_max,
            beta_ramp=self.config.beta_ramp, force_limit=self.config.force_limit,
        )
        if runtime.position.unavailable:
            runtime.note(runtime.position.unavailable)
        return force

    @staticmethod
    def _enter_brake(runtime: RobotRuntime, observation, config: OptionConfig) -> None:
        """Initialise the brake ramp at the moment BRAKE is entered, not one period later.

        The reference is seeded from the velocity this robot actually has at entry. Doing this
        in the BRAKE stage instead would let the first brake force be computed from the cruise
        reference and then jump on the following period, which is exactly the discontinuity the
        acceleration bound is supposed to exclude.
        """

        if runtime.brake_initialised:
            return
        if runtime.control.reference is None:
            runtime.control.reference = ReferenceVelocity(a_ref_max=config.a_ref_max)
        measured = observation.own
        runtime.control.reference.reset((measured.vx, measured.vy))
        runtime.reference = runtime.control.reference.value
        runtime.brake_initialised = True

    def _local_force(self, observation, runtime, dt, mode) -> Vec:
        runtime._dt = dt
        if mode == "brake":
            force = brake_force(observation, runtime.control, dt)
            runtime.reference = runtime.control.reference.value
            return force
        force, diagnostics = local_push_force(observation, runtime.control, dt, mode=mode)
        # the reference is logged every period, not only while braking, so the animation can draw
        # the real reference velocity instead of a placeholder
        runtime.reference = runtime.control.reference.value
        runtime.normal_jump = bool(diagnostics.get("normal_jump"))
        runtime.surface_speed = runtime.control.surface_speed
        if diagnostics.get("reason") == "surface_unobserved":
            runtime.note("surface_unobserved")
        return force

    # -- logging -------------------------------------------------------------
    def _row(self, env, step, stage, runtimes, active_robots, observations, joint_specs,
             force_source, dt, protocol) -> Dict[str, object]:
        object_state = env.backend.object_state()
        robots = env.backend.robot_states()
        return {
            "period": step,
            "time": round(step * dt, 6),
            "stage": stage.value if isinstance(stage, Stage) else str(stage),
            # READY is a per-robot state, so the decomposition the report needs is derived from
            # the robot stages: PUSH when every active robot pushes, READY when at least one is
            # holding contact at the pre-contact reference, APPROACH otherwise
            "phase": joint_phase(stage, runtimes, active_robots),
            "force_source": force_source,
            "protocol": protocol,
            "force": [list(runtimes[robot_id].force) for robot_id in (0, 1)],
            "true": {
                "object": list(object_state.as_tuple()),
                "robots": [list(state.as_tuple()) for state in robots],
            },
            "observed": {
                str(robot_id): {
                    "contact": bool(observations[robot_id].contacting),
                    "patch_distance": observations[robot_id].patch.distance
                    if observations[robot_id].patch else None,
                    "patch_normal": list(observations[robot_id].patch.normal_world)
                    if observations[robot_id].patch else None,
                    "patch_points": len(observations[robot_id].patch.points_robot)
                    if observations[robot_id].patch else 0,
                    "patch_confidence": observations[robot_id].patch.confidence
                    if observations[robot_id].patch else None,
                    "contact_point_robot": list(observations[robot_id].contact_point_robot)
                    if observations[robot_id].contact_point_robot else None,
                    "contact_impulse": float(observations[robot_id].contact_impulse or 0.0),
                    "reference": list(runtimes[robot_id].reference)
                    if runtimes[robot_id].reference else None,
                    "surface_speed": runtimes[robot_id].surface_speed,
                    "surface_speed_signed": runtimes[robot_id].control.surface_speed_signed,
                    "surface_speed_rms": runtimes[robot_id].control.surface_speed_rms,
                    # round-3 evidence provenance: which sample span backs the current verdict,
                    # and how far that evidence lags the present control period
                    "settle_evidence_span": runtimes[robot_id].settle_evidence_span,
                    "settle_evidence_lag": runtimes[robot_id].settle_evidence_lag,
                    "settle_new_sample": runtimes[robot_id].settle_new_sample,
                    "surface_unavailable_reason":
                        runtimes[robot_id].control.surface_unavailable_reason,
                    "stage": runtimes[robot_id].stage.value,
                    # --- arm X surface-position diagnostics (None for the other arms) ---
                    "beta": runtimes[robot_id].position.beta,
                    "signed_distance": runtimes[robot_id].position.signed_distance,
                    "gap_signed": runtimes[robot_id].position.gap_signed,
                    "q": list(runtimes[robot_id].position.q) if runtimes[robot_id].position.q else None,
                    "p_contact": list(runtimes[robot_id].position.p_contact)
                    if runtimes[robot_id].position.p_contact else None,
                    "p_ref": list(runtimes[robot_id].position.p_ref)
                    if runtimes[robot_id].position.p_ref else None,
                    "error": list(runtimes[robot_id].position.error),
                    "error_bounded": list(runtimes[robot_id].position.error_bounded),
                    "force_raw": list(runtimes[robot_id].position.force_raw),
                    "support": runtimes[robot_id].position.support,
                    "support_loss": runtimes[robot_id].position.support_loss,
                    "tangential_offset": runtimes[robot_id].position.tangential_offset,
                    "position_active": runtimes[robot_id].position_active,
                    "segment_world": [
                        list(_to_world(point, (observations[robot_id].own.x,
                                               observations[robot_id].own.y,
                                               observations[robot_id].own.angle)))
                        for point in (observations[robot_id].patch.segment_robot
                                      if observations[robot_id].patch else ())
                    ] or None,
                    # the kinematics the commanded force was computed from, i.e. the state
                    # *before* this period's step; the true{} block is the state after it
                    "own": list(observations[robot_id].own.__dict__.values())
                    if hasattr(observations[robot_id].own, "__dict__") else None,
                }
                for robot_id in (0, 1)
            },
            "penetration": float(env.backend.max_penetration()),
            "active_contact": {str(robot_id): runtimes[robot_id].active_contact for robot_id in (0, 1)},
            "push_wall": {str(robot_id): runtimes[robot_id].push_wall for robot_id in (0, 1)},
            "termination": env.termination,
        }


# ---------------------------------------------------------------------------
# small helpers used by the stage machine
# ---------------------------------------------------------------------------


    def _unavailable_row(self, env, step, stage, missing, dt, force_source, protocol) -> Dict[str, object]:
        """A control period in which the delayed observation had not arrived yet."""

        object_state = env.backend.object_state()
        robots = env.backend.robot_states()
        return {
            "period": step,
            "time": round(step * dt, 6),
            "stage": stage.value if isinstance(stage, Stage) else str(stage),
            "force_source": force_source,
            "protocol": protocol,
            "force": [[0.0, 0.0], [0.0, 0.0]],
            "true": {"object": list(object_state.as_tuple()),
                     "robots": [list(state.as_tuple()) for state in robots]},
            "observed": {},
            "observation_unavailable_for": list(missing),
            "penetration": float(env.backend.max_penetration()),
            "active_contact": {"0": 0.0, "1": 0.0},
            "push_wall": {"0": 0.0, "1": 0.0},
            "termination": env.termination,
        }


def joint_phase(stage, runtimes, active_robots) -> str:
    """The report's approach / ready / push / brake / settle phase, derived from robot stages."""

    if isinstance(stage, Stage) and stage in (Stage.BRAKE, Stage.SETTLE, Stage.DONE):
        return stage.value
    stages = [runtimes[robot_id].stage for robot_id in active_robots]
    if stages and all(value == Stage.PUSH for value in stages):
        return "push"
    if any(value == Stage.READY for value in stages):
        return "ready"
    return "approach"


def skill_parameters(skill) -> Dict[str, float]:
    if skill is None:
        return {}
    out: Dict[str, float] = {}
    for name in ("alpha", "speed", "active_duration", "force_limit"):
        if hasattr(skill, name):
            out[name] = float(getattr(skill, name))
    if hasattr(skill, "push_direction_world"):
        direction = getattr(skill, "push_direction_world")
        out["direction_x"] = float(direction[0])
        out["direction_y"] = float(direction[1])
    return out


def push_abort_trigger(
    runtimes, active_robots, config: OptionConfig
) -> Optional[Tuple[int, str]]:
    """First abort condition reached during PUSH, in the frozen priority order.

    ``face_switch`` (two consecutive normals more than 30 degrees apart) is checked before a
    patch loss, a contact loss and the push wall-clock cap, so a corner is never silently
    absorbed into a longer push.
    """

    for robot_id in active_robots:
        runtime = runtimes[robot_id]
        if runtime.normal_jump:
            return (robot_id, Exit.FACE_SWITCH.value)
        # arm X may not push against an extrapolated infinite plane: if the foot of the
        # perpendicular has left the observed segment for longer than the existing loss
        # tolerance, the shared protocol ends the push
        if runtime.position_active and runtime.position.support_loss > config.patch_loss_tolerance:
            return (robot_id, Exit.PATCH_SUPPORT_LOST.value)
        if runtime.patch_loss > config.patch_loss_tolerance:
            return (robot_id, Exit.PATCH_LOST.value)
        if runtime.contact_loss > config.contact_loss_tolerance:
            return (robot_id, Exit.CONTACT_LOST.value)
        if runtime.push_wall > getattr(config, "push_wall_limit", PUSH_WALL_LIMIT):
            return (robot_id, Exit.PUSH_WALL_TIMEOUT.value)
    return None


def stage_started_active(runtime: RobotRuntime, active_robots, runtimes, dt: float) -> bool:
    """Effective contact counts only while every active robot is touching (double push)."""

    if len(active_robots) == 1:
        return runtime.contact
    return all(runtimes[robot_id].contact for robot_id in active_robots)


def brake_finished(runtime: RobotRuntime, observation, config: OptionConfig, dt: float) -> bool:
    raw = (0.0, 0.0)
    if runtime.control.reference is None:
        return True
    reference = runtime.control.reference.update(raw, dt)
    speed = math.hypot(reference[0], reference[1])
    runtime.reference = reference
    return speed <= 1e-6
