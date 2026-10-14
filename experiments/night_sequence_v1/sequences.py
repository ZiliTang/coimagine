"""Round-5: one real option execution is one sequence, stored with O0 and the true exit instant T.

Task book section 3. Design notes that matter for correctness:

* The executor senses, decides and only then applies force. ``ExecutionResult.trajectory[0]`` is
  therefore the observation taken **before** the first control force, at the shared initial state --
  the true ``O0``. It is not "the first log after the action ran for one period".
* ``trajectory[-1]`` is the period the loop exited in, so its timestamp is the real exit instant
  ``T``. The stage machine breaks before appending that period's log row, which is why the old
  ``_exit_block`` was one period early.
* Every array is columnar and padded to ``MAX_STEPS``; ``steps`` says how much is real. Padding is
  never valid data and never enters a mask.
* Ground-truth object/robot state is written into a **separate** group so the input whitelist can
  reject it. It is used for auditing and for the trajectory figures only.

Nothing here steps extra physics or adds a control call: the sequence is exactly the execution.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.local_contact_option.contract import DELTA_PER_SPEED  # noqa: E402
from coimagine.local_contact_option.option import OptionExecutor  # noqa: E402
from coimagine.local_contact_option.roots import VARIANT_SPECS, apply_initial_velocities  # noqa: E402
from coimagine.local_contact_option.runner import build_snapshot  # noqa: E402
from coimagine.local_contact_option.skill import LocalSurfacePush  # noqa: E402

#: hard cap on stored periods; the frozen protocol cannot exceed it, and we assert that
MAX_STEPS = 256
BASELINE_VARIANT = next(spec for spec in VARIANT_SPECS if spec.name == "baseline")

STAGE_CODES = {"approach": 0, "ready": 1, "push": 2, "brake": 3, "settle": 4, "done": 5,
               "hold": 6, "idle": 7}
#: the full frozen ``Exit`` enum, in a fixed order. The first twelve are the round-4 class list; the
#: remaining three exist in the controller and must not be collapsed into "unknown" (a missing exit
#: reason would silently drop a real failure category).
EXIT_CODES = [
    "settled_observed", "contact_lost", "patch_lost", "settle_unobserved", "push_wall_timeout",
    "settle_timeout", "approach_timeout", "patch_support_lost", "surface_unobserved",
    "not_ready", "environment_termination", "wall_clock_end",
    "push_complete", "face_switch", "teammate_abort",
]

#: columnar float arrays stored per sequence, with their per-period shape after (steps,)
FLOAT_FIELDS = {
    "time": (),
    "own_world": (2, 6),            # x, y, angle, vx, vy, omega  (robot's own estimate)
    "teammate_world": (2, 6),       # the same six for the teammate, as shared
    "patch_distance": (2,),
    "patch_confidence": (2,),
    "patch_segment_robot": (2, 2, 2),
    "patch_normal_world": (2, 2),
    "patch_midpoint_robot": (2, 2),
    "contact_point_robot": (2, 2),
    "reference_robot": (2, 2),
    "force": (2, 2),
}
INT_FIELDS = {
    "stage_code": (),
    "patch_valid": (2,),
    "patch_points": (2,),
    "contacting": (2,),
    "contact_point_valid": (2,),
    "reference_valid": (2,),
}
#: offline truth, kept apart from anything a model may read
TRUTH_FLOAT_FIELDS = {
    "object_world": (3,),
    "robots_world": (2, 3),
}

#: the observation **at the exit instant T**. The protocol senses once per period at the period
#: start, so the last stored period was sensed at T - dt. Task book 3.1 asks for T itself, and
#: permits one sensing call after the exit: no physics is stepped and no control is applied, so the
#: environment is still exactly at T and this is a real observation of the real terminal state.
TERMINAL_FLOAT_FIELDS = {
    "terminal_own_world": (2, 6),
    "terminal_patch_distance": (2,),
    "terminal_patch_confidence": (2,),
    "terminal_patch_segment_robot": (2, 2, 2),
    "terminal_patch_normal_world": (2, 2),
    "terminal_patch_midpoint_robot": (2, 2),
    "terminal_teammate_world": (2, 6),
}
TERMINAL_INT_FIELDS = {
    "terminal_patch_valid": (2,),
    "terminal_patch_points": (2,),
    "terminal_contacting": (2,),
}


def frozen_option_config():
    from experiments.option_prediction_mechanism_v1.phase_b.collect_b import frozen_option_config as cfg
    return cfg()


def specs_from_command(command: Dict[str, object]) -> Tuple[Dict[int, object], Tuple[int, ...]]:
    """Rebuild the joint skill command from its recorded public payload."""

    alpha = command["alpha_rad"]
    speed = command["speed"]
    active = tuple(command["active_robots"])
    specs: Dict[int, object] = {}
    for robot_id in (0, 1):
        if robot_id not in active:
            specs[robot_id] = None
            continue
        specs[robot_id] = LocalSurfacePush(
            alpha=float(alpha[robot_id]), speed=float(speed[robot_id]),
            active_duration=float(command["active_duration"]),
            force_limit=float(command["force_limit"]),
            delta=DELTA_PER_SPEED * float(speed[robot_id]))
    return specs, active


def blank_sequence(steps_capacity: int = MAX_STEPS) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, shape in FLOAT_FIELDS.items():
        out[name] = np.zeros((steps_capacity, *shape), dtype=np.float32)
    for name, shape in INT_FIELDS.items():
        out[name] = np.zeros((steps_capacity, *shape), dtype=np.int16)
    for name, shape in TRUTH_FLOAT_FIELDS.items():
        out[name] = np.zeros((steps_capacity, *shape), dtype=np.float32)
    for name, shape in TERMINAL_FLOAT_FIELDS.items():
        out[name] = np.zeros(shape, dtype=np.float32)
    for name, shape in TERMINAL_INT_FIELDS.items():
        out[name] = np.zeros(shape, dtype=np.int16)
    return out


def _sense_at_exit(env, option_config, stage: str, teammate_cache, skill_cache) -> List[object]:
    """One sensing call per robot at the exit state. No stepping, no control, no physics change."""

    from coimagine.local_contact_option.option import build_observation

    out = []
    for robot_id in (0, 1):
        out.append(build_observation(
            env, option_config, robot_id, stage=stage, teammate=teammate_cache.get(robot_id),
            skill=skill_cache.get(robot_id), noise_rng=None))
    return out


def _fill_terminal(arrays: Dict[str, np.ndarray], observations) -> None:
    for robot_id in (0, 1):
        observation = observations[robot_id]
        own = observation.own
        arrays["terminal_own_world"][robot_id] = (own.x, own.y, own.angle,
                                                  own.vx, own.vy, own.omega)
        teammate = observation.teammate
        if teammate is not None and teammate.kinematics is not None:
            kinematics = teammate.kinematics
            arrays["terminal_teammate_world"][robot_id] = (
                kinematics.x, kinematics.y, kinematics.angle,
                kinematics.vx, kinematics.vy, kinematics.omega)
        patch = observation.patch
        if patch is not None and len(patch.points_robot) > 0:
            arrays["terminal_patch_valid"][robot_id] = 1
            arrays["terminal_patch_points"][robot_id] = len(patch.points_robot)
            arrays["terminal_patch_distance"][robot_id] = float(patch.distance)
            arrays["terminal_patch_confidence"][robot_id] = float(patch.confidence)
            arrays["terminal_patch_segment_robot"][robot_id] = np.asarray(
                patch.segment_robot, dtype=np.float32)
            arrays["terminal_patch_normal_world"][robot_id] = np.asarray(
                patch.normal_world, dtype=np.float32)
            arrays["terminal_patch_midpoint_robot"][robot_id] = np.asarray(
                patch.points_robot, dtype=np.float32).mean(axis=0)
        arrays["terminal_contacting"][robot_id] = 1 if observation.contacting else 0


def _fill_period(arrays: Dict[str, np.ndarray], index: int, observations, time_value: float,
                 stage_code: int, forces: Sequence[Sequence[float]]) -> None:
    arrays["time"][index] = float(time_value)
    arrays["stage_code"][index] = int(stage_code)
    for robot_id in (0, 1):
        observation = observations[robot_id]
        own = observation.own
        arrays["own_world"][index, robot_id] = (own.x, own.y, own.angle,
                                                own.vx, own.vy, own.omega)
        teammate = observation.teammate
        if teammate is not None and teammate.kinematics is not None:
            kinematics = teammate.kinematics
            arrays["teammate_world"][index, robot_id] = (
                kinematics.x, kinematics.y, kinematics.angle,
                kinematics.vx, kinematics.vy, kinematics.omega)
        patch = observation.patch
        if patch is not None and len(patch.points_robot) > 0:
            arrays["patch_valid"][index, robot_id] = 1
            arrays["patch_points"][index, robot_id] = len(patch.points_robot)
            arrays["patch_distance"][index, robot_id] = float(patch.distance)
            arrays["patch_confidence"][index, robot_id] = float(patch.confidence)
            arrays["patch_segment_robot"][index, robot_id] = np.asarray(
                patch.segment_robot, dtype=np.float32)
            arrays["patch_normal_world"][index, robot_id] = np.asarray(
                patch.normal_world, dtype=np.float32)
            points = np.asarray(patch.points_robot, dtype=np.float32)
            arrays["patch_midpoint_robot"][index, robot_id] = points.mean(axis=0)
        arrays["contacting"][index, robot_id] = 1 if observation.contacting else 0
        if observation.contact_point_robot is not None:
            arrays["contact_point_robot"][index, robot_id] = np.asarray(
                observation.contact_point_robot, dtype=np.float32)
            arrays["contact_point_valid"][index, robot_id] = 1
        # ``skill`` is the public parameter dict from ``skill_parameters``; the controller's own
        # reference velocity is not part of it, so it is recorded only when a caller supplies it
        skill = observation.skill
        reference = None
        if isinstance(skill, dict):
            reference = skill.get("reference_velocity")
        elif skill is not None and hasattr(skill, "reference_velocity"):
            reference = skill.reference_velocity
        if reference is not None:
            arrays["reference_robot"][index, robot_id] = np.asarray(reference, dtype=np.float32)
            arrays["reference_valid"][index, robot_id] = 1
        arrays["force"][index, robot_id] = np.asarray(forces[robot_id], dtype=np.float32)


def run_sequence(root, command: Dict[str, object], *, record_truth: bool = True) -> Optional[dict]:
    """Execute one option and return its columnar sequence. No extra physics, no extra control."""

    option = frozen_option_config()
    base = default_config()
    snapshot, config, _object, _robots, _report, velocities = build_snapshot(
        root, BASELINE_VARIANT, base)
    env = Continuous2DEnv(config)
    env.restore(snapshot, exact=True)
    apply_initial_velocities(env, velocities)
    specs, active = specs_from_command(command)
    executor = OptionExecutor(option, "V")
    result = executor.run(env, root_id=root.root_id, variant=BASELINE_VARIANT.name,
                          joint_specs=specs, active_robots=active, noise_rng=None,
                          record_trajectory=True)
    trajectory = result.trajectory or []
    steps = len(trajectory)
    if steps == 0 or steps > MAX_STEPS:
        return None
    arrays = blank_sequence()
    rows = result.rows
    for index, observations in enumerate(trajectory):
        time_value = float(observations[0].time)
        stage_code = STAGE_CODES.get(str(observations[0].stage).lower(), 8)
        forces = rows[index]["force"] if index < len(rows) else [[0.0, 0.0], [0.0, 0.0]]
        _fill_period(arrays, index, observations, time_value, stage_code, forces)
    # one sensing call at the true exit instant T (no step, no control)
    last = trajectory[-1]
    stage = str(last[0].stage)
    teammate_cache = {robot_id: last[robot_id].teammate for robot_id in (0, 1)}
    skill_cache = {robot_id: last[robot_id].skill for robot_id in (0, 1)}
    terminal_at_T = _sense_at_exit(env, option, stage, teammate_cache, skill_cache)
    _fill_terminal(arrays, terminal_at_T)
    if record_truth:
        for index, row in enumerate(rows):
            truth = row["true"]
            arrays["object_world"][index] = np.asarray(truth["object"][:3], dtype=np.float32)
            arrays["robots_world"][index] = np.asarray(
                [state[:3] for state in truth["robots"]], dtype=np.float32)
    return {
        "arrays": arrays,
        "steps": steps,
        "root_id": root.root_id,
        "exit_class": result.exit_reason,
        "primary_exit_reason": result.primary_exit_reason,
        "locally_settled": bool(result.locally_settled),
        "settled_observed": bool(result.settled_observed),
        "settled_after_abort": bool(result.settled_after_abort),
        #: round-6 fix: the option's real elapsed physical time. The stage machine can break before
        #: the force call, so a period may be sensed without the world advancing; counting periods
        #: (which include O0) over-counted the duration by exactly one control period.
        "duration_seconds": float(result.physics_steps) * float(config.physics.control_dt),
        #: the physical exit instant = physics steps actually taken x dt
        "T_exit": float(result.physics_steps) * float(config.physics.control_dt),
        #: the timestamp of the last sensed period inside the option
        "T_last_sensed": float(trajectory[-1][0].time),
        #: how many log rows exist: exactly the number of stepped periods
        "truth_steps": len(rows),
        #: invariance check, asserted per record by the collector
        "physics_steps_match_rows": bool(result.physics_steps == len(rows)),
        "terminal_at_T_patch_valid": [int(arrays["terminal_patch_valid"][r]) for r in (0, 1)],
        "wall_clock_steps": int(result.wall_clock_steps),
        "terminal_object": list(result.terminal_object),
        "terminal_robots": [list(state) for state in result.terminal_robots],
        "command": command,
        "active_robots": list(active),
    }


@dataclass
class ShardWriter:
    """Write shards of sequences atomically, with a manifest row per shard."""

    directory: Path
    split: str
    shard_size: int = 32

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer: List[dict] = []
        self.shards: List[dict] = []
        self.index = 0

    def add(self, record: dict) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.directory / f"{self.split}_{self.index:05d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        payload: Dict[str, np.ndarray] = {}
        names = (list(FLOAT_FIELDS) + list(INT_FIELDS) + list(TRUTH_FLOAT_FIELDS)
                 + list(TERMINAL_FLOAT_FIELDS) + list(TERMINAL_INT_FIELDS))
        for name in names:
            payload[name] = np.stack([item["arrays"][name] for item in self.buffer])
        payload["steps"] = np.asarray([item["steps"] for item in self.buffer], dtype=np.int32)
        payload["wall_clock_steps"] = np.asarray(
            [item["wall_clock_steps"] for item in self.buffer], dtype=np.int32)
        payload["exit_code"] = np.asarray(
            [EXIT_CODES.index(item["exit_class"]) if item["exit_class"] in EXIT_CODES else -1
             for item in self.buffer], dtype=np.int16)
        payload["duration_seconds"] = np.asarray(
            [item["duration_seconds"] for item in self.buffer], dtype=np.float32)
        payload["T_exit"] = np.asarray(
            [item["T_exit"] for item in self.buffer], dtype=np.float32)
        payload["T_last_sensed"] = np.asarray(
            [item["T_last_sensed"] for item in self.buffer], dtype=np.float32)
        payload["truth_steps"] = np.asarray(
            [item["truth_steps"] for item in self.buffer], dtype=np.int32)
        payload["physics_steps_match_rows"] = np.asarray(
            [1 if item["physics_steps_match_rows"] else 0 for item in self.buffer], dtype=np.int8)
        payload["terminal_object"] = np.asarray(
            [item["terminal_object"] for item in self.buffer], dtype=np.float32)
        payload["terminal_robots"] = np.asarray(
            [item["terminal_robots"] for item in self.buffer], dtype=np.float32)
        payload["command_alpha"] = np.asarray(
            [[-99.0 if item["command"]["alpha_rad"][r] is None else item["command"]["alpha_rad"][r]
              for r in (0, 1)] for item in self.buffer], dtype=np.float32)
        payload["command_speed"] = np.asarray(
            [[-99.0 if item["command"]["speed"][r] is None else item["command"]["speed"][r]
              for r in (0, 1)] for item in self.buffer], dtype=np.float32)
        payload["command_active"] = np.asarray(
            [[1 if r in item["active_robots"] else 0 for r in (0, 1)] for item in self.buffer],
            dtype=np.int16)
        payload["command_active_duration"] = np.asarray(
            [item["command"]["active_duration"] for item in self.buffer], dtype=np.float32)
        payload["command_force_limit"] = np.asarray(
            [item["command"]["force_limit"] for item in self.buffer], dtype=np.float32)
        payload["root_id"] = np.asarray([item["root_id"] for item in self.buffer], dtype=object)
        payload["record_id"] = np.asarray(
            [f"{item['root_id']}#cmd{item['command']['index']}" for item in self.buffer],
            dtype=object)
        payload["command_index"] = np.asarray(
            [item["command"]["index"] for item in self.buffer], dtype=np.int16)
        payload["settled_observed"] = np.asarray(
            [1 if item["settled_observed"] else 0 for item in self.buffer], dtype=np.int16)
        payload["settled_after_abort"] = np.asarray(
            [1 if item["settled_after_abort"] else 0 for item in self.buffer], dtype=np.int16)
        # numpy appends ".npz" when handed a bare path, so write through a file object and keep
        # the atomic rename
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
        self.shards.append({
            "path": str(path.relative_to(ROOT)), "split": self.split,
            "sequences": len(self.buffer), "bytes": path.stat().st_size,
            "steps_total": int(sum(item["steps"] for item in self.buffer)),
        })
        self.buffer = []
        self.index += 1


def sequence_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
