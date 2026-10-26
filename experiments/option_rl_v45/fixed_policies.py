"""Fixed-policy return comparison for the v4.5 option wrapper.

The question this answers is *not* "can PPO learn" but "does the reward contract contain a
positive-return path for pushing at all".  Six hand-written policies are run on the same
frozen test roots, with the same cost accounting as the trainer:

* ``wait2``        both robots always request ``WAIT_BRAKE[2 s]``;
* ``wait24``       both robots always request ``WAIT_BRAKE[24 s]``;
* ``random_push``  uniform over the 12 PUSH options (WAIT fallback when no patch is visible);
* ``random_legal`` uniform over every publicly legal option (the trainer's action mask);
* ``goal_push``    legal face-selection heuristic: push when the touched face is the best
                   available push direction, otherwise orbit towards a better face;
* ``goal_push_priv`` diagnostic only: reads the object pose and pushes along the local axis
                   with the larger remaining goal error.  This is the "is a competent push
                   rewarded?" reference; its privileged access is never used by the learner.

The ``old`` wrapper is also reproduced for ``wait2`` only: it stops silently at 24 skills and
does not charge the task-budget failure.  That is the accounting bug the fix removes; seeing
both numbers side by side is the point of the comparison.

All deployment-shaped policies read only ``boundary_observations`` and the public action
menu.  ``goal_push_priv`` is labelled privileged and only exists to bound what a competent
pusher can earn.  Outputs go to ``outputs/option_rl_v45/fixed_policies.json``.

Usage:
    .venv/bin/python experiments/option_rl_v45/fixed_policies.py --roots 16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coimagine.random_start import env as E, layout, spec  # noqa: E402

CONTROL_DT = spec.CONTROL_DT
MAX_SKILLS = 120          # same loop guard as the trainer
TRUNCATION_PENALTY = spec.FAILURE_PENALTY
OLD_MAX_SKILLS = 24       # the removed early-stop behaviour

ACTIONS = spec.ROBOT_ACTIONS

WAIT2 = next(a.index for a in ACTIONS if a.kind == "wait" and a.scale == 2.0)
WAIT24 = next(a.index for a in ACTIONS if a.kind == "wait" and a.scale == 24.0)
PUSH_INDICES = tuple(a.index for a in ACTIONS if a.kind == "push")
KINDS = tuple(a.kind for a in ACTIONS)
LEVELS = spec.PUSH_DISTANCES  # (0.10, 0.25, 0.50, 1.00)
ROBOT_GAP = spec.REPOSITION_GAP


def legal_mask(observation) -> np.ndarray:
    """Public feasibility: push/reposition need a visible patch; wait/follow always fit."""
    mask = np.asarray([kind in ("wait", "follow") for kind in KINDS], dtype=bool)
    if observation is not None and getattr(observation, "patch", None) is not None:
        mask[:] = True
    return mask


# ---------------------------------------------------------------------------
# fixed policies: (env, rng) -> (action_0, action_1)
# ---------------------------------------------------------------------------


def wait2(env, rng) -> Tuple[int, int]:
    return WAIT2, WAIT2


def wait24(env, rng) -> Tuple[int, int]:
    return WAIT24, WAIT24


def random_push(env, rng) -> Tuple[int, int]:
    observations = env.boundary_observations()
    chosen = []
    for observation in observations:
        if getattr(observation, "patch", None) is None:
            chosen.append(WAIT2)
        else:
            chosen.append(int(rng.choice(PUSH_INDICES)))
    return chosen[0], chosen[1]


def random_legal(env, rng) -> Tuple[int, int]:
    observations = env.boundary_observations()
    chosen = []
    for observation in observations:
        indices = np.flatnonzero(legal_mask(observation))
        chosen.append(int(rng.choice(indices)))
    return chosen[0], chosen[1]


def _rotate_quarter(vector: Tuple[float, float], quarters: int) -> Tuple[float, float]:
    """Rotate a 2-D vector by ``quarters`` x 90 deg (CCW positive)."""
    x, y = vector
    for _ in range(abs(quarters)):
        x, y = (-y, x) if quarters > 0 else (y, -x)
    return x, y


def _goal_action(observation) -> int:
    """One legal goal-directed action from a single robot's own observation.

    ``normal_outward`` points from the object surface towards the robot (``normal_world``
    points the other way, into the object).  Pushing moves the object along
    ``-normal_outward``.  The four box faces are the current outward normal rotated by
    multiples of 90 deg, so the heuristic picks the face whose inward push best points at
    the goal: push when that face is the one being touched, otherwise orbit towards it.
    Small requested pushes near the goal keep the lateral error recoverable.
    """
    patch = getattr(observation, "patch", None)
    if patch is None or not patch.valid:
        return WAIT2
    normal = patch.normal_outward
    gx, gy = observation.goal_xy
    dx, dy = gx - observation.own.x, gy - observation.own.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return WAIT2
    ux, uy = dx / distance, dy / distance
    candidates = []
    for quarters in (0, 1, -1, 2):
        rx, ry = _rotate_quarter(normal, quarters)
        candidates.append((rx * ux + ry * uy, quarters))
    score, quarters = min(candidates, key=lambda item: item[0])
    if quarters == 0 and score < -0.35:
        # the touched face is the best available push direction: push along its normal
        remaining = max(LEVELS[0], distance - 0.30)
        scale_index = next((i for i, level in enumerate(LEVELS) if level >= remaining),
                           len(LEVELS) - 1)
        return 2 * len(LEVELS) + scale_index      # PUSH type 2 = fastest normal push
    if quarters == 0:
        # the current face is still the best, but the goal is nearly lateral: pick the
        # better of the two neighbouring faces instead of pushing sideways
        neighbours = [item for item in candidates if item[1] != 0]
        score, quarters = min(neighbours, key=lambda item: item[0])
    direction_type = 3 if quarters > 0 else 4      # CCW = +90, CW = -90
    return direction_type * len(LEVELS) + len(LEVELS) - 1


def goal_push(env, rng) -> Tuple[int, int]:
    observations = env.boundary_observations()
    return _goal_action(observations[0]), _goal_action(observations[1])


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _push_scale_index(remaining: float) -> int:
    return next((i for i, level in enumerate(LEVELS) if level >= max(remaining, LEVELS[0])),
                len(LEVELS) - 1)


FACE_LOCAL = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def _face_index(normal, angle: float) -> int:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    scores = []
    for lx, ly in FACE_LOCAL:
        wx, wy = cos_a * lx - sin_a * ly, sin_a * lx + cos_a * ly
        scores.append(normal[0] * wx + normal[1] * wy)
    return max(range(4), key=scores.__getitem__)


def _goal_action_privileged(env, observation):
    """Diagnostic goal pusher that reads the object pose; never used by the learner.

    It is the "is a competent push even rewarded?" reference.  The object is a 1.0 x 0.25 m
    box.  The controller pushes along the local axis with the larger remaining goal error,
    from the face opposite the desired motion.  Navigation is face-index based: the robot
    orbits in a fixed direction until the observed face is the target face, which avoids the
    direction flipping that a purely angle-based rule shows at box corners.
    """
    patch = getattr(observation, "patch", None)
    if patch is None or not patch.valid:
        return WAIT2
    state = env.object_state()
    gx, gy = observation.goal_xy
    ex, ey = gx - state.x, gy - state.y
    cos_a, sin_a = math.cos(state.angle), math.sin(state.angle)
    local_x = cos_a * ex + sin_a * ey
    local_y = -sin_a * ex + cos_a * ey
    if abs(local_x) >= abs(local_y):
        sign = 1.0 if local_x >= 0.0 else -1.0
        push_local = (sign, 0.0)
        half_extent = spec.OBJECT_WIDTH / 2.0
        remaining = abs(local_x)
    else:
        sign = 1.0 if local_y >= 0.0 else -1.0
        push_local = (0.0, sign)
        half_extent = spec.OBJECT_HEIGHT / 2.0
        remaining = abs(local_y)
    outward_local = (-push_local[0], -push_local[1])
    target_face = FACE_LOCAL.index(outward_local)
    outward = (cos_a * outward_local[0] - sin_a * outward_local[1],
               sin_a * outward_local[0] + cos_a * outward_local[1])
    target = (state.x + outward[0] * (half_extent + ROBOT_GAP + spec.ROBOT_RADIUS),
              state.y + outward[1] * (half_extent + ROBOT_GAP + spec.ROBOT_RADIUS))
    current_face = _face_index(patch.normal_outward, state.angle)
    normal_dot = outward[0] * patch.normal_outward[0] + outward[1] * patch.normal_outward[1]
    if current_face != target_face:
        steps = (target_face - current_face) % 4
        direction_type = 3 if steps in (1, 2) else 4   # CCW for 1-2 quarters, CW for 3
        return direction_type * len(LEVELS) + len(LEVELS) - 1
    current_angle = math.atan2(observation.own.y - state.y, observation.own.x - state.x)
    target_angle = math.atan2(target[1] - state.y, target[0] - state.x)
    delta = _wrap(target_angle - current_angle)
    if abs(delta) < 0.15 and normal_dot > 0.85:
        speed_type = 0 if remaining < 0.30 else 2   # slow near the goal, fast when far
        return speed_type * len(LEVELS) + _push_scale_index(remaining - 0.02)
    direction_type = 3 if delta > 0.0 else 4
    orbit_radius = math.hypot(target[0] - state.x, target[1] - state.y)
    arc = abs(delta) * orbit_radius
    return direction_type * len(LEVELS) + _push_scale_index(arc)


def goal_push_privileged(env, rng) -> Tuple[int, int]:
    observations = env.boundary_observations()
    return _goal_action_privileged(env, observations[0]), WAIT2


POLICIES: Dict[str, Callable] = {
    "wait2": wait2,
    "wait24": wait24,
    "random_push": random_push,
    "random_legal": random_legal,
    "goal_push": goal_push,
    "goal_push_privileged": goal_push_privileged,
}


# ---------------------------------------------------------------------------
# runner: same accounting as train.run_episode
# ---------------------------------------------------------------------------


def run_episode(instance, policy, rng, *, max_skills: int,
                charge_truncation: bool) -> Dict[str, object]:
    env = E.RandomStartEnv(instance)
    initial = max(env.object_distance(), 1e-6)
    total_reward = 0.0
    components = {key: 0.0 for key in
                  ("progress", "distance_cost", "time_cost", "success_bonus", "failure_penalty")}
    counts: Counter = Counter()
    success = False
    terminated = False
    for _ in range(max_skills):
        a0, a1 = policy(env, rng)
        counts[ACTIONS[a0].kind] += 1
        counts[ACTIONS[a1].kind] += 1
        boundary = env.joint_step(a0, a1, record=False)
        total_reward += float(boundary["reward"]["total"])
        for key in components:
            components[key] += float(boundary["reward"].get(key, 0.0))
        success = bool(boundary["success"])
        terminated = boundary["termination"] is not None
        if terminated:
            break
    exit_reason = env.termination
    if not terminated and charge_truncation:
        total_reward += TRUNCATION_PENALTY
        components["failure_penalty"] += TRUNCATION_PENALTY
        exit_reason = "trainer_skill_cap"
    return {
        "return": float(total_reward),
        "success": bool(success),
        "progress": float(np.clip((initial - env.object_distance()) / initial, -1.0, 1.0)),
        "seconds": float(env.total_periods * CONTROL_DT),
        "exit_reason": exit_reason,
        "components": {key: float(value) for key, value in components.items()},
        "action_counts": {kind: int(count) for kind, count in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/option_rl_v45/fixed_policies.json")
    args = parser.parse_args()

    instances = layout.build_split("L0", "test", count=args.roots)[0]
    print(f"roots={len(instances)} env={spec.ENV_VERSION} actions={spec.ROBOT_ACTION_COUNT}", flush=True)

    cases: List[Tuple[str, str, int, bool]] = [
        ("old-wait2", "wait2", OLD_MAX_SKILLS, False),
        ("wait2", "wait2", MAX_SKILLS, True),
        ("wait24", "wait24", MAX_SKILLS, True),
        ("random_push", "random_push", MAX_SKILLS, True),
        ("random_legal", "random_legal", MAX_SKILLS, True),
        ("goal_push", "goal_push", MAX_SKILLS, True),
        ("goal_push_priv", "goal_push_privileged", MAX_SKILLS, True),
    ]
    report: Dict[str, object] = {"roots": [getattr(i, "name", str(i)) for i in instances],
                                 "env_version": spec.ENV_VERSION, "cases": {}}
    for label, policy_name, max_skills, charge in cases:
        rows = []
        for root_index, instance in enumerate(instances):
            rng = np.random.default_rng(args.seed * 100_003 + root_index)
            rows.append(run_episode(instance, POLICIES[policy_name], rng,
                                    max_skills=max_skills, charge_truncation=charge))
        summary = {
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_progress": float(np.mean([row["progress"] for row in rows])),
            "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
            "exit_reasons": dict(Counter(str(row["exit_reason"]) for row in rows)),
            "mean_components": {key: float(np.mean([row["components"][key] for row in rows]))
                                for key in rows[0]["components"]},
            "mean_actions": {kind: float(np.mean([row["action_counts"].get(kind, 0) for row in rows]))
                             for kind in ("push", "reposition", "wait", "follow")},
            "rows": rows,
        }
        report["cases"][label] = summary
        print(f"{label:14s} return={summary['mean_return']:+.3f} "
              f"success={summary['success_rate']:.3f} progress={summary['mean_progress']:+.3f} "
              f"seconds={summary['mean_seconds']:.1f} exits={summary['exit_reasons']}", flush=True)
        print(f"{'':14s} components={ {k: round(v, 3) for k, v in summary['mean_components'].items()} }", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
