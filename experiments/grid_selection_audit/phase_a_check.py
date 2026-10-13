"""Phase A acceptance: one real action and one model prediction on an existing scene.

Run: ``python experiments/grid_selection_audit/phase_a_check.py``

It reads only the frozen legacy assets and writes nothing. Exit code 0 means the scales,
the return direction and the action identity could be matched, which is the Phase A
acceptance condition.
"""

from __future__ import annotations

import sys

import numpy as np

from coimagine.grid.dynamics import JointSkill, LocalSkill, SKILL_HOLD, SKILL_PUSH, execute
from coimagine.grid.scenes import a0_public_sufficient
from coimagine.grid.sensing import ray_cast
from coimagine.grid.state import Direction
from coimagine.learning.belief_model import load_belief_model
from coimagine.learning.features import ACTION_IDS, observation_vector

CHECKPOINT = "outputs/grid/world_model/belief.npz"
ACTION = "push:0:E"
GOAL = frozenset({(2, 3), (3, 3), (2, 4), (3, 4)})


def main() -> int:
    scene = a0_public_sufficient()
    # Robot A pushes east alone; robot B holds. In this scene one push is enough.
    joint = JointSkill((LocalSkill(SKILL_PUSH, Direction.EAST), LocalSkill(SKILL_HOLD)))
    successor, result = execute(scene, joint)
    print(f"REAL  {ACTION}  cost={result.cost}  y_public={result.y_public}  "
          f"feedback={dict(result.feedback)}  failures={result.failures}")
    print(f"REAL  object {scene.obj.origin} -> {successor.obj.origin}  goal={successor.is_goal()}")

    observation = np.asarray(
        observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ()),
        dtype=np.float64,
    )
    model = load_belief_model(CHECKPOINT)
    dynamic = model.dynamic_state(observation)
    belief = model.parameter_belief(observation)
    states = model.step_actions(dynamic, belief)
    costs = model.cost_actions(states, belief)
    index = ACTION_IDS.index(ACTION)
    probability = states[index, 1 : 1 + 36]
    goal_mask = np.asarray(
        [1.0 if {(i % 6, i // 6), (i % 6 + 1, i // 6), (i % 6, i // 6 + 1), (i % 6 + 1, i // 6 + 1)} <= GOAL else 0.0
         for i in range(36)]
    )
    top = int(np.argmax(probability))
    print(f"MODEL {ACTION}  cost_hat={costs[index]:.3f}  "
          f"P(object in goal)={float(probability @ goal_mask):.3f}  "
          f"top cell=({top % 6},{top // 6})  mass_sum={float(probability.sum()):.3f}")
    print(f"ACTION ORDER {ACTION_IDS}")

    ok = (
        abs(float(result.cost) - 1.0) < 1e-9
        and result.y_public == "moved"
        and successor.is_goal()
        and abs(float(costs[index]) - 1.0) < 0.2
        and abs(float(probability.sum()) - 1.0) < 1e-6
        and float(probability @ goal_mask) > 0.5
    )
    print("PHASE_A_OK" if ok else "PHASE_A_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
