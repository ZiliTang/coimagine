"""Phase D: closed-loop baselines under the same scalar protocol (contract section 4).

Every method re-plans from the current true observation at each decision and executes only
the first window of the chosen sequence. Live and planner environments are separate; the
learned method never queries the true simulator.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.environment import (
    TERMINATION_NUMERIC,
    TERMINATION_OUT_OF_BOUNDS,
    TERMINATION_PENETRATION,
    TERMINATION_SUCCESS,
    TERMINATION_WINDOW,
    Continuous2DEnv,
)
from ..continuous2d.scenes import scene_by_name
from ..continuous2d.skills import joint_spec_to_dicts
from .features import BoundaryState
from .model import JointWindowMLP
from .pools import candidate_pool, root_library
from .rollout import (
    EXIT_NON_FINITE,
    SequenceEvaluation,
    distance_of_state,
    evaluate_sequence_exact,
    evaluate_sequence_learned,
)
from .contract import score_components

METHOD_EXACT = "exact_fixed_pool"
METHOD_LEARNED = "learned_fixed_pool"
METHOD_UNIFORM = "uniform_root"

PHYSICAL_FAILURES = (TERMINATION_NUMERIC, TERMINATION_OUT_OF_BOUNDS, TERMINATION_PENETRATION)
PLANNER_NO_FEASIBLE = "planner_no_feasible_plan"


@dataclass(frozen=True)
class ClosedLoopConfig:
    depths: Tuple[int, ...] = (1, 3)
    cap: int = 128
    horizon: int = 12
    scenes: Tuple[str, ...] = ("S0_v4", "S1_v4")
    episode_seeds: Tuple[int, ...] = (0, 1, 2)
    model_seeds: Tuple[int, ...] = (0, 1, 2)

    def as_dict(self) -> Dict[str, object]:
        return {
            "depths": list(self.depths),
            "cap": self.cap,
            "horizon": self.horizon,
            "scenes": list(self.scenes),
            "episode_seeds": list(self.episode_seeds),
            "model_seeds": list(self.model_seeds),
        }


def pool_seed(scene: str, window: int, episode_seed: int) -> int:
    """Plan-pool seed from public run identifiers only; never from a model prediction."""

    digest = hashlib.sha256(f"{scene}|{window}|{episode_seed}|forecast-pool-v1".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _argmax(evaluations: Sequence[SequenceEvaluation]) -> int:
    return max(
        range(len(evaluations)),
        key=lambda index: (evaluations[index].j, -evaluations[index].candidate_id),
    )


def run_episode(
    method: str,
    scene_name: str,
    episode_seed: int,
    depth: int,
    config: Continuous2DConfig,
    *,
    closed: Optional[ClosedLoopConfig] = None,
    model: Optional[JointWindowMLP] = None,
    model_seed: Optional[int] = None,
) -> Dict[str, object]:
    closed = closed or ClosedLoopConfig()
    scene = scene_by_name(scene_name)
    live = Continuous2DEnv(config)
    planner = Continuous2DEnv(config)
    live.reset(scene)
    rng = random.Random(pool_seed(scene_name, 0, episode_seed))
    started = time.time()
    plan_clock = 0.0
    total_time = 0.0
    windows: List[Dict[str, object]] = []
    prediction_nodes = 0
    exact_steps = 0
    prefix_steps = 0
    termination = TERMINATION_WINDOW
    no_feasible = False

    for window in range(closed.horizon):
        if live.termination is not None:
            break
        observation = live.observation()
        seed = pool_seed(scene_name, window, episode_seed)
        if method == METHOD_UNIFORM:
            options = root_library(observation, config)
            choice = random.Random(seed).choice(options)
            chosen_sequence = (choice.skills,)
            decision = {
                "window_index": window,
                "method": method,
                "pool_seed": seed,
                "candidate_count": len(options),
                "feasible_count": None,
                "rejected_candidates": 0,
                "rejection_rate": None,
                "chosen_candidate_id": int(choice.candidate_id),
                "chosen_sequence": [joint_spec_to_dicts(spec) for spec in chosen_sequence],
                "predicted_J": None,
                "predicted_exit_class": None,
                "plan_wall_clock": 0.0,
            }
        else:
            pool = candidate_pool(observation, config, depth, seed, cap=closed.cap)
            plan_started = time.time()
            if method == METHOD_EXACT:
                snapshot = live.export_state()
                evaluations = [
                    evaluate_sequence_exact(
                        planner, snapshot, scene, sequence, config, candidate_id=index
                    )
                    for index, sequence in enumerate(pool)
                ]
                exact_steps += sum(evaluation.physics_steps for evaluation in evaluations)
                prefix_steps += sum(evaluation.prefix_replay_steps for evaluation in evaluations)
            else:
                assert model is not None
                start = BoundaryState(
                    object=observation.object,
                    robots=observation.robots,
                    success_counter=live.success_counter,
                )
                evaluations = [
                    evaluate_sequence_learned(
                        model, scene, start, sequence, config, candidate_id=index
                    )
                    for index, sequence in enumerate(pool)
                ]
                prediction_nodes += sum(evaluation.prediction_nodes for evaluation in evaluations)
            plan_clock += time.time() - plan_started
            # A3: only window_complete / success may be executed. invalid_prediction,
            # non_finite and physical_failure are all rejected.
            feasible = [
                index for index, evaluation in enumerate(evaluations)
                if evaluation.feasible
            ]
            rejected = len(evaluations) - len(feasible)
            if not pool or not feasible:
                termination = PLANNER_NO_FEASIBLE
                no_feasible = True
                windows.append(
                    {
                        "window_index": window,
                        "method": method,
                        "pool_seed": seed,
                        "candidate_count": len(pool),
                        "chosen_candidate_id": -1,
                        "chosen_sequence": [],
                        "predicted_J": None,
                        "predicted_exit_class": PLANNER_NO_FEASIBLE,
                        "plan_wall_clock": time.time() - plan_started,
                        "rejected_candidates": rejected,
                        "rejection_rate": (rejected / len(evaluations)) if evaluations else None,
                        "stop_tick": live.tick,
                        "stop_elapsed_time": total_time,
                        "stop_terminal_distance": distance_of_state(
                            BoundaryState(
                                object=live.backend.object_state(),
                                robots=live.backend.robot_states(),
                                success_counter=live.success_counter,
                            ),
                            scene, config,
                        ),
                    }
                )
                break
            best = max(feasible, key=lambda index: (evaluations[index].j, -evaluations[index].candidate_id))
            chosen_sequence = pool[best]
            decision = {
                "window_index": window,
                "method": method,
                "pool_seed": seed,
                "candidate_count": len(pool),
                "feasible_count": len(feasible),
                "rejected_candidates": rejected,
                "rejection_rate": rejected / len(evaluations),
                "chosen_candidate_id": int(evaluations[best].candidate_id),
                "chosen_sequence": [joint_spec_to_dicts(spec) for spec in chosen_sequence],
                "predicted_J": float(evaluations[best].j),
                "predicted_exit_class": evaluations[best].exit_class,
                "predicted_components": evaluations[best].components.as_dict(),
                "plan_wall_clock": time.time() - plan_started,
            }

        decision["state_before"] = {
            "object": list(observation.object.as_tuple()),
            "robots": [list(state.as_tuple()) for state in observation.robots],
            "success_counter": live.success_counter,
        }
        result = live.execute_joint_skill(chosen_sequence[0])
        total_time += float(result.cost_components["time"])
        termination = result.termination
        decision["state_after"] = {
            "object": list(result.final.object.as_tuple()),
            "robots": [list(state.as_tuple()) for state in result.final.robots],
            "success_counter": result.final.success_counter,
        }
        decision["executed_duration"] = float(result.duration)
        decision["executed_termination"] = result.termination
        decision["executed_success"] = int(result.success)
        windows.append(decision)
        if result.termination in (TERMINATION_SUCCESS, *PHYSICAL_FAILURES):
            break

    success = any(int(entry.get("executed_success", 0)) for entry in windows)
    final_state = BoundaryState(
        object=live.backend.object_state(),
        robots=live.backend.robot_states(),
        success_counter=live.success_counter,
    )
    distance = 0.0 if success else distance_of_state(final_state, scene, config)
    failure = termination in PHYSICAL_FAILURES
    components = score_components(
        success=success,
        time=total_time,
        distance=distance,
        failure=failure,
        diagonal=(scene.arena[0] ** 2 + scene.arena[1] ** 2) ** 0.5,
    )
    return {
        "method": method,
        "scene_id": scene_name,
        "episode_seed": episode_seed,
        "H": depth,
        "model_seed": model_seed,
        "success": int(success),
        "termination": termination,
        "no_feasible_plan": int(no_feasible),
        "windows": len(windows),
        "completion_time": total_time,
        "terminal_distance": distance,
        "physical_failure": int(failure),
        "J": components.j,
        "components": components.as_dict(),
        "prediction_nodes": prediction_nodes,
        "exact_simulation_steps": exact_steps,
        "prefix_replay_steps": prefix_steps,
        "plan_wall_clock": plan_clock,
        "wall_clock": time.time() - started,
        "decisions": windows,
    }
