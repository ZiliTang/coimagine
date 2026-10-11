#!/usr/bin/env python
"""Read-only review of the round-20 diagnosis.

Emits ``outputs/grid/diagnosis/review.json``: the raw facts a code walkthrough needs.

* corrected full-set and reachable-subset denominators for the random baseline;
* per-episode results for the five training seeds on the fixed reachable batch, with the
  random baseline on that same batch and the exact model for reference;
* per-episode results for the module-replacement conditions, with the checkpoint and the
  scene ids;
* a full decomposition of the scene-6 ``wait`` versus ``push:1:E`` score at depth 2;
* a trace of which candidate ``_beam_value`` keeps at each level;
* whether the learned ``wait`` transition is the identity;
* whether depth 1 and depth 2 choose the same first action for the exact adapter over the
  whole evaluated batch.

Nothing here trains a *new* configuration or changes a method: the seed block re-runs the
exact seeds the diagnosis already used, so that the per-episode records the report did not
persist become inspectable.  The aggregate numbers must reproduce the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.diagnostics.belief_probe import (  # noqa: E402
    BRANCH_THRESHOLD,
    FAIL_COST,
    PROGRESS_WEIGHT,
    _beam_value,
    _cell_distance_distribution,
    _cut_value,
    plan_with_search,
)
from coimagine.diagnostics.diagnosis import (  # noqa: E402
    EVAL_SEED,
    _measurement_starts,
    _random_baseline,
    exact_reachability,
    run_lookahead_policy,
    run_module_replacement,
    run_seed_variance,
    summarise,
)
from coimagine.diagnostics.exact_adapter import ExactFactoredModel, exact_model_factory  # noqa: E402
from coimagine.grid.dynamics import FEEDBACK_SYMBOLS, execute  # noqa: E402
from coimagine.grid.planner import PHYSICS  # noqa: E402
from coimagine.grid.sensing import ray_cast  # noqa: E402
from coimagine.learning.belief_model import N_PHYSICS, load_belief_model  # noqa: E402
from coimagine.learning.data import _joint_for, pose_to_index, probe_scene  # noqa: E402
from coimagine.learning.features import ACTION_IDS, observation_vector  # noqa: E402

OUT = ROOT / "outputs" / "grid" / "diagnosis" / "review.json"


# ---------------------------------------------------------------------------
# Scene-6 style decomposition
# ---------------------------------------------------------------------------


def beam_trace(model, state, belief, distance_table, *, steps: int, beam: int = 1):
    """Mirror of ``_beam_value`` that also returns what it kept at every level."""

    levels = []
    states = [np.atleast_2d(state)]
    costs = [0.0]
    for level in range(steps):
        candidates = []
        candidate_costs = []
        for partial, spent in zip(states, costs):
            next_states = model.step_actions(partial, belief)
            next_costs = model.cost_actions(next_states, belief)
            for action in range(len(ACTION_IDS)):
                candidates.append(np.atleast_2d(next_states[action]))
                candidate_costs.append(spent + float(next_costs[action]))
        scored = [
            (
                candidate_costs[i] + _cut_value(candidates[i], distance_table, PROGRESS_WEIGHT, FAIL_COST),
                i,
            )
            for i in range(len(candidates))
        ]
        scored.sort()
        keep = [index for _value, index in scored[:beam]]
        levels.append(
            {
                "level": level + 1,
                "candidates": [
                    {
                        "index": index,
                        "action": ACTION_IDS[index % len(ACTION_IDS)],
                        "spent_cost": candidate_costs[index],
                        "cut": _cut_value(
                            candidates[index], distance_table, PROGRESS_WEIGHT, FAIL_COST
                        ),
                        "total": candidate_costs[index]
                        + _cut_value(
                            candidates[index], distance_table, PROGRESS_WEIGHT, FAIL_COST
                        ),
                        "kept": index in keep,
                    }
                    for _value, index in scored
                ],
            }
        )
        states = [candidates[i] for i in keep]
        costs = [candidate_costs[i] for i in keep]
    value = min(
        costs[i] + _cut_value(states[i], distance_table, PROGRESS_WEIGHT, FAIL_COST)
        for i in range(len(states))
    )
    return value, levels


def decompose_boundary(model, scene, belief, *, depth: int, beam: int = 1, label: str = ""):
    """Every term ``plan_with_search`` sums for one decision boundary."""

    goal = frozenset(scene.static.target)
    distance_table = _cell_distance_distribution(None, goal)
    observation = np.asarray(
        observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
    )
    dynamic = model.dynamic_state(observation)
    first_states = model.step_actions(dynamic, belief)
    first_costs = model.cost_actions(first_states, belief)

    rows: List[Dict[str, object]] = []
    for action in range(len(ACTION_IDS)):
        state = np.atleast_2d(first_states[action])
        cost = float(first_costs[action])
        probabilities = model.predict_outcome(state, action, belief)[0]
        expected = 0.0
        total_probability = 0.0
        branches = []
        for symbol in range(len(probabilities)):
            probability = float(probabilities[symbol])
            if probability < BRANCH_THRESHOLD:
                continue
            updated = model.update_belief(belief, dynamic, action, symbol)
            state_after = np.atleast_2d(model.step(dynamic, action, updated))
            value, levels = beam_trace(
                model, state_after, updated, distance_table, steps=depth - 1, beam=beam
            )
            expected += probability * value
            total_probability += probability
            branches.append(
                {
                    "symbol": symbol,
                    "symbol_name": FEEDBACK_SYMBOLS[symbol] if symbol < 4 else f"public{symbol}",
                    "probability": probability,
                    "posterior_before": [round(float(v), 6) for v in belief[0]],
                    "posterior_after": [round(float(v), 6) for v in updated[0]],
                    "value_after": value,
                    "contribution": probability * value,
                    "beam_levels": levels,
                }
            )
        tail = 0.0
        if total_probability < 1.0:
            tail = (1.0 - total_probability) * _cut_value(
                state, distance_table, PROGRESS_WEIGHT, FAIL_COST
            )
            expected += tail
        score = cost + expected
        cells = np.asarray(state[0, 1 : 1 + 37], dtype=float)
        rows.append(
            {
                "label": label,
                "action": ACTION_IDS[action],
                "first_cost": cost,
                "first_state_cell_mass_top3": sorted(
                    (round(float(v), 4), i) for i, v in enumerate(cells) if v > 0.01
                )[-3:],
                "first_state_cut": _cut_value(
                    state, distance_table, PROGRESS_WEIGHT, FAIL_COST
                ),
                "released_probability": float(state[0, 0]),
                "symbol_probabilities": [round(float(p), 6) for p in probabilities],
                "branches": branches,
                "tail_term": tail,
                "expected": expected,
                "score": score,
            }
        )
    chosen = min(rows, key=lambda row: (row["score"], row["action"]))
    return {
        "label": label,
        "depth": depth,
        "beam": beam,
        "progress_weight": PROGRESS_WEIGHT,
        "fail_cost": FAIL_COST,
        "branch_threshold": BRANCH_THRESHOLD,
        "rows": rows,
        "chosen": chosen["action"],
    }


def scene6_decomposition() -> Dict[str, object]:
    starts, _pre, _warm = _measurement_starts(
        n_scenes=60, seed=EVAL_SEED, warmup="probe:0:E"
    )
    entry = next(e for e in starts if e["scene_id"] == 6)
    scene = entry["scene"]
    model = ExactFactoredModel(static=scene.static)
    belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    warm_action, warm_symbol = entry["warmup"]
    observation0 = np.asarray(
        observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
    )
    dynamic0 = model.dynamic_state(observation0)
    belief = model.update_belief(belief, dynamic0, warm_action, warm_symbol)

    true_index = PHYSICS.index(entry["physics"])
    onehot = np.zeros((1, N_PHYSICS))
    onehot[0, true_index] = 1.0

    return {
        "scene_id": 6,
        "physics": entry["physics"].label,
        "physics_index": true_index,
        "origin": list(scene.obj.origin),
        "robots": [list(r.pose) for r in scene.robots],
        "exact_reachable_steps": exact_reachability(scene, horizon=8),
        "posterior_after_warmup": [round(float(v), 6) for v in belief[0]],
        "uniform_belief": decompose_boundary(model, scene, np.full((1, N_PHYSICS), 1.0 / N_PHYSICS), depth=2, label="uniform prior"),
        "warmup_posterior": decompose_boundary(model, scene, belief, depth=2, label="posterior after the warm-up probe"),
        "true_physics_onehot": decompose_boundary(model, scene, onehot, depth=2, label="true physics known"),
        "depth_1_warmup_posterior": decompose_boundary(model, scene, belief, depth=1, label="depth 1"),
        "depth_3_warmup_posterior": decompose_boundary(model, scene, belief, depth=3, label="depth 3"),
    }


# ---------------------------------------------------------------------------
# State estimate and one-step transition, with and without the parameter mixture
# ---------------------------------------------------------------------------


def state_and_one_step(
    path: str = "outputs/grid/world_model/belief.npz", n_scenes: int = 100
) -> Dict[str, object]:
    """One-step behaviour on real decision states, under the belief and under the true theta.

    Read-only and label-only use of the simulator.  ``state_top1`` is the encoder's current
    object cell; ``cell_top1`` is the transition's next object cell for every one of the six
    actions, once with the parameter one-hot (the training-time condition) and once with the
    model's own belief (the inference-time condition).  The gap between those two columns is
    the cost of querying the transition at the marginal state instead of a per-hypothesis
    state.
    """

    learned = load_belief_model(path)
    starts, _pre, _warm = _measurement_starts(n_scenes=n_scenes, seed=EVAL_SEED, warmup="probe:0:E")

    counters = {
        name: {
            "state_top1": 0,
            "state_cell_tv": 0.0,
            "cell_top1_belief": 0,
            "cell_top1_true_theta": 0,
            "moved_correct_belief": 0,
            "moved_correct_true_theta": 0,
            "expected_cell_error_belief": 0.0,
            "expected_cell_error_true_theta": 0.0,
        }
        for name in ("learned", "exact")
    }
    total = 0
    for entry in starts:
        scene = entry["scene"]
        theta_true = PHYSICS.index(entry["physics"])
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        true_cell = pose_to_index(scene.obj.origin)
        exact = ExactFactoredModel(static=scene.static)
        onehot = np.zeros((1, N_PHYSICS))
        onehot[0, theta_true] = 1.0
        models = {
            "learned": (learned, learned.parameter_belief(observation)),
            "exact": (exact, np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)),
        }
        for name, (model, belief) in models.items():
            dynamic = model.dynamic_state(observation)
            top1 = int(np.argmax(dynamic[0, 1:37]))
            counters[name]["state_top1"] += int(top1 == true_cell)
            counters[name]["state_cell_tv"] += float(
                1.0 - dynamic[0, 1 + true_cell]
            )
            states_belief = model.step_actions(dynamic, belief)
            states_true = model.step_actions(dynamic, onehot)
            for action, action_id in enumerate(ACTION_IDS):
                successor, result = execute(scene, _joint_for(None, action_id))
                true_next = pose_to_index(successor.obj.origin)
                moved = 1 if result.y_public == "moved" else 0
                for tag, states in (("belief", states_belief), ("true_theta", states_true)):
                    cells = np.asarray(states[action][1 : 1 + 36], dtype=float)
                    counters[name][f"cell_top1_{tag}"] += int(int(np.argmax(cells)) == true_next)
                    # "the object moved" = the successor cell is not the current cell
                    moved_probability = float(cells.sum() - cells[true_cell])
                    counters[name][f"moved_correct_{tag}"] += int(
                        (moved_probability >= 0.5) == bool(moved)
                    )
                    distance_table = _cell_distance_distribution(None, frozenset(scene.static.target))
                    counters[name][f"expected_cell_error_{tag}"] += float(
                        cells @ distance_table
                    ) - _expected_true_distance(true_next, scene)
        total += 1

    denominator = max(1, total * len(ACTION_IDS))
    return {
        "checkpoint": path,
        "decision_points": total,
        "actions_per_point": len(ACTION_IDS),
        "learned": {
            "state_top1_accuracy": counters["learned"]["state_top1"] / max(1, total),
            "state_true_cell_probability_mean": counters["learned"]["state_cell_tv"] / max(1, total),
            "cell_top1_accuracy_belief": counters["learned"]["cell_top1_belief"] / denominator,
            "cell_top1_accuracy_true_theta": counters["learned"]["cell_top1_true_theta"] / denominator,
            "moved_accuracy_belief": counters["learned"]["moved_correct_belief"] / denominator,
            "moved_accuracy_true_theta": counters["learned"]["moved_correct_true_theta"] / denominator,
            "expected_cell_error_belief": counters["learned"]["expected_cell_error_belief"] / denominator,
            "expected_cell_error_true_theta": counters["learned"]["expected_cell_error_true_theta"] / denominator,
        },
        "exact": {
            "state_top1_accuracy": counters["exact"]["state_top1"] / max(1, total),
            "cell_top1_accuracy_belief": counters["exact"]["cell_top1_belief"] / denominator,
            "cell_top1_accuracy_true_theta": counters["exact"]["cell_top1_true_theta"] / denominator,
            "moved_accuracy_belief": counters["exact"]["moved_correct_belief"] / denominator,
            "expected_cell_error_belief": counters["exact"]["expected_cell_error_belief"] / denominator,
        },
        "note": (
            "`belief` conditions the transition on the model's own 16-way posterior and one "
            "marginal dynamic state; `true_theta` uses the true parameter one-hot, which is the "
            "condition the transition was trained under.  The difference is the inference-time "
            "cost of not carrying a per-hypothesis state."
        ),
    }


def cost_and_likelihood_heads(
    path: str = "outputs/grid/world_model/belief.npz", n_scenes: int = 100
) -> Dict[str, object]:
    """What the cost head and the observation model output on real states.

    ``plan_with_search`` minimises ``cost + expected future``, and it weights its feedback
    branches by ``predict_outcome``; the same likelihood drives ``update_belief``.  A cost
    head that returns the wrong ordering changes the choice directly, and a weak likelihood
    changes the branch weights.  Both are read out here against the simulator's own values,
    under the model's own belief and under the true parameter one-hot (the training-time
    condition), for the learned model and the exact adapter alike.
    """

    learned = load_belief_model(path)
    starts, _pre, _warm = _measurement_starts(n_scenes=n_scenes, seed=EVAL_SEED, warmup="probe:0:E")
    true_costs = {"wait": 0.0, "probe:0:E": 0.3}
    cost_errors: List[float] = []
    argmin_match = 0
    likelihood = {
        name: {
            "correct": 0,
            "total": 0,
            "mass_on_truth": 0.0,
            "per_action_correct": {action: 0 for action in ACTION_IDS},
            "per_action_total": {action: 0 for action in ACTION_IDS},
        }
        for name in ("learned_belief", "learned_true_theta", "exact_belief", "exact_true_theta")
    }
    for entry in starts:
        scene = entry["scene"]
        theta_true = PHYSICS.index(entry["physics"])
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        onehot = np.zeros((1, N_PHYSICS))
        onehot[0, theta_true] = 1.0
        learned_belief = learned.parameter_belief(observation)
        exact = ExactFactoredModel(static=scene.static)
        exact_belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)

        conditions = {
            "learned_belief": (learned, learned_belief, learned.dynamic_state(observation)),
            "learned_true_theta": (learned, onehot, learned.dynamic_state(observation)),
            "exact_belief": (exact, exact_belief, exact.dynamic_state(observation)),
            "exact_true_theta": (exact, onehot, exact.dynamic_state(observation)),
        }
        for name, (model, belief, dynamic) in conditions.items():
            states = model.step_actions(dynamic, belief)
            if name == "learned_belief":
                predicted_costs = learned.cost_actions(states, belief)
                truth = np.asarray([true_costs.get(action, 1.0) for action in ACTION_IDS])
                cost_errors.extend(list(predicted_costs - truth))
                if int(np.argmin(predicted_costs)) == int(np.argmin(truth)):
                    argmin_match += 1
            for action, action_id in enumerate(ACTION_IDS):
                _successor, result = execute(scene, _joint_for(None, action_id))
                probabilities = model.predict_outcome(
                    np.atleast_2d(states[action]), action, belief
                )[0]
                true_symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
                entry_likelihood = likelihood[name]
                entry_likelihood["total"] += 1
                entry_likelihood["per_action_total"][action_id] += 1
                entry_likelihood["mass_on_truth"] += float(probabilities[true_symbol])
                if int(np.argmax(probabilities)) == true_symbol:
                    entry_likelihood["correct"] += 1
                    entry_likelihood["per_action_correct"][action_id] += 1

    def summarise_likelihood(entry) -> Dict[str, object]:
        return {
            "argmax_matches_true_feedback": f"{entry['correct']}/{entry['total']}",
            "argmax_accuracy": entry["correct"] / max(1, entry["total"]),
            "mean_probability_on_true_feedback": entry["mass_on_truth"] / max(1, entry["total"]),
            "per_action_accuracy": {
                action: f"{entry['per_action_correct'][action]}/{entry['per_action_total'][action]}"
                for action in ACTION_IDS
            },
        }

    return {
        "checkpoint": path,
        "decision_points": len(starts),
        "true_costs": {**true_costs, "other": 1.0},
        "cost_head": {
            "mean_signed_error": float(np.mean(cost_errors)),
            "mean_absolute_error": float(np.mean(np.abs(cost_errors))),
            "argmin_matches_true_cost": f"{argmin_match}/{len(starts)}",
        },
        "observation_head": {
            name: summarise_likelihood(entry) for name, entry in likelihood.items()
        },
    }


def _expected_true_distance(cell_index: int, scene) -> float:
    goal = frozenset(scene.static.target)
    origin = (cell_index % 6, cell_index // 6)
    members = [(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)]
    if not goal:
        return 0.0
    return float(max(min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in members))


# ---------------------------------------------------------------------------
# Learned wait transition
# ---------------------------------------------------------------------------


def learned_wait_identity(path: str = "outputs/grid/world_model/belief.npz") -> Dict[str, object]:
    model = load_belief_model(path)
    starts, _pre, _warm = _measurement_starts(n_scenes=100, seed=EVAL_SEED, warmup="probe:0:E")
    wait = ACTION_IDS.index("wait")
    cell_l1: List[float] = []
    released_delta: List[float] = []
    max_cell_mass_shift: List[float] = []
    for entry in starts:
        scene = entry["scene"]
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        state = model.dynamic_state(observation)
        belief = model.parameter_belief(observation)
        nxt = model.step(state, wait, belief)
        cell_l1.append(float(np.abs(nxt[0, 1:37] - state[0, 1:37]).sum() / 2.0))
        released_delta.append(float(abs(nxt[0, 0] - state[0, 0])))
        max_cell_mass_shift.append(
            float(np.abs(nxt[0, 1:37] - state[0, 1:37]).max())
        )
    return {
        "checkpoint": path,
        "decision_points": len(cell_l1),
        "cell_total_variation_mean": float(np.mean(cell_l1)),
        "cell_total_variation_max": float(np.max(cell_l1)),
        "largest_single_cell_mass_change_mean": float(np.mean(max_cell_mass_shift)),
        "released_probability_change_mean": float(np.mean(released_delta)),
        "note": (
            "The exact model's wait is the identity (0.0 on every entry); the learned "
            "transition is not, so `continue_from(first, wait)` moves the learned state and "
            "depth 2 need not equal depth 1 for the learned model."
        ),
    }


def exact_wait_identity(path: str = "outputs/grid/world_model/belief.npz") -> Dict[str, object]:
    """The same measurement for the exact adapter whose wait really is the identity."""

    starts, _pre, _warm = _measurement_starts(n_scenes=20, seed=EVAL_SEED, warmup="probe:0:E")
    wait = ACTION_IDS.index("wait")
    deltas: List[float] = []
    for entry in starts:
        scene = entry["scene"]
        model = ExactFactoredModel(static=scene.static)
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        state = model.dynamic_state(observation)
        belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
        nxt = model.step(state, wait, belief)
        deltas.append(float(np.abs(nxt[0] - state[0]).max()))
    return {
        "decision_points": len(deltas),
        "max_absolute_change_any_component": float(np.max(deltas)) if deltas else None,
    }


# ---------------------------------------------------------------------------
# Depth-1 versus depth-2 first action, over the whole measured batch
# ---------------------------------------------------------------------------


def depth_equivalence_scan(n_scenes: int = 100) -> Dict[str, object]:
    per_depth: Dict[int, Dict[int, Optional[str]]] = {1: {}, 2: {}, 3: {}}
    for depth in (1, 2, 3):
        records = run_lookahead_policy(
            model_factory=exact_model_factory(condition_on_feedback=True),
            name=f"exact-d{depth}", n_scenes=n_scenes, horizon=8, depth=depth,
        )
        for record in records:
            per_depth[depth][record.scene_id] = record.first_action
    ids = sorted(per_depth[1])
    same_12 = sum(1 for i in ids if per_depth[1][i] == per_depth[2][i])
    same_23 = sum(1 for i in ids if per_depth[2][i] == per_depth[3][i])
    differences = [
        {
            "scene_id": i,
            "depth_1": per_depth[1][i],
            "depth_2": per_depth[2][i],
            "depth_3": per_depth[3][i],
        }
        for i in ids
        if per_depth[1][i] != per_depth[2][i] or per_depth[2][i] != per_depth[3][i]
    ]
    return {
        "n_scenes": len(ids),
        "depth_1_equals_depth_2": f"{same_12}/{len(ids)}",
        "depth_2_equals_depth_3": f"{same_23}/{len(ids)}",
        "differences": differences,
    }


def main() -> int:
    result: Dict[str, object] = {}

    result["metrics_corrections"] = {
        "full_set": {
            "n": 300,
            "reachability_ceiling": "126/300 = 0.420",
            "random": _random_baseline(300, warmup="probe:0:E")["all"],
        },
        "exact_reachable_subset": {
            "n": 126,
            "reachability_ceiling": "126/126 = 1.000",
            "random": _random_baseline(300, warmup="probe:0:E")["exact_reachable"],
        },
        "note": (
            "The two rows must be compared within themselves: the full-set ceiling is 0.420 "
            "against a 0.123 random baseline; the reachable-subset ceiling is 1.000 against a "
            "0.294 random baseline.  The report's section 4 juxtaposed the full-set ceiling "
            "with the subset random baseline."
        ),
    }

    seed_block = run_seed_variance(
        seeds=(0, 1, 2, 3, 4), n_train_episodes=400, epochs=120, n_scenes=100,
        depths=(1, 2, 3), seed_eval=EVAL_SEED, record_episodes=True,
    )
    starts, _pre, _warm = _measurement_starts(n_scenes=100, seed=EVAL_SEED, warmup="probe:0:E")
    reachable_ids = sorted(
        entry["scene_id"]
        for entry in starts
        if exact_reachability(entry["scene"], horizon=8) is not None
    )
    result["seed_episodes"] = {
        "training": {"seeds": [0, 1, 2, 3, 4], "n_train_episodes": 400, "epochs": 120, "lead_in": 2},
        "eval": {"n_scenes": 100, "seed": EVAL_SEED, "warmup": "probe:0:E", "horizon": 8},
        "reachable_scene_ids": reachable_ids,
        "random_on_same_batch": _random_baseline(100, warmup="probe:0:E"),
        "summary": seed_block["summary"],
        "exact_model_baseline": {
            depth: {k: v for k, v in row.items() if k != "episodes"}
            for depth, row in seed_block["exact_model_baseline"].items()
        },
        "exact_model_episodes": {
            depth: row["episodes"] for depth, row in seed_block["exact_model_baseline"].items()
        },
        "per_seed": seed_block["per_seed"],
    }

    result["module_replacement_episodes"] = run_module_replacement(
        n_scenes=100, depth=2, beam=1, record_episodes=True
    )

    result["scene6"] = scene6_decomposition()
    result["state_and_one_step"] = state_and_one_step()
    result["cost_and_likelihood_heads"] = cost_and_likelihood_heads()
    result["learned_wait_transition"] = learned_wait_identity()
    result["exact_wait_transition"] = exact_wait_identity()
    result["depth_equivalence_scan"] = depth_equivalence_scan(n_scenes=100)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT}")
    print(json.dumps(result["metrics_corrections"], indent=2, default=str)[:1200])
    print("depth equivalence:", result["depth_equivalence_scan"]["depth_1_equals_depth_2"],
          result["depth_equivalence_scan"]["depth_2_equals_depth_3"])
    print("learned wait:", {k: v for k, v in result["learned_wait_transition"].items() if k != "note"})
    print("scene6 chosen:", {k: v["chosen"] for k, v in result["scene6"].items() if isinstance(v, dict) and "chosen" in v})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
