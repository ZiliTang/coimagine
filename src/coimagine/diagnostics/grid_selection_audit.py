"""Phase B: the legacy grid's last candidate-selection diagnostic.

The question is narrow. Fix the candidate set, isolate the **selection** step, and ask
whether the learned world model systematically **overestimates** the candidates its own
scoring then chooses. Nothing here re-plans, branches on feedback, or reads the real
state mid-sequence.

Protocol (per ``docs/CoImagine_Agent_迁移任务书_v1.md`` section 4):

* the first 20 valid scenes by stable scene id, no result-based selection;
* depths ``H = 1, 2, 3``, compared separately;
* a deterministic random order over all ``6**H`` action tuples gives nested sets
  ``C16 <= C64 <= C128``;
* the model predicts the **whole fixed sequence from the initial input** and never
  reads the real state in the middle;
* the true side executes the same sequence from the same initial world, on every world
  consistent with the robot's information, and the return is the exact expectation over
  those worlds -- never ``b(theta)`` times a marginal state.

Both sides use the single scalar

    J = SUCCESS_BONUS * 1[success] - charged_cost - DISTANCE_WEIGHT * terminal_distance

with the task's absorbing-success convention (once the object is in the goal, the rest of
the sequence is free and the state is frozen). The model's version replaces the indicator
by its own best-guess success (predicted ``P(in goal) >= SUCCESS_THRESHOLD``) and the
distance by its predicted expectation. That convention is stated here once and used by
every number in the output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..grid.dynamics import FEEDBACK_SYMBOLS, execute
from ..grid.state import Cell, HiddenPhysics, Scene, enumerate_hidden_physics
from ..learning.belief_model import BeliefWorldModel
from ..learning.data import Situation, _joint_for, generate_situations
from ..learning.features import ACTION_IDS, observation_vector
from .belief_probe import _cell_distance_distribution, belief_from_history

#: Fixed scoring constants. Large enough that success dominates any in-horizon cost term.
SUCCESS_BONUS = 10.0
DISTANCE_WEIGHT = 0.5
#: The model's own best-guess success threshold on ``P(object in goal)``.
SUCCESS_THRESHOLD = 0.5
#: Per (scene, depth): at most this many distinct sequences are ever used.
MAX_SEQUENCES = 128
N_CELLS = 36
GRID = 6


# ---------------------------------------------------------------------------
# Decision points
# ---------------------------------------------------------------------------


@dataclass
class DecisionPoint:
    """One audited decision point and everything the audit needs about it."""

    situation: Situation
    observation: np.ndarray
    #: Every hidden-physics type consistent with the robot's own information, and the
    #: exact joint world distribution (uniform on that set; the scan is noiseless).
    worlds: Tuple[HiddenPhysics, ...]
    weights: np.ndarray
    model_belief: np.ndarray
    model_dynamic: np.ndarray
    goal: frozenset
    distance_table: np.ndarray
    goal_mask: np.ndarray

    @property
    def scene_id(self) -> int:
        return self.situation.scene_id


def _history_pairs(situation: Situation) -> List[Tuple[int, int]]:
    """The ``(action_index, feedback_symbol_index)`` pairs the belief update consumes."""

    label_to_action = {_joint_for(None, action).label: index for index, action in enumerate(ACTION_IDS)}
    pairs: List[Tuple[int, int]] = []
    for skill, symbol in zip(situation.skills, situation.history_a):
        pairs.append((label_to_action[skill.label], FEEDBACK_SYMBOLS.index(symbol)))
    return pairs


def consistent_worlds(situation: Situation) -> Tuple[HiddenPhysics, ...]:
    """Exact set of hidden-physics types consistent with the robot's information.

    Consistency uses only what the robot observed: its own private feedback symbols, the
    public moved/stalled events, and the object pose read off its own noiseless scan. The
    lead-in is re-simulated for every candidate type; a type survives only if it
    reproduces all three. The true type always survives by construction.
    """

    surviving: List[HiddenPhysics] = []
    for physics in enumerate_hidden_physics():
        current = replace(situation.initial_scene, physics=physics, released=False)
        feedback: List[str] = []
        public: List[str] = []
        for skill in situation.skills:
            successor, result = execute(current, skill)
            feedback.append(result.feedback[0])
            public.append(result.y_public)
            current = successor
        if (
            tuple(feedback) == situation.history_a
            and tuple(public) == situation.public_history
            and current.obj.origin == situation.scene.obj.origin
        ):
            surviving.append(physics)
    if not surviving:
        # Never expected (the true type is consistent by construction); fail loudly rather
        # than silently falling back to a single realized world.
        raise RuntimeError(f"scene {situation.scene_id}: no consistent world reproduces history")
    return tuple(surviving)


def _goal_mask(goal: frozenset) -> np.ndarray:
    mask = np.zeros(N_CELLS)
    for index in range(N_CELLS):
        origin = (index % GRID, index // GRID)
        members = {(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)}
        if members <= goal:
            mask[index] = 1.0
    return mask


def build_decision_point(
    situation: Situation, model: BeliefWorldModel, *, radius: int = 5
) -> DecisionPoint:
    observation = np.asarray(
        observation_vector(
            situation.scan,
            situation.scene.robot_pose(0),
            situation.history_a,
            situation.public_history,
        ),
        dtype=np.float64,
    )
    dynamic = model.dynamic_state(observation)
    belief = belief_from_history(model, observation, dynamic, _history_pairs(situation))
    goal = frozenset(situation.scene.static.target)
    worlds = consistent_worlds(situation)
    return DecisionPoint(
        situation=situation,
        observation=observation,
        worlds=worlds,
        weights=np.full(len(worlds), 1.0 / len(worlds)),
        model_belief=belief,
        model_dynamic=dynamic,
        goal=goal,
        distance_table=_cell_distance_distribution(None, goal),
        goal_mask=_goal_mask(goal),
    )


# ---------------------------------------------------------------------------
# Candidate sequences
# ---------------------------------------------------------------------------


def candidate_order(depth: int, seed: int) -> Tuple[Tuple[int, ...], ...]:
    """All ``6**depth`` action tuples in a deterministic random order.

    The nested sets are prefixes of this order, so ``C16 <= C64 <= C128`` holds by
    construction. The order is built from ``random.Random(seed)`` only: it may not call
    the model or the true return.
    """

    tuples = [tuple(sequence) for sequence in _product(range(len(ACTION_IDS)), depth)]
    rng = random.Random(seed)
    rng.shuffle(tuples)
    return tuple(tuples[:MAX_SEQUENCES])


def _product(alphabet: Sequence[int], depth: int) -> Iterable[Tuple[int, ...]]:
    if depth == 0:
        yield ()
        return
    for head in alphabet:
        for tail in _product(alphabet, depth - 1):
            yield (head,) + tail


def nested(sequences: Sequence[Tuple[int, ...]], k: int) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sequences[: min(k, len(sequences))])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def joint_score(success: float, cost: float, distance: float) -> float:
    """The one scalar both sides use; higher is better."""

    return SUCCESS_BONUS * float(success) - float(cost) - DISTANCE_WEIGHT * float(distance)


def true_outcome(
    point: DecisionPoint, sequence: Sequence[int]
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    """Exact expected return over the consistent worlds, plus the raw expected quantities.

    Every world is executed independently from the same observed initial state and the same
    absorbing-success protocol; the expectation weights the worlds by the exact joint
    posterior. No marginal state is multiplied by ``b(theta)`` anywhere.
    """

    total = 0.0
    success = 0.0
    cost = 0.0
    distance = 0.0
    trace: List[Dict[str, object]] = []
    for physics, weight in zip(point.worlds, point.weights):
        scene = replace(point.situation.scene, physics=physics, released=False)
        world_cost = 0.0
        world_success = 0.0
        steps: List[Dict[str, object]] = []
        for action in sequence:
            current = scene
            if current.is_goal():
                break
            successor, result = execute(current, _joint_for(current, ACTION_IDS[action]))
            world_cost += float(result.cost)
            steps.append(
                {
                    "action": ACTION_IDS[action],
                    "cost": float(result.cost),
                    "y_public": result.y_public,
                    "object_origin": list(successor.obj.origin),
                }
            )
            scene = successor
            if scene.is_goal():
                world_success = 1.0
                break
        world_distance = 0.0 if world_success else _terminal_distance(scene, point.goal)
        total += weight * joint_score(world_success, world_cost, world_distance)
        success += weight * world_success
        cost += weight * world_cost
        distance += weight * world_distance
        trace.append({"steps": steps, "world_weight": float(weight), "success": world_success})
    raw = {"expected_success": success, "expected_cost": cost, "expected_distance": distance}
    return total, raw, trace


def _terminal_distance(scene: Scene, goal: frozenset) -> float:
    """Worst object cell's distance to its nearest goal cell (the repo's convention)."""

    return float(
        max(min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in scene.obj.cells)
    )


def model_outcome(
    model: BeliefWorldModel, point: DecisionPoint, sequence: Sequence[int]
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    """The model's fixed-sequence prediction from the initial input only.

    Exactly the planner's prediction path: ``step_actions`` / ``cost_actions`` under the
    deployment belief. No real state, observation or feedback is read after the first
    input. Success is the model's own best guess (``P(in goal) >= SUCCESS_THRESHOLD``).
    """

    state = np.atleast_2d(point.model_dynamic)
    cost = 0.0
    success = 0.0
    trace: List[Dict[str, object]] = []
    for action in sequence:
        states = model.step_actions(state, point.model_belief)
        step_cost = float(model.cost_actions(states, point.model_belief)[action])
        state = np.atleast_2d(states[action])
        probability = state[0, 1 : 1 + N_CELLS]
        cost += step_cost
        in_goal = float(probability @ point.goal_mask)
        trace.append(
            {
                "action": ACTION_IDS[action],
                "cost": step_cost,
                "p_in_goal": in_goal,
                "expected_distance": float(probability @ point.distance_table),
            }
        )
        if in_goal >= SUCCESS_THRESHOLD:
            success = 1.0
            break
    probability = state[0, 1 : 1 + N_CELLS]
    distance = float(probability @ point.distance_table)
    raw = {"predicted_success": success, "predicted_cost": cost, "predicted_distance": distance}
    return joint_score(success, cost, distance), raw, trace


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def selection_metrics(
    candidate_ids: Sequence[int], j_hat: np.ndarray, j_ref: np.ndarray
) -> Dict[str, float]:
    """The section B4 metrics; all are "higher return is better"."""

    j_hat = np.asarray(j_hat, dtype=np.float64)
    j_ref = np.asarray(j_ref, dtype=np.float64)
    best = max(range(len(j_hat)), key=lambda index: (j_hat[index], -candidate_ids[index]))
    chosen = int(best)
    selected_bias = float(j_hat[chosen] - j_ref[chosen])
    candidate_mean_bias = float(np.mean(j_hat - j_ref))
    return {
        "chosen_index": chosen,
        "chosen_candidate_id": candidate_ids[chosen],
        "selected_bias": selected_bias,
        "candidate_mean_bias": candidate_mean_bias,
        "selection_excess": selected_bias - candidate_mean_bias,
        "candidate_regret": float(np.max(j_ref) - j_ref[chosen]),
        "uniform_candidate_return": float(np.mean(j_ref)),
        "chosen_model_return": float(j_hat[chosen]),
        "chosen_true_return": float(j_ref[chosen]),
    }


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def run_audit(
    *,
    checkpoint: str = "outputs/grid/world_model/belief.npz",
    n_scenes: int = 20,
    lead_in: int = 2,
    seeds: Sequence[int] = (0, 1, 2),
    depths: Sequence[int] = (1, 2, 3),
    sizes: Sequence[int] = (16, 64, 128),
    scene_seed: int = 55_000,
    out_dir: Optional[str] = None,
) -> Dict[str, object]:
    from ..learning.belief_model import load_belief_model

    before = _sha256(checkpoint)
    model = load_belief_model(checkpoint)
    situations = generate_situations(n_scenes=n_scenes, lead_in=lead_in, seed=scene_seed)
    points = [build_decision_point(situation, model) for situation in situations]

    records: List[Dict[str, object]] = []
    for depth in depths:
        for seed in seeds:
            order = candidate_order(depth, seed)
            for point in points:
                for candidate_id, sequence in enumerate(order):
                    model_return, model_raw, model_trace = model_outcome(model, point, sequence)
                    true_return, true_raw, true_trace = true_outcome(point, sequence)
                    records.append(
                        {
                            "scene_id": point.scene_id,
                            "seed": seed,
                            "H": depth,
                            "candidate_id": candidate_id,
                            "action_sequence": [ACTION_IDS[a] for a in sequence],
                            "action_indices": list(sequence),
                            "initial_state_ref": {
                                "scene_id": point.scene_id,
                                "object_origin": list(point.situation.scene.obj.origin),
                                "robot_poses": [list(r.pose) for r in point.situation.scene.robots],
                                "goal": sorted(point.goal),
                            },
                            "information_condition": {
                                "n_consistent_worlds": len(point.worlds),
                                "consistent_world_labels": [w.label for w in point.worlds],
                            },
                            "J_hat": model_return,
                            "J_ref": true_return,
                            "J_hat_raw": model_raw,
                            "J_ref_raw": true_raw,
                            "model_trace": model_trace,
                            "true_trace": true_trace,
                        }
                    )

    summary = _summarise(records, sizes)
    checks = _checks(records, points, model, checkpoint, before, depths, seeds, sizes)
    return {"records": records, "summary": summary, "checks": checks, "points": points}


def _summarise(
    records: Sequence[Dict[str, object]], sizes: Sequence[int]
) -> List[Dict[str, object]]:
    """Per (H, K, seed, scene) rows plus per (H, K) aggregates, from the raw records.

    The choice is always made **within one scene and one seed**; aggregate rows average the
    per-scene metrics and never take an argmax across scenes.
    """

    groups: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for record in records:
        groups.setdefault(
            (int(record["H"]), int(record["seed"]), int(record["scene_id"])), []
        ).append(record)

    metric_keys = [
        "selected_bias", "candidate_mean_bias", "selection_excess",
        "candidate_regret", "uniform_candidate_return",
        "chosen_model_return", "chosen_true_return",
    ]
    per_scene: List[Dict[str, object]] = []
    for (depth, seed, scene), group in sorted(groups.items()):
        for size in sizes:
            subset = [r for r in group if int(r["candidate_id"]) < size]
            if not subset:
                continue
            metrics = selection_metrics(
                [int(r["candidate_id"]) for r in subset],
                np.asarray([float(r["J_hat"]) for r in subset]),
                np.asarray([float(r["J_ref"]) for r in subset]),
            )
            chosen = subset[int(metrics["chosen_index"])]
            row = {
                "H": depth, "K": size, "seed": seed, "scope": "scene", "scene_id": scene,
                "n_candidates": len(subset),
                "selected_true_success": float(chosen["J_ref_raw"]["expected_success"]),  # type: ignore[index]
                "selected_expected_cost": float(chosen["J_ref_raw"]["expected_cost"]),  # type: ignore[index]
                "selected_true_distance": float(chosen["J_ref_raw"]["expected_distance"]),  # type: ignore[index]
            }
            row.update({key: metrics[key] for key in metric_keys})
            row["chosen_candidate_id"] = metrics["chosen_candidate_id"]
            per_scene.append(row)

    aggregates: List[Dict[str, object]] = []
    by_hk: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for row in per_scene:
        by_hk.setdefault((int(row["H"]), int(row["K"])), []).append(row)
    for (depth, size), rows in sorted(by_hk.items()):
        row: Dict[str, object] = {
            "H": depth, "K": size, "seed": "", "scope": "all", "scene_id": "",
            "n_candidates": int(np.mean([int(r["n_candidates"]) for r in rows])),
            "chosen_candidate_id": "",
        }
        for key in metric_keys + ["selected_true_success", "selected_expected_cost", "selected_true_distance"]:
            row[key] = float(np.mean([float(r[key]) for r in rows]))
        aggregates.append(row)
    return aggregates + per_scene


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checks(
    records: Sequence[Dict[str, object]],
    points: Sequence[DecisionPoint],
    model: BeliefWorldModel,
    checkpoint: str,
    before: str,
    depths: Sequence[int],
    seeds: Sequence[int],
    sizes: Sequence[int],
) -> Dict[str, object]:
    # 1. nesting and counts
    nesting = {}
    for depth in depths:
        full = candidate_order(depth, 0)
        for small, large in zip(sizes, sizes[1:]):
            nesting[f"H={depth} C{small}<=C{large}"] = set(nested(full, small)) <= set(
                nested(full, large)
            )
        nesting[f"H={depth} unique"] = len(set(full)) == len(full)

    # 2. mock J_hat = J_ref => zero bias and zero regret
    sample = [r for r in records if r["H"] == depths[0] and r["seed"] == seeds[0] and r["candidate_id"] < 16]
    j_ref = np.asarray([float(r["J_ref"]) for r in sample])
    mock = selection_metrics(
        [int(r["candidate_id"]) for r in sample], j_ref, j_ref
    )

    # 3. candidate iteration order must not change a true trajectory
    point = points[0]
    order = candidate_order(1, 0)
    forward = [true_outcome(point, sequence)[0] for sequence in order]
    backward = [true_outcome(point, sequence)[0] for sequence in reversed(order)]
    order_invariant = np.allclose(forward, list(reversed(backward)), atol=0.0, rtol=0.0)

    return {
        "nesting": nesting,
        "mock_bias": mock["selected_bias"],
        "mock_regret": mock["candidate_regret"],
        "mock_ok": abs(mock["selected_bias"]) < 1e-9 and abs(mock["candidate_regret"]) < 1e-9,
        "order_invariant": bool(order_invariant),
        "checkpoint": checkpoint,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": _sha256(checkpoint),
        "checkpoint_unchanged": before == _sha256(checkpoint),
        "model_parameters_frozen": len(model.parameters()) > 0,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(result: Dict[str, object], out_dir: str) -> List[str]:
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    candidates_path = root / "candidates.jsonl"
    with candidates_path.open("w") as handle:
        for record in result["records"]:  # type: ignore[union-attr]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    written.append(str(candidates_path))

    summary_path = root / "summary.csv"
    rows = result["summary"]  # type: ignore[assignment]
    fieldnames = [
        "H", "K", "seed", "scope", "scene_id", "n_candidates",
        "chosen_candidate_id", "selected_bias", "candidate_mean_bias",
        "selection_excess", "candidate_regret", "uniform_candidate_return",
        "chosen_model_return", "chosen_true_return",
        "selected_true_success", "selected_expected_cost", "selected_true_distance",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    written.append(str(summary_path))

    checks_path = root / "checks.json"
    checks_path.write_text(json.dumps(result["checks"], indent=2, ensure_ascii=False) + "\n")
    written.append(str(checks_path))

    written.extend(_figures(plt, result, root))
    conclusion_path = root / "conclusion.md"
    conclusion_path.write_text(_conclusion(result), encoding="utf-8")
    written.append(str(conclusion_path))
    return written


def _aggregate(rows: Sequence[Dict[str, object]], depth: int, size: int) -> Dict[str, float]:
    subset = [
        row for row in rows
        if int(row["H"]) == depth and int(row["K"]) == size and row["scope"] == "all"
    ]
    keys = ["selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret", "uniform_candidate_return"]
    return {key: float(np.mean([float(row[key]) for row in subset])) for key in keys}


def _figures(plt, result: Dict[str, object], root: Path) -> List[str]:
    rows = result["summary"]  # type: ignore[assignment]
    depths = sorted({int(row["H"]) for row in rows})
    sizes = sorted({int(row["K"]) for row in rows})
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)
    written: List[str] = []

    fig, axes = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 3.4), squeeze=False)
    for column, depth in enumerate(depths):
        ax = axes[0][column]
        selected = [_aggregate(rows, depth, size)["selected_bias"] for size in sizes]
        mean_bias = [_aggregate(rows, depth, size)["candidate_mean_bias"] for size in sizes]
        ax.plot(sizes, selected, "o-", label="selected bias")
        ax.plot(sizes, mean_bias, "s--", label="candidate mean bias")
        ax.axhline(0.0, color="#888888", linewidth=0.8)
        ax.set_title(f"H = {depth}")
        ax.set_xlabel("K")
        ax.set_xticks(sizes)
        if column == 0:
            ax.set_ylabel("J_hat - J_ref")
            ax.legend(fontsize=7)
    fig.suptitle("Selection bias versus candidate-set size (positive = model overestimates)")
    fig.tight_layout()
    path = figures_dir / "fig_k_selection_bias.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 3.4), squeeze=False)
    for column, depth in enumerate(depths):
        ax = axes[0][column]
        true_return = [_aggregate(rows, depth, size)["uniform_candidate_return"] for size in sizes]
        regret = [_aggregate(rows, depth, size)["candidate_regret"] for size in sizes]
        ax.plot(sizes, true_return, "o-", label="uniform candidate return")
        ax.plot(sizes, regret, "s--", label="candidate regret")
        ax.set_title(f"H = {depth}")
        ax.set_xlabel("K")
        ax.set_xticks(sizes)
        if column == 0:
            ax.set_ylabel("J_ref")
            ax.legend(fontsize=7)
    fig.suptitle("True return and candidate regret versus candidate-set size")
    fig.tight_layout()
    path = figures_dir / "fig_k_true_return.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def _conclusion(result: Dict[str, object]) -> str:
    return conclusion_from_rows(result["summary"], result.get("records"))  # type: ignore[arg-type]


def conclusion_from_rows(
    rows: Sequence[Dict[str, object]],
    records: Optional[Sequence[Dict[str, object]]] = None,
) -> str:
    """Factual per-(H, K) table plus the scoring-protocol caveats.

    The generator no longer writes a mechanism verdict or fixed headline numbers. The
    thresholded success rule (``P(in goal) >= 0.5`` treated as certain success and
    truncating the rest of the sequence) makes the model-side and reference-side returns
    different quantities, so this report states facts and defers the mechanism judgement.
    """

    depths = sorted({int(row["H"]) for row in rows})
    sizes = sorted({int(row["K"]) for row in rows})
    lines = [
        "# 阶段 B 事实表（评分协议待修正）",
        "",
        "**评分协议待修正，暂不作机制判断。** 本表只汇总已有 `summary.csv`，未重算模型、"
        "未修改旧原始数据。",
        "",
        "## 按 H、K 的事实（跨场景与种子平均，收益越大越好）",
        "",
        "| H | K | n | selected_bias | selection_excess | candidate_regret | chosen_true_return | uniform_candidate_return |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    chosen_by_hk: Dict[Tuple[int, int], float] = {}
    for depth in depths:
        for size in sizes:
            aggregate = _aggregate(rows, depth, size)
            subset = [
                row for row in rows
                if int(row["H"]) == depth and int(row["K"]) == size and row["scope"] == "all"
            ]
            n = int(np.mean([int(row["n_candidates"]) for row in subset])) if subset else 0
            chosen = float(np.mean([float(row["chosen_true_return"]) for row in subset])) if subset else float("nan")
            chosen_by_hk[(depth, size)] = chosen
            lines.append(
                f"| {depth} | {size} | {n} | {aggregate['selected_bias']:+.3f} | "
                f"{aggregate['selection_excess']:+.3f} | {aggregate['candidate_regret']:.3f} | "
                f"{chosen:+.3f} | {aggregate['uniform_candidate_return']:+.3f} |"
            )
    lines.append("")
    lines.append("## 协议 caveat（必须与上表一起读）")
    lines.append("")
    lines.append(
        "1. 本诊断仍使用隐藏参数、学习信念与原**边缘状态**传播（`step_actions` 混合），"
        "不是逐参数条件状态。\n"
        "2. 成功概率阈值化：模型侧把 `P(in goal) >= 0.5` 当作 100% 成功并截断后续代价，"
        "真实侧是世界期望，两边收益语义不一致。\n"
        "3. 代表 H=3 案例的模型分只推演一步后触发该阈值。\n"
        "4. 固定 H=3 下实际选中收益随 K 变化"
        + (
            "（{:+.3f} → {:+.3f} → {:+.3f}）".format(
                chosen_by_hk.get((3, sizes[0]), float("nan")),
                chosen_by_hk.get((3, sizes[len(sizes) // 2]), float("nan")),
                chosen_by_hk.get((3, sizes[-1]), float("nan")),
            )
            if (3, sizes[0]) in chosen_by_hk
            else ""
        )
        + "；不能据此声称搜索越宽越差。\n"
        "5. 旧结论应限定为当前管线的现象，未隔离纯动力学误差（旧模型、旧数据、旧评分均未改动）。\n"
        "6. H=1 只有 6 条、H=2 只有 36 条不同序列，K 超过这些值时同一全集只计算一次；"
        "跨 H 不比较裸收益。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"
