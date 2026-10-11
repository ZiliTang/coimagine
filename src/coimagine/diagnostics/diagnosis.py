"""Round-20 diagnosis of the Stage-2 grid line.

Three things, in the order the task asks for them:

**A. one evaluation vocabulary.**  ``exact_reachability`` decides solvability by an
exhaustive search of the finite environment under the fixed action set and time limit,
instead of by whether some oracle happened to succeed.  ``summarise`` reports, separately
and with numerators and denominators, the full-set success rate, the mean cost over all
episodes, and the **conditional mean cost over successful episodes** (previously absent
everywhere), and names ``total cost / successes`` as its own quantity.  Episodes the team
had already solved before the measurement window -- during the lead-in, or by the common
warm-up action -- are counted and reported separately rather than silently scored as free
successes.

**B. the exact model inside the current planners.**  ``exact_adapter.ExactFactoredModel``
and ``exact_adapter.ExactBeliefModel`` implement the two learned planners' model
interfaces with the tabular dynamics, so the planner code, candidate set, scoring rule,
execution loop, termination and tie-break are byte-for-byte the ones already in the repo.
Only the model changes.

**C. an independent audit.**  ``brute_force_central_q`` recomputes the exact belief-space
action scores with a separate, memo-free implementation; ``cost_accounting`` checks that
each step's cost is charged once and that the failure cost is charged once, at truncation;
``scoring_identity_checks`` reconstructs ``plan_with_model``'s own score from its own
branch records.  ``collect_failure_traces`` records a representative boundary where a
non-productive action wins.

Everything is deterministic; every number in the report comes from a run of this module.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__
from ..grid.dynamics import FEEDBACK_SYMBOLS, execute
from ..grid.planner import (
    PHYSICS,
    ContactContext,
    DEFAULT_FAIL_COST,
    HYPOTHESES,
    HypKey,
    NullGridCodec,
    _cut_value,
    central_q_values,
    central_value,
    initial_contact_belief,
    private_posterior,
    successors_central,
    successors_codec,
)
from ..grid.sensing import ray_cast
from ..grid.state import ObjectState, RobotState, Scene, enumerate_hidden_physics
from ..learning.data import _joint_for, generate_situations
from ..learning.features import ACTION_IDS, GRID, N_CELL_BLOCKS, OBSERVATION_DIM, observation_vector
from . import exact_adapter
from .exact_adapter import ExactBeliefModel, ExactFactoredModel

#: The fixed evaluation protocol.  Same seed, scene count and warm-up as the project's
#: headline table, so the numbers here are comparable with it.
EVAL_SEED = 55_000
EVAL_SCENES = 300
ORACLE_PROBE_HORIZON = 16
FAIL_COST = DEFAULT_FAIL_COST
WARMUP_ACTION = "probe:0:E"
PHYSICS_TABLE = tuple(enumerate_hidden_physics())

_JOINTS = {action: _joint_for(None, action) for action in ACTION_IDS}
_WARMUP_INDEX = ACTION_IDS.index(WARMUP_ACTION)


# ===========================================================================
# A. one evaluation vocabulary
# ===========================================================================


def _canonical(scene: Scene) -> Scene:
    """Drop the heading, which the dynamics ignore, so the reachability search merges states."""

    return Scene(
        scene.static,
        scene.obj,
        tuple(RobotState(r.robot_id, r.pose, None) for r in scene.robots),
        scene.physics,
        scene.released,
    )


def _state_key(scene: Scene) -> Tuple[object, ...]:
    return (
        scene.obj.origin,
        scene.released,
        tuple((r.robot_id, r.pose) for r in scene.robots),
    )


def exact_reachability(
    scene: Scene, *, horizon: int, action_ids: Sequence[str] = ACTION_IDS
) -> Optional[int]:
    """Fewest actions that put the object in the goal, or ``None`` if it is unreachable.

    An exhaustive breadth-first search over the **finite** environment (object origin x
    released flag x the two robot poses), under the fixed action set and the same horizon
    the policy is given.  This is the "exact reachability" definition of solvability; the
    dynamics are deterministic, so the search is exact.
    """

    start = _canonical(scene)
    if start.is_goal():
        return 0
    seen = {_state_key(start)}
    frontier: deque = deque([(start, 0)])
    while frontier:
        current, depth = frontier.popleft()
        if depth >= horizon:
            continue
        for action in action_ids:
            successor, _result = execute(current, _JOINTS[action])
            if successor.is_goal():
                return depth + 1
            key = _state_key(successor)
            if key in seen:
                continue
            seen.add(key)
            frontier.append((successor, depth + 1))
    return None


@dataclass
class EpisodeRecord:
    scene_id: int
    policy: str
    physics: str
    success: int
    action_cost: float
    steps: int
    #: Fewest actions to the goal by exact search from the measured start, or ``None``.
    reachable_steps: Optional[int] = None
    exact_reachable: int = 0
    #: The full-information oracle's observed success at the long probe horizon (the
    #: project's historical definition, kept only so the two subsets can be compared).
    oracle_probe_success: int = 0
    pre_solved: int = 0
    first_action: Optional[str] = None


@dataclass
class MetricSummary:
    name: str
    n: int
    successes: int
    success_rate: float
    mean_action_cost_all: float
    mean_total_cost_all: float
    mean_total_cost_success: Optional[float]
    total_cost_per_success: Optional[float]
    mean_steps: float


def summarise(records: Sequence[EpisodeRecord], name: str) -> MetricSummary:
    """The one metric vocabulary.  Every field is named for exactly what it averages."""

    n = len(records)
    if n == 0:
        return MetricSummary(name, 0, 0, float("nan"), float("nan"), float("nan"), None, None, float("nan"))
    successes = sum(r.success for r in records)
    action_costs = [r.action_cost for r in records]
    total_costs = [r.action_cost + (0.0 if r.success else FAIL_COST) for r in records]
    successful_costs = [
        r.action_cost + (0.0 if r.success else FAIL_COST)
        for r in records
        if r.success
    ]
    total = sum(total_costs)
    return MetricSummary(
        name=name,
        n=n,
        successes=successes,
        success_rate=successes / n,
        mean_action_cost_all=float(np.mean(action_costs)),
        mean_total_cost_all=total / n,
        mean_total_cost_success=(float(np.mean(successful_costs)) if successes else None),
        total_cost_per_success=(total / successes if successes else None),
        mean_steps=float(np.mean([r.steps for r in records])),
    )


def summarise_table(records: Sequence[EpisodeRecord], groups: Dict[str, Callable[[EpisodeRecord], bool]]) -> List[MetricSummary]:
    return [summarise([r for r in records if predicate(r)], name) for name, predicate in groups.items()]


# ---------------------------------------------------------------------------
# Episode drivers
# ---------------------------------------------------------------------------


def _measurement_starts(
    *, n_scenes: int, seed: int, warmup: Optional[str]
) -> Tuple[List[Dict[str, object]], int, int]:
    """The scenes the policies are actually scored on, plus the two exclusion counts.

    Mirrors the project's headline protocol: drop an episode that is already solved, and
    drop one the common warm-up action solves.  The counts are returned, not discarded.
    """

    situations = generate_situations(
        n_scenes=n_scenes, lead_in=0, seed=seed, physics_pool=PHYSICS_TABLE
    )
    starts: List[Dict[str, object]] = []
    pre_solved = 0
    warm_solved = 0
    for situation in situations:
        scene = situation.scene
        record = {
            "scene_id": situation.scene_id,
            "physics": situation.physics,
            "scene": scene,
            "warmup": None,
        }
        if scene.is_goal():
            pre_solved += 1
            continue
        if warmup is not None:
            successor, result = execute(scene, _JOINTS[warmup])
            if successor.is_goal():
                warm_solved += 1
                continue
            record["scene"] = successor
            record["warmup"] = (
                ACTION_IDS.index(warmup),
                FEEDBACK_SYMBOLS.index(result.feedback[0]),
            )
        starts.append(record)
    return starts, pre_solved, warm_solved


def run_factored_policy(
    policy: str,
    *,
    model=None,
    exact_model: Optional[ExactFactoredModel] = None,
    model_factory: Optional[Callable[[Scene], object]] = None,
    n_scenes: int = EVAL_SCENES,
    horizon: int = 8,
    depth: int = 2,
    beam: int = 1,
    seed: int = EVAL_SEED,
    warmup: Optional[str] = WARMUP_ACTION,
    oracle_probe: bool = True,
) -> List[EpisodeRecord]:
    """Run one policy on the fixed set, using the repo's own episodes drivers unchanged.

    ``policy="search"`` drives ``belief_probe.run_search_episode`` and ``policy="lookahead"``
    drives ``belief_probe.run_belief_episode``; the model is either the learned one, the
    exact one or a hybrid supplied by ``model_factory``.
    """

    from .belief_probe import run_belief_episode, run_search_episode
    from .rollout_probe import run_episode

    starts, pre_solved, warm_solved = _measurement_starts(
        n_scenes=n_scenes, seed=seed, warmup=warmup
    )
    n_physics = len(PHYSICS)
    records: List[EpisodeRecord] = []
    for entry in starts:
        scene = entry["scene"]
        physics = entry["physics"]
        active = (
            model_factory(scene)
            if model_factory is not None
            else (exact_model if exact_model is not None else model)
        )
        belief = np.full((1, n_physics), 1.0 / n_physics)
        driver = run_search_episode if policy == "search" else run_belief_episode
        kwargs = dict(horizon=horizon, warmup=entry["warmup"])
        if policy == "search":
            kwargs.update(depth=depth, beam=beam)
        else:
            kwargs.update(depth=depth)
        success, cost, steps = driver(active, scene, belief, **kwargs)
        reachable = exact_reachability(scene, horizon=horizon)
        oracle_success = 0
        if oracle_probe:
            oracle_success, _c, _s = run_episode(
                scene, "oracle_true_state", None,
                horizon=ORACLE_PROBE_HORIZON, rng=np.random.default_rng(0),
            )
        records.append(
            EpisodeRecord(
                scene_id=int(entry["scene_id"]),
                policy=policy,
                physics=physics.label,
                success=int(success),
                action_cost=float(cost),
                steps=int(steps),
                reachable_steps=reachable,
                exact_reachable=int(reachable is not None),
                oracle_probe_success=int(oracle_success),
            )
        )
    for record in records:
        record.pre_solved = 0
    # attach the exclusion counts once, on the first record, for the report
    if records:
        records[0].policy = policy
    _EXCLUSIONS[(policy, seed, n_scenes)] = (pre_solved, warm_solved)
    return records


_EXCLUSIONS: Dict[Tuple[str, int, int], Tuple[int, int]] = {}


# ---------------------------------------------------------------------------
# B. the exact model inside the current planner (lookahead line)
# ---------------------------------------------------------------------------


def run_lookahead_policy(
    *,
    model=None,
    model_factory=None,
    name: str = "lookahead",
    n_scenes: int = EVAL_SCENES,
    horizon: int = 8,
    depth: int = 2,
    seed: int = EVAL_SEED,
    warmup: Optional[str] = WARMUP_ACTION,
    radius: int = 5,
    oracle_probe: bool = False,
) -> List[EpisodeRecord]:
    """The ``plan_with_model`` planner on the same fixed set, with a swappable model.

    The loop is ``world_model_probe.run_closed_loop``'s loop; the planner call inside it is
    ``world_model_probe.plan_with_model`` unchanged.  ``model_factory`` is rebuilt at every
    boundary from the deployment-legal context for the exact adapter.
    """

    from .rollout_probe import run_episode
    from .world_model_probe import _goal_cells, plan_with_model

    situations = generate_situations(
        n_scenes=n_scenes, lead_in=0, seed=seed, physics_pool=PHYSICS_TABLE
    )
    records: List[EpisodeRecord] = []
    for situation in situations:
        scene = situation.scene
        if scene.is_goal():
            continue
        goal = _goal_cells(scene)
        history_a: List[str] = []
        public_history: List[str] = []
        actions: List[str] = []
        if warmup is not None:
            successor, result = execute(scene, _JOINTS[warmup])
            if successor.is_goal():
                continue
            scene = successor
            actions.append(warmup)
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
        initial_robots = scene.robots
        measurement_start = scene
        first_action: Optional[str] = None
        total = 0.0
        steps = 0
        for _ in range(horizon):
            if scene.is_goal():
                break
            observation = np.asarray(
                observation_vector(
                    ray_cast(scene, 0, radius=radius),
                    scene.robot_pose(0),
                    tuple(history_a),
                    tuple(public_history),
                )
            )
            active = model
            if model_factory is not None:
                active = model_factory(
                    {
                        "observation": observation,
                        "static": scene.static,
                        "robots": scene.robots,
                        "initial_robots": initial_robots,
                        "actions": tuple(actions),
                        "history_a": tuple(history_a),
                        "public_history": tuple(public_history),
                        "radius": radius,
                    }
                )
            action_index, _branches = plan_with_model(active, observation, goal, depth=depth)
            action_id = ACTION_IDS[action_index]
            if first_action is None:
                first_action = action_id
            successor, result = execute(scene, _JOINTS[action_id])
            total += float(result.cost)
            steps += 1
            actions.append(action_id)
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            scene = successor
        reachable = exact_reachability(measurement_start, horizon=horizon)
        oracle_success = 0
        if oracle_probe:
            oracle_success, _c, _s = run_episode(
                scene, "oracle_true_state", None,
                horizon=ORACLE_PROBE_HORIZON, rng=np.random.default_rng(0),
            )
        records.append(
            EpisodeRecord(
                scene_id=situation.scene_id,
                policy=name,
                physics=situation.physics.label,
                success=int(scene.is_goal()),
                action_cost=total,
                steps=steps,
                reachable_steps=reachable,
                exact_reachable=int(reachable is not None),
                oracle_probe_success=int(oracle_success),
                first_action=first_action,
            )
        )
    return records


def _start_scene(situation, warmup: Optional[str]) -> Scene:
    scene = situation.scene
    if warmup is not None:
        scene, _result = execute(scene, _JOINTS[warmup])
    return scene


# ===========================================================================
# C. independent audit of the search, the scoring and the belief
# ===========================================================================


def brute_force_central_q(
    ctx: ContactContext,
    belief: Dict[Tuple[object, ...], float],
    candidates,
    depth: int,
    *,
    fail_cost: float = FAIL_COST,
) -> Dict[str, float]:
    """Action scores recomputed from scratch, with no memoisation and no shared cache.

    Same strategy class and same cut value as ``central_q_values``, but written as a plain
    recursion that calls ``execute`` directly.  A disagreement is a fact about the code, not
    about the planner's approximation.
    """

    def backup(entries: Dict[Tuple[object, ...], float], left: int) -> float:
        if not entries:
            return fail_cost
        if all(ObjectState(key[0]).within(ctx.static.target) for key in entries):
            return 0.0
        if left <= 0:
            risk = 0.0
            progress = 0.0
            for key, weight in entries.items():
                cells = ObjectState(key[0]).cells
                if cells <= ctx.static.target:
                    continue
                risk += weight
                progress += weight * max(
                    min(abs(c[0] - t[0]) + abs(c[1] - t[1]) for t in ctx.static.target)
                    for c in cells
                )
            return fail_cost * risk + 3.0 * progress
        best = float("inf")
        for candidate in candidates.candidates:
            scene_cost = 0.0
            buckets: Dict[Tuple[str, str, str], Dict[Tuple[object, ...], float]] = {}
            for (origin, robots, hypothesis), weight in entries.items():
                scene = Scene(
                    ctx.static,
                    ObjectState(origin),
                    robots,
                    PHYSICS[hypothesis[0]],
                    hypothesis[1],
                )
                successor, result = execute(scene, candidate.joint)
                scene_cost += weight * float(result.cost)
                key = (result.feedback[0], result.feedback[1], result.y_public)
                buckets.setdefault(key, {})
                child = (successor.obj.origin, successor.robots, (hypothesis[0], successor.released))
                buckets[key][child] = buckets[key].get(child, 0.0) + weight
            total = scene_cost
            for weights in buckets.values():
                branch_total = sum(weights.values())
                if branch_total <= 0.0:
                    continue
                child = {k: v / branch_total for k, v in weights.items()}
                total += branch_total * backup(child, left - 1)
            if total < best:
                best = total
        return best

    out: Dict[str, float] = {}
    for candidate in candidates.candidates:
        scene_cost = 0.0
        buckets: Dict[Tuple[str, str, str], Dict[Tuple[object, ...], float]] = {}
        for (origin, robots, hypothesis), weight in belief.items():
            scene = Scene(
                ctx.static, ObjectState(origin), robots, PHYSICS[hypothesis[0]], hypothesis[1]
            )
            successor, result = execute(scene, candidate.joint)
            scene_cost += weight * float(result.cost)
            key = (result.feedback[0], result.feedback[1], result.y_public)
            buckets.setdefault(key, {})
            child = (successor.obj.origin, successor.robots, (hypothesis[0], successor.released))
            buckets[key][child] = buckets[key].get(child, 0.0) + weight
        total = scene_cost
        for weights in buckets.values():
            branch_total = sum(weights.values())
            if branch_total <= 0.0:
                continue
            child = {k: v / branch_total for k, v in weights.items()}
            total += branch_total * backup(child, depth - 1)
        out[candidate.candidate_id] = total
    return out


def audit_exact_backup(*, depth: int = 2, seed: int = 7) -> Dict[str, object]:
    """Compare ``central_q_values`` with the independent recursion on a small scene set."""

    from .rollout_probe import _belief_values
    from ..grid.planner import contact_candidates

    rng = np.random.default_rng(seed)
    worst = 0.0
    checked = 0
    disagreements: List[Dict[str, object]] = []
    situations = generate_situations(n_scenes=6, lead_in=1, seed=seed, physics_pool=PHYSICS_TABLE)
    for situation in situations:
        scene = situation.scene
        ctx = ContactContext(scene.static, scene.obj.origin, scene.robots)
        candidates = contact_candidates(ctx)
        # an explicit, hand-built belief over four hypotheses consistent with the scene
        entries = {}
        for index in (0, 3, 8, 15):
            entries[(scene.obj.origin, scene.robots, (index, False))] = 0.25
        reference = central_q_values(ctx, entries, candidates, depth, fail_cost=FAIL_COST)
        independent = brute_force_central_q(ctx, entries, candidates, depth)
        for candidate_id, value in reference.items():
            checked += 1
            gap = abs(value - independent[candidate_id])
            worst = max(worst, gap)
            if gap > 1e-9:
                disagreements.append(
                    {
                        "scene_id": situation.scene_id,
                        "candidate": candidate_id,
                        "central": value,
                        "independent": independent[candidate_id],
                    }
                )
    return {
        "checked_candidates": checked,
        "max_abs_gap": worst,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:10],
        "depth": depth,
    }


def cost_accounting() -> Dict[str, object]:
    """Charge each step's cost once, and the failure cost once, at truncation."""

    from .rollout_probe import run_episode

    # A scene whose object is one push from the goal and whose physics let it move: the
    # full-information planner solves it in one step, so any extra cost is a bug.
    from ..learning.data import probe_scene

    physics = [p for p in PHYSICS if p.resistance == 1 and p.constraint == "free" and p.traction == ("grip", "grip")][0]
    scene = probe_scene((2, 0), physics)
    ctx = ContactContext(scene.static, scene.obj.origin, scene.robots)
    start_ok = exact_reachability(scene, horizon=4)
    success, cost, steps = run_episode(
        scene, "oracle_true_state", None, horizon=4, rng=np.random.default_rng(0)
    )
    # Truncation: a scene that cannot be solved must be charged exactly one failure cost.
    stuck = [p for p in PHYSICS if p.traction == ("slip", "slip")][0]
    stuck_scene = probe_scene((1, 1), stuck)
    _s2, cost2, steps2 = run_episode(
        stuck_scene, "oracle_true_state", None, horizon=3, rng=np.random.default_rng(0)
    )
    return {
        "solvable_start": {
            "reachable_steps": start_ok,
            "oracle_success": int(success),
            "action_cost": cost,
            "steps": steps,
            "cost_equals_steps": abs(cost - steps * 1.0) < 1e-9,
        },
        "truncated_start": {
            "physics": stuck.label,
            "oracle_success": int(_s2),
            "action_cost": cost2,
            "steps": steps2,
            "horizon": 3,
            "action_cost_le_horizon": cost2 <= steps2 + 1e-9,
        },
    }


@dataclass
class ScoringIdentity:
    """``plan_with_model``'s own branch record, recomputed."""

    action: str
    score: float
    first_cost: float
    continuation_cost: float
    continuation_distance: float
    terminal_distance: float
    terminal_cost: float
    progress_weight: float
    reconstructed: float
    #: What the pre-fix formula would have added a second time.
    old_formula_duplication: float


def scoring_identity_checks(*, progress_weight: float = 3.0, fail_cost: float = FAIL_COST) -> Dict[str, object]:
    """Reconstruct ``plan_with_model``'s score from its own returned components.

    The documented formula is ``cost1 + (cost2 + w * dist2 + terminal)`` at depth 2 and
    ``cost1 + w * dist1 + terminal`` at depth 1.  The check verifies the returned score
    matches its own components and, for the record, reports the extra ``w * dist`` the
    pre-fix formula added at depth 2 (the continuation was selected under ``cost + w * dist``
    and then ``w * dist`` was added again when scoring the first action).
    """

    from .world_model_probe import plan_with_model

    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})

    def make(cost: np.ndarray, cell: np.ndarray):
        class Stub:
            def _step(self, action: int) -> Dict[str, np.ndarray]:
                logits = np.full((1, 36), -30.0)
                logits[0, cell[action]] = 0.0
                return {
                    "cost": np.asarray([cost[action]]),
                    "pose_logits": logits,
                    "moved_logit": np.asarray([0.0]),
                    "observation": np.zeros((1, OBSERVATION_DIM)),
                }

            def predict_next(self, observation, action):
                return self._step(action)

            def continue_from(self, predicted, action):
                return self._step(action)

        return Stub()

    cost = np.asarray([1.0, 1.0, 1.0, 0.3, 1.0, 0.0])
    # Every candidate leaves the object in columns 0-2, so the cut keeps a non-zero
    # distance term and any duplication is visible.
    cell = np.asarray([0, 6, 12, 1, 7, 13])
    _chosen, branches = plan_with_model(
        make(cost, cell), np.zeros(OBSERVATION_DIM), goal, depth=2,
        progress_weight=progress_weight, fail_cost=fail_cost,
    )
    rows = []
    for branch in branches:
        action = int(branch["action_index"])
        first_cost = float(branch["first_cost"])
        terminal_distance = float(branch["expected_distance"])
        terminal_cost = 0.0 if terminal_distance < 0.5 else fail_cost
        if branch["continuation"] is None:
            continuation_cost = 0.0
            continuation_distance = terminal_distance
            reconstructed = first_cost + progress_weight * terminal_distance + terminal_cost
            duplication = 0.0
        else:
            cont = int(branch["continuation"])
            continuation_cost = float(cost[cont])
            continuation_distance = _stub_distance(int(cell[cont]), goal)
            reconstructed = (
                first_cost
                + continuation_cost
                + progress_weight * continuation_distance
                + terminal_cost
            )
            duplication = progress_weight * continuation_distance
        rows.append(
            asdict(
                ScoringIdentity(
                    action=ACTION_IDS[action],
                    score=float(branch["score"]),
                    first_cost=first_cost,
                    continuation_cost=continuation_cost,
                    continuation_distance=continuation_distance,
                    terminal_distance=terminal_distance,
                    terminal_cost=terminal_cost,
                    progress_weight=progress_weight,
                    reconstructed=reconstructed,
                    old_formula_duplication=duplication,
                )
            )
        )
    worst = max(abs(row["score"] - row["reconstructed"]) for row in rows)
    return {
        "rows": rows,
        "max_abs_gap_report_vs_reconstruction": worst,
        "old_formula_extra_term_per_action": {
            row["action"]: row["old_formula_duplication"] for row in rows
        },
        "depths_differ": {
            "depth_1_formula": "cost1 + w*dist1 + terminal",
            "depth_2_formula_before_fix": "cost1 + (cost2 + w*dist2) + w*dist2 + terminal",
            "depth_2_formula_after_fix": "cost1 + (cost2 + w*dist2 + terminal)",
        },
    }


def _stub_distance(cell_index: int, goal: frozenset) -> float:
    """The planner's own worst-cell distance for a one-hot pose at ``cell_index``.

    Same formula as the corrected ``world_model_probe._worst_cell_distance`` and the exact
    planner's cut: the **maximum over the object's cells of the minimum distance to a goal
    cell**.  A naive minimum over object cells returns 0 as soon as the object merely
    overlaps the goal, and the pre-fix planner used the other inversion (minimum over cells
    of the distance to the farthest goal cell), which is not a distance to the goal at all.
    """

    origin = (cell_index % GRID, cell_index // GRID)
    cells = [(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)]
    if all(cell in goal for cell in cells):
        return 0.0
    return float(max(min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in cells))


def protocol_checks() -> Dict[str, object]:
    """Branch probabilities normalise; theta is constant and released is correlated with it."""

    from ..grid.planner import contact_candidates, successors_codec

    start = _measurement_starts(n_scenes=1, seed=1, warmup=None)[0][0]["scene"]
    ctx = ContactContext(
        static=start.static,
        origin=(2, 2),
        robots=(
            RobotState(0, (1, 2), None),
            RobotState(1, (1, 3), None),
        ),
    )
    candidates = contact_candidates(ctx)
    belief = initial_contact_belief(ctx)
    normalisation = []
    codec = NullGridCodec()
    for candidate in candidates.candidates:
        _immediate, branches = successors_codec(
            ctx, belief, candidate.joint, codec, (), candidates
        )
        normalisation.append(sum(prob for prob, _child in branches.values()))
        for _key, (_prob, child) in branches.items():
            for (_origin, _robots, hypothesis, _ha, _hb) in child:
                assert hypothesis[0] in range(len(PHYSICS))

    # theta constant along an imagined release-then-push: the correlation
    # "gripped and released" must survive, not be recombined from independent marginals.
    latched_grip = (0, False)
    for index, physics in enumerate(PHYSICS):
        if physics.constraint == "constrained" and physics.traction[0] == "grip":
            latched_grip = (index, False)
            break
    entries = {(ctx.origin, ctx.robots, latched_grip): 1.0}
    release = next(c for c in candidates.candidates if c.candidate_id == "release:0:E")
    _cost, branches = successors_central(ctx, entries, release.joint)
    released_mass = 0.0
    physics_seen = set()
    total_mass = 0.0
    for _key, (_prob, child) in branches.items():
        for (_origin, _robots, hypothesis), weight in child.items():
            physics_seen.add(hypothesis[0])
            total_mass += weight
            if hypothesis[1]:
                released_mass += weight
    return {
        "branch_probability_min": min(normalisation),
        "branch_probability_max": max(normalisation),
        "branches_normalised": all(abs(v - 1.0) < 1e-9 for v in normalisation),
        "release_branches": len(branches),
        "release_child_mass": total_mass,
        "release_physics_indices": sorted(physics_seen),
        "theta_constant_along_branches": len(physics_seen) == 1,
        "released_mass_after_release": released_mass,
    }


def collect_failure_traces(
    *, n_scenes: int = 60, seed: int = EVAL_SEED, depth: int = 2, beam: int = 1, limit: int = 3
) -> List[Dict[str, object]]:
    """A representative boundary where the exact model's planner prefers a non-productive action.

    Every candidate's score, first cost, expected distance, release probability and the
    feedback branches that produced it are recorded, so the choice can be read off directly.
    """

    from .belief_probe import plan_with_search

    starts, _pre, _warm = _measurement_starts(
        n_scenes=n_scenes, seed=seed, warmup=WARMUP_ACTION
    )
    traces: List[Dict[str, object]] = []
    for entry in starts:
        scene = entry["scene"]
        if exact_reachability(scene, horizon=8) is None:
            continue
        model = ExactFactoredModel(static=scene.static)
        history_a: List[str] = []
        public_history: List[str] = []
        warm_action, warm_symbol = entry["warmup"]
        belief = np.full((1, len(PHYSICS)), 1.0 / len(PHYSICS))
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        dynamic = model.dynamic_state(observation)
        belief = model.update_belief(belief, dynamic, warm_action, warm_symbol)
        carried = model.step(dynamic, warm_action, belief)
        history_a.append(FEEDBACK_SYMBOLS[warm_symbol])
        current = scene
        for _ in range(4):
            if current.is_goal():
                break
            observation = np.asarray(
                observation_vector(
                    ray_cast(current, 0, radius=5), current.robot_pose(0),
                    tuple(history_a), tuple(public_history),
                )
            )
            observed = model.dynamic_state(observation)
            from .belief_probe import fuse_hidden_state

            dynamic = fuse_hidden_state(observed, carried)
            chosen, branches = plan_with_search(
                model, observation, frozenset(current.static.target),
                belief=belief, dynamic=dynamic, depth=depth, beam=beam,
            )
            action_id = ACTION_IDS[chosen]
            if action_id in ("wait", "probe:0:E") and not current.is_goal():
                traces.append(
                    {
                        "scene_id": entry["scene_id"],
                        "physics": entry["physics"].label,
                        "origin": current.obj.origin,
                        "robots": [r.pose for r in current.robots],
                        "chosen": action_id,
                        "branches": [
                            {
                                "action": b["action"],
                                "score": round(float(b["score"]), 6),
                                "first_cost": round(float(b["first_cost"]), 6),
                                "expected_distance": round(float(b["expected_distance"]), 6),
                                "released_probability": round(float(b["released_probability"]), 6),
                                "symbol_probabilities": [
                                    round(float(p), 6) for p in b["symbol_probabilities"]
                                ],
                            }
                            for b in sorted(branches, key=lambda b: b["score"])
                        ],
                    }
                )
                if len(traces) >= limit:
                    return traces
            successor, result = execute(current, _JOINTS[action_id])
            symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
            belief = model.update_belief(belief, dynamic, chosen, symbol)
            carried = model.step(dynamic, chosen, belief)
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            current = successor
    return traces


def lookahead_depth_check(*, checkpoint: str = "outputs/grid/world_model/latent_released.npz") -> Dict[str, object]:
    """Why depth does nothing in ``plan_with_model`` when ``wait`` really is a no-op.

    ``wait`` costs zero and, for a model that represents it exactly, leaves the imagined
    state unchanged.  Under the planner's own value it is therefore never worse than any
    other continuation, and a depth-2 score collapses algebraically onto the depth-1 score.
    The check measures this on a stub model whose ``wait`` is the identity, and on the exact
    tabular adapter whose ``wait`` transition really is the identity.  The learned checkpoint
    is reported as a contrast: its learned ``wait`` is not exactly the identity, so depth 1
    and depth 2 do differ there -- but for a different reason than "the search looked deeper".
    """

    from .world_model_probe import _goal_cells, plan_with_model

    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})
    cost = np.asarray([1.0, 1.0, 1.0, 0.3, 1.0, 0.0])
    cell = np.asarray([0, 6, 12, 1, 7, 13])
    wait_index = ACTION_IDS.index("wait")

    class Stub:
        """A model whose ``wait`` really is the identity and whose other continuations are worse.

        Every non-``wait`` continuation is given a strictly higher cost at the same state, so
        ``wait`` is the unique argmin of the continuation value and there is no way for the
        depth-2 search to improve on depth 1.
        """

        def _step(self, action: int) -> Dict[str, np.ndarray]:
            logits = np.full((1, 36), -30.0)
            logits[0, cell[action]] = 0.0
            return {
                "cost": np.asarray([cost[action]]),
                "pose_logits": logits,
                "moved_logit": np.asarray([0.0]),
                "observation": np.zeros((1, OBSERVATION_DIM)),
            }

        def predict_next(self, observation, action):
            return self._step(action)

        def continue_from(self, predicted, action):
            kept = dict(predicted)
            kept["cost"] = np.asarray(
                [0.0 if action == wait_index else float(cost[action]) + 5.0]
            )
            return kept

    _chosen1, branches1 = plan_with_model(Stub(), np.zeros(OBSERVATION_DIM), goal, depth=1)
    _chosen2, branches2 = plan_with_model(Stub(), np.zeros(OBSERVATION_DIM), goal, depth=2)

    output: Dict[str, object] = {
        "wait_action_index": wait_index,
        "stub_identity_wait": {
            "chosen_continuations": [branch["continuation"] for branch in branches2],
            "all_continuations_are_wait": all(
                branch["continuation"] == wait_index for branch in branches2
            ),
            "depth_1_scores": [round(float(b["score"]), 6) for b in branches1],
            "depth_2_scores": [round(float(b["score"]), 6) for b in branches2],
            "depth_1_and_2_identical": all(
                abs(branches1[i]["score"] - branches2[i]["score"]) < 1e-9
                for i in range(len(branches1))
            ),
        },
    }

    # The exact adapter, where `wait` is the identity by construction.
    from .exact_adapter import exact_model_factory

    starts, _pre, _warm = _measurement_starts(
        n_scenes=30, seed=EVAL_SEED, warmup=WARMUP_ACTION
    )
    per_depth: Dict[int, Dict[int, Optional[str]]] = {1: {}, 2: {}}
    for depth in (1, 2):
        records = run_lookahead_policy(
            model_factory=exact_model_factory(condition_on_feedback=True),
            name=f"exact-d{depth}", n_scenes=30, horizon=8, depth=depth,
        )
        for record in records:
            per_depth[depth][record.scene_id] = record.first_action
    comparable = [
        (per_depth[1][scene_id], per_depth[2][scene_id])
        for scene_id in per_depth[1]
        if scene_id in per_depth[2]
        and per_depth[1][scene_id] is not None
        and per_depth[2][scene_id] is not None
    ]
    output["exact_adapter"] = {
        "decision_points": len(comparable),
        "depth_1_and_2_same_first_action": sum(1 for one, two in comparable if one == two),
    }

    # Contrast: the learned checkpoint's `wait` is learned, not the identity.
    from ..learning.world_model import load_world_model

    model = load_world_model(checkpoint)
    same_chosen = 0
    wait_continuations = 0
    total = 0
    for entry in starts:
        scene = entry["scene"]
        observation = np.asarray(
            observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
        )
        first_choice, _b1 = plan_with_model(model, observation, _goal_cells(scene), depth=1)
        second_choice, b2 = plan_with_model(model, observation, _goal_cells(scene), depth=2)
        same_chosen += int(first_choice == second_choice)
        wait_continuations += sum(
            1 for branch in b2 if branch["continuation"] == wait_index
        )
        total += 1
    output["learned_checkpoint"] = {
        "checkpoint": checkpoint,
        "decision_points": total,
        "depth_1_and_2_same_first_action": same_chosen,
        "branches_whose_best_continuation_is_wait": f"{wait_continuations}/{total * len(ACTION_IDS)}",
    }
    return output


def distance_heuristic_check() -> Dict[str, object]:
    """The learned lookahead planner's distance-to-goal, before and after the fix.

    The pre-fix ``world_model_probe._worst_cell_distance`` computed the minimum over the
    object's cells of the distance to the *farthest* goal cell.  For an object in column 3
    with the goal on columns 4-5 -- one push away -- that reads 5-6 instead of 1, and it
    varies with the object's row even though the goal spans every row.  The corrected form
    is the exact planner's own cut heuristic, ``max over cells of min over goal``.
    """

    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})
    from .world_model_probe import _worst_cell_distance as corrected

    rows = []
    for origin in ((1, 0), (3, 0), (3, 2), (0, 3), (2, 3), (4, 1)):
        cells = [(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)]
        pre_fix = float(
            min(max(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in cells)
        )
        rows.append(
            {
                "origin": origin,
                "one_push_from_goal": origin[0] == 3,
                "pre_fix_value": pre_fix,
                "corrected_value": corrected(origin, goal),
                "fully_in_goal": all(cell in goal for cell in cells),
            }
        )
    return {
        "definition": "max over object cells c of (min over goal cells g of manhattan(c, g))",
        "rows": rows,
    }


def distance_sign_check() -> Dict[str, object]:
    """Hand-worked example for the signed distance error used in the report.

    Convention: ``predicted_distance - true_distance``, where the distance is the planner's
    worst-cell distance (min over goal cells of the max Manhattan distance from an object
    cell).  Positive means the model predicted the object **further** from the goal than it
    really was -- pessimistic -- not "closer".  Both signs are computed so the convention
    cannot be read backwards.
    """

    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})
    # Object predicted at origin (1, 0): cells {(1,0),(2,0),(1,1),(2,1)}, worst cell x=2.
    predicted = _stub_distance(1, goal)
    # Object truly at origin (3, 0): cells {(3,0),(4,0),(3,1),(4,1)}, worst cell x=3.
    true = _stub_distance(3, goal)
    return {
        "formula": "signed_error = predicted_worst_cell_distance - true_worst_cell_distance",
        "distance_definition": (
            "max over object cells c of (min over goal cells g of manhattan(c, g))"
        ),
        "positive_case": {
            "predicted_origin": (1, 0),
            "true_origin": (3, 0),
            "predicted_distance": predicted,
            "true_distance": true,
            "signed_error": predicted - true,
            "meaning": "model thought the object was further than it was (pessimistic)",
        },
        "negative_case": {
            "predicted_origin": (3, 0),
            "true_origin": (1, 0),
            "predicted_distance": true,
            "true_distance": predicted,
            "signed_error": true - predicted,
            "meaning": "model thought the object was closer than it was (optimistic)",
        },
    }


# ===========================================================================
# D. module replacement on the factored line
# ===========================================================================


def run_module_replacement(
    *,
    n_scenes: int = 100,
    horizon: int = 8,
    depth: int = 2,
    beam: int = 1,
    seed: int = EVAL_SEED,
    checkpoint: str = "outputs/grid/world_model/belief.npz",
    record_episodes: bool = False,
) -> List[Dict[str, object]]:
    """Exact/learned transition crossed with exact/learned observation.

    The four cells use the same planner, the same scenes and the same depth.  The two mixed
    cells are only meaningful because ``BeliefWorldModel`` and ``ExactFactoredModel`` share
    the state layout; that is checked by the run itself (a semantic mismatch raises).

    ``record_episodes`` adds the per-episode records (scene id, success, cost, reachable),
    which is what a paired trace needs; the aggregate rows are identical either way.
    """

    from ..learning.belief_model import load_belief_model
    from .exact_adapter import HybridFactoredModel

    learned = load_belief_model(checkpoint)
    rows: List[Dict[str, object]] = []
    conditions = {
        "exact_transition+exact_observation": lambda scene: ExactFactoredModel(static=scene.static),
        "learned_transition+learned_observation": lambda scene: learned,
        "exact_transition+learned_observation": lambda scene: HybridFactoredModel(
            ExactFactoredModel(static=scene.static), learned
        ),
        "learned_transition+exact_observation": lambda scene: HybridFactoredModel(
            learned, ExactFactoredModel(static=scene.static)
        ),
    }
    for name, factory in conditions.items():
        try:
            records = run_factored_policy(
                "search", model_factory=factory, n_scenes=n_scenes, horizon=horizon,
                depth=depth, beam=beam, seed=seed, oracle_probe=False,
            )
        except Exception as error:  # pragma: no cover - reported, not hidden
            rows.append({"condition": name, "status": f"could not run: {type(error).__name__}: {error}"})
            continue
        subset = [r for r in records if r.exact_reachable]
        row: Dict[str, object] = {
            "condition": name,
            "status": "ok",
            "checkpoint": (
                None if "learned" not in name else checkpoint
            ),
            "all": asdict(summarise(records, name)),
            "exact_reachable": asdict(summarise(subset, name)),
        }
        if record_episodes:
            row["episodes"] = [asdict(record) for record in records]
        rows.append(row)
    return rows


# ===========================================================================
# E. repeatability
# ===========================================================================


def run_seed_variance(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_train_episodes: int = 400,
    epochs: int = 120,
    n_scenes: int = 100,
    horizon: int = 8,
    depths: Sequence[int] = (1, 2, 3),
    beam: int = 1,
    seed_eval: int = EVAL_SEED,
    record_episodes: bool = False,
) -> Dict[str, object]:
    """Train the factored model from scratch under five seeds and evaluate it on one fixed set.

    ``record_episodes`` additionally stores, per seed and depth, the per-episode
    ``(scene_id, success, action_cost, steps)`` on the exact-reachable subset, so a paired
    trace against the exact model and the random baseline is possible.  It does not change
    training or the aggregate numbers.
    """

    from ..learning.belief_model import BeliefWorldModel, train_belief_model
    from ..learning.world_model_data import collect_trajectories

    out: Dict[str, object] = {"per_seed": [], "depths": list(depths)}
    for seed in seeds:
        trajectories = collect_trajectories(
            n_episodes=n_train_episodes, horizon=horizon, seed=seed, lead_in=2
        )
        model = BeliefWorldModel(seed=seed)
        train_belief_model(model, trajectories, epochs=epochs, seed=seed)
        row: Dict[str, object] = {"seed": seed, "success": {}}
        for depth in depths:
            records = run_factored_policy(
                "search", model=model, n_scenes=n_scenes, horizon=horizon,
                depth=depth, beam=beam, seed=seed_eval, oracle_probe=False,
            )
            subset = [r for r in records if r.exact_reachable]
            row["success"][str(depth)] = {
                "all": asdict(summarise(records, "all")),
                "exact_reachable": asdict(summarise(subset, "reachable")),
            }
            if record_episodes:
                row["success"][str(depth)]["episodes"] = [
                    {
                        "scene_id": r.scene_id,
                        "success": r.success,
                        "action_cost": r.action_cost,
                        "steps": r.steps,
                        "reachable_steps": r.reachable_steps,
                    }
                    for r in subset
                ]
            row.setdefault("eval_scenes", len(records))
            row.setdefault("eval_reachable", len(subset))
        out["per_seed"].append(row)
    # The exact model on the *same* fixed evaluation set, so a paired comparison is possible.
    from .exact_adapter import ExactFactoredModel

    exact_baseline: Dict[str, object] = {}
    for depth in depths:
        records = run_factored_policy(
            "search",
            model_factory=lambda scene: ExactFactoredModel(static=scene.static),
            n_scenes=n_scenes, horizon=horizon, depth=depth, beam=beam,
            seed=seed_eval, oracle_probe=False,
        )
        subset = [r for r in records if r.exact_reachable]
        exact_baseline[str(depth)] = {
            "all": asdict(summarise(records, "all")),
            "exact_reachable": asdict(summarise(subset, "reachable")),
            "eval_scenes": len(records),
            "eval_reachable": len(subset),
        }
        if record_episodes:
            exact_baseline[str(depth)]["episodes"] = [
                {
                    "scene_id": r.scene_id,
                    "success": r.success,
                    "action_cost": r.action_cost,
                    "steps": r.steps,
                    "reachable_steps": r.reachable_steps,
                }
                for r in subset
            ]
    out["exact_model_baseline"] = exact_baseline

    for depth in depths:
        values = [
            row["success"][str(depth)]["exact_reachable"]["success_rate"]
            for row in out["per_seed"]  # type: ignore[index]
        ]
        out.setdefault("summary", {})[str(depth)] = {
            "n_seeds": len(values),
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return out


# ===========================================================================
# Driver and report
# ===========================================================================


def _git_revision() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def _git_dirty() -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def run_diagnosis(
    output_dir: str = "outputs/grid/diagnosis",
    *,
    quick: bool = False,
    n_eval: int = EVAL_SCENES,
    n_reach: int = 150,
    n_lookahead: int = 60,
    n_replace: int = 100,
    n_seed_scenes: int = 100,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Dict[str, object]:
    """Run the whole round-20 diagnosis and write machine-readable output plus a report."""

    started = time.perf_counter()
    if quick:
        n_eval, n_reach, n_lookahead, n_replace, n_seed_scenes = 40, 40, 20, 30, 40
        seeds = tuple(seeds)[:2]
    result: Dict[str, object] = {
        "meta": {
            "package_version": __version__,
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty() or "(clean)",
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "eval_seed": EVAL_SEED,
            "eval_scenes": n_eval,
            "horizon": 8,
            "warmup_action": WARMUP_ACTION,
            "oracle_probe_horizon": ORACLE_PROBE_HORIZON,
            "fail_cost": FAIL_COST,
        }
    }

    # -- A: one vocabulary ---------------------------------------------------
    from .rollout_probe import run_episode

    situations = generate_situations(
        n_scenes=n_eval, lead_in=0, seed=EVAL_SEED, physics_pool=PHYSICS_TABLE
    )
    reachability_rows: List[Dict[str, object]] = []
    pre_solved = 0
    warm_solved = 0
    oracle_yes = 0
    reach_yes = 0
    for situation in situations:
        scene = situation.scene
        if scene.is_goal():
            pre_solved += 1
            continue
        successor, _result = execute(scene, _JOINTS[WARMUP_ACTION])
        if successor.is_goal():
            warm_solved += 1
            continue
        steps = exact_reachability(successor, horizon=8)
        oracle_success, _c, _s = run_episode(
            successor, "oracle_true_state", None,
            horizon=ORACLE_PROBE_HORIZON, rng=np.random.default_rng(0),
        )
        oracle_yes += int(oracle_success)
        reach_yes += int(steps is not None)
        reachability_rows.append(
            {
                "scene_id": situation.scene_id,
                "physics": situation.physics.label,
                "reachable_steps": steps,
                "oracle_probe_success": int(oracle_success),
            }
        )
    measured = len(reachability_rows)
    result["reachability"] = {
        "n_generated": n_eval,
        "excluded_already_solved": pre_solved,
        "excluded_warmup_solved": warm_solved,
        "n_measured": measured,
        "exact_reachable": reach_yes,
        "oracle_probe_success_subset": oracle_yes,
        "oracle_success_but_unreachable_at_horizon_8": sum(
            1 for r in reachability_rows if r["oracle_probe_success"] and r["reachable_steps"] is None
        ),
        "rows": reachability_rows,
    }

    # policies on the same measured set
    from ..learning.belief_model import load_belief_model
    from .exact_adapter import ExactFactoredModel

    policy_rows: List[Dict[str, object]] = []
    for name, factory in (
        ("exact_model_d2", lambda scene: ExactFactoredModel(static=scene.static)),
    ):
        records = run_factored_policy(
            "search", model_factory=factory, n_scenes=n_eval, horizon=8, depth=2, beam=1,
            oracle_probe=False,
        )
        policy_rows.append(
            {
                "policy": name,
                "all": asdict(summarise(records, "all")),
                "exact_reachable_subset": asdict(
                    summarise([r for r in records if r.exact_reachable], "reachable")
                ),
                "oracle_probe_subset": asdict(
                    summarise([r for r in records if r.oracle_probe_success], "oracle")
                ),
            }
        )
        result["random_baseline"] = _random_baseline(n_eval, warmup=WARMUP_ACTION)
    result["policies"] = policy_rows

    # -- B: exact model in the current planners ------------------------------
    factored: List[Dict[str, object]] = []
    learned_belief = load_belief_model("outputs/grid/world_model/belief.npz")
    for depth in (1, 2, 3):
        exact_records = run_factored_policy(
            "search", model_factory=lambda scene: ExactFactoredModel(static=scene.static),
            n_scenes=n_reach, horizon=8, depth=depth, beam=1, oracle_probe=False,
        )
        learned_records = run_factored_policy(
            "search", model=learned_belief, n_scenes=n_reach, horizon=8, depth=depth,
            beam=1, oracle_probe=False,
        )
        for label, records in (("exact", exact_records), ("learned", learned_records)):
            subset = [r for r in records if r.exact_reachable]
            factored.append(
                {
                    "planner": "plan_with_search",
                    "model": label,
                    "depth": depth,
                    "beam": 1,
                    "all": asdict(summarise(records, "all")),
                    "exact_reachable": asdict(summarise(subset, "reachable")),
                }
            )
    result["factored_depth_sweep"] = factored

    from ..learning.world_model import load_world_model
    from .exact_adapter import exact_model_factory

    lookahead: List[Dict[str, object]] = []
    latent = load_world_model("outputs/grid/world_model/latent_released.npz")
    for depth in (1, 2):
        for label, model, factory in (
            ("learned_latent_released", latent, None),
            ("exact", None, exact_model_factory(condition_on_feedback=True)),
            ("exact_unconditioned_belief", None, exact_model_factory(condition_on_feedback=False)),
        ):
            records = run_lookahead_policy(
                model=model, model_factory=factory, name=f"{label}-d{depth}",
                n_scenes=n_lookahead, horizon=8, depth=depth,
            )
            subset = [r for r in records if r.exact_reachable]
            lookahead.append(
                {
                    "planner": "plan_with_model",
                    "model": label,
                    "depth": depth,
                    "all": asdict(summarise(records, "all")),
                    "exact_reachable": asdict(summarise(subset, "reachable")),
                }
            )
    result["lookahead_model_swap"] = lookahead
    result["unloadable_checkpoints"] = _unloadable_checkpoints()

    # -- C: audit ------------------------------------------------------------
    result["audit"] = {
        "brute_force_vs_central": audit_exact_backup(depth=2),
        "scoring_identity": scoring_identity_checks(),
        "cost_accounting": cost_accounting(),
        "protocol": protocol_checks(),
        "distance_sign": distance_sign_check(),
        "distance_heuristic": distance_heuristic_check(),
        "lookahead_depth": lookahead_depth_check(),
    }
    result["failure_traces"] = collect_failure_traces(n_scenes=40, seed=EVAL_SEED, limit=2)

    # -- D: module replacement ----------------------------------------------
    result["module_replacement"] = run_module_replacement(
        n_scenes=n_replace, depth=2, beam=1
    )

    # -- E: repeatability ----------------------------------------------------
    result["seed_variance"] = run_seed_variance(
        seeds=seeds, n_scenes=n_seed_scenes, depths=(1, 2, 3)
    )

    result["meta"]["wall_clock_seconds"] = time.perf_counter() - started
    _write_outputs(result, output_dir)
    return result


def _random_baseline(n_scenes: int, *, warmup: str) -> Dict[str, object]:
    starts, _pre, _warm = _measurement_starts(
        n_scenes=n_scenes, seed=EVAL_SEED, warmup=warmup
    )
    rng = np.random.default_rng(EVAL_SEED)
    records: List[EpisodeRecord] = []
    for entry in starts:
        scene = entry["scene"]
        total = 0.0
        steps = 0
        current = scene
        first: Optional[str] = None
        for _ in range(8):
            if current.is_goal():
                break
            action = ACTION_IDS[int(rng.integers(len(ACTION_IDS)))]
            if first is None:
                first = action
            successor, result = execute(current, _JOINTS[action])
            current = successor
            total += float(result.cost)
            steps += 1
        reachable = exact_reachability(scene, horizon=8)
        records.append(
            EpisodeRecord(
                scene_id=int(entry["scene_id"]),
                policy="random",
                physics=entry["physics"].label,
                success=int(current.is_goal()),
                action_cost=total,
                steps=steps,
                reachable_steps=reachable,
                exact_reachable=int(reachable is not None),
                first_action=first,
            )
        )
    return {
        "all": asdict(summarise(records, "all")),
        "exact_reachable": asdict(
            summarise([r for r in records if r.exact_reachable], "reachable")
        ),
    }


def _unloadable_checkpoints() -> List[str]:
    import glob

    from ..learning.belief_model import load_belief_model
    from ..learning.world_model import load_world_model

    broken = []
    for path in sorted(glob.glob("outputs/grid/world_model/*.npz")):
        try:
            load_world_model(path)
            continue
        except Exception:
            pass
        try:
            load_belief_model(path)
        except Exception:
            broken.append(path)
    return broken


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _metric_line(summary: Dict[str, object], prefix: str = "") -> str:
    rate = summary["success_rate"]
    conditional = summary["mean_total_cost_success"]
    conditional_text = "undefined" if conditional is None else f"{conditional:.2f}"
    per_success = summary["total_cost_per_success"]
    per_success_text = "undefined" if per_success is None else f"{per_success:.2f}"
    return (
        f"{prefix}{summary['successes']}/{summary['n']} = {rate:.3f}; "
        f"mean action cost {summary['mean_action_cost_all']:.3f}; "
        f"mean total cost (all) {summary['mean_total_cost_all']:.2f}; "
        f"mean total cost (successful only) {conditional_text}; "
        f"total cost per success {per_success_text}"
    )


def format_diagnosis(result: Dict[str, object]) -> str:
    lines: List[str] = []
    meta = result["meta"]
    lines.append(
        f"git `{meta['git_revision']}` (dirty: {meta['git_dirty']}); "
        f"numpy {meta['numpy']}; eval seed {meta['eval_seed']}, {meta['eval_scenes']} scenes; "
        f"wall clock {meta['wall_clock_seconds']:.1f}s"
    )
    reach = result["reachability"]
    lines.append("")
    lines.append("**Solvability vocabulary.**")
    lines.append(
        f"- generated {reach['n_generated']}; already solved before the window "
        f"{reach['excluded_already_solved']}; solved by the common warm-up "
        f"{reach['excluded_warmup_solved']}; measured {reach['n_measured']}"
    )
    lines.append(
        f"- exact reachability at horizon 8: {reach['exact_reachable']}/{reach['n_measured']}"
    )
    lines.append(
        f"- oracle-success subset at horizon {meta['oracle_probe_horizon']}: "
        f"{reach['oracle_probe_success_subset']}/{reach['n_measured']}"
    )
    lines.append(
        f"- episodes the oracle probe solves but exact reachability says are out of reach "
        f"at horizon 8: {reach['oracle_success_but_unreachable_at_horizon_8']}"
    )
    random_row = result.get("random_baseline")
    if random_row:
        lines.append("- random: " + _metric_line(random_row["exact_reachable"], prefix="reachable "))
    lines.append("")
    lines.append("**Exact model in the factored planner (`plan_with_search`).**")
    lines.append("| model | depth | beam | reachable subset |")
    lines.append("|---|---|---|---|")
    for row in result["factored_depth_sweep"]:
        lines.append(
            f"| {row['model']} | {row['depth']} | {row['beam']} | "
            f"{_metric_line(row['exact_reachable'])} |"
        )
    lines.append("")
    lines.append("**Exact model in the lookahead planner (`plan_with_model`).**")
    lines.append("| model | depth | all | reachable subset |")
    lines.append("|---|---|---|---|")
    for row in result["lookahead_model_swap"]:
        lines.append(
            f"| {row['model']} | {row['depth']} | {_metric_line(row['all'])} | "
            f"{_metric_line(row['exact_reachable'])} |"
        )
    lines.append("")
    lines.append("**Module replacement (factored line, depth 2).**")
    lines.append("| transition | observation | reachable subset |")
    lines.append("|---|---|---|")
    for row in result["module_replacement"]:
        if row.get("status") != "ok":
            lines.append(f"| {row['condition']} | - | {row['status']} |")
            continue
        lines.append(
            f"| {row['condition']} | - | {_metric_line(row['exact_reachable'])} |"
        )
    lines.append("")
    lines.append("**Audit.**")
    audit = result["audit"]
    lines.append(
        f"- independent recursion vs `central_q_values`: "
        f"{audit['brute_force_vs_central']['checked_candidates']} candidates, "
        f"max gap {audit['brute_force_vs_central']['max_abs_gap']:.2e}"
    )
    lines.append(
        f"- scoring identity: max gap "
        f"{audit['scoring_identity']['max_abs_gap_report_vs_reconstruction']:.2e}; "
        f"extra term the pre-fix depth-2 formula added: "
        f"{audit['scoring_identity']['old_formula_extra_term_per_action']}"
    )
    lines.append(f"- cost accounting: {audit['cost_accounting']}")
    lines.append(f"- protocol: {audit['protocol']}")
    lines.append(f"- distance sign: {audit['distance_sign']}")
    lines.append("")
    lines.append("**Failure traces.**")
    for trace in result["failure_traces"]:
        lines.append(
            f"- scene {trace['scene_id']} ({trace['physics']}), object {trace['origin']}, "
            f"robots {trace['robots']}: chose `{trace['chosen']}`"
        )
        for branch in trace["branches"][:3]:
            lines.append(
                f"    - `{branch['action']}` score {branch['score']}, cost "
                f"{branch['first_cost']}, expected distance {branch['expected_distance']}"
            )
    lines.append("")
    lines.append("**Repeatability (factored line, training seeds).**")
    exact_baseline = result["seed_variance"].get("exact_model_baseline", {})
    for depth in sorted(exact_baseline, key=int):
        row = exact_baseline[depth]
        lines.append(
            f"- depth {depth} exact model (same fixed set): "
            f"{_metric_line(row['exact_reachable'])}"
        )
    for depth, summary in sorted(result["seed_variance"]["summary"].items()):
        lines.append(
            f"- depth {depth}: n={summary['n_seeds']} seeds, values "
            f"{['%.3f' % v for v in summary['values']]}, mean {summary['mean']:.3f}, "
            f"std {summary['std']:.3f}, range [{summary['min']:.3f}, {summary['max']:.3f}]"
        )
    if result["unloadable_checkpoints"]:
        lines.append("")
        lines.append(
            "**Checkpoints that no longer load with the current architecture:** "
            + ", ".join(result["unloadable_checkpoints"])
        )
    return "\n".join(lines)


def _write_outputs(result: Dict[str, object], output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "diagnosis.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))
    paths["json"] = str(json_path)
    md_path = out / "diagnosis.md"
    md_path.write_text(
        "# Round-20 diagnosis: raw summary\n\n```\n"
        + format_diagnosis(result)
        + "\n```\n"
    )
    paths["md"] = str(md_path)
    return paths
