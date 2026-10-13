"""True-model MPC baseline and the random control.

The planner builds an independent simulated copy of the current snapshot for every
candidate, rolls the *true* physics forward, and selects by :func:`scoring.ordering_key`.
Only the first joint-skill window of the chosen sequence is executed **on the live
environment**; then the world is observed again and the planner re-plans. This is a
real-model MPC baseline, **not** a global optimum or a performance ceiling.

Protocol guarantees enforced here:

* the live environment and the planner environment are distinct objects; the planner never
  mutates the live world, and the live world never runs through ``rollout``;
* a successful node is **absorbed**: it is carried through deeper search layers unchanged
  (same first-completion time and cost) and is never expanded again;
* a physically failed node (``numeric_failure`` / ``out_of_bounds`` / ``penetration_limit``)
  is terminal, is not expanded, and cannot be selected while any feasible path exists;
* if no feasible plan exists, :data:`PLANNER_NO_FEASIBLE` is returned and the episode ends
  without executing a known-failed candidate.

Depth is handled by a width-limited beam over sequences: depth 1 is exhaustive over the
candidate template, depth 3 keeps ``beam`` partial sequences per level and expands each.
Search nodes, candidate simulated steps, prefix replay steps and wall-clock time are
reported so depth 1 and depth 3 can be compared without pretending their compute budgets
are equal.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .candidates import JointCandidate, joint_candidates
from .config import Continuous2DConfig
from .environment import (
    TERMINATION_NUMERIC,
    TERMINATION_OUT_OF_BOUNDS,
    TERMINATION_PENETRATION,
    TERMINATION_SUCCESS,
    TERMINATION_WINDOW,
    Continuous2DEnv,
    SkillResult,
    TrajectoryResult,
)
from .scoring import ordering_key, orientation_error

#: Physical failures are terminal states of an episode.
TERMINAL_FAILURES = (
    TERMINATION_NUMERIC,
    TERMINATION_OUT_OF_BOUNDS,
    TERMINATION_PENETRATION,
)
#: Returned when every root candidate ends in a terminal physical failure.
PLANNER_NO_FEASIBLE = "planner_no_feasible_plan"


@dataclass
class PlanResult:
    first: Optional[JointCandidate]
    sequence: Tuple[JointCandidate, ...]
    key: Optional[Tuple[float, ...]]
    feasible: bool
    termination: Optional[str]
    nodes: int
    simulated_steps: int
    prefix_replay_steps: int
    wall_clock: float
    candidates_at_root: int
    failure_distribution: Dict[str, int] = field(default_factory=dict)
    predicted_success_time: Optional[float] = None
    predicted_windows: int = 0
    predicted_final_key: Optional[Tuple] = None

    @property
    def candidate_id(self) -> Optional[int]:
        return None if self.first is None else self.first.candidate_id


@dataclass
class _Node:
    key: Tuple[float, ...]
    sequence: Tuple[JointCandidate, ...]
    final: object
    root_candidate_id: int
    cumulative_time: float
    cumulative_usage: float
    success: bool
    success_time: Optional[float]
    success_usage: Optional[float]
    termination: Optional[str]
    feasible: bool

    @property
    def ordering(self) -> Tuple:
        return (self.feasible, self.key)


def evaluate(
    scratch: Continuous2DEnv,
    snapshot,
    specs: Sequence,
    *,
    root_candidate_id: int,
    cumulative_time: float = 0.0,
    cumulative_usage: float = 0.0,
    prefix_substeps: int = 1,
) -> Tuple[TrajectoryResult, Tuple[float, ...], Optional[str], Optional[float], Optional[float], int]:
    """Roll one window out on the true simulator and score the resulting node.

    Returns ``(result, key, failure_termination, success_time, success_usage,
    prefix_replay_steps)``. Each window's time and control usage are added exactly once.
    """

    replay_steps = len(snapshot.prefix) * prefix_substeps
    result = scratch.rollout(snapshot, list(specs), backend="simulator")
    usage = cumulative_usage + result.cost_components["force_usage"]
    time_used = cumulative_time + result.cost_components["time"]
    success = bool(result.success)
    failure = result.termination if result.termination in TERMINAL_FAILURES else None
    success_time = time_used if success else None
    success_usage = usage if success else None
    distance = 0.0 if success else scratch.passage_distance()
    key = ordering_key(
        success=success,
        success_time=success_time if success_time is not None else 0.0,
        control_usage=success_usage if success_usage is not None else usage,
        terminal_distance=distance,
        orientation=orientation_error(scratch.backend.object_state().angle, None),
        candidate_id=root_candidate_id,
    )
    return result, key, failure, success_time, success_usage, replay_steps


def plan(
    scratch: Continuous2DEnv,
    snapshot,
    config: Continuous2DConfig,
    *,
    depth: int,
    beam: int = 3,
) -> PlanResult:
    started = time.time()
    tree = _lookahead(scratch, snapshot, config, depth=depth, beam=beam)
    entries: List[_Node] = tree["entries"]
    if not entries:
        return PlanResult(
            first=None,
            sequence=(),
            key=None,
            feasible=False,
            termination=PLANNER_NO_FEASIBLE,
            nodes=tree["nodes"],
            simulated_steps=tree["steps"],
            prefix_replay_steps=tree["prefix_steps"],
            wall_clock=time.time() - started,
            candidates_at_root=tree["root_candidates"],
            failure_distribution=tree["failure_distribution"],
        )
    best = max(entries, key=lambda node: node.ordering)
    return PlanResult(
        first=best.sequence[0],
        sequence=best.sequence,
        key=best.key,
        feasible=True,
        termination=None,
        nodes=tree["nodes"],
        simulated_steps=tree["steps"],
        prefix_replay_steps=tree["prefix_steps"],
        wall_clock=time.time() - started,
        candidates_at_root=tree["root_candidates"],
        failure_distribution=tree["failure_distribution"],
        predicted_success_time=best.success_time,
        predicted_windows=len(best.sequence),
        predicted_final_key=best.final.as_key(),
    )


def _lookahead(
    scratch: Continuous2DEnv, snapshot, config: Continuous2DConfig, *, depth: int, beam: int
) -> Dict[str, object]:
    nodes = 0
    steps = 0
    prefix_steps = 0
    substeps = config.physics.substeps

    scratch.restore(snapshot, exact=True)
    observation = scratch.observation()
    root = joint_candidates(observation, config.candidates, config.control, config.physics)
    entries: List[_Node] = []
    failure_distribution: Dict[str, int] = {}
    for candidate in root:
        result, key, failure, success_time, success_usage, replay = evaluate(
            scratch,
            snapshot,
            [candidate.skills],
            root_candidate_id=candidate.candidate_id,
            prefix_substeps=substeps,
        )
        nodes += 1
        steps += int(result.cost_components["steps"])
        prefix_steps += replay
        feasible = failure is None
        if failure is not None:
            failure_distribution[failure] = failure_distribution.get(failure, 0) + 1
        entries.append(
            _Node(
                key=key,
                sequence=(candidate,),
                final=result.final,
                root_candidate_id=candidate.candidate_id,
                cumulative_time=result.cost_components["time"],
                cumulative_usage=result.cost_components["force_usage"],
                success=bool(result.success),
                success_time=success_time,
                success_usage=success_usage,
                termination=failure,
                feasible=feasible,
            )
        )
    entries = _keep_best([node for node in entries if node.feasible], beam)

    for _level in range(1, depth):
        expanded: List[_Node] = []
        carried = [node for node in entries if node.success]
        for node in entries:
            if node.success:
                # Absorbed success: carried unchanged, never expanded, never re-costed.
                continue
            scratch.restore(node.final, exact=True)
            observation = scratch.observation()
            candidates = joint_candidates(
                observation, config.candidates, config.control, config.physics
            )
            for candidate in candidates:
                result, key, failure, success_time, success_usage, replay = evaluate(
                    scratch,
                    node.final,
                    [candidate.skills],
                    root_candidate_id=node.root_candidate_id,
                    cumulative_time=node.cumulative_time,
                    cumulative_usage=node.cumulative_usage,
                    prefix_substeps=substeps,
                )
                nodes += 1
                steps += int(result.cost_components["steps"])
                prefix_steps += replay
                if failure is not None:
                    failure_distribution[failure] = failure_distribution.get(failure, 0) + 1
                expanded.append(
                    _Node(
                        key=key,
                        sequence=node.sequence + (candidate,),
                        final=result.final,
                        root_candidate_id=node.root_candidate_id,
                        cumulative_time=node.cumulative_time + result.cost_components["time"],
                        cumulative_usage=node.cumulative_usage + result.cost_components["force_usage"],
                        success=bool(result.success),
                        success_time=success_time,
                        success_usage=success_usage,
                        termination=failure,
                        feasible=failure is None,
                    )
                )
        pool = [node for node in [*expanded, *carried] if node.feasible]
        entries = _keep_best(pool, beam)
        if not entries:
            break
    return {
        "entries": entries,
        "nodes": nodes,
        "steps": steps,
        "prefix_steps": prefix_steps,
        "root_candidates": len(root),
        "failure_distribution": failure_distribution,
    }


def _keep_best(entries: List[_Node], beam: int) -> List[_Node]:
    return sorted(entries, key=lambda node: node.ordering, reverse=True)[:beam]


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    window_index: int
    chosen_candidate_id: int
    sequence_length: int
    predicted_sequence_labels: List[str]
    predicted_success_time: Optional[float]
    predicted_final_key: Optional[Tuple]
    key: Optional[Tuple]
    termination: Optional[str]
    node_candidates: int
    plan_wall_clock: float


@dataclass
class EpisodeResult:
    success: bool
    termination: str
    windows: int
    success_time: float
    control_usage: float
    terminal_distance: float
    chosen_candidate_ids: List[int] = field(default_factory=list)
    trace: List[SkillResult] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    wall_clock: float = 0.0
    plan_wall_clock: float = 0.0
    live_physics_steps: int = 0
    candidate_simulated_steps: int = 0
    prefix_replay_steps: int = 0
    candidates: int = 0
    plan_nodes: int = 0
    failure_distribution: Dict[str, int] = field(default_factory=dict)
    no_feasible_plan: bool = False


def run_mpc_episode(
    live_env: Continuous2DEnv,
    scene: str,
    config: Continuous2DConfig,
    *,
    depth: int,
    beam: int = 3,
    horizon: Optional[int] = None,
    seed: int = 0,
    planner_env: Optional[Continuous2DEnv] = None,
) -> EpisodeResult:
    """Run one episode: the live world persists, the planner gets only snapshots."""

    horizon = horizon or config.mpc.horizon_windows
    planner = planner_env if planner_env is not None else Continuous2DEnv(config)
    live_env.reset(scene, seed=seed)
    started = time.time()
    total_time = 0.0
    total_usage = 0.0
    live_steps = 0
    nodes = 0
    candidates = 0
    candidate_steps = 0
    prefix_steps = 0
    plan_clock = 0.0
    trace: List[SkillResult] = []
    decisions: List[DecisionRecord] = []
    chosen: List[int] = []
    failure_distribution: Dict[str, int] = {}
    termination = TERMINATION_WINDOW
    no_feasible = False

    for window in range(horizon):
        if live_env.termination is not None:
            break
        snapshot = live_env.export_state()
        plan_result = plan(planner, snapshot, config, depth=depth, beam=beam)
        nodes += plan_result.nodes
        candidates += plan_result.candidates_at_root
        candidate_steps += plan_result.simulated_steps
        prefix_steps += plan_result.prefix_replay_steps
        plan_clock += plan_result.wall_clock
        for reason, count in plan_result.failure_distribution.items():
            failure_distribution[reason] = failure_distribution.get(reason, 0) + count
        if not plan_result.feasible:
            # Do not execute a known-failed candidate.
            termination = PLANNER_NO_FEASIBLE
            no_feasible = True
            decisions.append(
                DecisionRecord(
                    window_index=window,
                    chosen_candidate_id=-1,
                    sequence_length=0,
                    predicted_sequence_labels=[],
                    predicted_success_time=None,
                    predicted_final_key=None,
                    key=None,
                    termination=PLANNER_NO_FEASIBLE,
                    node_candidates=plan_result.candidates_at_root,
                    plan_wall_clock=plan_result.wall_clock,
                )
            )
            break
        decisions.append(
            DecisionRecord(
                window_index=window,
                chosen_candidate_id=int(plan_result.candidate_id),
                sequence_length=plan_result.predicted_windows,
                predicted_sequence_labels=[candidate.label for candidate in plan_result.sequence],
                predicted_success_time=plan_result.predicted_success_time,
                predicted_final_key=plan_result.predicted_final_key,
                key=plan_result.key,
                termination=None,
                node_candidates=plan_result.candidates_at_root,
                plan_wall_clock=plan_result.wall_clock,
            )
        )
        chosen.append(int(plan_result.candidate_id))
        # Execute on the live world, which is never restored or rolled out.
        result = live_env.execute_joint_skill(plan_result.first.skills)
        trace.append(result)
        live_steps += len(result.steps) * config.physics.substeps
        total_time += result.cost_components["time"]
        total_usage += result.cost_components["force_usage"]
        termination = result.termination
        if result.termination in (TERMINATION_SUCCESS, *TERMINAL_FAILURES):
            break

    return EpisodeResult(
        success=any(result.success for result in trace),
        termination=termination,
        windows=len(trace),
        success_time=total_time,
        control_usage=total_usage,
        terminal_distance=live_env.passage_distance(),
        chosen_candidate_ids=chosen,
        trace=trace,
        decisions=decisions,
        wall_clock=time.time() - started,
        plan_wall_clock=plan_clock,
        live_physics_steps=live_steps,
        candidate_simulated_steps=candidate_steps,
        prefix_replay_steps=prefix_steps,
        candidates=candidates,
        plan_nodes=nodes,
        failure_distribution=failure_distribution,
        no_feasible_plan=no_feasible,
    )


def run_random_episode(
    live_env: Continuous2DEnv,
    scene: str,
    config: Continuous2DConfig,
    *,
    seed: int,
    horizon: Optional[int] = None,
) -> EpisodeResult:
    """Random control on the same persistent live world, with no restore."""

    horizon = horizon or config.mpc.horizon_windows
    rng = random.Random(seed)
    live_env.reset(scene, seed=seed)
    started = time.time()
    total_time = 0.0
    total_usage = 0.0
    live_steps = 0
    candidates = 0
    trace: List[SkillResult] = []
    choices: List[int] = []
    termination = TERMINATION_WINDOW

    for _window in range(horizon):
        if live_env.termination is not None:
            break
        observation = live_env.observation()
        options = joint_candidates(observation, config.candidates, config.control, config.physics)
        candidates += len(options)
        pick = rng.choice(options)
        choices.append(pick.candidate_id)
        result = live_env.execute_joint_skill(pick.skills)
        trace.append(result)
        live_steps += len(result.steps) * config.physics.substeps
        total_time += result.cost_components["time"]
        total_usage += result.cost_components["force_usage"]
        termination = result.termination
        if result.termination in (TERMINATION_SUCCESS, *TERMINAL_FAILURES):
            break

    return EpisodeResult(
        success=any(result.success for result in trace),
        termination=termination,
        windows=len(trace),
        success_time=total_time,
        control_usage=total_usage,
        terminal_distance=live_env.passage_distance(),
        chosen_candidate_ids=choices,
        trace=trace,
        wall_clock=time.time() - started,
        live_physics_steps=live_steps,
        candidates=candidates,
    )
