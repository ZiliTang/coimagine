"""Phase C runner: exact-simulation information-ambiguity mechanism experiment.

Implements CONTRACT section 3 exactly:

* two templates, at most 32 candidate pairs each (<= 64 pairs), never more;
* one common set of at most 8 deployable joint skills per pair, single skills first and at
  most two-skill sequences;
* the frozen scalar payoff

      completion  = 1 if the object satisfies the existing success rule else 0
      progress    = min(1, (d_start - d_end) / max(d_start, 1e-6)), d = max corner distance
      time_cost   = total wall-clock seconds of the skill sequence
      action_cost = sum(|F| * dt) / (F_max * T_total)
      Q_h(a)      = completion + 0.5*progress - 0.05*time_cost - 0.10*action_cost

* near-optimal set ``A*_h(delta) = {a : R_h(a) <= delta}`` with ``delta = 0.02`` fixed in
  advance, and the positive criterion ``A*_A ^ A*_B = {}`` with mutual regret > delta;
* repeatability: one pair is run twice and the difference is reported (must be exactly 0);
* a full replayable record for every pair that is a candidate positive.

Run::

    .venv/bin/python experiments/option_prediction_mechanism_v1/phase_c/run_phase_c.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time as walltime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
for extra in (ROOT / "src", ROOT / "experiments" / "option_prediction_mechanism_v1" / "phase_c"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import actions as ac  # noqa: E402
import mechanism_scene as ms  # noqa: E402
import perception as pc  # noqa: E402
from coimagine.continuous2d import Continuous2DEnv, default_config  # noqa: E402
from coimagine.continuous2d.config import Continuous2DConfig  # noqa: E402
from coimagine.continuous2d.scenes import Box, Pose, SceneSpec  # noqa: E402
from coimagine.continuous2d.scoring import distance_to_box  # noqa: E402
from coimagine.continuous2d.skills import Hold, JointSpec, MoveTo, skill_to_dict  # noqa: E402

#: One joint-skill window, from the mechanism scene module (recorded in every result).
WINDOW_STEPS = ms.WINDOW_STEPS


def mechanism_config(params: ms.MechanismParams):
    """The per-candidate frozen configuration (object size + window length)."""

    return ms.mechanism_config(params)


def payoff(success: bool, start_distance: float, end_distance: float, duration: float,
           impulse: float, *, max_force: float, config=None) -> Tuple[int, float, float, float, float]:
    """The frozen scalar payoff of CONTRACT 3.3 (returns completion, progress, costs, Q)."""

    completion = 1 if success else 0
    progress = min(1.0, (start_distance - end_distance) / max(start_distance, 1e-6))
    time_cost = duration
    action_cost = impulse / max(max_force * max(duration, 1e-9), 1e-9)
    q = completion + 0.5 * progress - 0.05 * time_cost - 0.10 * action_cost
    return completion, float(progress), float(time_cost), float(action_cost), float(q)


@dataclass(frozen=True)
class SequenceOutcome:
    """The real simulated outcome of one skill sequence in one world."""

    success: bool
    termination: str
    duration: float
    impulse: float
    start_distance: float
    end_distance: float
    final_object: Tuple[float, float, float, float, float, float]
    final_speed: float
    completion: int
    progress: float
    time_cost: float
    action_cost: float
    q: float
    steps: int
    #: The success hold timer reached at the end (>= success_hold_time means success).
    success_hold: float
    #: Largest penetration seen at the end of the last window.
    final_penetration: float
    #: True when the run produced a non-finite state, left the arena, penetrated beyond the
    #: limit, or commanded an impulse larger than the force limit allows (a numerical
    #: divergence, not a task outcome).
    diverged: bool
    #: The maximum object speed reached anywhere in the sequence (reporting only).
    peak_speed: float
    #: Wall-clock seconds of the real simulation (excluding the settle phase).
    wall_seconds: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


#: Settle phase appended after every action sequence: ``hold`` for both robots lets the
#: object come to rest so the existing success rule (inside the goal and slow for 0.5 s) can
#: actually register.  ``hold`` is one of the three existing skills and is already the
#: implicit continuation of every joint window, so this adds no new controller and no new
#: capability; it is charged in full in ``time_cost``.  It is applied identically in both
#: worlds and to every action, so it cannot bias the comparison.
SETTLE_STEPS = 120


def run_sequence(env: Continuous2DEnv, world: str, params: ms.MechanismParams,
                 skills: Sequence[ac.NamedJointSkill], *, settle: bool = True) -> SequenceOutcome:
    """Reset into a fresh world and execute the joint skill sequence with the real simulator."""

    config = mechanism_config(params)
    scene_a, scene_b = params.scenes()
    scene = scene_a if world == "A" else scene_b
    env.reset(scene)
    start_distance = distance_to_box(env.backend.object_corners(), scene.goal)

    impulse = 0.0
    duration = 0.0
    steps = 0
    success = False
    termination = "window_complete"
    peak_speed = 0.0
    wall_started = walltime.time()
    for entry in skills:
        result = env.execute_joint_skill(entry.joint)
        duration += result.duration
        steps += len(result.steps)
        impulse += sum(
            math.hypot(force[0], force[1]) * config.physics.control_dt
            for step in result.steps for force in step.applied_control
        )
        for step in result.steps:
            peak_speed = max(peak_speed, math.hypot(step.object.vx, step.object.vy))
        termination = result.termination
        if result.termination in ("success", "numeric_failure", "out_of_bounds", "penetration_limit"):
            success = result.success
            break
    if settle and not success and termination == "window_complete":
        result = env.execute_joint_skill((Hold(), Hold()))
        duration += result.duration
        steps += len(result.steps)
        impulse += sum(
            math.hypot(force[0], force[1]) * config.physics.control_dt
            for step in result.steps for force in step.applied_control
        )
        termination = result.termination
        success = result.success
    wall_seconds = walltime.time() - wall_started

    end_distance = distance_to_box(env.backend.object_corners(), scene.goal)
    state = env.backend.object_state()
    final_speed = math.hypot(state.vx, state.vy)
    success = bool(env.task_success()) or bool(success)
    penetration = float(env.backend.max_penetration())
    # Physical validity.  The controller clamps each robot's force to ``max_force``, so the
    # commanded impulse can never exceed ``2 * max_force * duration``; a larger value means
    # the engine produced a collision blow-up and the run is not a task outcome at all.  Such
    # runs are excluded from the payoff matrix instead of being scored.
    impulse_bound = 1.05 * 2.0 * config.physics.max_force * max(duration, 1e-9)
    _, _, _, _, q_probe = payoff(success, start_distance, end_distance, duration, impulse,
                                 max_force=config.physics.max_force)
    diverged = bool(
        termination in ("numeric_failure", "out_of_bounds", "penetration_limit")
        or steps == 0
        or not math.isfinite(state.x) or not math.isfinite(state.y)
        or not math.isfinite(state.angle)
        or impulse > impulse_bound
        or peak_speed > 5.0
        or abs(q_probe) > 10.0
    )
    completion, progress, time_cost, action_cost, q = payoff(
        success, start_distance, end_distance, duration, impulse,
        max_force=config.physics.max_force,
    )
    return SequenceOutcome(
        success=bool(success), termination=termination, duration=float(duration),
        impulse=float(impulse), start_distance=float(start_distance), end_distance=float(end_distance),
        final_object=state.as_tuple(), final_speed=float(final_speed), completion=completion,
        progress=progress, time_cost=time_cost, action_cost=action_cost, q=q, steps=steps,
        success_hold=float(env.success_counter), final_penetration=penetration,
        diverged=diverged, peak_speed=float(peak_speed), wall_seconds=float(wall_seconds),
    )


# ---------------------------------------------------------------------------
# candidate pairs
# ---------------------------------------------------------------------------


@dataclass
class CandidatePair:
    """One pair of legal worlds plus the shared skill set."""

    pair_id: str
    template: str
    params: ms.MechanismParams
    skills: Tuple[ac.NamedJointSkill, ...]
    sequences: Tuple[Tuple[str, ...], ...] = ()
    note: str = ""

    def skill_by_id(self) -> Dict[str, ac.NamedJointSkill]:
        return {entry.skill_id: entry for entry in self.skills}


def two_skill_sequences(skills: Sequence[ac.NamedJointSkill]) -> List[Tuple[str, str]]:
    """All ordered two-skill sequences (single skills are evaluated separately)."""

    ids = [entry.skill_id for entry in skills]
    return [(first, second) for first in ids for second in ids]


def evaluate_pair(pair: CandidatePair, *, max_sequences: Optional[int] = None,
                  single_skill_windows: int = 1, sequence_windows: int = 1) -> Dict[str, object]:
    """Compute the observation-identity numbers and the Q matrix of a pair."""

    params = pair.params
    scene_a, scene_b = params.scenes()
    if not (legality_ok(scene_a) and legality_ok(scene_b)):
        return {"pair_id": pair.pair_id, "template": pair.template, "legal": False,
                "reason": "initial configuration not legal"}

    env = Continuous2DEnv(mechanism_config(params))
    env.reset(scene_a)
    obs_a = pc.observe(scene_a, params.object_pose("A"), env.backend.robot_states())
    env.reset(scene_b)
    obs_b = pc.observe(scene_b, params.object_pose("B"), env.backend.robot_states())
    diff = pc.observation_differences(obs_a, obs_b)
    identity_max = max(diff.values())
    identity_ok = identity_max <= 1e-12

    by_id = pair.skill_by_id()
    q: Dict[str, Dict[str, float]] = {"A": {}, "B": {}}
    outcomes: Dict[str, Dict[str, object]] = {"A": {}, "B": {}}
    replay: Dict[str, Dict[str, object]] = {"A": {}, "B": {}}

    single_ids = [entry.skill_id for entry in pair.skills]
    for world, scene in (("A", scene_a), ("B", scene_b)):
        for skill_id in single_ids:
            sequence = (skill_id,) * single_skill_windows
            sim = Continuous2DEnv(mechanism_config(params))
            outcome = run_sequence(sim, world, params, [by_id[sid] for sid in sequence])
            q[world]["/".join(sequence)] = outcome.q
            outcomes[world]["/".join(sequence)] = outcome.as_dict()
            replay[world]["/".join(sequence)] = {
                "skills": [by_id[sid].as_dict() for sid in sequence],
                "snapshot_hash": snapshot_hash(sim),
                "scene_hash": scene_hash(scene),
            }

    sequences = pair.sequences or tuple((sid,) for sid in single_ids)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    for sequence in sequences:
        key = "/".join(sequence)
        if key in q["A"]:
            continue
        for world in ("A", "B"):
            sim = Continuous2DEnv(mechanism_config(params))
            outcome = run_sequence(sim, world, params, [by_id[sid] for sid in sequence])
            q[world][key] = outcome.q
            outcomes[world][key] = outcome.as_dict()
            replay[world][key] = {
                "skills": [by_id[sid].as_dict() for sid in sequence],
                "snapshot_hash": snapshot_hash(sim),
                "scene_hash": scene_hash(scene_a if world == "A" else scene_b),
            }

    diverged = {
        world: sorted(key for key, value in outcomes[world].items() if value.get("diverged"))
        for world in ("A", "B")
    }
    analysis = analyse_q(q, delta=DELTA, diverged=diverged)
    analysis["diverged_actions"] = diverged
    analysis["divergence_clean"] = not (diverged["A"] or diverged["B"])
    if not analysis.get("positive"):
        analysis.setdefault("positive_reason",
                            "near-optimal sets intersect or mutual regret <= delta")
    return {
        "pair_id": pair.pair_id,
        "template": pair.template,
        "name": params.name,
        "legal": True,
        "params": params_as_dict(params),
        "skills": [entry.as_dict() for entry in pair.skills],
        "observation_identity": {
            "max_abs_difference": identity_max,
            "within_tolerance": bool(identity_ok),
            "per_quantity": diff,
            "robot_poses_equal": all(
                obs_a.robots[i].pose == obs_b.robots[i].pose for i in (0, 1)
            ),
            "robot_velocities_equal": all(
                obs_a.robots[i].velocity == obs_b.robots[i].velocity for i in (0, 1)
            ),
            "map_equal": diff.get("map.goal", 1.0) == 0.0 and diff.get("map.obstacles", 1.0) == 0.0,
            "short_faces_outside_range": not any(
                r.scan.short_face_visible for r in (*obs_a.robots, *obs_b.robots)
            ),
            "history": "cold start (no prior commands); both worlds have empty history",
        },
        "q": q,
        "outcomes": outcomes,
        "analysis": analysis,
        "replay": replay,
        "note": pair.note,
    }


def analyse_q(q: Dict[str, Dict[str, float]], *, delta: float,
              diverged: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, object]:
    """Near-optimal sets, regrets, mutual regret and the 50/50 reference (CONTRACT 3.3).

    Sequeneces that diverged numerically are removed from the payoff matrix before the
    analysis; if that empties an action set the pair is inconclusive, not positive.
    """

    dropped = {world: list((diverged or {}).get(world, ())) for world in ("A", "B")}
    actions = sorted(
        (set(q["A"]) & set(q["B"])) - set(dropped["A"]) - set(dropped["B"])
    )
    if not actions:
        return {"disjoint": False, "reason": "no common valid action", "positive": False,
                "dropped_diverged": dropped}
    best_a = max(q["A"][a] for a in actions)
    best_b = max(q["B"][a] for a in actions)
    regrets = {a: (max(best_a - q["A"][a], best_b - q["B"][a])) for a in actions}
    near_a = sorted(a for a in actions if best_a - q["A"][a] <= delta)
    near_b = sorted(a for a in actions if best_b - q["B"][a] <= delta)
    argmax_a = max(actions, key=lambda a: (q["A"][a], a))
    argmax_b = max(actions, key=lambda a: (q["B"][a], a))
    mutual_regret = max(best_a - q["A"][argmax_b], best_b - q["B"][argmax_a])
    all_knowing = 0.5 * best_a + 0.5 * best_b
    common = max(0.5 * q["A"][a] + 0.5 * q["B"][a] for a in actions)
    common_action = max(actions, key=lambda a: (0.5 * q["A"][a] + 0.5 * q["B"][a], a))
    return {
        "actions": actions,
        "best_A": best_a,
        "best_B": best_b,
        "argmax_A": argmax_a,
        "argmax_B": argmax_b,
        "near_optimal_A": near_a,
        "near_optimal_B": near_b,
        "near_optimal_disjoint": len(set(near_a) & set(near_b)) == 0,
        "regret_A": {a: best_a - q["A"][a] for a in actions},
        "regret_B": {a: best_b - q["B"][a] for a in actions},
        "mutual_regret": mutual_regret,
        "delta": delta,
        "positive": bool(len(set(near_a) & set(near_b)) == 0 and mutual_regret > delta),
        "all_knowing_50_50": all_knowing,
        "common_action_50_50": common,
        "common_action_50_50_id": common_action,
        "information_gap_50_50": all_knowing - common,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def legality_ok(scene: SceneSpec) -> bool:
    from coimagine.continuous2d.legality import legality_report

    return bool(legality_report(scene)["legal"])


def params_as_dict(params: ms.MechanismParams) -> Dict[str, object]:
    payload = asdict(params)
    payload["goal"] = asdict(params.goal)
    payload["obstacles"] = [asdict(box) for box in params.obstacles]
    return payload


def scene_hash(scene: SceneSpec) -> str:
    from coimagine.continuous2d.legality import scene_content_hash

    return scene_content_hash(scene)


def snapshot_hash(env: Continuous2DEnv) -> str:
    snapshot = env.export_state()
    payload = json.dumps({
        "tick": snapshot.tick,
        "object": list(snapshot.object.as_tuple()),
        "robots": [list(state.as_tuple()) for state in snapshot.robots],
        "success_counter": snapshot.success_counter,
        "termination": snapshot.termination,
        "prefix_len": len(snapshot.prefix),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_candidates() -> List[CandidatePair]:
    """The frozen candidate-pair search: exactly the templates and budgets of CONTRACT 3.2."""

    pairs: List[CandidatePair] = []
    pairs += build_t1_candidates()
    pairs += build_t2_candidates()
    return pairs


# -- T1: two robots on the same long edge, object shifted along that edge --------------


def build_t1_candidates() -> List[CandidatePair]:
    pairs: List[CandidatePair] = []
    setups = [
        # (object centre x, shift, robot x, object length, goal, description)
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(11.60, 2.00, 0.60, 0.35), "baseline long edge"),
        (8.50, 1.00, (8.00, 10.00), 6.0, Box(11.60, 2.00, 0.60, 0.35), "smaller shift"),
        (8.50, 2.00, (8.00, 10.00), 6.0, Box(11.60, 2.00, 0.60, 0.35), "larger shift"),
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(12.20, 2.00, 0.60, 0.35), "goal further"),
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(11.60, 2.30, 0.60, 0.35), "goal offset laterally"),
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(11.60, 1.70, 0.60, 0.35), "goal offset other side"),
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(11.60, 2.00, 0.60, 0.22), "tight goal slot"),
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(11.60, 2.00, 0.45, 0.40), "wide-short goal"),
        (8.50, 1.50, (8.40, 9.20), 6.0, Box(11.60, 2.00, 0.60, 0.35), "robots closer together"),
        (8.50, 1.50, (7.60, 9.40), 6.0, Box(11.60, 2.00, 0.60, 0.35), "robots symmetric about centre"),
        (9.00, 1.20, (8.40, 10.60), 6.0, Box(12.00, 2.00, 0.60, 0.35), "wider stance"),
        (8.00, 1.20, (7.40, 9.60), 6.0, Box(11.20, 2.00, 0.60, 0.35), "shifted left staging"),
        (8.50, 1.50, (8.00, 10.00), 5.0, Box(11.60, 2.00, 0.60, 0.35), "shorter object"),
        (8.50, 1.50, (8.00, 10.00), 7.0, Box(11.60, 2.00, 0.60, 0.35), "longer object"),
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(11.60, 2.00, 0.80, 0.30), "wide goal"),
        (8.50, 1.50, (8.00, 10.00), 6.0, Box(11.00, 1.80, 0.55, 0.30), "diagonal goal"),
        (8.50, 1.50, (8.20, 9.60), 6.0, Box(11.40, 2.10, 0.50, 0.30), "asymmetric robots"),
        (9.00, 1.00, (8.20, 10.20), 6.0, Box(12.00, 1.90, 0.55, 0.30), "lateral goal shift"),
    ]
    for index, (cx, shift, robot_x, length, goal, note) in enumerate(setups):
        if index >= MAX_PAIRS_PER_TEMPLATE:
            break
        for variant, action_set in enumerate(("symmetric", "one_sided")):
            if len(pairs) >= MAX_PAIRS_PER_TEMPLATE:
                break
            params = ms.MechanismParams(
                name=f"T1_p{index:02d}_{variant}", object_centre_x=cx, shift=shift,
                goal=goal, robot_x=robot_x, object_len=length,
            )
            skills = (ac.symmetric_action_set(params, force_limit=15.0)
                      if action_set == "symmetric"
                      else ac.one_sided_action_set(params, force_limit=15.0))
            pairs.append(CandidatePair(
                pair_id=f"T1-{index:02d}-{action_set}", template="T1", params=params,
                skills=tuple(skills), note=note,
            ))
    return pairs


# -- T2: same local contact observations, a known passage makes the action differ ------


def build_t2_candidates() -> List[CandidatePair]:
    """T2: the same visible long edge, but a known passage/obstacle decides which action works.

    The passage is part of the *known static map* (identical in both worlds).  The two worlds
    differ only in where the object's centre of mass is along the edge, exactly as in T1, so
    the observation-identity argument is unchanged; the payoff now depends on whether the
    object can be threaded through the passage.
    """

    pairs: List[CandidatePair] = []
    setups = [
        # (object centre x, shift, robot x, object length, goal, obstacles, note)
        (7.00, 1.50, (6.50, 8.50), 6.0, Box(11.60, 2.00, 0.60, 0.35),
         (Box(10.00, 0.90, 0.09, 0.90), Box(10.00, 3.10, 0.09, 0.90)),
         "passage at x=10, gap 1.30"),
        (7.00, 1.50, (6.50, 8.50), 6.0, Box(11.60, 2.00, 0.60, 0.35),
         (Box(10.00, 1.10, 0.09, 1.10), Box(10.00, 2.90, 0.09, 1.10)),
         "passage gap 0.70"),
        (7.00, 2.00, (6.50, 8.50), 6.0, Box(11.60, 2.00, 0.60, 0.35),
         (Box(10.00, 1.10, 0.09, 1.10), Box(10.00, 2.90, 0.09, 1.10)),
         "passage gap 0.70, larger shift"),
        (7.00, 1.50, (6.50, 8.50), 6.0, Box(11.20, 2.00, 0.60, 0.35),
         (Box(9.60, 1.10, 0.09, 1.10), Box(9.60, 2.90, 0.09, 1.10)),
         "earlier passage"),
        (7.50, 1.50, (6.80, 9.00), 6.0, Box(11.60, 1.60, 0.60, 0.35),
         (Box(10.00, 0.70, 0.09, 0.70), Box(10.00, 2.50, 0.09, 0.70)),
         "passage offset from the goal line"),
        (7.00, 1.50, (6.50, 8.50), 6.0, Box(11.60, 2.00, 0.60, 0.35),
         (Box(10.00, 0.60, 0.09, 0.60), Box(10.00, 2.40, 0.09, 0.60)),
         "narrow offset passage"),
        (7.00, 1.00, (6.50, 8.50), 6.0, Box(11.60, 2.00, 0.60, 0.35),
         (Box(10.00, 1.10, 0.09, 1.10), Box(10.00, 2.90, 0.09, 1.10)),
         "smaller shift through passage"),
        (7.00, 2.00, (6.50, 8.50), 7.0, Box(11.60, 2.00, 0.60, 0.35),
         (Box(10.00, 1.10, 0.09, 1.10), Box(10.00, 2.90, 0.09, 1.10)),
         "longer object through passage"),
    ]
    for index, (cx, shift, robot_x, length, goal, obstacles, note) in enumerate(setups):
        if index >= MAX_PAIRS_PER_TEMPLATE:
            break
        for variant, action_set in enumerate(("symmetric", "one_sided")):
            if len(pairs) >= MAX_PAIRS_PER_TEMPLATE:
                break
            params = ms.MechanismParams(
                name=f"T2_p{index:02d}_{variant}", object_centre_x=cx, shift=shift,
                goal=goal, obstacles=tuple(obstacles), robot_x=robot_x, object_len=length,
            )
            skills = (ac.symmetric_action_set(params, force_limit=15.0)
                      if action_set == "symmetric"
                      else ac.one_sided_action_set(params, force_limit=15.0))
            pairs.append(CandidatePair(
                pair_id=f"T2-{index:02d}-{action_set}", template="T2", params=params,
                skills=tuple(skills), note=note,
            ))
    return pairs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="outputs/option_prediction_mechanism_v1")
    parser.add_argument("--run-id", default="phase_c")
    parser.add_argument("--two-skill", action="store_true",
                        help="also enumerate all ordered two-skill sequences (slower)")
    parser.add_argument("--repeat-check", action="store_true",
                        help="run the first positive pair twice and report the difference")
    parser.add_argument("--only", default=None, help="evaluate only this pair_id")
    args = parser.parse_args()

    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_candidates()
    if args.only:
        pairs = [pair for pair in pairs if pair.pair_id == args.only]
    started = walltime.time()
    records: List[Dict[str, object]] = []
    positives: List[Dict[str, object]] = []
    for index, pair in enumerate(pairs):
        if args.two_skill:
            pair.sequences = tuple(two_skill_sequences(pair.skills))
        record = evaluate_pair(pair)
        records.append(record)
        if record.get("analysis", {}).get("positive"):
            positives.append(record)
        elapsed = walltime.time() - started
        print(f"[{index + 1}/{len(pairs)}] {pair.pair_id} "
              f"identity={record.get('observation_identity', {}).get('max_abs_difference')} "
              f"positive={record.get('analysis', {}).get('positive')} "
              f"elapsed={elapsed:.1f}s", flush=True)

    summary = {
        "run_id": args.run_id,
        "delta": DELTA,
        "window_steps": WINDOW_STEPS,
        "control_dt": default_config().physics.control_dt,
        "window_seconds": WINDOW_STEPS * default_config().physics.control_dt,
        "frozen_config_hash": default_config().config_hash(),
        "per_candidate_config_hash": "recorded in each record",
        "templates": list(TEMPLATES),
        "pairs_total": len(pairs),
        "pairs_per_template": {
            template: sum(1 for pair in pairs if pair.template == template) for template in TEMPLATES
        },
        "positives": [record["pair_id"] for record in positives],
        "elapsed_seconds": walltime.time() - started,
        "notes": {
            "scene": "explicit long-rectangle mechanism scene (CONTRACT 3.2)",
            "sensing": {
                "scan_range": pc.SCAN_RANGE,
                "scan_rays": pc.SCAN_RAYS,
                "fov_half_angle_deg": math.degrees(pc.FOV_HALF_ANGLE),
                "noise_sigma": 0.0,
                "declared": "frozen 0.6 m range and 72 rays; a forward +/-60 deg cone is the "
                            "one declared narrowing (removes the arena behind each robot)",
            },
            "history": "cold start: no commands before the decision, identical (empty) history",
            "robot_poses": "identical global poses in both worlds (required; verified)",
        },
        "records": records,
    }
    if args.repeat_check and pairs:
        target = positives[0] if positives else records[0]
        pair = next(pair for pair in pairs if pair.pair_id == target["pair_id"])
        first = evaluate_pair(pair)
        second = evaluate_pair(pair)
        summary["repeatability"] = {
            "pair_id": pair.pair_id,
            "q_identical": first["q"] == second["q"],
            "max_q_abs_difference": max(
                abs(first["q"][world][key] - second["q"][world][key])
                for world in ("A", "B") for key in first["q"][world]
            ),
            "outcomes_identical": first["outcomes"] == second["outcomes"],
        }
        print("repeatability", json.dumps(summary["repeatability"]), flush=True)

    (out_dir / "phase_c_records.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    write_tables(summary, out_dir)
    print(f"wrote {out_dir / 'phase_c_records.json'}")
    print("PHASE_C_RUN_OK")
    return 0


def write_tables(summary: Dict[str, object], out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Phase C raw table\n")
    lines.append(f"delta = {summary['delta']}, pairs = {summary['pairs_total']}, "
                 f"positives = {len(summary['positives'])}\n")
    for record in summary["records"]:
        ident = record.get("observation_identity", {})
        analysis = record.get("analysis", {})
        lines.append(f"\n## {record['pair_id']} ({record.get('note', '')})\n")
        lines.append(f"- legal: {record.get('legal')}\n")
        lines.append(f"- observation max abs difference: {ident.get('max_abs_difference')}\n")
        lines.append(f"- near-optimal A: {analysis.get('near_optimal_A')}\n")
        lines.append(f"- near-optimal B: {analysis.get('near_optimal_B')}\n")
        lines.append(f"- mutual regret: {analysis.get('mutual_regret')}\n")
        lines.append(f"- 50/50 all-knowing: {analysis.get('all_knowing_50_50')}, "
                     f"common: {analysis.get('common_action_50_50')}, "
                     f"gap: {analysis.get('information_gap_50_50')}\n")
        q = record.get("q", {})
        if q:
            lines.append("\n| action | Q_A | Q_B |\n|---|---|---|\n")
            for key in sorted(q["A"]):
                lines.append(f"| {key} | {q['A'][key]:+.5f} | {q['B'][key]:+.5f} |\n")
    (out_dir / "phase_c_table.md").write_text("".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
