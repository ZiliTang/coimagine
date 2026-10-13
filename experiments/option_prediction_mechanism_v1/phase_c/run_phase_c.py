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

#: Frozen positive tolerance (CONTRACT 3.3).
DELTA = 0.02
#: The two search templates and the per-template cap of CONTRACT 3.2.
TEMPLATES = ("T1", "T2")
MAX_PAIRS_PER_TEMPLATE = 32


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
    identity_ok = pc.observation_is_identical(diff, tolerance=1e-12)

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
    # PRIMARY convention: the literal CONTRACT 3.3 reading -- every action of the common
    # executable set, including sequences whose finite Q is a catastrophic engine outcome.
    # This is the *conservative* choice for the disjointness claim because it can only ADD
    # actions to the near-optimal sets, which makes an empty intersection harder to obtain.
    analysis = analyse_q(q, delta=DELTA, diverged=None)
    # SECONDARY convention: numerically diverged sequences removed, because the controller
    # clamps force to F_max, so an impulse above 2*F_max*T is not a physical task outcome.
    analysis_excluded = analyse_q(q, delta=DELTA, diverged=diverged)
    analysis["divergence_excluded_convention"] = {
        "near_optimal_A": analysis_excluded.get("near_optimal_A"),
        "near_optimal_B": analysis_excluded.get("near_optimal_B"),
        "mutual_regret": analysis_excluded.get("mutual_regret"),
        "near_optimal_disjoint": analysis_excluded.get("near_optimal_disjoint"),
        "positive": analysis_excluded.get("positive"),
        "dropped_actions": analysis_excluded.get("dropped_diverged"),
    }
    analysis["diverged_actions"] = diverged
    # A positive additionally requires that the action each world actually prefers is a
    # physically valid outcome in that world, not an engine blow-up.
    if analysis.get("positive"):
        preferred_A = analysis.get("argmax_A")
        preferred_B = analysis.get("argmax_B")
        if preferred_A in diverged["A"] or preferred_B in diverged["B"]:
            analysis["positive"] = False
            analysis["positive_reason"] = (
                "the preferred action of a world diverged numerically, so the apparent "
                "difference is an engine artefact, not a task outcome"
            )
    analysis["divergence_clean"] = not (diverged["A"] or diverged["B"])
    # The observation-identity check is a *precondition* of the proposition: if the two worlds
    # can be told apart from the allowed shared input, the pair says nothing about ambiguity.
    analysis["observation_identical"] = bool(identity_ok)
    if not identity_ok:
        analysis["positive"] = False
        analysis["positive_reason"] = (
            "observations already distinguish the two worlds "
            f"(max abs difference {identity_max!r})"
        )
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
        "q_valid_in_both": sorted(set(q["A"]) & set(q["B"]) - set(diverged["A"]) - set(diverged["B"])),
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


#: Reference-relative contact offsets used by every candidate pair.  The reference point is
#: the midpoint of the two known robot poses, i.e. part of the static map, so the same skill
#: resolves to the same world contact point in both worlds.
CONTACT_OFFSETS = (-1.0, 0.0, 1.0)


def _make_pair(template: str, index: int, params: ms.MechanismParams, flavour: str,
               note: str) -> CandidatePair:
    skills = (ac.symmetric_action_set(params, force_limit=15.0, offsets=CONTACT_OFFSETS)
              if flavour == "symmetric"
              else ac.one_sided_action_set(params, force_limit=15.0, offsets=CONTACT_OFFSETS))
    return CandidatePair(
        pair_id=f"{template}-{index:02d}-{flavour}", template=template, params=params,
        skills=tuple(skills), note=f"{note} [{flavour}]",
    )


def build_t1_candidates() -> List[CandidatePair]:
    """T1: two robots on the same long straight edge, object shifted along that edge.

    The frozen grid is the whole T1 budget: 32 pairs.  It varies the tangential shift, the
    object length, the goal placement, the robot stance and the action-set flavour.  Shape,
    size, mass, goal, obstacles, arena and the **global robot poses** are identical inside a
    pair; only ``object_pose.x`` changes by ``shift``.
    """

    specs: List[Tuple] = []
    for shift, length in ((0.5, 6.0), (1.0, 5.0), (1.0, 6.0), (1.0, 7.0),
                          (1.5, 5.0), (1.5, 6.0), (2.0, 5.0)):
        for goal_x in (11.0, 11.5, 12.0):
            specs.append((8.5, shift, length, (8.0, 10.0),
                          Box(goal_x, 2.5, 3.5, 2.0),
                          f"shift={shift} L={length} goal_x={goal_x}"))
    specs += [
        (8.0, 1.5, 6.0, (7.5, 9.5), Box(11.0, 2.5, 3.5, 2.0), "left staging"),
        (9.0, 1.5, 6.0, (8.5, 10.5), Box(12.0, 2.5, 3.5, 2.0), "right staging"),
        (8.5, 1.5, 6.0, (8.0, 10.0), Box(11.5, 2.0, 3.5, 2.0), "goal on the edge line"),
        (8.5, 1.5, 6.0, (8.0, 10.0), Box(11.5, 3.0, 3.5, 2.0), "goal displaced in +y"),
        (8.5, 1.5, 6.0, (8.0, 10.0), Box(11.5, 2.5, 3.5, 0.4), "tight goal band (negative control)"),
        (8.5, 0.5, 6.0, (8.0, 10.0), Box(11.5, 2.5, 3.5, 2.0), "small shift (negative control)"),
        (8.5, 2.0, 5.0, (8.0, 10.0), Box(11.5, 2.5, 3.5, 2.0), "large shift, short object"),
        (8.5, 1.5, 6.0, (7.0, 9.0), Box(11.5, 2.5, 3.5, 2.0), "both robots left of centre"),
        (8.5, 1.0, 6.0, (8.0, 10.0), Box(11.0, 2.5, 3.5, 2.0), "shift 1.0 goal 11.0"),
        (8.5, 1.0, 6.0, (8.0, 10.0), Box(12.0, 2.5, 3.5, 2.0), "shift 1.0 goal 12.0"),
        (8.5, 1.5, 5.0, (8.0, 10.0), Box(11.5, 2.5, 3.5, 2.0), "shift 1.5 short object"),
        (8.5, 1.0, 7.0, (8.0, 10.0), Box(11.5, 2.0, 3.5, 3.0), "long object low goal band"),
        (8.5, 1.0, 6.0, (8.5, 10.5), Box(11.5, 2.5, 3.5, 2.0), "right stance shift 1.0"),
        (8.5, 1.5, 5.0, (8.5, 10.5), Box(11.5, 2.5, 3.5, 2.0), "right stance short object"),
    ]
    pairs: List[CandidatePair] = []
    for index, (cx, shift, length, robot_x, goal, note) in enumerate(specs[:MAX_PAIRS_PER_TEMPLATE]):
        flavour = "symmetric" if index % 2 == 0 else "one_sided"
        params = ms.MechanismParams(
            name=f"T1_p{index:02d}_{flavour}", object_centre_x=cx, shift=shift,
            goal=goal, robot_x=robot_x, object_len=length,
            reference_x=0.5 * (robot_x[0] + robot_x[1]),
        )
        pairs.append(_make_pair("T1", index, params, flavour, note))
    return pairs


def _t2_wall(wall_x: float, gap_center: float, gap: float, height: float) -> Tuple[Box, Box]:
    lower_h = gap_center - gap / 2.0
    upper_start = gap_center + gap / 2.0
    return (
        Box(wall_x, lower_h / 2.0, 0.09, lower_h / 2.0),
        Box(wall_x, (upper_start + height) / 2.0, 0.09, (height - upper_start) / 2.0),
    )


def build_t2_candidates() -> List[CandidatePair]:
    """T2: same legal local contact observations, a known passage decides the action.

    The passage is part of the known static map and is identical in both worlds.  The two
    worlds differ only in the object's tangential position, exactly as in T1.  ``wall_x`` is
    always strictly east of both worlds' object, so the initial configuration is legal in
    both, and it is beyond the 0.6 m scan range from every robot stance.  Again 32 pairs.
    """

    specs: List[Tuple] = []
    for shift, length in ((0.5, 6.0), (1.0, 5.0), (1.0, 6.0), (1.0, 7.0),
                          (1.5, 5.0), (1.5, 6.0)):
        for wall_x in (12.0, 12.5):
            for gap in (1.5, 2.5):
                specs.append((8.5, shift, length, (8.0, 10.0), Box(13.5, 2.5, 3.5, 2.5),
                              (wall_x, 2.5, gap),
                              f"shift={shift} L={length} wall={wall_x} gap={gap}"))
    specs += [
        (8.0, 1.5, 6.0, (7.5, 9.5), Box(13.0, 2.5, 3.5, 2.5), (12.0, 2.5, 2.5), "left staging"),
        (9.0, 1.5, 6.0, (8.5, 10.5), Box(14.0, 2.5, 3.5, 2.5), (13.0, 2.5, 2.5), "right staging"),
        (8.5, 1.5, 6.0, (8.0, 10.0), Box(13.5, 3.4, 3.5, 2.5), (12.5, 2.5, 2.5), "goal above the opening"),
        (8.5, 1.5, 6.0, (8.0, 10.0), Box(13.5, 2.5, 3.5, 2.5), (12.5, 3.4, 2.5), "opening above the goal"),
        (8.5, 1.5, 6.0, (8.0, 10.0), Box(13.5, 2.5, 3.5, 0.4), (12.5, 2.5, 2.5), "tight goal band (negative control)"),
        (8.5, 0.5, 6.0, (8.0, 10.0), Box(13.5, 2.5, 3.5, 2.5), (12.5, 2.5, 2.5), "small shift (negative control)"),
        (8.5, 1.5, 6.0, (7.0, 9.0), Box(13.5, 2.5, 3.5, 2.5), (12.5, 2.5, 2.5), "both robots left of centre"),
        (8.5, 1.5, 6.0, (8.0, 10.0), Box(13.5, 2.5, 3.5, 2.5), (11.9, 2.5, 1.2), "narrow opening (collision probe)"),
        (8.5, 1.0, 6.0, (8.0, 10.0), Box(13.0, 2.5, 3.5, 2.5), (12.0, 2.5, 2.5), "shift 1.0 wall 12.0"),
        (8.5, 1.0, 6.0, (8.0, 10.0), Box(13.5, 2.5, 3.5, 2.5), (12.5, 2.5, 2.5), "shift 1.0 wall 12.5"),
        (8.5, 1.5, 5.0, (8.0, 10.0), Box(12.5, 2.5, 3.5, 2.5), (11.8, 2.5, 2.5), "short object wall 11.8"),
        (8.5, 1.0, 6.0, (8.5, 10.5), Box(13.5, 2.5, 3.5, 2.5), (12.5, 2.5, 2.5), "right stance shift 1.0"),
    ]
    pairs: List[CandidatePair] = []
    for index, (cx, shift, length, robot_x, goal, wall, note) in enumerate(specs[:MAX_PAIRS_PER_TEMPLATE]):
        flavour = "symmetric" if index % 2 == 0 else "one_sided"
        wall_x, gap_center, gap = wall
        params = ms.MechanismParams(
            name=f"T2_p{index:02d}_{flavour}", object_centre_x=cx, shift=shift,
            goal=goal, obstacles=_t2_wall(wall_x, gap_center, gap, 4.0),
            robot_x=robot_x, object_len=length,
            reference_x=0.5 * (robot_x[0] + robot_x[1]),
        )
        pairs.append(_make_pair("T2", index, params, flavour, note))
    return pairs


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
        def _stable(record):
            return {
                world: {
                    key: {k: v for k, v in value.items() if k != "wall_seconds"}
                    for key, value in record["outcomes"][world].items()
                }
                for world in ("A", "B")
            }

        summary["repeatability"] = {
            "pair_id": pair.pair_id,
            "q_identical": first["q"] == second["q"],
            "max_q_abs_difference": max(
                abs(first["q"][world][key] - second["q"][world][key])
                for world in ("A", "B") for key in first["q"][world]
            ),
            "outcomes_identical": _stable(first) == _stable(second),
            "note": "wall_seconds (real simulation time) is excluded; the physical outcomes, "
                    "commands, durations and impulses are compared exactly",
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
