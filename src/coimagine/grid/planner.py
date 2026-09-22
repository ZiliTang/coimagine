"""Belief-space planning for the Stage 2 grid, and the six baseline classes.

Planning-stage precondition
---------------------------
The planner operates **at contact**: both robots have reached the object using their
own sensing and the prior map, so each robot privately knows the object pose from
contact geometry plus its own localisation. Object-localisation error is therefore a
*sensing* diagnostic reported separately by :mod:`coimagine.grid.belief`; it is not
part of the contact decision problem. The remaining hidden state is the contact
physics ``(resistance, constraint, traction_A, traction_B, released)`` = 32
hypotheses.

Observation model used for planning
-----------------------------------
A boundary branches over each robot's own load/slip feedback (four symbols) and the
public ``moved`` / ``stalled`` event. The occupancy scan is used to establish
contact and localisation, not as a planning branch. This restriction keeps the
planner exact and tractable and is listed as a limitation.

Planner classes
---------------
``public``   B = 0: the commit rule sees public information only
``codec``    a fixed B-bit message per robot, public commit, future messages modelled
``replan``   the same codec, but the lookahead never predicts the next message (C2 control)
``limited``  the same codec with a shallower lookahead (computation-limit diagnostic)
``central``  both robots' feedback histories shared (complete observation sharing)
``full``     true physics known (full-information lower bound)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .candidates import Candidate, CandidateSet, generate_candidates
from .dynamics import JointSkill, execute
from .state import (
    Cell,
    ObjectState,
    RobotState,
    Scene,
    StaticMap,
    enumerate_hidden_physics,
)

HypKey = Tuple[int, bool]  # (physics index, released)
#: ``(origin, robots, hypothesis, history A, history B)``.
#: The robot poses are part of the entry because a successful push advances the
#: pushers, so different hypotheses can end a boundary with different poses.
PEntry = Tuple[Cell, Tuple[RobotState, ...], HypKey, Tuple[str, ...], Tuple[str, ...]]
PBelief = Dict[PEntry, float]
#: ``(origin, robots, hypothesis)`` -- for planners that see both histories
CEntry = Tuple[Cell, Tuple[RobotState, ...], HypKey]
CBelief = Dict[CEntry, float]

PHYSICS = enumerate_hidden_physics()
HYPOTHESES: Tuple[HypKey, ...] = tuple(
    (index, released)
    for index in range(len(PHYSICS))
    for released in (False, True)
)

#: Candidate kinds meaningful once the robots are at contact. Contact acquisition
#: (repositioning) is handled by the public geometric layer.
CONTACT_KINDS = frozenset({"co_push", "push", "probe", "release", "wait"})
#: The pose-agnostic public set also allows repositioning, because a robot that does
#: not know where the object is must be allowed to go and look.
PUBLIC_KINDS = CONTACT_KINDS | {"reposition"}

DEFAULT_FAIL_COST = 20.0

FEEDBACK_SYMBOLS = ("drove", "idle", "slipped", "strained")


@dataclass(frozen=True)
class ContactContext:
    """Public geometry at the planning stage: map, object origin, robot poses."""

    static: StaticMap
    origin: Cell
    robots: Tuple[RobotState, ...]

    def scene(self, hyp: HypKey, origin: Optional[Cell] = None) -> Scene:
        return Scene(
            static=self.static,
            obj=ObjectState(self.origin if origin is None else origin),
            robots=self.robots,
            physics=PHYSICS[hyp[0]],
            released=hyp[1],
        )


def _useful_direction(ctx: ContactContext, direction) -> bool:
    """Public prune: a push is useful only if it does not move the object away."""

    if not ctx.static.target:
        return True
    moved = ObjectState(ctx.origin).translated(direction.delta)
    if not all(ctx.static.in_bounds(c) and not ctx.static.is_wall(c) for c in moved.cells):
        return False

    def distance(cells) -> int:
        return min(
            abs(c[0] - t[0]) + abs(c[1] - t[1]) for c in cells for t in ctx.static.target
        )

    return distance(moved.cells) <= distance(ObjectState(ctx.origin).cells)


def _geometrically_plausible(ctx: ContactContext, joint: JointSkill) -> bool:
    """Keep only candidates some robot can actually execute from this contact.

    Contact geometry does not depend on the hidden physics, so this pruning uses
    public information only and never leaks the answer.
    """

    from .dynamics import (
        SKILL_HOLD,
        SKILL_PROBE,
        SKILL_PUSH,
        SKILL_RELEASE,
        SKILL_WAIT,
        push_contact_ok,
        release_contact_ok,
    )

    scene = ctx.scene((0, False))
    active = False
    for robot_id, skill in enumerate(joint.skills):
        if skill.kind in (SKILL_PUSH, SKILL_PROBE):
            if not push_contact_ok(scene, robot_id, skill.direction):
                return False
            if not _useful_direction(ctx, skill.direction):
                return False
            active = True
        elif skill.kind == SKILL_RELEASE:
            if not release_contact_ok(scene, robot_id):
                return False
            active = True
        elif skill.kind in (SKILL_HOLD, SKILL_WAIT):
            continue
    return active


def contact_candidates(ctx: ContactContext, *, public: bool = False) -> CandidateSet:
    """Contact candidates.

    ``public=True`` returns the **pose-agnostic template set**: every direction and
    every contact skill, with no geometric pruning by the object's position. This is
    what a robot that does not know where the object is may legitimately enumerate,
    and it is the set the calibrated testbed uses. ``public=False`` additionally
    prunes by the (true) object geometry, which is a privileged operation and is
    kept only for the earlier reference experiments.
    """

    full = generate_candidates(
        ctx.static,
        tuple(r.pose for r in ctx.robots),
        max_reposition_targets=64 if public else 4,
    )
    kinds = PUBLIC_KINDS if public else CONTACT_KINDS
    kept = [c for c in full.candidates if c.kind in kinds]
    if public:
        return CandidateSet(tuple(kept))
    return CandidateSet(
        tuple(c for c in kept if _geometrically_plausible(ctx, c.joint))
    )


def belief_from_posterior(posterior, robots: Tuple[RobotState, ...]) -> PBelief:
    """Convert a scan-based posterior over ``(origin, physics, released)`` into the
    planner's public entry format.

    This is the bridge that lets the planner run on a robot's **estimated** object
    pose instead of the simulator's true one.
    """

    belief: PBelief = {}
    for hypothesis, weight in posterior.items():
        if weight <= 0.0:
            continue
        index = PHYSICS.index(hypothesis.physics)
        entry: PEntry = (
            hypothesis.origin,
            robots,
            (index, bool(hypothesis.released)),
            (),
            (),
        )
        belief[entry] = belief.get(entry, 0.0) + weight
    return _normalise(belief)  # type: ignore[return-value]


def initial_contact_belief(ctx: ContactContext) -> PBelief:
    """Uniform over the 16 contact types, all **unreleased**.

    An episode starts before any release, so ``released=True`` is not a valid
    initial hypothesis; it becomes reachable only through a successful release.
    """

    initial = tuple((index, False) for index in range(len(PHYSICS)))
    weight = 1.0 / len(initial)
    return {(ctx.origin, ctx.robots, hyp, (), ()): weight for hyp in initial}


@lru_cache(maxsize=300_000)
def _execute_cached(scene: Scene, joint: JointSkill):
    """Deterministic transition, memoised: the planner calls it thousands of times."""

    return execute(scene, joint)


def _normalise(weights: Mapping[object, float]) -> Dict[object, float]:
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in weights.items() if v > 0.0}


def _belief_key(belief: Mapping[object, float]) -> Tuple[object, ...]:
    return tuple(sorted(belief.items(), key=lambda kv: repr(kv[0])))


# --------------------------------------------------------------------------
# Private posterior over hypotheses
# --------------------------------------------------------------------------

_SIM_CACHE: Dict[Tuple[object, ...], Tuple[str, ...]] = {}


def implied_history(
    ctx: ContactContext, skills: Sequence[JointSkill], hyp: HypKey, robot_id: int
) -> Tuple[str, ...]:
    """The feedback history one hypothesis implies for one robot."""

    key = (ctx, tuple(skills), hyp, robot_id)
    cached = _SIM_CACHE.get(key)
    if cached is not None:
        return cached
    scene = ctx.scene(hyp)
    history: List[str] = []
    for skill in skills:
        scene, result = execute(scene, skill)
        history.append(result.feedback[robot_id])
    out = tuple(history)
    _SIM_CACHE[key] = out
    return out


_POSTERIOR_CACHE: Dict[Tuple[object, ...], Dict[HypKey, float]] = {}


def private_posterior(
    ctx: ContactContext,
    skills: Sequence[JointSkill],
    robot_id: int,
    history: Tuple[str, ...],
) -> Dict[HypKey, float]:
    """``P(hypothesis | own feedback history)``, uniform over consistent hypotheses."""

    key = (ctx, tuple(skills), robot_id, history)
    cached = _POSTERIOR_CACHE.get(key)
    if cached is not None:
        return cached
    out: Dict[HypKey, float] = {}
    for hyp in HYPOTHESES:
        if implied_history(ctx, skills, hyp, robot_id) == history:
            out[hyp] = 1.0
    result = _normalise(out)  # type: ignore[assignment]
    _POSTERIOR_CACHE[key] = result  # type: ignore[assignment]
    return result  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Message codecs
# --------------------------------------------------------------------------


class GridCodec:
    name = "null"
    bits = 0

    def encode(
        self,
        ctx: ContactContext,
        skills: Sequence[JointSkill],
        robot_id: int,
        history: Tuple[str, ...],
        candidates: CandidateSet,
    ) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def _levels(self) -> int:
        return 1 << self.bits


class NullGridCodec(GridCodec):
    """``B = 0``: no task message."""

    name = "null"
    bits = 0

    def encode(self, ctx, skills, robot_id, history, candidates) -> int:
        return 0


class ObservationGridCodec(GridCodec):
    """Transmit the current local feedback symbol."""

    name = "observation"

    def __init__(self, bits: int = 2) -> None:
        self.bits = bits

    def encode(self, ctx, skills, robot_id, history, candidates) -> int:
        if not history:
            return 0
        index = FEEDBACK_SYMBOLS.index(history[-1])
        return index % self._levels()


class ModeGridCodec(GridCodec):
    """Quantise the private posterior that the cooperative push will succeed."""

    name = "posterior-mode"

    def __init__(self, bits: int = 1) -> None:
        self.bits = bits

    def encode(self, ctx, skills, robot_id, history, candidates) -> int:
        posterior = private_posterior(ctx, skills, robot_id, history)
        if not posterior:
            return 0
        probability = 0.0
        for hyp, weight in posterior.items():
            physics = PHYSICS[hyp[0]]
            free = not (physics.constraint == "constrained" and not hyp[1])
            if free and physics.effective_capacity((0, 1)) >= physics.resistance:
                probability += weight
        levels = self._levels()
        index = int(min(max(probability, 0.0), 1.0) * levels)
        return min(index, levels - 1)


class ConsequenceGridCodec(GridCodec):
    """Action-conditioned consequence summary over the public candidates."""

    name = "consequence"

    def __init__(self, bits: int = 1) -> None:
        self.bits = bits

    def encode(self, ctx, skills, robot_id, history, candidates) -> int:
        posterior = private_posterior(ctx, skills, robot_id, history)
        if not posterior:
            return 0
        predicted: List[float] = []
        for candidate in candidates.candidates:
            success = 0.0
            for hyp, weight in posterior.items():
                scene = ctx.scene(hyp)
                successor, _result = _execute_cached(scene, candidate.joint)
                if successor.obj.within(ctx.static.target):
                    success += weight
            predicted.append(success)
        levels = self._levels()
        if predicted and levels >= len(predicted):
            best_index, best_value = 0, -1.0
            for index, value in enumerate(predicted):
                if value > best_value + 1e-12:
                    best_index, best_value = index, value
            return best_index
        value = predicted[0] if predicted else 0.0
        index = int(min(max(value, 0.0), 1.0) * levels)
        return min(index, levels - 1)


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------


def successors_codec(
    ctx: ContactContext,
    belief: PBelief,
    joint: JointSkill,
    codec: GridCodec,
    skills: Sequence[JointSkill],
    candidates: CandidateSet,
) -> Tuple[float, Dict[Tuple[str, int, int], Tuple[float, PBelief]]]:
    """Fixed-codec public transition: branch over ``(y, m_A, m_B)``."""

    new_skills = tuple(skills) + (joint,)
    buckets: Dict[Tuple[str, int, int], Dict[object, float]] = {}
    immediate = 0.0
    for (origin, robots, hyp, h_a, h_b), weight in belief.items():
        scene = Scene(
            static=ctx.static, obj=ObjectState(origin), robots=robots,
            physics=PHYSICS[hyp[0]], released=hyp[1],
        )
        successor, result = _execute_cached(scene, joint)
        immediate += weight * result.cost
        new_h_a = h_a + (result.feedback[0],)
        new_h_b = h_b + (result.feedback[1],)
        m_a = codec.encode(ctx, new_skills, 0, new_h_a, candidates)
        m_b = codec.encode(ctx, new_skills, 1, new_h_b, candidates)
        key = (result.y_public, m_a, m_b)
        bucket = buckets.setdefault(key, {})
        entry: PEntry = (
            successor.obj.origin,
            successor.robots,
            (hyp[0], successor.released),
            new_h_a,
            new_h_b,
        )
        bucket[entry] = bucket.get(entry, 0.0) + weight

    out: Dict[Tuple[str, int, int], Tuple[float, PBelief]] = {}
    for key, weights in buckets.items():
        out[key] = (sum(weights.values()), _normalise(weights))  # type: ignore[arg-type]
    return immediate, out


def successors_central(
    ctx: ContactContext, belief: CBelief, joint: JointSkill
) -> Tuple[float, Dict[Tuple[str, str, str], Tuple[float, CBelief]]]:
    buckets: Dict[Tuple[str, str, str], Dict[object, float]] = {}
    immediate = 0.0
    for (origin, robots, hyp), weight in belief.items():
        scene = Scene(
            static=ctx.static, obj=ObjectState(origin), robots=robots,
            physics=PHYSICS[hyp[0]], released=hyp[1],
        )
        successor, result = _execute_cached(scene, joint)
        immediate += weight * result.cost
        key = (result.feedback[0], result.feedback[1], result.y_public)
        bucket = buckets.setdefault(key, {})
        entry: CEntry = (successor.obj.origin, successor.robots, (hyp[0], successor.released))
        bucket[entry] = bucket.get(entry, 0.0) + weight
    out: Dict[Tuple[str, str, str], Tuple[float, CBelief]] = {}
    for key, weights in buckets.items():
        out[key] = (sum(weights.values()), _normalise(weights))  # type: ignore[arg-type]
    return immediate, out


def successors_public_only(
    ctx: ContactContext, belief: PBelief, joint: JointSkill
) -> Tuple[float, Dict[str, Tuple[float, CBelief]]]:
    """Branch on the public event only; never predicts the next message (C2 control)."""

    buckets: Dict[str, Dict[object, float]] = {}
    immediate = 0.0
    for (origin, robots, hyp, _h_a, _h_b), weight in belief.items():
        scene = Scene(
            static=ctx.static, obj=ObjectState(origin), robots=robots,
            physics=PHYSICS[hyp[0]], released=hyp[1],
        )
        successor, result = _execute_cached(scene, joint)
        immediate += weight * result.cost
        bucket = buckets.setdefault(result.y_public, {})
        entry: CEntry = (successor.obj.origin, successor.robots, (hyp[0], successor.released))
        bucket[entry] = bucket.get(entry, 0.0) + weight
    out: Dict[str, Tuple[float, CBelief]] = {}
    for key, weights in buckets.items():
        out[key] = (sum(weights.values()), _normalise(weights))  # type: ignore[arg-type]
    return immediate, out


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


#: Weight on the distance-to-goal heuristic used at the lookahead horizon.
#:
#: It must exceed the per-step action cost. With weight 1.0 (one cell of progress
#: worth exactly one step) making progress and standing still are equivalent under the
#: heuristic, so a belief-space planner facing a risky push prefers the free "wait" and
#: never moves: measured, the equal-information policy then idles for the whole
#: horizon. Weight 3.0 makes progress strictly preferable while still being a
#: heuristic rather than a reward. The same weight is applied to every planner.
DEFAULT_PROGRESS_WEIGHT = 3.0


def _cut_value(
    belief: Mapping[object, float],
    static: StaticMap,
    fail_cost: float,
    progress_weight: Optional[float] = None,
) -> float:
    """Heuristic cost-to-go at the lookahead cut.

    Two terms, both heuristics rather than rewards:

    * ``fail_cost * risk`` -- a risk penalty on hypotheses that have not reached the
      goal by the cut. The true terminal failure cost is charged once, at the real end
      of the episode, so this term is a *lookahead* device. It is kept because removing
      it makes a failed action and a neutral preparation action tie, and the fixed
      tie-break then picks the failing one: measured, the belief-space planner stopped
      solving a task the full-information planner solves in three steps. An unproductive
      action must look worse than a productive one.
    * ``progress_weight * distance`` -- distance to the goal, computed over the **worst**
      object cell. Taking the minimum over the object's cells returns 0 as soon as the
      object merely overlaps the goal region, which makes a half-on-the-goal object look
      finished.

    The weight must exceed the per-step action cost, otherwise one cell of progress is
    worth exactly one step and standing still is as good as advancing. The same function
    is applied identically to every planner.
    """

    # Read the module constant at call time rather than binding it as a default
    # argument: a default binds at definition, so mutating the constant later silently
    # has no effect, which made an earlier weight sweep look like the weight did nothing.
    if progress_weight is None:
        progress_weight = DEFAULT_PROGRESS_WEIGHT

    risk = 0.0
    progress = 0.0
    for entry, weight in belief.items():
        cells = ObjectState(entry[0]).cells
        if cells <= static.target:
            continue
        risk += weight
        progress += weight * max(
            min(abs(c[0] - t[0]) + abs(c[1] - t[1]) for t in static.target)
            for c in cells
        )
    return fail_cost * risk + progress_weight * progress


def _goal_reached(belief: Mapping[object, float], static: StaticMap) -> bool:
    """True when every surviving hypothesis already has the object in the goal.

    The episode ends there, so no further action may be charged. Without this check
    the planners happily "act" inside the goal and pay a probe for nothing.
    """

    if not belief:
        return False
    return all(
        ObjectState(entry[0]).within(static.target) for entry in belief
    )


def _as_pbelief(belief: CBelief) -> PBelief:
    return {
        (origin, robots, hyp, (), ()): weight
        for (origin, robots, hyp), weight in belief.items()
    }


def codec_value(
    ctx: ContactContext,
    belief: PBelief,
    codec: GridCodec,
    skills: Sequence[JointSkill],
    candidates: CandidateSet,
    depth: int,
    *,
    fail_cost: float = DEFAULT_FAIL_COST,
    predict_messages: bool = True,
    memo: Optional[Dict[object, float]] = None,
) -> float:
    if _goal_reached(belief, ctx.static):
        return 0.0
    if depth <= 0:
        return _cut_value(belief, ctx.static, fail_cost)
    memo = memo if memo is not None else {}
    key = (
        ctx, _belief_key(belief), codec.name, codec.bits, tuple(skills), depth, predict_messages,
    )
    if key in memo:
        return memo[key]
    best = float("inf")
    for candidate in candidates.candidates:
        if predict_messages:
            immediate, branches = successors_codec(
                ctx, belief, candidate.joint, codec, skills, candidates
            )
            total = immediate
            for _obs, (prob, child) in branches.items():
                total += prob * codec_value(
                    ctx, child, codec, tuple(skills) + (candidate.joint,), candidates,
                    depth - 1, fail_cost=fail_cost, predict_messages=True, memo=memo,
                )
        else:
            immediate, branches = successors_public_only(ctx, belief, candidate.joint)
            total = immediate
            for _obs, (prob, child) in branches.items():
                total += prob * codec_value(
                    ctx, _as_pbelief(child), codec, tuple(skills) + (candidate.joint,),
                    candidates, depth - 1, fail_cost=fail_cost,
                    predict_messages=False, memo=memo,
                )
        if total < best:
            best = total
    memo[key] = best
    return best


def central_value(
    ctx: ContactContext,
    belief: CBelief,
    candidates: CandidateSet,
    depth: int,
    *,
    fail_cost: float = DEFAULT_FAIL_COST,
    memo: Optional[Dict[object, float]] = None,
) -> float:
    if _goal_reached(belief, ctx.static):
        return 0.0
    if depth <= 0:
        return _cut_value(belief, ctx.static, fail_cost)
    memo = memo if memo is not None else {}
    key = (ctx, _belief_key(belief), "central", depth)
    if key in memo:
        return memo[key]
    best = float("inf")
    for candidate in candidates.candidates:
        immediate, branches = successors_central(ctx, belief, candidate.joint)
        total = immediate
        for _obs, (prob, child) in branches.items():
            total += prob * central_value(
                ctx, child, candidates, depth - 1, fail_cost=fail_cost, memo=memo
            )
        if total < best:
            best = total
    memo[key] = best
    return best


def central_q_values(
    ctx: ContactContext,
    belief: CBelief,
    candidates: CandidateSet,
    depth: int,
    *,
    fail_cost: float = DEFAULT_FAIL_COST,
    memo: Optional[Dict[object, float]] = None,
) -> Dict[str, float]:
    """Per-action Q values under the exact belief-space backup.

    This is the equal-information reference: the planner does not know the hidden
    physics, but it **updates its belief along the lookahead**, branching on the joint
    observation. That is what lets it value an action that is informative as well as one
    that is productive. A value function that keeps the belief frozen cannot do this, and
    measured, such a planner waits in more than half of all decisions: with the risk
    penalty identical for a productive and an unproductive action, standing still looks
    as good as trying whenever the belief puts success below about one third.
    """

    memo = memo if memo is not None else {}
    out: Dict[str, float] = {}
    for candidate in candidates.candidates:
        immediate, branches = successors_central(ctx, belief, candidate.joint)
        total = immediate
        for _obs, (prob, child) in branches.items():
            total += prob * central_value(
                ctx, child, candidates, depth - 1, fail_cost=fail_cost, memo=memo
            )
        out[candidate.candidate_id] = total
    return out


def full_value(
    ctx: ContactContext,
    hyp: HypKey,
    candidates: CandidateSet,
    depth: int,
    *,
    fail_cost: float = DEFAULT_FAIL_COST,
    start_origin: Optional[Cell] = None,
    start_robots: Optional[Tuple[RobotState, ...]] = None,
) -> float:
    """Full-information optimum: the true physics is known."""

    def value(
        origin: Cell, robots: Tuple[RobotState, ...], hypothesis: HypKey, left: int
    ) -> float:
        if ObjectState(origin).within(ctx.static.target):
            return 0.0
        if left <= 0:
            # At the cut the value must still depend on the state, otherwise every
            # action looks equally bad, the cheapest one wins, and a "wait" (which is
            # free) becomes the argmin of a nominally value-optimal policy. The same
            # distance-to-goal heuristic used by the belief-space value functions is
            # applied here so all planners share one cut value.
            return _cut_value(
                {(origin, robots, hypothesis, (), ()): 1.0}, ctx.static, fail_cost
            )
        best = float("inf")
        for candidate in candidates.candidates:
            scene = Scene(
                static=ctx.static, obj=ObjectState(origin), robots=robots,
                physics=PHYSICS[hypothesis[0]], released=hypothesis[1],
            )
            successor, result = execute(scene, candidate.joint)
            total = result.cost + value(
                successor.obj.origin,
                successor.robots,
                (hypothesis[0], successor.released),
                left - 1,
            )
            if total < best:
                best = total
        return best

    return value(
        start_origin if start_origin is not None else ctx.origin,
        start_robots if start_robots is not None else ctx.robots,
        hyp,
        depth,
    )


# --------------------------------------------------------------------------
# Planner specification and action selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerSpec:
    kind: str  # public | codec | replan | limited | central | full
    codec: Optional[GridCodec] = None
    depth: int = 2

    @property
    def label(self) -> str:
        if self.kind in ("public", "codec", "replan", "limited"):
            codec = self.codec or NullGridCodec()
            return f"{self.kind}:{codec.name}:B{codec.bits}:d{self.depth}"
        return self.kind


def planner_spec(kind: str, bits: int = 1, depth: int = 2) -> PlannerSpec:
    """Build a planner from a short name, so all planners share one candidate set."""

    if kind == "public":
        return PlannerSpec("public", NullGridCodec(), depth)
    if kind == "codec":
        return PlannerSpec("codec", ModeGridCodec(bits), depth)
    if kind == "consequence":
        return PlannerSpec("codec", ConsequenceGridCodec(bits), depth)
    if kind == "observation":
        return PlannerSpec("codec", ObservationGridCodec(max(2, bits)), depth)
    if kind == "replan":
        return PlannerSpec("replan", ModeGridCodec(bits), depth)
    if kind == "limited":
        return PlannerSpec("limited", ModeGridCodec(bits), depth)
    if kind == "central":
        return PlannerSpec("central", None, depth)
    if kind == "full":
        return PlannerSpec("full", None, depth)
    raise ValueError(f"Unknown planner kind {kind!r}")


def choose_action(
    spec: PlannerSpec,
    ctx: ContactContext,
    belief: PBelief,
    candidates: CandidateSet,
    skills: Sequence[JointSkill] = (),
    *,
    fail_cost: float = DEFAULT_FAIL_COST,
    true_hyp: Optional[HypKey] = None,
    current_ctx: Optional[ContactContext] = None,
) -> Tuple[Optional[Candidate], float]:
    """Pick the first joint action for a planner class.

    ``ctx`` is the **initial** context and is what the codec uses to reconstruct a
    hypothesis' implied feedback history. ``current_ctx`` is the state as it is now,
    which is what the geometry and the full-information planner must use; without it
    a multi-step plan is evaluated from stale robot poses.
    """

    current = current_ctx if current_ctx is not None else ctx

    memo: Dict[object, float] = {}
    best_candidate: Optional[Candidate] = None
    best_value = float("inf")
    if _goal_reached(belief, ctx.static):
        return None, 0.0

    if spec.kind == "central":
        shared: CBelief = {}
        for (origin, robots, hyp, _a, _b), weight in belief.items():
            key = (origin, robots, hyp)
            shared[key] = shared.get(key, 0.0) + weight
        shared = _normalise(shared)  # type: ignore[assignment]
        for candidate in candidates.candidates:
            immediate, branches = successors_central(ctx, shared, candidate.joint)
            total = immediate
            for _obs, (prob, child) in branches.items():
                total += prob * central_value(
                    ctx, child, candidates, spec.depth - 1, fail_cost=fail_cost
                )
            if total < best_value:
                best_value, best_candidate = total, candidate
        return best_candidate, best_value

    if spec.kind == "full":
        if true_hyp is None:
            raise ValueError("The full-information planner needs the true hypothesis.")
        for candidate in candidates.candidates:
            successor, result = _execute_cached(current.scene(true_hyp), candidate.joint)
            total = result.cost + full_value(
                current, (true_hyp[0], successor.released), candidates,
                spec.depth - 1, fail_cost=fail_cost,
                start_origin=successor.obj.origin,
                start_robots=successor.robots,
            )
            if total < best_value:
                best_value, best_candidate = total, candidate
        return best_candidate, best_value

    predict = spec.kind != "replan"
    codec = spec.codec or NullGridCodec()
    for candidate in candidates.candidates:
        if predict:
            immediate, branches = successors_codec(
                ctx, belief, candidate.joint, codec, skills, candidates
            )
            total = immediate
            for _obs, (prob, child) in branches.items():
                total += prob * codec_value(
                    ctx, child, codec, tuple(skills) + (candidate.joint,), candidates,
                    spec.depth - 1, fail_cost=fail_cost, predict_messages=True, memo=memo,
                )
        else:
            immediate, branches = successors_public_only(ctx, belief, candidate.joint)
            total = immediate
            for _obs, (prob, child) in branches.items():
                total += prob * codec_value(
                    ctx, _as_pbelief(child), codec, tuple(skills) + (candidate.joint,),
                    candidates, spec.depth - 1, fail_cost=fail_cost,
                    predict_messages=False, memo=memo,
                )
        if total < best_value:
            best_value, best_candidate = total, candidate
    return best_candidate, best_value


# --------------------------------------------------------------------------
# Closed-loop evaluation (exact expectation over the deterministic protocol)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    planner: str
    expected_cost: float
    success_prob: float
    steps: float
    nodes: int


def _filtered_belief(
    spec: PlannerSpec,
    ctx: ContactContext,
    belief: PBelief,
    joint: JointSkill,
    realised,
    true_h_a: Tuple[str, ...],
    true_h_b: Tuple[str, ...],
    candidates: CandidateSet,
    skills: Sequence[JointSkill] = (),
) -> PBelief:
    """The planner's next public belief after the realised observation."""

    codec = spec.codec or NullGridCodec()
    keep_history = spec.kind in ("codec", "limited", "central")

    if spec.kind == "central":
        wanted = (realised.feedback[0], realised.feedback[1], realised.y_public)

        def matches(result) -> bool:
            return (
                result.feedback[0],
                result.feedback[1],
                result.y_public,
            ) == wanted

    elif spec.kind == "replan":
        def matches(result) -> bool:
            return result.y_public == realised.y_public

    else:
        full_skills = tuple(skills) + (joint,)
        new_h_a = true_h_a + (realised.feedback[0],)
        new_h_b = true_h_b + (realised.feedback[1],)
        wanted_messages = (
            codec.encode(ctx, full_skills, 0, new_h_a, candidates),
            codec.encode(ctx, full_skills, 1, new_h_b, candidates),
        )

        def matches(result) -> bool:
            if result.y_public != realised.y_public:
                return False
            return (
                codec.encode(ctx, full_skills, 0, true_h_a + (result.feedback[0],), candidates),
                codec.encode(ctx, full_skills, 1, true_h_b + (result.feedback[1],), candidates),
            ) == wanted_messages

    out: PBelief = {}
    for (origin, robots, hyp, h_a, h_b), weight in belief.items():
        scene = Scene(
            static=ctx.static, obj=ObjectState(origin), robots=robots,
            physics=PHYSICS[hyp[0]], released=hyp[1],
        )
        successor, result = _execute_cached(scene, joint)
        if not matches(result):
            continue
        new_hyp: HypKey = (hyp[0], successor.released)
        if keep_history:
            entry: PEntry = (
                successor.obj.origin,
                successor.robots,
                new_hyp,
                h_a + (result.feedback[0],),
                h_b + (result.feedback[1],),
            )
        else:
            entry = (successor.obj.origin, successor.robots, new_hyp, (), ())
        out[entry] = out.get(entry, 0.0) + weight
    return _normalise(out)  # type: ignore[return-value]


def evaluate_planner(
    spec: PlannerSpec,
    ctx: ContactContext,
    true_hyp: HypKey,
    *,
    horizon: int = 3,
    fail_cost: float = DEFAULT_FAIL_COST,
    initial_belief: Optional[PBelief] = None,
    public_candidates: bool = False,
    candidate_ids: Optional[Sequence[str]] = None,
    model_hyp: Optional[HypKey] = None,
) -> EpisodeResult:
    """Evaluate a receding-horizon planner.

    ``true_hyp`` is the physics the *environment* executes. ``model_hyp`` is the
    physics the *planner* uses; when they differ the planner is running with a wrong
    world model at the same information level, which is how the model-error gap is
    measured (as opposed to the value-of-information gap, which is irreducible).
    """
    """Exact closed-loop expected cost of a receding-horizon planner.

    The controller is re-planned at every boundary with its own (possibly
    message-blind) lookahead, while execution always follows the real protocol: the
    real feedback arrives, the same codec produces the real messages, and the public
    belief is filtered accordingly. All quantities are deterministic in Stage 2, so
    this is an exact expectation, not a Monte-Carlo estimate.
    """

    nodes = [0]

    def rollout(
        belief: PBelief,
        state_ctx: ContactContext,
        hyp: HypKey,
        true_h_a: Tuple[str, ...],
        true_h_b: Tuple[str, ...],
        t: int,
        skills: Tuple[JointSkill, ...],
    ) -> Tuple[float, float, float]:
        nodes[0] += 1
        if ObjectState(state_ctx.origin).within(ctx.static.target):
            return 0.0, 1.0, 0.0
        if t >= horizon:
            return fail_cost, 0.0, 0.0

        # Candidates are regenerated from public information at every boundary, so
        # both robots always agree on the ID space even as the geometry changes.
        candidates = contact_candidates(state_ctx, public=public_candidates)
        if candidate_ids is not None:
            allowed = set(candidate_ids)
            candidates = CandidateSet(
                tuple(c for c in candidates.candidates if c.candidate_id in allowed)
            )
        chosen, _value = choose_action(
            spec, ctx, belief, candidates, skills, fail_cost=fail_cost,
            # Vary the model's *physics parameters* only: the released flag is an
            # observed contact state and is held equal to the environment's.
            true_hyp=((model_hyp[0], hyp[1]) if model_hyp is not None else hyp),
            current_ctx=state_ctx,
        )
        if chosen is None:
            return fail_cost, 0.0, 0.0

        successor_true, realised = _execute_cached(state_ctx.scene(hyp), chosen.joint)
        new_hyp: HypKey = (hyp[0], successor_true.released)
        new_ctx = ContactContext(
            static=ctx.static,
            origin=successor_true.obj.origin,
            robots=successor_true.robots,
        )
        next_belief = _filtered_belief(
            spec, state_ctx, belief, chosen.joint, realised, true_h_a, true_h_b,
            candidates, skills,
        )
        cost, success, steps = rollout(
            next_belief,
            new_ctx,
            new_hyp,
            true_h_a + (realised.feedback[0],),
            true_h_b + (realised.feedback[1],),
            t + 1,
            skills + (chosen.joint,),
        )
        if not next_belief:
            # The observation is impossible under every surviving hypothesis; the
            # team cannot recover. Charge the failure cost explicitly.
            return realised.cost + fail_cost, 0.0, 1.0
        return realised.cost + cost, success, 1.0 + steps

    belief = initial_belief if initial_belief is not None else initial_contact_belief(ctx)
    expected_cost, success, steps = rollout(belief, ctx, true_hyp, (), (), 0, ())
    return EpisodeResult(
        planner=spec.label,
        expected_cost=expected_cost,
        success_prob=success,
        steps=steps,
        nodes=nodes[0],
    )
