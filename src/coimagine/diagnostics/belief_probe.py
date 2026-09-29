"""Does the factored model actually estimate the hidden parameters?

Training loss is not the question. The questions are:

1. **Is the learned likelihood right?** The observation model must reproduce the structure the
   simulator has: after a contact action, ``slipped`` means this robot slipped, and anything else
   means it gripped. If it does not, the Bayes update built on it is meaningless.
2. **Does the belief update concentrate on the truth?** Run the sequential update
   ``b'(θ) ∝ P(y | a, s, θ) b(θ)`` over a real sequence of actions and feedback and check the
   posterior against the true parameters.
3. **Does it refuse to claim what it cannot know?** The teammate's traction is not reachable
   through this robot's own feedback, so the posterior over it must stay near the prior. A model
   that confidently guesses it is worse than one that stays uncertain.

The reference for (2) is not another learned number: `own traction` and `the object's constraint
mode` have exact identifiability structure that was measured directly from the simulator, so the
comparison is against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..grid.dynamics import FEEDBACK_SYMBOLS, execute
from ..grid.state import enumerate_hidden_physics
from ..learning.autograd import Tensor, concat
from ..learning.belief_model import BeliefWorldModel, observation_symbol_index
from ..learning.data import _joint_for, generate_situations, pose_to_index
from ..learning.features import ACTION_IDS, observation_vector
from ..learning.world_model import N_ACTIONS, N_CELLS, N_PHYSICS, Trajectory
from ..learning.world_model_data import collect_trajectories
from ..grid.sensing import apply_noise, ray_cast, SensorNoise

PHYSICS_TABLE = tuple(enumerate_hidden_physics())
#: The hidden quantities the belief is marginalised into, and how to read them off a θ index.
PARAMETER_VIEWS: Dict[str, Tuple[str, ...]] = {
    "own traction": ("grip", "slip"),
    "teammate traction": ("grip", "slip"),
    "resistance": ("light", "heavy"),
    "constraint": ("free", "latched"),
}


def theta_index(physics) -> int:
    return PHYSICS_TABLE.index(physics)


def marginal(belief: np.ndarray, view: str) -> np.ndarray:
    """Marginalise a belief over the 16 combinations into a named two-valued view."""

    out = np.zeros(2)
    for index, physics in enumerate(PHYSICS_TABLE):
        if view == "own traction":
            value = 0 if physics.traction[0] == "grip" else 1
        elif view == "teammate traction":
            value = 0 if physics.traction[1] == "grip" else 1
        elif view == "resistance":
            value = 0 if physics.resistance == 1 else 1
        elif view == "constraint":
            value = 0 if physics.constraint == "free" else 1
        else:
            raise KeyError(view)
        out[value] += belief[index]
    return out


def true_value(physics, view: str) -> int:
    if view == "own traction":
        return 0 if physics.traction[0] == "grip" else 1
    if view == "teammate traction":
        return 0 if physics.traction[1] == "grip" else 1
    if view == "resistance":
        return 0 if physics.resistance == 1 else 1
    return 0 if physics.constraint == "free" else 1


@dataclass
class BeliefProbe:
    #: Post-action accuracy of the marginal, per number of contact actions taken.
    accuracy: Dict[str, List[float]] = field(default_factory=dict)
    #: Mean posterior probability mass on the truth, same indexing.
    mass_on_truth: Dict[str, List[float]] = field(default_factory=dict)
    #: How much the posterior moves away from uniform: 0 means "learned nothing".
    moved: Dict[str, List[float]] = field(default_factory=dict)
    #: Fraction of runs where the model named the wrong value with confidence above 0.8.
    confident_errors: Dict[str, float] = field(default_factory=dict)
    steps: int = 0
    runs: int = 0


def run_belief_probe(
    model: BeliefWorldModel,
    *,
    n_runs: int = 120,
    horizon: int = 4,
    seed: int = 55_000,
    radius: int = 5,
) -> BeliefProbe:
    """Follow real episodes and update the belief from the real feedback each step."""

    situations = generate_situations(
        n_scenes=n_runs, lead_in=2, seed=seed, physics_pool=PHYSICS_TABLE
    )
    views = list(PARAMETER_VIEWS)
    probe = BeliefProbe(
        accuracy={v: [0.0] * horizon for v in views},
        mass_on_truth={v: [0.0] * horizon for v in views},
        moved={v: [0.0] * horizon for v in views},
        steps=horizon,
        runs=len(situations),
    )
    confident = {v: 0 for v in views}
    counts = [0] * horizon
    random = np.random.default_rng(seed)
    noise = SensorNoise()

    for situation in situations:
        scene = situation.scene
        history_a = list(situation.history_a)
        public_history = list(situation.public_history)
        belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)  # a uniform prior, deliberately
        for step in range(horizon):
            if scene.is_goal():
                break
            scan = apply_noise(ray_cast(scene, 0, radius=radius), noise, random)
            observation = np.asarray(
                observation_vector(
                    scan, scene.robot_pose(0), tuple(history_a), tuple(public_history)
                )
            )
            dynamic = model.dynamic_state(observation)
            # act with a policy that makes contact, so the evidence is informative
            action_id = "co_push:E" if step == 0 else "push:0:E"
            action = ACTION_IDS.index(action_id)
            successor, result = execute(scene, _joint_for(scene, action_id))
            symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
            # the belief update uses the imagined next dynamic state, as it would when planning
            belief = model.update_belief(belief, dynamic, action, symbol)

            counts[step] += 1
            for view in views:
                posterior = marginal(belief[0], view)
                truth = true_value(scene.physics, view)
                probe.accuracy[view][step] += float(posterior.argmax() == truth)
                probe.mass_on_truth[view][step] += float(posterior[truth])
                probe.moved[view][step] += float(abs(posterior - 0.5).sum())
                if posterior.argmax() != truth and posterior.max() > 0.8:
                    confident[view] += 1

            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            scene = successor

    for view in views:
        for step in range(horizon):
            count = max(1, counts[step])
            probe.accuracy[view][step] /= count
            probe.mass_on_truth[view][step] /= count
            probe.moved[view][step] /= count
        probe.confident_errors[view] = confident[view] / max(1, sum(counts))
    return probe


def format_belief_probe(probe: BeliefProbe) -> str:
    lines = [
        "Posterior over each hidden quantity after 1..N contact actions, starting from a "
        "**uniform prior** and updated only through the learned likelihood. `mass` is the "
        "probability placed on the true value (0.5 = no information). `moved` is the total "
        "variation from uniform (0 = learned nothing).",
        "",
        "| hidden quantity | " + " | ".join(f"after {i + 1}" for i in range(probe.steps)) + " |",
        "|---" * (probe.steps + 1) + "|",
    ]
    for view in probe.accuracy:
        cells = " | ".join(
            f"acc {probe.accuracy[view][i]:.2f} / mass {probe.mass_on_truth[view][i]:.2f}"
            for i in range(probe.steps)
        )
        lines.append(f"| {view} | {cells} |")
    lines += [
        "",
        "| hidden quantity | total variation from uniform | confident errors (wrong with p>0.8) |",
        "|---|---|---|",
    ]
    for view in probe.moved:
        final = probe.moved[view][probe.steps - 1]
        lines.append(f"| {view} | {final:.3f} | {probe.confident_errors[view]:.3f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Is the learned likelihood right?
# ---------------------------------------------------------------------------


@dataclass
class LikelihoodRow:
    action: str
    view: str
    symbol: str
    truth: str
    learned: float
    empirical: float


def compare_likelihoods(
    model: BeliefWorldModel,
    *,
    n_scenes: int = 400,
    seed: int = 77_000,
    actions: Sequence[str] = ("push:0:E", "release:0:E", "co_push:E"),
    view: str = "own traction",
    symbol: str = "slipped",
) -> List[LikelihoodRow]:
    """The learned ``P(symbol | action, θ)`` against the simulator's own frequency.

    The empirical column is measured by running the simulator over scenes of every parameter
    combination; the learned column is what the observation model says for the same θ, averaged
    over the dynamic states the encoder infers from those scenes.
    """

    rows: List[LikelihoodRow] = []
    for action_id in actions:
        index = ACTION_IDS.index(action_id)
        for value, label in ((0, PARAMETER_VIEWS[view][0]), (1, PARAMETER_VIEWS[view][1])):
            scenes = []
            for physics in PHYSICS_TABLE:
                if true_value(physics, view) != value:
                    continue
                scenes.extend(
                    s.scene
                    for s in generate_situations(
                        n_scenes=max(2, n_scenes // 16), lead_in=2, seed=seed,
                        physics_pool=(physics,),
                    )
                    if not s.scene.is_goal()
                )
            empirical_hits = 0
            learned_total = 0.0
            symbol_index = FEEDBACK_SYMBOLS.index(symbol)
            random = np.random.default_rng(seed)
            for scene in scenes:
                scan = apply_noise(ray_cast(scene, 0, radius=5), SensorNoise(), random)
                observation = np.asarray(
                    observation_vector(scan, scene.robot_pose(0), (), ())
                )
                dynamic = model.dynamic_state(observation)
                belief = np.zeros((1, N_PHYSICS))
                belief[0, theta_index(scene.physics)] = 1.0
                # The symbol is the *outcome* of the action, so it is conditioned on the state
                # the transition produces -- the same state the training objective used. Feeding
                # the pre-action state here was a bug in this probe and made the learned
                # likelihood look far weaker than it is.
                next_dynamic = model.step(dynamic, index, belief)
                # marginalise over the belief rather than reading one arbitrary parameter
                # setting; the belief is a one-hot on this scene's true physics
                per_theta = model.observe(next_dynamic, index, belief)
                likelihood = belief[0] @ per_theta[0]
                learned_total += float(likelihood[symbol_index])
                _, result = execute(scene, _joint_for(scene, action_id))
                empirical_hits += int(result.feedback[0] == symbol)
            n = max(1, len(scenes))
            rows.append(
                LikelihoodRow(
                    action=action_id,
                    view=view,
                    symbol=symbol,
                    truth=label,
                    learned=learned_total / n,
                    empirical=empirical_hits / n,
                )
            )
    return rows


def format_likelihoods(rows: Sequence[LikelihoodRow]) -> str:
    lines = [
        "The learned likelihood beside the simulator's own frequency, for the same parameter "
        "values. This is the quantity the belief update is built on, so it is checked directly "
        "rather than inferred from a training loss.",
        "",
        "| action | parameter value | P(symbol | action, value): learned | empirical |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.action} | {row.view} = {row.truth} | {row.learned:.3f} | {row.empirical:.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planning with belief-space imagination
# ---------------------------------------------------------------------------

#: How much a step of imagined progress is worth, matching the exact planner's weight so the
#: comparison is about the model rather than about a differently tuned objective.
PROGRESS_WEIGHT = 3.0
FAIL_COST = 20.0
#: Branch only on symbols the model thinks are plausible; the tail contributes nothing but cost.
BRANCH_THRESHOLD = 0.02


def _cell_distance_distribution(cells: np.ndarray, goal: frozenset) -> np.ndarray:
    """Worst-cell distance to the goal for every object cell, weighted by its probability."""

    out = np.zeros(N_CELLS)
    for index in range(N_CELLS):
        origin = (index % 6, index // 6)
        members = [(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)]
        # Worst cell's distance to its **nearest** goal cell: max over the object's cells of the
        # min over goal cells. Both directions were inverted here -- it computed the *minimum*
        # over cells of the distance to the *farthest* goal cell -- which inflated the heuristic
        # roughly two-fold and, worse, made it depend on the object's y coordinate even though
        # the goal spans every row. The table is the search's only guidance past the horizon, so
        # that error propagates straight into every branch score.
        out[index] = float(
            max(min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in members)
        )
    return out


def belief_from_history(
    model: BeliefWorldModel,
    observation: np.ndarray,
    dynamic: np.ndarray,
    history: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """The parameter belief by **explicit Bayes over the real action/feedback history**.

    Deliberately *not* the encoder's parameter head. Measured on a scene whose true physics is
    (light, latched, this robot grips), the encoder put 0.973 on "this robot slips" -- confidently
    wrong -- while the explicit filter built on the learned likelihood identified own traction
    perfectly. The encoder head is a shortcut that fits spurious correlations in the lead-in
    history; the likelihood is the part that was verified against the simulator, so the update is
    done analytically from it, exactly as the repository's exact reference does.
    """

    belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    state = dynamic
    for action, symbol in history:
        # the state handed in is the one *before* the action; the update derives each
        # hypothesis's own next state internally
        belief = model.update_belief(belief, state, action, symbol)
        state = model.step(state, action, belief)
    return belief


def plan_with_belief(
    model: BeliefWorldModel,
    observation: np.ndarray,
    goal: frozenset,
    *,
    history: Optional[Sequence[Tuple[int, int]]] = None,
    belief_override: Optional[np.ndarray] = None,
    dynamic_override: Optional[np.ndarray] = None,
    depth: int = 2,
    beam: int = 1,
    progress_weight: float = PROGRESS_WEIGHT,
    fail_cost: float = FAIL_COST,
) -> Tuple[int, List[Dict[str, object]]]:
    """Two-level lookahead, delegating to :func:`plan_with_search`.

    This used to carry its own copy of the scoring, and that copy had four defects. The belief was
    updated on the imagined feedback while the continuation still started from the state computed
    under the prior. The failure cost was multiplied by ``0.0`` in a leftover expression, so it was
    charged flat rather than by the probability of failure. The second action's cost was never
    counted, making it free. And the failure cost was then charged a second time. Keeping one
    scoring path removes the whole class of problem, and it is the only way the two planners can be
    compared at all.
    """

    dynamic = (
        np.atleast_2d(dynamic_override)
        if dynamic_override is not None
        else model.dynamic_state(observation)
    )
    if belief_override is not None:
        belief = np.atleast_2d(belief_override)
    elif history is None:
        belief = model.parameter_belief(observation)
    else:
        belief = belief_from_history(model, observation, dynamic, history)
    return plan_with_search(
        model, observation, goal, belief=belief, dynamic=dynamic,
        depth=depth, beam=beam, progress_weight=progress_weight, fail_cost=fail_cost,
    )

@dataclass
class PrepareProbe:
    """Does the planner prepare, and does it do so only when preparation is warranted?"""

    latched_release_rate: float
    free_release_rate: float
    latched_n: int
    free_n: int
    latched_belief_mass: float
    free_belief_mass: float
    #: Split by whether releasing can work at all, which is the breakdown that matters. A latched
    #: object that this robot cannot release is as unprepared-for as a free one: the correct rate
    #: there is zero, and only the "latched and gripped" column should be high.
    latched_releasable_n: int = 0
    latched_releasable_rate: float = float("nan")
    latched_unreleasable_n: int = 0
    latched_unreleasable_rate: float = float("nan")


def run_prepare_probe(
    model: BeliefWorldModel,
    *,
    n_scenes: int = 240,
    seed: int = 55_000,
    lead_in: int = 1,
    oracle_belief: bool = False,
) -> PrepareProbe:
    """Ask the planner for its action once the belief has been earned by real contact.

    The probe drives its own lead-in rather than reading a stored history: the Bayes update needs
    the ``(action, symbol)`` pair that produced each feedback symbol, and a stored symbol alone
    does not determine it. Driving the actions makes the pairs known by construction, which is
    also how the planner is used for real.

    The question is whether it spends a step releasing a **latched** object and does not waste one
    on a free object. The belief mass on `latched` is reported beside the choice, because a
    planner that releases blindly demonstrates nothing.
    """

    situations = generate_situations(
        n_scenes=n_scenes, lead_in=0, seed=seed, physics_pool=PHYSICS_TABLE
    )
    latched_choices: List[int] = []
    free_choices: List[int] = []
    releasable_choices: List[int] = []
    unreleasable_choices: List[int] = []
    latched_mass: List[float] = []
    free_mass: List[float] = []
    random = np.random.default_rng(seed)
    noise = SensorNoise()
    release = ACTION_IDS.index("release:0:E")

    for situation in situations:
        scene = situation.scene
        if scene.is_goal():
            continue
        belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
        # The dynamic state must be computed even when there is no lead-in, otherwise a probe
        # with lead_in=0 skips every scene and silently reports NaN.
        dynamic = model.dynamic_state(
            np.asarray(
                observation_vector(
                    ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ()
                )
            )
        )
        # earn the belief with real contact: the action and its symbol are both known here
        for _ in range(lead_in):
            scan = apply_noise(ray_cast(scene, 0, radius=5), noise, random)
            observation = np.asarray(
                observation_vector(scan, scene.robot_pose(0), (), ())
            )
            dynamic = model.dynamic_state(observation)
            action = ACTION_IDS.index("probe:0:E")
            successor, result = execute(scene, _joint_for(scene, "probe:0:E"))
            symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
            belief = model.update_belief(belief, dynamic, action, symbol)
            scene = successor
            if scene.is_goal():
                break
        if scene.is_goal():
            continue
        scan = apply_noise(ray_cast(scene, 0, radius=5), noise, random)
        observation = np.asarray(observation_vector(scan, scene.robot_pose(0), (), ()))
        goal = frozenset(scene.static.target)
        # With `oracle_belief` the planner is handed the true parameters. That separates *model
        # quality* from *information*: if it prepares when it knows and not when it does not, the
        # model and the planner are correct and the residual gap is the value of information.
        handed = belief
        if oracle_belief:
            handed = np.zeros((1, N_PHYSICS))
            handed[0, theta_index(scene.physics)] = 1.0
        chosen, _branches = plan_with_belief(
            model, observation, goal, belief_override=handed
        )
        mass = float(marginal(handed[0], "constraint")[1])
        if scene.physics.constraint == "constrained" and not scene.released:
            latched_choices.append(chosen)
            latched_mass.append(mass)
            if scene.physics.traction[0] == "grip":
                releasable_choices.append(chosen)
            else:
                unreleasable_choices.append(chosen)
        elif scene.physics.constraint == "free":
            free_choices.append(chosen)
            free_mass.append(mass)

    def rate(choices: Sequence[int]) -> float:
        return float(np.mean([c == release for c in choices])) if choices else float("nan")

    return PrepareProbe(
        latched_release_rate=rate(latched_choices),
        free_release_rate=rate(free_choices),
        latched_n=len(latched_choices),
        free_n=len(free_choices),
        latched_belief_mass=float(np.mean(latched_mass)) if latched_mass else float("nan"),
        free_belief_mass=float(np.mean(free_mass)) if free_mass else float("nan"),
        latched_releasable_n=len(releasable_choices),
        latched_releasable_rate=rate(releasable_choices),
        latched_unreleasable_n=len(unreleasable_choices),
        latched_unreleasable_rate=rate(unreleasable_choices),
    )


def format_prepare_probe(probe: PrepareProbe) -> str:
    return "\n".join(
        [
            "Does the planner spend its first step preparing, and only where that is warranted?",
            "",
            "| situation | scenes | chose `release` first | belief mass on `latched` |",
            "|---|---|---|---|",
            f"| latched, not yet released | {probe.latched_n} | **{probe.latched_release_rate:.3f}** | "
            f"{probe.latched_belief_mass:.3f} |",
            f"| free | {probe.free_n} | {probe.free_release_rate:.3f} | "
            f"{probe.free_belief_mass:.3f} |",
            f"| latched **and this robot grips** (releasing can work) | "
            f"{probe.latched_releasable_n} | **{probe.latched_releasable_rate:.3f}** | - |",
            f"| latched **and this robot slips** (releasing cannot work) | "
            f"{probe.latched_unreleasable_n} | **{probe.latched_unreleasable_rate:.3f}** | - |",
        ]
    )


# ---------------------------------------------------------------------------
# Does the parameter belief change the *multi-step* prediction?
# ---------------------------------------------------------------------------


@dataclass
class MultiStepSensitivity:
    """How much the imagined future depends on the parameter belief.

    The single-step check (``P(released | release)`` conditioned on the parameters) is necessary
    but not sufficient: a planner needs the parameters to change what it expects several steps
    ahead. This measures the imagined object position and released probability ``k`` steps out,
    under two different beliefs, from the same observation.
    """

    steps: int
    cell_divergence: List[float] = field(default_factory=list)
    released_gap: List[float] = field(default_factory=list)
    best_action_agreement: List[float] = field(default_factory=list)


def measure_multistep_sensitivity(
    model: BeliefWorldModel,
    *,
    n_scenes: int = 200,
    steps: int = 3,
    seed: int = 55_000,
    sequence: Sequence[str] = ("release:0:E", "push:0:E", "push:0:E"),
) -> MultiStepSensitivity:
    """Imagine the same action sequence under the true parameters and under a wrong belief.

    If the parameters only mattered for one step, the two imaginations would converge; if they
    matter, the divergence should persist and grow with depth.
    """

    situations = generate_situations(
        n_scenes=n_scenes, lead_in=0, seed=seed, physics_pool=PHYSICS_TABLE
    )
    actions = [ACTION_IDS.index(name) for name in sequence]
    out = MultiStepSensitivity(steps=steps)
    out.cell_divergence = [0.0] * steps
    out.released_gap = [0.0] * steps
    out.best_action_agreement = [0.0] * steps
    counts = 0
    noise = SensorNoise()
    random = np.random.default_rng(seed)

    for situation in situations:
        scene = situation.scene
        if scene.is_goal():
            continue
        observation = np.asarray(
            observation_vector(
                apply_noise(ray_cast(scene, 0, radius=5), noise, random),
                scene.robot_pose(0), (), (),
            )
        )
        dynamic = model.dynamic_state(observation)
        # the true parameters, and a belief that is wrong about them on purpose
        true_belief = np.zeros((1, N_PHYSICS))
        true_belief[0, theta_index(scene.physics)] = 1.0
        wrong = scene.physics
        for candidate in PHYSICS_TABLE:  # first combination with different constraint + traction
            if (candidate.constraint != wrong.constraint
                    and candidate.traction[0] != wrong.traction[0]):
                wrong = candidate
                break
        wrong_belief = np.zeros((1, N_PHYSICS))
        wrong_belief[0, theta_index(wrong)] = 1.0

        goals = frozenset(scene.static.target)
        distance_table = _cell_distance_distribution(None, goals)
        true_state, wrong_state = dynamic, dynamic
        for depth in range(steps):
            action = actions[min(depth, len(actions) - 1)]
            true_state = model.step(true_state, action, true_belief)
            wrong_state = model.step(wrong_state, action, wrong_belief)
            out.cell_divergence[depth] += float(
                0.5 * np.abs(true_state[0, 1 : 1 + N_CELLS] - wrong_state[0, 1 : 1 + N_CELLS]).sum()
            )
            out.released_gap[depth] += abs(float(true_state[0, 0] - wrong_state[0, 0]))
            # do the two beliefs agree on which continuation is best?
            def best(state, belief):
                values = []
                for second in range(len(ACTION_IDS)):
                    nxt = model.step(state, second, belief)
                    values.append(float(nxt[0, 1 : 1 + N_CELLS] @ distance_table))
                return int(np.argmin(values))
            out.best_action_agreement[depth] += float(
                best(true_state, true_belief) == best(wrong_state, wrong_belief)
            )
        counts += 1

    n = max(1, counts)
    out.cell_divergence = [v / n for v in out.cell_divergence]
    out.released_gap = [v / n for v in out.released_gap]
    out.best_action_agreement = [v / n for v in out.best_action_agreement]
    return out


def format_multistep_sensitivity(result: MultiStepSensitivity) -> str:
    lines = [
        "The same action sequence imagined under the **true** parameters and under a belief that "
        "is wrong about the constraint mode and this robot's traction. `cell divergence` is the "
        "total variation between the two imagined object-cell distributions; `best continuation "
        "agreement` is how often the two beliefs pick the same next action.",
        "",
        "| step | cell divergence | released gap | best continuation agreement |",
        "|---|---|---|---|",
    ]
    for depth in range(result.steps):
        lines.append(
            f"| {depth + 1} | {result.cell_divergence[depth]:.3f} | "
            f"{result.released_gap[depth]:.3f} | "
            f"{result.best_action_agreement[depth]:.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Closed-loop comparison in the task itself
# ---------------------------------------------------------------------------


def fuse_hidden_state(observed_dynamic: np.ndarray, carried_dynamic: np.ndarray) -> np.ndarray:
    """Combine what the observation can see with what only the transition knows.

    The object's cell and the robot positions are observable, so they are taken from the current
    observation. ``released`` is **hidden**: a successful release leaves the observation
    byte-identical, so re-reading it from the observation forgets that it happened, and a planner
    that forgets keeps predicting that pushes fail and idles forever. It is carried forward by the
    transition instead, exactly like the parameter belief.
    """

    fused = np.array(observed_dynamic, copy=True)
    fused[:, 0] = carried_dynamic[:, 0]
    return fused


def run_belief_episode(
    model: BeliefWorldModel,
    scene,
    belief: np.ndarray,
    *,
    horizon: int = 8,
    radius: int = 5,
    depth: int = 2,
    warmup: Optional[Tuple[int, int]] = None,
) -> Tuple[int, float, int]:
    """One episode where the planner imagines with the model and the belief is carried along.

    ``warmup`` is an ``(action, symbol)`` pair the robot already experienced before this episode
    began; it seeds the belief. Without it the belief starts uniform, which is the honest
    zero-information case.
    """

    goal = frozenset(scene.static.target)
    history_a: List[str] = []
    public_history: List[str] = []
    carried: Optional[np.ndarray] = None  # the hidden part, propagated by the transition

    def observe(current) -> np.ndarray:
        observation = np.asarray(
            observation_vector(
                ray_cast(current, 0, radius=radius), current.robot_pose(0),
                tuple(history_a), tuple(public_history),
            )
        )
        observed = model.dynamic_state(observation)
        return observed if carried is None else fuse_hidden_state(observed, carried)

    if warmup is not None:
        action, symbol = warmup
        dynamic = observe(scene)
        belief = model.update_belief(belief, dynamic, action, symbol)
        carried = model.step(dynamic, action, belief)
        history_a.append(FEEDBACK_SYMBOLS[symbol])

    total = 0.0
    steps = 0
    for _ in range(horizon):
        if scene.is_goal():
            break
        dynamic = observe(scene)
        observation = np.asarray(
            observation_vector(
                ray_cast(scene, 0, radius=radius), scene.robot_pose(0),
                tuple(history_a), tuple(public_history),
            )
        )
        chosen, _branches = plan_with_belief(
            model, observation, goal, belief_override=belief, depth=depth,
            dynamic_override=dynamic,
        )
        successor, result = execute(scene, _joint_for(scene, ACTION_IDS[chosen]))
        total += float(result.cost)
        steps += 1
        symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
        belief = model.update_belief(belief, dynamic, chosen, symbol)
        carried = model.step(dynamic, chosen, belief)
        history_a.append(result.feedback[0])
        public_history.append(result.y_public)
        scene = successor
    return int(scene.is_goal()), total, steps


@dataclass
class ComparisonRow:
    policy: str
    episodes: int
    success_rate: float
    mean_cost: float
    cost_per_success: float
    mean_steps: float


def label_solvable(episodes, *, horizon: int = 16, warmup_action: str = "probe:0:E"):
    """Which scenes the full-information oracle can solve at all, given a generous horizon.

    Needed because the success metric has a ceiling here: the oracle scores 0.417-0.425 and does
    **not** improve when the horizon goes from 8 to 12 to 16, so about 58% of episodes are
    physically unsolvable. Comparing planners on raw success therefore measures mostly how they
    behave on unwinnable episodes.
    """

    from .rollout_probe import run_episode

    flags = []
    for situation in episodes:
        scene = situation.scene
        if scene.is_goal():
            flags.append(False)
            continue
        successor, _ = execute(scene, _joint_for(scene, warmup_action))
        if successor.is_goal():
            flags.append(False)
            continue
        success, _cost, _steps = run_episode(
            successor, "oracle_true_state", None, horizon=horizon,
            rng=np.random.default_rng(0),
        )
        flags.append(bool(success))
    return flags


def run_belief_comparison(
    model: BeliefWorldModel,
    *,
    n_episodes: int = 60,
    horizon: int = 8,
    seed: int = 55_000,
    depth: int = 4,
    beam: int = 3,
    warmup_action: str = "probe:0:E",
    solvable_only: bool = False,
    oracle_probe_horizon: int = 16,
) -> List[ComparisonRow]:
    """Every policy on the **same** scenes, from the **same** starting state.

    All policies first perform one common warm-up action, so nobody gets a different initial
    configuration and the learned planner has one real ``(action, symbol)`` pair to seed its
    belief with. Measurement starts after the warm-up.
    """

    from .rollout_probe import run_episode

    situations = generate_situations(
        n_scenes=n_episodes, lead_in=0, seed=seed, physics_pool=PHYSICS_TABLE
    )
    if solvable_only:
        keep = label_solvable(
            situations, horizon=oracle_probe_horizon, warmup_action=warmup_action
        )
        situations = [s for s, k in zip(situations, keep) if k]
    tallies: Dict[str, List[Tuple[int, float, int]]] = {}
    random = np.random.default_rng(seed)
    warm_index = ACTION_IDS.index(warmup_action)

    for situation in situations:
        scene = situation.scene
        if scene.is_goal():
            continue
        successor, result = execute(scene, _joint_for(scene, warmup_action))
        if successor.is_goal():
            continue
        symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
        belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)

        tallies.setdefault("belief model + belief-space search", []).append(
            run_search_episode(
                model, successor, belief, horizon=horizon, depth=depth, beam=beam,
                warmup=(warm_index, symbol),
            )
        )
        tallies.setdefault("belief model, two-level lookahead", []).append(
            run_belief_episode(
                model, successor, belief, horizon=horizon, depth=2,
                warmup=(warm_index, symbol),
            )
        )
        # the full-information ceiling and the equal-information exact reference, from the same
        # state, via the closed loop already used elsewhere in the project
        for name, policy in (
            ("oracle (true state + true model)", "oracle_true_state"),
            ("exact belief-space reference", "true_model_belief"),
        ):
            success, cost, steps = run_episode(
                successor, policy, model, horizon=horizon,
                rng=np.random.default_rng(0),
            )
            tallies.setdefault(name, []).append((success, cost, steps))
        # a random-action baseline on the same scenes
        current = successor
        total, steps = 0.0, 0
        for _ in range(horizon):
            if current.is_goal():
                break
            action_id = ACTION_IDS[int(random.integers(len(ACTION_IDS)))]
            current, outcome = execute(current, _joint_for(current, action_id))
            total += float(outcome.cost)
            steps += 1
        tallies.setdefault("random", []).append((int(current.is_goal()), total, steps))

    rows: List[ComparisonRow] = []
    for name, records in tallies.items():
        rate = float(np.mean([r[0] for r in records]))
        mean_cost = float(np.mean([r[1] for r in records]))
        rows.append(
            ComparisonRow(
                policy=name,
                episodes=len(records),
                success_rate=rate,
                mean_cost=mean_cost,
                cost_per_success=(mean_cost / rate if rate > 0 else float("inf")),
                mean_steps=float(np.mean([r[2] for r in records])),
            )
        )
    return rows


def format_comparison(rows: Sequence[ComparisonRow]) -> str:
    lines = [
        "| planner | episodes | success | mean cost | cost per success | mean steps |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        per = "n/a" if row.cost_per_success == float("inf") else f"{row.cost_per_success:.2f}"
        lines.append(
            f"| {row.policy} | {row.episodes} | {row.success_rate:.3f} | "
            f"{row.mean_cost:.3f} | {per} | {row.mean_steps:.2f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Depth-limited belief-space rollout
# ---------------------------------------------------------------------------


def _cut_value(
    state: np.ndarray, distance_table: np.ndarray, progress_weight: float, fail_cost: float
) -> float:
    """Value of an imagined state at the search horizon.

    Progress is the expected distance of the object from the goal under the imagined cell
    distribution, and the failure cost is charged once when the object is not home. Charging it
    per branch instead would add the same constant everywhere and drown the progress term, which
    is what made the earlier two-level planner prefer cheap neutral actions.
    """

    cells = state[0, 1 : 1 + N_CELLS]
    distance = float(cells @ distance_table)
    # The failure cost is charged in proportion to the probability that the object is **not**
    # home, not flat. Charging it flat whenever the *expected* distance is above zero means a
    # branch with a 79% chance of having finished still pays the full penalty, which makes every
    # branch look equally bad and lets a free `wait` win. The exact reference weights it by the
    # probability of failure, and this is the same quantity.
    at_goal = float(cells[distance_table < 0.5].sum())
    return progress_weight * distance + fail_cost * (1.0 - at_goal)


def _beam_value(
    model: BeliefWorldModel,
    state: np.ndarray,
    belief: np.ndarray,
    distance_table: np.ndarray,
    *,
    steps: int,
    beam: int,
    progress_weight: float,
    fail_cost: float,
) -> float:
    """Best value reachable in ``steps`` actions, keeping the ``beam`` best partial plans.

    Every level accumulates the **cost of the actions taken** and only the surviving plans are
    expanded, which is what makes a horizon of three or four affordable: a full tree would be
    ``6**steps`` expansions per branch.
    """

    states = [np.atleast_2d(state)]
    costs = [0.0]
    for _ in range(steps):
        candidates: List[np.ndarray] = []
        candidate_costs: List[float] = []
        for partial, spent in zip(states, costs):
            next_states = model.step_actions(partial, belief)
            next_costs = model.cost_actions(next_states, belief)
            for action in range(len(ACTION_IDS)):
                candidates.append(np.atleast_2d(next_states[action]))
                candidate_costs.append(spent + float(next_costs[action]))
        scored = [
            (candidate_costs[i] + _cut_value(candidates[i], distance_table, progress_weight, fail_cost), i)
            for i in range(len(candidates))
        ]
        scored.sort()
        keep = [index for _value, index in scored[:beam]]
        states = [candidates[i] for i in keep]
        costs = [candidate_costs[i] for i in keep]
    return min(
        costs[i] + _cut_value(states[i], distance_table, progress_weight, fail_cost)
        for i in range(len(states))
    )


def plan_with_search(
    model: BeliefWorldModel,
    observation: np.ndarray,
    goal: frozenset,
    *,
    belief: Optional[np.ndarray] = None,
    dynamic: Optional[np.ndarray] = None,
    depth: int = 4,
    beam: int = 3,
    progress_weight: float = PROGRESS_WEIGHT,
    fail_cost: float = FAIL_COST,
    symbol_threshold: float = 0.05,
) -> Tuple[int, List[Dict[str, object]]]:
    """Plan by rolling the belief forward, branching on feedback at the **first** step only.

    Branching on feedback at every depth is exponential; branching once and then continuing
    open-loop under the updated belief is the standard approximation, and it is the step that
    matters because the first action is the one being chosen. The exact `central_value` reference
    branches all the way down, and the gap to it is reported rather than hidden.
    """

    belief = model.parameter_belief(observation) if belief is None else np.atleast_2d(belief)
    dynamic = (
        model.dynamic_state(observation) if dynamic is None else np.atleast_2d(dynamic)
    )
    distance_table = _cell_distance_distribution(None, goal)

    first_states = model.step_actions(dynamic, belief)
    first_costs = model.cost_actions(first_states, belief)

    branches: List[Dict[str, object]] = []
    for action in range(len(ACTION_IDS)):
        state = np.atleast_2d(first_states[action])
        cost = float(first_costs[action])
        probabilities = model.predict_outcome(state, action, belief)[0]
        expected = 0.0
        total_probability = 0.0
        for symbol in range(len(probabilities)):
            probability = float(probabilities[symbol])
            if probability < symbol_threshold:
                continue
            updated = model.update_belief(belief, dynamic, action, symbol)
            # Re-derive the state **under the posterior**. `state` was computed under the prior
            # belief, so without this the imagined feedback sharpens the parameters while the
            # dynamic state still reflects the old mixture: imagining that a release succeeded
            # would leave `released` at its prior value, and the continuation would still believe
            # pushes fail. That suppresses exactly the preparation branches the planner is
            # supposed to prefer.
            state_after = np.atleast_2d(model.step(dynamic, action, updated))
            value = _beam_value(
                model, state_after, updated, distance_table,
                steps=depth - 1, beam=beam,
                progress_weight=progress_weight, fail_cost=fail_cost,
            )
            expected += probability * value
            total_probability += probability
        # account for the symbols skipped by the threshold instead of silently dropping them
        if total_probability < 1.0:
            expected += (1.0 - total_probability) * _cut_value(
                state, distance_table, progress_weight, fail_cost
            )
        branches.append(
            {
                "action": ACTION_IDS[action],
                "action_index": action,
                "score": cost + expected,
                "first_cost": cost,
                "expected_distance": float(state[0, 1 : 1 + N_CELLS] @ distance_table),
                "released_probability": float(state[0, 0]),
                "symbol_probabilities": probabilities,
                "continuation_actions": [],
                "state": state[0],
            }
        )
    chosen = min(branches, key=lambda b: (b["score"], b["action"]))
    return int(chosen["action_index"]), branches


def run_search_episode(
    model: BeliefWorldModel,
    scene,
    belief: np.ndarray,
    *,
    horizon: int = 8,
    radius: int = 5,
    depth: int = 4,
    beam: int = 3,
    warmup: Optional[Tuple[int, int]] = None,
) -> Tuple[int, float, int]:
    """The closed-loop episode driven by :func:`plan_with_search`."""

    goal = frozenset(scene.static.target)
    history_a: List[str] = []
    public_history: List[str] = []
    carried: Optional[np.ndarray] = None

    def observe(current):
        observation = np.asarray(
            observation_vector(
                ray_cast(current, 0, radius=radius), current.robot_pose(0),
                tuple(history_a), tuple(public_history),
            )
        )
        observed = model.dynamic_state(observation)
        return observation, (observed if carried is None else fuse_hidden_state(observed, carried))

    if warmup is not None:
        action, symbol = warmup
        _obs, dynamic = observe(scene)
        belief = model.update_belief(belief, dynamic, action, symbol)
        carried = model.step(dynamic, action, belief)
        history_a.append(FEEDBACK_SYMBOLS[symbol])

    total = 0.0
    steps = 0
    for _ in range(horizon):
        if scene.is_goal():
            break
        observation, dynamic = observe(scene)
        chosen, _branches = plan_with_search(
            model, observation, goal, belief=belief, dynamic=dynamic,
            depth=depth, beam=beam,
        )
        successor, result = execute(scene, _joint_for(scene, ACTION_IDS[chosen]))
        total += float(result.cost)
        steps += 1
        symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
        belief = model.update_belief(belief, dynamic, chosen, symbol)
        carried = model.step(dynamic, chosen, belief)
        history_a.append(result.feedback[0])
        public_history.append(result.y_public)
        scene = successor
    return int(scene.is_goal()), total, steps
