"""Does a message pay when the teammate is *necessary*?

Round 21's rule: message value shows up only where the teammate's contribution is
**required**, not merely cheaper. This module builds exactly that decision and measures the
value of one bit.

The construction, using only physics values the model already has:

* robot A is **slipping** -- A knows this about itself, and it means A can never move the
  object alone at any force, so "push solo" is not an option;
* robot B's traction is **private** to B;
* the object is light, so a **co-push works exactly when B grips**.

So the only way to succeed is to co-push, and whether that works is decided by a fact that
lives entirely in the teammate's private state. Without a message the team must hedge
between attempting a push that may be impossible and declining a task it could have won.
With one bit, it knows.

Costs are the ones the project already uses: a push at force ``f`` costs ``1 + f``; a failed
push costs an extra penalty; and declining a task that was winnable costs the same penalty,
so "do nothing" cannot look free.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .. import __version__
from ..grid import scene_from_ascii
from ..grid.continuous import FORCE_MIN, push_outcome
from ..grid.state import HiddenPhysics, Scene

#: A push at force f costs this plus f.
PUSH_BASE_COST = 1.0
#: Charged for a push that fails, and for declining a task that was winnable.
#:
#: It must clearly exceed the cost of a successful attempt, otherwise declining a winnable
#: task breaks even with winning it and the task objective disappears. An earlier value of
#: 2.0 was exactly the cost of a successful co-push, which made "do nothing" as good as
#: "succeed" and hid the value of the message. The project's planner uses a failure cost of
#: 20 against action costs of order 1, and this follows the same convention.
FAIL_PENALTY = 10.0
#: The angle the team pushes along (inside the east angular window).
ANGLE = 0.0

def necessity_scene(teammate_grip: bool, resistance: int = 1) -> Scene:
    """A slipping, both robots on the west face, goal to the east.

    The grid is indexed ``[y][x]`` and emitted top row first, so the object sits at
    world origin ``(2, 1)``, robot A at ``(1, 1)`` and B at ``(1, 2)``: both are on the
    object's west face, so a co-push is immediately executable.
    """

    grid = [["."] * 6 for _ in range(6)]
    for dx in (0, 1):
        for dy in (0, 1):
            grid[1 + dy][2 + dx] = "O"
    for y in range(6):
        grid[y][4] = "T"
        grid[y][5] = "T"
    grid[1][1] = "A"
    grid[2][1] = "B"
    rows = ["".join(grid[y]) for y in range(5, -1, -1)]
    return scene_from_ascii(
        rows,
        name="necessity",
        hidden=HiddenPhysics(
            resistance=resistance,
            constraint="free",
            traction=("slip", "grip" if teammate_grip else "slip"),
        ),
    )


@dataclass
class NecessityRecord:
    scene_id: int
    teammate_grips: int
    solvable: int
    cost: Dict[str, float]
    success: Dict[str, int]
    regret: Dict[str, float]


@dataclass
class PolicySummary:
    name: str
    n: int
    mean_regret: float
    success_rate: float


@dataclass
class NecessityResult:
    summaries: List[PolicySummary]
    by_teammate: Dict[str, List[PolicySummary]]
    message_value: float
    success_gain: float
    records: List[NecessityRecord] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


POLICIES: Tuple[str, ...] = ("oracle", "silent", "one_bit", "always_co", "always_decline")


def _play(scene: Scene, mode: str, force: float) -> Tuple[float, int]:
    """Cost and success of one choice in the simulator."""

    if mode == "decline":
        return 0.0, 0
    forces = {0: force} if mode == "solo" else {0: force, 1: force}
    outcome = push_outcome(scene, forces, ANGLE)
    if outcome.moved:
        return PUSH_BASE_COST + force, 1
    return PUSH_BASE_COST + force + FAIL_PENALTY, 0


def run_necessity_comparison(
    *, n_scenes: int = 200, seed: int = 0, resistance: int = 1
) -> NecessityResult:
    started = time.perf_counter()
    records: List[NecessityRecord] = []
    rng = np.random.default_rng(seed)

    for scene_id in range(n_scenes):
        teammate_grips = bool(rng.integers(0, 2))
        scene = necessity_scene(teammate_grips, resistance=resistance)

        # What is actually available? A slips, so solo is impossible at any force.
        co_threshold = 1.0 if teammate_grips else float("inf")
        solvable = int(np.isfinite(co_threshold))
        oracle_cost, oracle_success = _play(
            scene, "co", co_threshold if solvable else FORCE_MIN
        )
        if not solvable:
            oracle_cost, oracle_success = 0.0, 0

        # A silent team hedges. It knows its own traction (slipping) and the resistance, so
        # it knows the co-push needs the teammate; it puts probability 1/2 on the teammate
        # gripping, and declines if that looks better than risking a failed attempt.
        p_grip = 0.5
        co_force = 1.0
        co_expected = (
            p_grip * (PUSH_BASE_COST + co_force)
            + (1 - p_grip) * (PUSH_BASE_COST + co_force + FAIL_PENALTY)
        )
        decline_expected = p_grip * FAIL_PENALTY  # misses a winnable task half the time
        if co_expected <= decline_expected:
            silent_cost, silent_success = _play(scene, "co", co_force)
        else:
            silent_cost, silent_success = 0.0, 0

        # One bit: the teammate's traction is known exactly.
        if teammate_grips:
            onebit_cost, onebit_success = _play(scene, "co", co_force)
        else:
            onebit_cost, onebit_success = 0.0, 0

        always_co_cost, always_co_success = _play(scene, "co", co_force)
        always_decline_cost, always_decline_success = 0.0, 0

        def regret(cost: float, success: int) -> float:
            """Regret against the oracle, charging the penalty for missing a win."""

            if solvable and not success:
                return FAIL_PENALTY + (cost - oracle_cost)
            return cost - oracle_cost

        records.append(
            NecessityRecord(
                scene_id=scene_id,
                teammate_grips=int(teammate_grips),
                solvable=solvable,
                cost={
                    "oracle": oracle_cost,
                    "silent": silent_cost,
                    "one_bit": onebit_cost,
                    "always_co": always_co_cost,
                    "always_decline": always_decline_cost,
                },
                success={
                    "oracle": oracle_success,
                    "silent": silent_success,
                    "one_bit": onebit_success,
                    "always_co": always_co_success,
                    "always_decline": always_decline_success,
                },
                regret={
                    "oracle": 0.0,
                    "silent": regret(silent_cost, silent_success),
                    "one_bit": regret(onebit_cost, onebit_success),
                    "always_co": regret(always_co_cost, always_co_success),
                    "always_decline": regret(
                        always_decline_cost, always_decline_success
                    ),
                },
            )
        )

    def summarise(name: str, subset: Sequence[NecessityRecord]) -> List[PolicySummary]:
        out: List[PolicySummary] = []
        for policy in POLICIES:
            if not subset:
                out.append(PolicySummary(policy, 0, float("nan"), float("nan")))
                continue
            out.append(
                PolicySummary(
                    name=policy,
                    n=len(subset),
                    mean_regret=float(np.mean([r.regret[policy] for r in subset])),
                    success_rate=float(np.mean([r.success[policy] for r in subset])),
                )
            )
        return out

    groups = {
        "teammate_grips": [r for r in records if r.teammate_grips],
        "teammate_slips": [r for r in records if not r.teammate_grips],
    }
    overall = summarise("all", records)
    by_name = {s.name: s for s in overall}

    return NecessityResult(
        summaries=overall,
        by_teammate={k: summarise(k, v) for k, v in groups.items()},
        message_value=by_name["silent"].mean_regret - by_name["one_bit"].mean_regret,
        success_gain=by_name["one_bit"].success_rate - by_name["silent"].success_rate,
        records=records,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_scenes": n_scenes,
            "resistance": resistance,
            "push_base_cost": PUSH_BASE_COST,
            "fail_penalty": FAIL_PENALTY,
            "wall_clock_seconds": time.perf_counter() - started,
        },
    )


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_necessity_markdown(result: NecessityResult) -> str:
    lines = [
        "The teammate is **necessary**: this robot slips, so only a co-push can move the "
        "object, and a co-push works exactly when the teammate grips.",
        "",
        f"Value of one bit: **{result.message_value:.3f}** regret; success gain "
        f"**{result.success_gain:+.3f}**.",
        "",
        "| group | policy | n | mean regret | success rate |",
        "|---|---|---|---|---|",
    ]
    for summary in result.summaries:
        lines.append(
            f"| all | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
            f"{summary.success_rate:.3f} |"
        )
    for group, summaries in result.by_teammate.items():
        for summary in summaries:
            lines.append(
                f"| {group} | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
                f"{summary.success_rate:.3f} |"
            )
    return "\n".join(lines)


def write_necessity_report(result: NecessityResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "necessity.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "summaries": [asdict(s) for s in result.summaries],
                "by_teammate": {
                    k: [asdict(s) for s in v] for k, v in result.by_teammate.items()
                },
                "message_value": result.message_value,
                "success_gain": result.success_gain,
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "necessity.md"
    md_path.write_text(
        "# Message value when the teammate is necessary\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_scenes']} scenes, resistance "
        f"{result.meta['resistance']}\n"
        f"- push costs {result.meta['push_base_cost']} + force; failure or a missed win "
        f"costs {result.meta['fail_penalty']}\n\n"
        + format_necessity_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths


# ---------------------------------------------------------------------------
# Learned version: can a model extract the message's information on its own?
# ---------------------------------------------------------------------------


@dataclass
class LearnedNecessitySummary:
    scheme: str
    bits: int
    n: int
    accuracy: float
    balanced_accuracy: float
    mean_regret: float
    success_rate: float
    silent_baseline_regret: float
    oracle_regret: float


def necessity_samples(
    *,
    n_scenes: int = 200,
    message_scheme: str = "none",
    seed: int = 0,
    radius: int = 5,
    lead_in: int = 0,
) -> List[Tuple[Tuple[float, ...], Tuple[float, ...], int, int]]:
    """(observation, action, label, teammate_grips) tuples for the necessity decision.

    ``lead_in`` defaults to zero on purpose. Any real lead-in that produces public
    evidence -- for instance a co-push that moved the object -- would reveal the
    teammate's traction through the public record, and the message would be redundant.
    With a fresh scene the only channel carrying that fact is the message, which is
    exactly the comparison being measured.
    """

    from ..grid.dynamics import execute
    from ..grid.sensing import SensorNoise, apply_noise, ray_cast
    from ..learning.continuous_data import continuous_action_vector
    from ..learning.data import _joint_for
    from ..learning.features import observation_vector
    from ..learning.messages import message_vector

    rng = np.random.default_rng(seed)
    out = []
    for scene_id in range(n_scenes):
        teammate_grips = bool(rng.integers(0, 2))
        origin = (2, 1 + int(rng.integers(0, 2)))
        scene = necessity_scene(teammate_grips)
        scan = apply_noise(ray_cast(scene, 0, radius=radius), SensorNoise(), rng)

        history_a: List[str] = []
        public_history: List[str] = []
        current = scene
        for _ in range(lead_in):
            successor, result = execute(current, _joint_for(current, "push:0:E"))
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            current = successor

        observation = observation_vector(
            scan, current.robot_pose(0), tuple(history_a), tuple(public_history)
        ) + message_vector(message_scheme, current.physics, ())
        action = continuous_action_vector(ANGLE, 1.0)
        label = int(push_outcome(scene, {0: 1.0, 1: 1.0}, ANGLE).moved)
        out.append((observation, action, label, int(teammate_grips)))
    return out


def _optimal_push_threshold(force: float) -> float:
    """The probability above which attempting the push beats declining.

    Derived from the declared costs rather than hard-coded: attempting costs ``1 + f``
    plus the penalty with probability ``1 - p``; declining costs the penalty with
    probability ``p``. Attempting is better when ``p >= (1 + f + penalty) / (2 * penalty)``.
    """

    return (PUSH_BASE_COST + force + FAIL_PENALTY) / (2.0 * FAIL_PENALTY)


def run_learned_necessity(
    *,
    schemes: Sequence[str] = ("none", "traction"),
    n_train_scenes: int = 400,
    n_test_scenes: int = 200,
    epochs: int = 400,
    seed: int = 0,
) -> List[LearnedNecessitySummary]:
    """Train the same model with and without the message and compare decision quality."""

    from ..learning.features import OBSERVATION_DIM
    from ..learning.messages import message_dim, payload_bits
    from ..learning.mlp import MultiHeadMLP, train

    out: List[LearnedNecessitySummary] = []
    for scheme in schemes:
        split = OBSERVATION_DIM + message_dim(scheme)
        train_samples = necessity_samples(
            n_scenes=n_train_scenes, message_scheme=scheme, seed=seed
        )
        test_samples = necessity_samples(
            n_scenes=n_test_scenes, message_scheme=scheme, seed=seed + 40_000
        )
        obs = np.asarray([s[0][:split] for s in train_samples], dtype=np.float64)
        action = np.asarray([s[1] for s in train_samples], dtype=np.float64)
        label = np.asarray([s[2] for s in train_samples], dtype=np.float64)
        model = MultiHeadMLP(
            seed=seed, obs_dim=split, action_dim=action.shape[1]
        )
        targets = np.concatenate(
            [label[:, None], np.zeros((len(label), 2))], axis=1
        )
        train(model, obs, action, targets, epochs=epochs, seed=seed)

        test_obs = np.asarray([s[0][:split] for s in test_samples], dtype=np.float64)
        test_action = np.asarray([s[1] for s in test_samples], dtype=np.float64)
        test_label = np.asarray([s[2] for s in test_samples], dtype=np.float64)
        prob = model.predict_outcomes(test_obs, test_action)[:, 0]
        predicted = (prob >= _optimal_push_threshold(1.0)).astype(int)

        accuracy = float((predicted == test_label).mean())
        recalls = []
        if (test_label == 1).any():
            recalls.append(float((predicted[test_label == 1] == 1).mean()))
        if (test_label == 0).any():
            recalls.append(float((predicted[test_label == 0] == 0).mean()))

        regrets: List[float] = []
        successes: List[int] = []
        for sample, decision in zip(test_samples, predicted):
            teammate_grips = bool(sample[3])
            solvable = 1 if teammate_grips else 0
            oracle_cost = PUSH_BASE_COST + 1.0 if solvable else 0.0
            if decision:
                cost = (
                    PUSH_BASE_COST + 1.0
                    if solvable
                    else PUSH_BASE_COST + 1.0 + FAIL_PENALTY
                )
                success = solvable
            else:
                cost = 0.0
                success = 0
            if solvable and not success:
                regrets.append(FAIL_PENALTY + (cost - oracle_cost))
            else:
                regrets.append(cost - oracle_cost)
            successes.append(success)

        # The silent analytic baseline, for reference: it never has the information.
        silent_regrets = []
        for sample in test_samples:
            solvable = 1 if sample[3] else 0
            oracle_cost = PUSH_BASE_COST + 1.0 if solvable else 0.0
            # declining is optimal below the threshold for p = 1/2
            decision = 0.5 >= _optimal_push_threshold(1.0)
            if decision:
                cost = PUSH_BASE_COST + 1.0 if solvable else PUSH_BASE_COST + 1.0 + FAIL_PENALTY
                success = solvable
            else:
                cost, success = 0.0, 0
            if solvable and not success:
                silent_regrets.append(FAIL_PENALTY + (cost - oracle_cost))
            else:
                silent_regrets.append(cost - oracle_cost)

        out.append(
            LearnedNecessitySummary(
                scheme=scheme,
                bits=payload_bits(scheme),
                n=len(test_samples),
                accuracy=accuracy,
                balanced_accuracy=float(np.mean(recalls)) if recalls else float("nan"),
                mean_regret=float(np.mean(regrets)),
                success_rate=float(np.mean(successes)),
                silent_baseline_regret=float(np.mean(silent_regrets)),
                oracle_regret=0.0,
            )
        )
    return out


def format_learned_necessity_markdown(
    summaries: Sequence[LearnedNecessitySummary],
) -> str:
    lines = [
        f"A co-push is attempted when the predicted success probability reaches "
        f"{_optimal_push_threshold(1.0):.2f}, the threshold implied by the declared costs.",
        "",
        "| scheme | bits | n | accuracy | balanced accuracy | mean regret | success |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.scheme} | {summary.bits} | {summary.n} | "
            f"{summary.accuracy:.3f} | {summary.balanced_accuracy:.3f} | "
            f"{summary.mean_regret:.3f} | {summary.success_rate:.3f} |"
        )
    if summaries:
        lines += [
            "",
            f"For reference, the silent analytic policy scores "
            f"{summaries[0].silent_baseline_regret:.3f} regret and the oracle 0.000.",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-step episodes: does the message value survive once outcomes accumulate?
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSummary:
    horizon: int
    policy: str
    n: int
    mean_cost: float
    success_rate: float
    regret_first_decision: float
    regret_later_decisions: float


def run_episode_message_value(
    *,
    horizons: Sequence[int] = (1, 2, 4),
    n_episodes: int = 400,
    seed: int = 0,
) -> List[EpisodeSummary]:
    """A task needing ``h`` co-pushes, with and without one bit at the start.

    Two modelling decisions matter and are declared:

    * **The failure cost scales with the horizon** (``10 * h``). With a flat failure cost,
      a long successful episode costs more than failing it, so the task stops being worth
      doing at all -- there is no completion reward anywhere in the cost model. Scaling it
      keeps "winning" strictly better than "failing" at every horizon while staying within
      the project's "action cost plus explicit failure cost" convention.
    * **A failed attempt costs only the attempt.** Charging the full episode-failure cost to
      every failed push would make trying never worthwhile, and learning requires trying.

    The oracle knows the teammate's traction throughout. The **message** policy learns it
    from one bit at the first decision. The **silent** policy starts at even odds and
    updates from the public outcome, which identifies a deterministic co-push after a
    single attempt -- so its regret should be concentrated in the first decision.
    """

    rng = np.random.default_rng(seed)
    out: List[EpisodeSummary] = []

    for horizon in horizons:
        failure_cost = 10.0 * horizon
        attempt_cost = PUSH_BASE_COST + 1.0

        # One episode set per horizon, shared by every policy. Drawing separately for each
        # policy would compare them on different episodes and swamp a one-attempt effect
        # with sampling noise.
        episode_grips = [bool(rng.integers(0, 2)) for _ in range(n_episodes)]

        for policy in ("oracle", "message", "silent"):
            costs: List[float] = []
            successes: List[int] = []
            first_regrets: List[float] = []
            later_regrets: List[float] = []

            for grips in episode_grips:
                solvable = 1 if grips else 0
                oracle_total = attempt_cost * horizon if solvable else failure_cost

                if policy == "oracle":
                    plan_try = [bool(solvable)] * horizon
                elif policy == "message":
                    plan_try = [grips] * horizon
                else:
                    # silent: begin at even odds; the optimal action under p=0.5 is to
                    # decline, but declining can never reveal anything, so a silent team
                    # that declines stays ignorant. It therefore compares trying once
                    # (which may win the whole episode) against the certain failure.
                    p_first = 0.5
                    expected_try = p_first * attempt_cost * horizon + (
                        1 - p_first
                    ) * (attempt_cost + failure_cost)
                    expected_decline = failure_cost
                    try_first = expected_try < expected_decline
                    if not try_first:
                        plan_try = [False] * horizon
                    else:
                        # try; a successful first attempt reveals that the teammate grips,
                        # a failed one that it slips
                        plan_try = [True] + [grips] * (horizon - 1)

                total = 0.0
                pushed = 0
                for step in range(horizon):
                    if step >= len(plan_try) or not plan_try[step]:
                        break
                    total += attempt_cost
                    if not grips:
                        break
                    pushed += 1
                won = 1 if pushed == horizon else 0
                if not won:
                    total += failure_cost

                costs.append(total)
                successes.append(won)
                episode_regret = total - oracle_total
                # attribute it: the first decision versus everything after
                if horizon == 1:
                    first_regrets.append(episode_regret)
                else:
                    first_regrets.append(0.0)
                    later_regrets.append(episode_regret)

            out.append(
                EpisodeSummary(
                    horizon=horizon,
                    policy=policy,
                    n=n_episodes,
                    mean_cost=float(np.mean(costs)),
                    success_rate=float(np.mean(successes)),
                    regret_first_decision=(
                        float(np.mean(first_regrets)) if first_regrets else 0.0
                    ),
                    regret_later_decisions=(
                        float(np.mean(later_regrets)) if later_regrets else 0.0
                    ),
                )
            )
    return out


def format_episode_markdown(summaries: Sequence[EpisodeSummary]) -> str:
    lines = [
        "| horizon | policy | mean cost | success rate |",
        "|---|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.horizon} | {summary.policy} | {summary.mean_cost:.3f} | "
            f"{summary.success_rate:.3f} |"
        )
    lines += ["", "Message value (silent cost minus message cost):", ""]
    by_horizon: Dict[int, Dict[str, EpisodeSummary]] = {}
    for summary in summaries:
        by_horizon.setdefault(summary.horizon, {})[summary.policy] = summary
    for horizon in sorted(by_horizon):
        group = by_horizon[horizon]
        value = group["silent"].mean_cost - group["message"].mean_cost
        lines.append(
            f"- horizon {horizon}: **{value:.3f}** "
            f"(silent success {group['silent'].success_rate:.3f} versus message "
            f"{group['message'].success_rate:.3f})"
        )
    return "\n".join(lines)
