"""Threshold probe: infer the force a push needs, when force costs something.

Round 18 formulated the continuous case as binary classification over an (angle, force)
grid and it failed: 3.7% of the labels are positive, so the model either predicts "never
feasible" or over-predicts, and the region is wrong either way.

The physics says the right object is a **scalar**: inside the angular window a push works
exactly when ``force >= resistance / sum(traction)``. That single number is what the robot
cannot observe, so the probe regresses it directly, with two targets:

* **can push at all** -- whether a push at maximum force works (false when the robot
  slips, when the object is latched, or when the direction is blocked);
* **force threshold** -- the smallest force that works, regressed only where a push is
  possible (else the target is infinite and is masked out).

The threshold is decision-relevant only if force costs something, which it does here:
a push at force ``f`` costs ``1 + f``. So pushing too softly fails and pushing too hard
wastes energy, and the optimal choice is exactly the threshold. That makes the metric a
**regret in the task's own units**, not an abstract error.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .. import __version__
from ..grid.continuous import FORCE_MAX, FORCE_MIN, force_threshold, push_outcome
from ..grid.planner import PHYSICS
from ..learning.continuous_data import continuous_features
from ..learning.data import generate_situations, pose_to_index
from ..learning.features import OBSERVATION_DIM
from ..learning.messages import (
    SCHEME_ORDER,
    message_dim,
    message_vector,
    payload_bits,
)
from ..grid.state import HiddenPhysics, Scene
from ..learning.mlp import MultiHeadMLP, train

#: Base cost of a push; the total cost of a push at force f is BASE + f, so a heavier
#: push really is more expensive. Declared here and used identically for every policy.
PUSH_BASE_COST = 1.0

#: Regret charged when the policy pushes and the push fails, or declines a push that was
#: possible. Fixed before the comparison and identical for every policy.
FAIL_PENALTY = 2.0

#: Pushing direction used by the probe: straight east, inside the angular window.
PROBE_ANGLE = 0.0

BASELINES: Tuple[str, ...] = ("oracle", "learned", "always_max", "always_min")


@dataclass
class ThresholdRecord:
    scene_id: int
    true_physics: str
    can_push: int
    true_threshold: float
    predicted_can_push: float
    predicted_threshold: float
    mechanism: str
    regret: Dict[str, float]
    force_used: Dict[str, float]
    #: Whether this robot is the one that slips; recorded so the caller can look at the
    #: subgroup where the teammate's traction is what matters.
    self_slips: int = 0


@dataclass
class PolicySummary:
    name: str
    n: int
    mean_regret: float
    success_rate: float


@dataclass
class ThresholdProbeResult:
    summaries: List[PolicySummary]
    by_mechanism: Dict[str, List[PolicySummary]]
    threshold_mae: float
    can_push_balanced_accuracy: float
    n_pushable: int
    records: List[ThresholdRecord] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def _threshold_training(situations, message_scheme: str = "none"):
    """The two supervised targets plus the observation, for a set of situations.

    Shared by the threshold probe and the continuous rollout so both train and evaluate
    the identical model.
    """

    features: List[Tuple[float, ...]] = []
    can_push: List[float] = []
    thresholds: List[float] = []
    mask: List[float] = []
    physics_index: List[int] = []
    poses: List[int] = []
    from ..learning.data import pose_to_index

    for situation in situations:
        scene, pushable, threshold = _targets_for(situation)
        features.append(
            continuous_features(
                situation.scan,
                scene.robot_pose(0),
                situation.history_a,
                situation.public_history,
                PROBE_ANGLE,
                FORCE_MAX,
            )
            # The received message is appended to the observation. It is a declared
            # codebook over the sender's own private state, and it is the only channel
            # through which the teammate's traction can reach this robot.
            + message_vector(
                message_scheme, situation.physics, situation.history_b
            )
        )
        can_push.append(float(pushable))
        thresholds.append(threshold if np.isfinite(threshold) else 0.0)
        mask.append(float(pushable))
        physics_index.append(PHYSICS.index(situation.physics))
        poses.append(pose_to_index(scene.obj.origin))
    x = np.asarray(features, dtype=np.float64)
    split = OBSERVATION_DIM + message_dim(message_scheme)
    return (
        x[:, :split],
        x[:, split:],
        np.asarray(can_push, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(mask, dtype=np.float64),
        np.asarray(physics_index, dtype=np.int64),
        np.asarray(poses, dtype=np.int64),
    )


def _targets_for(situation):
    """The two supervised targets for one decision point, plus the observables."""

    scene = situation.scene
    at_max = push_outcome(scene, {0: FORCE_MAX}, PROBE_ANGLE)
    can_push = int(at_max.moved)
    threshold = force_threshold(scene, [0])
    return scene, can_push, threshold


def run_threshold_probe(
    *,
    n_train_situations: int = 250,
    n_test_situations: int = 120,
    epochs: int = 500,
    seed: int = 0,
    lead_in: int = 2,
    message_scheme: str = "none",
) -> ThresholdProbeResult:
    started = time.perf_counter()
    train_situations = generate_situations(
        n_scenes=n_train_situations, lead_in=lead_in, seed=seed
    )
    test_situations = generate_situations(
        n_scenes=n_test_situations, lead_in=lead_in, seed=seed + 70_000
    )

    obs, action, can_push, thresholds, mask, physics_index, poses = _threshold_training(
        train_situations, message_scheme
    )
    positive_rate = float(can_push.mean())
    model = MultiHeadMLP(
        seed=seed,
        obs_dim=obs.shape[1],  # the observation plus any received message
        action_dim=action.shape[1],
        positive_weight=(1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0,
    )
    targets = np.concatenate(
        [can_push[:, None], np.zeros((len(can_push), 2))], axis=1
    )
    train(
        model, obs, action, targets, physics_index, poses,
        thresholds=thresholds, threshold_mask=mask,
        epochs=epochs, seed=seed,
    )

    # -- evaluation on held-out situations -------------------------------------
    records: List[ThresholdRecord] = []
    for situation in test_situations:
        scene, pushable, threshold = _targets_for(situation)
        feature = np.asarray(
            [
                continuous_features(
                    situation.scan, scene.robot_pose(0), situation.history_a,
                    situation.public_history, PROBE_ANGLE, FORCE_MAX,
                )
                + message_vector(
                    message_scheme, situation.physics, situation.history_b
                )
            ],
            dtype=np.float64,
        )
        split = OBSERVATION_DIM + message_dim(message_scheme)
        obs_one = feature[:, :split]
        act_one = feature[:, split:]
        prob_push = float(model.predict_outcomes(obs_one, act_one)[0, 0])
        predicted_threshold = float(model.predict_threshold(obs_one, act_one)[0])
        predicted_threshold = float(
            min(max(predicted_threshold, FORCE_MIN), FORCE_MAX)
        )
        pushes = prob_push >= 0.5

        mechanism = []
        if situation.physics.constraint == "constrained":
            mechanism.append("latched")
        if situation.physics.resistance >= 2:
            mechanism.append("heavy")
        if situation.physics.traction[0] == "slip":
            mechanism.append("self_slips")
        if situation.physics.traction[1] == "slip":
            mechanism.append("teammate_slips")

        def regret_for(pushes: bool, force: float) -> Tuple[float, float, float]:
            """Regret, success and the force spent for one policy's choice."""

            if not pushes:
                if pushable:
                    return FAIL_PENALTY, 0.0, 0.0
                return 0.0, 0.0, 0.0
            outcome = push_outcome(scene, {0: force}, PROBE_ANGLE)
            if not outcome.moved:
                return FAIL_PENALTY + force, 0.0, force
            return force - threshold, 1.0, force

        choices = {
            "oracle": (pushable, threshold if np.isfinite(threshold) else 0.0),
            "learned": (pushes, predicted_threshold),
            "always_max": (True, FORCE_MAX),
            "always_min": (True, 0.25),
        }
        regret: Dict[str, float] = {}
        force_used: Dict[str, float] = {}
        for name, (does_push, force) in choices.items():
            value, _success, spent = regret_for(bool(does_push), force)
            regret[name] = value
            force_used[name] = spent

        records.append(
            ThresholdRecord(
                self_slips=int(situation.physics.traction[0] == "slip"),
                scene_id=situation.scene_id,
                true_physics=situation.physics.label,
                can_push=pushable,
                true_threshold=threshold if np.isfinite(threshold) else float("inf"),
                predicted_can_push=prob_push,
                predicted_threshold=predicted_threshold,
                mechanism="+".join(mechanism) if mechanism else "plain",
                regret=regret,
                force_used=force_used,
            )
        )

    pushable_records = [r for r in records if r.can_push]
    threshold_mae = (
        float(
            np.mean(
                [abs(r.predicted_threshold - r.true_threshold) for r in pushable_records]
            )
        )
        if pushable_records
        else float("nan")
    )
    correct = [
        (r.predicted_can_push >= 0.5) == bool(r.can_push) for r in records
    ]
    recalls = []
    if any(r.can_push for r in records):
        recalls.append(
            float(
                np.mean(
                    [
                        (r.predicted_can_push >= 0.5)
                        for r in records
                        if r.can_push
                    ]
                )
            )
        )
    if any(not r.can_push for r in records):
        recalls.append(
            float(
                np.mean(
                    [
                        (r.predicted_can_push < 0.5)
                        for r in records
                        if not r.can_push
                    ]
                )
            )
        )

    def summarise(name: str, subset: Sequence[ThresholdRecord]) -> List[PolicySummary]:
        out = []
        for policy in BASELINES:
            if not subset:
                out.append(PolicySummary(policy, 0, float("nan"), float("nan")))
                continue
            out.append(
                PolicySummary(
                    name=policy,
                    n=len(subset),
                    mean_regret=float(np.mean([r.regret[policy] for r in subset])),
                    success_rate=float(
                        np.mean([r.regret[policy] <= r.true_threshold + 1e-9 for r in subset])
                    ),
                )
            )
        return out

    groups: Dict[str, List[ThresholdRecord]] = defaultdict(list)
    for record in records:
        groups[record.mechanism].append(record)

    return ThresholdProbeResult(
        summaries=summarise("all", records),
        by_mechanism={k: summarise(k, v) for k, v in sorted(groups.items())},
        threshold_mae=threshold_mae,
        can_push_balanced_accuracy=float(np.mean(recalls)) if recalls else float("nan"),
        n_pushable=len(pushable_records),
        records=records,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_train_situations": len(train_situations),
            "n_test_situations": len(test_situations),
            "epochs": epochs,
            "positive_rate": positive_rate,
            "message_scheme": message_scheme,
            "message_bits": payload_bits(message_scheme),
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


def format_threshold_markdown(result: ThresholdProbeResult) -> str:
    lines = [
        f"Pushable decision points: {result.n_pushable} of {len(result.records)}. "
        f"Can-push balanced accuracy {result.can_push_balanced_accuracy:.3f}; "
        f"threshold MAE on pushable points {result.threshold_mae:.3f} force units.",
        "",
        "| group | policy | n | mean regret | success rate |",
        "|---|---|---|---|---|",
    ]
    for summary in result.summaries:
        lines.append(
            f"| all | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
            f"{summary.success_rate:.3f} |"
        )
    for group, summaries in result.by_mechanism.items():
        for summary in summaries:
            lines.append(
                f"| {group} | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
                f"{summary.success_rate:.3f} |"
            )
    return "\n".join(lines)


def write_threshold_report(result: ThresholdProbeResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "threshold_probe.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "summaries": [asdict(s) for s in result.summaries],
                "by_mechanism": {k: [asdict(s) for s in v] for k, v in result.by_mechanism.items()},
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "threshold_probe.md"
    md_path.write_text(
        "# Threshold probe: inferring the force a push needs\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_train_situations']} training / "
        f"{result.meta['n_test_situations']} held-out situations, "
        f"{result.meta['epochs']} epochs\n"
        f"- push costs {result.meta['push_base_cost']} + force; failure penalty "
        f"{result.meta['fail_penalty']}\n"
        f"- wall clock: {result.meta['wall_clock_seconds']:.1f}s\n\n"
        + format_threshold_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths


# ---------------------------------------------------------------------------
# Message comparison: does one bounded message close the teammate-slipping gap?
# ---------------------------------------------------------------------------


@dataclass
class MessageSchemeSummary:
    scheme: str
    bits: int
    n: int
    overall_regret: float
    overall_success: float
    threshold_mae: float
    can_push_balanced_accuracy: float
    teammate_slips_n: int
    teammate_slips_regret: float
    teammate_slips_success: float


def run_message_comparison(
    *,
    schemes: Sequence[str] = SCHEME_ORDER,
    n_train_situations: int = 200,
    n_test_situations: int = 100,
    epochs: int = 400,
    seed: int = 0,
) -> List[MessageSchemeSummary]:
    """Train the same threshold model under each message scheme and compare.

    The comparison is restricted in scope on purpose: one message per decision, sent by
    the teammate, decoded into the receiver's observation. There is no joint commit rule
    and no message-dependent reference belief, so this measures the **value of the message
    content** at a fixed bit budget, which is the C1 question, not the whole protocol.
    """

    out: List[MessageSchemeSummary] = []
    for scheme in schemes:
        result = run_threshold_probe(
            n_train_situations=n_train_situations,
            n_test_situations=n_test_situations,
            epochs=epochs,
            seed=seed,
            message_scheme=scheme,
        )
        by_name = {s.name: s for s in result.summaries}
        slips = [
            record
            for record in result.records
            if record.mechanism and "teammate_slips" in record.mechanism
        ]
        out.append(
            MessageSchemeSummary(
                scheme=scheme,
                bits=int(result.meta["message_bits"]),
                n=len(result.records),
                overall_regret=by_name["learned"].mean_regret,
                overall_success=by_name["learned"].success_rate,
                threshold_mae=result.threshold_mae,
                can_push_balanced_accuracy=result.can_push_balanced_accuracy,
                teammate_slips_n=len(slips),
                teammate_slips_regret=(
                    float(np.mean([r.regret["learned"] for r in slips])) if slips else float("nan")
                ),
                teammate_slips_success=(
                    float(np.mean([r.regret["learned"] <= 1e-9 for r in slips]))
                    if slips
                    else float("nan")
                ),
            )
        )
    return out


def format_message_comparison_markdown(summaries: Sequence[MessageSchemeSummary]) -> str:
    lines = [
        "| scheme | bits | overall regret | overall success | threshold MAE | "
        "teammate-slipping n | its regret | its success |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.scheme} | {summary.bits} | {summary.overall_regret:.3f} | "
            f"{summary.overall_success:.3f} | {summary.threshold_mae:.3f} | "
            f"{summary.teammate_slips_n} | {summary.teammate_slips_regret:.3f} | "
            f"{summary.teammate_slips_success:.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Joint decision: solo push versus co-push, where the teammate's traction matters
# ---------------------------------------------------------------------------


@dataclass
class JointRecord:
    scene_id: int
    teammate_slips: int
    mechanism: str
    true_best_mode: str
    chosen_mode: str
    regret: float
    success: int


@dataclass
class JointSummary:
    scheme: str
    bits: int
    n: int
    mean_regret: float
    success_rate: float
    teammate_slips_n: int
    teammate_slips_regret: float
    teammate_slips_success: float


def _joint_scene(situation):
    """The staged scene: both robots already on the object's west face."""

    from ..grid.state import ObjectState, Scene
    from ..grid.dynamics import SKILL_HOLD  # noqa: F401

    scene = situation.scene
    return Scene(
        static=scene.static,
        obj=ObjectState(scene.obj.origin),
        robots=scene.robots,
        physics=scene.physics,
        released=scene.released,
    )


def _co_push_contact(scene) -> bool:
    """Whether both robots can push east right now."""

    object_cells = scene.obj.cells
    for robot in scene.robots:
        if (robot.pose[0] + 1, robot.pose[1]) not in object_cells:
            return False
    return True


def run_joint_message_comparison(
    *,
    schemes: Sequence[str] = SCHEME_ORDER,
    n_train_situations: int = 200,
    n_test_situations: int = 100,
    epochs: int = 400,
    seed: int = 0,
) -> List[JointSummary]:
    """Solo-push versus co-push, with and without a message about the teammate.

    In this decision the teammate's traction genuinely matters -- it is what decides
    whether a co-push can succeed -- so it is the setting in which a message about
    "I am slipping" can be expected to help. The single-robot version of this probe
    cannot show any message effect, because there the teammate is irrelevant by
    construction.
    """

    from ..grid.continuous import FORCE_MIN, push_outcome

    out: List[JointSummary] = []
    for scheme in schemes:
        result = run_threshold_probe(
            n_train_situations=n_train_situations,
            n_test_situations=n_test_situations,
            epochs=epochs,
            seed=seed,
            message_scheme=scheme,
        )
        records = result.records
        # The threshold probe already supplies the message-augmented observation, so the
        # joint evaluation re-uses its situations through the same generator.
        situations = generate_situations(
            n_scenes=n_test_situations, lead_in=2, seed=seed + 70_000
        )
        joint_records: List[JointRecord] = []

        for situation, record in zip(situations, records):
            scene = situation.scene
            if not _co_push_contact(scene):
                continue
            physics = situation.physics
            resistance = float(physics.resistance)
            tractions = [
                1.0 if physics.traction[i] == "grip" else 0.0 for i in (0, 1)
            ]
            solo_threshold = (
                resistance / tractions[0] if tractions[0] > 0 else float("inf")
            )
            co_threshold = (
                resistance / sum(tractions) if sum(tractions) > 0 else float("inf")
            )

            def simulate(mode: str, force: float):
                forces = {0: force} if mode == "solo" else {0: force, 1: force}
                return push_outcome(scene, forces, PROBE_ANGLE)

            def cost(mode: str, force: float) -> Tuple[float, int]:
                """Cost of choosing ``mode`` at ``force``; the optimum is the threshold."""

                outcome = simulate(mode, force)
                if outcome.moved:
                    return PUSH_BASE_COST + force, 1
                return PUSH_BASE_COST + force + FAIL_PENALTY, 0

            best = None
            for mode, threshold in (("solo", solo_threshold), ("co", co_threshold)):
                if not np.isfinite(threshold):
                    continue
                candidate = cost(mode, min(max(threshold, FORCE_MIN), 2.0))
                if best is None or candidate[0] < best[0]:
                    best = (*candidate, mode)
            if best is None:
                continue  # nothing works from here
            oracle_cost, oracle_success, oracle_mode = best

            # The policy's view: without a message it does not know the teammate's
            # traction, so it only uses the solo option; with a message it can evaluate
            # the co option too. Both use the same declared costs.
            if scheme == "none":
                # A *fair* silent baseline: it cannot see the teammate's traction, so it
                # picks the mode with the lower expected cost under a declared prior.
                # "Always solo" would be a straw man, since a silent policy can still
                # reason about the chance that the teammate grips.
                p_grip = 0.5
                options = []
                for mode, threshold_for in (
                    ("solo", lambda grips: resistance / tractions[0] if tractions[0] else float("inf")),
                    ("co", lambda grips: (
                        resistance / (tractions[0] + (1.0 if grips else 0.0))
                        if (tractions[0] + (1.0 if grips else 0.0)) > 0
                        else float("inf")
                    )),
                ):
                    expected = 0.0
                    for grips, prob in ((True, p_grip), (False, 1.0 - p_grip)):
                        threshold = threshold_for(grips)
                        if not np.isfinite(threshold):
                            expected += prob * (PUSH_BASE_COST + FORCE_MAX + FAIL_PENALTY)
                            continue
                        force = min(max(threshold, FORCE_MIN), 2.0)
                        expected += prob * (
                            PUSH_BASE_COST + force + FAIL_PENALTY * (0.0 if force >= threshold else 1.0)
                        )
                    options.append((expected, mode, threshold_for(physics.traction[1] == "grip")))
                options.sort(key=lambda item: (item[0], item[1]))
                _, chosen_mode, chosen_threshold = options[0]
                chosen_force = (
                    min(max(chosen_threshold, FORCE_MIN), 2.0)
                    if np.isfinite(chosen_threshold)
                    else FORCE_MIN
                )
            else:
                # With a message the teammate's traction is known exactly.
                grips = physics.traction[1] == "grip"
                co_known = resistance / (tractions[0] + (1.0 if grips else 0.0)) if (
                    tractions[0] + (1.0 if grips else 0.0)
                ) > 0 else float("inf")
                solo_known = resistance / tractions[0] if tractions[0] else float("inf")
                if np.isfinite(co_known) and (
                    not np.isfinite(solo_known)
                    or PUSH_BASE_COST + co_known < PUSH_BASE_COST + solo_known
                ):
                    chosen_mode, chosen_force = "co", min(max(co_known, FORCE_MIN), 2.0)
                elif np.isfinite(solo_known):
                    chosen_mode, chosen_force = "solo", min(max(solo_known, FORCE_MIN), 2.0)
                else:
                    chosen_mode, chosen_force = "solo", FORCE_MIN
            chosen_cost, chosen_success = cost(chosen_mode, chosen_force)

            joint_records.append(
                JointRecord(
                    scene_id=situation.scene_id,
                    teammate_slips=int(physics.traction[1] == "slip"),
                    mechanism=record.mechanism,
                    true_best_mode=oracle_mode,
                    chosen_mode=chosen_mode,
                    regret=chosen_cost - oracle_cost,
                    success=chosen_success,
                )
            )

        slips = [r for r in joint_records if r.teammate_slips]
        out.append(
            JointSummary(
                scheme=scheme,
                bits=int(result.meta["message_bits"]),
                n=len(joint_records),
                mean_regret=(
                    float(np.mean([r.regret for r in joint_records]))
                    if joint_records
                    else float("nan")
                ),
                success_rate=(
                    float(np.mean([r.success for r in joint_records]))
                    if joint_records
                    else float("nan")
                ),
                teammate_slips_n=len(slips),
                teammate_slips_regret=(
                    float(np.mean([r.regret for r in slips])) if slips else float("nan")
                ),
                teammate_slips_success=(
                    float(np.mean([r.success for r in slips])) if slips else float("nan")
                ),
            )
        )
    return out


def format_joint_comparison_markdown(summaries: Sequence[JointSummary]) -> str:
    lines = [
        "| scheme | bits | n | mean regret | success | teammate-slipping n | "
        "its regret | its success |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.scheme} | {summary.bits} | {summary.n} | "
            f"{summary.mean_regret:.3f} | {summary.success_rate:.3f} | "
            f"{summary.teammate_slips_n} | {summary.teammate_slips_regret:.3f} | "
            f"{summary.teammate_slips_success:.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Continuous parameters in a closed loop
# ---------------------------------------------------------------------------


def _continuous_scene(physics: HiddenPhysics) -> Scene:
    """Object two pushes from the goal, both robots on its west face."""

    from ..grid import scene_from_ascii

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
    return scene_from_ascii(rows, name="continuous", hidden=physics)


def run_continuous_episode(
    physics: HiddenPhysics,
    policy: str,
    model: MultiHeadMLP,
    *,
    horizon: int = 6,
    radius: int = 5,
    seed: int = 0,
) -> Tuple[int, float, int]:
    """One episode in which each step chooses a force.

    A push at force ``f`` costs ``1 + f``. On success the object advances one cell along the
    chosen axis and the pushing robot advances with it, so the same contact holds for the next
    step. The direction is fixed east; **the force is the decision**, which is what the
    threshold probe regresses, so this is the continuous decision inside a closed loop.
    """

    from dataclasses import replace

    from ..grid.continuous import FORCE_MIN, push_outcome
    from ..grid.sensing import SensorNoise, apply_noise, ray_cast
    from ..grid.state import ObjectState, RobotState

    rng = np.random.default_rng(seed)
    scene = _continuous_scene(physics)
    history_a: List[str] = []
    public_history: List[str] = []
    total = 0.0
    steps = 0

    for step in range(horizon):
        if scene.is_goal():
            return 1, total, step
        scan = apply_noise(ray_cast(scene, 0, radius=radius), SensorNoise(), rng)
        feature = np.asarray(
            [
                continuous_features(
                    scan, scene.robot_pose(0), tuple(history_a),
                    tuple(public_history), PROBE_ANGLE, FORCE_MAX,
                )
            ],
            dtype=np.float64,
        )
        split = OBSERVATION_DIM
        obs_one = feature[:, :split]
        act_one = feature[:, split:]

        gripper = 1.0 if physics.traction[0] == "grip" else 0.0
        if policy == "oracle":
            force = (
                min(max(physics.resistance / gripper, FORCE_MIN), FORCE_MAX)
                if gripper > 0
                else FORCE_MIN
            )
            attempt = gripper > 0
        elif policy == "learned":
            probability = float(model.predict_outcomes(obs_one, act_one)[0, 0])
            force = float(model.predict_threshold(obs_one, act_one)[0])
            force = min(max(force, FORCE_MIN), FORCE_MAX)
            attempt = probability >= 0.5
        elif policy == "always_max":
            force, attempt = FORCE_MAX, True
        elif policy == "always_min":
            force, attempt = 0.25, True
        else:
            raise ValueError(policy)

        if not attempt:
            break
        steps += 1
        total += PUSH_BASE_COST + force
        outcome = push_outcome(scene, {0: force}, PROBE_ANGLE)
        if not outcome.moved:
            break
        delta = outcome.direction.delta
        moved = ObjectState(scene.obj.origin).translated(delta)
        robots = tuple(
            RobotState(r.robot_id, (r.pose[0] + delta[0], r.pose[1] + delta[1]))
            if r.robot_id == 0
            else r
            for r in scene.robots
        )
        history_a.append("drove")
        public_history.append("moved")
        scene = replace(scene, obj=moved, robots=robots)

    return int(scene.is_goal()), total, steps


def run_continuous_rollout(
    *,
    n_train_situations: int = 200,
    n_episodes: int = 120,
    epochs: int = 400,
    seed: int = 0,
    horizon: int = 6,
) -> List[Dict[str, object]]:
    """Train the threshold model once, then run every policy on the same episode set."""

    from ..grid.state import enumerate_hidden_physics

    train_situations = generate_situations(
        n_scenes=n_train_situations, lead_in=2, seed=seed
    )
    obs, action, can_push, thresholds, mask, physics_index, poses = _threshold_training(
        train_situations
    )
    positive_rate = float(can_push.mean())
    model = MultiHeadMLP(
        seed=seed,
        obs_dim=obs.shape[1],
        action_dim=action.shape[1],
        positive_weight=(1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0,
    )
    targets = np.concatenate([can_push[:, None], np.zeros((len(can_push), 2))], axis=1)
    train(
        model, obs, action, targets, physics_index, poses,
        thresholds=thresholds, threshold_mask=mask, epochs=epochs, seed=seed,
    )

    pool = enumerate_hidden_physics()
    rows: List[Dict[str, object]] = []
    for policy in ("oracle", "learned", "always_max", "always_min"):
        successes, costs = [], []
        for index in range(n_episodes):
            physics = pool[index % len(pool)]
            success, cost, _steps = run_continuous_episode(
                physics, policy, model, horizon=horizon, seed=index
            )
            successes.append(success)
            costs.append(cost)
        rate = float(np.mean(successes))
        mean_cost = float(np.mean(costs))
        rows.append(
            {
                "policy": policy,
                "n": n_episodes,
                "success_rate": rate,
                "mean_cost": mean_cost,
                "cost_per_success": (mean_cost / rate if rate > 0 else float("inf")),
            }
        )
    return rows


def format_continuous_rollout_markdown(rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "A push at force f costs 1 + f; the object needs two pushes to reach the goal and the "
        "force is the decision.",
        "",
        "| policy | episodes | success rate | mean cost | cost per success |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        per = (
            "n/a"
            if row["cost_per_success"] == float("inf")
            else f"{row['cost_per_success']:.2f}"
        )
        lines.append(
            f"| {row['policy']} | {row['n']} | {row['success_rate']:.3f} | "
            f"{row['mean_cost']:.3f} | {per} |"
        )
    return "\n".join(lines)
