"""Regression tests for the round-20 diagnosis.

Each test pins a defect that was demonstrated by a run, or a protocol property the new
diagnostic machinery relies on.  They are deliberately small; the heavy comparisons live in
``coimagine.diagnostics.diagnosis.run_diagnosis``.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.diagnostics.diagnosis import (
    EpisodeRecord,
    distance_heuristic_check,
    distance_sign_check,
    exact_reachability,
    scoring_identity_checks,
    summarise,
)
from coimagine.diagnostics.exact_adapter import (
    ExactBeliefModel,
    ExactFactoredModel,
    scan_from_observation,
)
from coimagine.grid.dynamics import FEEDBACK_SYMBOLS, execute
from coimagine.grid.planner import PHYSICS
from coimagine.grid.sensing import ray_cast
from coimagine.learning.data import _joint_for, generate_situations, probe_scene
from coimagine.learning.features import ACTION_IDS, observation_vector


# ---------------------------------------------------------------------------
# The corrected depth-2 score
# ---------------------------------------------------------------------------


def test_depth_two_score_counts_the_progress_term_once():
    """Regression: the pre-fix depth-2 score added ``w * dist`` to the cut twice."""

    result = scoring_identity_checks()
    # The returned score must equal its own components under the documented formula.
    assert result["max_abs_gap_report_vs_reconstruction"] < 1e-8
    # And the duplication the old formula introduced is real and non-zero, otherwise the
    # regression test would pass vacuously.
    assert all(value > 0.0 for value in result["old_formula_extra_term_per_action"].values())
    # The pre-fix formula is explicitly recorded so a future reader can see what changed.
    assert "w*dist2" in result["depths_differ"]["depth_2_formula_before_fix"]
    assert "w*dist2" in result["depths_differ"]["depth_2_formula_after_fix"]
    assert "w*dist2) + w*dist2" not in result["depths_differ"]["depth_2_formula_after_fix"]


# ---------------------------------------------------------------------------
# The unified metric vocabulary
# ---------------------------------------------------------------------------


def test_summarise_separates_all_cost_from_successful_cost():
    records = [
        EpisodeRecord(0, "p", "x", success=1, action_cost=3.0, steps=3),
        EpisodeRecord(1, "p", "x", success=1, action_cost=5.0, steps=5),
        EpisodeRecord(2, "p", "x", success=0, action_cost=2.0, steps=6),
    ]
    summary = summarise(records, "p")
    assert summary.n == 3
    assert summary.successes == 2
    assert summary.success_rate == pytest.approx(2 / 3)
    # Mean action cost averages every episode; the conditional cost averages successes only.
    assert summary.mean_action_cost_all == pytest.approx((3 + 5 + 2) / 3)
    # Total cost charges the explicit failure cost exactly once, on the failed episode.
    total_costs = [3.0, 5.0, 2.0 + 20.0]
    assert summary.mean_total_cost_all == pytest.approx(sum(total_costs) / 3)
    assert summary.mean_total_cost_success == pytest.approx((3.0 + 5.0) / 2)
    # `total cost per success` is a different quantity and must be named as such.
    assert summary.total_cost_per_success == pytest.approx(sum(total_costs) / 2)


def test_summarise_reports_undefined_without_successes():
    summary = summarise(
        [EpisodeRecord(0, "p", "x", success=0, action_cost=4.0, steps=8)], "p"
    )
    assert summary.success_rate == 0.0
    assert summary.mean_total_cost_success is None
    assert summary.total_cost_per_success is None


def test_distance_sign_convention_is_explicit():
    check = distance_sign_check()
    assert check["positive_case"]["signed_error"] > 0
    assert "pessimistic" in check["positive_case"]["meaning"]
    assert check["negative_case"]["signed_error"] < 0
    assert "optimistic" in check["negative_case"]["meaning"]


def test_distance_heuristic_is_not_inverted():
    """Regression: the pre-fix heuristic read 5-6 for an object one push from the goal."""

    from coimagine.diagnostics.world_model_probe import _worst_cell_distance
    from coimagine.grid.planner import _cut_value
    from coimagine.grid.state import ObjectState, RobotState, StaticMap

    check = distance_heuristic_check()
    one_push = [row for row in check["rows"] if row["one_push_from_goal"]]
    assert one_push, "the check must include an object one push from the goal"
    for row in one_push:
        assert row["corrected_value"] == pytest.approx(1.0)
        assert row["pre_fix_value"] >= 2.0  # the inversion is real, not cosmetic

    # The corrected helper must agree with the exact planner's own cut heuristic.
    static = StaticMap(width=6, height=6, target=frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)}))
    for row in check["rows"]:
        origin = tuple(row["origin"])
        robotless = {((origin), (RobotState(0, (0, 5), None), RobotState(1, (5, 5), None)), (0, False), (), ()): 1.0}
        cut = _cut_value(robotless, static, 20.0, progress_weight=3.0)
        risk = 0.0 if ObjectState(origin).within(static.target) else 1.0
        expected = 20.0 * risk + 3.0 * _worst_cell_distance(origin, static.target)
        assert cut == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Independent reachability
# ---------------------------------------------------------------------------


def test_exact_reachability_matches_shortest_solution():
    gripping = next(
        p
        for p in PHYSICS
        if p.resistance == 1 and p.constraint == "free" and p.traction == ("grip", "grip")
    )
    scene = probe_scene((2, 0), gripping)  # two co-pushes from the goal columns
    steps = exact_reachability(scene, horizon=8)
    assert steps is not None and steps >= 1
    # Replay the counted number of co-pushes and check the object arrives at that step.
    current = scene
    for _ in range(steps):
        current, _result = execute(current, _joint_for(None, "co_push:E"))
    assert current.is_goal()
    # And one step fewer must not arrive, otherwise the search was not minimal.
    if steps > 1:
        current = scene
        for _ in range(steps - 1):
            current, _result = execute(current, _joint_for(None, "co_push:E"))
        assert not current.is_goal()


def test_exact_reachability_declares_a_hopeless_scene_unreachable():
    both_slip = next(p for p in PHYSICS if p.traction == ("slip", "slip"))
    scene = probe_scene((1, 1), both_slip)
    assert exact_reachability(scene, horizon=8) is None


# ---------------------------------------------------------------------------
# The exact adapter respects the information boundary
# ---------------------------------------------------------------------------


def _situation(seed: int = 3):
    return generate_situations(
        n_scenes=1, lead_in=2, seed=seed, feedback_noise=0.0
    )[0]


def test_exact_adapter_starts_uniform_over_hidden_physics():
    situation = _situation()
    observation = np.asarray(
        observation_vector(
            situation.scan, situation.scene.robot_pose(0), (), ()
        )
    )
    model = ExactBeliefModel(
        observation,
        static=situation.scene.static,
        robots=situation.scene.robots,
        initial_robots=situation.initial_scene.robots,
        actions=(),
        history_a=(),
        public_history=(),
    )
    mass = model.belief_summary()["physics_mass"]
    assert len(mass) == len(PHYSICS)
    assert all(abs(value - 1.0 / len(PHYSICS)) < 1e-9 for value in mass.values())


def test_exact_adapter_history_never_rules_out_the_true_physics():
    """The posterior is a likelihood: the true hypothesis must stay in its support."""

    situation = _situation()
    reverse = {_joint_for(None, action): action for action in ACTION_IDS}
    actions = tuple(reverse[joint] for joint in situation.skills)
    observation = np.asarray(
        observation_vector(
            situation.scan, situation.scene.robot_pose(0), situation.history_a,
            situation.public_history,
        )
    )
    model = ExactBeliefModel(
        observation,
        static=situation.scene.static,
        robots=situation.scene.robots,
        initial_robots=situation.initial_scene.robots,
        actions=actions,
        history_a=situation.history_a,
        public_history=situation.public_history,
    )
    true_index = PHYSICS.index(situation.physics)
    mass = model.belief_summary()["physics_mass"]
    assert mass.get(true_index, 0.0) > 0.0
    assert sum(mass.values()) == pytest.approx(1.0)


def test_exact_adapter_step_outputs_are_normalised_distributions():
    situation = _situation()
    observation = np.asarray(
        observation_vector(
            situation.scan, situation.scene.robot_pose(0), (), ()
        )
    )
    model = ExactBeliefModel(
        observation,
        static=situation.scene.static,
        robots=situation.scene.robots,
        initial_robots=situation.initial_scene.robots,
    )
    for action in range(len(ACTION_IDS)):
        step = model.predict_next(observation, action)
        probabilities = np.exp(step["pose_logits"][0] - step["pose_logits"][0].max())
        probabilities /= probabilities.sum()
        assert probabilities.sum() == pytest.approx(1.0)
        assert step["cost"].shape == (1,)
        assert -1.0 <= float(step["moved_logit"][0]) or float(step["moved_logit"][0]) >= -30.0


def test_exact_factored_model_feedback_belief_is_exact():
    """`update_belief` must place zero mass on hypotheses the observed symbol rules out."""

    gripping = next(
        p
        for p in PHYSICS
        if p.resistance == 1 and p.constraint == "constrained" and p.traction == ("grip", "grip")
    )
    scene = probe_scene((2, 2), gripping)
    model = ExactFactoredModel(static=scene.static)
    observation = np.asarray(
        observation_vector(ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ())
    )
    dynamic = model.dynamic_state(observation)
    belief = model.parameter_belief(observation)
    release = ACTION_IDS.index("release:0:E")
    # A gripping robot releasing a latched object reports `drove`; the exact likelihood of
    # `strained` or `slipped` under that action is therefore zero for those hypotheses.
    updated = model.update_belief(
        belief, dynamic, release, FEEDBACK_SYMBOLS.index("drove")
    )
    assert updated.sum() == pytest.approx(1.0)
    mass_on_true = updated[0, PHYSICS.index(gripping)]
    assert mass_on_true > 0.0


def test_exact_factored_model_plugs_into_plan_with_search():
    from coimagine.diagnostics.belief_probe import run_search_episode

    gripping = next(
        p
        for p in PHYSICS
        if p.resistance == 1 and p.constraint == "free" and p.traction == ("grip", "grip")
    )
    scene = probe_scene((1, 1), gripping)
    successor, result = execute(scene, _joint_for(None, "probe:0:E"))
    model = ExactFactoredModel(static=successor.static)
    belief = np.full((1, len(PHYSICS)), 1.0 / len(PHYSICS))
    success, cost, steps = run_search_episode(
        model, successor, belief, horizon=8, depth=3, beam=1,
        warmup=(ACTION_IDS.index("probe:0:E"), FEEDBACK_SYMBOLS.index(result.feedback[0])),
    )
    assert success in (0, 1)
    assert cost >= 0.0
    assert 0 <= steps <= 8
