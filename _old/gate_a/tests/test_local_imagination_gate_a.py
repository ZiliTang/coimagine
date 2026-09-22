from coimagine.core.types import Direction, OptionKind, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import (
    generate_local_imagination_channel_showcase,
    generate_local_imagination_showcase,
)
from coimagine.evaluation.episodes import rollout_receding_horizon
from coimagine.evaluation.local_imagination import successor_support_diagnostics
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch


def test_current_observations_support_only_one_heavy_contact() -> None:
    instance = generate_local_imagination_showcase(0)
    env = GridEnvironment(observation_radius=6)
    generator = GridOptionGenerator(candidate_budget=8)
    planner = ExactTopologySearch(env, generator, horizon=3, max_arity=3)
    beliefs = planner.initialize_beliefs(instance.state)

    for belief in beliefs.values():
        push_slots = {
            option.slot_token
            for option in generator.generate(belief)
            if option.kind == OptionKind.PUSH
        }
        assert push_slots <= {"north:0"}
    assert any(
        option.kind == OptionKind.REPOSITION
        for option in generator.generate(beliefs[1])
    )


def test_h3_iet_and_oracle_find_reposition_approach_push_chain() -> None:
    instance = generate_local_imagination_showcase(0)
    env = GridEnvironment(observation_radius=6)
    generator = GridOptionGenerator(candidate_budget=8)

    h2 = ExactTopologySearch(env, generator, horizon=2, max_arity=3)
    beliefs_h2 = h2.initialize_beliefs(instance.state)
    h2_iet = h2.plan(instance.state, beliefs_h2, mode=PlannerMode.IET)
    assert all(option.kind == OptionKind.IDLE for option in h2_iet.joint_option.options)

    h3 = ExactTopologySearch(env, generator, horizon=3, max_arity=3)
    beliefs_h3 = h3.initialize_beliefs(instance.state)
    cost = h3.plan(instance.state, beliefs_h3, mode=PlannerMode.COST)
    iet = h3.plan(instance.state, beliefs_h3, mode=PlannerMode.IET)
    oracle = h3.plan(
        instance.state,
        beliefs_h3,
        mode=PlannerMode.PRIVILEGED_IET_ORACLE,
    )

    assert all(option.kind == OptionKind.IDLE for option in cost.joint_option.options)
    assert iet.joint_option.stable_key == oracle.joint_option.stable_key
    assert [
        {option.kind for option in step.joint_option.options}
        for step in iet.trace
    ] == [
        {OptionKind.APPROACH, OptionKind.REPOSITION},
        {OptionKind.APPROACH, OptionKind.IDLE},
        {OptionKind.PUSH},
    ]
    assert iet.value > cost.value


def test_receding_horizon_h3_completes_open_chain() -> None:
    instance = generate_local_imagination_showcase(0)
    h2 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.IET,
        horizon=2,
        candidate_budget=8,
        max_arity=3,
        max_steps=5,
        observation_radius=6,
    )
    h3 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.IET,
        horizon=3,
        candidate_budget=8,
        max_arity=3,
        max_steps=5,
        observation_radius=6,
    )
    cost_h3 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.COST,
        horizon=3,
        candidate_budget=8,
        max_arity=3,
        max_steps=5,
        observation_radius=6,
    )
    assert not h2.completed
    assert h3.completed
    assert h3.option_count == 3
    assert not cost_h3.completed


def test_optimistic_surprise_is_detected_by_privileged_oracle() -> None:
    instance = generate_local_imagination_showcase(0, surprise_obstacle=True)
    env = GridEnvironment(observation_radius=6)
    generator = GridOptionGenerator(candidate_budget=8)
    planner = ExactTopologySearch(env, generator, horizon=3, max_arity=3)
    beliefs = planner.initialize_beliefs(instance.state)
    iet = planner.plan(instance.state, beliefs, mode=PlannerMode.IET)
    oracle = planner.plan(
        instance.state,
        beliefs,
        mode=PlannerMode.PRIVILEGED_IET_ORACLE,
    )
    assert iet.joint_option.stable_key != oracle.joint_option.stable_key
    assert any(option.kind == OptionKind.REPOSITION for option in iet.joint_option.options)
    assert all(option.kind == OptionKind.IDLE for option in oracle.joint_option.options)


def test_successor_support_diagnostics_distinguish_cost_and_iet() -> None:
    record = successor_support_diagnostics(
        generate_local_imagination_showcase(0),
        horizon=3,
        observation_radius=6,
        candidate_budget=8,
        max_arity=3,
    )
    assert record.optimistic_new_option_count > 0
    assert record.oracle_new_option_count > 0
    assert record.cost_new_option_count == 0
    assert record.new_option_precision == 1.0
    assert record.new_option_recall == 1.0
    assert record.iet_matches_oracle_root


def test_surprise_support_reports_optimistic_false_positives() -> None:
    record = successor_support_diagnostics(
        generate_local_imagination_showcase(0, surprise_obstacle=True),
        horizon=3,
        observation_radius=6,
        candidate_budget=8,
        max_arity=3,
    )
    assert record.option_false_positive_count > 0
    assert record.edge_false_positive_count > 0
    assert record.new_option_precision == 0.0
    assert record.new_option_recall is None
    assert record.retained_oracle_gain is None


def test_channel_showcase_has_open_passage_and_multiple_blockers() -> None:
    instance = generate_local_imagination_channel_showcase(0)
    state = instance.state
    channel_y = int(state.task.metadata["channel_y"])
    assert len(state.blockers) == 4
    assert (0, channel_y) not in state.walls
    assert (state.width - 1, channel_y) not in state.walls
    assert state.blockers["H"].required_pushers == 2
    assert tuple(state.task.metadata["reposition_anchors"])


def test_mixed_benchmark_randomizes_positions_and_keeps_easy_cst_cases() -> None:
    from coimagine.envs.grid.generation.procedural import (
        generate_local_imagination_benchmark_variant,
    )

    instances = [generate_local_imagination_benchmark_variant(seed) for seed in range(8)]
    assert len({tuple(robot.position for robot in item.state.robots.values()) for item in instances}) > 4
    easy = next(item for item in instances if item.state.task.metadata["dependency_depth"] <= 2)
    trace = rollout_receding_horizon(
        easy,
        mode=PlannerMode.COST,
        horizon=2,
        candidate_budget=10,
        max_arity=3,
        max_steps=6,
        observation_radius=6,
    )
    assert trace.completed


def test_mixed_benchmark_hard_open_case_requires_iet_h3() -> None:
    from coimagine.envs.grid.generation.procedural import (
        generate_local_imagination_benchmark_variant,
    )

    instance = generate_local_imagination_benchmark_variant(3)
    assert instance.state.task.metadata["dependency_depth"] == 3
    assert not instance.state.task.metadata["surprise_obstacle"]
    cst_h3 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.COST,
        horizon=3,
        candidate_budget=10,
        max_arity=3,
        max_steps=6,
        observation_radius=6,
    )
    iet_h2 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.IET,
        horizon=2,
        candidate_budget=10,
        max_arity=3,
        max_steps=6,
        observation_radius=6,
    )
    iet_h3 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.IET,
        horizon=3,
        candidate_budget=10,
        max_arity=3,
        max_steps=6,
        observation_radius=6,
    )
    assert not cst_h3.completed
    assert not iet_h2.completed
    assert iet_h3.completed


def test_v08_explicit_family_strata_and_surprise_split() -> None:
    from coimagine.envs.grid.generation.procedural import (
        LOCAL_IMAGINATION_FAMILIES,
        generate_local_imagination_benchmark_case,
    )

    open_cases = [
        generate_local_imagination_benchmark_case(11, family=family, scene_type="open")
        for family in LOCAL_IMAGINATION_FAMILIES
    ]
    assert {item.state.task.metadata["scene_family"] for item in open_cases} == set(
        LOCAL_IMAGINATION_FAMILIES
    )
    assert all(item.state.task.metadata["scene_type"] == "open" for item in open_cases)
    surprise = generate_local_imagination_benchmark_case(
        11,
        family="reposition_approach_push",
        scene_type="surprise",
    )
    assert surprise.state.task.metadata["surprise_obstacle"]
    assert surprise.state.task.metadata["benchmark_version"] == "v0.8"


def test_v08_stratified_quick_counts_and_paired_summary() -> None:
    from coimagine.evaluation.local_imagination import (
        run_stratified_horizon_study,
        summarize_dependency_study,
        summarize_paired_study,
    )

    records, support = run_stratified_horizon_study(
        open_per_family=1,
        surprise_cases=1,
        jobs=1,
    )
    assert len(records) == 5 * 2 * 3
    assert len(support) == 2
    dependency = summarize_dependency_study(records)
    open_families = {
        row["scene_family"] for row in dependency if row["scene_type"] == "open"
    }
    assert len(open_families) == 4
    paired = summarize_paired_study(records)
    assert any(row["comparison"] == "IET H=3 vs CST H=3" for row in paired)


def test_v010_randomized_channel_generator_has_two_sided_unscreened_variation() -> None:
    from coimagine.envs.grid.generation.procedural import (
        generate_randomized_channel_benchmark_case,
    )

    instances = [generate_randomized_channel_benchmark_case(seed) for seed in range(48)]
    assert {item.state.task.metadata["difficulty_variant"] for item in instances} == {
        "easy",
        "moderate",
        "hard",
    }
    assert len({len(item.state.blockers) for item in instances}) >= 3
    assert len({item.state.task.metadata["observation_radius"] for item in instances}) >= 3
    assert len({tuple(robot.position for robot in item.state.robots.values()) for item in instances}) >= 12
    assert any(item.state.task.metadata["future_support_required"] for item in instances)
    assert any(not item.state.task.metadata["future_support_required"] for item in instances)
    assert {
        item.state.task.metadata["initial_position_mode"] for item in instances
    } == {"same_side", "split_side", "cross_side_required"}
    assert any(
        Direction.NORTH in blocker.allowed_push_directions
        for item in instances
        for blocker in item.state.blockers.values()
    )
    assert any(
        Direction.SOUTH in blocker.allowed_push_directions
        for item in instances
        for blocker in item.state.blockers.values()
    )
    for item in instances:
        state = item.state
        channel_y = int(state.task.metadata["channel_y"])
        assert (0, channel_y) not in state.walls
        assert (state.width - 1, channel_y) not in state.walls
        assert state.task.metadata["benchmark_version"] == "v0.10"
        assert int(state.task.metadata["episode_max_steps"]) >= 3 * len(state.blockers)
        anchors = tuple(state.task.metadata["reposition_anchors"])
        assert any(y < channel_y for _, y in anchors)
        assert any(y > channel_y + 1 for _, y in anchors)
        assert set(robot.position for robot in state.robots.values()).isdisjoint(state.walls)
        assert set(robot.position for robot in state.robots.values()).isdisjoint(
            set().union(*(set(blocker.cells) for blocker in state.blockers.values()))
        )


def test_v010_randomized_worker_preserves_three_m_step_floor() -> None:
    from coimagine.envs.grid.generation.procedural import (
        generate_randomized_channel_benchmark_case,
    )
    from coimagine.evaluation.local_imagination import (
        resolve_randomized_channel_budgets,
    )

    for difficulty in ("easy", "moderate", "hard"):
        instance = generate_randomized_channel_benchmark_case(17, difficulty=difficulty)
        candidate, steps, floor = resolve_randomized_channel_budgets(
            instance,
            candidate_budget=5,
            max_steps_cap=6,
        )
        assert candidate == 5
        assert floor == 3 * len(instance.state.blockers)
        assert steps >= floor



def test_v09_randomized_channel_summary_accepts_mixed_outcomes() -> None:
    from dataclasses import replace

    from coimagine.evaluation.local_imagination import (
        summarize_randomized_channel_study,
    )
    from coimagine.evaluation.records import EpisodeResultRecord

    base = EpisodeResultRecord(
        instance_id="x", suite="randomized_channel", task_name="channel", seed=0,
        method="CST H=3", planner_mode="cost", horizon=3, observation_radius=6,
        robot_count=2, entity_count=3, completed=False, option_count=5,
        total_duration=5.0, final_potential=0.5, realized_return=0.3,
        total_planning_seconds=0.01, mean_planning_seconds=0.002,
        first_action="", first_predicted_continuation="", first_created_edges=0,
        first_destroyed_edges=0, first_event="none", scene_family="randomized_channel",
        dependency_depth=3, scene_type="open", difficulty_variant="easy",
    )
    records = [
        base,
        replace(base, instance_id="y", seed=1, completed=True, final_potential=1.0),
        replace(base, instance_id="z", seed=2, difficulty_variant="hard", method="CoImagine-IET H=3", completed=True, final_potential=1.0),
    ]
    summary = summarize_randomized_channel_study(records)
    easy_cst = next(row for row in summary if row["difficulty"] == "easy" and row["method"] == "CST")
    assert easy_cst["completion_rate"] == 0.5
    assert any(row["difficulty"] == "all" for row in summary)


def test_v010_randomized_summary_plot_clamps_roundoff_yerr(tmp_path) -> None:
    from coimagine.reporting.local_gate_a import plot_randomized_channel_summary

    rows = []
    for difficulty in ("easy", "moderate", "hard"):
        for method in ("CST", "CoImagine-IET"):
            rows.append(
                {
                    "difficulty": difficulty,
                    "method": method,
                    "horizon": 3,
                    "completion_rate": 0.5,
                    # Deliberately invert the bounds by machine-scale amounts to
                    # reproduce the Matplotlib 3.10 negative-yerr failure.
                    "completion_ci_low": 0.5000000000000001,
                    "completion_ci_high": 0.4999999999999999,
                    "final_potential_median": 0.5,
                    "final_potential_q25": 0.5000000000000001,
                    "final_potential_q75": 0.4999999999999999,
                    "planning_ms_median": 1.0,
                    "planning_ms_q25": 1.0000000000000002,
                    "planning_ms_q75": 0.9999999999999998,
                }
            )
    plot_randomized_channel_summary(rows, tmp_path)
    assert (tmp_path / "fig_gate_a_randomized_channel_summary.pdf").exists()
    assert (tmp_path / "fig_gate_a_randomized_channel_summary.png").exists()
