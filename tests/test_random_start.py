"""Focused tests for the random-start round (scene legality, sensor, skills, reward, protocol).

These are deliberately small: each test builds one or two instances so the suite stays fast. The
heavier acceptance batteries live in ``experiments/random_start_reposition_v1/``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from coimagine.random_start import env as E
from coimagine.random_start import geometry as G
from coimagine.random_start import layout, lidar, skills, verify
from coimagine.random_start.spec import (
    GOAL_DISTANCE_MAX,
    GOAL_DISTANCE_MIN,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    ROBOT_ACTION_COUNT,
    ROBOT_OBJECT_GAP_MAX,
    ROBOT_OBJECT_GAP_MIN,
    ROBOT_ACTIONS,
    ROBOT_RADIUS,
)


@pytest.fixture(scope="module")
def instances():
    panel, _ = layout.build_split("L0", "train", count=3)
    return panel


def test_sampled_scenes_satisfy_the_contract(instances):
    report = layout.scene_legality_report(instances)
    assert report["ok"], report["problems"]
    for instance in instances:
        polygon = G.box_polygon(instance.object_pose.x, instance.object_pose.y,
                                instance.object_pose.angle, OBJECT_WIDTH / 2.0,
                                OBJECT_HEIGHT / 2.0)
        for pose in instance.robot_poses:
            gap = G.circle_polygon_gap((pose.x, pose.y), ROBOT_RADIUS, polygon)
            assert ROBOT_OBJECT_GAP_MIN - 1e-9 <= gap <= ROBOT_OBJECT_GAP_MAX + 1e-9
        distance = math.dist((instance.object_pose.x, instance.object_pose.y),
                             instance.goal_centre)
        assert GOAL_DISTANCE_MIN - 1e-9 <= distance <= GOAL_DISTANCE_MAX + 1e-9


def test_scanner_classes_and_no_return(instances):
    instance = instances[0]
    states = [(pose.x, pose.y, pose.angle, 0.0, 0.0, 0.0) for pose in instance.robot_poses]
    scan = lidar.scan_lidar(instance.arena, (), (instance.object_pose.x, instance.object_pose.y,
                                                 instance.object_pose.angle), states, 0, 0, 0.0)
    assert len(scan.distances) == 72
    assert all(0.0 <= distance <= lidar.LIDAR_RANGE + 1e-9
               for distance, valid in zip(scan.distances, scan.valid) if valid)
    assert set(scan.category) <= {lidar.CATEGORY_NONE, lidar.CATEGORY_OBJECT,
                                  lidar.CATEGORY_STATIC, lidar.CATEGORY_TEAMMATE}
    # a scan never reports an object-class hit beyond the range
    for distance, valid, category in zip(scan.distances, scan.valid, scan.category):
        if valid and category == lidar.CATEGORY_OBJECT:
            assert distance <= lidar.LIDAR_RANGE


def test_scanner_matches_independent_ray_march(instances):
    report = verify.check_lidar_against_brute_force(instances[:1])
    assert report["ok"], report


def test_fitted_normal_comes_from_returns(instances):
    report = verify.check_fit_is_from_returns(instances)
    assert report["ok"], report


def test_information_boundary_counterfactual():
    report = verify.check_information_boundary()
    assert report["ok"], report


def test_one_world_advance_per_joint_period(instances):
    report = verify.check_one_world_per_period(instances[0])
    assert report["ok"], report


def test_reward_matches_an_independent_recomputation(instances):
    report = verify.check_reward_recomputation(instances)
    assert report["ok"], report


def test_success_hold_is_an_environment_clock(instances):
    report = verify.check_hold_across_skills_and_wait(instances[0])
    assert report["ok"], report


def test_wait_brake_never_commands_a_push(instances):
    environment = E.RandomStartEnv(instances[0])
    environment.joint_step(3, 13)
    boundary = environment.joint_step(13, 13)
    rows = environment.rows[-boundary["periods"]:]
    assert all(math.hypot(*row["r0"]["desired_velocity"]) <= 1e-12 for row in rows)
    assert all(entry["exit_reason"] in ("wait_complete", "safety_abort", "timeout",
                                        "joint_window_end", None)
               for entry in (boundary["skill_ends"][0],))


def test_every_action_index_is_used_and_unique():
    assert len(ROBOT_ACTIONS) == ROBOT_ACTION_COUNT
    assert len({action.index for action in ROBOT_ACTIONS}) == ROBOT_ACTION_COUNT
    assert {action.kind for action in ROBOT_ACTIONS} == {"push", "reposition", "wait"}


def test_observation_whitelist_has_no_forbidden_field(instances):
    environment = E.RandomStartEnv(instances[0])
    observation = environment.observe(0, 0, 0.0)
    assert observation.forbidden_fields() == ()


def test_tick_time_axis_and_branch_restore(instances):
    assert verify.check_tick_and_time_axis(instances[0])["ok"]
    assert verify.check_branch_restore(instances[0])["ok"]
