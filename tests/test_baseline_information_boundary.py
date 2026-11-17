"""Nested diagnostic data must never be serialized to generated programs."""
from dataclasses import replace
import json

from experiments.planar_tasks.interface import (
    TaskEnvironment, policy_observation, SCAN_FIELDS, KINEMATICS_FIELDS, PATCH_FIELDS,
)
from experiments.planar_tasks.tasks import task_catalog


def test_raw_labels_and_future_nested_fields_do_not_cross_policy_boundary():
    env = TaskEnvironment(task_catalog()['planar_wall_anchor'].instance)
    observations = env._env.boundary_observations()
    expected = policy_observation(observations, [{}, {}], 0)
    changed = []
    for observation in observations:
        scan = replace(observation.scan, raw_labels=('PRIVATE_OBJECT_ID',) * len(observation.scan.valid))
        object.__setattr__(scan, 'future_private_pose', (12, 34, 56))
        changed.append(replace(observation, scan=scan))
    assert policy_observation(changed, [{}, {}], 0) == expected
    for robot in expected['robots']:
        assert set(robot['scan']) == set(SCAN_FIELDS)
        assert set(robot['own']) == set(robot['teammate']) == set(KINEMATICS_FIELDS)
        assert all(set(p) == set(PATCH_FIELDS) for p in robot['patches'])
    assert 'raw_labels' not in json.dumps(expected)


def test_future_nested_kinematics_and_patch_fields_are_not_serialized():
    env = TaskEnvironment(task_catalog()['planar_wall_anchor'].instance)
    observations = env._env.boundary_observations()
    original = policy_observation(observations, [{}, {}], 0)
    for observation in observations:
        object.__setattr__(observation.own, 'hidden_mass', 777)
        for patch in observation.patches:
            object.__setattr__(patch, 'hidden_face_id', 'secret')
    assert policy_observation(observations, [{}, {}], 0) == original
