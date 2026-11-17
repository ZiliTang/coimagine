"""Lost surfaces and braking must report zero normal command without crashing."""
import math

import pytest

from coimagine.random_start import spec as R
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.system import OptionSystemEnv
from experiments.planar_tasks.tasks import task_catalog


@pytest.fixture
def observation():
    # Construct sensors only: no physics integration or policy rollout.
    world = OptionSystemEnv(task_catalog()['planar_open_anchor'].instance)
    return world.boundary_observations()[0]


@pytest.mark.parametrize('action_id', [3, 34, 37])
def test_lost_surface_preserves_exit_and_zero_command(observation, monkeypatch, action_id):
    skill = S.make_skill(S.ACTIONS[action_id])
    skill.pending_exit_reason = 'effect_face_lost'
    monkeypatch.setattr(skill, '_observe_face', lambda obs: None)
    force, info = skill.step(observation, R.CONTROL_DT)
    assert skill.done and skill.exit_reason == 'effect_face_lost'
    assert skill.radial_command == 0.
    assert info['effect']['radial'] == 0.
    assert info['desired_velocity'] == (0., 0.)
    assert all(math.isfinite(x) for x in force)


def test_braking_does_not_reuse_previous_normal_command(observation):
    skill = S.make_skill(S.ACTIONS[3])
    skill.step(observation, R.CONTROL_DT)
    assert skill.initial_face is not None
    skill.stage = 'brake'
    skill.radial_command = .17
    _, info = skill.step(observation, R.CONTROL_DT)
    assert skill.radial_command == 0.
    assert info['effect']['radial'] == 0.
    assert info['desired_velocity'] == (0., 0.)
