"""Event, joint-mask and PPO semantics at real controller boundaries."""
import math
import numpy as np
import pytest
import torch

from coimagine.continuous2d.scenes import Pose
from coimagine.random_start.layout import InstanceSpec
from experiments.option_rl_v20.event_env import EventConfig, EventEnv, legal_pair
from experiments.option_rl_v20.interface import Policy
from experiments.option_rl_v20.train import load_checkpoint, save_checkpoint
from experiments.option_rl_v45.train import RunningMeanStd
from experiments.option_rl_v19 import spec as S

def event_world(case, config=EventConfig()):
    if case == 'approach_east':
        obj = Pose(2.5, 2., -math.pi/2)
        robots = (Pose(2.095, 2., 0.), Pose(4.4, .5, 0.))
        goal = (2.95, 2.)
    elif case == 'dual_north':
        obj = Pose(2.5, 2., 0.)
        robots = (Pose(2.75, 1.595, math.pi/2+.12), Pose(2.25, 1.595, math.pi/2-.12))
        goal = (2.5, 2.45)
    elif case == 'south_after_reposition':
        obj = Pose(2.5, 2., 0.)
        robots = (Pose(2.48, 1.595, math.pi/2), Pose(4.4, .5, 0.))
        goal = (2.5, 1.55)
    else:
        raise ValueError(case)
    return EventEnv(InstanceSpec(name=case, map='L0', split='test', index=0, seed=1,
                    arena=(5., 4.), obstacles=(), object_pose=obj,
                    robot_poses=robots, goal_centre=goal), config)


def test_single_push_contact_then_two_progress_events_keep_q_and_accumulate_reward():
    world = event_world('approach_east')
    first = world.step(2, 44)
    assert first['reason'] == 'contact' and first['contact'][0]
    assert first['progress'] == [0., 0.]
    skill = world.env.slots[0].skill
    q_id = skill.candidate.id
    events = [first, world.step(2, 44), world.step(2, 44)]
    assert [e['reason'] for e in events] == ['contact', 'progress', 'progress']
    assert all(.2 <= e['progress'][0] < .23 for e in events[1:])
    assert world.env.slots[0].skill is skill and skill.candidate.id == q_id
    assert world.env.skills_used == 3
    assert sum(e['reward'] for e in events) == pytest.approx(world.env.reward_total, abs=1e-9)
    assert sum(e['components']['time_cost'] for e in events) == pytest.approx(-.04*sum(e['elapsed_s'] for e in events))


def test_dual_push_contact_then_progress_and_reposition_progress():
    world = event_world('dual_north')
    contact = world.step(10, 10)
    assert contact['reason'] == 'contact' and contact['contact'] == [True, True]
    step = world.step(10, 10)
    assert step['reason'] == 'progress' and max(step['progress']) >= .2
    move = event_world('south_after_reposition').step(43, 44)
    assert move['reason'] == 'progress' and .2 <= move['progress'][0] < .23


def test_timeout_wait_and_invalid_pairs():
    world = event_world('approach_east', EventConfig(acquisition_timeout_s=.10))
    event = world.step(2, 44)
    assert event['reason'] == 'acquisition_timeout' and event['periods'] == 2
    move_world = event_world('south_after_reposition', EventConfig(progress_timeout_s=.05))
    move_timeout = move_world.step(43, 44)
    assert move_timeout['reason'] == 'progress_timeout' and move_timeout['periods'] == 1
    wait = world.step(44, 44)
    assert wait['reason'] == 'wait' and wait['elapsed_s'] == pytest.approx(1.)
    for a, b in ((2, 43), (43, 2), (32, 44), (45, 44), (42, 44)):
        with pytest.raises(ValueError):
            world.step(a, b)


def test_contact_loss_produces_one_decision_after_stable_loss():
    world = event_world('approach_east')
    assert world.step(2, 44)['reason'] == 'contact'
    # Perturb the physical fixture after contact, then let the normal control
    # and sensor path detect separation on subsequent ticks.
    body = world.env.env.backend.robot_bodies[0]
    body.position = (body.position.x-.3, body.position.y)
    body.velocity = (0., 0.)
    event = world.step(2, 44)
    assert event['reason'] == 'contact_lost'
    assert event['elapsed_s'] == pytest.approx(.25)
    assert not event['contact'][0]


def test_policy_probabilities_obey_joint_mask_and_round_trip():
    world = event_world('dual_north')
    obs, m0, m1 = world.observe()
    torch.manual_seed(51)
    policy = Policy(len(obs), 64)
    x = torch.tensor(obs)[None, :]
    gen = torch.Generator().manual_seed(52)
    with torch.no_grad():
        for _ in range(100):
            actions, logp, value = policy.act(x, (m0, m1), generator=gen)
            a, b = (int(v.item()) for v in actions)
            assert m0[a] and m1[b] and legal_pair(S.ACTIONS[a], S.ACTIONS[b])
            distributions, recomputed_value = policy.distributions(x, (m0, m1), a0=actions[0])
            assert torch.isfinite(logp) and torch.isfinite(value)
            assert logp.item() == pytest.approx(sum(d.log_prob(v).item() for d, v in zip(distributions, actions)), abs=1e-6)
            assert value.item() == pytest.approx(recomputed_value.item())
        assert int(policy.legal_pairs.sum()) == 1097


def test_tick_splitting_preserves_same_fixed_action_physics_and_reward():
    one, four = event_world('approach_east'), event_world('approach_east')
    four.env.decision_ticks = 4
    one.env.boundary_observations()
    for _ in range(10):
        parts = [one.env.joint_step(2, 44) for _ in range(4)]
        boundary = four.env.joint_step(2, 44)
        np.testing.assert_allclose(one.env.object_state().as_tuple(), four.env.object_state().as_tuple(), rtol=0, atol=1e-12)
        for a, b in zip(one.env.robot_states(), four.env.robot_states()):
            np.testing.assert_allclose(a.as_tuple(), b.as_tuple(), rtol=0, atol=1e-12)
        assert sum(p['reward']['total'] for p in parts) == pytest.approx(boundary['reward']['total'], abs=1e-12)


def test_checkpoint_round_trip_checks_action_and_event_contract(tmp_path):
    world = event_world('approach_east')
    obs, m0, m1 = world.observe()
    model = Policy(len(obs), 64)
    normalizer = RunningMeanStd(len(obs))
    normalizer.update(obs[None, :])
    optimizers = dict(actor=torch.optim.Adam(model.actor_parameters()),
                      critic=torch.optim.Adam(model.critic_parameters()))
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, model, normalizer, optimizers, 1, 51)
    restored, normalizer2, data = load_checkpoint(path)
    np.testing.assert_array_equal(normalizer.mean, normalizer2.mean)
    x = torch.as_tensor(obs)[None, :]
    with torch.no_grad():
        p1 = model.distributions(x, (m0, m1))[0][0].probs
        p2 = restored.distributions(x, (m0, m1))[0][0].probs
    torch.testing.assert_close(p1, p2, rtol=0., atol=0.)
    data['metadata']['action_version'] = 'different-action-table'
    torch.save(data, path)
    with pytest.raises(ValueError, match='incompatible'):
        load_checkpoint(path)
