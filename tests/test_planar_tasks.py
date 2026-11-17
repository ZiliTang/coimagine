"""Physical geometry, information boundary, rotation access and bounded process I/O."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from coimagine.continuous2d.scenes import Pose, Box
from experiments.planar_tasks.tasks import task_catalog, geometry_audit
from experiments.planar_tasks.interface import TaskEnvironment, policy_observation, validate_action
from experiments.planar_tasks.policy import JsonlPolicy
from experiments.planar_tasks.runner import RunBudget, run_episode


def wall():
    return task_catalog()['planar_wall_anchor']


def test_catalog_is_deterministic_and_all_variants_stay_in_development():
    first, second = task_catalog(), task_catalog()
    assert len(first) == 9
    assert {t.family for t in first.values()} == {'open', 'wall', 'detour'}
    assert [t.sha256 for t in first.values()] == [t.sha256 for t in second.values()]
    assert len({t.root_id for t in first.values()}) == 3
    for t in first.values():
        assert t.instance.split == 'development'
        assert geometry_audit(t.instance)['legal']
        assert t.record()['feasibility'] == 'unverified'


def test_audit_checks_actual_obstacles_robot_overlap_and_goal():
    inst = wall().instance
    assert not geometry_audit(replace(inst, object_pose=Pose(1.05, 2., 0.)))['legal']
    assert not geometry_audit(replace(inst, robot_poses=(Pose(1.05, 2.), inst.robot_poses[1])))['legal']
    assert not geometry_audit(replace(inst, goal_centre=(1.05, 2.)))['legal']
    # This arbitrary obstacle is not any old map's built-in box.
    assert not geometry_audit(replace(inst, obstacles=(Box(1.45, 1.55, .1, .1),)))['legal']


def test_rotation_exposed_without_changing_v2_rl_contract():
    from experiments.option_rl_v21.event_env import action_masks, legal_pair
    from experiments.option_rl_v21 import spec as S
    world = TaskEnvironment(wall().instance)
    obs = world.observe()
    assert [36, 36] in obs['legal_pairs']
    assert not legal_pair(S.ACTIONS[36], S.ACTIONS[36])
    assert not action_masks(world._env.boundary_observations())[0][36]
    world.step(dict(a0=36, a1=36, ticks=3))
    assert world._env.total_periods == world._env.world_advances == 3
    assert world._env.coordinator.rows[-1]['families'] == ['ROTATE', 'ROTATE']
    assert world.observe()['robots'][0]['tick'] == 3


def test_payload_uses_only_sensors_and_whitelisted_execution_feedback():
    world = TaskEnvironment(wall().instance)
    observations = world._env.boundary_observations()
    plain = policy_observation(observations, [{}, {}], 0)
    poisoned = policy_observation(observations,
        [dict(object_pose=[99, 99], reward=99, true_friction=99), {}], 0)
    assert plain == poisoned
    text = json.dumps(plain, allow_nan=False)
    for key in ('object_pose', 'true_friction', 'reward', 'scene_meta', 'root_id', 'obstacles'):
        assert f'"{key}"' not in text
    # Holding observable inputs fixed also fixes the menu when hidden metadata changes.
    world._env.instance = replace(world._env.instance, seed=999, name='hidden_other_root')
    assert policy_observation(observations, [{}, {}], 0) == plain
    # Change hidden dynamics while holding positions/sensor history fixed.
    before = world.observe()
    world._env.env.backend.object_body.mass *= 2
    assert world.observe() == before


@pytest.mark.parametrize('action', [dict(a0=True, a1=44, ticks=1),
    dict(a0=44, a1=44, ticks=0), dict(a0=44, a1=44, ticks=41),
    dict(a0=38, a1=38, ticks=1), dict(a0=32, a1=36, ticks=1),
    dict(a0=44, a1=44, ticks=1, reward=0)])
def test_invalid_commands_do_not_advance_physics(action):
    world = TaskEnvironment(wall().instance)
    with pytest.raises(ValueError):
        world.step(action)
    assert world._env.total_periods == 0


def test_jsonl_policy_state_persists_and_partial_line_times_out(tmp_path):
    program = "import sys,json\nn=0\nfor line in sys.stdin:\n n+=1\n print(json.dumps({'n':n}),flush=True)"
    p = JsonlPolicy([sys.executable, '-u', '-c', program], tmp_path/'ok.stderr')
    try:
        assert p.request({'reset': True}, 2.) == {'n': 1}
        assert p.request({}, 2.) == {'n': 2}
    finally:
        p.close()
    p = JsonlPolicy([sys.executable, '-u', '-c',
        "import sys,time;sys.stdin.readline();sys.stdout.write('{');sys.stdout.flush();time.sleep(5)"],
        tmp_path/'partial.stderr')
    try:
        with pytest.raises(TimeoutError):
            p.request({}, .15)
    finally:
        p.close()
    assert p.process.poll() is not None


def test_runner_keeps_incomplete_denominator_and_replay_clock(tmp_path):
    out = tmp_path/'smoke'
    result = run_episode(wall(), [sys.executable, '-m', 'experiments.planar_tasks.policy'],
                         out, RunBudget(max_decisions=2, wall_seconds=10.))
    assert result['termination'] == 'decision_budget_exhausted'
    assert result['status'] == 'incomplete' and not result['success']
    assert result['attempted_episodes'] == 1
    assert result['task_cost_seconds'] == 180.
    assert result['executed_control_ticks'] == 8
    replay = json.loads((out/'replay.json').read_text())
    assert replay['frames'][-1]['t'] == pytest.approx(.4)
    assert len(replay['frames']) == len(replay['strip']) == 8
    assert replay['scene_meta']['obstacles']
    transcript = [json.loads(line) for line in (out/'sensors.jsonl').read_text().splitlines()]
    assert [r['request']['observation']['robots'][0]['tick'] for r in transcript if 'request' in r] == [0, 4]
    with pytest.raises(FileExistsError):
        run_episode(wall(), [], out)


def test_testbed_preserves_obstacles_and_rejects_illegal_drag():
    from experiments.option_rl_v19 import testbed_server as server
    inst = server.SCENES['planar_wall_anchor']
    meta = server._scene_dict(inst)
    assert meta['obstacles'][0]['half_w'] == .15
    restored = server._scene_instance(inst.name, meta['initial']['object'], meta['initial']['robots'])
    assert restored.obstacles == inst.obstacles
    with pytest.raises(ValueError, match='illegal planar scene'):
        server._scene_instance(inst.name, dict(x=1.05, y=2., angle=0.), meta['initial']['robots'])


def test_runner_records_invalid_policy_output_without_retry_or_free_failure(tmp_path):
    program = "import sys;sys.stdin.readline();print('{\"a0\":999,\"a1\":44,\"ticks\":1}',flush=True)"
    out = tmp_path/'invalid'
    result = run_episode(wall(), [sys.executable, '-u', '-c', program], out,
                         RunBudget(max_decisions=2, wall_seconds=10.))
    assert result['termination'] == 'execution_error'
    assert result['error']['type'] == 'ValueError'
    assert result['executed_control_ticks'] == result['decisions'] == 0
    assert result['attempted_episodes'] == 1 and result['task_cost_seconds'] == 180.
    assert len((out/'sensors.jsonl').read_text().splitlines()) == 2
