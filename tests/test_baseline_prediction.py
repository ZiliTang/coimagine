"""Real nominal engine consistency, independent branches and RPC boundaries."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import time

import pytest

from experiments.planar_tasks.baseline.prediction import (
    PredictionService, PredictionLedger, LIMITS, validate_hypothesis)
from experiments.planar_tasks.interface import TaskEnvironment
from experiments.planar_tasks.tasks import task_catalog
from experiments.planar_tasks.policy import JsonlPolicy


def hypothesis(instance):
    return dict(arena=list(instance.arena), goal=list(instance.goal_centre),
                goal_radius=instance.goal_radius,
                object=[instance.object_pose.x, instance.object_pose.y, instance.object_pose.angle, 0, 0, 0],
                robots=[[r.x, r.y, r.angle, 0, 0, 0] for r in instance.robot_poses],
                obstacles=[[b.cx, b.cy, b.half_w, b.half_h, b.angle] for b in instance.obstacles])


def rpc(service, op, **args):
    service.begin_decision()
    return service.rpc(dict(method='predict', operation=op, **args), time.monotonic()+10)


@pytest.mark.parametrize('name', ['planar_open_anchor', 'planar_wall_anchor', 'planar_detour_anchor'])
def test_independent_nominal_rollout_and_fork_equal_execution(name):
    instance = task_catalog()[name].instance
    # Offline calibration uses a declared fixture, NOT a deployment observation.
    real = TaskEnvironment(instance)
    s = PredictionService(ledger=PredictionLedger(dict(LIMITS, decision_cpu_seconds=10)))
    first = rpc(s, 'create', hypothesis=hypothesis(instance)); h = first['handle']
    assert first['observation'] == real.observe()
    for pair in ([2, 2], [36, 36], [44, 44]):
        observation = real.observe()
        if pair not in observation['legal_pairs']:
            pair = [44, 44]
        action = dict(a0=pair[0], a1=pair[1], ticks=20)
        real.step(action)
        predicted = rpc(s, 'step', handle=h, action=action)
        assert predicted['state']['object'] == asdict(real._env.object_state())
        assert predicted['state']['robots'] == [asdict(r) for r in real._env.robot_states()]
        assert predicted['observation'] == real.observe()
    before = s.result(h)
    ticks = s.ledger.usage['ticks']
    fork = rpc(s, 'fork', handle=h)
    assert fork['state'] == before['state']
    assert fork['observation'] == before['observation']
    assert s.ledger.usage['ticks'] - ticks == 60
    rpc(s, 'step', handle=fork['handle'], action=dict(a0=44, a1=44, ticks=10))
    assert s.result(h)['state'] == before['state']
    assert real._env.total_periods == 60


def test_schema_no_execution_references_or_hidden_fields():
    h = hypothesis(task_catalog()['planar_open_anchor'].instance)
    for key in ('task', 'seed', 'execution_handle', 'friction', 'controller_state', 'snapshot'):
        with pytest.raises(ValueError):
            validate_hypothesis(dict(h, **{key: 1}))
    for value in (float('nan'), float('inf'), True, 1000):
        bad = deepcopy(h); bad['object'][0] = value
        with pytest.raises(ValueError):
            validate_hypothesis(bad)
    a, b = PredictionService(), PredictionService()
    # Unrelated execution world changes cannot change predictions.
    unrelated = TaskEnvironment(task_catalog()['planar_wall_anchor'].instance)
    ra = rpc(a, 'create', hypothesis=h)
    unrelated.step(dict(a0=44, a1=44, ticks=5))
    rb = rpc(b, 'create', hypothesis=h)
    assert ra['state'] == rb['state'] and ra['observation'] == rb['observation']
    with pytest.raises(ValueError):
        rpc(a, 'step', handle='execution', action=dict(a0=44, a1=44, ticks=1))


def test_budget_partial_step_discard_and_parent_preservation():
    s = PredictionService(ledger=PredictionLedger(dict(LIMITS, ticks=6, decision_cpu_seconds=10)))
    h = rpc(s, 'create', hypothesis=hypothesis(task_catalog()['planar_open_anchor'].instance))['handle']
    rpc(s, 'step', handle=h, action=dict(a0=44, a1=44, ticks=4))
    before = s.result(h)['state']
    with pytest.raises(RuntimeError, match='ticks budget'):
        rpc(s, 'fork', handle=h)
    assert s.ledger.usage['ticks'] == 6
    assert s.result(h)['state'] == before and len(s.worlds) == 1
    with pytest.raises(RuntimeError, match='ticks budget'):
        rpc(s, 'step', handle=h, action=dict(a0=44, a1=44, ticks=1))
    assert h not in s.worlds


def test_per_decision_limits_cannot_reset_via_rpc():
    s = PredictionService(ledger=PredictionLedger(dict(LIMITS, decision_calls=2)))
    request = dict(method='predict', operation='status')
    for _ in range(2):
        s.rpc(request, time.monotonic()+2)
    with pytest.raises(RuntimeError, match='per-decision calls'):
        s.rpc(request, time.monotonic()+2)
    s.begin_decision()
    s.rpc(request, time.monotonic()+2)


@pytest.mark.parametrize('available', [False, True])
def test_nested_runtime_rpc_and_no_real_environment_route(tmp_path, available):
    program = tmp_path/'p.py'
    program.write_text('''import json,sys
for line in sys.stdin:
    print('__COIMAGINE_RPC__'+json.dumps({'method':'predict','operation':'status'}),flush=True)
    reply=json.loads(sys.stdin.readline())
    print(json.dumps({'available':reply['ok']}),flush=True)
''')
    service = PredictionService()
    policy = JsonlPolicy([sys.executable, str(program)], tmp_path/'stderr',
                          rpc_handler=service.rpc if available else None)
    try:
        for _ in range(2):
            assert policy.request({'observation': {}}, 2) == {'available': available}
        with pytest.raises(ValueError):
            rpc(service, 'reset', task='planar_open_anchor')
    finally:
        policy.close()


def test_predictor_inside_real_sandbox_episode(tmp_path):
    import json
    import shutil
    from experiments.planar_tasks.baseline.session import package_command
    from experiments.planar_tasks.runner import run_episode, RunBudget
    task = task_catalog()['planar_open_anchor']
    package = tmp_path/'package'; package.mkdir()
    base = Path('experiments/planar_tasks/baseline')
    shutil.copyfile(base/'policy_host.py', package/'__policy_host__.py')
    shutil.copyfile(base/'predictor_client.py', package/'predictor_client.py')
    # Infrastructure fixture contains a synthetic hypothesis, not a research policy.
    (package/'policy.py').write_text('from predictor_client import Predictor\n'
        'class Policy:\n'
        ' def __init__(self,contract): self.p=Predictor()\n'
        ' def act(self,observation):\n'
        f'  r=self.p.create({hypothesis(task.instance)!r})\n'
        '  self.p.step(r["handle"],dict(a0=44,a1=44,ticks=2))\n'
        '  self.p.close(r["handle"])\n'
        '  return dict(a0=44,a1=44,ticks=2)\n')
    result = run_episode(task, package_command(package), tmp_path/'episode',
                         RunBudget(10, 10, 3), max_control_ticks=2,
                         prediction_profile=dict(version='nominal-v1', runtime=True))
    assert result['error'] is None, result
    assert result['executed_control_ticks'] == 2
    assert result['prediction_usage']['ticks'] == 2
    records = [json.loads(line) for line in (tmp_path/'episode/prediction.jsonl').read_text().splitlines()]
    assert len(records) == 3 and all(r['error'] is None for r in records)
