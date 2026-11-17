"""Offline known-initial-state calibration, never exposed to policy authors."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from experiments.planar_tasks.baseline.prediction import PredictionService
from experiments.planar_tasks.interface import TaskEnvironment
from experiments.planar_tasks.tasks import task_catalog


def hypothesis(instance):
    return dict(arena=list(instance.arena), goal=list(instance.goal_centre), goal_radius=instance.goal_radius,
        object=[instance.object_pose.x, instance.object_pose.y, instance.object_pose.angle, 0, 0, 0],
        robots=[[r.x, r.y, r.angle, 0, 0, 0] for r in instance.robot_poses],
        obstacles=[[b.cx, b.cy, b.half_w, b.half_h, b.angle] for b in instance.obstacles])


def main():
    out = Path('outputs/agentic/prediction_calibration_20261110_01')
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for name in ('planar_open_anchor', 'planar_wall_anchor', 'planar_detour_anchor'):
        instance = task_catalog()[name].instance
        real = TaskEnvironment(instance)
        service = PredictionService(log_path=out/(name+'.jsonl'))
        def rpc(op, **fields):
            service.begin_decision()
            return service.rpc(dict(method='predict', operation=op, **fields), time.monotonic()+5)
        first = rpc('create', hypothesis=hypothesis(instance)); handle = first['handle']
        rows = []
        for pair in ([2, 2],)*4 + ([36, 36], [44, 44]):
            if real._env.termination is not None:
                break
            obs = real.observe()
            if pair not in obs['legal_pairs']:
                pair = [44, 44]
            action = dict(a0=pair[0], a1=pair[1], ticks=40)
            real.step(action)
            pred = rpc('step', handle=handle, action=action)
            state = dict(object=asdict(real._env.object_state()), robots=[asdict(r) for r in real._env.robot_states()])
            rows.append(dict(action=action, ticks=real._env.total_periods, state_equal=pred['state']==state,
                observations_equal=pred['observation']==real.observe(), termination=real._env.termination))
        before = service.result(handle)
        fork = rpc('fork', handle=handle)
        results.append(dict(task=name, comparisons=rows, execution_ticks=real._env.total_periods,
            fork_state_equal=fork['state']==before['state'], fork_observation_equal=fork['observation']==before['observation'],
            prediction_usage=service.ledger.usage))
    passed = all(r['fork_state_equal'] and r['fork_observation_equal'] and
        all(c['state_equal'] and c['observations_equal'] for c in r['comparisons']) for r in results)
    summary = dict(passed=passed, scope='offline exact initial hypothesis; not online estimation quality',
                   execution_tick_cap=720, prediction_tick_cap=1440, results=results)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
