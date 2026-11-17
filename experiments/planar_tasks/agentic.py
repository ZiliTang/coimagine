"""Bounded generate/run/repair/freeze pilot; all fixtures remain DEVELOPMENT.

External chat API and in-session Codex authors share packet, trials and freeze
selection, but synthesis access/token budgets differ: this is not a model ranking.
"""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .interface import TaskEnvironment, contract
from .runner import RunBudget, run_episode, write_json
from .sandbox import SANDBOX, policy_command
from .tasks import task_catalog

DEV_TASKS = ('planar_open_anchor', 'planar_wall_anchor')
PILOT_VERSION = 'agentic-option-pilot-v2-wire-examples'
PROMPT = '''Implement one persistent Python 3 JSONL policy for two cooperative pushing robots.
This is bounded program synthesis: generate code, inspect development feedback, revise;
then freeze a single general program. No LLM calls occur during deployment.
Each stdin line has observation; the first also has reset=true and contract.
The exact wire envelope is {"observation": {"robots": [...], "legal_pairs": [...]},
"reset": true, "contract": {...}}. Read request["observation"]["robots"],
NOT request["robots"]. Subsequent envelopes still have the observation wrapper.
JSON legal_pairs are lists of lists; convert them to a set of tuples before
checking tuple membership, or compare lists with lists.
Emit one JSON line {"a0": int, "a1": int, "ticks": int} and flush stdout.
Use only the Python standard library. Program memory persists within an episode.
Choose ONLY observation.legal_pairs; ticks 1..40, dt .05 seconds, horizon 90 seconds.
Contract gives option semantics. Same action continues the controller, including
failure latches. A different option requests a brake/commit transition. Allow time
for acquisition and transitions; do not equate a legal action with a feasible one.
BRAKE actively resists motion; YIELD may vacate a blocked partner's workspace.
Scan angles are world angles; origin is world xy; category 1=object, 2=static,
3=teammate, 0=none. Patch points_world and inward normal_world are sensor-derived.
No object pose, full map, mass, friction, task name or seed is given to deployment.
Estimate what you need from sensors/history. Do not hardcode scene coordinates,
identify fixture IDs, read files, import simulator code or call a network service.
Deployment is isolated from repository, credentials and network in bubblewrap.
A single object is transported to the shared goal. Other obstacles are fixed.
Development covers open transport and near-wall release; rotation/reposition may
help. Frozen transfer also includes detours, mirrors and shifts; all are development
fixtures, NOT an independent benchmark. Success and terminal distance may be given
as development scoring feedback, never in the deployed observation.
Maximum three candidate programs, each tried once on both development anchors.
The packet's candidate_budget overrides this maximum when smaller.
Every failed/invalid candidate consumes an attempt. Select by total successes,
then lower sum task cost, then lower sum terminal goal distance, then earlier index.
Keep all attempts. Do not alter physics, options, masks, cost or success rules.
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def locked(session):
    with (Path(session)/'.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def initialize(path, model, access, maximum_candidates=3):
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 3:
        raise ValueError('maximum_candidates must be an integer in 1..3')
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    config = dict(version=PILOT_VERSION, model=model, synthesis_access=access,
        maximum_candidates=maximum_candidates, development_tasks=list(DEV_TASKS),
        transfer_tasks=[n for n in task_catalog() if n not in DEV_TASKS],
        run_budget=dict(max_decisions=1800, wall_seconds=120., response_seconds=3.),
        selection='max(successes), min(sum cost), min(sum terminal distance), earliest',
        split='development only; transfer not independently held-out',
        started_unix=time.time(), deployment_sandbox=SANDBOX)
    write_json(path/'session.json', config)
    # These are legal sensor examples, not state dumps or privileged geometry.
    examples = {name:TaskEnvironment(task_catalog()[name].instance).observe() for name in DEV_TASKS}
    from coimagine.random_start.env import EnvConfig
    env_config = EnvConfig()
    packet = dict(instructions=PROMPT, contract=contract(), initial_observation_examples=examples,
        candidate_budget=maximum_candidates,
        wire_example=dict(reset=True,contract=contract(),observation=examples[DEV_TASKS[0]]),
        public_success_rule={key:getattr(env_config,key) for key in
            ('success_distance','success_linear','success_angular','success_hold')})
    write_json(path/'packet.json', packet)
    return config


def sensor_feedback(directory):
    """Compact chronological diagnostics derived only from transmitted sensors."""
    samples = []
    last_time = -10.
    for line in (directory/'sensors.jsonl').read_text().splitlines():
        item = json.loads(line)
        if 'request' not in item:
            continue
        obs = item['request']['observation']
        t = obs['robots'][0]['time']
        if t-last_time < 5.:
            continue
        last_time = t
        samples.append(dict(time=t, robots=[dict(own=r['own'],
            contacting_object=r['contacting_object'], execution=r['execution'],
            visible_object_bounds=visible_bounds(r)) for r in obs['robots']]))
    decisions = json.loads((directory/'decisions.json').read_text())
    segments = []
    for d in decisions:
        if segments and segments[-1]['actions'] == d['actions']:
            segments[-1]['end_tick'] = d['end_tick']
        else:
            segments.append(dict(actions=d['actions'],start_tick=d['start_tick'],end_tick=d['end_tick']))
    return dict(sensor_samples=samples, action_segments=segments[:60],
                action_segment_count=len(segments))


def visible_bounds(robot):
    points = [p for patch in robot['patches'] for p in patch['points_world'] if patch['valid']]
    return None if not points else [min(p[0] for p in points),min(p[1] for p in points),
                                    max(p[0] for p in points),max(p[1] for p in points)]


def trial(session, code, note=''):
    session = Path(session)
    with locked(session):
        config = json.loads((session/'session.json').read_text())
        if (session/'frozen.json').exists():
            raise ValueError('session is frozen; no further development')
        count = len(list(session.glob('candidate_*')))
        if count >= config['maximum_candidates']:
            raise ValueError('candidate budget exhausted')
        directory = session/f'candidate_{count:02d}'
        directory.mkdir()
        program = directory/'policy.py'
        program.write_text(code)
        provenance = dict(generated_agent_baseline=True, model=config['model'],
            synthesis_access=config['synthesis_access'], candidate=count,
            policy_sha256=digest(program), method='bounded coding-agent option adaptation')
        write_json(directory/'submission.json', dict(note=note, provenance=provenance,
                   submitted_unix=time.time()))
        result = dict(candidate=count, policy_sha256=digest(program), episodes=[])
        try:
            ast.parse(code)
        except SyntaxError as exc:
            result['compile_error'] = str(exc)
            result['selection_key'] = [0, 360., 1000000., count]
        else:
            for task in config['development_tasks']:
                episode_dir = directory/task
                try:
                    summary = run_episode(task_catalog()[task], policy_command(program), episode_dir,
                        RunBudget(**config['run_budget']), provenance=provenance, security_sandbox=SANDBOX)
                except Exception as exc:
                    # Preserve the attempted denominator even if setup/snapshotting
                    # fails before the runner can produce its own summary.
                    summary = dict(task=task, success=False, status='incomplete',
                        termination='infrastructure_error', attempted_episodes=1,
                        successful_episodes=0, task_cost_seconds=180.,
                        final_goal_distance_diagnostic=None,
                        error=dict(type=type(exc).__name__, message=str(exc)))
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    write_json(episode_dir/'infrastructure_error.json',summary)
                feedback = {}
                if (episode_dir/'sensors.jsonl').exists():
                    try:
                        feedback = sensor_feedback(episode_dir)
                    except Exception as exc:
                        feedback = dict(feedback_error=dict(type=type(exc).__name__,message=str(exc)))
                result['episodes'].append(dict(summary=summary, **feedback))
                write_json(directory/'feedback.json',result)
            ss = [e['summary'] for e in result['episodes']]
            result['selection_key'] = [-sum(s['success'] for s in ss),
                sum(s['task_cost_seconds'] for s in ss),
                sum(s['final_goal_distance_diagnostic'] if s['final_goal_distance_diagnostic'] is not None
                    else 1000000. for s in ss), count]
        write_json(directory/'feedback.json',result)
        return result


def freeze(session):
    session = Path(session)
    with locked(session):
        if (session/'frozen.json').exists():
            return json.loads((session/'frozen.json').read_text())
        candidates = [json.loads(p.read_text()) for p in session.glob('candidate_*/feedback.json')]
        candidates = [c for c in candidates if 'compile_error' not in c]
        if not candidates:
            raise ValueError('no compilable candidate to freeze')
        best = min(candidates, key=lambda c:c['selection_key'])
        source = session/f'candidate_{best["candidate"]:02d}'/'policy.py'
        if digest(source) != best['policy_sha256']:
            raise ValueError('candidate changed after trial')
        target = session/'frozen_policy.py'
        target.write_bytes(source.read_bytes())
        frozen = dict(candidate=best['candidate'],policy_sha256=digest(target),
                      selected_unix=time.time(),selection_key=best['selection_key'])
        write_json(session/'frozen.json', frozen)
        return frozen


def transfer(session):
    session = Path(session)
    with locked(session):
        frozen = json.loads((session/'frozen.json').read_text())
        config = json.loads((session/'session.json').read_text())
        policy = session/'frozen_policy.py'
        if digest(policy) != frozen['policy_sha256']:
            raise ValueError('frozen policy hash mismatch')
        directory = session/'transfer'
        directory.mkdir(exist_ok=False)
        summaries=[]
        for task in config['transfer_tasks']:
            summaries.append(run_episode(task_catalog()[task], policy_command(policy), directory/task,
                RunBudget(**config['run_budget']), provenance=dict(generated_agent_baseline=True,
                model=config['model'],policy_sha256=frozen['policy_sha256'], phase='frozen development transfer'),
                security_sandbox=SANDBOX))
        result=dict(split=config['split'],episodes=summaries,
                    successes=sum(s['success'] for s in summaries),attempts=len(summaries))
        write_json(directory/'summary.json',result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='op',required=True)
    init=sub.add_parser('init');init.add_argument('--model',required=True);init.add_argument('--access',required=True)
    init.add_argument('--maximum-candidates',type=int,default=3)
    tr=sub.add_parser('trial');tr.add_argument('--policy',required=True);tr.add_argument('--note',default='')
    sub.add_parser('freeze');sub.add_parser('transfer')
    for s in sub.choices.values():s.add_argument('--session',required=True)
    a=p.parse_args()
    if a.op=='init': result=initialize(a.session,a.model,a.access,a.maximum_candidates)
    elif a.op=='trial':result=trial(a.session,Path(a.policy).read_text(),a.note)
    elif a.op=='freeze':result=freeze(a.session)
    else:result=transfer(a.session)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
