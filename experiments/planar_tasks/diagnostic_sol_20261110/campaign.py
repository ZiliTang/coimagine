"""Preregistered prompt comparison using unchanged v1.3.5 execution machinery.

The only runtime override is driver.BASE for its system-prompt file. This is
explicitly hashed in preflight, logged in every model request, and mirrored into
the author workspace. No simulator, option, observation or score is patched.
"""
import argparse
import json
import os
from pathlib import Path
import time

from experiments.planar_tasks.baseline import driver, seal
from experiments.planar_tasks.baseline.config import validate_config
from experiments.planar_tasks.baseline.evaluation import EvaluationBudget, evaluate
from experiments.planar_tasks.baseline.panels import create_panel
from experiments.planar_tasks.baseline.session import Session

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOTS = {name: REPO / 'outputs/agentic' / f'diagnostic_sol_{name}_20261110_01'
         for name in ('control', 'modular')}
PANEL = REPO / 'outputs/agentic/panels/diagnostic_sol_20261110_01.json'


def save(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def prepare():
    seal.verify_release()
    if PANEL.exists() or any(root.exists() for root in ROOTS.values()):
        raise FileExistsError('comparison already prepared; no overwrite')
    create_panel(PANEL, seed=2026111001, per_family=10)
    config = json.loads((seal.BASE / 'development_config.json').read_text())
    config['model'].update(provider='codex_external_relay', name='gpt-6.1-sol',
                           settings={'reasoning_effort': 'high'})
    config['budget']['model_call_seconds'] = 300
    validate_config(config)
    evaluation = json.loads((seal.BASE / 'evaluation_config.json').read_text())
    base = (seal.BASE / 'prompt.txt').read_text()
    common = (HERE / 'common.txt').read_text()
    for condition, root in ROOTS.items():
        root.mkdir(parents=True); (root / 'relay').mkdir(); (root / 'author_prompt').mkdir()
        save(root / 'development_config.json', config)
        save(root / 'evaluation_config.json', evaluation)
        (root / 'author_prompt/prompt.txt').write_text(base + '\n\n' + common + '\n\n' +
                                                    (HERE / f'{condition}.txt').read_text())
        files = {str(p): seal.digest(p) for p in HERE.iterdir() if p.is_file()}
        files.update({str(root / p): seal.digest(root / p) for p in
                      ('development_config.json', 'evaluation_config.json', 'author_prompt/prompt.txt')})
        save(root / 'preflight.json', {'condition': condition, 'prepared_unix': time.time(),
            'release_sha256': seal.digest(seal.LOCK), 'files_sha256': files,
            'panel_path': str(PANEL), 'panel_sha256': seal.digest(PANEL),
            'model': 'gpt-6.1-sol', 'requested_reasoning_effort': 'high',
            'fork_turns': 'none', 'replicates_per_condition': 1,
            'native_tokens_and_cost': 'unavailable; relay calls are not native model calls',
            'author_access': 'relay-only instruction, not OS isolation of author',
            'prompt_override': 'driver.BASE only; mirrored instructions.txt, exact prompt in model events',
            'scope': 'same-family transfer, fixed shape/dynamics; not evidence of unknown dynamics',
            'primary_comparison': 'unrestricted CAP versus researcher-prescribed task-effect modular CAP',
            'shared_final_development_regression': '9 fixtures, 1800 ticks each, inside development total',
            'solvability': 'success traces are witnesses only for their instances; other cases unverified'})
    print(json.dumps({'roots': {k: str(v) for k, v in ROOTS.items()}, 'panel': str(PANEL)}))


class Relay:
    def __init__(self, root):
        self.root = root / 'relay'

    def __call__(self, payload, timeout):
        index = len(list(self.root.glob('request_*.json'))) + 1
        save(self.root / f'request_{index:03d}.json', payload)
        response = self.root / f'response_{index:03d}.json'
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if response.exists():
                return json.loads(response.read_text())
            time.sleep(.1)
        raise TimeoutError('external author relay deadline; no retry')


def run(condition):
    root = ROOTS[condition]
    if (root / 'execution.json').exists():
        raise FileExistsError('campaign already attempted; no resume')
    expected = json.loads((root / 'preflight.json').read_text())
    if seal.digest(seal.LOCK) != expected['release_sha256']:
        raise ValueError('baseline release changed')
    for name, digest in expected['files_sha256'].items():
        if seal.digest(name) != digest:
            raise ValueError('preregistered file changed: ' + name)
    if seal.digest(PANEL) != expected['panel_sha256']:
        raise ValueError('panel changed')
    record = {'status': 'running', 'phase': 'synthesis', 'pid': os.getpid(),
              'condition': condition, 'started_unix': time.time()}
    save(root / 'execution.json', record)
    try:
        session = Session.create(root / 'session', json.loads((root / 'development_config.json').read_text()))
        try:
            driver.BASE = root / 'author_prompt'
            (session.workspace / 'instructions.txt').write_text((driver.BASE / 'prompt.txt').read_text())
            session.event('diagnostic_prompt', {'condition': condition}, {
                'prompt_sha256': seal.digest(driver.BASE / 'prompt.txt'),
                'preflight_sha256': seal.digest(root / 'preflight.json')})
            result = driver.drive(session, Relay(root))
        finally:
            session.close()
        save(root / 'synthesis_summary.json', result)
        if result['status'] == 'submitted':
            record.update(phase='independent_evaluation', synthesis_finished_unix=time.time())
            save(root / 'execution.json', record)
            result = evaluate(root / 'session', PANEL, root / 'evaluation',
                              EvaluationBudget(**json.loads((root / 'evaluation_config.json').read_text())))
            record.update(status=result['status'], phase='finished',
                          scheduled_episodes=result['scheduled_episodes'],
                          attempted_episodes=result['attempted_episodes'],
                          successful_episodes=result['successful_episodes'])
        else:
            record.update(status='incomplete', phase='finished', stop_reason='no_explicit_submission',
                          synthesis_stop_reason=result['stop_reason'], evaluation='not_run')
    except Exception as exc:
        record.update(status='incomplete', phase='finished',
                      error={'type': type(exc).__name__, 'message': str(exc)})
    record['finished_unix'] = time.time(); save(root / 'execution.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'control', 'modular'))
    command = parser.parse_args().command
    prepare() if command == 'prepare' else run(command)
