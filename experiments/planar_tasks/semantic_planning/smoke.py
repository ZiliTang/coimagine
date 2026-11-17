"""Explicit short physical integration check; never calls a model provider."""
import argparse
import json
from pathlib import Path
import time

from experiments.planar_tasks.authoring_v2 import campaign_config
from .artifact import create_revision, verify_revision, sha
from .session import create_session, verify_frozen


def run(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    revision = destination / 'revision'
    create_revision(Path(__file__).with_name('reference.py'), revision)
    verify_revision(revision)
    tasks = ['planar_open_anchor', 'planar_wall_anchor', 'planar_detour_anchor']
    config = campaign_config(dict(provider='synthetic', name='no-model-invoked', settings={},
                                  source_access='hidden', accounting='reserved_output_tokens_and_input_bytes'), tasks)
    config.pop('finalization')
    config['budget'].update(wall_seconds=180, episode_wall_seconds=30, response_seconds=5,
                            policy_tests=3, control_ticks=240, resets=3)
    session = create_session(destination / 'session', config)
    report = dict(purpose='short physical integration, not a success-rate evaluation',
                  status='running', model_calls=0, tasks=[], declared_execution_tick_cap=240,
                  declared_session_wall_seconds=180)
    def save():
        (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        response = session.tool('write_file', dict(path='heuristic.py', text=(revision / 'heuristic.py').read_text()))
        if not response['ok']:
            raise RuntimeError(response)
        for task in tasks:
            response = session.tool('test_policy', dict(files=['heuristic.py'], task=task, max_ticks=80))
            row = dict(task=task, response=response)
            report['tasks'].append(row)
            save()
            if not response['ok']:
                raise RuntimeError(response)
            result = response['result']
            directory = session.path / 'private' / result['test_id'] / 'episode'
            traces = []
            for line in (directory / 'policy.stderr').read_text().splitlines():
                try:
                    traces.append(json.loads(line))
                except ValueError:
                    pass
            semantic = [t for t in traces if t.get('kind') == 'history_semantic_v0']
            search = [t for t in traces if t.get('kind') == 'option_search_v3']
            roots = [t for t in semantic if t['stage'] == 'value' and t['source'] == 'real_observation_estimate']
            row.update(semantic_records=len(semantic), search_records=len(search),
                       root_real_sample_counts=[t['features']['real_samples'] for t in roots],
                       root_synthetic_sample_counts=[t['features']['synthetic_samples'] for t in roots],
                       hook_errors=[e for t in search for e in t['hook_errors']],
                       model_errors=[e for t in search for e in t['events'] if e['status'] == 'model_error'])
            save()
            if result['summary'].get('error') or row['hook_errors'] or row['model_errors']:
                raise RuntimeError('execution/hook/predictor integration failed; see report')
            if not roots or not search or any(t['features']['synthetic_samples'] for t in roots):
                raise RuntimeError('missing causal logs or synthetic memory contamination')
        response = session.tool('submit', dict(files=['heuristic.py']))
        report['submission'] = response
        if not response['ok']:
            raise RuntimeError(response)
        verify_frozen(session.path)
        if (session.path / 'frozen/package/heuristic.py').read_bytes() != (revision / 'heuristic.py').read_bytes():
            raise RuntimeError('submitted knowledge identity mismatch')
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report['usage'] = dict(session.state['usage'])
        report['prediction_usage'] = dict(session.state.get('prediction_usage', {}))
        report['model_calls'] = session.state['usage'].get('model_calls', 0)
        session.close()
        report['wall_seconds'] = time.monotonic() - started
        save()
        files = {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*') if p.is_file()}
        (destination / 'files_sha256.json').write_text(json.dumps(files, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    report = run(args.destination)
    print(json.dumps({k: report[k] for k in ('status', 'wall_seconds', 'model_calls', 'usage', 'prediction_usage')}, indent=2))
