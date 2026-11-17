"""Explicit, bounded projections of experiment artifacts for human inspection."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import time


def redact(value):
    if isinstance(value, dict):
        return {k: ('[redacted]' if any(s in k.lower() for s in
                ('api_key', 'authorization', 'password', 'secret', 'reasoning_content')) else redact(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'\bsk-[A-Za-z0-9_-]{12,}', '[redacted]', value)
    return value


def inside(root, path):
    """Reject every symlink component, even if it would point inside the root."""
    root = Path(root).resolve()
    path = Path(path)
    relative = path.relative_to(root)
    if '..' in relative.parts:
        raise ValueError('path traversal')
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('symlink artifact unavailable')
    if not path.resolve().is_relative_to(root):
        raise ValueError('outside artifact root')
    return path


def read_json(root, path, default=None):
    try:
        path = inside(root, path)
        if path.stat().st_size > 8 * 1024 * 1024:
            return default
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return default


def identifier(kind, relative):
    return kind + '_' + hashlib.sha256(str(relative).encode()).hexdigest()[:20]


SUMMARY_FIELDS = ('task', 'status', 'success', 'termination', 'executed_control_ticks',
                  'simulated_seconds', 'wall_seconds', 'task_cost_seconds', 'error')
CUMULATIVE_FIELDS = ('model_calls', 'model_input_bytes', 'model_output_tokens', 'tool_calls',
                     'python_total_seconds', 'resets', 'control_ticks', 'policy_tests')


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.runs = {}
        self.replays = {}
        self.evaluations = {}

    def scan(self):
        runs = {}
        evaluations = {}
        for directory, children, files in os.walk(self.root, followlinks=False):
            path = Path(directory)
            children[:] = sorted(n for n in children if n not in
                ('source', 'source_snapshot', 'frozen_code', '__pycache__', '.git')
                and not (path / n).is_symlink())
            if 'session.json' in files:
                config = read_json(self.root, path / 'session.json', {})
                if config and ('model' in config):
                    run_id = identifier('run', path.relative_to(self.root))
                    runs[run_id] = path
                    children[:] = []
            elif 'manifest.json' in files and read_json(self.root, path / 'manifest.json', {}).get('kind') == 'frozen_evaluation':
                evaluations[identifier('eval', path.relative_to(self.root))] = path
                children[:] = []
        self.runs, self.evaluations = runs, evaluations
        return self

    def list_evaluations(self, run_id=None):
        items = []
        for eval_id, path in self.evaluations.items():
            manifest = read_json(self.root, path / 'manifest.json', {})
            linked = str(self.run(run_id)) if run_id else None
            if linked and manifest.get('session_path') != linked:
                continue
            summary = read_json(self.root, path / 'summary.json', {})
            results = []
            # In-progress test outcomes stay hidden until the batch summary exists.
            for row in summary.get('results', []):
                result = {k: row.get(k) for k in ('index', 'task', 'family', 'status', 'success', 'termination', 'attempted')}
                episode = row.get('episode_path')
                result['replay'] = self.register_replay(path / episode / 'replay.json') if episode else None
                results.append(result)
            items.append({'id': eval_id, 'path': str(path.relative_to(self.root)),
                'status': summary.get('status', 'in_progress'),
                'panel_sha256': manifest.get('panel_sha256'),
                'scheduled_episodes': manifest.get('scheduled_episodes'),
                'attempted_episodes': summary.get('attempted_episodes'),
                'successful_episodes': summary.get('successful_episodes'),
                'success_rate': summary.get('success_rate'),
                'results': results, 'budget': manifest.get('budget'),
                'submission_manifest_sha256': manifest.get('submission_manifest_sha256')})
        return redact(items)

    def run(self, run_id):
        if run_id not in self.runs:
            raise KeyError('unknown run')
        return inside(self.root, self.runs[run_id])

    def overview(self, run_id):
        path = self.run(run_id)
        config = read_json(self.root, path / 'session.json', {})
        state = read_json(self.root, path / 'state.json', {})
        legacy = not isinstance(config.get('model'), dict)
        frozen = read_json(self.root, path / ('frozen.json' if legacy else 'frozen/manifest.json'), {})
        model = config.get('model')
        if legacy:
            model = {'name': model, 'provider': 'deepseek' if str(model).startswith('deepseek') else 'Codex session',
                     'settings': {}, 'source_access': config.get('synthesis_access', 'not recorded')}
        return redact({'id': run_id, 'path': str(path.relative_to(self.root)),
            'kind': 'legacy_fixed_feedback' if legacy else 'autonomous', 'model': model,
            'version': state.get('version', config.get('version')),
            'purpose': config.get('purpose', 'legacy_development'),
            'status': state.get('status', 'frozen' if frozen else 'unknown'),
            'stage': 'finalizing' if state.get('finalizing') and state.get('status') == 'developing' else state.get('status'),
            'stop_reason': state.get('stop_reason'), 'frozen': bool(frozen),
            'created_unix': state.get('created_unix', config.get('started_unix')),
            'release_sha256': state.get('release_sha256'),
            'config_sha256': state.get('config_sha256')})

    def list_runs(self):
        return sorted((self.overview(k) for k in self.runs),
                      key=lambda r: r.get('created_unix') or 0, reverse=True)

    def register_replay(self, path):
        path = inside(self.root, path)
        if not path.is_file():
            return None
        replay_id = identifier('replay', path.relative_to(self.root))
        self.replays[replay_id] = path
        return '/replay?run=' + replay_id

    def episodes(self, run_id):
        path = self.run(run_id)
        legacy = self.overview(run_id)['kind'] == 'legacy_fixed_feedback'
        patterns = ('candidate_*/*/summary.json', 'transfer/*/summary.json') if legacy else ('private/test_*/episode/summary.json',)
        episodes = []
        for pattern in patterns:
            for summary_path in sorted(path.glob(pattern)):
                summary = read_json(self.root, summary_path, {})
                if not summary:
                    continue
                item = {k: summary.get(k) for k in SUMMARY_FIELDS}
                relative = str(summary_path.parent.relative_to(path))
                item.update(id=relative, group=relative.split('/')[0] if legacy else relative.split('/')[1],
                            split='development', replay=self.register_replay(summary_path.parent / 'replay.json'))
                episodes.append(item)
        if not legacy:
            recorded = {item['group'] for item in episodes}
            state = read_json(self.root, path / 'state.json', {})
            for test in state.get('tests', []):
                if test.get('id') in recorded:
                    continue
                episodes.append({'id': test.get('id'), 'group': test.get('id'), 'task': test.get('task'),
                    'status': test.get('status'), 'success': test.get('success'), 'error': test.get('error'),
                    'termination': test.get('error', {}).get('type') if isinstance(test.get('error'), dict) else None,
                    'split': 'development', 'replay': None})
        return redact(episodes)

    def versions(self, run_id):
        path = self.run(run_id)
        legacy = self.overview(run_id)['kind'] == 'legacy_fixed_feedback'
        versions = []
        framework = read_json(self.root, path / 'framework.json', {})
        entry = 'heuristic.py' if str(framework.get('kind', '')).startswith('option-search-') else 'policy.py'
        patterns = ('candidate_*/policy.py', 'frozen_policy.py') if legacy else (
            f'private/python_*/workspace/{entry}', f'private/test_*/package/{entry}',
            f'private/submission_*/package/{entry}', f'frozen/package/{entry}')
        for pattern in patterns:
            for policy in sorted(path.glob(pattern)):
                try:
                    inside(self.root, policy)
                    data = policy.read_bytes()
                except (ValueError, OSError):
                    continue
                relative = str(policy.relative_to(path))
                label = relative.split('/')[0] if legacy else (relative.split('/')[1] if relative.startswith('private/') else 'frozen')
                versions.append({'id': identifier('version', relative), 'label': label,
                    'path': relative, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                    'selected': relative.startswith('frozen'),
                    'kind': 'submission' if relative.startswith('frozen') else
                            ('submission_attempt' if relative.startswith('private/submission_') else
                             'python_snapshot' if '/workspace/' in relative else 'tested_package')})
        return versions

    def policy(self, run_id, version_id):
        versions = self.versions(run_id)
        version = next((v for v in versions if v['id'] == version_id), None)
        if version is None:
            raise KeyError('unknown policy version')
        path = inside(self.root, self.run(run_id) / version['path'])
        with path.open('rb') as stream:
            data = stream.read(256 * 1024 + 1)
        return dict(version, text=redact(data[:256 * 1024].decode(errors='replace')),
                    truncated=len(data) > 256 * 1024)

    def diff(self, run_id, before, after):
        left, right = self.policy(run_id, before), self.policy(run_id, after)
        return {'text': ''.join(difflib.unified_diff(left['text'].splitlines(True), right['text'].splitlines(True),
                    fromfile=left['path'], tofile=right['path'])),
                'truncated': left['truncated'] or right['truncated']}

    def events(self, run_id, start=0, count=100):
        from itertools import islice
        path = self.run(run_id) / 'private/events.jsonl'
        if not path.is_file():
            return {'items': [], 'next': None, 'note': 'Legacy pilot has no autonomous event stream.'}
        inside(self.root, path)
        rows = []
        with path.open() as stream:
            for line in islice(stream, start, start + count + 1):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a concurrent writer may not yet have finished its last line
                request, result = row.get('request') or {}, row.get('result') or {}
                item = {k: row.get(k) for k in ('index', 'kind', 'time_unix')}
                item['ok'] = result.get('ok')
                item['error'] = result.get('error')
                if row.get('kind') == 'model':
                    response = result.get('response') or {}
                    message = ((response.get('choices') or [{}])[0].get('message') or {})
                    item.update(label=request.get('model'), wall_seconds=result.get('wall_seconds'),
                        usage=response.get('usage'), calls=[x.get('function', {}).get('name') for x in message.get('tool_calls', [])],
                        content=str(message.get('content') or '')[:4000])
                else:
                    arguments = request.get('arguments') or request
                    item.update(label=request.get('name', request.get('method')),
                        arguments={k: arguments[k] for k in ('path', 'task', 'seconds', 'max_ticks', 'files') if k in arguments})
                    tool_result = result.get('result') or {}
                    if isinstance(tool_result, dict):
                        item['summary'] = {k: tool_result[k] for k in ('run_id', 'test_id', 'status', 'summary') if k in tool_result}
                rows.append(redact(item))
        return {'items': rows[:count], 'next': start + count if len(rows) > count else None}

    def detail(self, run_id):
        path = self.run(run_id)
        overview = self.overview(run_id)
        config = read_json(self.root, path / 'session.json', {})
        state = read_json(self.root, path / 'state.json', {})
        usage = state.get('reported_model_usage', [])
        if overview['kind'] == 'legacy_fixed_feedback':
            usage = [x.get('usage') for x in read_json(self.root, path / 'api_status.json', [])]
        reported = {k: sum((u or {}).get(k, 0) or 0 for u in usage) for k in
                    ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        if not any(isinstance(u, dict) for u in usage):
            reported = None
        usage_source = 'session provider responses'
        audited = read_json(self.root, path / 'provider_usage_audit.json', {})
        if (state.get('status') != 'developing' and isinstance(audited.get('events'), int)
                and audited['events'] == state.get('events')
                and audited.get('last_event_sha256') == state.get('last_event_hash')
                and isinstance(audited.get('reported_usage'), dict)
                and all(type(audited['reported_usage'].get(k)) is int and audited['reported_usage'][k] >= 0
                        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))):
            reported = audited['reported_usage']
            usage_source = audited.get('basis', 'audited returned provider events')
        accounting = 'Reservations are hard limits; provider tokens are separately reported. Monetary cost is not inferred.'
        if overview['model'].get('provider') == 'codex_external_relay':
            accounting = ('External Codex author: model_calls counts relay exchanges. Output tokens are requested '
                          'reservations, not measured or enforced native Codex usage. Tokens, reasoning and cost '
                          'are unknown; experimental tool/physics limits are enforced. Not a token/cost-matched run.')
        used = dict(state.get('usage', {}))
        if overview['kind'] == 'autonomous':
            for key in CUMULATIVE_FIELDS:
                if key in config.get('budget', {}):
                    used.setdefault(key, 0)
        submission_remaining={}
        if 'finalization_start_usage' in state:
            for key,cap in config.get('finalization',{}).items():
                if key=='wall_seconds':
                    end=state.get('finished_monotonic')
                    if end is None and state.get('status')=='developing' and state.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip():end=time.monotonic()
                    if end is None:continue
                    spent=end-state['finalization_started_monotonic']
                    total_left=config['budget'][key]-(end-state['started_monotonic'])
                else:
                    spent=used.get(key,0)-state['finalization_start_usage'].get(key,0)
                    total_left=config['budget'][key]-used.get(key,0)
                submission_remaining[key]=max(0,min(total_left,cap-spent))
        return redact(dict(overview, config=config, budget={
            'limits': config.get('budget', read_json(self.root, path / 'api_budget.json', {})),
            'used': used, 'reported_model_usage': reported, 'reported_usage_source': usage_source,
            'finalization': config.get('finalization', {}),
            'submission_remaining': submission_remaining,
            'accounting': accounting,
            'observed_unix': time.time()},
            versions=self.versions(run_id), episodes=self.episodes(run_id),
            tests=state.get('tests', []), episodes_development=state.get('episodes', [])))
