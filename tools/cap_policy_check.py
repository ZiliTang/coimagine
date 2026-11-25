"""Audit Python CAP decisions on frozen public observations, without running physics."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from experiments.planar_tasks.authoring_v2.policy_checks import check_boundary
from experiments.planar_tasks.authoring_v2.policy_check_specs import check_declared_uses, validate_source_rules
from experiments.planar_tasks.relational_search.contract import canonical, digest, public_observation, validate_action
from experiments.planar_tasks.relational_search.motion import CODEBOOK_HASH, VERSION as ACTION_VERSION
from experiments.planar_tasks.sandbox import policy_command
from experiments.planar_tasks.semantic_rl import stream_features
from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory
from experiments.planar_tasks.semantic_rl.typed_compiler import compile_source, stream_config, validate_source


# Runs in the same fork-per-query sandbox as deployed typed-stream programs.
# Production and audit are invoked in different forks of the identical compiled module.
_AUDIT_WORKER = """
normal_request = stream_request
def stream_request(namespace, request):
    if request['op'] == 'advance':
        return normal_request(namespace, request)
    memory = request['memory']
    observation = request['observation']
    if request['op'] == 'choose':
        # Validate production context with exactly the deployment validator.
        # Memory was already updated at this boundary; keep it unchanged.
        committed = dict(namespace, update_memory=lambda prior, event: prior)
        context = normal_request(committed, dict(op='advance', memory=memory,
                                                 event=dict(observation=observation)))
        return context['proposals']
    if request['op'] != 'audit':
        raise ValueError('unsupported audit operation')
    menu = request['menu']
    packet = namespace['audit_policy'](copy.deepcopy(memory), copy.deepcopy(observation), copy.deepcopy(menu))
    if not isinstance(packet, dict) or 'reviews' in packet:
        raise ValueError('audit_policy returns decision metadata; host supplies reviews')
    rows = []
    for pair in menu:
        review = namespace['review_push'](copy.deepcopy(memory), copy.deepcopy(observation), list(pair))
        if not isinstance(review, dict):
            raise ValueError('review_push must return a dictionary')
        if 'pair' in review and review['pair'] != pair:
            raise ValueError('review_push returned a different pair')
        for group in ('peer_predictions', 'widths'):
            for fact in review.pop(group, []):
                if 'pair' in fact and fact['pair'] != pair:
                    raise ValueError('review_push evidence describes another pair')
                packet.setdefault(group, []).append(dict(fact, pair=list(pair)))
        rows.append(dict(review, pair=pair))
    packet['reviews'] = rows
    return packet

stream_serve("/work/semantics.so")
"""


class _AuditProgram(StreamProgram):
    def start(self):
        self.close()
        self.directory = tempfile.TemporaryDirectory(prefix='coimagine_policy_check_')
        worker = Path(self.directory.name) / 'worker.py'
        compiled, self.compilation = compile_source(self.source)
        runtime = Path(stream_features.__file__).parent
        worker.write_text((runtime / 'feature_worker.py').read_text() + '\n'
                          + (runtime / 'stream_worker.py').read_text() + '\n' + _AUDIT_WORKER)
        self.process = subprocess.Popen(
            policy_command(worker, artifacts=((compiled / 'semantics.so', 'semantics.so'),)),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, bufsize=0,
        )


def lint_source(source):
    """Check the opt-in source interface, declarations and direct-name review linkage."""
    validate_source(source)
    stream_config(source)
    tree = ast.parse(source)
    rules = validate_source_rules(tree)
    functions = {item.name: item for item in tree.body if isinstance(item, ast.FunctionDef)}
    for name, args in [('audit_policy', ['memory', 'observation', 'menu']),
                       ('review_push', ['memory', 'observation', 'pair'])]:
        node = functions.get(name)
        if node is None:
            raise ValueError(f'missing required hook {name}')
        if ([a.arg for a in node.args.args] != args or node.args.posonlyargs or node.args.vararg
                or node.args.kwonlyargs or node.args.kwarg or node.args.defaults):
            raise ValueError(f'expected {name}({", ".join(args)})')
        if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)):
            raise ValueError(f'unimplemented hook {name}')
    reachable = set()
    pending = ['prepare_context', 'update_memory']
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for child in ast.walk(functions[name]):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in functions:
                pending.append(child.func.id)
    if 'review_push' not in reachable:
        raise ValueError('review_push must be called from prepare_context or update_memory via named functions')
    return dict(status='ok', interface='cap-policy-check-v1',
                source_sha256=hashlib.sha256(source.encode()).hexdigest(), rules=rules)


def _observation(raw):
    obs = public_observation(raw)
    if canonical(obs) != canonical(raw):
        raise ValueError('observation contains fields outside the public protocol')
    if 'entity_legal_pairs' not in obs:
        raise ValueError('missing per-target public action menu')
    if obs.get('action_codebook_hash', CODEBOOK_HASH) != CODEBOOK_HASH or obs.get('action_version', ACTION_VERSION) != ACTION_VERSION:
        raise ValueError('incompatible public action codebook')
    return obs


def public_menu(observation):
    target = observation['goal']['target_id']
    seen = any(e['id'] == target and e['points'] and e['last_seen_tick'] == observation['tick']
               for e in observation['entities'])
    if not seen:
        return []
    return [p for p in observation['entity_legal_pairs'][target] if min(p) < 32]


def _selected(action, obs):
    validate_action(action, obs)
    if action['target_id'] == obs['goal']['target_id'] and min(action['a0'], action['a1']) < 32:
        return [action['a0'], action['a1']]
    return None


def _inputs(program, source_hash, boundary):
    if 'events' in boundary:
        if 'memory' in boundary or 'observation' in boundary:
            raise ValueError('choose events or a committed boundary')
        events = boundary['events']
    elif 'memory' in boundary:
        if boundary.get('source_sha256') != source_hash:
            raise ValueError('committed memory must carry the exact source_sha256')
        memory = boundary['memory']
        if not isinstance(memory, dict) or len(json.dumps(memory, allow_nan=False).encode()) > 8192:
            raise ValueError('invalid committed memory')
        return _observation(boundary['observation']), memory, 'source_bound_memory'
    else:
        events = [{'observation': boundary['observation'], 'action': None}]
    if not isinstance(events, list) or not events:
        raise ValueError('expected one or more public events from reset')
    history = StreamHistory()
    memory = None
    for event in events:
        if not isinstance(event, dict) or set(event) != {'observation', 'action'}:
            raise ValueError('each ordinary committed event requires observation and preceding action')
        obs = _observation(event['observation'])
        if memory is None and obs['tick'] != 0:
            raise ValueError('replay requires a tick-zero reset or source-bound committed memory')
        history.append(obs, event['action'])
        dt = None if history.previous_tick is None else obs['tick'] - history.previous_tick
        memory = program.request(dict(op='advance', memory=memory,
                                      event=dict(observation=obs, action=event['action'], executed_ticks=dt)))['memory']
    return obs, memory, 'public_events_from_reset'


def run_boundary(source, boundary):
    """Check one ordinary decision. Return findings or raise on an invalid/incomplete audit."""
    interface = lint_source(source)
    allowed = {'events', 'observation', 'memory', 'action', 'source_sha256', 'observation_sha256'}
    if not isinstance(boundary, dict) or set(boundary) - allowed:
        raise ValueError('unsupported boundary fields; supply public input only')
    source_hash = interface['source_sha256']
    if 'source_sha256' in boundary and boundary['source_sha256'] != source_hash:
        raise ValueError('frozen policy source_sha256 mismatch')
    program = _AuditProgram(source)
    try:
        obs, memory, origin = _inputs(program, source_hash, boundary)
        obs_hash = digest(obs)
        if 'observation_sha256' in boundary and boundary['observation_sha256'] != obs_hash:
            raise ValueError('public observation_sha256 mismatch')
        menu = public_menu(obs)
        proposals = program.request(dict(op='choose', memory=memory, observation=obs))
        if not isinstance(proposals, list) or not proposals:
            raise ValueError('production prepare_context returned no first command')
        action = proposals[0]
        selected = _selected(action, obs)
        if 'action' in boundary and action != boundary['action']:
            raise ValueError('production first command disagrees with recorded action')
        packet = program.request(dict(op='audit', memory=memory, observation=obs, menu=menu))
    finally:
        program.close()
    report = check_boundary(menu, packet, selected=selected)
    check_declared_uses(interface['rules'], packet)
    return dict(report, packet=packet, action=action, menu=menu, selected=selected,
                target_id=obs['goal']['target_id'], source_sha256=source_hash,
                observation_sha256=obs_hash, tick=obs['tick'], menu_scope='public_goal_translation_pairs',
                input_origin=origin, actual_command_binding='isolated_production_first_proposal')


def replay(sample):
    """Validate a stored diagnostic, retaining its trust level and optional action binding."""
    if 'observation' in sample:
        obs = _observation(sample['observation'])
        menu = public_menu(obs)
        if 'menu' in sample and canonical(menu) != canonical(sample['menu']):
            raise ValueError('recorded menu differs from public observation')
        if 'observation_sha256' in sample and digest(obs) != sample['observation_sha256']:
            raise ValueError('observation_sha256 mismatch')
        origin = 'public_observation'
    elif 'menu' in sample:
        menu = sample['menu']
        origin = 'recorded_menu'
    else:
        raise ValueError('replay needs a public observation or recorded menu')
    kwargs = {'selected': sample['selected']} if 'selected' in sample else {}
    if 'action' in sample:
        if origin == 'public_observation':
            selected = _selected(sample['action'], obs)
        else:
            if type(sample.get('target_id')) is not str:
                raise ValueError('recorded action needs the recorded target_id or public observation')
            action = sample['action']
            selected = [action['a0'], action['a1']] if (
                action['target_id'] == sample['target_id'] and min(action['a0'], action['a1']) < 32
            ) else None
        if kwargs and selected != kwargs['selected']:
            raise ValueError('recorded selected pair disagrees with action')
        kwargs['selected'] = selected
    report = check_boundary(menu, sample['packet'], **kwargs)
    return dict(report, input_origin=origin, actual_command_binding='record_only' if kwargs else 'unbound')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    lint = subs.add_parser('lint')
    lint.add_argument('--policy', required=True, type=Path)
    run = subs.add_parser('run')
    run.add_argument('--policy', required=True, type=Path)
    run.add_argument('--boundary', required=True, type=Path)
    cached = subs.add_parser('replay')
    cached.add_argument('--packet', required=True, type=Path)
    for sub in (run, cached, lint):
        sub.add_argument('--out', type=Path)
    submit = subs.add_parser('submit')
    submit.add_argument('--policy', required=True, type=Path)
    submit.add_argument('--boundary', action='append', required=True, type=Path)
    submit.add_argument('--out', type=Path, required=True)
    verify = subs.add_parser('verify')
    verify.add_argument('--submission', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'lint':
            result = lint_source(args.policy.read_bytes().decode('utf-8'))
        elif args.command == 'run':
            result = run_boundary(args.policy.read_bytes().decode('utf-8'), json.loads(args.boundary.read_bytes()))
        elif args.command == 'submit':
            from experiments.planar_tasks.authoring_v2.checked_submission import submit_checked
            result = submit_checked(args.policy, args.boundary, args.out)
        elif args.command == 'verify':
            from experiments.planar_tasks.authoring_v2.checked_submission import verify_checked
            result = verify_checked(args.submission)
        else:
            samples = json.loads(args.packet.read_text())
            result = [replay(s) for s in samples] if isinstance(samples, list) else replay(samples)
            if result == []:
                raise ValueError('empty replay list')
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.command in ('run', 'lint', 'replay') and args.out:
            args.out.write_text(output+'\n')
        else:
            print(output)
        reports = result if isinstance(result, list) else [result]
        return 1 if any(r.get('status') == 'violation' for r in reports) else 0
    except (OSError, KeyError, TypeError, ValueError, SyntaxError, RuntimeError, TimeoutError) as exc:
        print(f'Incomplete policy check: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
