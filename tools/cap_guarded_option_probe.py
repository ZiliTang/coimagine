"""Small, reproducible guarded-policy comparisons on independently supplied task models."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.planar_tasks.authoring_v2 import guarded_options
from experiments.planar_tasks.authoring_v2.guarded_options import compile_policy
from experiments.planar_tasks.relational_search.contract import public_observation, validate_action


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'experiments/planar_tasks/authoring_v2/guarded_push_example.json'
PUSH_PROFILE = EXAMPLE.with_name('guarded_push_profile.json')
ARCHIVED = ROOT / 'outputs/semantic_rl/cap_checked_pair_v1/authoring_verified/development_inputs'
BOUNDARY_NAMES = (
    'diag_dev_nat_001_first_noninitial_push.json',
    'diag_dev_nat_002_initial.json',
    'diag_dev_nat_003_first_noninitial_push.json',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def digest_spec(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def parse_json(data):
    def unique_fields(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key {key!r}')
            result[key] = value
        return result

    def reject_constant(name):
        raise ValueError(f'non-finite JSON number {name!r}')

    return json.loads(data, object_pairs_hook=unique_fields, parse_constant=reject_constant)


def check_source_files(program_path, profile_path):
    report = {'schema': 'guarded-option-check-v1',
              'checker_sha256': digest(Path(guarded_options.__file__).read_bytes())}
    try:
        specs = []
        for key, path in [('program', program_path), ('profile', profile_path)]:
            raw = path.read_bytes()
            report[key + '_sha256'] = digest(raw)
            specs.append(parse_json(raw))
        report.update(compile_policy(*specs).check())
    except (ValueError, TypeError, KeyError, OSError) as exc:
        report.update(status='invalid', error=str(exc))
        return report, 2
    return report, 0 if report['status'] == 'proved' else 1


def outcome(symbol, change, post):
    return {'observe': symbol, 'change': change, 'post': post}


def active(requires, outcomes, quality='model'):
    return {'requires': requires, 'quality': quality, 'outcomes': outcomes}


def host_profile(cell, options, metrics):
    return {'units': {m: 'mm' for m in metrics}, 'cells': {'start': cell}, 'options': options}


def push_profile():
    return parse_json(PUSH_PROFILE.read_bytes())


def yield_profile():
    return host_profile({'peer_gap_mm': [340, 345]}, {
        'push_with_yield': active({'peer_gap_mm': [300, None]}, [
            outcome('scan', {'advance_mm': [40, 80], 'peer_toward_mm': [-60, 60]},
                    {'peer_gap_mm': ['sub', 'peer_gap_mm', 'peer_toward_mm']})]),
        'cross_behind': {'requires': {'peer_gap_mm': [325, None]}},
    }, ['peer_gap_mm', 'advance_mm', 'peer_toward_mm'])


def scan_profile(hidden):
    options = {
        'scan_both_sides': active({}, [
            outcome('occluded' if hidden else 'left_open', {},
                    {'left_gap_mm': [200, 200], 'right_gap_mm': [0, 0]}),
            outcome('occluded' if hidden else 'right_open', {},
                    {'left_gap_mm': [0, 0], 'right_gap_mm': [200, 200]})]),
        'pass_left': {'requires': {'left_gap_mm': [150, None]}},
        'pass_right': {'requires': {'right_gap_mm': [150, None]}},
    }
    return host_profile({'left_gap_mm': [0, 200], 'right_gap_mm': [0, 200]},
                        options, ['left_gap_mm', 'right_gap_mm'])


def cabinet_profile():
    def grip(span):
        return active({'opening_mm': [80, None]}, [
            outcome('grasped', {'loaded_width_mm': span},
                    {'margin_mm': ['sub', 'opening_mm', 'loaded_width_mm']})])
    return host_profile({'opening_mm': [110, 110]}, {
        'grip_slim': grip([80, 85]),
        'grip_crosswise': grip([105, 125]),
        'withdraw': {'requires': {'margin_mm': [5, None]}},
    }, ['opening_mm', 'loaded_width_mm', 'margin_mm'])


def rule(name, when, option, expect, then):
    return {'rules': [dict(name=name, when=when, option=option, expect=expect, then=then)]}


def trials(example):
    push = push_profile()
    yield example, push, 'short_push', 'proved'
    long = deepcopy(example)
    long['rules'][0].update(option='push_long', expect={'advance_mm': [150, 200]})
    yield long, push, 'long_push', 'counterexample'

    for quality, expected in [('empirical', 'conditional'), ('unknown', 'unknown')]:
        unverified = deepcopy(push)
        unverified['options']['push_short']['quality'] = quality
        yield example, unverified, f'short_push_{quality}', expected

    # The declared state entirely misses the only productive guard.
    outside = deepcopy(push)
    outside['cells']['start']['wall_mm'] = [340, 350]
    yield example, outside, 'outside_author_guard', 'no_action'

    # A syntactically enabled command can still violate the host option's entry condition.
    short_clearance = deepcopy(push)
    short_clearance['cells']['start']['wall_mm'] = [180, 190]
    wide_guard = deepcopy(example)
    wide_guard['rules'][0]['when'] = {'wall_mm': [180, 320]}
    yield wide_guard, short_clearance, 'unsafe_entry', 'counterexample'
    yield rule('hold', {}, 'STOP', {}, {}), push, 'always_stop', 'no_action'
    stop_after = deepcopy(example)
    stop_after['rules'][0]['then'] = {'scan': 'STOP'}
    yield stop_after, push, 'push_then_only_stop', 'counterexample'

    peer = yield_profile()
    for name, peer_claim in [('peer_omitted', None), ('peer_assumed_stationary', [0, 0]),
                             ('peer_motion_disclosed', [-60, 60])]:
        expect = {'advance_mm': [40, 80]}
        if peer_claim is not None:
            expect['peer_toward_mm'] = peer_claim
        author = rule(name, {'peer_gap_mm': [340, 345]}, 'push_with_yield', expect,
                      {'scan': 'cross_behind'})
        yield author, peer, name, 'counterexample'
    # Sufficient clearance for handoff isolates the *missing declaration* from a
    # physically inadmissible continuation in the other peer cases.
    roomy = deepcopy(peer)
    roomy['cells']['start']['peer_gap_mm'] = [500, 505]
    omitted = rule('roomy_yield', {}, 'push_with_yield', {'advance_mm': [40, 80]},
                   {'scan': 'cross_behind'})
    yield omitted, roomy, 'roomy_peer_omitted', 'unknown'

    observed = rule('scan_then_pass', {}, 'scan_both_sides', {},
                    {'left_open': 'pass_left', 'right_open': 'pass_right'})
    yield observed, scan_profile(False), 'visible_successor', 'proved'
    for side in ('left', 'right'):
        occluded = rule('scan_then_pass', {}, 'scan_both_sides', {},
                        {'occluded': f'pass_{side}'})
        yield occluded, scan_profile(True), f'occluded_commit_{side}', 'counterexample'

    cabinet = cabinet_profile()
    for option, span, expected in [('grip_slim', [80, 85], 'proved'),
                                   ('grip_crosswise', [105, 125], 'counterexample')]:
        author = rule('grasp_and_withdraw', {'opening_mm': [110, 110]}, option,
                      {'loaded_width_mm': span}, {'grasped': 'withdraw'})
        yield author, cabinet, f'cabinet_{option}', expected


def audit_case(program, profile, name, expected):
    result = {'case': name, 'expected': expected, 'program_sha256': digest_spec(program),
              'profile_sha256': digest_spec(profile)}
    compiled = compile_policy(program, profile)
    audit = compiled.check()
    actual = audit['status']
    if actual != expected:
        raise AssertionError(f'{name}: wanted {expected}, got {json.dumps(audit)}')
    cell_case, = audit['cases']
    dispatch = compiled.decide('start')
    opt_in = compiled.decide('start', allow_empirical=True)
    primary = program['rules'][0]['option']
    if dispatch != (primary if actual == 'proved' else None):
        raise AssertionError(f'{name}: default dispatch {dispatch}')
    if opt_in != (primary if actual in ('proved', 'conditional') else None):
        raise AssertionError(f'{name}: empirical dispatch {opt_in}')
    continuations = {symbol: compiled.resume('start', symbol)
                     for symbol in program['rules'][0]['then']}
    if compiled.resume('start', 'unrecognised-public-observation') is not None:
        raise AssertionError(f'{name}: unexpected observation dispatched')
    if actual == 'proved' and continuations != program['rules'][0]['then']:
        raise AssertionError(f'{name}: continuation mismatch {continuations}')
    if actual != 'proved' and any(v is not None for v in continuations.values()):
        raise AssertionError(f'{name}: unproved continuation {continuations}')
    result.update(status=actual, dispatch=dispatch, dispatch_allow_empirical=opt_in,
                  next_public_dispatch=continuations, issues=cell_case['issues'],
                  branches=cell_case['branches'])
    return result


def check_public_menus(policy, directory):
    """Transport a synthetic accepted token onto an archived public action menu."""
    token = policy.decide('start')
    if token != 'push_short':
        raise AssertionError(f'unexpected option token {token!r}')
    rows = []
    for name in BOUNDARY_NAMES:
        raw = (directory / name).read_bytes()
        sample = json.loads(raw)
        packet = sample['observation'] if 'observation' in sample else sample['events'][-1]['observation']
        obs = public_observation(packet)
        command = {'target_id': obs['goal']['target_id'], 'a0': 7, 'a1': 7, 'ticks': 12}
        issued = validate_action(command, obs)
        rows.append(dict(boundary=name, input_sha256=digest(raw), tick=obs['tick'],
                         action_version=obs.get('action_version'), accepted_command=issued))
    return rows


def run(public_edges):
    example = parse_json(EXAMPLE.read_bytes())
    rows = [audit_case(p, m, name, expected) for p, m, name, expected in trials(example)]
    menus = [] if public_edges is None else check_public_menus(
        compile_policy(example, push_profile()), public_edges)
    return {
        'schema': 'guarded-option-probe-v1',
        'scope': 'Synthetic independent host intervals. Conditional on these abstract models.',
        'units': 'mm; integer closed intervals',
        'public_menu_scope': ('Replays only public menu validation of the accepted token. '
                              'Synthetic wall intervals are not measured at these boundaries. '
                              'Ticks cap duration and do not promise box progress.'),
        'author_example': str(EXAMPLE.relative_to(ROOT)),
        'author_example_sha256': digest(EXAMPLE.read_bytes()),
        'host_push_profile': str(PUSH_PROFILE.relative_to(ROOT)),
        'host_push_profile_sha256': digest(PUSH_PROFILE.read_bytes()),
        'checker_sha256': digest(Path(guarded_options.__file__).read_bytes()),
        'probe_sha256': digest(Path(__file__).read_bytes()),
        'cases': rows,
        'archived_public_menus': menus,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command')
    check = subcommands.add_parser('check', help='check an author program against a separate host profile')
    check.add_argument('--program', type=Path, required=True, help='author rule JSON')
    check.add_argument('--profile', type=Path, required=True, help='trusted host profile JSON')
    check.add_argument('--out', dest='check_out', type=Path, help='write JSON here (default stdout)')
    parser.add_argument('--out', type=Path, help='write the compact JSON report')
    edges = parser.add_mutually_exclusive_group()
    edges.add_argument('--public-edges', type=Path, default=ARCHIVED,
                       help='directory with the three archived development boundary packets')
    edges.add_argument('--static-only', action='store_true', help='run the portable interval examples only')
    args = parser.parse_args()
    if args.command == 'check':
        report, code = check_source_files(args.program, args.profile)
        destination = args.check_out if args.check_out is not None else args.out
        if destination and destination.resolve() in {args.program.resolve(), args.profile.resolve()}:
            check.error('report output would overwrite an input file')
        payload = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload)
        else:
            print(payload, end='')
        return code

    report = run(None if args.static_only else args.public_edges)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for case in report['cases']:
        print(f"{case['case']:25} {case['status']}")
    print(f"Public action menus accepted: {len(report['archived_public_menus'])}")


if __name__ == '__main__':
    raise SystemExit(main())
