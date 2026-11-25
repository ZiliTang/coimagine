"""Host/author contract tests on tiny public packets, with one isolated end-to-end run."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from experiments.planar_tasks.relational_search.contract import public_observation
from tools.cap_policy_check import lint_source, main, replay, run_boundary

EXAMPLE = Path(__file__).resolve().parents[1] / 'experiments/planar_tasks/authoring_v2/checked_policy_example.py'


def public_input(tick=0):
    pairs = [[7, 7], [7, 46], [44, 44]]
    fixture = {
        'tick': tick, 'active_target': 'box', 'legal_pairs': pairs,
        'goal': {'target_id': 'box', 'xy': [2., 2.], 'radius': .2},
        'robots': [{'patches': [{'valid': True, 'normal_world': [2**-.5]*2, 'span_m': .42}]} for _ in range(2)],
        'entities': [{'id': 'box', 'points': [[1., 1.]], 'last_seen_tick': tick}],
        'entity_legal_pairs': {'box': pairs}, 'global_legal_pairs': [[44, 44]],
    }
    return public_observation(fixture)


def test_static_hooks_and_production_linkage():
    src = EXAMPLE.read_text()
    assert lint_source(src)['status'] == 'ok'
    renamed = src.replace('def audit_policy(', 'def absent(')
    with pytest.raises(ValueError, match='missing required hook audit_policy'):
        lint_source(renamed)
    disconnected = src.replace("if review_push(memory, observation, pair)['verdict'] == 'pass':", 'if True:')
    with pytest.raises(ValueError, match='must be called'):
        lint_source(disconnected)
    mismatched_args = src.replace('def audit_policy(memory, observation, menu)', 'def audit_policy(memory, observation, other)')
    with pytest.raises(ValueError, match='expected audit_policy'):
        lint_source(mismatched_args)


def test_replay_bound_to_public_menu_and_labels_recorded_output():
    obs = public_input()
    packet = dict(version=1, proposed=[[7, 7]], claim='selected', selected=[7, 7], widths=[], peer_predictions=[],
                  reviews=[dict(pair=[7, 7], verdict='pass', peer_robots=[]),
                           dict(pair=[7, 46], verdict='unknown', peer_robots=[])])
    record = dict(observation=obs, packet=packet, action=dict(target_id='box', a0=7, a1=7, ticks=10))
    verified = replay(record)
    assert verified['status'] == 'ok'
    assert verified['actual_command_binding'] == 'record_only'
    assert verified['counts']['legal'] == 2
    with pytest.raises(ValueError, match='recorded menu differs'):
        replay(dict(record, menu=[[7, 7]]))
    missing = deepcopy(record)
    missing['packet']['reviews'].pop()
    with pytest.raises(ValueError, match='reviews missing'):
        replay(missing)
    extra = deepcopy(record)
    extra['observation']['truth'] = {}
    with pytest.raises(ValueError, match='outside the public protocol'):
        replay(extra)


def test_reject_unbound_source_memory_and_midstream_restart():
    src = EXAMPLE.read_text()
    for boundary, reason in [({'observation': public_input(), 'memory': {}}, 'source_sha256'),
                             ({'observation': public_input(20)}, 'tick-zero')]:
        with pytest.raises(ValueError, match=reason):
            run_boundary(src, boundary)


def test_full_public_event_replay_and_cli(tmp_path):
    src = EXAMPLE.read_text()
    first = public_input()
    second = public_input(3)
    action = dict(target_id='box', a0=7, a1=7, ticks=10)
    boundary = {'events': [dict(observation=first, action=None), dict(observation=second, action=action)], 'action': action}
    inp = tmp_path / 'prefix.json'
    report_file = tmp_path / 'report.json'
    inp.write_text(json.dumps(boundary))
    assert main(['run', '--policy', str(EXAMPLE), '--boundary', str(inp), '--out', str(report_file)]) == 0
    report = json.loads(report_file.read_text())
    assert report['selection_checked'] and report['actual_command_binding'] == 'isolated_production_first_proposal'
    assert report['action'] == action and report['tick'] == 3 and report['counts']['legal'] == 2
    assert report['source_sha256'] == lint_source(src)['source_sha256']
    assert replay(report)['counts'] == report['counts']


def test_disagreement_with_actual_production_and_cli_exit(tmp_path):
    # The author declares abstention even though the ordinary selector emits a PUSH.
    src = EXAMPLE.read_text().replace("return dict(version=1, proposed=proposed, selected=selected,",
                                     "selected = None\n    return dict(version=1, proposed=proposed, selected=selected,")
    result = run_boundary(src, dict(observation=public_input()))
    assert result['status'] == 'violation'
    assert 'selection_mismatch' in {f['code'] for f in result['findings']}
    packet = tmp_path / 'packet.json'
    packet.write_text(json.dumps(result))
    assert main(['replay', '--packet', str(packet)]) == 1
