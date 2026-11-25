import ast
import json
from pathlib import Path

import pytest

from experiments.planar_tasks.authoring_v2.checked_submission import submit_checked, verify_checked
from experiments.planar_tasks.authoring_v2.policy_check_specs import check_declared_uses, validate_source_rules


def spec(widths, models):
    return validate_source_rules(ast.parse(
        f"POLICY_CHECK_RULES = {{'version': 1, 'widths': {widths!r}, 'peer_models': {models!r}}}"
    ))


def test_static_proof_direction_and_peer_role():
    with pytest.raises(ValueError, match='direction of inference'):
        spec({'visible_overlap': ['insufficient']}, {})
    with pytest.raises(ValueError, match='hypothesis'):
        spec({}, {'YIELD': [['stationary', 'exact']]})
    rules = spec({'certified_interval': ['insufficient', 'sufficient', 'unknown']},
                 {'HOLD': [['stationary', 'hypothesis']]})
    observation = {'widths': [], 'peer_predictions': [
        {'pair': [7, 44], 'robot': 1, 'model': 'stationary', 'scope': 'hypothesis'}
    ]}
    check_declared_uses(rules, observation)
    observation['peer_predictions'][0]['pair'] = [7, 46]
    with pytest.raises(ValueError, match='YIELD'):
        check_declared_uses(rules, observation)


def test_static_rules_require_unambiguous_literals():
    with pytest.raises(ValueError, match='duplicate key'):
        validate_source_rules(ast.parse("POLICY_CHECK_RULES = {'widths': {}, 'widths': {}}"))
    with pytest.raises(ValueError, match='literal'):
        validate_source_rules(ast.parse('POLICY_CHECK_RULES = make_rules()'))


def test_real_submission_verifies_frozen_source_and_retains_evidence(tmp_path):
    from experiments.planar_tasks.relational_search.contract import public_observation
    from tools.cap_policy_check import main

    policy = Path(__file__).resolve().parents[1] / 'experiments/planar_tasks/authoring_v2/checked_policy_example.py'
    observation = public_observation({
        'tick': 0, 'active_target': 'box', 'legal_pairs': [[7, 46], [44, 44]],
        'goal': {'target_id': 'box', 'xy': [2., 2.], 'radius': .2},
        'robots': [{'patches': [{'valid': True, 'normal_world': [.7071, .7071], 'span_m': .42}]} for _ in (0, 1)],
        'entities': [{'id': 'box', 'points': [[1., 1.]], 'last_seen_tick': 0}],
        'entity_legal_pairs': {'box': [[7, 46], [44, 44]]}, 'global_legal_pairs': [[44, 44]],
    })
    sample = tmp_path / 'public.json'
    sample.write_text(json.dumps({'observation': observation}))
    destination = tmp_path / 'candidate01'
    assert main(['submit', '--policy', str(policy), '--boundary', str(sample), '--out', str(destination)]) == 0
    receipt = verify_checked(destination)
    assert receipt['cases'][0]['status'] == 'ok'
    with pytest.raises(FileExistsError):
        submit_checked(policy, [sample], destination)
    wrong = tmp_path / 'wrong.json'
    wrong.write_text(json.dumps(dict(observation=observation, action=None)))
    with pytest.raises(ValueError, match='recorded action'):
        submit_checked(policy, [sample, wrong], tmp_path / 'partial')
    assert not (tmp_path / 'partial').exists()
    (destination / 'policy.py').write_text('changed')
    assert main(['verify', '--submission', str(destination)]) != 0
