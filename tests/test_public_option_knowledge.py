"""Knowledge delivery/provenance and controller-contract checks; no physics rollout."""
from copy import deepcopy
import json
import pytest
from experiments.planar_tasks.authoring import knowledge, create_session, open_session, verify_frozen
from experiments.planar_tasks.authoring import driver
from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.interface import contract
from experiments.option_rl_v19.unknown_shape import make_unknown_skill, UnknownShapePush, SharedMotionEstimator
from experiments.option_rl_v13.push_controller import ContinuousPush
from experiments.option_rl_v21.system import YieldSlot
from experiments.option_rl_v21 import spec as S


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, 'LOCK', tmp_path/'knowledge_release.json')
    knowledge.freeze()
    return knowledge.verify()


def config():
    c = json.loads((seal.BASE/'development_config.json').read_text())
    c['model'].update(provider='synthetic', name='no-network-model', settings={})
    c['prediction'] = dict(version='nominal-v1', runtime=True)
    c['budget'].update(wall_seconds=60, model_calls=2, model_call_output_tokens=512,
                       model_output_tokens=1024, model_input_bytes=200000)
    c.pop('finalization', None); c.pop('submission_settings', None)
    return c


@pytest.mark.parametrize('method', ['free', 'option_search_v3'])
@pytest.mark.parametrize('finalizing', [False, True])
def test_delivery_before_budget_accounting_and_atomic_submission(tmp_path, monkeypatch, release, method, finalizing):
    campaign = config()
    if finalizing:
        campaign['budget'].update(model_calls=4, model_output_tokens=2048)
        campaign['finalization'] = dict(model_calls=2, model_output_tokens=1024,
                                       model_input_bytes=100000, tool_calls=4, wall_seconds=20)
    s = create_session(tmp_path/'session', campaign, method=method)
    try:
        guide = knowledge.GUIDE.read_text()
        assert (s.workspace/'option_guide.md').read_text() == guide
        c = json.loads((s.workspace/'contract.json').read_text())
        assert c['public_skill_knowledge']['sha256'] == seal.digest(knowledge.GUIDE)
        assert {k:v for k,v in c.items() if k in contract()} == json.loads(json.dumps(contract()))
        with pytest.raises(ValueError, match='host-owned'):
            s.write_file('option_guide.md', 'changed')
        if method == 'free':
            name = 'policy.py'
            source = 'class Policy:\n def __init__(self,contract): pass\n def act(self,obs): return dict(a0=44,a1=44,ticks=1)\n'
        else:
            name = 'heuristic.py'
            source = 'def evaluate_state(node,memory): return 0.\ndef rank_actions(parent,candidates,memory): return dict(costs=[0.]*len(candidates))\n'
        captured = []
        if finalizing:
            s.finalize('submission-only interface check')
        def fake(complete, payload, timeout):
            captured.append(deepcopy(payload))
            assert guide in payload['messages'][0]['content']
            assert 'option_guide.md' in payload['messages'][1]['content']
            if finalizing:
                assert [t['function']['name'] for t in payload['tools']] == ['submit']
            return {'ok':True, 'response':dict(model='no-network-model', usage={}, choices=[dict(
                finish_reason='tool_calls', message=dict(content=None, tool_calls=[dict(id='submit1',
                type='function', function=dict(name='submit', arguments=json.dumps(dict(files=[name], contents={name:source}))))]))])}
        monkeypatch.setattr(driver, 'bounded_completion', fake)
        assert driver.drive(s, None)['status'] == 'submitted'
        assert s.state['usage']['model_input_bytes'] == len(json.dumps(captured[0], ensure_ascii=False).encode())
        assert s.state['usage'].get('control_ticks', 0) == 0
        assert s.state['usage'].get('policy_tests', 0) == 0
        assert verify_frozen(s.path)['public_skill_knowledge']['version'] == knowledge.VERSION
    finally:
        s.close()
    reopened = open_session(tmp_path/'session'); reopened.close()
    manifest = tmp_path/'session/frozen/manifest.json'
    m = json.loads(manifest.read_text()); m.pop('public_skill_knowledge')
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='provenance'):
        verify_frozen(tmp_path/'session')


@pytest.mark.parametrize('filename', ['option_guide.md', 'instructions.txt', 'contract.json'])
def test_script_side_material_tampering_is_detected(tmp_path, release, filename):
    s = create_session(tmp_path/'session', config())
    try:
        (s.workspace/filename).write_text('tampered by script')
        with pytest.raises(ValueError, match='material changed'):
            s._active()
    finally:
        s.close()


def test_same_guide_for_both_authoring_conditions(tmp_path, release):
    a = create_session(tmp_path/'free', config(), method='free')
    b = create_session(tmp_path/'search', config())
    try:
        assert (a.workspace/'option_guide.md').read_bytes() == (b.workspace/'option_guide.md').read_bytes()
        assert a.config['budget'] == b.config['budget']
    finally:
        a.close(); b.close()


def test_documented_dispatch_and_transition_semantics():
    estimator = SharedMotionEstimator()
    factory = lambda a: make_unknown_skill(a, estimator=estimator, action_spec=S)
    assert isinstance(factory(S.ACTIONS[0]), UnknownShapePush)
    assert type(factory(S.ACTIONS[32])) is ContinuousPush
    slot = YieldSlot(factory)
    # Commit without stepping a world; only test the documented state machine.
    slot.pending = S.ACTIONS[0]; slot._commit(); first = slot.skill
    slot.select(S.ACTIONS[3]); assert slot.skill is first and slot.pending is None
    slot.select(S.ACTIONS[8]); assert slot.skill is first and slot.pending is None
    first.done = True
    slot.select(S.ACTIONS[9]); assert slot.pending is None and slot.skill.done
    slot.select(S.ACTIONS[0]); assert slot.pending is S.ACTIONS[0]
    slot.select(S.ACTIONS[46]); assert slot.pending is None and slot.action.kind == 'yield'
    slot.select(S.ACTIONS[44]); assert slot.pending is S.ACTIONS[44]
    slot.pending = S.ACTIONS[32]; slot._commit(); first = slot.skill; first.done = True
    slot.select(S.ACTIONS[34]); assert slot.skill is first and slot.pending is None
    slot.select(S.ACTIONS[35]); assert slot.pending is S.ACTIONS[35]
