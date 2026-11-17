"""Mechanism and deployment-contract checks, not learned-policy performance."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from experiments.planar_tasks.semantic_planning import reference
from experiments.planar_tasks.semantic_planning.artifact import (
    compile_source, create_revision, verify_revision, validate_source,
)
from experiments.planar_tasks.semantic_planning.runtime import SemanticRuntime, visible, validate_trees
from experiments.planar_tasks.option_search_v3.search import Search, Budget, action_space, root_state
from experiments.planar_tasks.option_search_v3 import build
from experiments.planar_tasks.option_search_v3.hook_client import HookClient
from experiments.planar_tasks import interface


def observation(decision=0, contact=False):
    own = dict(x=0., y=0., angle=0., vx=0., vy=0., omega=0.)
    return dict(protocol='test', decision=decision, legal_pairs=[[3, 3], [44, 44]],
                robots=[dict(robot_id=i, own=own, teammate=own, scan={}, patches=[], patch=None,
                             elapsed_seconds=decision * 2., goal_xy=[3., 0.], contacting_object=contact,
                             execution=dict(stage='push' if contact else 'approach')) for i in range(2)])


def context(obs):
    return dict(observation=obs, hypothesis=dict(object=[0., 0., 0., 0., 0., 0.],
                robots=[[0.] * 6, [0.] * 6]), previous_action=None)


def node(obs, path=(), ident=0, ticks=None):
    return dict(id=ident, observation=obs, source='predicted' if path else 'real_observation_estimate',
                state=root_state(context(obs)['hypothesis']), goal=[3., 0.],
                incoming_action=path[-1] if path else None, path_context=list(path),
                elapsed_ticks=sum(a['ticks'] for a in path) if ticks is None else ticks,
                depth=len(path))


def runtime(log=None, **kwargs):
    return SemanticRuntime(reference.initial_memory, reference.update_memory,
                           kwargs.get('features', reference.semantic_features),
                           kwargs.get('trees', reference.TREES), 'test', emit=(log if log is not None else []).append)


def test_sensor_projection_matches_existing_contract_and_drops_privileged_fields():
    from experiments.planar_tasks.semantic_planning import runtime as core
    for name in ('OBS_FIELDS', 'KINEMATICS_FIELDS', 'SCAN_FIELDS', 'PATCH_FIELDS', 'FEEDBACK_FIELDS'):
        assert getattr(core, name) == getattr(interface, name)
    clean = observation()
    contaminated = deepcopy(clean)
    contaminated['true_state'] = {'friction': 1000}
    contaminated['robots'][0]['hidden_mode'] = 'blocked'
    contaminated['robots'][0]['own']['truth'] = 12
    contaminated['robots'][0]['execution']['pair_phase'] = 'SECRET'
    assert visible(clean) == visible(contaminated)
    outputs = []
    for obs in (clean, contaminated):
        log = []
        r = runtime(log)
        r.evaluate(node(obs), context(obs))
        outputs.append(log)
    assert outputs[0] == outputs[1]


def test_root_idempotence_and_forecast_siblings_never_change_real_memory():
    r = runtime()
    obs = observation()
    root = node(obs)
    r.evaluate(root, context(obs))
    r.rank(root, [dict(a0=3, a1=3, ticks=40)], context(obs))
    assert r.real_memory['real_samples'] == 1
    before = deepcopy(r.real_memory)
    a, b = dict(a0=3, a1=3, ticks=40), dict(a0=44, a1=44, ticks=20)
    r.evaluate(node(observation(contact=True), [a], 1), context(obs))
    r.evaluate(node(observation(), [b], 2), context(obs))
    assert r.real_memory == before
    assert r.branches[r.path_key([a])][0]['last_both_contact_time'] == 0.
    assert r.branches[r.path_key([b])][0]['last_both_contact_time'] is None
    r.evaluate(node(observation(contact=False), [a, b], 3), context(obs))
    assert r.branches[r.path_key([a, b])][0]['synthetic_samples'] == 2
    next_obs = observation(1)
    r.evaluate(node(next_obs), dict(context(next_obs), previous_action=a))
    assert r.real_memory['real_samples'] == 2
    assert r.real_memory['synthetic_samples'] == 0
    assert r.real_memory['last_both_contact_time'] is None
    assert set(r.branches) == {()}


def test_feature_mutation_cannot_modify_stored_memory_or_frame():
    def mutating(memory, frame, action):
        memory['corrupted'] = True
        frame['observation']['legal_pairs'].clear()
        return {'q': 1.}
    r = runtime(features=mutating, trees={'value': {'cost': 1.}, 'rank': {'cost': {'feature': 'q'}}})
    obs = observation()
    r.evaluate(node(obs), context(obs))
    assert 'corrupted' not in r.real_memory
    assert r.branches[()][1]['observation']['legal_pairs'] == obs['legal_pairs']


def test_unknown_has_explicit_branch_and_nonfinite_costs_fail():
    tree = dict(feature='missing', op='eq', value=True,
                yes={'cost': 1.}, no={'cost': 2.}, unknown={'cost': 3.})
    r = runtime(trees={'value': tree, 'rank': tree})
    obs = observation()
    assert r.evaluate(node(obs), context(obs)) == 3.
    bad = deepcopy(tree)
    del bad['unknown']
    with pytest.raises(ValueError):
        validate_trees({'value': bad, 'rank': tree})
    with pytest.raises(ValueError):
        validate_trees({'value': {'cost': float('nan')}, 'rank': tree})


class Adapter:
    def __init__(self, r):
        self.runtime = r
        self.nodes = {}
    def begin(self, context, deadline):
        self.context = context
        self.nodes = {}
    def register(self, ident, node, deadline):
        self.nodes[ident] = node
        return self.runtime.evaluate(node, self.context)
    def rank(self, ident, candidates, deadline):
        return self.runtime.rank(self.nodes[ident], candidates, self.context)


class NoModel:
    def cost(self, *args):
        raise AssertionError('zero expansion test cannot predict')


def test_same_current_observation_different_history_changes_tree_and_search_action():
    # Deliberately constructed diagnostic, NOT evidence this contact rule is useful.
    tree = dict(feature='has_contact_evidence', op='eq', value=True,
                yes={'cost': {'feature': 'push_cost'}},
                no={'cost': {'feature': 'brake_cost'}}, unknown={'cost': 100.})
    def features(memory, frame, action):
        return dict(has_contact_evidence=memory['last_both_contact_time'] is not None,
                    push_cost=0. if action and action['a0'] == 3 else 100.,
                    brake_cost=0. if action and action['a0'] == 44 else 100.)
    selected, paths = [], []
    for prior_contact in (False, True):
        logs = []
        r = runtime(logs, features=features, trees={'value': {'cost': 0.}, 'rank': tree})
        first = observation(0, prior_contact)
        r.evaluate(node(first), context(first))
        current = observation(1, False)
        action, trace = Search(Adapter(r), Budget(expansions=0)).choose(current, context(current), NoModel())
        selected.append(action['a0'])
        paths.append([x for x in logs if x['stage'] == 'rank'][0]['candidates'][0]['branch'])
        assert trace['root_action_support'] == 80
    assert selected == [44, 3]
    assert paths[0][0]['branch'] == 'no' and paths[1][0]['branch'] == 'yes'


@pytest.mark.parametrize('ticks', [1, 7, 40])
def test_arbitrary_legal_command_is_still_immediately_proposable(ticks):
    target = dict(a0=44, a1=44, ticks=ticks)
    leaf = dict(cost=100., proposals=[dict(target, cost=-100.)])
    r = runtime(trees={'value': {'cost': 100.}, 'rank': leaf})
    obs = observation()
    action, trace = Search(Adapter(r), Budget(expansions=0, candidate_batch=1)).choose(obs, context(obs), NoModel())
    assert action == target
    assert trace['selected_status'] == 'unexpanded'
    support = {(a.a0, a.a1, a.ticks) for a in action_space(obs)}
    assert support == {(a, b, t) for a, b in obs['legal_pairs'] for t in range(1, 41)}


def test_illegal_proposal_fails_and_search_reports_hook_error():
    r = runtime(trees={'value': {'cost': 0.}, 'rank': {'cost': 0., 'proposals': [dict(a0=999, a1=999, ticks=40, cost=-1.)]}})
    obs = observation()
    action, trace = Search(Adapter(r), Budget(expansions=0)).choose(obs, context(obs), NoModel())
    assert trace['hook_errors'][0]['stage'] == 'rank_actions'
    assert [action['a0'], action['a1']] in obs['legal_pairs']


def test_memory_limit_and_reused_real_boundary_rejected():
    with pytest.raises(ValueError):
        SemanticRuntime(lambda: {'x': 'a' * 9000}, lambda m, e: m, lambda *a: {}, reference.TREES, 'bad')
    r = runtime()
    r.evaluate(node(observation()), context(observation()))
    with pytest.raises(ValueError, match='same real boundary'):
        r.evaluate(node(observation(contact=True)), context(observation()))


def test_revision_lineage_integrity_and_no_host_execution(tmp_path):
    source = Path(reference.__file__)
    first = tmp_path / 'v0'
    m = create_revision(source, first)
    assert verify_revision(first)['knowledge_id'] == m['knowledge_id']
    with pytest.raises(FileExistsError):
        create_revision(source, first)
    with pytest.raises(ValueError, match='evidence'):
        create_revision(source, tmp_path / 'missing', parent=first)
    evidence = tmp_path / 'diagnosis.json'
    evidence.write_text('{"status": "synthetic test only"}')
    second = create_revision(source, tmp_path / 'v1', parent=first, evidence=[evidence], reason='lineage test')
    assert second['parent']['knowledge_id'] == m['knowledge_id']
    (first / 'knowledge.py').write_text('tampered')
    with pytest.raises(ValueError):
        verify_revision(first)
    bad = source.read_text() + '\nopen("BAD", "w").write("x")\n'
    with pytest.raises(ValueError):
        compile_source(bad)
    with pytest.raises(ValueError):
        validate_source(source.read_text().replace('def initial_memory():', 'def initial_memory(x=print("BAD")):'))


def test_exported_program_uses_actual_frozen_worker_protocol(tmp_path, monkeypatch):
    # Real subprocess IPC, no execution physics and no LLM.
    compiled, _ = compile_source(Path(reference.__file__).read_text())
    source = tmp_path / 'heuristic.py'
    source.write_text(compiled)
    package = tmp_path / 'package'
    build.build_package(package, source)
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    monkeypatch.chdir(package)
    client = HookClient()
    try:
        obs = observation()
        client.begin(context(obs), time.monotonic() + 5)
        assert client.register(0, node(obs), time.monotonic() + 5) == pytest.approx(10.)
        reply = client.rank(0, [dict(a0=3, a1=3, ticks=40)], time.monotonic() + 5)
        assert reply == dict(costs=[10.], proposals=[])
    finally:
        client.close()
    build.verify_package(package)


def test_summary_adapter_handles_real_request_response_format(tmp_path):
    from experiments.planar_tasks.authoring_v2 import campaign_config
    from experiments.planar_tasks.semantic_planning.session import create_session
    config = campaign_config(dict(provider='synthetic', name='no-network', settings={},
                                  source_access='hidden', accounting='reserved_output_tokens_and_input_bytes'),
                             ['planar_open_anchor'])
    session = create_session(tmp_path / 'session', config)
    try:
        session.state['tests'].append(dict(id='test_0000', status='finished', task='planar_open_anchor'))
        path = session.path / 'private/test_0000/episode'
        path.mkdir(parents=True)
        obs = observation()
        (path / 'sensors.jsonl').write_text(json.dumps({'request': {'observation': obs}}) + '\n' +
                                           json.dumps({'response': dict(a0=3, a1=3, ticks=40)}) + '\n')
        assert session._observations('test_0000') == [obs]
        assert session.test_summary('test_0000')['observations']['samples'] == 1
        assert session.world is None
    finally:
        session.close()
