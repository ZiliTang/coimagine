"""M0 mechanism/contract tests and explicitly bounded M1 rigid-body probes."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.planar_tasks.relational_search.contract import History, action_space, public_observation, validate_action
from experiments.planar_tasks.relational_search.evidence import Outcome, compare, Ledger, group_split
from experiments.planar_tasks.relational_search.fit import Ranker, fit
from experiments.planar_tasks.relational_search.search import Budget, choose


def obs(tick=0, labels=('a', 'b')):
    return dict(protocol='test', tick=tick, decision=tick, goal=dict(target_id=labels[0], xy=[3., 0.], radius=.1),
                robots=[], entities=[dict(id=k, points=[[1., float(i)]], last_seen_tick=tick) for i, k in enumerate(labels)],
                legal_pairs=[[0, 0], [44, 44]])


def action(ticks=20, target='a', a=0):
    return dict(target_id=target, a0=a, a1=a, ticks=ticks)


def test_public_projection_and_hidden_invariance():
    a, b = obs(), obs()
    a['mass'] = 5
    b['mass'] = 500
    a['entities'][0]['truth'] = {'friction': .1}
    assert public_observation(a) == public_observation(b)
    b['entities'][0]['last_seen_tick'] = 1
    with pytest.raises(ValueError):
        public_observation(b)


def test_candidates_exhaustive_and_equivariant():
    commands = list(action_space(obs()))
    assert len(commands) == (2 * 2 + 1) * 40
    assert set(c['ticks'] for c in commands) == set(range(1, 41))
    renamed = list(action_space(obs(labels=('z', 'y'))))
    assert [dict(c, target_id={'a': 'z', 'b': 'y', None: None}[c['target_id']]) for c in commands] == renamed
    for a in commands:
        validate_action(a, obs())


@pytest.mark.parametrize('change', [dict(ticks=True), dict(ticks=41), dict(target_id='hidden'), dict(a0=5)])
def test_invalid_commands(change):
    with pytest.raises(ValueError):
        validate_action(dict(action(), **change), obs())


def test_stale_target_and_history_isolation():
    o = obs(10)
    o['entities'][1]['last_seen_tick'] = 0
    with pytest.raises(ValueError):
        validate_action(action(target='b'), o)
    h = History()
    h.append(obs())
    child = h.fork()
    child.append(obs(20), action())
    assert len(h.events) == 1 and len(child.events) == 2
    with pytest.raises(ValueError):
        child.append(obs(20), action())
    child.events[-1]['observation']['goal']['xy'][0] = 999
    assert h.events[0]['observation']['goal']['xy'] == [3., 0.]


def test_history_window_and_real_duration():
    h = History()
    h.append(obs())
    for tick in range(1, 70):
        h.append(obs(tick), action(1))
    assert len(h.events) == 64
    with pytest.raises(ValueError):
        h.append(obs(72), action(1))


def test_evidence_unknowns_and_comparability():
    def outcome(status, cost, **kw):
        return Outcome(**dict(dict(origin='prefix-hash', action=action(), continuation='shared-v1',
                                   budget={'ticks': 100}, source='execution', status=status, cost=cost), **kw))
    good = outcome('success', 12.)
    bad = outcome('task_failure', 180.)
    assert compare(good, bad)['preferred']['cost'] == 12.
    assert compare(good, outcome('search_unknown', None))['status'] == 'unknown'
    assert compare(good, good)['status'] == 'tie'
    with pytest.raises(ValueError):
        compare(good, outcome('success', 3., origin='other-prefix'))
    with pytest.raises(ValueError):
        outcome('task_failure', None)


def test_ledger_atomic_and_group_split():
    ledger = Ledger(dict(execution=20, prediction=10))
    ledger.charge(execution=5)
    with pytest.raises(RuntimeError):
        ledger.charge(execution=1, prediction=11)
    assert ledger.used == dict(execution=5, prediction=0)
    with pytest.raises(ValueError):
        ledger.charge(execution=-5)
    assert group_split('mother', 'sealed') == group_split('mother', 'sealed')


def test_pairwise_fit_missingness_and_checkpoint():
    model, losses = fit([[1., None], [2., None]], [[-1., 0.], [-2., 0.]])
    assert losses[-1] < losses[0] * .2
    assert (model.score([[1., None], [2., None]]) > model.score([[-1., 0.], [-2., 0.]])).all()
    restored = Ranker.restore(model.record())
    assert np.array_equal(model.score([[1., None]]), restored.score([[1., None]]))
    with pytest.raises(ValueError):
        fit([[float('inf')]], [[0.]])


def test_residual_offset_keeps_semantic_model_fixed():
    semantic, _ = fit([[1.]], [[-1.]])
    seal = semantic.record()
    margin = semantic.score([[1.]]) - semantic.score([[-1.]])
    residual, _ = fit([[2.]], [[-2.]], offsets=margin, residual_penalty=.1)
    assert semantic.record() == seal
    assert np.isfinite(residual.weights).all()


def test_executable_semantics_window_and_branch_memory():
    from experiments.planar_tasks.relational_search.features import FeatureProgram
    def update(m, event):
        m['samples'] += 1
        m['ticks'] += event['executed_ticks'] or 0
        return m
    program = FeatureProgram(lambda: dict(samples=0, ticks=0), update,
                             lambda m, o, a: dict(samples=m['samples'], elapsed=m['ticks'], missing=None),
                             ['samples', 'elapsed', 'missing'])
    h = History(); h.append(obs())
    child = h.fork(); child.append(obs(20), action())
    assert program.evaluate(child, action()) == [2., 20., None]
    assert program.evaluate(h, action()) == [1., 0., None]
    assert program.evaluate(h, action()) == [1., 0., None]
    oversized = FeatureProgram(lambda: {'x': 'a'*8193}, update, lambda m, o, a: {}, ['x'])
    with pytest.raises(ValueError):
        oversized.evaluate(h, action())


class Model:
    def __init__(self, shorten=False, fail=False):
        self.shorten, self.fail = shorten, fail

    def predict(self, handle, a):
        if self.fail:
            raise RuntimeError('model failure')
        ticks = 1 if self.shorten else a['ticks']
        o = obs(handle + ticks)
        return dict(handle=handle+ticks, observation=o, ticks=ticks, cost_seconds=ticks*.05,
                    terminal=False, success=False)


def test_search_priority_separate_from_cost_and_actual_horizon():
    h = History(); h.append(obs()); seal = h.sha256
    result = choose(h, 0, Model(), lambda h, a: 1e9 if a['a0'] == 0 else -1e9,
                    lambda h: 10., Budget(prediction_ticks=160, calls=4, batch=4))
    assert h.sha256 == seal
    assert result['action'] is not None
    assert result['selected']['cost'] < 100
    assert all(e['actual_ticks'] == e['requested_ticks'] == 40 for e in result['trace'])
    assert result['trace'][-1]['fair']


def test_search_model_errors_charged_and_no_unpredicted_action():
    h = History(); h.append(obs())
    result = choose(h, 0, Model(fail=True), lambda h, a: 0., lambda h: 1.,
                    Budget(prediction_ticks=80, calls=2))
    assert result['action'] is None
    assert result['prediction_ticks'] == 80
    assert all(e['status'] == 'error' for e in result['trace'])


def test_search_rejects_one_tick_substitution():
    h = History(); h.append(obs())
    result = choose(h, 0, Model(shorten=True), lambda h, a: 0., lambda h: 1.,
                    Budget(prediction_ticks=80, calls=2))
    assert result['action'] is None
    assert all('shortened' in e['error'] for e in result['trace'])


def test_small_prediction_budget_reaches_shorter_windows():
    h = History(); h.append(obs())
    result = choose(h, 0, Model(), lambda h, a: 0., lambda h: 1.,
                    Budget(prediction_ticks=10, calls=1, batch=4))
    assert result['action']['ticks'] == 10


def test_global_brake_when_all_targets_occluded():
    o = obs(); o['entities'] = []
    assert len(list(action_space(o))) == 40
    validate_action(action(target=None, a=44), o)
    with pytest.raises(ValueError):
        validate_action(action(target=None), o)


def test_target_handoff_requires_real_slow_boundaries():
    from experiments.planar_tasks.relational_search.target_binding import TargetBinding
    binding = TargetBinding()
    def o(t, speed=0.):
        p = obs(t)
        p['robots'] = [dict(own=dict(vx=speed, vy=0.)) for _ in range(2)]
        return p
    first = binding.select(action(), o(0, 1.))
    assert first['phase'] == 'switch_brake' and first['command']['target_id'] is None
    assert not binding.select(action(), o(20))['commit']
    assert binding.select(action(), o(40))['commit']
    assert binding.current == 'a'
    assert binding.select(action(), o(60))['phase'] == 'continue'
    assert not binding.select(action(target='b'), o(80))['commit']
    assert binding.select(action(target='b'), o(100))['commit']
    assert binding.current == 'b'
    with pytest.raises(ValueError):
        binding.select(action(target='b'), o(100))


def test_physics_probe_public_labels_and_exact_prefix_replay():
    from experiments.planar_tasks.relational_search.physics_probe import PhysicsProbe, Fixture
    p = PhysicsProbe()
    initial = p.observation()
    assert 'mass' not in str(initial)
    heavy = PhysicsProbe(Fixture(blocker_mass=30.))
    assert initial == heavy.observation()
    # B is occluded by A in this fixture: do not invent B in the public table.
    assert all(e['id'] != 'entity-b' for e in initial['entities'])
    for _ in range(10):
        p.step_forces(((10., 0.), (10., 0.)))
    assert p.replay().audit_state() == p.audit_state()
    with pytest.raises(NotImplementedError):
        p.backend.export_state()


def test_dataset_roundtrip_unknowns_tampering_and_group_leakage(tmp_path):
    from experiments.planar_tasks.relational_search.dataset import record, write, read_training, verify
    h = History(); h.append(obs())
    def outcome(a, status, cost):
        return Outcome(h.sha256, a, 'same-continuation', {'ticks': 100}, 'execution', status, cost)
    row = record('mother', 'train', h, outcome(action(), 'success', 5.),
                 outcome(action(a=44), 'task_failure', 180.))
    unknown = record('other-mother', 'validation', h, outcome(action(), 'success', 5.),
                     outcome(action(a=44), 'search_unknown', None))
    path = tmp_path/'evidence.jsonl'
    manifest = write(path, [row, unknown])
    assert manifest['rows'] == 2 and len(read_training(path)) == 1
    with pytest.raises(FileExistsError):
        write(path, [row])
    altered = deepcopy(row); altered['comparison']['preferred']['cost'] = 0
    with pytest.raises(ValueError):
        verify(altered)
    leaked = deepcopy(row); leaked['split'] = 'test'
    with pytest.raises(ValueError):
        write(tmp_path/'leaked.jsonl', [row, leaked])


def test_learned_history_feature_changes_search_not_only_scores():
    from experiments.planar_tasks.relational_search.smoke import mechanism_fixture
    result = mechanism_fixture()
    old = Ranker.restore(result['models']['initial']['checkpoint'])
    revised = Ranker.restore(result['models']['revised']['checkpoint'])
    h = History()
    past = obs(); past['entities'][0]['points'] = [[-.3, 0.]]
    h.append(past); h.append(obs(20), action())
    class WitnessModel:
        def predict(self, handle, a):
            success = a['a0'] == 44
            return dict(handle=handle+a['ticks'], observation=obs(handle+a['ticks']), ticks=a['ticks'],
                        cost_seconds=a['ticks']*.05, terminal=success, success=success)
    def scorer(model, new):
        def score(history, a):
            command = 1. if a['a0'] == 0 else -1.
            x = [command]
            if new:
                x.append(command * history.events[0]['observation']['entities'][0]['points'][0][0])
            return float(model.score([x])[0])
        return score
    budget = Budget(calls=1, prediction_ticks=40, batch=4)
    before = choose(h, 20, WitnessModel(), scorer(old, False), lambda h: 100., budget)
    after = choose(h, 20, WitnessModel(), scorer(revised, True), lambda h: 100., budget)
    assert before['action']['a0'] == 0 and after['action']['a0'] == 44
    assert not before['selected']['success'] and after['selected']['success']
    assert before['prediction_ticks'] == after['prediction_ticks'] == 40
    assert len(h.events) == 2
