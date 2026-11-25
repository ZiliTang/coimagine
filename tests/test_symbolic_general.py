"""Counterexamples where existence, priority, framing, or hidden branches matter."""
from copy import deepcopy

import pytest

from experiments.planar_tasks.authoring_v2.symbolic_general.core import Policy, extract


def v(n): return ['var', n]
def lit(x, u='mm'): return ['lit', x, u]
def ge(n, x): return ['ge', v(n), lit(x)]


def example():
    metrics = {'gap': {'unit': 'mm', 'domain': [0, 1000]},
               'face': {'unit': 'bool'}, 'goal': {'unit': 'bool'}}
    def opt(req, productive=True):
        return dict(requires=req, writes=list(metrics), frame=[], productive=productive)
    host = dict(metrics=metrics, options={
        'orbit': opt(ge('gap', 100)), 'push': opt(['and', ge('gap', 150), v('face')]),
        'brake': opt(['lit', True, 'bool'], False)})
    program = dict(version=1, derived={'margin': ['sub', v('gap'), lit(150)]},
                   rules=[dict(name='push', when=v('face'), option='push'),
                          dict(name='orbit', when=['not', v('face')], option='orbit')],
                   terminal=v('goal'))
    return program, host


def test_actual_selected_action_not_existential_safe_alternative():
    p, h = example()
    # A brake is possible but not selected. It cannot rescue the false proof.
    p['rules'].append(dict(name='brake', when=['lit', True, 'bool'], option='brake'))
    q = Policy(p, h)
    assert q.select(dict(gap=120, face=True, goal=False))['executable'] is False
    r = q.check({'gap': [200, 200], 'face': False, 'goal': False}, 'orbit',
                [dict(updates={'gap': [120, 120], 'face': True, 'goal': False}, path_safe=True)], quality='model')
    assert r['status'] == 'counterexample'
    assert r['selection_reasoning']=='exact_score_and_priority'
    assert r['branches'][0]['possible_rules'] == ['push']


def test_merged_guard_shortcut_requires_all_enabled_rules_executable():
    p,h=example()
    for rule in p['rules']:
        rule['when']=['and',rule['when'],h['options'][rule['option']]['requires']]
    p['rules'].append(dict(name='stop',when=['lit',True,'bool'],option='brake'))
    q=Policy(p,h)
    result=q.check({'gap':[200,200]},'orbit',
        [dict(updates={'gap':[0,170],'face':None,'goal':False},path_safe=True)],enumerate_rules=False)
    assert result['status']=='conditional'
    assert result['selection_reasoning']=='certified_merged_guards'
    assert q.select(dict(gap=120,face=True,goal=False))['option']=='brake'


def test_symbolic_selection_partitions_uncertainty_without_oracle():
    p, h = example(); q = Policy(p, h)
    result = q.check({'gap': [200, 210], 'face': False, 'goal': False}, 'orbit',
                     [dict(updates={'gap': [160, 180], 'face': None, 'goal': False}, path_safe=True)], quality='model')
    assert result['status'] == 'proved'
    assert result['branches'][0]['possible_rules'] == ['push', 'orbit']
    for face in (False, True):
        assert q.select(dict(gap=170, face=face, goal=False))['executable']


def test_nonobservable_branch_does_not_choose_hidden_safe_action():
    p, h = example(); q = Policy(p, h)
    branches = [dict(label='clear', updates={'gap': [170, 170], 'face': True, 'goal': False}, path_safe=True),
                dict(label='narrow', updates={'gap': [120, 120], 'face': True, 'goal': False}, path_safe=True)]
    result = q.check({'gap': [200, 200], 'face': False, 'goal': False}, 'orbit', branches)
    assert result['status'] == 'counterexample'


def test_missing_effect_invalidates_written_metric():
    p, h = example(); q = Policy(p, h)
    result = q.check({'gap': [200, 200], 'face': True, 'goal': False}, 'push',
                     [dict(updates={'face': True, 'goal': False}, path_safe=True)])
    assert result['status'] == 'counterexample'


def test_joint_outcome_union_preserves_correlation():
    p, h = example(); q = Policy(p, h)
    branches = [dict(updates={'gap': [160, 170], 'face': True, 'goal': False}, path_safe=True),
                dict(updates={'gap': [110, 130], 'face': False, 'goal': False}, path_safe=True)]
    assert q.check({'gap': [200, 200]}, 'orbit', branches)['status'] == 'conditional'
    # The Cartesian enclosure introduces a real conservative counterexample.
    broad = [dict(updates={'gap': [110, 170], 'face': None, 'goal': False}, path_safe=True)]
    assert q.check({'gap': [200, 200]}, 'orbit', broad)['status'] == 'counterexample'


def test_score_and_stable_tie_break_match_concrete_selector():
    p, h = example()
    p['rules'] = [dict(name='push', when=['lit', True, 'bool'], option='push', score=['div', v('margin'), lit(1, '1')])]
    # Dimensional score is rejected, rather than silently mixing mm and rankings.
    with pytest.raises(ValueError, match='dimensionless'): Policy(p, h)
    p['rules'] = [dict(name='push', when=['lit', True, 'bool'], option='push', score=lit(2, '1')),
                  dict(name='brake', when=['lit', True, 'bool'], option='brake', score=lit(2, '1'))]
    q = Policy(p, h)
    assert q.select(dict(gap=170, face=True, goal=False))['rule'] == 'push'
    r = q.check({'gap': [200, 200]}, 'orbit',
                [dict(updates={'gap': [170, 170], 'face': True, 'goal': False}, path_safe=True)])
    assert r['branches'][0]['possible_rules'] == ['push']


def test_no_coverage_and_path_safety_are_not_safety_proofs():
    p, h = example(); p['rules'] = p['rules'][:1]; q = Policy(p, h)
    r = q.check({'gap': [200, 200]}, 'orbit', [dict(updates={'face': False, 'goal': False}, path_safe=True)])
    assert r['status'] == 'counterexample'
    r = q.check({'gap': [200, 200]}, 'orbit', [dict(updates={'gap': [200, 200], 'face': True, 'goal': False})])
    assert r['status'] == 'unknown'


def test_terminal_is_separate_from_safe_stop():
    p, h = example(); q = Policy(p, h)
    assert q.select(dict(gap=0, face=False, goal=True))['terminal']
    r = q.check({'gap': [200, 200]}, 'orbit', [dict(updates={'gap': [0, 0], 'goal': True}, path_safe=True)], quality='model')
    assert r['status'] == 'proved'
    assert not r['branches'][0]['productive_possible']


def test_dependency_invalidation_and_literal_extraction():
    p, h = example(); q = Policy(p, h)
    assert q.dependencies(['gap'])['derived'] == ['margin']
    assert set(q.dependencies(['gap'])['requirements']) == {'push', 'orbit'}
    assert extract('SYMBOLIC_POLICY = '+repr(p)) == p
    with pytest.raises((ValueError, TypeError)): extract('SYMBOLIC_POLICY = make_policy()')


def test_editing_derived_definition_invalidates_downstream_rule():
    p,h=example();p['derived']['wide']=['ge',v('margin'),lit(0)]
    p['rules'][0]['when']=['and',v('face'),v('wide')]
    q=Policy(p,h);affected=q.dependencies(['margin'])
    assert affected['derived']==['margin','wide']
    assert affected['rules']==['push']
    assert affected['selection_competitors']==['push','orbit']
    assert affected['requirements']==[]  # physical gap preconditions did not change


def test_units_frames_and_empty_prediction_cannot_prove():
    p, h = example()
    bad = deepcopy(p); bad['derived']['bad'] = ['add', v('gap'), lit(2, 's')]
    with pytest.raises(ValueError): Policy(bad, h)
    h['options']['orbit']['frame'] = ['gap']
    with pytest.raises(ValueError, match='framed'): Policy(p, h)
    p, h = example(); q = Policy(p, h)
    with pytest.raises(ValueError, match='empty current'):
        q.check({'gap': [-20, -10]}, 'orbit', [dict(updates={}, path_safe=True)])
    assert q.check({'gap': [200, 200]}, 'orbit', [])['status'] == 'unknown'
