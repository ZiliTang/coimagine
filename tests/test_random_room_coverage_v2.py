from copy import deepcopy
import pytest
from experiments.planar_tasks.authoring_v2.random_room_coverage_v2 import panel_rows, choose_cuts, prefix_audit
from experiments.planar_tasks.authoring_v2.random_room_tasks import generate
from experiments.planar_tasks.relational_search.contract import digest


def test_new_splits_are_disjoint_balanced_and_outcome_independent():
    a=panel_rows(2026111741,'newdev',200);b=panel_rows(2026111742,'fresh_test',100)
    old={digest(r['fixture']) for r in generate()['cases']}
    assert len(a)==200 and len(b)==100
    assert len({r['fixture_sha256'] for r in a+b})==300
    assert not old.intersection(r['fixture_sha256'] for r in a+b)
    for n in range(2,6):
        assert sum(r['movable_count']==n for r in a)==50
        assert sum(r['movable_count']==n for r in b)==25


def result(actions):
    return dict(public_rows=[dict(action=None,observation={'tick':0})]+[
        dict(action=a,observation={'tick':(i+1)*40}) for i,a in enumerate(actions)],
        replay={'frames':[{'tick':i,'x':float(i)} for i in range(len(actions)*40+1)]})


def test_cuts_keep_remaining_physical_budget_and_use_executed_prefix():
    a=result([1]*40);b=result([1]*3+[2]*37)
    assert choose_cuts(a,b)==dict(first_divergence=3,cuts=[3,11])
    b=result([1]*35+[2]*5)
    assert all(a['public_rows'][i]['observation']['tick']<=1200 for i in choose_cuts(a,b)['cuts'])


def test_handoff_audit_rejects_hidden_physics_drift_despite_same_public_input():
    ref=result([1]*10);actual=deepcopy(ref);actual['prefix_replay']={'executed':3}
    assert prefix_audit(actual,ref,3)['physics_equal']
    actual['replay']['frames'][80]['x']+=.001
    with pytest.raises(ValueError,match='physics prefix changed'):prefix_audit(actual,ref,3)
