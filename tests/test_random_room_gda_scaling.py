import pytest

from experiments.planar_tasks.authoring_v2 import random_room_gda_scaling as S


def test_frozen_cohort_sizes_and_no_overlap():
    groups = [S.panel(2026111751,'development',0), S.panel(2026111752,'development',1), S.panel(2026111753,'fresh_test')]
    assert [len(x) for x in groups] == [200,200,200]
    assert S.assert_disjoint(groups) == 600
    assert len({r['id'] for group in groups for r in group}) == 600
    for group in groups:
        assert all(sum(r['movable_count']==n for r in group)==50 for n in range(2,6))
    with pytest.raises(ValueError,match='duplicate fixture'):
        S.assert_disjoint([groups[0],groups[0]])


def test_success_selection_never_promotes_fast_failures():
    rows=[dict(case_id='a',condition='CAP-RR-002',success=True,ticks=30),
          dict(case_id='a',condition='CAP-RR-001',success=True,ticks=30),
          dict(case_id='a',condition='CAP-RR-003',success=False,ticks=1),
          dict(case_id='b',condition='CAP-RR-001',success=False,ticks=2)]
    assert S.select_successes(rows) == [rows[1]]


def test_conditions_have_matched_seed_budget_and_new_paths():
    assert S.conditions()==['control_s17','control_s29','control_s43','expanded_s17','expanded_s29','expanded_s43']
    assert S.model_path('expanded_s17').is_relative_to(S.BASE)
    assert not S.BASE.is_relative_to(S.OLD)
    assert 400*10+200*(10+len(S.conditions()))+100*len(S.conditions())==7800
