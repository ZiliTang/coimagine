import json
from pathlib import Path
import pytest
from tools.priority_transfer import select, inside, bind_output


def row(case,teacher,success,ticks):
    return dict(case_id=case,teacher=teacher,success=success,ticks=ticks,split='development')


def test_global_priority_preserves_solvable_union_not_fastest_teacher():
    rows=[row('a','CAP1',True,10),row('a','CAP2',True,5),
          row('b','CAP1',True,8),row('b','CAP2',False,40),
          row('c','CAP1',False,40),row('c','CAP2',False,40)]
    s=select(rows)
    assert s['ranking'][0]==dict(teacher='CAP1',success=2,cases=3)
    assert [r['case_id'] for r in s['arms']['fastest']]==['a','b']
    assert [r['teacher'] for r in s['arms']['fastest']]==['CAP2','CAP1']
    assert [r['teacher'] for r in s['arms']['priority']]==['CAP1','CAP1']


def test_rank_ties_are_stable_and_failures_never_demonstrated():
    rows=[row('a','CAP2',True,2),row('a','CAP1',True,2)]
    for rs in [rows,rows[::-1]]:
        s=select(rs)
        assert s['arms']['priority'][0]['teacher']=='CAP1'
        assert s['arms']['fastest'][0]['teacher']=='CAP1'


def test_incomplete_duplicate_or_test_labels_rejected():
    good=[row('a','A',True,3),row('a','B',True,2)]
    for bad in [good+[good[0]],good+[row('b','A',True,3)],
                [dict(good[0],split='fresh_test'),good[1]]]:
        with pytest.raises(ValueError):select(bad)


def test_path_mapping_uses_destination_and_rejects_escape(tmp_path):
    new=tmp_path/'tzl'/'repo'/'coimagine'/'bundle';new.mkdir(parents=True)
    assert inside(new,'trajectories/a.json.gz')==new/'trajectories/a.json.gz'
    for rel in ['../outside','/home/zilitang/original']:
        with pytest.raises(ValueError):inside(new,rel)
    (new/'escape').symlink_to(tmp_path)
    with pytest.raises(ValueError):inside(new,'escape/file')


def test_output_cannot_reuse_another_bundle(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'manifest.json').write_text('{}')
    (bundle/'runtime_sources.json').write_text('{}')
    out=tmp_path/'work';bind_output(bundle,out);bind_output(bundle,out)
    (bundle/'manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):bind_output(bundle,out)
