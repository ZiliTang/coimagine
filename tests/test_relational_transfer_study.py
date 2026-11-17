from collections import Counter
from pathlib import Path
import json
import pytest
from experiments.planar_tasks.authoring_v2 import relational_transfer_panels as P
from experiments.planar_tasks.authoring_v2 import relational_transfer_study as R


def test_panels_deterministic_disjoint_and_stratified():
    panels=P.build();assert panels==P.build();seen=set()
    for name,n in [('train',48),('iid',24),('transfer',48),('roots',8)]:
        rows=panels[name]['cases'];assert len(rows)==n
        assert Counter(r['relation'] for r in rows)==dict(blocked=n//2,clear=n//2)
        for r in rows:
            assert r['fixture_sha256']==P.digest(r['fixture'])
            assert r['fixture_sha256'] not in seen;seen.add(r['fixture_sha256'])
            assert P.relation(r['fixture'])['overlap']==(r['relation']=='blocked')
    assert {r['layout'] for r in panels['train']['cases']}=={r['layout'] for r in panels['iid']['cases']}
    assert not {r['layout'] for r in panels['train']['cases']}&{r['layout'] for r in panels['transfer']['cases']}


def test_comparison_is_paired_not_just_rate():
    rows=[dict(split='transfer',condition=c,case_id=str(i),layout='west',success=s) for c,vals in [('G',[True,False]),('GA',[False,True]),('DA',[True,False]),('GDA',[False,True])] for i,s in enumerate(vals)]
    p=R.summarize(rows)['paired']['transfer']['GA-G']
    assert p['net_success']==0 and p['gained']==['1'] and p['lost']==['0']
    with pytest.raises(ValueError,match='cases differ'):R.summarize(rows[:-1])


def test_missing_source_review_does_not_freeze_author(monkeypatch,tmp_path):
    monkeypatch.setattr(R,'HOST',tmp_path);monkeypatch.setattr(R,'final_manifest',lambda:({'feature_sid':'s006'},{'runs':[],'reservations':{}}))
    monkeypatch.setattr(R.TransferCampaign,'verify_sources',lambda self:None)
    def mutation(*args,**kwargs):raise AssertionError('must not mutate before review')
    monkeypatch.setattr(R.S,'save',mutation);monkeypatch.setattr(R.S,'finish',mutation)
    with pytest.raises(FileNotFoundError):R.prepare()
