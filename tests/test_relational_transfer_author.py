from pathlib import Path
import pytest
from experiments.planar_tasks.authoring_v2.relational_transfer_author import TransferCampaign,TransferSession
from experiments.planar_tasks.relational_search.transfer_tasks import cases
from experiments.planar_tasks.relational_search.pipeline import read

@pytest.fixture
def session(tmp_path):
    fs={f'dev_{i:02d}':c['fixture'] for i,c in enumerate(cases()[:4])}
    c=TransferCampaign.create(tmp_path/'campaign',fs,wall_seconds=120)
    return TransferSession(c.path)


def neutral():
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
    groups=dict(candidate={'duration_match':'A'},state={'contact_fraction':'D','blocked_fraction':'D','observed_entities':'D'})
    return dict(source=source,summary='neutral interface test',groups=groups)


def test_development_allowlist_and_disabled_training(session):
    s=session.tool('status',{});assert set(s['tasks'])=={'dev_00','dev_01','dev_02','dev_03'}
    assert s['limits']['fits']==0
    for op in ['fit','run_rl','reset','probe']:
        with pytest.raises(ValueError,match='unknown'):session.tool(op,{})
    with pytest.raises(ValueError,match='undeclared'):session.tool('test',dict(sid='s000',task='west_blocked'))
    with pytest.raises(ValueError,match='fitted'):session.tool('test',dict(sid='s000',task='dev_00',fit_id='f000'))


def test_submit_groups_metered_execution_and_trace(session):
    bad=neutral();bad['groups']['state']={}
    with pytest.raises(ValueError,match='groups'):session.tool('submit',bad)
    submission=session.tool('submit',neutral());assert submission['id']=='s000'
    r=session.tool('test',dict(sid='s000',task='dev_00',max_ticks=4,max_decisions=1))
    assert r['ticks']==4 and r['error'] is None
    s=session.tool('status',{});assert s['used']['attempts']==s['used']['resets']==1 and s['used']['execution_ticks']==4
    assert not read(session.c.path/'campaign.json')['reservations']
    assert session.tool('trace',dict(rid=r['id'],start=0,count=1))['rows'][0]['observation']['tick']==0
    assert session.tool('trace',dict(rid=r['id'],start=0,count=1,truth=True))['scope']=='privileged offline author only'
    session.c.verify_sources()
