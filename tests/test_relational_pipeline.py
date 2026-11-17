"""Real-physics pipeline boundaries, evidence scope, isolated features and viewer."""
from copy import deepcopy
import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest

from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search import pipeline_semantics as F
from experiments.planar_tasks.relational_search.pipeline import Campaign, read, save, unpack, perturb
from experiments.planar_tasks.relational_search.pipeline_server import handler_for
from experiments.planar_tasks.authoring_v2.relational import RelationalSession


BRAKE=dict(target_id=None,a0=44,a1=44,ticks=4)
SOURCE="""NAMES=['duration']
def initial_memory():return {}
def update_memory(memory,event):return memory
def semantic_features(memory,observation,action):return {'duration':action['ticks']}
"""


@pytest.fixture
def campaign(tmp_path):
    c=Campaign.create(tmp_path/'case',sequence=[BRAKE,BRAKE])
    c.new_round()
    return c


def test_real_replay_causal_trace_and_public_author_boundary(campaign):
    c=campaign;a=c.replay('r000')
    assert a['ticks']==8 and a['task_cost_seconds']==180 and a['status']=='task_failure'
    rows=unpack(c._apath('r000',a['id'])/'public.json.gz')
    assert [x['observation']['tick'] for x in rows]==[0,4,8]
    h=X.history_at(rows,1);assert h.events[-1]['observation']['tick']==4
    context=c.context('r000',attempt_id=a['id'],boundary=1)
    assert context['history_hash']==h.sha256
    assert not {'fixture','replay','scene','final_audit','delta'}&set(context)
    assert 'mass' not in json.dumps(context)
    state=read(c.path/'campaign.json')
    assert state['used']['execution_ticks']==8 and not state['reservations']
    assert state['used']['resets']==2


def test_illegal_action_kept_unknown_not_negative(campaign):
    c=campaign;r=c._round('r000')
    r['original_commands']=[BRAKE,dict(target_id='hidden',a0=2,a1=2,ticks=4)]
    save(c._rpath('r000')/'round.json',r)
    a=c.replay('r000')
    assert a['status']=='tool_error' and a['task_cost_seconds'] is None
    assert a['detected_boundary']==1 and a['ticks']==4
    assert len(unpack(c._apath('r000',a['id'])/'replay.json.gz')['frames'])==5


def test_repair_preserves_exact_prefix_and_does_not_create_preferences(campaign):
    c=campaign;a=c.replay('r000')
    t=c.author_turn('r000',dict(actor='engineering',summary='接线检查：延长停稳后缀。',output={}))
    repaired=c.repair('r000',attempt_id=a['id'],boundary=1,suffix=[dict(BRAKE,ticks=8)],turn_id=t['turn_id'])
    assert repaired['ticks']==12 and repaired['metadata']['repair_boundary']==1
    assert 'not a local action preference' in repaired['metadata']['supervision_scope']
    assert c._round('r000')['comparisons']==[]
    with pytest.raises(ValueError):c.repair('r000',attempt_id=a['id'],boundary=99,suffix=[BRAKE],turn_id=t['turn_id'])


def test_same_prefix_comparisons_keep_ties_and_charge_replays(campaign):
    c=campaign;a=c.replay('r000')
    comp=c.compare('r000',attempt_id=a['id'],boundary=1,
                   candidates=[BRAKE,dict(BRAKE,ticks=8)],continuation=[BRAKE])
    assert comp['preference_pairs']==0
    assert all(x['prefix_verified'] for x in comp['branches'])
    # First run 8 + (prefix4 + candidate4 + continuation4) + (4+8+4).
    assert read(c.path/'campaign.json')['used']['execution_ticks']==36
    rows=[json.loads(l) for l in (c._rpath('r000')/'comparisons/c000.jsonl').read_text().splitlines()]
    assert rows[0]['comparison']['status']=='tie'
    with pytest.raises(ValueError,match='no comparable'):F.fit_models(rows,SOURCE)


def test_reservation_survives_host_failure_and_audit(campaign,monkeypatch):
    def fail(*a,**k):raise RuntimeError('simulated host interruption')
    monkeypatch.setattr(X,'execute',fail)
    with pytest.raises(RuntimeError):campaign.replay('r000')
    recovered=Campaign(campaign.path);audit=recovered.audit()
    assert audit['used']['execution_ticks']==8 and audit['reservations']=={'r000/a000':8}
    assert recovered._round('r000')['attempts'][0]['reservation_retained']


def test_atomic_budget_rejection_and_unknown_tool(campaign):
    c=campaign;s=read(c.path/'campaign.json');s['limits']['execution_ticks']=7;save(c.path/'campaign.json',s)
    with pytest.raises(RuntimeError):c.replay('r000')
    assert read(c.path/'campaign.json')['used']==s['used']
    assert not c._round('r000')['attempts']
    with pytest.raises(ValueError):RelationalSession(c.path).tool('truth',{})


def test_perturbation_group_and_parent_require_success(campaign):
    c=campaign
    c.new_round({'entity-b.x':.1,'robot1.x':.1})
    assert c._round('r000')['group']==c._round('r001')['group']
    assert c._round('r000')['split']==c._round('r001')['split']=='train'
    with pytest.raises(ValueError):c.new_round({'entity-b.x':999})
    with pytest.raises(ValueError):c.new_round(parent='r000')


def test_disjoint_aligned_circle_polygon_has_no_false_contact():
    # R1 x=3.1 aligns with the distant horizontal obstacle's left edge.
    # Raw narrow-phase queries previously returned penetration .18 m.
    f=perturb(X.seed_fixture(),{'entity-b.x':.1,'robot1.x':.1},.35)
    s,boxes=X.fixture(f)
    from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv
    env=MultiObjectOptionEnv(s,boxes)
    assert not env.env.backend.contacts()
    assert max(env.env.backend.penetration_by_pair().values())==0


def test_isolated_semantic_worker_cannot_read_repo_or_inherit_truth(campaign):
    c=campaign;h=__import__('experiments.planar_tasks.relational_search.contract',fromlist=['History']).History()
    h.append(c.context('r000')['observation'])
    assert F.evaluate(SOURCE,[(h,BRAKE)])['values']==[[4]]
    source=SOURCE.replace("{'duration':action['ticks']}","{'duration':__import__('os').path.exists('/home/zilitang/repo/coimagine')}")
    assert F.evaluate(source,[(h,BRAKE)])['values']==[[False]]
    malicious=deepcopy(h);malicious.events[-1]['observation']['truth_mass']=400
    source=SOURCE.replace("{'duration':action['ticks']}","{'duration':observation.get('truth_mass')}")
    assert F.evaluate(source,[(malicious,BRAKE)])['values']==[[None]]
    # No positive/negative example ordering or other histories in the process.
    source=SOURCE.replace("{'duration':action['ticks']}","{'duration':len(request)}")
    assert F.evaluate(source,[(h,BRAKE),(h,dict(BRAKE,ticks=8))])['values']==[[1],[1]]


def test_semantic_versions_are_immutable_and_refit_rejects_no_data(campaign):
    c=campaign;t=c.author_turn('r000',dict(actor='engineering',summary='测试特征。',output={}))
    s=c.submit_semantics('r000',source=SOURCE,turn_id=t['turn_id'])
    s2=c.submit_semantics('r000',source=SOURCE,turn_id=t['turn_id'])
    assert s['id']!=s2['id']
    with pytest.raises(ValueError):c.fit('r000',s['id'])
    assert read(c.path/'fits/f000.json')['status']=='error'


def test_fit_same_dataset_three_conditions_and_train_only_normalization(campaign):
    # Synthetic labels test fitting, not physical success. Histories are real
    # observations; held-out durations must not change the training scaler.
    from experiments.planar_tasks.relational_search.contract import History
    from experiments.planar_tasks.relational_search.evidence import Outcome
    from experiments.planar_tasks.relational_search.dataset import record
    h=History();h.append(campaign.context('r000')['observation'])
    def pair(group,split,ticks):
        a=dict(BRAKE,ticks=ticks)
        return record(group,split,h,Outcome(h.sha256,a,'fixture',{},'execution','success',1.),
                      Outcome(h.sha256,BRAKE,'fixture',{},'execution','task_failure',180.))
    data=[pair('train','train',8),pair('validation','validation',40)]
    f=F.fit_models(data,SOURCE)
    assert f['training_pairs']==1 and f['pair_count']==2
    assert f['models']['S']['checkpoint']['mean']==[6.]
    assert all(m['metrics']['train']['count']==1 for m in f['models'].values())
    actions=[BRAKE,dict(BRAKE,ticks=8)]
    score=F.score_candidates(SOURCE,f,h,actions,'S');assert score[1]>score[0]
    assert f['mother_groups_by_split']['validation']==['validation']
    with pytest.raises(ValueError,match='group crosses'):
        F.fit_models([pair('same','train',8),pair('same','test',40)],SOURCE)


def test_source_change_refused_and_frozen_extra_file_detected(campaign,monkeypatch):
    c=campaign
    sources=read(c.path/'sources.json');original=deepcopy(sources);key=next(iter(sources));sources[key]='bad'
    save(c.path/'sources.json',sources)
    with pytest.raises(ValueError,match='source changed'):c.replay('r000')
    with pytest.raises(ValueError):c.freeze()
    save(c.path/'sources.json',original)
    # Historical verification checks copied source, independently of worktree.
    c.freeze();(c.path/'extra.json').write_text('{}')
    with pytest.raises(ValueError,match='inventory'):c.verify_frozen()


@pytest.mark.parametrize('field',['empty_sequence','nan_goal'])
def test_creation_rejects_empty_or_nonfinite_configuration(tmp_path,field):
    with pytest.raises(ValueError):
        if field=='empty_sequence':Campaign.create(tmp_path/'bad',sequence=[])
        else:
            f=X.seed_fixture();f['goal']['x']=float('nan');Campaign.create(tmp_path/'bad',fixture=f)


def test_initial_observation_plan_submission_isolated_and_replayable(campaign):
    c=campaign
    t=c.author_turn('r000',dict(actor='llm',model=dict(name='test-double'),summary='测试同一初始观测函数。',output={}))
    source="def make_sequence(observation,seed_commands):\n    assert 'fixture' not in observation\n    assert not __import__('os').path.exists('/home/zilitang/repo/coimagine')\n    return seed_commands\n"
    a=c.test_plan('r000',source=source,turn_id=t['turn_id'])
    assert a['ticks']==8 and a['kind']=='generated_plan'
    assert (c.path/'plans'/(a['metadata']['plan_sha256']+'.py')).read_text()==source
    assert 'no online replanning' in a['metadata']['scope']


def test_native_model_usage_not_reported_as_zero_calls(tmp_path):
    c=Campaign.create(tmp_path/'native',sequence=[BRAKE],model=dict(provider='codex_native',name='test-double'))
    assert read(c.path/'campaign.json')['llm_usage']['requests'] is None


def test_freeze_detects_evidence_edits_and_denies_new_work(campaign):
    c=campaign;c.replay('r000');assert c.freeze()['status']=='verified'
    with pytest.raises(ValueError):c.new_round()
    save(c.path/'fixture.json',{})
    with pytest.raises(ValueError):c.verify_frozen()


def test_human_server_readonly_replay_and_trace_pagination(campaign):
    c=campaign;c.replay('r000')
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(c.path.parent))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        def get(p):return json.load(urlopen(base+p))
        assert get('/api/campaigns')[0]['id']=='case'
        prefix='/api/campaigns/case/rounds/r000/attempts/a000/'
        assert len(get(prefix+'replay')['frames'])==9
        assert len(get(prefix+'trace?start=1&count=1')['rows'])==1
        with pytest.raises(HTTPError) as ex:urlopen(Request(base+'/api/campaigns',data=b'{}'))
        assert ex.value.code==405
        for url in ('/api/campaigns/..','/api/campaigns/case/rounds/../attempts/a000/replay',prefix+'trace?count=99'):
            with pytest.raises(HTTPError):urlopen(base+url)
        assert 'Research Pipeline' in urlopen(base+'/').read().decode()
    finally:server.shutdown();server.server_close();thread.join()
