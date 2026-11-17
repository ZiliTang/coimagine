"""Corrected author/deployment boundary, finite search and real feedback wiring."""
from copy import deepcopy
from itertools import islice
import pytest
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search import pipeline_semantics as F
from experiments.planar_tasks.relational_search import live_search as L
from experiments.planar_tasks.relational_search import live_execution as E
from experiments.planar_tasks.relational_search.live_scorer import FittedScorer
from experiments.planar_tasks.relational_search.contract import History,action_space,canonical
from experiments.planar_tasks.relational_search.pipeline import read,save,unpack
from experiments.planar_tasks.authoring_v2.relational_live import LiveCampaign,LiveSession

BRAKE=dict(target_id=None,a0=44,a1=44,ticks=4)
SOURCE="""NAMES=['duration']
def initial_memory():return {}
def update_memory(memory,event):return memory
def semantic_features(memory,observation,action):return {'duration':action['ticks']}
"""

def history():
    s,b=X.fixture(X.seed_fixture());h=History();h.append(X.RecordedEnv(s,b).public());return h

class Model:
    root=0
    def __init__(self,h,**kwargs):self.states={0:h.events[-1]['observation']};self.roots=[self.states[0]['tick']]
    def cost_ticks(self,handle,a):return a['ticks']
    def predict(self,handle,a):
        o=deepcopy(self.states[handle]);o['tick']+=a['ticks']
        for e in o['entities']:e['last_seen_tick']=o['tick']
        k=len(self.states);self.states[k]=o
        return dict(observation=o,ticks=a['ticks'],cost_seconds=a['ticks']*.05,terminal=False,success=False,handle=k)
    def record(self):return dict(root_tick=self.states[0]['tick'])
    def close(self):pass


def test_candidate_set_same_and_scoring_changes_expansion():
    h=history();o=h.events[-1]['observation']
    assert {canonical(a) for a in L.candidates(o)}=={canonical(a) for a in action_space(o)}
    # Exactly one simulated command: changing rank changes the selected action.
    b=L.Budget(calls=1,prediction_ticks=40,batch=72)
    short=L.choose(h,Model(h),lambda h,aa:[-a['ticks'] for a in aa],b)
    long=L.choose(h,Model(h),lambda h,aa:[a['ticks'] for a in aa],b)
    assert short['action']['ticks']==4 and long['action']['ticks']==40
    assert short['trace'][0]['requested_ticks']==short['trace'][0]['actual_ticks']==4
    assert long['selection_horizon_ticks']==40


def test_shorter_prefix_cannot_win_by_cheaper_elapsed_cost(monkeypatch):
    monkeypatch.setattr(L,'candidates',lambda o:iter([dict(BRAKE,ticks=40),BRAKE]))
    h=history();r=L.choose(h,Model(h),lambda h,aa:[a['ticks'] for a in aa],L.Budget(calls=3,prediction_ticks=120,horizon_ticks=120,batch=2))
    assert r['selection_horizon_ticks']==120 and len(r['selected_path'])==3
    class Short(Model):
        def predict(self,handle,a):
            out=super().predict(handle,a);out['ticks']=1;return out
    bad=L.choose(h,Short(h),lambda h,aa:[0 for a in aa],L.Budget(calls=1))
    assert bad['action'] is None and 'shortened' in bad['trace'][0]['error']


def test_batch_scorer_matches_training_and_strips_hidden_input():
    from experiments.planar_tasks.relational_search.evidence import Outcome
    from experiments.planar_tasks.relational_search.dataset import record
    h=history();a=dict(BRAKE,ticks=8)
    rows=[record('synthetic','train',h,Outcome(h.sha256,a,'fixture',{},'execution','success',1.),Outcome(h.sha256,BRAKE,'fixture',{},'execution','task_failure',180.))]
    fitted=F.fit_models(rows,SOURCE)
    for condition in ('S','E','M'):
        scorer=FittedScorer(SOURCE,fitted,condition)
        assert scorer(h,[a,BRAKE])==pytest.approx(F.score_candidates(SOURCE,fitted,h,[a,BRAKE],condition))
        assert scorer(h,[BRAKE,a])==pytest.approx(scorer(h,[a,BRAKE])[::-1])
    # Runtime schema strips hidden fields as well as the training worker.
    h.events[-1]['observation']['truth_mass']=400
    hidden=SOURCE.replace("action['ticks']","observation.get('truth_mass', 4)")
    from experiments.planar_tasks.relational_search.contract import digest
    fitted['semantic_sha256']=digest(hidden)
    assert FittedScorer(hidden,fitted,'S')(h,[a,BRAKE])[0]==FittedScorer(hidden,fitted,'S')(h,[a,BRAKE])[1]


def test_real_execution_replans_from_feedback_not_predicted_suffix():
    roots=[]
    class Tracking(Model):
        def __init__(self,h,**kw):super().__init__(h,**kw);roots.append(h.events[-1]['observation']['tick'])
    def scores(source,fitted,condition):
        def score(h,aa):
            return [100 if a['target_id'] is None and a['a0']==44 and a['a1']==44 and a['ticks']==4 else -a['ticks'] for a in aa]
        return score
    r=E.run(X.seed_fixture(),'',{},max_ticks=8,max_decisions=2,budget=L.Budget(calls=1,batch=72),model_factory=Tracking,score_factory=scores)
    assert roots==[0,4] and [d['real_tick'] for d in r['decisions']]==[0,4]
    assert r['summary']['ticks']==8 and len(r['commands'])==2
    assert [row['observation']['tick'] for row in r['public_rows']]==[0,4,8]


def test_live_author_truth_deployment_and_budget_accounting(tmp_path,monkeypatch):
    c=LiveCampaign.create(tmp_path/'live',sequence=[BRAKE]);s=LiveSession(c.path)
    s.tool('new_round',{})
    ctx=s.tool('author_context',{'rid':'r000'})
    assert 'fixture' in ctx and 'scene' in ctx
    assert 'fixture' not in s.tool('context',{'rid':'r000'})
    knowledge=s.tool('knowledge',{})
    assert set(knowledge)=={'0_option_guide.md','3_option_guide.md','live_guide.md'}
    with pytest.raises(ValueError):s.tool('test_plan',{})
    t=s.tool('brief',dict(rid='r000',turn=dict(actor='engineering',summary='接线测试',output={})))
    save(c.path/'fits/f000.json',{'semantic_id':'s000'});(c.path/'semantics').mkdir();(c.path/'semantics/s000.py').write_text('')
    import experiments.planar_tasks.authoring_v2.relational_live as module
    def fake(value,source,fitted,**kw):
        kw['on_prediction']('reserve',600,12);kw['on_prediction']('settle',40,1)
        r=X.execute(value,[BRAKE]);r.update(commands=[BRAKE],decisions=[]);return r
    monkeypatch.setattr(module,'run',fake)
    a=s.tool('test_closed_loop',dict(rid='r000',fit_id='f000',turn_id=t['turn_id'],max_ticks=40))
    state=read(c.path/'campaign.json')
    assert state['used']['execution_ticks']==4 and state['used']['prediction_ticks']==40 and state['used']['prediction_calls']==1
    assert not state['reservations']
    assert read(c._apath('r000',a['id'])/'request.json')['commands']==[BRAKE]
    assert s.tool('decisions',dict(rid='r000',attempt_id=a['id']))['total']==0
