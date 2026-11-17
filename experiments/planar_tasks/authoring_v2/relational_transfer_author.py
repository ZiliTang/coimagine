"""Metered four-fixture clean native author profile; no fit/RL endpoints.

Use authoring_v2 shared skill knowledge. The native author's filesystem exclusion
is instruction-based; deployed authored code still runs in the existing sandbox.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,time,shutil
from .semantic_rl import SemanticRLCampaign,SemanticRLSession
from experiments.planar_tasks.relational_search.pipeline import Campaign,read,save,ROOT,identifier,unpack
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.typed_compiler import stream_config,CACHE

VERSION='relational-transfer-clean-author-v1'
GUIDES=['experiments/planar_tasks/authoring/option_guide.md','experiments/planar_tasks/relational_search/option_guide.md','experiments/planar_tasks/semantic_rl/ACTION_SCALE.md','experiments/planar_tasks/semantic_rl/STREAM_SEMANTICS.md','experiments/planar_tasks/semantic_rl/stream_example.py','experiments/planar_tasks/authoring_v2/README.md','experiments/planar_tasks/authoring_v2/experiment_api.md']


class TransferCampaign(SemanticRLCampaign):
    @classmethod
    def create(cls,path,fixtures,*,wall_seconds=5400):
        if set(fixtures)!={'dev_00','dev_01','dev_02','dev_03'}:raise ValueError('exactly four declared author fixtures')
        cfg=EventConfig(mode='duration')
        c=Campaign.create(path,fixture=fixtures['dev_00'],sequence=[dict(target_id=None,a0=44,a1=44,ticks=4)],limits=dict(execution_ticks=144000,resets=100,attempts=100,rounds=0,fits=0,author_turns=150),model=dict(provider='native-subagent',name='gpt-6.1-sol',reasoning_effort='high',clean_context=True,filesystem_isolation='instruction-only'))
        state=read(c.path/'campaign.json');state.update(profile=VERSION,deadline=time.time()+wall_seconds,tasks=list(fixtures),submissions=[],runs=[],semantic_fits=[],stage1=dict(feature_runtime='typed_stream',event_config=asdict(cfg),action_protocol=protocol(cfg),max_ticks=1800,max_decisions=300))
        state['limits']['feature_batches']=60000;state['used']['feature_batches']=0
        state['author_protocol']=dict(version=VERSION,only_declared_fixtures=True,no_prior_semantics=True,no_host_witnesses=True,offline_access='own fixtures, own truth replays, supplied simulator source',deployment='causal public history only',no_fit=True,no_rl=True,wall_seconds=wall_seconds,final='explicit FINAL.json; freeze then collect same-budget comparisons',feature_groups='D: measured/history descriptions; A: recommendations, policy phase/control state, route-stage/time schedule; missingness masked with values')
        for name,value in fixtures.items():
            scene,boxes=X.fixture(value);save(c.path/'tasks'/name/'fixture.json',value);save(c.path/'tasks'/name/'initial_public.json',X.RecordedEnv(scene,boxes).public())
        sources=read(c.path/'sources.json')
        for i,rel in enumerate(GUIDES):
            data=(ROOT/rel).read_bytes();(c.path/'knowledge'/f'transfer_{i}_{Path(rel).name}').write_bytes(data)
            dest=c.path/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);sources[rel]=hashlib.sha256(data).hexdigest()
        save(c.path/'sources.json',sources);state['source_hash']=digest(sources);save(c.path/'campaign.json',state);save(c.path/'knowledge/transfer_execution_protocol.json',dict(stage1=state['stage1'],limits=state['limits'],author_protocol=state['author_protocol']))
        return cls(path)

    def submit(self,source,summary,groups):
        with self.locked():
            self._check();cfg=stream_config(source)
            self._charge(feature_batches=2)
            f=self._features(source);h=f.new_history();h.append(read(self.path/'tasks'/self.state['tasks'][0]/'initial_public.json'))
            try:
                f.panel(h,ActionTree(h.events[-1]['observation']),{})
                if set(groups)!={'candidate','state'} or set(groups['candidate'])!=set(f.names) or set(groups['state'])!=set(cfg['state_names']):raise ValueError('groups must exactly name candidate and state features')
                if any(v not in ('D','A') for g in groups.values() for v in g.values()):raise ValueError('feature group must be D or A')
                sid=f's{len(self.state["submissions"]):03d}';p=self.path/'semantics';p.mkdir(exist_ok=True);(p/(sid+'.py')).write_text(source)
                item=dict(id=sid,sha256=digest(source),summary=summary,names=f.names,semantic_protocol=f.semantic_protocol,groups=groups,compilation=f.worker.compilation)
                save(p/(sid+'.build.json'),f.worker.compilation);shutil.copyfile(CACHE/f.worker.compilation['key']/'semantics.so',p/(sid+'.so'))
                self.state['submissions'].append(item);self._save();return item
            finally:f.close()

    def test(self,sid,task,**kwargs):
        if kwargs.get('fit_id') is not None:raise ValueError('no fitted policies in author development')
        if kwargs.get('max_decisions',300)>300:raise ValueError('max_decisions <=300')
        kwargs.setdefault('max_decisions',300)
        return super().test(sid,task,**kwargs)

    def trace(self,rid,start=0,count=3,truth=False):
        if type(start) is not int or start<0 or type(count) is not int or not 1<=count<=10:raise ValueError('trace page')
        r=unpack(self.path/'runs'/identifier(rid)/'result.json.gz')
        if truth:
            replay=r['replay'];rows=replay['frames']
            return dict(scope='privileged offline author only',scene=replay['scene'],total=len(rows),rows=rows[start:start+count])
        rows=r['public_rows'];return dict(total=len(rows),rows=rows[start:start+count])


class TransferSession(SemanticRLSession):
    def __init__(self,path):self.c=TransferCampaign(path)
    def tool(self,name,arguments):
        if name=='status':
            s=read(self.c.path/'campaign.json');return {k:s[k] for k in ['profile','model','limits','used','deadline','tasks','submissions','runs','author_protocol']}
        if name=='knowledge':return {p.name:p.read_text() for p in (self.c.path/'knowledge').glob('transfer_*')}
        methods=dict(brief=self.c.brief,submit=self.c.submit,test=self.c.test,detail=self.c.detail,trace=self.c.trace)
        if name not in methods:raise ValueError('unknown transfer endpoint')
        p=self.c.path/'author_tools'/f'{time.time_ns()}_{name}.json';save(p,dict(tool=name,arguments=arguments,status='running'))
        try:
            result=methods[name](**arguments);save(p,dict(tool=name,arguments=arguments,status='complete',result=result));return result
        except Exception as e:save(p,dict(tool=name,arguments=arguments,status='error',error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--request',required=True);a=p.parse_args();r=read(a.request);print(json.dumps(TransferSession(a.campaign).tool(r['name'],r.get('arguments',{})),ensure_ascii=False,allow_nan=False))
