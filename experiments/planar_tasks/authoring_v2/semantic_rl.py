"""Metered first-stage native author experiment; immutable source snapshots."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import time
import torch
from experiments.planar_tasks.relational_search.pipeline import Campaign, read, save, pack, unpack, identifier, ROOT
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import digest, History
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.learning import demonstration_traces, warm_start, ppo_update
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.action_semantics.program import Program


FIXTURE_PROFILES = ('stage1-v1', 'stage1-wall-anchor-v2')


def development_fixtures(profile='stage1-v1'):
    if profile not in FIXTURE_PROFILES:
        raise ValueError('unknown semantic RL fixture profile')
    clear=X.seed_fixture()
    free=deepcopy(clear);free['movable_objects']=[]
    # A longitudinal baseline plus a different contact geometry at the wall.
    wall=deepcopy(free);wall['object'].update(x=.125,y=2.,angle=0.)
    wall['robots']=[dict(x=.38,y=1.22,angle=0.),dict(x=.38,y=2.78,angle=0.)]
    if profile == 'stage1-wall-anchor-v2':
        anchor = X.T.task_catalog()['planar_wall_anchor'].instance
        wall = dict(scene_id='planar_wall_anchor', object=asdict(anchor.object_pose),
                    robots=[asdict(p) for p in anchor.robot_poses],
                    goal=dict(x=anchor.goal_centre[0], y=anchor.goal_centre[1], radius=anchor.goal_radius),
                    movable_objects=[], overrides={})
    return dict(open_transport=free,wall_release=wall,clear_transport=clear)


class SemanticRLCampaign(Campaign):
    @classmethod
    def create(cls,path,*,fixture_profile='stage1-v1',model=None,execution_mode='progress',feature_runtime='typed_stream'):
        event_config=EventConfig(mode=execution_mode)
        if feature_runtime not in ("legacy","cached_fork","typed_batch","typed_stream"):raise ValueError("feature runtime")
        fixtures = development_fixtures(fixture_profile)
        c=super().create(path,limits=dict(execution_ticks=50000,resets=70,attempts=70,rounds=3,fits=3,author_turns=100),
                        model=model or dict(provider='native',name='gpt-6.1-sol',reasoning_effort='high'))
        s=read(c.path/'campaign.json')
        s.update(profile='semantic-rl-stage1-v1',fixture_profile=fixture_profile,deadline=time.time()+5400,
                 tasks=list(fixtures),submissions=[],runs=[],semantic_fits=[],
                 stage1=dict(condition='M',supervised_epochs=200,seed=17,rl_episodes=6,
                             max_ticks=1800,max_decisions=900,episode_wall_seconds=420,
                             event_config=asdict(event_config),feature_runtime=feature_runtime,
                             action_protocol=protocol(event_config) if execution_mode=="duration" else None,nonrecommended_penalty=0.))
        s['limits']['feature_batches']=30000;s['used']['feature_batches']=0
        save(c.path/'campaign.json',s)
        for name,fixture in fixtures.items():
            X.fixture(fixture)
            save(c.path/'tasks'/name/'fixture.json',fixture)
            scene,boxes=X.fixture(fixture)
            save(c.path/'tasks'/name/'initial_public.json',X.RecordedEnv(scene,boxes).public())
        guides=['experiments/planar_tasks/authoring/option_guide.md',
                'experiments/planar_tasks/relational_search/option_guide.md',
                'experiments/planar_tasks/semantic_rl/README.md',
                'experiments/planar_tasks/semantic_rl/ACTION_SCALE.md',
                'experiments/planar_tasks/authoring_v2/semantic_rl_guide.md']
        if feature_runtime=='typed_batch':
            guides += ['experiments/planar_tasks/semantic_rl/TYPED_SEMANTICS.md',
                       'experiments/planar_tasks/semantic_rl/typed_example.py']
        if feature_runtime=='typed_stream':
            guides += ['experiments/planar_tasks/semantic_rl/STREAM_SEMANTICS.md',
                       'experiments/planar_tasks/semantic_rl/stream_example.py']
        sources=read(c.path/'sources.json')
        for i,rel in enumerate(guides):
            data=(ROOT/rel).read_bytes()
            (c.path/'knowledge'/f'stage1_{i}_{Path(rel).name}').write_bytes(data)
            target=c.path/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            sources[rel]=hashlib.sha256(data).hexdigest()
        save(c.path/'sources.json',sources);s['source_hash']=digest(sources);save(c.path/'campaign.json',s)
        save(c.path/'knowledge/stage1_execution_protocol.json',s['stage1'])
        return cls(path)

    def _event_config(self):
        return EventConfig(**self.state['stage1']['event_config'])

    def _features(self,source):
        config=self.state['stage1']
        if config.get('feature_runtime')=='typed_stream':
            from experiments.planar_tasks.semantic_rl.typed_compiler import stream_config
            stream_config(source)
        return Features(source,fast_embedding=True,runtime=config.get('feature_runtime','legacy'),
                        action_protocol=config.get('action_protocol'))

    def _check(self):
        self.verify_sources()
        if time.time()>self.state['deadline']:raise RuntimeError('campaign wall budget exhausted')

    def brief(self,summary,output=None):
        with self.locked():
            self._check();self._charge(author_turns=1)
            index=self.state['used']['author_turns']
            save(self.path/'briefs'/f'{index:03d}.json',dict(summary=summary,output=output,at=time.time()))
            return dict(id=index)

    def submit(self,source,summary):
        with self.locked():
            self._check()
            if self.state['stage1'].get('feature_runtime') not in ('typed_batch','typed_stream') and 'def prefix_semantic_features' not in source:
                raise ValueError('explicit prefix semantics required')
            sid=f's{len(self.state["submissions"]):03d}'
            # Validate isolated prefix schema, no task execution or success claim.
            streaming=self.state['stage1'].get('feature_runtime')=='typed_stream'
            self._charge(feature_batches=2 if streaming else 1)
            f=self._features(source)
            h=f.new_history();h.append(read(self.path/'tasks/open_transport/initial_public.json'))
            try:f.panel(h,ActionTree(h.events[-1]['observation']),{})
            finally:f.close()
            (self.path/'semantics').mkdir(exist_ok=True);(self.path/'semantics'/f'{sid}.py').write_text(source)
            item=dict(id=sid,sha256=digest(source),summary=summary,names=f.names)
            if f.semantic_protocol is not None:item['semantic_protocol']=f.semantic_protocol
            if f.runtime in ('typed_batch','typed_stream'):
                from experiments.planar_tasks.semantic_rl.typed_compiler import CACHE
                import shutil
                item['compilation']=f.worker.compilation
                save(self.path/'semantics'/f'{sid}.build.json',f.worker.compilation)
                shutil.copyfile(CACHE/f.worker.compilation['key']/'semantics.so',self.path/'semantics'/f'{sid}.so')
            self.state['submissions'].append(item);self._save();return item

    def _source(self,sid):return (self.path/'semantics'/f'{identifier(sid)}.py').read_text()

    def test(self,sid,task,fit_id=None,max_ticks=1800,max_decisions=900,deterministic=True,seed=17,phase='author'):
        with self.locked():
            self._check()
            if task not in self.state['tasks']:raise ValueError('undeclared task')
            if not 1<=max_ticks<=1800 or not 1<=max_decisions<=900:raise ValueError('episode cap')
            source=self._source(sid);f=self._features(source);model=None
            if fit_id is not None:
                model,meta=Policy.load(self.path/'semantic_fits'/identifier(fit_id)/'policy.pt',f.sha256)
            rid=f'a{len(self.state["runs"]):03d}';p=self.path/'runs'/rid
            reserve_features=((5 if f.runtime=='typed_stream' else 4) if model else 1)*max_decisions+((2 if f.runtime=='typed_stream' else 1) if model else 0)
            self._charge(attempts=1,resets=1,execution_ticks=max_ticks,feature_batches=reserve_features)
            self.state['reservations'][rid]=dict(ticks=max_ticks,features=reserve_features)
            self.state['runs'].append(dict(id=rid,sid=sid,task=task,fit_id=fit_id,status='running',phase=phase))
            self._save();p.mkdir(parents=True)
            save(p/'request.json',dict(sid=sid,task=task,fit_id=fit_id,max_ticks=max_ticks,max_decisions=max_decisions,seed=seed,deterministic=deterministic,phase=phase))
            result,steps=rollout(read(self.path/'tasks'/task/'fixture.json'),source,model=model,features=f,
                    max_ticks=max_ticks,max_decisions=max_decisions,wall_seconds=min(420,max(1,self.state['deadline']-time.time())),
                    seed=seed,deterministic=deterministic,config=self._event_config(),fast_feedback=True,
                    on_progress=lambda v:save(p/'progress.json',dict(v,updated=time.time())))
            f.close()
            result.update(split='train',task=task,source_id=sid)
            pack(p/'result.json.gz',result)
            if steps:pack(p/'batch.json.gz',steps)
            save(p/'summary.json',result['summary'])
            # CAP uses one feature worker per attempted decision, including a final error.
            actual_features=f.calls if model or f.runtime in ('typed_batch','typed_stream') else result['summary']['decisions']+int(bool(result['summary']['error']))
            self.state['used']['execution_ticks']-=max_ticks-result['summary']['ticks']
            self.state['used']['feature_batches']-=reserve_features-actual_features
            del self.state['reservations'][rid]
            self.state['runs'][-1].update(result['summary']);self._save()
            return dict(id=rid,task=task,fit_id=fit_id,**result['summary'])

    def fit_semantics(self,sid,run_ids):
        with self.locked():
            self._check();source=self._source(sid)
            if len(run_ids)!=len(set(run_ids)):raise ValueError('duplicate demonstrations')
            examples=[];seen=set()
            for rid in run_ids:
                r=unpack(self.path/'runs'/identifier(rid)/'result.json.gz')
                if r['summary']['status']!='success' or r['split']!='train':raise ValueError('successful train demos required')
                # No truncation/subsampling; all completed boundaries enter.
                examples.append(r);seen.add(r['task'])
            if seen!=set(self.state['tasks']):raise ValueError('successful demonstration required for every declared task')
            reserve=(5 if self.state['stage1'].get('feature_runtime')=='typed_stream' else 4)*sum(len(r['public_rows'])-1 for r in examples)
            self._charge(fits=1,feature_batches=reserve)
            fid=f'f{len(self.state["semantic_fits"]):03d}';p=self.path/'semantic_fits'/fid;p.mkdir(parents=True)
            self.state['semantic_fits'].append(dict(id=fid,sid=sid,run_ids=run_ids,status='running'))
            self.state['reservations'][fid]=dict(features=reserve);self._save()
            features=self._features(source);traces=[]
            for r in examples:
                traces.extend(demonstration_traces(r,features,split='train',mode=self._event_config().mode))
                save(p/'progress.json',dict(stage='features',histories=len(traces),feature_batches=features.calls,updated=time.time()))
            pack(p/'dataset.json.gz',traces)
            model,curve=warm_start(traces,features,epochs=200,condition='M',seed=17,
                                   on_epoch=lambda row:save(p/'progress.json',dict(stage='fit',**row,updated=time.time())))
            correct=0;factor_correct=0
            with torch.no_grad():
                for tr in traces:
                    match=[int(model.distribution(panel)[0].logits.argmax())==i for panel,i in zip(tr['panels'],tr['indices'])]
                    correct+=int(all(match));factor_correct+=sum(match)
            model.save(p/'policy.pt',dict(sid=sid,run_ids=run_ids,event_config=asdict(self._event_config()),epochs=200,condition='M'))
            save(p/'curve.json',curve)
            result=dict(id=fid,sid=sid,histories=len(traces),exact_teacher_match=correct,factor_match=factor_correct,
                        final_loss=curve[-1],status='complete')
            features.close()
            save(p/'summary.json',result)
            self.state['used']['feature_batches']-=reserve-features.calls
            del self.state['reservations'][fid];self.state['semantic_fits'][-1].update(result);self._save()
            return result

    def detail(self,rid,boundary=None):
        r=unpack(self.path/'runs'/identifier(rid)/'result.json.gz')
        if boundary is None:
            return dict(summary=r['summary'],last_decisions=r['decisions'][-5:],public_boundaries=len(r['public_rows']),
                        result_path=str(self.path/'runs'/rid/'result.json.gz'))
        return r['public_rows'][boundary]

    def run_rl(self,sid,fid):
        # Only after the SAME frozen fit succeeds on all declared tasks.
        state=read(self.path/'campaign.json')
        passed={r['task'] for r in state['runs'] if r['sid']==sid and r.get('fit_id')==fid and r.get('success')}
        if passed!=set(state['tasks']):raise ValueError('supervised recovery gate not passed')
        source=self._source(sid);model,meta=Policy.load(self.path/'semantic_fits'/fid/'policy.pt',digest(source))
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        # Save each policy version; existing test() provides metering + raw batches.
        previous=fid;results=[]
        for episode in range(6):
            task=state['tasks'][episode%len(state['tasks'])]
            run=self.test(sid,task,fit_id=previous,deterministic=False,seed=117+episode,phase='rl')
            r=unpack(self.path/'runs'/run['id']/'result.json.gz')
            steps=unpack(self.path/'runs'/run['id']/'batch.json.gz') if (self.path/'runs'/run['id']/'batch.json.gz').exists() else []
            update=ppo_update(model,optimizer,steps,bootstrap=r['summary']['bootstrap']) if steps and r['summary']['eligible_for_ppo'] else dict(updates=0)
            previous=f'rl{episode:03d}';p=self.path/'semantic_fits'/previous;p.mkdir()
            model.save(p/'policy.pt',dict(meta,rl_episode=episode));save(p/'update.json',update)
            torch.save(dict(optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),p/'optimizer.pt')
            results.append(dict(run=run,update=update,fit_id=previous));save(self.path/'rl_progress.json',results)
        evaluations=[self.test(sid,t,fit_id=previous,phase='post_rl') for t in state['tasks']]
        return dict(episodes=results,post_rl=evaluations,final_fit_id=previous)


class SemanticRLSession:
    def __init__(self,path):self.c=SemanticRLCampaign(path)
    def tool(self,name,arguments):
        if name=='status':return read(self.c.path/'campaign.json')
        if name=='detail':return self.c.detail(**arguments)
        if name=='knowledge':return {p.name:p.read_text() for p in (self.c.path/'knowledge').glob('stage1_*')}
        methods=dict(brief=self.c.brief,submit=self.c.submit,test=self.c.test,fit=self.c.fit_semantics)
        if name not in methods:raise ValueError('unknown author endpoint')
        key=f'{time.time_ns()}_{name}';p=self.c.path/'author_tools'/f'{key}.json'
        save(p,dict(tool=name,arguments=arguments,status='running'))
        try:
            result=methods[name](**arguments);save(p,dict(tool=name,arguments=arguments,status='complete',result=result));return result
        except Exception as exc:
            save(p,dict(tool=name,arguments=arguments,status='error',error=str(exc)));raise
