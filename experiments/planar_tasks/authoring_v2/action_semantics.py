"""Versioned action proposal + nonlinear ranking author experiment profile."""
import hashlib,time
from copy import deepcopy
from .relational_live import LiveCampaign,LiveSession
from experiments.planar_tasks.relational_search.pipeline import read,save,pack,identifier,ROOT
from experiments.planar_tasks.relational_search.contract import History,digest
from experiments.planar_tasks.relational_search.pipeline_execution import history_at
from experiments.planar_tasks.action_semantics.program import Program
from experiments.planar_tasks.action_semantics import fitting,execution


class ActionCampaign(LiveCampaign):
    @classmethod
    def create(cls,path,**kwargs):
        c=super().create(path,**kwargs);s=read(c.path/'campaign.json')
        s['limits']['feature_batches']=10000;s['used']['feature_batches']=0
        s['action_profile']=dict(version='action-semantics-mlp-v1',episode_wall_seconds=300,max_decisions=225,
            modes=['proposal','learned','search'],architecture='32-tanh-32-tanh; shared candidate scorer',
            equivalence='proposal mode only, same causal history/action/timing/determinism; learned/search performance not guaranteed')
        rel='experiments/planar_tasks/action_semantics/guide.md';data=(ROOT/rel).read_bytes()
        p=c.path/'sources'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        sources=read(c.path/'sources.json');sources[rel]=hashlib.sha256(data).hexdigest();save(c.path/'sources.json',sources)
        s['source_hash']=digest(sources);(c.path/'knowledge'/'action_guide.md').write_bytes(data)
        save(c.path/'campaign.json',s);return cls(path)

    def _source(self,sid):return (self.path/'semantics'/(identifier(sid)+'.py')).read_text()
    def _brief(self,rid,turn_id):
        r=self._round(rid)
        if not any(t['id']==turn_id for t in r['author_turns']):raise ValueError('record brief first')
        return r

    def submit_action_semantics(self,rid,source,turn_id):
        with self.locked():
            self.verify_sources();r=self._brief(rid,turn_id);h=History();h.append(read(self._rpath(rid)/'initial_public.json'))
            probe=Program(source,lambda:self._charge(feature_batches=1)).evaluate(h)
            sid=f's{len(self.state["semantics"]):03d}';p=self.path/'semantics';p.mkdir(exist_ok=True);(p/(sid+'.py')).write_text(source)
            meta=dict(id=sid,round_id=rid,turn_id=turn_id,sha256=digest(source),names=probe['names'],profile='action-semantics-mlp-v1',validation='schema only',initial_proposals=probe['proposals'],invalid_proposals=probe['invalid_proposals'])
            self.state['semantics'].append(meta);self._save();r['semantics'].append(sid);save(self._rpath(rid)/'round.json',r);return meta

    def inspect_action(self,rid,semantic_id,attempt_id=None,boundary=None,fit_id=None,condition='S'):
        with self.locked():
            self.verify_sources()
            if attempt_id is None:h=History();h.append(read(self._rpath(rid)/'initial_public.json'))
            else:
                from experiments.planar_tasks.relational_search.pipeline import unpack
                h=history_at(unpack(self._apath(rid,attempt_id)/'public.json.gz'),boundary)
            program=Program(self._source(semantic_id),lambda:self._charge(feature_batches=1));packet=program.evaluate(h)
            scores=execution.Scorer(program,read(self.path/'fits'/(identifier(fit_id)+'.json')),condition).packet_scores(h,packet) if fit_id else None
            order=sorted(range(len(packet['actions'])),key=lambda i:-scores[i]) if scores is not None else range(len(packet['actions']))
            return dict(history_hash=h.sha256,proposals=packet['proposals'],invalid_proposals=packet['invalid_proposals'],candidate_count=len(packet['actions']),top=[dict(action=packet['actions'][i],score=None if scores is None else scores[i],features=packet['values'][i]) for i in list(order)[:10]])

    def fit_action_semantics(self,rid,semantic_id,demonstration_refs=None,comparison_refs=None):
        with self.locked():
            self.verify_sources();source=self._source(semantic_id);self._round(rid)
            rounds=[self._round(r) for r in self.state['rounds']]
            demos=demonstration_refs if demonstration_refs is not None else [(r['id'],a['id']) for r in rounds if r['split']=='train' for a in r['attempts'] if a['status']=='success']
            comps=comparison_refs if comparison_refs is not None else [(r['id'],c['id']) for r in rounds if r['split']=='train' for c in r['comparisons'] if c['status']=='complete']
            for refs in (demos,comps):
                if len(refs)!=len(set(tuple(x) for x in refs)):raise ValueError('duplicate reference')
                for rr,aa in refs:identifier(rr);identifier(aa)
            self._charge(fits=1);fid=f'f{len(self.state["fits"]):03d}';p=self.path/'fits';p.mkdir(exist_ok=True)
            meta=dict(id=fid,round_id=rid,semantic_id=semantic_id,status='running',demonstration_refs=demos,comparisons=comps)
            self.state['fits'].append(meta);self._save();save(p/(fid+'.json'),meta)
            try:
                data=fitting.build(self,source,demos,comps,on_feature=lambda:self._charge(feature_batches=1),progress=lambda n,total:save(p/(fid+'_progress.json'),dict(histories=n,total=total,updated=time.time())))
                pack(p/(fid+'_dataset.json.gz'),data)
                result=fitting.fit_dataset(data,source);result.update(meta,status='complete');save(p/(fid+'.json'),result);meta['status']='complete'
            except Exception as exc:
                meta.update(status='error',error=str(exc));save(p/(fid+'.json'),meta);raise
            finally:
                self._save();r=self._round(rid);r['fits'].append(fid);save(self._rpath(rid)/'round.json',r)
            return dict(fit_id=fid,demonstrations=result['demonstrations'],physical_pairs=result['physical_pairs'],models={k:v['metrics'] for k,v in result['models'].items()})

    def test_action_policy(self,rid,semantic_id,turn_id,mode='proposal',fit_id=None,condition='S',max_ticks=900,max_decisions=225):
        with self.locked():
            self.verify_sources();r=self._brief(rid,turn_id);profile=self.state['action_profile'];source=self._source(semantic_id)
            if mode not in profile['modes']:raise ValueError('mode')
            if type(max_ticks) is not int or not 1<=max_ticks<=1800:raise ValueError('tick limit')
            if type(max_decisions) is not int or not 1<=max_decisions<=profile['max_decisions']:raise ValueError('decision limit')
            fitted=read(self.path/'fits'/(identifier(fit_id)+'.json')) if mode!='proposal' else None
            if fitted is not None:execution.Scorer(Program(source),fitted,condition)
            aid=f'a{len(r["attempts"]):03d}';p=self._apath(rid,aid);key=rid+'/'+aid
            self._charge(attempts=1,resets=1,execution_ticks=max_ticks);self.state['reservations'][key]=max_ticks;self._save();p.mkdir(parents=True)
            meta=dict(semantic_id=semantic_id,fit_id=fit_id,condition=condition,mode=mode,turn_id=turn_id,full_root=True)
            save(p/'request.json',dict(kind='closed_loop',metadata=meta,max_ticks=max_ticks,max_decisions=max_decisions))
            item=dict(id=aid,kind='closed_loop',status='running',metadata=meta,created=time.time());r['attempts'].append(item);r['status']='running';save(self._rpath(rid)/'round.json',r)
            reserve=None
            def prediction(phase,ticks,calls):
                nonlocal reserve
                if phase=='reserve':self._charge(prediction_ticks=ticks,prediction_calls=calls);reserve=(ticks,calls)
                else:
                    self.state['used']['prediction_ticks']-=reserve[0]-ticks;self.state['used']['prediction_calls']-=reserve[1]-calls;self._save();reserve=None
            try:
                result=execution.run(read(self._rpath(rid)/'fixture.json'),source,mode=mode,fitted=fitted,condition=condition,max_ticks=max_ticks,max_decisions=max_decisions,wall_seconds=profile['episode_wall_seconds'],on_prediction=prediction,on_feature=lambda:self._charge(feature_batches=1),on_progress=lambda n,t,d:save(p/'progress.json',dict(boundary=n,tick=t,last_action=d['executed_action'],updated=time.time())))
                for file,field in [('public','public_rows'),('replay','replay'),('decisions','decisions')]:pack(p/(file+'.json.gz'),result[field])
                for file,field in [('summary','summary'),('events','events'),('executed_commands','commands')]:save(p/(file+'.json'),result[field])
                request=read(p/'request.json');request['commands']=result['commands'];save(p/'request.json',request)
                item.update(result['summary']);self.state['used']['execution_ticks']-=max_ticks-item['ticks'];del self.state['reservations'][key];self._save()
            except Exception as exc:
                item.update(status='infrastructure_error',error=str(exc),reservation_retained=True);save(p/'summary.json',item);raise
            finally:r['status']='awaiting_author';save(self._rpath(rid)/'round.json',r)
            return deepcopy(item)


class ActionSession(LiveSession):
    def __init__(self,path):self.campaign=ActionCampaign(path)
    def _tool(self,name,arguments):
        if name in ('submit_semantics','fit','test_closed_loop','test_plan','apply_ranker'):raise ValueError('use action-semantic profile endpoints')
        if name in ('submit_action_semantics','fit_action_semantics','inspect_action','test_action_policy'):return getattr(self.campaign,name)(**arguments)
        if name=='knowledge':return {n:(self.campaign.path/'knowledge'/n).read_text() for n in ('0_option_guide.md','3_option_guide.md','action_guide.md')}
        result=super()._tool(name,arguments)
        if name=='status':result['action_profile']=read(self.campaign.path/'campaign.json')['action_profile']
        return result
