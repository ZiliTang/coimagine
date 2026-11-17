"""Clean-author profile: privileged offline development, public fitted online ranks."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import time
from .relational import RelationalSession
from experiments.planar_tasks.relational_search.pipeline import Campaign,read,save,pack,unpack,identifier,ROOT
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.relational_search.live_search import Budget
from experiments.planar_tasks.relational_search.live_execution import run


class LiveCampaign(Campaign):
    @classmethod
    def create(cls,path,**kwargs):
        campaign=super().create(path,**kwargs)
        state=read(campaign.path/'campaign.json')
        state['limits'].update(prediction_ticks=200000,prediction_calls=10000)
        state['used'].update(prediction_ticks=0,prediction_calls=0)
        rel='experiments/planar_tasks/relational_search/live_guide.md'
        guide=(ROOT/rel).read_bytes()
        target=campaign.path/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(guide)
        sources=read(campaign.path/'sources.json');sources[rel]=hashlib.sha256(guide).hexdigest()
        save(campaign.path/'sources.json',sources);state['source_hash']=digest(sources)
        (campaign.path/'knowledge'/'live_guide.md').write_bytes(guide)
        state['live_profile']=dict(version='relational-replanning-v1',search=asdict(Budget()),
            episode_wall_seconds=300,max_decisions=90,
            author_access='simulator implementation, truth, replay, authored probes; no prior experiment semantic solutions',
            deployed_access='causal public history and candidate only',
            leaf='shared visible-support distance / 0.3 m/s',
            selection='success first; else longest reached nonterminal leaf horizon; min task+leaf cost, rank breaks ties')
        save(campaign.path/'campaign.json',state)
        return cls(path)

    def test_closed_loop(self,rid,*,fit_id,turn_id,condition='M',max_ticks=400,max_decisions=30,prefix=None):
        with self.locked():
            self.verify_sources();r=self._round(rid);profile=self.state['live_profile']
            if not any(t['id']==turn_id for t in r['author_turns']):raise ValueError('record decision brief first')
            if type(max_ticks) is not int or not 1<=max_ticks<=1800:raise ValueError('execution budget')
            if type(max_decisions) is not int or not 1<=max_decisions<=profile['max_decisions']:raise ValueError('decision budget')
            from experiments.planar_tasks.relational_search.pipeline_execution import validate_commands
            prefix=validate_commands(prefix or [],True);total=max_ticks+sum(a['ticks'] for a in prefix)
            if total>1800:raise ValueError('prefix plus execution exceeds episode cap')
            fitted=read(self.path/'fits'/(identifier(fit_id)+'.json'))
            source=(self.path/'semantics'/(identifier(fitted['semantic_id'])+'.py')).read_text()
            aid=f'a{len(r["attempts"]):03d}';key=rid+'/'+aid;p=self._apath(rid,aid)
            self._charge(attempts=1,resets=1,execution_ticks=total)
            self.state['reservations'][key]=total;self._save();p.mkdir(parents=True)
            metadata=dict(fit_id=fit_id,semantic_id=fitted['semantic_id'],condition=condition,turn_id=turn_id,
                          full_root=not bool(prefix),search_profile=profile['search'])
            save(p/'request.json',dict(kind='closed_loop',prefix=prefix,max_ticks=max_ticks,max_decisions=max_decisions,metadata=metadata))
            item=dict(id=aid,kind='closed_loop',status='running',metadata=metadata,created=time.time(),reserved_ticks=total)
            r['attempts'].append(item);r['status']='running';save(self._rpath(rid)/'round.json',r)
            reserve=None
            def prediction(phase,ticks,calls):
                nonlocal reserve
                if phase=='reserve':
                    self._charge(prediction_ticks=ticks,prediction_calls=calls);reserve=(ticks,calls)
                else:
                    self.state['used']['prediction_ticks']-=reserve[0]-ticks
                    self.state['used']['prediction_calls']-=reserve[1]-calls
                    self._save();reserve=None
            def progress(n,tick,d):
                save(p/'progress.json',dict(boundary=n,tick=tick,updated=time.time(),last_action=d['executed_action'],
                    selected_path=d.get('selected_path'),prediction_ticks=self.state['used']['prediction_ticks']))
            try:
                result=run(read(self._rpath(rid)/'fixture.json'),source,fitted,condition=condition,prefix=prefix,
                    max_ticks=max_ticks,max_decisions=max_decisions,budget=Budget(**profile['search']),
                    wall_seconds=profile['episode_wall_seconds'],on_prediction=prediction,on_progress=progress)
                pack(p/'public.json.gz',result['public_rows']);pack(p/'replay.json.gz',result['replay'])
                pack(p/'decisions.json.gz',result['decisions']);save(p/'events.json',result['events'])
                save(p/'summary.json',result['summary']);save(p/'executed_commands.json',result['commands'])
                request=read(p/'request.json');request['commands']=result['commands'];save(p/'request.json',request)
                item.update(result['summary']);self.state['used']['execution_ticks']-=total-item['ticks']
                del self.state['reservations'][key];self._save()
            except Exception as exc:
                item.update(status='infrastructure_error',error=str(exc),reservation_retained=True)
                save(p/'summary.json',item);raise
            finally:r['status']='awaiting_author';save(self._rpath(rid)/'round.json',r)
            return deepcopy(item)


class LiveSession(RelationalSession):
    def __init__(self,path):self.campaign=LiveCampaign(path)

    def tool(self,name,arguments):
        if read(self.campaign.path/'campaign.json')['status']=='frozen' and name in ('author_context','author_replay','decisions'):
            return self._tool(name,arguments)
        return super().tool(name,arguments)

    def _tool(self,name,arguments):
        c=self.campaign
        if name=='knowledge':
            return {name:(c.path/'knowledge'/name).read_text() for name in ('0_option_guide.md','3_option_guide.md','live_guide.md')}
        if name in ('test_plan','apply_ranker'):raise ValueError('not a deployment endpoint in the corrected profile; use test_closed_loop')
        if name=='test_closed_loop':return c.test_closed_loop(**arguments)
        if name=='author_context':
            rid=arguments['rid'];r=c._round(rid)
            return dict(round=r,fixture=read(c._rpath(rid)/'fixture.json'),scene=read(c._rpath(rid)/'scene.json'),
                public_initial=read(c._rpath(rid)/'initial_public.json'),role='privileged OFFLINE author; not scorer input')
        if name=='author_replay':
            p=c._apath(arguments['rid'],arguments['attempt_id']);data=unpack(p/'replay.json.gz')
            start=arguments.get('start',0);count=arguments.get('count',5)
            if type(start) is not int or start<0 or type(count) is not int or not 1<=count<=100:raise ValueError('replay page')
            return dict(scene=data['scene'],total=len(data['frames']),frames=data['frames'][start:start+count],role='offline author truth')
        if name=='probe':
            with c.locked():return c._execute(arguments['rid'],arguments['commands'],'author_probe',dict(role='offline author experiment'))
        if name=='decisions':
            data=unpack(c._apath(arguments['rid'],arguments['attempt_id'])/'decisions.json.gz')
            start=arguments.get('start',0);count=arguments.get('count',1)
            if type(start) is not int or start<0 or type(count) is not int or not 1<=count<=5:raise ValueError('decision page')
            return dict(total=len(data),decisions=data[start:start+count])
        result=super()._tool(name,arguments)
        if name=='context':result['instruction']='Offline author may inspect implementation/truth/replays through this clean profile; deployment features remain public-history-only.'
        if name=='status':result['live_profile']=read(c.path/'campaign.json')['live_profile']
        return result
