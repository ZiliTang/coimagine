"""Random-room CAP author: metered batches, immutable library commits, dev only."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,json,time,multiprocessing as mp,shutil,hashlib,traceback
from .relational_transfer_author import TransferCampaign,TransferSession,GUIDES
from experiments.planar_tasks.relational_search.pipeline import Campaign,read,save,pack,ROOT
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.environment import EventConfig

VERSION='random-room-cap-library-v1'
BASE=ROOT/'outputs/semantic_rl/random_room_author_20261117_01'
HOST=ROOT/'outputs/semantic_rl_authors/random_room_20261117_01'


def worker(job):
    import torch
    torch.set_num_threads(1)
    p=Path(job['path']);save(p/'started.json',dict(at=time.time()))
    try:
        result,_=rollout(job['fixture'],job['source'],config=EventConfig(mode='duration'),max_ticks=job['max_ticks'],max_decisions=300,wall_seconds=300,deterministic=True,seed=17,fast_feedback=True)
        started=time.monotonic();pack(p/'result.json.gz',result);io=time.monotonic()-started
        summary=dict(result['summary'],storage_seconds=io)
        save(p/'summary.json',summary)
        return summary
    except Exception:
        save(p/'worker_error.json',dict(traceback=traceback.format_exc()));raise


class RoomCampaign(TransferCampaign):
    @classmethod
    def create(cls,path,fixtures):
        first=list(fixtures.values())[:4]
        parent=TransferCampaign.create(path,{f'dev_{i:02d}':v for i,v in enumerate(first)},wall_seconds=8*3600)
        s=read(parent.path/'campaign.json')
        shutil.rmtree(parent.path/'tasks')
        for name,value in fixtures.items():
            scene,boxes=X.fixture(value);save(parent.path/'tasks'/name/'fixture.json',value)
            save(parent.path/'tasks'/name/'initial_public.json',X.RecordedEnv(scene,boxes).public())
        s.update(profile=VERSION,tasks=list(fixtures),library=[],round_batches=[])
        s['limits'].update(attempts=4000,resets=4000,execution_ticks=7200000,feature_batches=2500000,author_turns=2000)
        s['author_protocol'].update(version=VERSION,max_library_policies=10,wall_seconds=8*3600,final='FINAL.json plus up to ten immutable committed CAP versions; host performs supervised integration',only_declared_fixtures=True)
        save(parent.path/'campaign.json',s)
        save(parent.path/'knowledge/transfer_execution_protocol.json',dict(stage1=s['stage1'],limits=s['limits'],author_protocol=s['author_protocol']))
        return cls(path)

    def batch(self,sid,tasks,max_ticks=1800):
        if not isinstance(tasks,list) or not tasks or len(set(tasks))!=len(tasks) or len(tasks)>160:raise ValueError('1..160 distinct dev tasks')
        if type(max_ticks)!=int or not 1<=max_ticks<=1800:raise ValueError('tick cap')
        with self.locked():
            self._check()
            if any(t not in self.state['tasks'] for t in tasks):raise ValueError('undeclared task')
            source=self._source(sid);requests=[]
            self._charge(attempts=len(tasks),resets=len(tasks),execution_ticks=max_ticks*len(tasks),feature_batches=301*len(tasks))
            for t in tasks:
                rid=f'a{len(self.state["runs"]):05d}';idx=len(self.state['runs']);p=self.path/'runs'/rid
                r=dict(id=rid,sid=sid,task=t,fit_id=None,status='reserved',phase='author_batch')
                self.state['runs'].append(r);self.state['reservations'][rid]=dict(ticks=max_ticks,features=301)
                save(p/'request.json',dict(r,max_ticks=max_ticks));requests.append((r,idx))
            self._save()
        rows=[];pending={};cursor=0;errors=[]
        with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
            while pending or cursor<len(requests):
                while cursor<len(requests) and len(pending)<16:
                    r,idx=requests[cursor];cursor+=1
                    job=dict(path=str(self.path/'runs'/r['id']),source=source,fixture=read(self.path/'tasks'/r['task']/'fixture.json'),max_ticks=max_ticks)
                    pending[pool.submit(worker,job)]=(r,idx)
                done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    r,idx=pending.pop(future)
                    try:
                        summary=future.result();row=dict(r,**summary)
                        with self.locked():
                            self.state['used']['execution_ticks']-=max_ticks-summary['ticks']
                            # Actual typed-stream CAP requests are one per decision incl. failing attempt.
                            used=summary['decisions']+int(bool(summary['error']))
                            self.state['used']['feature_batches']-=301-used
                            del self.state['reservations'][r['id']];self.state['runs'][idx]=row;self._save()
                        rows.append(row)
                    except Exception as e:errors.append(dict(id=r['id'],error=repr(e)))
                del done
        if errors:raise RuntimeError(errors)
        return dict(sid=sid,completed=len(rows),success=sum(r['success'] for r in rows),ticks=sum(r['ticks'] for r in rows),rows=rows)

    def commit(self,sid,summary):
        with self.locked():
            self._check()
            if len(self.state['library'])>=10:raise ValueError('ten-policy library cap')
            if sid in [r['sid'] for r in self.state['library']]:raise ValueError('already committed')
            source=self._source(sid)
            tested={r['task']:r for r in self.state['runs'] if r['sid']==sid and r.get('success') is not None and r.get('ticks',0)>0 and read(self.path/'runs'/r['id']/'request.json').get('max_ticks')==1800}
            missing=[t for t in self.state['tasks'] if t not in tested]
        if missing:self.batch(sid,missing)
        with self.locked():
            tested={r['task']:r for r in self.state['runs'] if r['sid']==sid and r.get('success') is not None and read(self.path/'runs'/r['id']/'request.json').get('max_ticks')==1800}
            if set(tested)!=set(self.state['tasks']):raise ValueError('full-budget coverage required; batch all tasks first')
            covered={t for p in self.state['library'] for t in p['success_tasks']}
            success=sorted(t for t,r in tested.items() if r['success'])
            name=f'CAP-RR-{len(self.state["library"])+1:03d}'
            item=dict(name=name,sid=sid,sha256=digest(source),summary=summary,success_tasks=success,new_success_tasks=sorted(set(success)-covered),runs={t:r['id'] for t,r in tested.items()},at=time.time())
            save(self.path/'library'/name/'manifest.json',item);(self.path/'library'/name/'policy.py').write_text(source)
            self.state['library'].append(item);self._save()
            return dict(item,coverage_union=len(covered|set(success)),remaining=sorted(set(self.state['tasks'])-(covered|set(success))))


class RoomSession(TransferSession):
    def __init__(self,path):self.c=RoomCampaign(path)
    def tool(self,name,arguments):
        if name=='status':
            s=read(self.c.path/'campaign.json');return {k:s[k] for k in ['profile','model','limits','used','deadline','tasks','submissions','library','author_protocol']}
        if name in ('batch','commit'):
            p=self.c.path/'author_tools'/f'{time.time_ns()}_{name}.json';save(p,dict(tool=name,arguments=arguments,status='running'))
            try:
                result=getattr(self.c,name)(**arguments);save(p,dict(tool=name,arguments=arguments,status='complete',result=result));return result
            except Exception as e:save(p,dict(tool=name,arguments=arguments,status='error',error=repr(e)));raise
        return super().tool(name,arguments)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--request',required=True);a=p.parse_args();r=read(a.request)
    print(json.dumps(RoomSession(a.campaign).tool(r['name'],r.get('arguments',{})),ensure_ascii=False,allow_nan=False))
