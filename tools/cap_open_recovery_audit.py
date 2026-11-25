"""Offline replay for the frozen author's local claim. Never pass physical frames into CAP."""
from contextlib import contextmanager
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

from tools import cap_open_recovery_round as study

OUT=study.OUT


class Evidence:
    def __init__(self):
        self.events=[]
        self.frames=[]
        self.parent_keys=None

    def record(self,obs,memory,action,kind):
        # Keep compact public state added by the candidate and the existing module slot.
        data={k:v for k,v in memory.items() if k not in self.parent_keys or k in ('slot','compact')}
        self.events.append(dict(t=obs['tick'],kind=kind,action=action,state=data,
                                touch=[r['contacting_object'] for r in obs['robots']]))
    def boundary(self,obs,memory,action,pending):
        self.record(obs,memory,action,'pending' if pending else 'boundary')
    def checkpoint(self,obs,memory,action,original,reason):
        self.record(obs,memory,action,reason)
    @contextmanager
    def frames_from(self,original,fixture):
        with original(fixture) as (world,other):
            env=world.env.env
            target=world.replay()['scene']['primary_id']
            body=env.backend.bodies[target]
            def sample():
                assert len(self.frames)==world.env.total_periods
                self.frames.append([round(float(v),6) for v in (*body.position,body.angle)])
            sample()
            update=world.detector.update
            def after(packet):
                r=update(packet)
                sample()
                return r
            world.detector.update=after
            yield world,other


def record(job):
    key,fixture,src_hook,parent_src,tag,parent_keys=job
    path=OUT/'local_observations'/tag/(key+'.json.gz')
    if path.exists():return
    evidence=Evidence(); evidence.parent_keys=parent_keys
    phase=study.old.prefix.s.phase
    original=phase.world
    with patch.object(phase,'world',lambda f:evidence.frames_from(original,f)):
        _,result=study.episode((key,fixture,False,None,'boundary',False,src_hook,parent_src),evidence)
    assert result['result']==study.read(OUT/'runs/confirm'/tag/'full'/(key+'.json'))['result']
    path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix('.tmp')
    with gzip.open(temp,'wt',compresslevel=2) as f:
        json.dump(dict(frames=evidence.frames,events=evidence.events,
                       boundaries=[{k:r[k] for k in ('tick','gap','public_history','action','from_event')}
                                   for r in result['boundaries']]),f,separators=(',',':'))
    temp.replace(path)


def replay(tag,keys):
    fixtures=study.read(OUT/'fixtures.json')
    src=study.seal(tag)
    pmem=study.read(next(study.old.PUBLIC.joinpath('snapshots').glob('*.json')))['memory']
    # Keys present in s003. Added state from the previous patch remains visible.
    jobs=[(k,fixtures[k],study.enabled(src),None,tag,set(pmem)) for k in keys]
    study.old.compile_source(src)
    with ProcessPoolExecutor(min(12,len(keys)),mp_context=mp.get_context('fork'),
                             initializer=study.old.isolated.initialize,initargs=(src,)) as pool:
        list(pool.map(record,jobs))


def load_record(tag,key):
    with gzip.open(OUT/'local_observations'/tag/(key+'.json.gz'),'rt') as f:return json.load(f)


def first_change(parent,candidate):
    """Find the first changed command or interruption along a paired trajectory."""
    a=parent['boundaries'];b=candidate['boundaries']
    for i,(x,y) in enumerate(zip(a,b)):
        if x['tick']!=y['tick']:
            return min(x['tick'],y['tick']), 'interrupt'
        if x['public_history']!=y['public_history']:
            raise ValueError('public history changed before action or duration')
        if x['action']!=y['action']:
            return x['tick'], 'choice'
    if len(a)!=len(b):
        extra=(a if len(a)>len(b) else b)[min(len(a),len(b))]
        return extra['tick'],'interrupt'
    return None,None


def timing(name,keys):
    rows=[]
    for k in keys:
        a=load_record('P',k);b=load_record(name,k)
        t,kind=first_change(a,b)
        base=study.read(OUT/'runs/confirm/P/full'/(k+'.json'))
        other=study.read(OUT/'runs/confirm'/name/'full'/(k+'.json'))
        if t is None and base['result']['history']!=other['result']['history']:
            raise ValueError('unexplained distinct trajectory in '+k)
        start=base['effects'][base['anchor']['window_end']]['tick'] if base['anchor'] else None
        rows.append(dict(room=k,first_change=t,change_type=kind,stall_confirmation=start,
                         relative_tick=t-start if start is not None and t is not None else None,
                         parent=base['result']['success'],candidate=other['result']['success']))
    return rows
