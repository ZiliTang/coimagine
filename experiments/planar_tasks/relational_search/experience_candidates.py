"""Observed-history retrieval of witnessed actions, without a trajectory cursor.

No true geometry, fixture, absolute timestamp, boundary index, remaining time,
future observation or success label is accepted by the retrieval descriptor.
Record provenance is returned for inspection, never used in similarity ranking.
"""
from copy import deepcopy
import numpy as np
from .contract import public_observation,validate_action,canonical


def observation_vector(raw):
    o=public_observation(raw);goal=o['goal']['target_id'];result=[]
    result.extend(o['goal']['xy']);result.append(o['goal']['radius'])
    # Instance labels only distinguish target roles; no numeric/lexical ID feature.
    result.extend([float(o.get('active_target')==goal),float(o.get('active_target') is not None)])
    for r in o['robots']:
        own=r['own'];result.extend(own[k] for k in ('x','y','vx','vy','omega'))
        result.extend([np.cos(own['angle']),np.sin(own['angle'])])
        result.extend(float(r.get(k,False)) for k in ('contacting_object','contacting_static','contacting_teammate'))
        scan=r['scan'];result.extend(float(d) if v else 0. for d,v in zip(scan['distances'],scan['valid']))
        result.extend(float(v) for v in scan['valid'])
        result.extend(float(v==goal) for v in scan['entity_ids'])
    return np.asarray(result,dtype=float)


def descriptor(history):
    # Last three observed packets, chronological; missing past is explicit.
    packets=[observation_vector(e['observation']) for e in history.events[-3:]]
    size=len(packets[-1]);out=[]
    for i in range(3-len(packets)):out.extend([0.]*(size+1))
    for packet in packets:out.extend([1.,*packet])
    # Recent actually committed commands, not the requested candidate.
    goal=history.events[-1]['observation']['goal']['target_id']
    for event in ([None]*2+history.events)[-2:]:
        a=None if event is None else event['action']
        out.extend([0.]*7 if a is None else [1.,float(a['target_id'] is None),float(a['target_id']==goal),a['a0']/79.,a['a1']/79.,a['ticks']/40.,1.])
    return np.asarray(out,dtype=float)


class ExperienceCandidates:
    def __init__(self,records,max_actions=4,neighbors=16):
        if not records or not 1<=max_actions<=16 or neighbors<max_actions:raise ValueError('retrieval limits')
        self.records=deepcopy(records);self.max_actions=max_actions;self.neighbors=neighbors
        self.x=np.asarray([r['descriptor'] for r in records],dtype=float)
        if self.x.ndim!=2 or not np.isfinite(self.x).all():raise ValueError('finite descriptor matrix')
        self.scale=np.maximum(self.x.std(axis=0),.05)

    @classmethod
    def from_trajectories(cls,trajectories,**kwargs):
        from .contract import History
        records=[]
        for provenance,rows,commands in trajectories:
            h=History()
            for i,row in enumerate(rows[:-1]):
                h.append(row['observation'],row['action'])
                action=validate_action(commands[i],h.events[-1]['observation'])
                records.append(dict(provenance=dict(trajectory=provenance,boundary=i),
                                    descriptor=descriptor(h).tolist(),action=action))
        return cls(records,**kwargs)

    def retrieve(self,history):
        q=descriptor(history)
        distances=np.mean(((self.x-q)/self.scale)**2,axis=1)
        order=np.argsort(distances,kind='stable')[:self.neighbors];out=[];seen=set()
        for i in order:
            item=self.records[int(i)];action=item['action']
            try:validate_action(action,history.events[-1]['observation'])
            except ValueError:continue
            key=canonical(action)
            if key in seen:continue
            seen.add(key);out.append(dict(action=deepcopy(action),distance=float(distances[i]),provenance=deepcopy(item['provenance'])))
            if len(out)>=self.max_actions:break
        return out
