"""Goal-relative, entity-bound public geometry for relation-transfer attribution.

No authored program memory, hidden body geometry, fixture labels or task IDs enter G.
The fixed frame comes from the first visible target point mean and public goal.
All action IDs keep their original legal support; only scorer features are rotated.
"""
from collections import deque
import math
import numpy as np
from .features import Features
from .memory_flow import current_event
from .actions import FIELDS

VERSION='relation-transfer-relative-inputs-v1'
ENTITY_CAPACITY=9


def action_vector(index,theta):
    out=np.zeros(11,dtype=float)
    if index is None:return out
    if 0<=index<32:kind=0;d,s=divmod(index,4);speed=.1*(s+1);angle=d*math.pi/4-theta
    elif 47<=index<79:kind=1;d,s=divmod(index-47,4);speed=.1*(s+1);angle=d*math.pi/4-theta
    else:
        kind=2 if 32<=index<=34 else 3 if 35<=index<=37 else {40:4,43:5,44:6,46:7}.get(index)
        if kind is None:raise ValueError('noncanonical public action')
        speed=.02*(1+(index-32)%3) if kind in (2,3) else .3 if kind in (4,5) else 0.;angle=None
    out[kind]=1.;out[8]=speed
    if angle is not None:out[9:]=[math.cos(angle),math.sin(angle)]
    return out


def action_encoding(prefix,goal_id,theta):
    # Common to EVERY arm: prefix masks, target roles, relative action parameters.
    flags=[float(k in prefix) for k in FIELDS]
    role=[float(prefix.get('target_id') is None),float(prefix.get('target_id')==goal_id)] if 'target_id' in prefix else [0.,0.]
    duration=np.zeros(41)
    if 'ticks' in prefix:
        t=prefix['ticks']
        if type(t) is not int or not 1<=t<=40:raise ValueError('public duration1..40')
        duration[0]=t*.05;duration[t]=1.
    return np.concatenate([flags,role,action_vector(prefix.get('a0'),theta),action_vector(prefix.get('a1'),theta),duration])


class RelativeGeometry:
    def __init__(self,length=8):self.length=length;self.episode=None;self.index=-1

    def rotate(self,xy):
        a=np.asarray(xy,dtype=float);return a@self.rotation.T

    def points(self,xy):return self.rotate(np.asarray(xy,dtype=float)-self.origin)

    def advance(self,h):
        event=current_event(h);o=event['observation']
        if h.episode!=self.episode:
            if h.index!=0:raise ValueError('relative geometry requires complete causal prefix')
            self.episode=h.episode;self.index=-1;self.frames=deque(maxlen=self.length);self.entities={}
            self.origin=np.asarray(o['goal']['xy'],dtype=float);goal=o['goal']['target_id']
            pts=next((e['points'] for e in o['entities'] if e['id']==goal and e['points']),None)
            self.measured_frame=pts is not None
            anchor=np.mean(pts,axis=0) if pts is not None else np.mean([[r['own']['x'],r['own']['y']] for r in o['robots']],axis=0)
            delta=self.origin-anchor;self.theta=math.atan2(delta[1],delta[0])
            c,s=math.cos(self.theta),math.sin(self.theta);self.rotation=np.array([[c,s],[-s,c]])
        if h.index==self.index:
            if h.sha256!=self.chain:raise ValueError('changed relative history')
            return
        if h.index!=self.index+1:raise ValueError('skipped relative history')
        now=o['tick'];self.goal_id=o['goal']['target_id'];visible=set()
        for e in o['entities']:
            if e['points']:
                pts=self.points(e['points']);self.entities[e['id']]=dict(last=e['last_seen_tick'],count=len(pts),mean=pts.mean(0),lo=pts.min(0),hi=pts.max(0),std=pts.std(0));visible.add(e['id'])
        def entity_row(label):
            e=self.entities.get(label)
            return np.array([0.,0.]+[np.nan]*10) if e is None else np.array([float(label in visible),1.,(now-e['last'])*.05,math.log1p(e['count']),*e['mean'],*e['lo'],*e['hi'],*e['std']])
        self.rows={label:entity_row(label) for label in self.entities}
        others=sorted((k for k in self.entities if k!=self.goal_id),key=lambda k:tuple(self.entities[k]['mean']))
        if len(others)>ENTITY_CAPACITY-1:raise ValueError('entity capacity exceeded')
        erows=[entity_row(self.goal_id)]+[entity_row(k) for k in others]
        erows += [np.array([0.,0.]+[np.nan]*10)]*(ENTITY_CAPACITY-len(erows))
        vals=[now*.05,o['decision'],o['goal']['radius'],float(self.measured_frame),len(visible),float(o['active_target']==self.goal_id),float(o['active_target'] is None)]
        self.robot_xy=[]
        for robot in o['robots']:
            own=robot['own'];xy=self.points([own['x'],own['y']]);self.robot_xy.append(xy)
            vals += [*xy,math.cos(own['angle']-self.theta),math.sin(own['angle']-self.theta),*self.rotate([own['vx'],own['vy']]),own['omega']]
            vals += [float(robot[k]) for k in ['contacting_object','contacting_static','contacting_teammate']]
            ex=robot['execution'];vals += [(np.nan if ex.get('done') is None else float(ex['done'])),float(ex.get('stage') in ('blocked','move_blocked'))]
            for key in ['desired_velocity','reference_velocity','force']:
                pair=ex.get(key);vals += [np.nan,np.nan] if pair is None else list(self.rotate(pair))
            patches=[self.rotate(p['normal_world']) for p in robot.get('patches',[]) if p.get('valid')]
            vals += list(np.mean(patches,axis=0)) if patches else [np.nan,np.nan]
            width,height=robot['arena'];corners=self.points([[0,0],[width,0],[width,height],[0,height]])
            corners=np.array(sorted(corners.tolist(),key=lambda p:(round(p[0],10),round(p[1],10))))
            vals += list(corners.ravel())+[robot['force_limit'],robot['remaining_seconds']]
            scan=robot['scan'];angles=np.asarray(scan['ray_angles'])-self.theta
            order=sorted(range(72),key=lambda j:round((float(angles[j])+math.pi)%(2*math.pi),10))
            vals += [math.cos(angles[order[0]]),math.sin(angles[order[0]])]
            for j in order:
                valid=scan['valid'][j];label=scan['entity_ids'][j];category=scan['category'][j]
                role=0 if not valid else 1 if label==self.goal_id else 2 if label is not None else 3 if category==2 else 4
                vals += [scan['distances'][j],float(valid)]+[float(role==k) for k in range(5)]
        previous=event['action']
        vals += action_encoding(previous or {},self.goal_id,self.theta).tolist()
        x=np.concatenate([np.asarray(vals),*erows]);self.frames.append(x)
        if self.index<0:self.initial=x.copy()
        recent=[np.full_like(x,np.nan)]*(self.length-len(self.frames))+list(self.frames)
        self.vector=np.concatenate([self.initial,*recent]);self.event_width=len(x);self.index=h.index;self.chain=h.sha256

    def candidate_geometry(self,prefix):
        if 'target_id' not in prefix or prefix['target_id'] is None:return np.full(16,np.nan)
        row=self.rows.get(prefix['target_id'],np.array([0.,0.]+[np.nan]*10))
        center=row[4:6];return np.concatenate([row,*(np.asarray(r)-center for r in self.robot_xy)])


class TransferFeatures(Features):
    def __init__(self,*args,groups,**kwargs):
        super().__init__(*args,**kwargs);self.groups=groups;self.geometry=RelativeGeometry(self.semantic_protocol['history_length']);self.layout=None

    def panel(self,history,tree,prefix):
        self.geometry.advance(history);p=super().panel(history,tree,prefix)
        s=np.asarray(p['semantic'],dtype=float);slots=len(self.schema.slots);length=self.semantic_protocol['history_length'];capacity=self.semantic_protocol['state_capacity'];window=length*(capacity+2)
        p['attribution_context']=np.concatenate([self.geometry.vector,s[0,slots:slots+window]])
        geom=np.array([self.geometry.candidate_geometry(q) for q in p['choices']]);common=np.array([action_encoding(q,tree.observation['goal']['target_id'],self.geometry.theta) for q in p['choices']])
        p['attribution_candidates']=np.concatenate([s[:,:slots],s[:,slots+window:slots+window+1],geom,common],axis=1)
        if self.layout is None:
            cg=['G']*len(self.geometry.vector)
            state_names=self.semantic_protocol['state_names']
            if set(self.groups['candidate'])!=set(self.names) or set(self.groups['state'])!=set(state_names):raise ValueError('authored group schema mismatch')
            for _ in range(length):cg += [self.groups['state'][n] for n in state_names]+['unused']*(capacity-len(state_names))+['common','common']
            xg=[self.groups['candidate'][n] if n is not None else 'unused' for n in self.schema.slots]+['A']+['G']*16+['common']*common.shape[1]
            self.layout=dict(version=VERSION,context_groups=cg,candidate_groups=xg,semantic_groups=self.groups,semantic_slots=list(self.schema.slots),state_names=list(state_names),geometry_event_width=self.geometry.event_width,geometry_history='initial plus last H; entity-preserving goal-relative public scans and poses; lossy, not full belief',action_encoding='original support, goal-relative parameters, common40-way duration',frame='first visible target sample mean toward public goal; fallback team mean with validity flag',history_length=length)
        return p
