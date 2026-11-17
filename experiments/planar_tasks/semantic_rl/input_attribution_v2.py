"""Controlled input attribution. No changes to public actions or authored semantics.

G is an explicitly lossy geometric history summary, not a complete-history model.
A conservatively includes program phase/progress as well as recommendation signals.
Every arm uses the same normalized history/candidate projections and scorer.
Version 1 is retained unchanged. The two repair factors can be disabled for probes.
The factorization avoids repeating a boundary's history for every candidate.
"""
from collections import deque
from contextlib import contextmanager
import math
import time
import numpy as np
import torch
from torch import nn
from .features import Features
from .actions import ActionTree, FIELDS
from .memory_flow import current_event

VERSION = 'fixed-coverage-input-attribution-v2'
ARMS = ('G', 'GA', 'DA', 'GDA')
# These are program-derived control outputs, not independent physical descriptions.
CONTROL_NAMES = frozenset(('phase_clear','phase_brake','phase_relocate','phase_transport',
    'phase_rotate','phase_time_margin','phase_age','approach_progress','target_handoff',
    'target_role_match','robot0_intent_match','robot1_intent_match','duration_error',
    'robot0_direction_dot','robot1_direction_dot','robot0_speed_error','robot1_speed_error',
    'rotation_sign_compatibility','duration_cap_match','legacy_target_match',
    'legacy_robot0_match','legacy_robot1_match','legacy_duration_match'))


def numeric(x):
    return np.nan if x is None else float(x)


def observation_vector(event):
    """Only public measurements; no CAP memory, phase, true centers or task IDs.

    All 72 scan ranges/categories per robot are retained. Variable point clouds
    use measured bounding boxes, means and counts, grouped by goal/non-goal role.
    """
    o=event['observation']; goal=o['goal']['target_id']
    v=[o['tick']*.05, o['decision'], *o['goal']['xy'],o['goal']['radius'],len(o['entities'])]
    for target in (True,False):
        entities=[e for e in o['entities'] if (e['id']==goal)==target]
        points=[p for e in entities for p in e['points']]
        if points:
            a=np.asarray(points); v += [len(entities),len(points),*a.mean(0),*a.min(0),*a.max(0),
                (o['tick']-max(e['last_seen_tick'] for e in entities))*.05]
        else:v += [len(entities),0]+[np.nan]*7
    for r in o['robots']:
        v += [r['own'][k] for k in ('x','y','angle','vx','vy','omega')]
        v += [float(r[k]) for k in ('contacting_object','contacting_static','contacting_teammate')]
        v += [numeric(r['execution'].get('done')),float(r['execution'].get('stage') in ('blocked','move_blocked'))]
        for key in ('desired_velocity','reference_velocity','force'):
            a=r['execution'].get(key);v += [np.nan,np.nan] if a is None else list(a)
        patches=[p for p in r.get('patches',[]) if p.get('valid')]
        for axis in (0,1):
            vals=[p['normal_world'][axis] for p in patches]
            v += [np.mean(vals) if vals else np.nan,min(vals) if vals else np.nan,max(vals) if vals else np.nan]
        v += list(r['arena'])+[r['force_limit'],r['remaining_seconds']]
        scan=r['scan'];assert len(scan['distances'])==len(scan['category'])==72
        v += scan['distances']+scan['category']
    a=event['action']
    v += [0.,0.,0.,0.,0.] if a is None else [1.,float(a['target_id']==goal),float(a['a0']),float(a['a1']),a['ticks']*.05]
    return np.asarray(v,dtype=np.float64)


class GeometryHistory:
    def __init__(self):self.episode=None;self.index=-1

    def advance(self,history):
        if history.episode!=self.episode:
            if history.index!=0:raise ValueError('geometry history requires the real prefix')
            self.episode=history.episode;self.index=-1;self.frames=deque(maxlen=8)
        if history.index==self.index:
            if history.sha256!=self.chain:raise ValueError('changed geometry history')
            return
        if history.index!=self.index+1:raise ValueError('skipped geometry history')
        x=observation_vector(current_event(history))
        if self.index<0:
            self.initial=x.copy();self.total=np.zeros_like(x);self.count=np.zeros_like(x)
            self.minimum=np.full_like(x,np.inf);self.maximum=np.full_like(x,-np.inf)
        valid=np.isfinite(x);self.total+=np.where(valid,x,0);self.count+=valid
        self.minimum=np.minimum(self.minimum,np.where(valid,x,np.inf))
        self.maximum=np.maximum(self.maximum,np.where(valid,x,-np.inf))
        self.frames.append(x);self.index=history.index;self.chain=history.sha256
        mean=np.divide(self.total,self.count,out=np.full_like(x,np.nan),where=self.count>0)
        lo=np.where(np.isfinite(self.minimum),self.minimum,np.nan)
        hi=np.where(np.isfinite(self.maximum),self.maximum,np.nan)
        recent=[np.full_like(x,np.nan)]*(8-len(self.frames))+list(self.frames)
        self.vector=np.concatenate([self.initial,mean,lo,hi,*recent])


class AttributionFeatures(Features):
    def __init__(self,*args,**kwargs):
        self.duration_onehot=kwargs.pop("duration_onehot", True)
        super().__init__(*args,**kwargs);self.geometry_history=GeometryHistory();self.layout=None

    def panel(self,history,tree,prefix):
        self.geometry_history.advance(history)
        p=super().panel(history,tree,prefix)
        s=np.asarray(p['semantic'],dtype=np.float64);raw=np.asarray(p['raw'],dtype=np.float64)
        slots=len(self.schema.slots);window=self.semantic_protocol['history_length']*(self.semantic_protocol['state_capacity']+2)
        # Candidate-specific legacy geometry stays in G; prefix masks/encoding in every arm.
        p['attribution_context']=np.concatenate([self.geometry_history.vector,s[0,slots:slots+window]])
        p['attribution_candidates']=np.concatenate([s[:,:slots],s[:,slots+window:slots+window+1],raw[:,:18],raw[:,18:]],axis=1)
        if self.duration_onehot:p['attribution_candidates']=append_duration_encoding(p['attribution_candidates'])
        if self.layout is None:
            cg=['G']*len(self.geometry_history.vector)
            names=self.semantic_protocol['state_names'];capacity=self.semantic_protocol['state_capacity']
            for _ in range(self.semantic_protocol['history_length']):
                cg += [('A' if n in CONTROL_NAMES else 'D') for n in names]+['unused']*(capacity-len(names))+['common','common']
            xg=[('unused' if n is None else 'A' if n in CONTROL_NAMES else 'D') for n in self.schema.slots]
            xg += ['A']+['G']*18+['common']*(raw.shape[1]-18)
            if self.duration_onehot:xg += ['common']*40
            self.layout=dict(version=VERSION,context_groups=cg,candidate_groups=xg,semantic_slots=list(self.schema.slots),state_names=list(names),geometry_event_width=len(observation_vector(current_event(history))),geometry_blocks=['initial','mean_all','min_all','max_all']+['recent_'+str(i) for i in range(8)],geometry_note='all public scan ranges/categories; measured entity/patch summary; no full-history sufficiency claim',control_names=sorted(CONTROL_NAMES),duration_onehot=self.duration_onehot)
        assert len(self.layout['context_groups'])==len(p['attribution_context'])
        assert len(self.layout['candidate_groups'])==p['attribution_candidates'].shape[1]
        return p


def append_duration_encoding(x):
    """Encode only the already selected prefix's legal duration, never a label.

    v1 fixed layout: prefix-known ticks flag at 86; seconds at 87. Before
    the duration field is revealed all new slots remain zero, not missing.
    """
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=248:raise ValueError('expected v1 248-column candidates')
    known=x[:,86]==1
    ticks=np.rint(np.where(known,x[:,87],0)*20).astype(np.int64)
    if np.any(known & ((ticks<1)|(ticks>40))):raise ValueError('duration outside public 1..40 support')
    if not np.allclose(x[known,87]*20,ticks[known]):raise ValueError('nonintegral duration')
    out=np.zeros((len(x),40),dtype=x.dtype)
    out[np.flatnonzero(known),ticks[known]-1]=1
    return np.concatenate([x,out],axis=1)


def upgraded_layout(layout,duration_onehot=True):
    import copy
    layout=copy.deepcopy(layout);layout['version']=VERSION
    layout['duration_onehot']=duration_onehot
    if duration_onehot:layout['candidate_groups'] += ['common']*40
    return layout


def feature_mask(groups,arm):
    if arm not in ARMS:raise ValueError('unknown attribution arm')
    return np.array([g=='common' or g in arm for g in groups],dtype=np.float32)


def statistics(arrays,width):
    count=np.zeros(width);total=np.zeros(width)
    for a in arrays:
        a=np.asarray(a,dtype=np.float64).reshape(-1,width);count+=np.isfinite(a).sum(0);total+=np.nansum(a,0)
    mean=total/np.maximum(count,1);squares=np.zeros(width)
    for a in arrays:
        a=np.asarray(a,dtype=np.float64).reshape(-1,width);squares+=np.nansum((a-mean)**2,0)
    std=np.sqrt(squares/np.maximum(count,1))
    return mean.astype('float32'),np.where(std<1e-5,1.,std).astype('float32')


class Scorer(nn.Module):
    def __init__(self,layout,arm,context_stats,candidate_stats,source_sha,action_protocol,*,projection_norm=True):
        super().__init__();self.layout=layout;self.arm=arm;self.semantic_sha256=source_sha;self.action_protocol=action_protocol
        for key,stats,groups in [('context',context_stats,layout['context_groups']),('candidate',candidate_stats,layout['candidate_groups'])]:
            self.register_buffer(key+'_mean',torch.as_tensor(stats[0],dtype=torch.float32))
            self.register_buffer(key+'_std',torch.as_tensor(stats[1],dtype=torch.float32))
            self.register_buffer(key+'_mask',torch.tensor(feature_mask(groups,arm)))
        self.context_linear=nn.Linear(2*len(layout['context_groups']),32,bias=False)
        self.candidate_linear=nn.Linear(2*len(layout['candidate_groups']),32)
        # Fixed unit gain prevents reintroducing arbitrary scale before tanh.
        self.projection_norm=projection_norm
        self.context_norm=nn.LayerNorm(32,elementwise_affine=False) if projection_norm else nn.Identity()
        self.candidate_norm=nn.LayerNorm(32,elementwise_affine=False) if projection_norm else nn.Identity()
        self.tail=nn.Sequential(nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))

    def normalize(self,x,key):
        x=torch.as_tensor(np.asarray(x,dtype=np.float32));missing=torch.isnan(x)
        if not torch.isfinite(x[~missing]).all():raise ValueError('infinite attribution features')
        mean=getattr(self,key+'_mean');std=getattr(self,key+'_std');mask=getattr(self,key+'_mask')
        values=((torch.where(missing,mean,x)-mean)/std).clamp(-10,10)*mask
        return torch.cat([values,missing.float()*mask],-1)

    def logits(self,context,candidates,owners=None,normalized=False):
        c=context if normalized else self.normalize(context,'context')
        x=candidates if normalized else self.normalize(candidates,'candidate')
        cc=self.context_norm(self.context_linear(c))
        xx=self.candidate_norm(self.candidate_linear(x))
        return self.tail(xx+(cc if owners is None else cc[owners])).flatten()

    def decide(self,history,features,*,generator=None,deterministic=True,temperature=1.):
        if not deterministic:raise ValueError('attribution evaluation is deterministic')
        if features.sha256!=self.semantic_sha256 or features.action_protocol!=self.action_protocol:raise ValueError('attribution source/protocol mismatch')
        prefix={};panels=[];indices=[];logps=[]
        with features.decision(history):
            tree=ActionTree(current_event(history)['observation'])
            for key in FIELDS:
                p=features.panel(history,tree,prefix)
                if features.layout!=self.layout:raise ValueError('attribution feature layout mismatch')
                q=self.logits(p['attribution_context'],p['attribution_candidates'])/temperature
                j=int(q.argmax());logps.append(q[j]-q.logsumexp(0));indices.append(j);prefix=p['choices'][j]
                # Replay needs choices/proposals, not the large training tensors.
                panels.append({k:p[k] for k in ('choices','proposals')})
        return dict(action=prefix,panels=panels,indices=indices,logp=float(torch.stack(logps).sum()),value=0.,temperature=temperature,recommended=prefix in panels[0]['proposals'],legal_action_count=tree.count,scored_prefixes=sum(len(p['choices']) for p in panels),action_protocol=self.action_protocol)

    def value(self,panel):return torch.tensor(0.)

    def save(self,path,metadata):
        torch.save(dict(version=VERSION,projection_norm=self.projection_norm,layout=self.layout,arm=self.arm,state=self.state_dict(),source_sha=self.semantic_sha256,action_protocol=self.action_protocol,metadata=metadata),path)

    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d['version']!=VERSION:raise ValueError('attribution checkpoint version')
        s=d['state'];model=cls(d['layout'],d['arm'],(s['context_mean'],s['context_std']),(s['candidate_mean'],s['candidate_std']),d['source_sha'],d['action_protocol'],projection_norm=d['projection_norm']);model.load_state_dict(s);return model


def train(dataset,ids,layout,arm,source_sha,action_protocol,*,updates=16800,seed=17,wall_seconds=1800,progress=None,projection_norm=True):
    """Identical examples/order/architecture/objective per arm; no residual penalty.

    Rows index complete teacher boundaries. Four conditional losses are summed.
    There is no arm-specific regularizer handicapping the geometry-only model.
    """
    torch.set_num_threads(1);torch.manual_seed(seed)
    contexts=dataset['contexts'];candidates=dataset['candidates'];offsets=dataset['offsets'];targets=dataset['targets']
    cstats=statistics([contexts[ids]],contexts.shape[1])
    ranges=[(offsets[i,0],offsets[i,-1]) for i in ids]
    xstats=statistics([candidates[a:b] for a,b in ranges],candidates.shape[1])
    model=Scorer(layout,arm,cstats,xstats,source_sha,action_protocol,projection_norm=projection_norm)
    prepared_c=model.normalize(contexts[ids],'context')
    # Candidate normalization is chunked and restricted to this data subset.
    chunks=[];local_offsets=[];cursor=0
    for i in ids:
        a,b=offsets[i,0],offsets[i,-1];chunks.append(model.normalize(candidates[a:b],'candidate'))
        local_offsets.append(offsets[i]-a+cursor);cursor+=b-a
    prepared_x=torch.cat(chunks);del chunks
    opt=torch.optim.Adam(model.parameters(),lr=.003);curve=[];step=0;epoch=0;start=time.monotonic()
    while step<updates:
        for batch in torch.randperm(len(ids)).split(32):
            if step>=updates:break
            if time.monotonic()-start>wall_seconds:raise TimeoutError('attribution optimizer wall budget')
            xs=[];owners=[];sizes=[];labels=[]
            for j,k in enumerate(batch.tolist()):
                off=local_offsets[k];xs.append(prepared_x[off[0]:off[-1]])
                owners.extend([j]*int(off[-1]-off[0]));sizes.extend(np.diff(off).tolist());labels.extend(targets[ids[k]].tolist())
            q=model.logits(prepared_c[batch],torch.cat(xs),torch.tensor(owners),normalized=True)
            terms=[p.logsumexp(0)-p[j] for p,j in zip(q.split(sizes),labels)]
            loss=torch.stack(terms).sum()/len(batch)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
            if step%100==0 or step==updates:
                row=dict(update=step,epoch=epoch,imitation=float(loss.detach()),elapsed_seconds=time.monotonic()-start);curve.append(row)
                if progress:progress(row)
        epoch+=1
    return model,curve


def accuracy(model,dataset,ids):
    correct=0;fields=np.zeros(4,dtype=int);errors=[]
    with torch.no_grad():
        for i in ids:
            off=dataset['offsets'][i];sizes=np.diff(off).tolist()
            q=model.logits(dataset['contexts'][i],dataset['candidates'][off[0]:off[-1]])
            matches=[int(p.argmax())==int(j) for p,j in zip(q.split(sizes),dataset['targets'][i])]
            fields+=matches;correct+=all(matches)
            if not all(matches):errors.append(int(i))
    return dict(boundaries=len(ids),exact_actions=correct,field_matches=fields.tolist(),error_boundaries=errors)
