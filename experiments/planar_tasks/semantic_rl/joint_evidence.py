"""Controlled candidate-conditioned D/A evidence over frozen CAP programs.

The enriched and joint arms receive identical operands. The joint arm additionally
exposes bounded products linking a program's measured conditions to compatibility
with that SAME program's proposal. No outcomes, task IDs or hidden state enter.
Original CAP channels and all legal choices remain intact. This first version
tests binding/parameter evidence, not new predicate invention or a world model.
"""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
from .handoff_features import HandoffLibraryFeatures as LibraryFeatures
from .relational_transfer_features import action_vector
from experiments.planar_tasks.relational_search.contract import digest

VERSION='candidate-bound-da-evidence-v1'
MODES=('current','enriched','joint')
CONDITIONS=('target_seen','target_estimate_age','target_contacts','blocked_robots',
            'object_progress_5s','stagnation_seconds','nav0_distance','nav1_distance')
SCALES=np.array([1.,5.,2.,2.,.5,5.,2.,2.])
# 8 conditions; 26 proposal operands; 12 compatibilities; 8x4 bound products.
WIDTH=78


def compatibility(prefix,proposal):
    """Unknown fields remain missing, including target-free versus unchosen target."""
    out=[float(prefix['target_id']==proposal['target_id']) if 'target_id' in prefix else np.nan]
    for key in ('a0','a1'):
        if key not in prefix:
            out += [np.nan]*4
            continue
        x,y=action_vector(prefix[key],0),action_vector(proposal[key],0)
        same=float(np.argmax(x[:8])==np.argmax(y[:8]))
        dot=float(x[9:]@y[9:]) if np.linalg.norm(x[9:]) and np.linalg.norm(y[9:]) else np.nan
        out += [same,dot,(x[8]-y[8])/.4,float(prefix[key]==proposal[key])]
    out += [float(prefix['a0']==proposal['a0'] and prefix['a1']==proposal['a1'])
            if 'a0' in prefix and 'a1' in prefix else np.nan,
            (prefix['ticks']-proposal['ticks'])/40. if 'ticks' in prefix else np.nan,
            float(all(v==proposal[k] for k,v in prefix.items()))]
    return np.asarray(out,dtype=float)


def evidence_block(state,proposal,prefix,goal_id,theta,legal,mode):
    if mode not in MODES:raise ValueError('unknown evidence mode')
    if mode=='current':return np.zeros(WIDTH)
    conditions=np.array([np.nan if state.get(n) is None else state[n] for n in CONDITIONS])/SCALES
    conditions=np.clip(conditions,-2,2)
    if proposal is None:
        return np.concatenate([conditions,np.full(WIDTH-8,np.nan)])
    operands=np.concatenate([[float(legal),float(proposal['target_id'] is None),
        float(proposal['target_id']==goal_id),proposal['ticks']/40.],
        action_vector(proposal['a0'],theta),action_vector(proposal['a1'],theta)])
    c=compatibility(prefix,proposal)
    # Each gate is conditional on the selected target, so unrelated entities
    # cannot borrow another proposal's support. Incomplete pairs stay unknown.
    gates=np.array([c[0],c[0]*c[4],c[0]*c[8],c[0]*c[9]])
    bound=(conditions[:,None]*gates[None,:]).ravel() if mode=='joint' else np.zeros(32)
    return np.concatenate([conditions,operands,c,bound])


class JointEvidenceFeatures:
    def __init__(self,entries,action_protocol,mode):
        if mode not in MODES:raise ValueError('mode')
        self.base=LibraryFeatures(entries,action_protocol,execution_backend='trusted_inprocess')
        self.mode=mode;self.action_protocol=action_protocol;self.runtime=self.base.runtime
        self.source=self.base.source;self.sha256=digest(dict(base=self.base.sha256,version=VERSION,mode=mode))
        self.semantic_protocol=dict(self.base.semantic_protocol,evidence_version=VERSION,evidence_mode=mode)
        self.layout=None

    @property
    def calls(self):return self.base.calls
    def close(self):self.base.close()
    def new_history(self):return self.base.new_history()
    @contextmanager
    def decision(self,history):
        with self.base.decision(history):yield

    def panel(self,history,tree,prefix):
        p=self.base.panel(history,tree,prefix)
        blocks=[]
        for member in self.base.members:
            state=dict(zip(member.semantic_protocol['state_names'],member.worker.frames[-1]['state']))
            rec=member.worker.recommendation()
            blocks.append(np.array([evidence_block(state,rec['raw_first'],q,
                tree.observation['goal']['target_id'],self.base.members[0].geometry.theta,
                rec['first_legal'],self.mode) for q in p['choices']]))
        p['attribution_candidates']=np.concatenate([p['attribution_candidates'],*blocks],axis=1)
        if self.layout is None:
            self.layout=deepcopy(self.base.layout)
            self.layout.update(evidence_version=VERSION,evidence_mode=self.mode,
                evidence_conditions=list(CONDITIONS),evidence_width_per_cap=WIDTH,
                evidence_note='same raw operands for enriched/joint; joint adds explicit32 products; current padded with zeros; original semantics retained')
            self.layout['candidate_groups']+=['common']*(WIDTH*len(self.base.members))
        return p
