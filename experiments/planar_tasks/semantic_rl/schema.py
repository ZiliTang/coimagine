"""Append-only named semantic slots and explicit checkpoint migration."""
from dataclasses import dataclass
from copy import deepcopy
import torch
from .policy import Policy


@dataclass(frozen=True)
class Schema:
    slots: tuple

    def __post_init__(self):
        names=[n for n in self.slots if n is not None]
        if not 1<=len(self.slots)<=128 or not names or len(set(names))!=len(names) or any(not isinstance(n,str) or not n for n in names):
            raise ValueError('invalid semantic slots')

    @classmethod
    def create(cls,names,capacity=64):
        if len(names)>capacity:raise ValueError('semantic capacity exhausted')
        return cls(tuple(names)+tuple([None]*(capacity-len(names))))

    def extend(self,names):
        if len(set(names))!=len(names) or not set(n for n in self.slots if n is not None)<=set(names):
            raise ValueError('existing semantic names cannot be removed or renamed')
        slots=list(self.slots)
        for n in names:
            if n not in slots:
                if None not in slots:raise ValueError('semantic capacity exhausted; explicit wider migration required')
                slots[slots.index(None)]=n
        return Schema(tuple(slots))

    def encode(self,names,row):
        if set(names)!={n for n in self.slots if n is not None} or len(row)!=len(names):
            raise ValueError('semantic schema mismatch')
        values=dict(zip(names,row))
        return [0. if n is None else values[n] for n in self.slots]

    def to_dict(self):return dict(version='named-semantic-slots-v1',slots=list(self.slots))


def migrate(model,names,new_sha256,*,capacity=64,calibration=None):
    """Version boundary only: copy old coordinates; zero new input weights.

    Old scale statistics stay fixed. New slots default to mean0/std1 (clipped
    by Scale); optional calibration contains TRAIN-only scalar rows for new names.
    Optimizer MUST be recreated; no old on-policy batches are reusable.
    """
    old_schema=Schema(tuple(model.schema['slots'])) if model.schema else None
    if old_schema:
        schema=old_schema.extend(names);old_slots=len(old_schema.slots)
        old_names={n:i for i,n in enumerate(old_schema.slots) if n is not None}
    else:
        if new_sha256 != model.semantic_sha256:
            raise ValueError('migrate legacy model with its original semantic names/source first')
        old_slots=len(names);old_names={n:i for i,n in enumerate(names)}
        schema=Schema.create(names,capacity)
    old_dim=len(model.s.mean);tail=old_dim-old_slots
    history_tail=0 if model.semantic_protocol is None else model.semantic_protocol['history_length']*(model.semantic_protocol['state_capacity']+2)
    if tail!=166+history_tail:raise ValueError('unexpected prefix feature layout')
    new_dim=len(schema.slots)+tail
    target=Policy([dict(semantic=[[0.]*new_dim],raw=[[0.]*len(model.e.mean)],semantic_protocol=model.semantic_protocol)],new_sha256,model.condition)
    target.schema=schema.to_dict()
    target.action_protocol=model.action_protocol
    mapping={i:schema.slots.index(n) for n,i in old_names.items()}
    mapping.update({old_slots+i:len(schema.slots)+i for i in range(tail)})
    # Preserve all non-input parameters; assign the expanded first-layer columns.
    state=deepcopy(model.state_dict());dest=target.state_dict()
    for k in dest:
        if k in ('s.mean','s.std','semantic.0.weight','critic.0.weight'):continue
        dest[k]=state[k].clone()
    dest['s.mean'].zero_();dest['s.std'].fill_(1.)
    dest['semantic.0.weight'].zero_();dest['critic.0.weight'].zero_()
    for old,new in mapping.items():
        dest['s.mean'][new]=state['s.mean'][old];dest['s.std'][new]=state['s.std'][old]
        for offset_old,offset_new in [(old,new),(old+old_dim,new+new_dim)]:
            dest['semantic.0.weight'][:,offset_new]=state['semantic.0.weight'][:,offset_old]
            dest['critic.0.weight'][:,offset_new]=state['critic.0.weight'][:,offset_old]
    dest['critic.0.weight'][:,2*new_dim:]=state['critic.0.weight'][:,2*old_dim:]
    if calibration is not None:
        if calibration.get('split')!='train':raise ValueError('new scale calibration must be train only')
        for name,rows in calibration['values'].items():
            if name in old_names or name not in schema.slots:raise ValueError('only new slots may be calibrated')
            x=torch.tensor([v for v in rows if v is not None],dtype=torch.float32)
            if not len(x) or not torch.isfinite(x).all():raise ValueError('invalid calibration')
            i=schema.slots.index(name);dest['s.mean'][i]=x.mean();std=x.std(unbiased=False)
            dest['s.std'][i]=std if std>=1e-5 else 1.
    target.load_state_dict(dest)
    return target,dict(old_source=model.semantic_sha256,new_source=new_sha256,schema=schema.to_dict(),
                       optimizer='reset required',old_batches='discard; recollect with new source',
                       new_scale='train calibration' if calibration is not None else 'mean0/std1; Scale clip +/-10')
