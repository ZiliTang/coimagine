"""Frozen experience retrieval and causal CAP routing, without simulator access.

R0 retrieves from initial outcome tables and keeps one closed-loop program.
Rt retrieves from successful trajectory boundaries at each decision. Routing
is separate from CAP execution so a learned selector can use the same contract.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Protocol

import numpy as np

from experiments.planar_tasks.relational_search.contract import digest
from .features import Features
from .relational_transfer_features import RelativeGeometry
from .stream_history import StreamHistory

VERSION = 'program-retrieval-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GeometryEncoder:
    """Same public entity/scan geometry as G; no CAP advice or task identity.

    Snapshot retains the current vector in the episode's initial goal-relative
    frame, including the G encoder's last-seen entity cache and previous action.
    Thus it is a compact current estimate, not a history-free raw image.
    History uses exactly the initial plus eight-boundary G representation.
    """
    def __init__(self, representation='snapshot'):
        if representation not in ('snapshot','history'):
            raise ValueError('unknown representation')
        self.representation=representation
        self.geometry=RelativeGeometry(8)

    def advance(self, history):
        self.geometry.advance(history)
        return np.array(self.geometry.frames[-1] if self.representation=='snapshot' else self.geometry.vector, dtype=np.float32, copy=True)


class Router(Protocol):
    """Replacement boundary for learned routing; no environment/fixture input."""
    names: list
    representation: str
    identity: str
    metadata: dict

    def select(self, vector: np.ndarray) -> dict:
        """Return cap name plus inspectable evidence; unknown outcomes stay unknown."""
        ...


class RetrievalIndex:
    def __init__(self, vectors, labels, names, metadata, *, evidence, representation='snapshot', k=1):
        self.vectors=np.asarray(vectors,dtype=np.float32)
        self.labels=np.asarray(labels,dtype=float)
        self.names=list(names)
        self.metadata=dict(metadata)
        self.evidence=evidence
        self.representation=representation
        self.k=int(k)
        if evidence not in ('initial_outcomes','successful_prefix') or representation not in ('snapshot','history'):
            raise ValueError('index protocol')
        if len(set(names))!=len(names) or not names or self.k<1:
            raise ValueError('names/k')
        if self.vectors.ndim!=2 or len(self.vectors)==0 or np.isinf(self.vectors).any():
            raise ValueError('nonempty finite-or-missing matrix required')
        if self.labels.shape!=(len(self.vectors),len(names)):
            raise ValueError('outcome shape')
        finite=np.isfinite(self.labels)
        if not np.isin(self.labels[finite],[0,1]).all():
            raise ValueError('outcomes must be0/1/unknown')
        if evidence=='successful_prefix' and not ((self.labels==1).sum(1)==1).all():
            raise ValueError('each prefix has one witnessed successful CAP')
        self.mean,self.scale=self.statistics(self.vectors)
        self.encoded=self.encode(self.vectors)
        counts=finite.sum(0)
        self.global_success=np.divide(np.nansum(self.labels,axis=0),counts,out=np.full(len(names),np.nan),where=counts>0)
        if not np.isfinite(self.global_success).any():
            raise ValueError('no observed outcomes')
        self.identity=digest(dict(version=VERSION,names=names,evidence=evidence,representation=representation,k=k,metadata=metadata,
            vector_sha=hashlib.sha256(self.vectors.tobytes()).hexdigest(),labels_sha=hashlib.sha256(self.labels.tobytes()).hexdigest()))

    @staticmethod
    def statistics(x):
        valid=np.isfinite(x)
        n=valid.sum(0)
        mean=np.divide(np.where(valid,x,0).sum(0),n,out=np.zeros(x.shape[1]),where=n>0)
        var=np.divide((np.where(valid,x-mean,0)**2).sum(0),n,out=np.zeros(x.shape[1]),where=n>0)
        return mean,np.maximum(np.sqrt(var),.1)

    def encode(self, x):
        x=np.asarray(x)
        if x.shape[-1]!=self.vectors.shape[1] or np.isinf(x).any():
            raise ValueError('query schema/nonfinite error')
        mask=np.isfinite(x)
        return np.concatenate([np.where(mask,(x-self.mean)/self.scale,0),mask.astype(float)],axis=-1).astype(np.float32)

    def select(self, vector):
        q=self.encode(vector)
        if q.ndim!=1:
            raise ValueError('one query required')
        # Deterministic exact search. Chunking bounds temporary distance storage.
        distances=[]
        for start in range(0,len(self.encoded),1024):
            d=self.encoded[start:start+1024]-q
            distances.append(np.einsum('ij,ij->i',d,d)/len(q))
        distances=np.concatenate(distances)
        neighbors=np.argsort(distances,kind='stable')[:min(self.k,len(distances))]
        y=self.labels[neighbors]
        weights=1/(1+distances[neighbors])
        known=np.isfinite(y)
        denominator=(weights[:,None]*known).sum(0)
        scores=np.divide((weights[:,None]*np.nan_to_num(y,nan=0)).sum(0),denominator,
                         out=np.full(len(self.names),np.nan),where=denominator>0)
        if self.evidence=='successful_prefix':
            # Votes from witnessed successful prefixes, NOT empirical failure
            # probabilities for untried receiving programs.
            scores=(weights[:,None]*(y==1)).sum(0)/weights.sum()
        fallback=not np.any(np.isfinite(scores)&(scores>0))
        use=self.global_success if fallback else scores
        chosen=int(np.argmax(np.where(np.isfinite(use),use,-np.inf)))
        return dict(cap=self.names[chosen],fallback=fallback,
            score_kind='success_vote' if self.evidence=='successful_prefix' else 'observed_local_success_rate',
            scores=[float(x) if np.isfinite(x) else None for x in scores],
            neighbors=[dict(row=int(i),distance=float(distances[i])) for i in neighbors])

    def save(self, path):
        path=Path(path)
        path.mkdir(parents=True,exist_ok=False)
        np.savez(path/'arrays.npz',vectors=self.vectors,labels=self.labels)
        info=dict(version=VERSION,names=self.names,metadata=self.metadata,evidence=self.evidence,
            representation=self.representation,k=self.k,identity=self.identity,arrays_sha256=sha(path/'arrays.npz'))
        (path/'index.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        path=Path(path)
        info=json.loads((path/'index.json').read_text())
        if info['version']!=VERSION or sha(path/'arrays.npz')!=info['arrays_sha256']:
            raise ValueError('index integrity failure')
        with np.load(path/'arrays.npz',allow_pickle=False) as a:
            index=cls(a['vectors'],a['labels'],info['names'],info['metadata'],evidence=info['evidence'],representation=info['representation'],k=info['k'])
        if index.identity!=info['identity']:
            raise ValueError('index identity failure')
        return index


class RoutedFeatures:
    """Adapter for existing rollout(features=..., model=None).

    Every member observes the SAME actual history, including when unselected.
    No hypothetical recommendation is fed back as an executed action.
    """
    runtime='typed_stream'

    def __init__(self,entries,router:Router,action_protocol,*,mode='initial',backend='trusted_inprocess'):
        if mode not in ('initial','boundary'):
            raise ValueError('routing mode')
        if [e['name'] for e in entries]!=router.names:
            raise ValueError('router/library order mismatch')
        expected={e['name']:digest(e['source']) for e in entries}
        if router.metadata.get('sources')!=expected or router.metadata.get('action_protocol')!=action_protocol:
            raise ValueError('router/library knowledge seal mismatch')
        self.entries=entries;self.router=router;self.mode=mode;self.action_protocol=action_protocol
        self.source=json.dumps(dict(version=VERSION,library=expected,index=router.identity,mode=mode),sort_keys=True)
        self.sha256=digest(self.source)
        self.semantic_protocol=dict(version=VERSION,mode=mode,representation=router.representation,index=router.identity)
        self.members=[]
        try:
            for e in entries:
                self.members.append(Features(e['source'],action_protocol=action_protocol,execution_backend=backend,
                    trusted_source_sha256=digest(e['source']) if backend=='trusted_inprocess' else None))
        except BaseException:
            self.close();raise
        self.encoder=GeometryEncoder(router.representation)
        self.episode=None;self.last_index=-1;self.trace=[];self.selected=None

    def new_history(self):
        return StreamHistory()

    def close(self):
        for member in self.members:
            member.close()

    @contextmanager
    def decision(self,history):
        if history.episode!=self.episode:
            if history.index!=0:
                raise ValueError('routing requires complete real prefix')
            self.episode=history.episode;self.last_index=-1;self.selected=None;self.trace=[]
            self.encoder=GeometryEncoder(self.router.representation)
        if history.index not in (self.last_index,self.last_index+1):
            raise ValueError('skipped actual history')
        started=time.monotonic()
        with ExitStack() as stack:
            for member in self.members:
                stack.enter_context(member.decision(history))
            self.query=self.encoder.advance(history)
            if self.selected is None and self.mode=='initial':
                self.initial_selection=self.router.select(self.query)
                self.selected=self.initial_selection['cap']
            self.last_index=history.index
            self.prepare_seconds=time.monotonic()-started
            yield

    def recommendation(self,history):
        with self.decision(history):
            started=time.monotonic()
            route=self.initial_selection if self.mode=='initial' else self.router.select(self.query)
            retrieval_seconds=time.monotonic()-started
            selected=route['cap']
            if selected not in self.router.names:
                raise ValueError('unknown routed CAP')
            old=self.selected
            self.selected=selected
            started=time.monotonic()
            packet=self.members[self.router.names.index(selected)].worker.recommendation()
            # Preserve first-legal behavior of standalone CAP. Do not rescue an
            # illegal advice by silently trying all other experts.
            self.trace.append(dict(index=history.index,history_hash=history.sha256,selected=selected,
                switched=old is not None and old!=selected,route=deepcopy(route),
                action=deepcopy(packet.get('raw_first')),first_legal=packet['first_legal'],
                prepare_seconds=self.prepare_seconds,retrieval_seconds=retrieval_seconds,
                recommendation_seconds=time.monotonic()-started))
            return packet
