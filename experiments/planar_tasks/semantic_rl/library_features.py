"""Flat union of separately authored causal programs, not a discrete expert router.

Each program consumes the SAME actually executed public history. Descriptive and
recommendation channels are concatenated; the scorer retains the full legal tree.
"""
from contextlib import contextmanager,ExitStack
import json
import numpy as np
from .features import Features
from .relational_transfer_features import TransferFeatures
from experiments.planar_tasks.relational_search.contract import digest

VERSION='flat-cap-library-features-v1'

class LibraryFeatures:
    def __init__(self,entries,action_protocol,execution_backend='isolated'):
        self.entries=entries;self.action_protocol=action_protocol;self.runtime='typed_stream';self.layout=None
        self.source=json.dumps([(e['name'],e['source'],e['groups']) for e in entries],sort_keys=True)
        self.sha256=digest(self.source);self.members=[]
        try:
            for e in entries:
                self.members.append(TransferFeatures(e['source'],groups=e['groups'],fast_embedding=True,action_protocol=action_protocol,execution_backend=execution_backend,trusted_source_sha256=digest(e['source']) if execution_backend=='trusted_inprocess' else None))
        except Exception:
            self.close();raise
        if not self.members:raise ValueError('nonempty frozen library required')
        self.semantic_protocol=dict(version=VERSION,members=[dict(name=e['name'],sha256=digest(e['source']),protocol=f.semantic_protocol) for e,f in zip(entries,self.members)])

    @property
    def calls(self):return sum(f.calls for f in self.members)
    def new_history(self):return self.members[0].new_history()
    def close(self):
        for f in self.members:f.close()
    @contextmanager
    def decision(self,history):
        with ExitStack() as stack:
            for f in self.members:stack.enter_context(f.decision(history))
            yield
    def panel(self,history,tree,prefix):
        panels=[f.panel(history,tree,prefix) for f in self.members]
        first=panels[0];contexts=[];candidates=[];cg=[];xg=[];proposals=[]
        for i,(f,p) in enumerate(zip(self.members,panels)):
            if p['choices']!=first['choices']:raise ValueError('library changed legal support')
            cm=np.array([g!='unused' and (i==0 or g in ('D','A','common')) for g in f.layout['context_groups']])
            xm=np.array([g!='unused' and (i==0 or g in ('D','A')) for g in f.layout['candidate_groups']])
            contexts.append(p['attribution_context'][cm]);candidates.append(p['attribution_candidates'][:,xm])
            cg.extend(g for g,v in zip(f.layout['context_groups'],cm) if v)
            xg.extend(g for g,v in zip(f.layout['candidate_groups'],xm) if v)
            for a in p['proposals']:
                if a not in proposals:proposals.append(a)
        if self.layout is None:
            self.layout=dict(version=VERSION,context_groups=cg,candidate_groups=xg,member_layouts=[f.layout for f in self.members],members=[e['name'] for e in self.entries],note='flat semantic channels; one shared public geometry encoding; full public conditional action support; each program updated from actual shared history')
        return dict(choices=first['choices'],proposals=proposals,attribution_context=np.concatenate(contexts),attribution_candidates=np.concatenate(candidates,axis=1),action_protocol=self.action_protocol,history_hash=history.sha256)

class GeometryOnlyLibraryFeatures:
    """Exact G projection without evaluating any CAP; no advice can break G runs."""
    def __init__(self,entries,action_protocol,layout):
        from .relational_transfer_features import RelativeGeometry
        self.source=json.dumps([(e['name'],e['source'],e['groups']) for e in entries],sort_keys=True)
        self.sha256=digest(self.source);self.action_protocol=action_protocol;self.layout=layout
        self.runtime='typed_stream';self.calls=0;self.episode=None;self.index=-1
        self.semantic_protocol=dict(version=VERSION,geometry_only_projection=True,members=layout['members'])
        self.lengths=[p['history_length'] for p in layout['member_layouts']]
        self.geometry=RelativeGeometry(self.lengths[0])
    def close(self):pass
    def new_history(self):
        from .stream_history import StreamHistory
        return StreamHistory()
    def advance(self,h):
        from .memory_flow import current_event
        self.geometry.advance(h)
        if self.episode!=h.episode:self.episode=h.episode;self.index=-1;self.ticks=[]
        if self.index!=h.index:
            if h.index!=self.index+1:raise ValueError('skipped geometry-only history')
            self.ticks.append(current_event(h)['observation']['tick']);self.ticks=self.ticks[-max(self.lengths):];self.index=h.index
    @contextmanager
    def decision(self,h):
        self.advance(h);yield
    def panel(self,h,tree,prefix):
        from .relational_transfer_features import action_encoding
        self.advance(h);choices=tree.choices(prefix)
        c=np.full(len(self.layout['context_groups']),np.nan);x=np.full((len(choices),len(self.layout['candidate_groups'])),np.nan)
        cg=np.array(self.layout['context_groups']);xg=np.array(self.layout['candidate_groups'])
        c[cg=='G']=self.geometry.vector
        common=[]
        for n in self.lengths:
            ticks=self.ticks[-n:];common.extend([v for _ in range(n-len(ticks)) for v in (0.,np.nan)])
            common.extend(v for t in ticks for v in (1.,(self.ticks[-1]-t)*.05))
        c[cg=='common']=common
        x[:,xg=='G']=np.array([self.geometry.candidate_geometry(q) for q in choices])
        x[:,xg=='common']=np.array([action_encoding(q,tree.observation['goal']['target_id'],self.geometry.theta) for q in choices])
        return dict(choices=choices,proposals=[],attribution_context=c,attribution_candidates=x,action_protocol=self.action_protocol,history_hash=h.sha256)
