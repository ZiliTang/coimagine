"""Same-history inference batching; labels and other histories never enter."""
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from experiments.planar_tasks.sandbox import policy_command
from .contract import canonical,digest,public_observation,validate_action
from .pipeline_semantics import embedding
from .fit import Ranker

WRAPPER='''
import json,sys,math
from copy import deepcopy
r=json.loads(sys.stdin.read())
if not isinstance(NAMES,(tuple,list)) or not 1<=len(NAMES)<=16 or len(set(NAMES))!=len(NAMES):raise ValueError('names')
memory=initial_memory();previous=None
for e in r['history']:
    o=e['observation']
    memory=update_memory(memory,dict(observation=deepcopy(o),action=None if previous is None else deepcopy(e['action']),executed_ticks=None if previous is None else o['tick']-previous))
    if not isinstance(memory,dict) or len(json.dumps(memory,allow_nan=False).encode())>8192:raise ValueError('memory')
    previous=o['tick']
rows=[]
for a in r['actions']:
    values=semantic_features(deepcopy(memory),deepcopy(r['history'][-1]['observation']),deepcopy(a))
    if set(values)!=set(NAMES):raise ValueError('schema')
    row=[values[k] for k in NAMES]
    if any(v is not None and (type(v) not in (int,float,bool) or not math.isfinite(v)) for v in row):raise ValueError('features')
    rows.append(row)
print(json.dumps(dict(names=list(NAMES),values=rows),allow_nan=False))
'''


class FittedScorer:
    def __init__(self,source,fitted,condition='M'):
        if digest(source)!=fitted['semantic_sha256']:raise ValueError('semantic/fit version mismatch')
        if condition not in ('S','E','M'):raise ValueError('condition')
        self.source=source;self.fitted=fitted;self.condition=condition
        self.model=Ranker.restore(fitted['models'][condition]['checkpoint'])
        self.base=Ranker.restore(fitted['models'][condition]['semantic_checkpoint']) if condition=='M' else None
        self.calls=0;self.candidates=0

    def __call__(self,history,actions):
        if not 1<=len(actions)<=256:raise ValueError('inference batch bound')
        for a in actions:validate_action(a,history.events[-1]['observation'])
        self.calls+=1;self.candidates+=len(actions)
        semantic=None
        if self.condition!='E':
            request=dict(history=[dict(observation=public_observation(e['observation']),action=e['action']) for e in history.events],actions=actions)
            with tempfile.TemporaryDirectory(prefix='coimagine_live_features_') as tmp:
                p=Path(tmp);(p/'policy.py').write_text(self.source+'\n'+WRAPPER)
                with (p/'stdout').open('w+b') as out,(p/'stderr').open('w+b') as err:
                    r=subprocess.run(policy_command(p/'policy.py'),input=canonical(request).encode(),stdout=out,stderr=err,timeout=30)
                    out.seek(0);err.seek(0);data=out.read(1048576);error=err.read(2000)
                if r.returncode:raise RuntimeError('feature worker: '+error.decode(errors='replace'))
            value=json.loads(data)
            if value['names']!=self.fitted['semantic_names'] or len(value['values'])!=len(actions):raise ValueError('inference schema')
            semantic=value['values']
        features=semantic if self.condition=='S' else [embedding(history,a) for a in actions]
        scores=self.model.score(features)
        if self.base is not None:scores+=self.base.score(semantic)
        if not np.isfinite(scores).all():raise ValueError('nonfinite scores')
        return scores.tolist()
