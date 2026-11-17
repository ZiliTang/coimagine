"""Hermetic same-history proposal AND feature evaluation, without outcome labels."""
from pathlib import Path
from copy import deepcopy
from itertools import islice
import json,subprocess,tempfile
import numpy as np
from experiments.planar_tasks.sandbox import policy_command
from experiments.planar_tasks.relational_search.action_scale import duration_seconds
from experiments.planar_tasks.relational_search.contract import public_observation,validate_action,canonical,digest
from experiments.planar_tasks.relational_search.live_search import candidates

WRAPPER='''
import json,sys,math
from copy import deepcopy
request=json.loads(sys.stdin.read());memory=initial_memory();previous=None
for event in request['history']:
 o=event['observation'];memory=update_memory(memory,dict(observation=deepcopy(o),action=None if previous is None else deepcopy(event['action']),executed_ticks=None if previous is None else o['tick']-previous))
 if not isinstance(memory,dict) or len(json.dumps(memory,allow_nan=False).encode())>8192:raise ValueError('memory limit')
 previous=o['tick']
if not isinstance(NAMES,(tuple,list)) or not 1<=len(NAMES)<=24 or len(set(NAMES))!=len(NAMES) or any(not isinstance(n,str) or not 0<len(n)<=100 for n in NAMES):raise ValueError('feature names')
proposals=propose_actions(deepcopy(memory),deepcopy(o))
if not isinstance(proposals,list) or len(proposals)>16:raise ValueError('at most16 proposals')
# Model-produced proposals and host candidates are unlabelled. Input order is
# content-hash canonicalized, independent of whether a candidate was demonstrated.
actions=[]
for a in proposals+request['candidates']:
 if a not in actions:actions.append(a)
values=[]
for a in actions:
 v=semantic_features(deepcopy(memory),deepcopy(o),deepcopy(a),deepcopy(proposals))
 if not isinstance(v,dict) or set(v)!=set(NAMES):raise ValueError('feature keys')
 row=[v[n] for n in NAMES]
 if any(x is not None and (type(x) not in (float,int,bool) or not math.isfinite(x)) for x in row):raise ValueError('features numeric')
 values.append(row)
print(json.dumps(dict(names=list(NAMES),proposals=proposals,actions=actions,values=values),allow_nan=False))
'''


def action_encoding(o,a):
    # Shared candidate encoding; IDs are roles, never ordinal class labels.
    v=[duration_seconds(a['ticks']),float(a['target_id'] is None),float(a['target_id']==o['goal']['target_id'])]
    for k in ('a0','a1'):
        one=[0.]*79;one[a[k]]=1.;v.extend(one)
    return v


def proposal_encoding(a,proposals):
    if not proposals:return [0.]*7
    first=proposals[0]
    return [float(a==first),float(a in proposals),float(a['target_id']==first['target_id']),
            float(a['a0']==first['a0']),float(a['a1']==first['a1']),
            (a['ticks']-first['ticks'])*.05,min(abs(a['ticks']-p['ticks'])*.05 for p in proposals if p['a0']==a['a0'] and p['a1']==a['a1'] and p['target_id']==a['target_id']) if any(p['a0']==a['a0'] and p['a1']==a['a1'] and p['target_id']==a['target_id'] for p in proposals) else 2.]


class Program:
    def __init__(self,source,on_call=None,*,max_names=24):
        if not isinstance(source,str) or len(source.encode())>49152:raise ValueError('source at most48KiB')
        if type(max_names) is not int or not 1<=max_names<=128:raise ValueError('semantic capacity 1..128')
        self.max_names=max_names
        self.source=source;self.sha256=digest(source);self.calls=0;self.on_call=on_call

    def evaluate(self,history,extras=(),generic=True):
        o=public_observation(history.events[-1]['observation'])
        supplied=list(islice(candidates(o),48)) if generic else []
        for a in extras:
            validate_action(a,o)
            if a not in supplied:supplied.append(deepcopy(a))
        if len(supplied)>128:raise ValueError('host candidate limit')
        supplied=sorted(supplied,key=lambda a:digest(a))
        request=dict(history=[dict(observation=public_observation(e['observation']),action=e['action']) for e in history.events],candidates=supplied)
        if self.on_call:self.on_call()
        self.calls+=1
        with tempfile.TemporaryDirectory(prefix='coimagine_action_semantics_') as d:
            p=Path(d);(p/'policy.py').write_text(self.source+'\n'+WRAPPER.replace('len(NAMES)<=24',f'len(NAMES)<={self.max_names}'))
            with (p/'out').open('w+b') as out,(p/'err').open('w+b') as err:
                proc=subprocess.run(policy_command(p/'policy.py'),input=canonical(request).encode(),stdout=out,stderr=err,timeout=30)
                out.seek(0);err.seek(0);data=out.read(1048576);error=err.read(2000)
            if proc.returncode:raise ValueError('action semantic worker: '+error.decode(errors='replace'))
        result=json.loads(data);legal=[];invalid=[]
        for a in result['proposals']:
            try:validate_action(a,o)
            except (ValueError,TypeError,KeyError) as exc:invalid.append(dict(action=a,error=str(exc)));continue
            if a not in legal:legal.append(a)
        actions=[];values=[]
        for a,v in zip(result['actions'],result['values']):
            try:validate_action(a,o)
            except (ValueError,TypeError,KeyError):continue
            if a not in actions:actions.append(a);values.append(v)
        # The first *raw* proposal is the CAP-equivalence contract. An illegal
        # first proposal is explicit failure, not silently replaced by a later one.
        first=result['proposals'][0] if result['proposals'] else None
        return dict(names=result['names'],proposals=legal,raw_first=first,first_legal=first in legal if first else False,
                    invalid_proposals=invalid,actions=actions,values=values,
                    semantic=[v+proposal_encoding(a,legal)+action_encoding(o,a) for a,v in zip(actions,values)])
