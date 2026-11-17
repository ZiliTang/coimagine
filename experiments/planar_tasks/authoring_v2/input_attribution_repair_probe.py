"""Training-only, predeclared 2x2 repair probe; no physics or held-out labels."""
from pathlib import Path
import json, time
import numpy as np
import torch
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as I
from . import input_attribution_study as V1

HOST=Path('outputs/semantic_rl_authors/input_attribution_v2_20261117_01')

def compact(ds,ids,onehot):
    rows=[];offsets=[];cursor=0
    for i in ids:
        off=ds['offsets'][i];rows.append(np.array(ds['candidates'][off[0]:off[-1]]))
        offsets.append(off-off[0]+cursor);cursor+=int(off[-1]-off[0])
    x=np.concatenate(rows)
    return dict(contexts=np.array(ds['contexts'][ids]),candidates=I.append_duration_encoding(x) if onehot else x,offsets=np.array(offsets),targets=np.array(ds['targets'][ids]))

def activations(model,ds,ids):
    saturated=[];derivative=[];ranges=[];ties=[]
    with torch.no_grad():
        for i in ids:
            off=ds['offsets'][i];c=model.normalize(ds['contexts'][i],'context');x=model.normalize(ds['candidates'][off[0]:off[-1]],'candidate')
            z=model.context_norm(model.context_linear(c))+model.candidate_norm(model.candidate_linear(x))
            saturated.append(float((z.abs()>3).float().mean()));derivative.append(float((1-z.tanh().square()).mean()))
            q=model.tail(z).flatten()[(off[-2]-off[0]):];ranges.append(float(q.max()-q.min()));ties.append(int((q==q.max()).sum()))
    return dict(saturated_fraction=float(np.mean(saturated)),mean_tanh_derivative=float(np.mean(derivative)),duration_logit_range=float(np.mean(ranges)),duration_max_ties=float(np.mean(ties)))

def main():
    torch.set_num_threads(1);HOST.mkdir(parents=True,exist_ok=True)
    path=HOST/'probe';path.mkdir()
    seals={str(V1.DATA):V1.S.Campaign(V1.DATA).verify_frozen()}
    ds=V1.dataset();layout=V1.S.read(V1.DATA/'layout.json');ex=V1.S.read(V1.DATA/'extraction.json')
    ids=np.random.default_rng(2026111705).choice(len(ds['contexts']),128,replace=False).tolist()
    protocol=dict(training_boundary_ids=ids,subset_seed=2026111705,fit_seed=17,updates=2000,batch=32,lr=.003,arms=['G'],factors=['projection_layernorm','duration_onehot40'],no_physics=True,no_test_inputs=True,seals=seals)
    V1.S.save(path/'protocol.json',protocol);results={}
    for norm in (False,True):
        for hot in (False,True):
            name=f'norm{int(norm)}_onehot{int(hot)}';data=compact(ds,ids,hot);start=time.monotonic()
            model,curve=I.train(data,list(range(128)),I.upgraded_layout(layout,hot),'G',ex['source_sha'],ex['action_protocol'],updates=2000,wall_seconds=180,projection_norm=norm)
            result=dict(accuracy=I.accuracy(model,data,list(range(128))),activation=activations(model,data,list(range(128))),seconds=time.monotonic()-start,curve=curve)
            model.save(path/(name+'.pt'),dict(probe=True));V1.S.save(path/(name+'.json'),result);results[name]=result
            print(name,result['accuracy']['exact_actions'],result['accuracy']['field_matches'],result['activation'],flush=True)
    V1.S.save(path/'RESULT.json',results)

if __name__=='__main__':main()
