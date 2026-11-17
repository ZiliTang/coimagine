"""Predeclared training-only canary:64 boundaries,2000 updates, four arms."""
from pathlib import Path
import argparse,json
import numpy as np
from . import relational_transfer_study as R
from .input_attribution_repair_probe import activations
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as I


def compact(ds,ids):
    rows=[];off=[];cursor=0
    for i in ids:
        a,b=ds['offsets'][i,0],ds['offsets'][i,-1];rows.append(np.array(ds['candidates'][a:b]));off.append(ds['offsets'][i]-a+cursor);cursor+=b-a
    return dict(contexts=np.array(ds['contexts'][ids]),candidates=np.concatenate(rows),offsets=np.asarray(off),targets=np.array(ds['targets'][ids]))


def run(data,out):
    data=Path(data);out=Path(out);out.mkdir(exist_ok=False,parents=True);ds={k:np.load(data/(k+'.npy'),mmap_mode='r') for k in ['contexts','candidates','offsets','targets']}
    n=min(64,len(ds['contexts']));ids=np.random.default_rng(2026111714).choice(len(ds['contexts']),n,replace=False).tolist();ds=compact(ds,ids);ex=R.S.read(data/'extraction.json');layout=R.S.read(data/'layout.json');results={}
    R.S.save(out/'protocol.json',dict(training_boundary_ids=ids,updates=2000,seed=17,selection_seed=2026111714,gate='each arm >=95% exact teacher actions and <20% saturated first-layer units; training-only diagnostic, not model selection',no_evaluation=True))
    for arm in I.ARMS:
        model,curve=I.train(ds,list(range(n)),layout,arm,ex['source_sha'],ex['action_protocol'],updates=2000,wall_seconds=180)
        accuracy=I.accuracy(model,ds,list(range(n)));activation=activations(model,ds,list(range(n)))
        results[arm]=dict(accuracy=accuracy,activation=activation);model.save(out/(arm+'.pt'),dict(canary=True));R.S.save(out/(arm+'.curve.json'),curve);print('CANARY',arm,accuracy['exact_actions'],'/',n,flush=True)
    accepted=all(v['accuracy']['exact_actions']/n>=.95 and v['activation']['saturated_fraction']<.2 for v in results.values());R.S.save(out/'RESULT.json',dict(accepted=accepted,arms=results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.data,a.output)
