"""Replay the common public prefix and describe the first differing joint option."""
import argparse
import ast
import csv
import gzip
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('outputs/semantic_rl')
OUT = ROOT / 'cap_semantic_disagreement_20261119_01'
OLD = [ROOT/'cap_inverse_scale_20261119_01/evaluation/q1001', ROOT/'cap_global_attention_20261119_01/evaluation']
CAPS = ['009','003','001','010','006','007','002']
SOURCE = Path('outputs/transfers/cap_trace_value_dev_v1/policies/CAP-RR-009.py')


def cap_geometry():
    # Only these unannotated arithmetic functions are needed for prefix replay.
    tree = ast.parse(SOURCE.read_text())
    selected = {'initial_memory','dot','dist','direction','estimate','update_memory','scan_points',
                'all_points','clearances','update_navigation_escape'}
    tree.body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected]
    assert len(tree.body) == len(selected)
    env = {'math':math}
    exec(compile(tree,str(SOURCE),'exec'),env)
    return env


CAP = cap_geometry()


def geometry(rows):
    mem = CAP['initial_memory']()
    previous = 0
    for row in rows:
        tick = row['observation']['tick']
        CAP['update_memory'](mem,dict(row,executed_ticks=tick-previous))
        previous = tick
    return mem


def describe(mem,o,a):
    codes = [a['a0'],a['a1']]
    derived = {}
    goal = o['goal']['xy']
    pos = [[r['own']['x'],r['own']['y']] for r in o['robots']]
    g = mem['geom']
    target = o['goal']['target_id']
    pushing = [j for j,v in enumerate(codes) if a['target_id']==target and v<32]
    kind = 'push' if pushing else ('rotate' if all(32<=v<=37 for v in codes) else
                                  ('move' if any(v>=47 for v in codes) else 'other'))
    derived.update(codes=codes,ticks=a['ticks'],kind=kind,target=a['target_id'])
    if g is None:
        return derived
    c=g[:2]; u=g[2:4]; v=[-u[1],u[0]]
    dgoal=[goal[i]-c[i] for i in range(2)]
    gd=math.hypot(*dgoal)
    dgoal=[x/max(.001,gd) for x in dgoal]
    dot=CAP['dot']; normals=[u,[-u[0],-u[1]],v,[-v[0],-v[1]]]
    normal=max(normals,key=lambda n:dot(n,dgoal))
    def rear(n):
        support=.125 if abs(dot(n,v))>.7 else .5
        return [c[i]-(support+.18)*n[i] for i in range(2)]
    staging=rear(normal)
    clear=mem['clearance']
    predicted=[p.copy() for p in pos]
    pushes=[]
    for j,code in enumerate(codes):
        if j in pushing:
            k=code//4; d=CAP['direction'](k)
            held=(o['active_target']==target and o['robots'][j]['contacting_object'] and
                  o['robots'][j]['execution'].get('stage')=='push' and
                  o['robots'][j]['execution'].get('direction_valid') is True)
            pushes.append(dict(driver=j,distance=math.dist(pos[j],rear(d)),alignment=dot(d,dgoal),
                               clearance=clear[k],held=bool(held)))
        elif 47<=code<=78:
            k=(code-47)//4; d=CAP['direction'](k)
            travel=min(a['ticks']*.05*.1*(1+(code-47)%4),max(0,clear[8+8*j+k]))
            predicted[j]=[pos[j][i]+travel*d[i] for i in range(2)]
    before=min(math.dist(p,staging) for p in pos)
    after=min(math.dist(p,staging) for p in predicted)
    derived.update(pushes=pushes,stage_gain=before-after)
    return derived


def load(path):
    with gzip.open(path,'rt') as f:
        return json.load(f)


def extract(job):
    cid,dirs=job
    ga,att=[load(p/cid/'result.json.gz') for p in dirs]
    models=(ga,att)
    assert all(m['summary']['complete'] and m['summary']['error'] is None for m in models)
    out=dict(id=cid,ga=int(ga['summary']['success']),att=int(att['summary']['success']))
    for i,(x,y) in enumerate(zip(ga['decisions'],att['decisions'])):
        row=ga['public_rows'][i]
        assert row['history_hash']==att['public_rows'][i]['history_hash']
        assert x['candidates']==y['candidates'] and x['sources']==y['sources']
        if x['action']==y['action']:
            continue
        o=row['observation']; mem=geometry(ga['public_rows'][:i+1]); g=mem['geom']
        if g:
            mem['clearance']=CAP['clearances'](o,g,mem['target'],mem)
        out.update(step=i+1,tick=o['tick'],hash=row['history_hash'],geometry=g,
                   goal=o['goal']['xy'],goal_distance=math.dist(g[:2],o['goal']['xy']) if g else None,
                   still=mem['still'],stall=mem['stall'],contacts=[r['contacting_object'] for r in o['robots']],
                   robots=[[r['own']['x'],r['own']['y']] for r in o['robots']],
                   stages=[r['execution'].get('stage') for r in o['robots']],previous=row['action'])
        for name,decision,model in zip(('G','T'),(x,y),models):
            a=describe(mem,o,decision['action'])
            a['caps']=[CAPS[j] for j in decision['sources'][decision['chosen']]]
            a['slot']=decision['chosen']
            out[name]=a
            # Audit-only suffix. No selection function takes these fields.
            frames=model['replay']['frames']; target=o['goal']['target_id']
            distances=[math.dist(f['entities'][target][:2],o['goal']['xy']) for f in frames if f['tick']>=o['tick']]
            out['audit_'+name]=dict(stop=model['summary']['termination'],ticks=model['summary']['ticks'],
                                     final=distances[-1],minimum=min(distances))
        return out
    assert len(ga['decisions'])==len(att['decisions']) and out['ga']==out['att']
    assert ga['public_rows'][-1]['history_hash']==att['public_rows'][-1]['history_hash']
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=['discovery','oldcheck','fresh','neighbors'])
    args=parser.parse_args()
    dirs=OLD if args.split in ('discovery','oldcheck') else [OUT/'evaluation'/a for a in ('ga','attention')]
    if dirs==OLD:
        cases=[p.name for p in dirs[0].iterdir() if p.is_dir() and p.name.startswith('inverse_test_')
               and p.name.startswith('inverse_test_2_')==(args.split=='oldcheck')]
    else:
        cases=[r['case_id'] for r in json.loads((OUT/(args.split+'_panel.json')).read_text())]
    assert len(cases)==dict(discovery=400,oldcheck=200,fresh=200,neighbors=96)[args.split]
    with ProcessPoolExecutor(5) as pool:
        rows=list(pool.map(extract,[(c,dirs) for c in sorted(cases)],chunksize=2))
    (OUT/(args.split+'_prefixes.json')).write_text(json.dumps(rows,allow_nan=False))


if __name__=='__main__':
    main()
