"""Frozen CAP banks crossed with PUSH execution and observable routing features."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from experiments.option_rl_v19.unknown_shape import direction_matches
from experiments.planar_tasks.relational_search.motion import ACTIONS
from experiments.planar_tasks.semantic_rl import push_phases
from experiments.planar_tasks.semantic_rl.candidate_semantics import endpoints
from experiments.planar_tasks.semantic_rl.corridor_semantics import corridor_semantics
from experiments.planar_tasks.semantic_rl.phase_aware_cap import ContactAwareCAP, position, signature, translating
from experiments.planar_tasks.semantic_rl.push_contact_recovery import clearance
from coimagine.random_start.spec import ROBOT_RADIUS
from tools import cap_push_phase_comparison as phase
from tools import cap_semantic_library_eval as old
from tools import cap_semantic_library_pilot as lib

OUT=phase.OUT.parent/'cap_option_semantic_factorial_20261123_01'
DOC=phase.cap.ROOT/'docs/research/key_findings/2026-11-23_option_semantic_factorial'
CONTROLLERS=('coupled','split_precision')
FEATURES=('basic','corridor','contact','both')
SEEDS=(17,29,43)
BANKS=lib.ARMS
MAX_TICKS=phase.MAX_TICKS
MAX_DECISIONS=phase.MAX_DECISIONS
read,save=phase.cap.read,phase.cap.save
MODULES={}
LIBRARIES={}
RULES=[]
PRECISION=ContactAwareCAP(precision=True)
MODELS={}


def collect_identities():
    known={r['content_sha256'] for r in read(phase.cap.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json')}
    known.update(read(OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    required={'robots','object','goal','room','movable_objects'}
    def visit(tree):
        if isinstance(tree,dict):
            if set(tree)>=required:
                known.add(phase.input_hash(tree))
            else:
                for v in tree.values():visit(v)
        elif isinstance(tree,list):
            for v in tree:visit(v)
    for name in ('panel.json','panels.json','held.json','fresh.json','test.json','development.json','natural.json','variants.json','families.json'):
        for path in OUT.parent.rglob(name):
            if OUT not in path.parents:
                visit(read(path))
    return known


def freeze():
    known=collect_identities()
    dev=dict(phase.bench.DEV)
    held={}
    for seed,destination,limit,tag in [(662314121,dev,200,'dev_a'),(662314122,dev,40,'dev_b'),
                                        (662314123,held,200,'test_a'),(662314124,held,200,'test_b')]:
        counts={}
        for r in phase.cap.rooms.generate(seed)['cases']:
            n=r['movable_count']
            counts[n]=counts.get(n,0)+1
            if counts[n]>limit//4:continue
            f=r['fixture'];identity=phase.input_hash(f)
            assert identity not in known
            known.add(identity)
            destination[tag+'_'+r['id']]=f
    assert len(dev)==len(held)==400
    libraries={}
    for bank in BANKS:
        old.base.host.RoomCampaign(lib.OUT/bank).verify_frozen()
        members=read(lib.OUT/f'{bank}_library.json')
        for m in members:
            assert phase.cap.digest(Path(m['path']).read_text())==m['sha']
        libraries[bank]=members
    rules=read(lib.OUT/'conditions.json')
    sources={name:hashlib.sha256((phase.cap.ROOT/name).read_bytes()).hexdigest() for name in (
             'tools/cap_option_semantic_factorial.py','experiments/planar_tasks/semantic_rl/push_phases.py',
             'experiments/planar_tasks/semantic_rl/phase_aware_cap.py',
             'experiments/planar_tasks/semantic_rl/corridor_semantics.py')}
    protocol=dict(dev=phase.cap.digest(dev),held=phase.cap.digest(held),libraries=libraries,rules=rules,
                  code=sources,controllers=CONTROLLERS,feature_arms=FEATURES,seeds=SEEDS,
                  update_steps=8000,batch=64,lr=.0003,physical_budgets=[MAX_TICKS,MAX_DECISIONS],
                  teacher='fastest successful member, per controller and bank, whole trajectory',
                  primary='held room-clustered contact feature gain over basic, compare across push executors and CAP banks',
                  scope='frozen authored programs; automatic verified split with near-goal pacing only')
    for name,obj in [('protocol',protocol),('development',dev),('held',held)]:
        p=OUT/(name+'.json')
        if p.exists():assert phase.cap.digest(read(p))==phase.cap.digest(obj)
        else:save(p,obj)


def initialize():
    global LIBRARIES,RULES
    torch.set_num_threads(1)
    protocol=read(OUT/'protocol.json')
    RULES=protocol['rules']
    LIBRARIES={bank:[x['sha'] for x in items] for bank,items in protocol['libraries'].items()}
    for items in protocol['libraries'].values():
        for row in items:
            if row['sha'] not in MODULES:
                text=Path(row['path']).read_text()
                assert phase.cap.digest(text)==row['sha']
                MODULES[row['sha']]=old.dq.load_native(text)[0]


def observe(w):
    return phase.cap.current_event(w.history)['observation']


def remaining(w):
    return MAX_TICKS-w.env.total_periods


def ended(w):
    return w.env.termination is not None or remaining(w)<=0 or w.decision_count>=MAX_DECISIONS


def tick_action(w,action,controller,update,counts):
    if controller=='coupled' or len(translating(action))!=1:
        w.step(action,remaining_ticks=remaining(w));update(w.history)
        return
    contact=push_phases.contact(w,action,min(remaining(w),action['ticks']))
    counts['contact_'+contact.status]+=1
    if contact.ticks:update(w.history)
    left=action['ticks']-contact.ticks
    if contact.status!='ready' or left<=0 or ended(w):
        assert contact.ticks
        return
    default=dict(action,ticks=left)
    push=PRECISION.choose_push(observe(w),action,left)
    counts['retimed']+=push!=default
    drive=push_phases.push_from_contact(w,push,min(remaining(w),push['ticks']))
    if drive.ticks:update(w.history)
    elif not contact.ticks:
        w.step(dict(target_id=None,a0=44,a1=44,ticks=1),remaining_ticks=remaining(w))
        update(w.history)
    counts['drive_'+drive.status]+=1


def output(w,counts):
    return dict(success=int(w.env.success),ticks=w.env.total_periods,decisions=w.decision_count,
                history=w.history.sha256,termination=w.env.termination,counts=dict(counts))


def single(job):
    controller,sid,room,fixture,panel,path=job
    policy=phase.FrozenCAP(MODULES[sid])
    counts=Counter();route=[]
    with phase.world(fixture) as (w,_):
        policy.update(w.history)
        while not ended(w):
            a=policy.propose(observe(w))
            route.append(a)
            tick_action(w,a,controller,policy.update,counts)
        result=output(w,counts)
    if panel=='development' and result['success']:result['actions']=route
    if controller=='coupled' and room in phase.bench.DEV:
        original=phase.bench.SOURCE_DIR/'runs/s025'/(room+'.json')
        basis=read(OUT/'protocol.json')['libraries']['usual'][0]['sha']
        if sid==basis:
            ref=read(original)
            assert (result['success'],result['ticks'],result['history'])==(int(ref['success']),ref['ticks'],ref['history'])
    save(path,result)


def banks(w,bank):
    policies=[phase.FrozenCAP(MODULES[sid]) for sid in LIBRARIES[bank]]
    def update(history):
        for p in policies:p.update(history)
    update(w.history)
    return policies,update


def candidates(policies,obs):
    pool=[]
    support=[]
    for i,p in enumerate(policies):
        try:a=p.propose(obs)
        except ValueError:
            # An expert's first proposal can become illegal off its own trajectory.
            continue
        if a not in pool:
            pool.append(a);support.append(np.zeros(len(policies)))
        support[pool.index(a)][i]=1.
    if not pool:
        pool=[dict(target_id=None,a0=44,a1=44,ticks=1)]
        support=[np.zeros(len(policies))]
    return pool,support


def reference_angle(obs):
    points=next((e['points'] for e in obs['entities'] if e['id']==obs['goal']['target_id'] and e['points']),[])
    xy=np.mean(points,0) if points else np.mean([position(obs,i) for i in (0,1)],0)
    dx,dy=np.subtract(obs['goal']['xy'],xy)
    return math.atan2(dy,dx)


def contact_features(obs,action,old_action,old_positions,contexts):
    result=np.zeros(14)
    target=action['target_id']
    for i in translating(action):
        own,returns,labels,patches=contexts[i]
        occupied=returns[labels!=target]
        option=ACTIONS[action[f'a{i}']]
        axis=np.asarray(option.effect_direction)
        robot=obs['robots'][i]
        active=obs['active_target']==target
        touched=active and robot['contacting_object']
        best=None
        if active:
            for patch in patches:
                normal=np.asarray(patch['normal_world'])
                if not direction_matches(normal,axis):
                    continue
                q=np.mean(patch['points_world'],axis=0)
                station=q-normal*(ROBOT_RADIUS+.15)
                distance=float(np.linalg.norm(station-own))
                gap=clearance(own,station,occupied)
                test=(distance,q,normal,gap)
                if best is None or distance<best[0]:best=test
        same=bool(old_action and old_action['target_id']==target and i in translating(old_action)
                  and old_action[f'a{i}']//4==action[f'a{i}']//4)
        changed=0. if old_positions is None else float(np.linalg.norm(own-old_positions[i]))
        # Positive tactile evidence is distinct from visible direction support.
        values=[float(touched),float(best is not None),0.,0.,0.,float(same),float(same)*changed]
        if best is not None:
            d,q,n,gap=best
            values[2:5]=[d,min(gap,.8),float(n@axis)]
        result[7*i:7*i+7]=values
    return result


def encoder(obs,pool,support,angle,decision,old_action,old_positions,features):
    common=np.r_[old.base.geom(obs),obs['tick']/MAX_TICKS,decision/MAX_DECISIONS]
    corridor=np.zeros(len(RULES))
    if features in ('both','corridor'):
        estimate=corridor_semantics(obs)
        estimate['observed']=estimate['kind']
        corridor=np.array([lib.matches(r,estimate) for r in RULES])
    contexts=[]
    if features in ('both','contact'):
        for i,r in enumerate(obs['robots']):
            returns,_,labels=endpoints(r)
            patches=[p for p in (r['patches'] or []) if p['valid'] and len(p['points_world'])>=3]
            contexts.append((position(obs,i),returns,labels,patches))
    out=[]
    for a,s in zip(pool,support):
        local=contact_features(obs,a,old_action,old_positions,contexts) if contexts else np.zeros(14)
        out.append(np.r_[common,old.dq.action_encoding(a,obs['goal']['target_id'],angle),s,corridor,local])
    return np.stack(out).astype('float32')


@torch.inference_mode()
def episode(job):
    controller,bank,features,seed,room,fixture,path,teacher=job
    panels=[];targets=[];counts=Counter()
    with phase.world(fixture) as (w,_):
        policies,update=banks(w,bank)
        angle=reference_angle(observe(w))
        old_action=old_positions=None
        index=0
        while not ended(w):
            obs=observe(w)
            pool,sources=candidates(policies,obs)
            def features_for(arm):
                return encoder(obs,pool,sources,angle,w.decision_count,old_action,old_positions,arm)
            if teacher is not None:
                action=teacher['actions'][index]
                selected=pool.index(action)
                if len(pool)>1:
                    x=features_for('both')
                    padded=np.full((4,x.shape[-1]),np.nan,dtype=np.float32)
                    padded[:len(pool)]=x
                    panels.append(padded);targets.append(selected)
            elif len(pool)==1:
                action=pool[0]
            else:
                score,mu,sd=MODELS[controller,bank,features,seed]
                x=features_for(features)
                vals=score(torch.from_numpy((x-mu)/sd)).flatten().numpy()
                assert np.isfinite(vals).all()
                action=pool[int(np.argmax(vals))]
            old_action=action
            old_positions=[position(obs,i) for i in (0,1)]
            tick_action(w,action,controller,update,counts)
            index+=1
        result=output(w,counts)
    path.parent.mkdir(exist_ok=True,parents=True)
    if teacher is not None:
        assert index==len(teacher['actions']) and result['history']==teacher['history'] and result['success']
        temp=path.with_suffix('.tmp.npz')
        np.savez_compressed(temp,x=np.array(panels,dtype=np.float32),y=np.array(targets,dtype=np.int64))
        temp.replace(path)
    else:
        save(path,result)


def execute(func,jobs,stage,workers):
    index=6 if func is episode else 5
    total=len(jobs)
    jobs=[j for j in jobs if not j[index].exists()]
    if not jobs:return
    print(stage,total-len(jobs),total,flush=True)
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(func,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),total-len(jobs)+1):
            f.result()
            if i%100==0 or i==total:
                progress=OUT/('progress_pacing.json' if stage.startswith('timing') else 'progress.json')
                save(progress,dict(stage=stage,done=i,total=total))
                print(stage,i,total,flush=True)


def singles_path(controller,sid,room,panel):
    return OUT/'singles'/panel/controller/sid/(room+'.json')


def collect(dev,held,workers):
    members=list(MODULES)
    jobs=[(controller,sid,k,f,panel,singles_path(controller,sid,k,panel))
          for panel,rows in [('development',dev),('held',held)]
          for controller in CONTROLLERS
          for k,f in rows.items()
          for sid in members]
    execute(single,jobs,'single CAPs',workers)


def extract(dev,workers):
    jobs=[]
    for controller in CONTROLLERS:
        for bank,members in LIBRARIES.items():
            for k,f in dev.items():
                options=[read(singles_path(controller,sid,k,'development')) for sid in members]
                wins=[r for r in options if r['success']]
                if wins:
                    jobs.append((controller,bank,'both',0,k,f,OUT/'features'/controller/bank/(k+'.npz'),
                                 min(wins,key=lambda r:r['ticks'])))
    execute(episode,jobs,'teacher boundary replay',workers)


def network(dim):
    return old.bank_model(dim)


def fit():
    for controller in CONTROLLERS:
        for bank in BANKS:
            files=sorted((OUT/'features'/controller/bank).glob('*.npz'))
            with_data=[]
            for path in files:
                with np.load(path) as r:
                    if len(r['y']):with_data.append((r['x'],r['y']))
            x=np.concatenate([v[0] for v in with_data])
            y=torch.from_numpy(np.concatenate([v[1] for v in with_data]))
            valid=np.isfinite(x[:,:,0])
            assert valid[np.arange(len(y)),y.numpy()].all()
            mask=torch.from_numpy(valid)
            for arm in FEATURES:
                arr=x.copy()
                if arm in ('basic','corridor'):arr[:,:,-14:]=0.
                if arm in ('basic','contact'):arr[:,:,-14-len(RULES):-14]=0.
                mu=arr[valid].mean(0)
                sd=np.maximum(arr[valid].std(0),.1)
                z=torch.from_numpy(np.nan_to_num((arr-mu)/sd))
                for seed in SEEDS:
                    ident=(controller,bank,arm,seed)
                    tag='_'.join(map(str,ident))
                    root=OUT/'models'
                    root.mkdir(exist_ok=True)
                    path=root/(tag+'.pt')
                    model=network(z.shape[-1])
                    if path.exists():
                        obj=torch.load(path,weights_only=True)
                        assert np.array_equal(obj['mean'].numpy(),mu) and np.array_equal(obj['sd'].numpy(),sd)
                        model.load_state_dict(obj['weights'])
                    else:
                        torch.manual_seed(seed)
                        model=network(z.shape[-1])
                        opt=torch.optim.Adam(model.parameters(),lr=.0003)
                        for _ in range(8000):
                            ids=torch.randint(len(y),(64,))
                            logits=model(z[ids]).squeeze(-1).masked_fill(~mask[ids],-1e9)
                            loss=torch.nn.functional.cross_entropy(logits,y[ids])
                            opt.zero_grad();loss.backward();opt.step()
                        with torch.no_grad():
                            acc=float((model(z).squeeze(-1).masked_fill(~mask,-1e9).argmax(1)==y).float().mean())
                        tmp=path.with_suffix('.tmp')
                        torch.save(dict(weights=model.state_dict(),mean=torch.from_numpy(mu),sd=torch.from_numpy(sd)),tmp)
                        tmp.replace(path)
                        save(root/(tag+'.json'),dict(accuracy=acc,boundaries=len(y),teachers=len(files),seed=seed))
                    MODELS[ident]=(model.eval(),mu,sd)
                    print('fit',tag,flush=True)


def evaluate(held,workers):
    jobs=[(a,b,c,s,k,f,OUT/'eval'/a/b/c/f's{s}'/(k+'.json'),None)
          for a in CONTROLLERS for b in BANKS for c in FEATURES for s in SEEDS for k,f in held.items()]
    execute(episode,jobs,'mixed CAP closed loops',workers)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('freeze','collect','run'),default='run')
    p.add_argument('--workers',type=int,default=18)
    args=p.parse_args()
    freeze()
    if args.stage=='freeze':return
    dev,held=read(OUT/'development.json'),read(OUT/'held.json')
    initialize()
    collect(dev,held,args.workers)
    if args.stage=='collect':return
    extract(dev,args.workers)
    fit()
    evaluate(held,args.workers)
    from tools.cap_option_semantic_factorial_report import main as report
    report()


if __name__=='__main__':
    main()
