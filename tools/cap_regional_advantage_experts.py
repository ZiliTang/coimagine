"""Check whether the advantage of two development-selected CAPs transfers across the same families."""
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from tools import cap_regional_advantage as ref

study = ref.study
OUT = ref.OUT/'experts'
SIDS = ref.cpp.SID_ORDER
PAIR = None


def select():
    csv_path = study.DOCS/'per_scene.csv'
    with csv_path.open() as fp:
        rows = [r for r in csv.DictReader(fp) if r['panel']=='dev']
    assert len(rows)==160
    primary = min(SIDS,key=lambda s:(-sum(int(r[s]) for r in rows),s))
    alternate = min((s for s in SIDS if s!=primary),
                    key=lambda s:(-sum(int(r[s]) for r in rows if r[primary]=='0'),s))
    return primary,alternate


def run(f, switch=None, inspect=False, check=None):
    module = ref.cpp.load()[0]
    memory = [module.initial_memory(i) for i in PAIR]
    env = ref.ppo.environment(f)
    stages = {}
    try:
        for step in range(300):
            history = env.history
            event = study.dq.current_event(history)
            obs = event['observation']
            event = dict(event,executed_ticks=None if history.previous_tick is None else obs['tick']-history.previous_tick)
            if check and step == check['step']:
                assert history.sha256==check['prefix']
            actions=[]
            for k,i in enumerate(PAIR):
                memory[k] = module.update_memory(i,memory[k],event)
                a = memory[k]['context']['proposals'][0]
                study.dq.validate_action(a,obs)
                actions.append(a)
            if inspect and len(stages)<3:
                names=[]
                if step==0:
                    names.append('initial')
                if actions[0]!=actions[1] and 'divergence' not in stages:
                    names.append('divergence')
                if 'contact' not in stages and any(r['contacting_object'] for r in obs['robots']):
                    names.append('contact')
                if names:
                    v=[step,obs['tick'],*study.ext.old.geom(obs),int(actions[0]==actions[1])]
                    for r in obs['robots']:
                        v.extend([r['contacting_object'],r['contacting_static'],r['contacting_teammate'],
                                  r['own']['vx'],r['own']['vy'],r['own']['omega']])
                    for a in actions:
                        v.extend(study.dq.action_encoding(a,obs['goal']['target_id'],0.))
                    rec=dict(step=step,tick=obs['tick'],prefix=history.sha256,
                             features=[float(z) if np.isfinite(z) else None for z in v])
                    stages.update({name:rec for name in names})
            action = actions[int(switch is not None and step>=switch)]
            env.step(action,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        if check:
            assert step>=check['step']
        return dict(success=int(env.env.success),ticks=env.env.total_periods,steps=step+1,
                    history=env.history.sha256),stages
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE=None


def collect(job):
    p,idx = job
    cm=(0,1,2,5)[idx]
    file=OUT/'cases'/f"{p['parent']}_{cm}.json"
    if file.exists():
        return study.read(file)['runs']
    f=p['fixtures'][idx]
    first,stages=run(f,inspect=True)
    divergence=stages.get('divergence')
    contact=stages.get('contact')
    count=1
    other=first
    if divergence:
        other,_=run(f,switch=0,check=divergence)
        count+=1
    labels={name:other for name in stages}
    if contact and divergence and contact['step']>divergence['step']:
        labels['contact'],_=run(f,switch=contact['step'],check=contact)
        count+=1
    if p['repeat'] and cm==0:
        assert [run(f)[0],run(f,switch=0)[0]] == [first,other]
        count+=2
    study.save(file,dict(parent=p['parent'],cm=cm,fold=p['fold'],objects=p['objects'],
                         baseline=first,stages=stages,labels=labels,runs=count))
    return count


def main():
    global PAIR
    names=select()
    PAIR=tuple(SIDS.index(s) for s in names)
    ref.cpp.load()
    parents=study.read(ref.OUT/'panel.json')
    protocol=dict(pair=names,selection='first maximizes dev160 successes, second adds most in failures of first, name breaks ties',
                  panel=study.ext.old.digest(parents),source=study.sha(__file__),main=study.read(ref.OUT/'protocol.json'))
    assert protocol['panel']==protocol['main']['panel']
    study.save(OUT/'protocol.json',protocol)
    dev=study.read(study.OUT/study.ext.DEV)
    for name,switch in zip(names,[None,0]):
        for cid in ('dev_2_00','dev_3_00','dev_4_00','dev_5_00'):
            path = study.OUT/('sweeps' if name.startswith('n_') else 'baseline')/name/(cid+'.json')
            old=study.read(path)
            actual,_=run(dev[cid],switch=switch)
            assert old['complete'] and actual['history']==old['history'] and actual['success']==int(old['success'])
    jobs=[(p,i) for p in parents for i in range(4)]
    runs=0
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        for n,f in enumerate(as_completed([pool.submit(collect,j) for j in jobs]),1):
            runs+=f.result()
            study.save(OUT/'progress.json',dict(fixtures=n,total=len(jobs),rollouts=runs))


if __name__=='__main__':
    main()
