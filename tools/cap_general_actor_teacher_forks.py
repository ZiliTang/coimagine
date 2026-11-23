"""Which teacher disagreements are consequential with a fixed LayerNorm continuation?"""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch

from tools import cap_general_actor_forks as study

OUT=study.OUT/'teacher_disagreements'


@torch.inference_mode()
def intervene(fixture,path,step,action):
    env=study.dq.fast_runtime.ScopedEnvironment(fixture,study.dq.study.CFG,fast_feedback=True,record_details=False,
                                                history_factory=study.dq.fast_runtime.AuditedCachedHistory)
    library=study.dq.Proposals()
    try:
        for i in range(300):
            panel=library.panel(env.history)
            if i<=step:
                assert env.history.sha256==path[i]['h']
                request=path[i]['a'] if i<step else action
                assert request in panel[4]
            else:
                request=study.choose(study.MODELS[1],panel)
            env.step(request,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        assert i>=step
        assert env.env.termination in ('success','task_budget_exhausted') or (env.env.termination is None and (i==299 or env.env.total_periods==1800))
        return int(env.env.success)
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE=None


def case(job):
    name,f,prior=job
    wins=[(study.read(study.campaign.OUT/'runs'/sid/(name+'.json'))['ticks'],sid) for sid in study.SIDS if prior[sid]=='1']
    teacher=min(wins)[1]
    path=study.rollout(f,None,teacher=study.sets.unzip(study.campaign.OUT/'runs'/teacher/(name+'.json.gz')))
    assert path['success'] and path['hash']==study.read(study.campaign.OUT/'runs'/teacher/(name+'.json'))['history']
    steps=path['steps']
    groups=[]
    for correct in (0,1):
        indices=[i for i,r in enumerate(steps) if r['choices'][correct]==r['a'] and r['choices'][1-correct]!=r['a']]
        label=dict(count=len(indices))
        if indices:
            i=indices[0]
            r=steps[i]
            label.update(step=i,tick=r['t'],prefix=r['h'],success=[intervene(f,steps,i,a) for a in r['choices']])
        groups.append(label)
    return dict(room=name,old_only=groups[0],ln_only=groups[1])


def main():
    torch.set_num_threads(1)
    _,rooms,study.SIDS=study.scale.inputs()
    assert study.campaign.base.digest(rooms)==study.read(study.base.OUT/'protocol.json')['panel']
    study.MODELS=[study.sets.checkpoint(study.scale.OUT),study.fit.load_best()]
    study.dq.MODULES=[study.dq.load_native((study.campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in study.SIDS]
    with (study.base.OUT/'per_scene.csv').open() as f:
        jobs=[(r['room'],rooms[r['room']],r) for r in csv.DictReader(f) if r['union']=='1']
    assert len(jobs)==144
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(6,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(case,j) for j in jobs if not (OUT/(j[0]+'.json')).exists()]
        for future in as_completed(futures):
            row=future.result()
            study.save(OUT/(row['room']+'.json'),row)


if __name__=='__main__':
    main()
