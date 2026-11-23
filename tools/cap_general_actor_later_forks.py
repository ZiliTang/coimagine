"""Single-decision repairs at four later disagreements on each discordant losing path."""
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from tools import cap_general_actor_forks as study

OUT=study.OUT/'later'


def case(job):
    key,fixture,prior=job
    loser=int(prior['actor800'])
    path=study.rollout(fixture,loser)
    expected=prior['ln_history' if loser else 'actor800_history']
    assert path['hash']==expected and path['success']==0
    trace=[dict(a=r['a'],tick=r['t']+r['dt']) for r in path['steps']]
    replay=study.rollout(fixture,None,teacher=trace)
    assert replay['hash']==expected
    path=replay['steps']
    disagreements=[i for i,v in enumerate(path) if v['choices'][0]!=v['choices'][1]]
    candidates=disagreements[1:]
    selected=sorted(set(np.linspace(0,len(candidates)-1,min(4,len(candidates))).round().astype(int).tolist()))
    wins=[]
    for sid in study.SIDS:
        r=study.read(study.campaign.OUT/'runs'/sid/(key+'.json'))
        if r['success']:
            wins.append((r['ticks'],sid))
    teacher=study.sets.unzip(study.campaign.OUT/'runs'/min(wins)[1]/(key+'.json.gz')) if wins else []
    tested=[]
    for j in selected:
        i=candidates[j]
        item=path[i]
        action=item['choices'][1-loser]
        assert item['a']==item['choices'][loser]
        result=study.rollout(fixture,loser,cut=i,override=action,ref=path)
        on=i<len(teacher) and trace[:i]==[dict(a=r['a'],tick=r['tick']) for r in teacher[:i]]
        label=('alternative' if teacher[i]['a']==action else 'actual' if teacher[i]['a']==item['a'] else 'third') if on else 'off_prefix' if teacher else 'no_teacher'
        tested.append(dict(step=i,tick=item['t'],prefix=item['h'],actual=item['a'],alternative=action,success=result['success'],teacher=label))
    return dict(room=key,loser=loser,first=disagreements[0],n_later=len(candidates),tested=tested)


def main():
    torch.set_num_threads(1)
    _,rooms,study.SIDS=study.scale.inputs()
    assert study.campaign.base.digest(rooms)==study.read(study.base.OUT/'protocol.json')['panel']
    study.MODELS=[study.sets.checkpoint(study.scale.OUT),study.fit.load_best()]
    study.dq.MODULES=[study.dq.load_native((study.campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in study.SIDS]
    with (study.base.OUT/'per_scene.csv').open() as f:
        cases=[(r['room'],rooms[r['room']],r) for r in csv.DictReader(f) if r['actor800']!=r['layernorm']]
    assert len(cases)==61
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(6,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(case,job) for job in cases if not (OUT/(job[0]+'.json')).exists()]
        for future in as_completed(futures):
            row=future.result()
            study.save(OUT/(row['room']+'.json'),row)


if __name__=='__main__':
    main()
