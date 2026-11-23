"""Matched 800-room comparison: known successful actions on identical teacher prefixes."""
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tools import cap_general_actor_800 as scale
from tools.cap_general_actor_800 import campaign, dq, imitation, original, paired, read, save

OUT = campaign.OUT/'action_actor_800_sets_s17'


def unzip(path):
    with gzip.open(path,'rt') as f:
        return json.load(f)


def progress(stage, **info):
    save(OUT/'progress.json',dict(stage=stage,**info))


def sources_for(rooms, sids):
    return {k:{s:read(campaign.OUT/'runs'/s/(k+'.json')) for s in sids} for k in rooms}


def replay(rooms, outcomes, sids, reference, phase):
    jobs = []
    for k,f in sorted(rooms.items()):
        successes = [(r['ticks'],s) for s,r in outcomes[k].items() if r['success']]
        if successes:
            _,s = min(successes)
            jobs.append((k,f,s,sids.index(s),reference/s,'replay'))
    dq.progress(phase,completed=0,total=len(jobs))
    return dq.batch(jobs,12,phase)


def labels(records, data, outcomes, sids, archived):
    target = np.zeros_like(data['valid'])
    offset = multi_rooms = starts = audited = 0
    rates = []
    for r in sorted(records,key=lambda r:r['case_id']):
        k = r['case_id']
        n = r['boundaries']
        live = np.array([outcomes[k][s]['success'] for s in sids],dtype=bool)
        assert all(outcomes[k][s]['complete'] and not outcomes[k][s]['error'] for s in sids)
        src = data['sources'][offset:offset+n].astype(bool)
        chosen = data['chosen'][offset:offset+n]
        assert (data['boundary'][offset:offset+n]==np.arange(n)).all()
        local = target[offset:offset+n]
        for j in range(n):
            assert np.all(src[j].sum(0)[live]==1)
            local[j] = (src[j] & live).any(1)
            # A deterministic CAP agrees with the real prefix up to its first different request.
            live &= src[j,chosen[j]]
        assert all(outcomes[k][s]['history']==r['history'] for s,keep in zip(sids,live) if keep)
        assert live[sids.index(r['cap_id'])]
        if k in archived:
            # Independently check every eligible prefix against the original action and tick logs.
            baseline = unzip(campaign.OUT/'runs'/r['cap_id']/(k+'.json.gz'))
            expected = np.zeros_like(local)
            for i,s in enumerate(sids):
                if not outcomes[k][s]['success']:
                    continue
                other = unzip(campaign.OUT/'runs'/s/(k+'.json.gz'))
                for j,(left,right) in enumerate(zip(other,baseline)):
                    action_index = np.flatnonzero(src[j,:,i]).item()
                    expected[j,action_index] = True
                    if left['a'] != right['a']:
                        assert action_index != chosen[j]
                        break
                    assert left['tick']==right['tick'] and action_index==chosen[j]
            np.testing.assert_array_equal(expected,local)
            audited += n
        mixed = local.sum(1)>1
        rates.append(float(mixed.mean()))
        multi_rooms += int(mixed.any())
        starts += int(mixed[0])
        offset += n
    assert offset==len(target) and not (target & ~data['valid']).any()
    assert target[np.arange(offset),data['chosen']].all()
    return target,dict(rooms=len(records),boundaries=offset,mixed=int((target.sum(1)>1).sum()),
                       mixed_rooms=multi_rooms,mixed_starts=starts,room_uniform=float(np.mean(rates)),
                       archived_boundaries_checked=audited)


@torch.inference_mode()
def metrics(model,data,targets):
    chosen = []
    masses = []
    for start in range(0,len(targets),128):
        span = slice(start,start+128)
        x,c,s,valid,_ = imitation.batch(data,span)
        logits = model.values(x,c,s).masked_fill(~valid,-torch.inf)
        chosen.append(logits.argmax(1).numpy())
        mass = (logits.softmax(1)*torch.from_numpy(targets[span])).sum(1)
        masses.append(mass.numpy())
    pred = np.concatenate(chosen)
    mass = np.concatenate(masses)
    multi_action = data['valid'].sum(1)>1
    multi_label = targets.sum(1)>1
    hit = targets[np.arange(len(pred)),pred]
    return pred,dict(informative=int(multi_action.sum()),supported=int(hit[multi_action].sum()),
                     fastest=int((pred[multi_action]==data['chosen'][multi_action]).sum()),
                     mixed=int(multi_label.sum()),mixed_supported=int(hit[multi_label].sum()),
                     mean_set_mass_multiple_candidates=float(mass[multi_action].mean()),
                     mean_set_mass_multiple_labels=float(mass[multi_label].mean()))


def checkpoint(directory):
    m = imitation.actor(original.BASE)
    m.load_state_dict(torch.load(directory/'actor.pt',weights_only=True))
    return m.eval()


def training_records(scratch,old,sids):
    base = scale.OUT
    new = read(base/'natural.json') | read(base/'variants.json')
    outcomes = unzip(base/'new_outcomes.json.gz')
    actions = unzip(base/'new_teacher_actions.json.gz')
    reference = scratch/'references'
    for k,seq in actions.items():
        directory = reference/seq['sid']
        directory.mkdir(exist_ok=True,parents=True)
        save(directory/(k+'.json'),outcomes[k][seq['sid']])
        scale.save_gzip(directory/(k+'.json.gz'),seq['steps'])
    records = replay(new,outcomes,sids,reference,'重放新增教师')
    outcomes.update(sources_for(old,sids))
    teachers = read(base/'teachers.json')
    for k in old:
        if k in teachers:
            records.append(read(original.BASE/'shards'/teachers[k]['sid']/(k+'.json')))
    assert {r['case_id']:(r['cap_id'],r['history']) for r in records} == {k:(v['sid'],v['history']) for k,v in teachers.items()}
    return records,outcomes


def experiment(scratch):
    old,held,sids = scale.inputs()
    base = scale.OUT
    settings = read(base/'protocol.json')
    assert settings['held']==campaign.base.digest(held)
    assert settings['normalizers']==campaign.sha(original.BASE/'normalizers.npz')
    assert settings['sids']==sids and settings['updates']==imitation.STEPS==24000 and settings['seed']==imitation.SEED
    reference_result = read(base/'FINISHED.json')
    assert reference_result['checkpoint']==campaign.sha(base/'actor.pt')
    protocol = dict(**settings,baseline_sha=reference_result['checkpoint'],target='negative log total known-success action probability at exact teacher prefix',teacher_sha=campaign.sha(base/'teachers.json'))
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==protocol
    save(OUT/'protocol.json',protocol)
    dq.OUT = scratch
    dq.MODULES = [dq.load_native((campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in sids]
    dq.progress = progress
    records,outcomes = training_records(scratch,old,sids)
    data = scale.load_teachers(records)
    targets,inventory = labels(records,data,outcomes,sids,set(old))
    assert inventory['rooms']==491 and inventory['boundaries']==15601
    assert (targets[data['root']<10000].sum(1)>1).sum()==207
    save(OUT/'labels.json',inventory)
    progress('拟合集合监督',total=imitation.STEPS)
    model = (checkpoint(OUT) if (OUT/'actor.pt').exists() else
             imitation.fit(data,base=original.BASE,out=OUT,target_sets=targets))
    hard = checkpoint(base)
    models = dict(hard=hard,sets=model)
    train = {name:metrics(m,data,targets)[1] for name,m in models.items()}
    del data,targets
    dq.MODEL = model
    progress('配对闭环',completed=0,total=len(held))
    episodes = dq.batch([(k,f,'sets',None,None,'fresh') for k,f in sorted(held.items())],12,'配对闭环')
    with (base/'per_scene.csv').open() as f:
        rows = {r['room']:r for r in csv.DictReader(f)}
    assert len(rows)==200 and rows.keys()==held.keys()=={r['case_id'] for r in episodes}
    for episode in episodes:
        assert episode['termination'] in ('success','task_budget_exhausted') or (
            not episode['success'] and (episode['ticks']==1800 or episode['boundaries']==300))
        rows[episode['case_id']].update(sets=int(episode['success']),sets_history=episode['history'])
    observed = sources_for(held,sids)
    test_records = replay(held,observed,sids,campaign.OUT/'runs','验证首次分歧')
    test = scale.load_teachers(test_records)
    supported,held_inventory = labels(test_records,test,observed,sids,set(held))
    test_metrics = {}
    for name,m in models.items():
        pred,test_metrics[name] = metrics(m,test,supported)
        offset = 0
        for r in sorted(test_records,key=lambda r:r['case_id']):
            room,n = r['case_id'],r['boundaries']
            span = slice(offset,offset+n)
            difference = np.flatnonzero(pred[span]!=test['chosen'][span])
            trace = unzip(scratch/'evaluation/fresh'/(room+'.json.gz')) if name=='sets' else None
            first = int(difference[0]) if len(difference) else n
            if trace is not None:
                assert [t['chosen'] for t in trace[:first]]==test['chosen'][span][:first].tolist()
                if first<n:
                    assert trace[first]['chosen']==pred[offset+first]
            row = rows[room]
            if first==n:
                assert int(row['sets' if name=='sets' else 'actor800'])==1
                assert row['sets_history' if name=='sets' else 'actor800_history']==r['history']
            row[name+'_step'] = first if first<n else ''
            row[name+'_path'] = ('identical' if first==n else 'supported' if supported[offset+first,pred[offset+first]] else 'unverified')
            offset += n
    for r in rows.values():
        for name in models:
            r.setdefault(name+'_step','')
            r.setdefault(name+'_path','no_teacher')
    table = list(rows.values())
    with (OUT/'per_scene.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    def values(key):
        return np.array([int(r[key]) for r in table])
    first_diff = {}
    for name,col in [('hard','actor800'),('sets','sets')]:
        first_diff[name] = {}
        for label in ('identical','supported','unverified','no_teacher'):
            group = [r for r in table if r[name+'_path']==label]
            first_diff[name][label] = dict(rooms=len(group),success=sum(int(r[col]) for r in group))
    result = dict(vs_hard=paired(values('actor800'),values('sets'),values('objects')),
                  union=int(values('union').sum()),unique={k:int(values(k)[np.sum([values(s) for s in sids],axis=0)==1].sum()) for k in ('actor800','sets')},
                  first_difference=first_diff,training=train,held=test_metrics,test_labels=held_inventory,
                  checkpoint=campaign.sha(OUT/'actor.pt'))
    save(OUT/'FINISHED.json',result)
    (OUT/'predictions.npy').unlink(missing_ok=True)
    progress('完成')
    print(result,flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if not (OUT/'FINISHED.json').is_file():
        torch.set_num_threads(1)
        with TemporaryDirectory(prefix='coimagine-sets-',dir='/dev/shm') as temp:
            experiment(Path(temp))
