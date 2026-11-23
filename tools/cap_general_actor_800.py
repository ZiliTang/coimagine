"""Fixed-budget 330 versus 800 training-room comparison for the frozen action actor."""
import csv
import gzip
import json
from tempfile import TemporaryDirectory
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tools import cap_general_specialists_actor as original
from tools import cap_general_specialists as campaign
from tools import cap_regional_directq as dq
from tools import cap_regional_fastest_actor as imitation
from tools.cap_regional_evaluate import paired

OUT = campaign.OUT/'action_actor_800_s17'
SEED = 66218000
read, save = campaign.read, campaign.save


def fixed_file(name, value):
    path = OUT/name
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen input mismatch at {path}')
    else:
        save(path, value)


def inputs():
    old = read(campaign.OUT/'development.json') | read(campaign.OUT/'extra.json')
    held = read(campaign.OUT/'held.json')
    state = read(campaign.STATE)
    assert campaign.base.host.RoomCampaign(campaign.CAM).verify_frozen() == state['seal']
    sids = [r['sid'] for r in state['rounds']]
    assert len(old) == 330 and len(held) == 200 and len(sids) == 5
    return old, held, sids


def natural():
    result = {}
    for batch, per_stratum in enumerate((50, 7)):
        counts = Counter()
        for row in campaign.base.rooms.generate(SEED+batch)['cases']:
            n = row['movable_count']
            if counts[n] < per_stratum:
                j = counts[n] + 50*batch
                result[f'nat_{100+n}_{j:02d}'] = row['fixture']
                counts[n] += 1
    assert len(result) == 228
    return result


def collect_member(sid, rooms):
    path = OUT/'runs'/sid
    missing = {k:f for k,f in rooms.items() if not (path/(k+'.json')).exists()}
    if missing:
        source = (campaign.CAM/'semantics'/(sid+'.py')).read_text()
        campaign.base.run_batch(source, missing, path, 16)
    for k in rooms:
        r = read(path/(k+'.json'))
        if not r['complete'] or r['error']:
            raise RuntimeError((sid,k,r))


def collect():
    OUT.mkdir(exist_ok=True)
    old,held,sids = inputs()
    norm = original.BASE/'normalizers.npz'
    fixed_file('protocol.json',dict(sids=sids, generation_seed=SEED, seed=17, updates=24000,
               canonical='old 330 + 228 natural + 242 variants of new generalist failures',
               old_actor=campaign.sha(original.OUT/'actor.pt'), normalizers=campaign.sha(norm),
               old_pool=campaign.base.digest(old), held=campaign.base.digest(held)))
    new = natural()
    known = {campaign.input_hash(f) for f in [*old.values(),*held.values()]}
    natural_hashes = {campaign.input_hash(f) for f in new.values()}
    assert len(natural_hashes)==len(new) and not natural_hashes & known
    known.update(natural_hashes)
    fixed_file('natural.json',new)
    save(OUT/'stage.json',dict(stage='自然房间基线采集',sid=sids[0]))
    collect_member(sids[0],new)
    failed = [k for k in sorted(new) if not read(OUT/'runs'/sids[0]/(k+'.json'))['success']]
    random.Random(SEED).shuffle(failed)
    if not failed:
        raise ValueError('no failed parent rooms')
    parents = {}
    variants = {}
    for j in range(242):
        k = failed[j % len(failed)]
        cid = f'var_{200+j//100}_{j%100:02d}'
        f = campaign.base.variant(dict(case_id=k,fixture=new[k]), SEED*1000+j)
        fingerprint = campaign.input_hash(f)
        assert fingerprint not in known
        known.add(fingerprint)
        parents[cid] = k
        variants[cid] = f
    fixed_file('parents.json',parents)
    fixed_file('variants.json',variants)
    added = new | variants
    assert len(old | added)==800 and not held.keys() & (old|added).keys()
    for sid in sids:
        save(OUT/'stage.json',dict(stage='新增开发场景采集',sid=sid))
        collect_member(sid,added)
    print('collection verified',len(added)*len(sids),flush=True)
    save(OUT/'stage.json',dict(stage='新增轨迹已齐'))


def save_gzip(path,value):
    tmp = path.with_suffix('.pending')
    with gzip.open(tmp,'wt',compresslevel=6) as f:
        json.dump(value,f,separators=(',',':'))
    with gzip.open(tmp,'rt') as f:
        assert json.load(f) == value
    tmp.replace(path)


def compact_collection(rooms,sids):
    """Archive and verify the 2,350 outcomes and only the selected target sequences."""
    outcomes = {}
    actions = {}
    for k in sorted(rooms):
        records = {sid:read(OUT/'runs'/sid/(k+'.json')) for sid in sids}
        assert all(r['complete'] and not r['error'] for r in records.values())
        outcomes[k] = records
        choices = [(r['ticks'],s) for s,r in records.items() if r['success']]
        best = min(choices)[1] if choices else None
        for sid in sids:
            path = OUT/'runs'/sid/(k+'.json.gz')
            if sid == best:
                with gzip.open(path,'rt') as f:
                    target = [dict(a=v['a'],tick=v['tick']) for v in json.load(f)]
                assert len(target)==records[sid]['decisions'] and target[-1]['tick']==records[sid]['ticks']
                actions[k] = dict(sid=sid, steps=target)
                save_gzip(path,target)
            else:
                path.unlink(missing_ok=True)
    save_gzip(OUT/'new_outcomes.json.gz',outcomes)
    save_gzip(OUT/'new_teacher_actions.json.gz',actions)


def clear_staging():
    """Only remove run files represented in the two verified campaign-local archives."""
    with gzip.open(OUT/'new_outcomes.json.gz','rt') as f:
        outcomes=json.load(f)
    with gzip.open(OUT/'new_teacher_actions.json.gz','rt') as f:
        actions=json.load(f)
    assert len(outcomes)==470 and sum(map(len,outcomes.values()))==2350
    assert len(actions)==read(OUT/'FINISHED.json')['added_teachers']
    for k,records in outcomes.items():
        for sid,row in records.items():
            path=OUT/'runs'/sid/(k+'.json')
            if path.exists():
                assert read(path)==row
                path.unlink()
            trace=path.with_suffix('.json.gz')
            if trace.exists():
                with gzip.open(trace,'rt') as f:
                    assert actions[k]==dict(sid=sid,steps=json.load(f))
                trace.unlink()
    runs=OUT/'runs'
    if runs.exists():
        for directory in runs.iterdir():
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        if not any(runs.iterdir()):
            runs.rmdir()
    pred=OUT/'predictions.npy'
    pred.unlink(missing_ok=True)


def teacher_shards(rooms, sids, runs, phase):
    jobs = []
    for k,f in sorted(rooms.items()):
        choices = []
        for i,sid in enumerate(sids):
            rec = read(runs/sid/(k+'.json'))
            assert rec['complete'] and not rec['error']
            if rec['success']:
                choices.append((rec['ticks'],sid,i))
        if choices:
            _,sid,i = min(choices)
            jobs.append((k,f,sid,i,runs/sid,'replay'))
    save(OUT/'stage.json',dict(stage=phase,total=len(jobs)))
    return dq.batch(jobs,12,phase)


def load_teachers(records):
    fields = ('x','candidates','sources','valid','chosen','root','boundary')
    records = sorted(records,key=lambda r:r['case_id'])
    total = sum(r['boundaries'] for r in records)
    arrays = {}
    start = 0
    for r in records:
        assert r['success']
        with np.load(r['path']) as shard:
            end = start + r['boundaries']
            for field in fields:
                a = shard[field]
                if field not in arrays:
                    arrays[field] = np.empty((total,*a.shape[1:]),a.dtype)
                arrays[field][start:end] = a
            np.testing.assert_array_equal(shard['boundary'],np.arange(end-start))
            assert shard['duration'].sum() == r['ticks']
        start = end
    assert len(np.unique(arrays['root'])) == len(records)
    return arrays


def subgroup_fit(model,data,cases):
    # Score once and report all / natural / perturbed without copying large feature matrices.
    all_fit,predictions = imitation.assess(model,data)
    result = dict(all=all_fit)
    correct = predictions == data['chosen']
    active = data['valid'].sum(1)>1
    for prefix in ('nat','var'):
        roots = [int(k.split('_')[-2])*100+int(k.split('_')[-1]) for k in cases if k.startswith(prefix+'_')]
        selected = active & np.isin(data['root'],roots)
        result[prefix] = dict(informative=int(selected.sum()), matched=int(correct[selected].sum()))
    return result


def fit_and_evaluate():
    if (OUT/'FINISHED.json').exists():
        clear_staging()
        return
    torch.set_num_threads(1)
    old,held,sids = inputs()
    added = read(OUT/'natural.json') | read(OUT/'variants.json')
    assert len(old|added)==800
    settings = read(OUT/'protocol.json')
    assert settings['old_actor']==campaign.sha(original.OUT/'actor.pt')
    assert settings['normalizers']==campaign.sha(original.BASE/'normalizers.npz')
    assert settings['held']==campaign.base.digest(held)
    dq.MODULES = [dq.load_native((campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in sids]
    compact_collection(added,sids)
    with TemporaryDirectory(prefix='coimagine-actor800-',dir='/dev/shm') as scratch:
        dq.OUT = Path(scratch)
        def progress(stage, **info):
            save(OUT/'progress.json',dict(stage=stage,**info))
        dq.progress = progress
        learn_and_compare(added,held,sids)
    clear_staging()


def learn_and_compare(added,held,sids):
    new_shards = teacher_shards(added,sids,OUT/'runs','校验新增最快教师')
    old_teachers = read(original.OUT/'teachers.json')
    old_shards = [read(original.BASE/'shards'/r['sid']/(k+'.json')) for k,r in old_teachers.items()]
    assert len(old_shards) == 222
    all_shards = old_shards + new_shards
    fixed_file('teachers.json',{r['case_id']:dict(sid=r['cap_id'],ticks=r['ticks'],boundaries=r['boundaries'],
                                                 history=r['history']) for r in all_shards})
    small = imitation.actor(original.BASE)
    small.load_state_dict(torch.load(original.OUT/'actor.pt',weights_only=True))
    small.eval()
    if not (OUT/'pretest.json').exists():
        pretest = subgroup_fit(small,load_teachers(new_shards),[r['case_id'] for r in new_shards])
        save(OUT/'pretest.json',pretest)
    if not (OUT/'actor.pt').exists():
        save(OUT/'stage.json',dict(stage='拟合 800 场景 actor',updates=imitation.STEPS))
        large = imitation.fit(load_teachers(all_shards),base=original.BASE,out=OUT)
    else:
        large = imitation.actor(original.BASE)
        large.load_state_dict(torch.load(OUT/'actor.pt',weights_only=True))
    large.eval()
    dq.MODEL = large
    save(OUT/'stage.json',dict(stage='同批 200 房间闭环',total=len(held)))
    episodes = dq.batch([(k,f,'actor800',None,None,'fresh') for k,f in sorted(held.items())],
                        12,'同批 200 房间闭环')
    by_room = {r['case_id']:r for r in episodes}
    with (original.OUT/'per_scene.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==len(held) and {r['room'] for r in rows}==set(by_room)==set(held)
    for r in rows:
        episode = by_room[r['room']]
        assert episode['termination'] in ('task_budget_exhausted','success') or (
            not episode['success'] and (episode['boundaries']==300 or episode['ticks']==1800))
        r.update(actor800=int(episode['success']), actor800_ticks=episode['ticks'],
                 actor800_decisions=episode['boundaries'], actor800_history=episode['history'])
    with (OUT/'per_scene.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    test_shards = teacher_shards(held,sids,campaign.OUT/'runs','留出成功教师离线诊断')
    test = load_teachers(test_shards)
    held_fit = {name:imitation.assess(model,test)[0] for name,model in [('330',small),('800',large)]}
    def values(key):
        return np.array([int(r[key]) for r in rows])
    oracle_count = np.sum([values(s) for s in sids],axis=0)
    summary = dict(teacher_rooms=len(all_shards),added_teachers=len(new_shards),
                   held_teacher_rooms=len(test_shards), held_fit=held_fit, fit=read(OUT/'fit.json')[-1],
                   old_on_added=read(OUT/'pretest.json'),checkpoint=campaign.sha(OUT/'actor.pt'),
                   vs_330=paired(values('actor'),values('actor800'),values('objects')),
                   vs_Q=paired(values('mix'),values('actor800'),values('objects')),
                   vs_best=paired(values('s027'),values('actor800'),values('objects')),
                   coverage={str(i):dict(rooms=int((oracle_count==i).sum()),
                       old=int(values('actor')[oracle_count==i].sum()),
                       new=int(values('actor800')[oracle_count==i].sum())) for i in range(6)})
    save(OUT/'FINISHED.json',summary)
    save(OUT/'stage.json',dict(stage='完成'))
    print(summary,flush=True)


def geometry_probe():
    """Change only public geometry, retaining the original proposal set and budget."""
    torch.set_num_threads(1)
    old,_,sids = inputs()
    _,data = imitation.teacher_data(old,sids,original.BASE)
    rng = np.random.default_rng(17)
    roots = data['root']
    other = rng.permutation(len(roots))
    same = roots[other] == roots
    while same.any():
        other[same] = rng.integers(len(roots),size=int(same.sum()))
        same = roots[other] == roots
    summary = {}
    with torch.inference_mode():
        for directory in (original.OUT,OUT):
            model = imitation.actor(original.BASE)
            model.load_state_dict(torch.load(directory/'actor.pt',weights_only=True))
            model.eval()
            width = model.layout['geometry_width']
            changed = eligible = 0
            for start in range(0,len(roots),96):
                x,c,s,valid,_ = imitation.batch(data,slice(start,start+96))
                swapped = x.clone()
                swapped[:,:width] = torch.from_numpy(data['x'][other[start:start+len(x)],:width])
                a = model.values(x,c,s).masked_fill(~valid,-torch.inf).argmax(1)
                b = model.values(swapped,c,s).masked_fill(~valid,-torch.inf).argmax(1)
                multi = valid.sum(1)>1
                changed += int(((a!=b)&multi).sum())
                eligible += int(multi.sum())
            summary[directory.name] = dict(changed=changed,multiple_actions=eligible)
    save(OUT/'geometry_probe.json',summary)


if __name__ == '__main__':
    if sys.argv[1:] == ['geometry-probe']:
        geometry_probe()
    elif sys.argv[1:] == ['evaluate']:
        fit_and_evaluate()
    elif not sys.argv[1:]:
        if not (OUT/'FINISHED.json').exists():
            collect()
        fit_and_evaluate()
    else:
        raise SystemExit('use no arguments, or evaluate to reuse an existing collection')
