"""Matched-budget neighborhood augmentation of frozen CAP development families."""
from coimagine.random_start.spec import ROBOT_DIAMETER
import copy
import gzip
import json
import math
import multiprocessing as mp
import random
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import cap_retention_supervised as history
from . import random_room_tasks as rooms
from .cap_trace_value_long import sha
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_data, fast_runtime
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import CAPS

BASE, PARENT = history.BASE, history.PARENT
OUT = BASE/'cap_family_augmentation_20261119_01'
BUNDLE = Path(read(PARENT/'inputs.json')['bundle'])
PRIORITY = read(PARENT/'inputs.json')['priority_bundle']
ARMS = ('uniform', 'targeted')
SEED = 2026111927
BATCH, MIN_ROUNDS, MAX_ROUNDS, VISIT_CAP = 32, 4, 8, 4


def publish(stage, **changes):
    path = OUT/'live/state.json'
    state = read(path) if path.exists() else dict(arms=counters([]), phases={})
    state.update(stage=stage, time=time.time(), **changes)
    save(path, state)


def content(f):
    return digest({k:v for k,v in f.items() if k != 'scene_id'})


def known_cases():
    known = read(BUNDLE/'used_scene_identities.json')
    for path in (PARENT/'panels/final_test.json', PARENT/'panels/online_training.json',
                 history.prior.SEALED, BASE/'cap_trace_ppo_continue_20261118_01/confirmation.json',
                 BASE/'cap_retention_supervised_20261119_01/panel.json'):
        known.extend(read(path)['cases'])
    return known


def freeze():
    (OUT/'live').mkdir(exist_ok=True,parents=True)
    (OUT/'live/index.html').write_bytes(Path(__file__).with_name('cap_family_progress.html').read_bytes())
    if not (OUT/'live/state.json').exists():
        publish('准备开发初态与独立评估面板')
    if (OUT/'protocol.json').exists():
        for name, h in read(OUT/'protocol.json')['files'].items():
            assert sha(OUT/name) == h
        return
    split = read(PARENT/'split.json')
    fixtures = {r['case_id']:r['fixture'] for r in read(BUNDLE/'scenes.json')}
    winners = defaultdict(list)
    for row in read(BUNDLE/'index.json'):
        if row['summary']['success']:
            winners[row['case_id']].append(row['cap_id'])
    rng = random.Random(SEED)
    # Sample a training-only, solvable 200-family pool; retain all singleton witnesses.
    unique = sorted(k for k in split['train'] if len(winners[k]) == 1)
    others = sorted(k for k in split['train'] if len(winners[k]) > 1)
    roots = unique + rng.sample(others, 200-len(unique))
    rng.shuffle(roots)
    anchors = [dict(case_id=k, fixture=fixtures[k], winners=winners[k], content_sha256=content(fixtures[k])) for k in roots]
    assert len(unique) == 99 and set(roots).isdisjoint(split['validation'])
    known = known_cases()
    used = {r['content_sha256'] for r in known} | {content(f) for f in fixtures.values()}
    ids = {r['case_id'] for r in known} | set(fixtures)
    test = []
    for row in rooms.generate(SEED+1)['cases']:
        f = row['fixture']
        h, case = content(f), 'family_eval_'+row['id']
        assert h not in used and case not in ids
        used.add(h)
        ids.add(case)
        test.append(dict(case_id=case, fixture=f, content_sha256=h, movable_count=row['movable_count']))
    assert len(test) == 200
    save(OUT/'anchors.json', anchors)
    save(OUT/'evaluation_panel.json', test)
    frozen = read(PARENT/'ALL_MODELS_FROZEN.json')['files']
    path = 'models/GA_s17/offline.pt'
    assert sha(PARENT/path) == frozen[path]
    save(OUT/'protocol.json',dict(seed=SEED, training_seed=17, arms=list(ARMS), all_caps=CAPS,
         training_roots=608, anchors=len(anchors), singleton_anchors=len(unique),
         prim='targeted minus uniform closed-loop success on fresh exactly-one-of-seven CAP subset',
         secondary='whole panel, seven-CAP union, outside union; both vs frozen unaugmented GA-Q',
         batches=dict(variants_per_arm=BATCH, minimum=MIN_ROUNDS, maximum=MAX_ROUNDS,
                      max_per_family_per_arm=VISIT_CAP, stop='after a matched round, both arms >=201 distinct successful action sequences'),
         sampling='same pool; control random least visited; targeted lowest original + newly observed successful trajectories; random ties',
         perturbation='independent uniform +/-0.065m for x,y and +/-0.10 rad for angle of target, clutter and both robots; fixed obstacles and goal; room generation geometry rejection',
         learning='full original training data plus all successes and failures from assigned neighbors, boundary uniform, original normalizers and GA architecture, 16k V then 8k Q from scratch',
         test='200 freshly generated families, all seven CAPs once plus three greedy GA-Q policies; fixed checkpoints', baseline_sha256=frozen[path],
         files={k:sha(OUT/k) for k in ['anchors.json','evaluation_panel.json']}))


def variant(parent, seed):
    rng = random.Random(seed)
    original = parent['fixture']
    obstacles = [rooms.polygon(b['cx'],b['cy'],2*b['half_w'],2*b['half_h'],b['angle']) for b in original['room']['obstacles']]
    for _ in range(10000):
        f = copy.deepcopy(original)
        bodies = [f['object'], *f['movable_objects'], *f['robots']]
        for b in bodies:
            b['x'] += rng.uniform(-.065,.065)
            b['y'] += rng.uniform(-.065,.065)
            b['angle'] += rng.uniform(-.10,.10)
        shapes = [rooms.polygon(b['x'],b['y'],b.get('width',w),b.get('height',h),b['angle'])
                  for b,(w,h) in zip(bodies,[(.25,1.)]+[(0,0)]*len(f['movable_objects'])+[(ROBOT_DIAMETER,ROBOT_DIAMETER)]*2,strict=True)]
        if any(not rooms.inside(p) or any(not rooms.separated(p,q) for q in obstacles+shapes[:i]) for i,p in enumerate(shapes)):
            continue
        t,g=f['object'],f['goal']
        goal=rooms.polygon(g['x'],g['y'],.25,1.,t['angle'])
        if math.hypot(t['x']-g['x'],t['y']-g['y']) < 1.8 or not rooms.inside(goal):
            continue
        if any(not rooms.separated(goal,p,.1) for p in obstacles+shapes[:-2]):
            continue
        if not all(.65<=math.hypot(r['x']-t['x'],r['y']-t['y'])<=1.4 for r in f['robots']):
            continue
        return f
    raise ValueError('no legal neighborhood for '+parent['case_id'])


def initialize():
    fast_runtime.initialize(str(BUNDLE), PRIORITY)
    full_data.LIBRARY = fast_runtime.W.LIBRARY


def worker(job):
    result = fast_runtime.run(job)
    if not result['complete']:
        raise RuntimeError(result)
    if job.get('training'):
        path = Path(job['directory'])/'result.json.gz'
        row = dict(path=str(path), sha256=sha(path), success=result['success'],
                   cap_id=job['cap'], case_id=job['case']['case_id'])
        extracted = full_data.extract(('.', str(OUT/job['arm']), row, job['root_number']))
        return dict(case_id=row['case_id'], parent=job['case']['parent'], arm=job['arm'], cap=job['cap'],
                    success=bool(result['success']), replay=str(path), sha256=row['sha256'], shard=extracted)
    return result


def make_round(number, parents, prior, forbidden):
    path = OUT/f'round_{number:02d}_plan.json'
    if path.exists():
        rows=read(path)
        assert all(r['content_sha256'] not in forbidden for r in rows)
        forbidden.update(r['content_sha256'] for r in rows)
        return rows
    rows=[]
    for idx,arm in enumerate(ARMS):
        done=[r for r in prior if r['arm']==arm]
        visits=Counter(r['parent'] for r in done if r['cap']==CAPS[0])
        wins=Counter(r['parent'] for r in done if r['success'])
        rng=random.Random(SEED+100*number+idx)
        eligible=[p for p in parents if visits[p['case_id']]<VISIT_CAP]
        rng.shuffle(eligible)
        if arm=='uniform':
            eligible.sort(key=lambda p: visits[p['case_id']])
        else:
            eligible.sort(key=lambda p:len(p['winners'])+wins[p['case_id']])
        assert len(eligible)>=BATCH
        for i,p in enumerate(eligible[:BATCH]):
            f=variant(p,rng.getrandbits(64))
            h=content(f)
            assert h not in forbidden
            forbidden.add(h)
            rows.append(dict(arm=arm, parent=p['case_id'], case_id=f'aug_{arm}_{number}_{i:02d}',
                             root_number=10000+number*BATCH+i, fixture=f, content_sha256=h,
                             base_winners=p['winners']))
    save(path,rows)
    return rows


def counters(records):
    return {arm:dict(success=sum(r['success'] for r in records if r['arm']==arm),
                     runs=sum(r['arm']==arm for r in records),
                     families=len({r['parent'] for r in records if r['arm']==arm})) for arm in ARMS}



def action_sequence(row):
    with gzip.open(row['replay'], 'rt') as stream:
        episode=json.load(stream)
    assert episode['summary']['success']
    # Hash only the full sequence of actual policy requests. Different scenes
    # with exactly the same action list count once for this diversity statistic.
    return digest([e['requested'] for e in episode['events']])


def record_distinct(round_rows, pool, path):
    if path.exists():
        signatures=read(path)
    else:
        successes=[r for r in round_rows if r['success']]
        signatures={r['sha256']:signature for r,signature in zip(successes,pool.map(action_sequence,successes),strict=True)}
        save(path,signatures)
    return {arm:{signatures[r['sha256']] for r in round_rows if r['arm']==arm and r['success']} for arm in ARMS}


def collect(require_distinct=True):
    freeze()
    if (OUT/'collection.json').exists() and (not require_distinct or 'unique_sequences' in read(OUT/'collection.json')):
        return
    parents=read(OUT/'anchors.json')
    forbidden={r['content_sha256'] for r in known_cases()+read(OUT/'evaluation_panel.json')+parents}
    records=[]
    unique={arm:set() for arm in ARMS}
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for number in range(1,MAX_ROUNDS+1):
            plans=make_round(number,parents,records,forbidden)
            dest=OUT/f'round_{number:02d}_results.json'
            if dest.exists():
                records.extend(read(dest))
            else:
                jobs=[dict(training=True, arm=p['arm'],root_number=p['root_number'],case=p,cap=cap,condition=cap,
                           directory=str(OUT/'training_runs'/p['case_id']/cap)) for p in plans for cap in CAPS]
                random.Random(SEED+number).shuffle(jobs)
                current=[]
                for future in as_completed([pool.submit(worker,j) for j in jobs]):
                    current.append(future.result())
                    if len(current)%8==0:
                        publish('采集并提取轨迹特征',arms=counters(records+current),
                                phases={'匹配采集批次':f'{number-1} 完成，本批 {len(current)} / {len(jobs)}',
                                        '拟合':'0 / 2','新面板执行':'0 / 2000'},
                                detail='每个合法扰动初态执行七个原 CAP。所有完整成败都进入所属组的离线训练数据。')
                current.sort(key=lambda r:(r['case_id'],CAPS.index(r['cap'])))
                save(dest,current)
                records.extend(current)
            counts=counters(records)
            signatures=record_distinct(read(dest),pool,OUT/f'round_{number:02d}_sequences.json')
            for arm in ARMS:
                unique[arm].update(signatures[arm])
                counts[arm]['distinct']=len(unique[arm])
            if number>=MIN_ROUNDS and min(len(group) for group in unique.values())>=201:
                break
    assert all(v['runs']==number*BATCH*7 for v in counts.values())
    save(OUT/'collection.json',dict(arms=counts, rounds=number,records=records,unique_sequences={k:len(v) for k,v in unique.items()}))
    publish('采集已完成',arms=counts,phases={'匹配采集批次':f'{number} / {number}','拟合':'0 / 2','新面板执行':'0 / 2000'})


if __name__=='__main__':
    try:
        collect()
    except BaseException:
        publish('执行中断',error=traceback.format_exc())
        raise
