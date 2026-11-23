"""Inverse-frequency family sampling with fixed, nested complete seven-CAP datasets."""
import multiprocessing as mp
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import cap_family_augmentation as family
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save

PARENT, CAPS = family.PARENT, family.CAPS
OUT = family.BASE/'cap_inverse_scale_20261119_01'
DOSES = (250, 500, 750, 1001)
BATCH = 16
SEED = 2026111951
EVAL_BATCHES = 3


def state(stage, **values):
    save(OUT/(stage+'_state.json'),values)


def freeze():
    path=OUT/'protocol.json'
    if path.exists():
        protocol=read(path)
        assert all(sha(OUT/f)==v for f,v in protocol['files'].items())
        return
    OUT.mkdir(parents=True,exist_ok=True)
    source=[*family.known_cases(),*read(family.OUT/'evaluation_panel.json')]
    for p in sorted(family.OUT.glob('round_*_plan.json')):
        source.extend(read(p))
    used={r['content_sha256'] for r in source}
    ids={r['case_id'] for r in source}
    anchors=read(family.OUT/'anchors.json')
    assert len(anchors)==200 and all(p['case_id'] in read(PARENT/'split.json')['train'] for p in anchors)
    assert len({p['case_id'] for p in anchors})==len(anchors)
    w=[1/len(p['winners']) for p in anchors]
    for p,weight in zip(anchors,w,strict=True):
        p['probability']=weight/sum(w)
    save(OUT/'anchors.json',anchors)
    test=[]
    for n in range(EVAL_BATCHES):
        for r in family.rooms.generate(SEED+100+n)['cases']:
            f=r['fixture']; h=family.content(f); cid=f"inverse_test_{n}_{r['id']}"
            assert h not in used and cid not in ids
            used.add(h);ids.add(cid)
            test.append(dict(case_id=cid,fixture=f,content_sha256=h,movable_count=r['movable_count']))
    save(OUT/'evaluation_panel.json',test)
    save(OUT/'forbidden.json',sorted(used))
    frozen=read(PARENT/'ALL_MODELS_FROZEN.json')['files']['models/GA_s17/offline.pt']
    assert sha(PARENT/'models/GA_s17/offline.pt')==frozen
    save(path,dict(training_seed=17, collection_seed=SEED, doses=[0,*DOSES], caps=CAPS,
         sampling='iid parents with replacement; fixed weight 1/k for original k>0, normalized over 200 prior training anchors; do not update weights',
         axes='new distinct complete requested joint-action sequences on successful real runs within this collection; additionally report all runs, boundaries and parents',
         nested='first full neighbor (all seven CAPs, successes + failures) reaching each threshold in predetermined generation order; original 4256 train runs always included',
         fit='GA offline from scratch for each nonzero dose, 16000 V + 8000 Q updates, fixed original normalizers and boundary-uniform sampling; zero reuses archived seed17',
         primary='1001 minus zero on previously unseen scenes solved by exactly one of the seven frozen CAPs',
         secondary='all, union, multiple, outside; adjacent and intermediate differences descriptive; no best checkpoint selection',
         uncertainty='10000 paired resamples of scenes stratified by movable count, fixed training seed; fractions within each subset',
         eval='600 freshly generated roots with 150 for each movable count 2..5, same roots for seven CAPs and five frozen models',
         baseline_sha256=frozen,files={f:sha(OUT/f) for f in ('anchors.json','evaluation_panel.json','forbidden.json')}))


def run_one(job):
    case,cap=job
    directory=OUT/'rollouts'/case['case_id']/cap
    args=dict(case=case,cap=cap,condition=cap,directory=str(directory))
    result=family.fast_runtime.run(args)
    if not result['complete'] or result['error']:
        raise RuntimeError(result)
    path=directory/'result.json.gz'
    row=dict(path=str(path),sha256=sha(path),success=result['success'],cap_id=cap,case_id=case['case_id'])
    shard=family.full_data.extract(('.',str(OUT),row,10000+case['index']))
    signature=family.action_sequence(dict(replay=str(path))) if result['success'] else None
    return dict(case_id=row['case_id'], index=case['index'], parent=case['parent'], cap=cap,
                success=bool(result['success']),sha256=row['sha256'],replay=str(path),sequence=signature,shard=shard)


def stats(rows):
    return dict(runs=len(rows),success=sum(r['success'] for r in rows),distinct=len({r['sequence'] for r in rows if r['success']}),
                families=len({r['parent'] for r in rows}),boundaries=sum(r['shard']['boundaries'] for r in rows))


def collect():
    freeze()
    if (OUT/'collection.json').exists():return
    anchors=read(OUT/'anchors.json')
    forbidden=set(read(OUT/'forbidden.json'))
    weights=[p['probability'] for p in anchors]
    archive=set()
    records=[]
    sequence_set=set()
    completed=set()
    batch=0
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=family.initialize) as pool:
        while len(completed)<len(DOSES):
            batch+=1
            rng=random.Random(SEED+10000*batch)
            plan=[]
            for i in range(BATCH):
                parent=rng.choices(anchors,weights=weights)[0]
                f=family.variant(parent,rng.getrandbits(64))
                h=family.content(f)
                assert h not in forbidden and h not in archive
                archive.add(h)
                number=(batch-1)*BATCH+i
                plan.append(dict(index=number,parent=parent['case_id'],case_id=f'inv_{number:05d}',fixture=f,content_sha256=h))
            planfile=OUT/'batches'/f'{batch:03d}_plan.json'
            if planfile.exists():assert read(planfile)==plan
            else:save(planfile,plan)
            dest=planfile.with_name(f'{batch:03d}_results.json')
            if dest.exists():
                rows=read(dest)
            else:
                rows=[]
                jobs=[(case,cap) for case in plan for cap in CAPS]
                for future in as_completed([pool.submit(run_one,j) for j in jobs]):
                    rows.append(future.result())
                    if len(rows)%8==0:
                        state('collection',batch=batch,complete=False,**stats(records+rows))
                rows.sort(key=lambda r:(r['index'],CAPS.index(r['cap'])))
                save(dest,rows)
            for j in range(0,len(rows),len(CAPS)):
                group=rows[j:j+len(CAPS)]
                assert [r['cap'] for r in group]==CAPS and len({r['index'] for r in group})==1
                records.extend(group)
                sequence_set.update(r['sequence'] for r in group if r['success'])
                for d in DOSES:
                    if d not in completed and len(sequence_set)>=d:
                        descriptor=dict(threshold=d,batch=batch,neighbors=len(records)//7,**stats(records))
                        prefix=OUT/'doses'/f'{d}.json'
                        if prefix.exists():assert read(prefix)==descriptor
                        else:save(prefix,descriptor)
                        completed.add(d)
            state('collection',batch=batch,complete=False,**stats(records))
    save(OUT/'collection.json',dict(batches=batch,neighbors=len(records)//7,**stats(records)))
    state('collection',batch=batch,complete=True,**stats(records))


if __name__=='__main__':
    import traceback
    try:collect()
    except BaseException:
        state('collection',error=traceback.format_exc())
        raise
