"""Evaluate the original authored router on the locked expanding-pool confirmation panel."""
import csv
from tools import cap_expanding_pool as study
from tools.cap_option_handoff_opportunity_report import paired

router = study.d.source
read = study.e.read


def run_room(key):
    row = router.play(study.ROOMS[key])
    result = row['result']
    study.e.save(study.OUT/'held_teacher'/(key+'.json'),
                 {k:result[k] for k in ('success','ticks','history')})


def main():
    study.setup()
    keys = study.WAVES[5]
    assert len(keys)==200 and keys==read(study.OUT/'lock.json')['held']
    router.ROUTE = study.d.TEACHER
    freeze = read(router.OUT/'frozen.json')
    study.e.pool(run_room,keys,study.WORKERS)
    with (study.DOC/'held.csv').open() as file:
        existing = {row['room']:row for row in csv.DictReader(file)}
    assert set(keys)==set(existing)
    rows = []
    for key in keys:
        r = existing[key]
        outcome = read(study.OUT/'held_teacher'/(key+'.json'))['success']
        rows.append(dict(room=key,stratum=r['stratum'],teacher=outcome,
                         **{m:int(r[m]) for m in ('A','B','initial','final')}))
    with (study.DOC/'held_teacher.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(profile=study.PROFILE,router_sha=freeze['router_sha'],revision=freeze['selection']['revision'],
                   held_seed=study.HOLD_SEED)
    summary['paired'] = {m:paired([r['teacher'] for r in rows],[r[m] for r in rows]) for m in ('initial','final','A','B')}
    summary['strata'] = {g:{m:sum(r[m] for r in rows if r['stratum']==g) for m in ('teacher','initial','final')} for g in ('A_only','B_only','both','neither')}
    study.e.save(study.OUT/'teacher_summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
