"""Fit four independent dose checkpoints and evaluate the complete frozen curve."""
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from . import cap_inverse_scale as study
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save

OUT, CAPS, DOSES=study.OUT, study.CAPS, study.DOSES


def checkpoint(d):
    return (study.PARENT if d==0 else OUT/f'fit_{d}')/'models/GA_s17/offline.pt'


def fit_pending():
    active={}
    with ExitStack() as logs:
        while True:
            for d,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    if code:raise RuntimeError(f'fit {d} exited {code}; see fit_{d}.log')
                    del active[d]
            statefile=OUT/'collection_state.json'
            if statefile.exists() and (error:=read(statefile).get('error')):raise RuntimeError(error)
            done=[d for d in DOSES if (checkpoint(d).parent/'offline_complete.json').exists()]
            collected=(OUT/'collection.json').exists()
            if collected and len(done)==len(DOSES) and not active:return
            for i,d in enumerate(DOSES):
                if d in done or d in active or not (OUT/'doses'/f'{d}.json').exists():continue
                if len(active)>=(2 if collected else 1):break
                if i and not (OUT/'parts'/str(DOSES[i-1])/'packed/manifest.json').exists():break
                logfile=logs.enter_context((OUT/f'fit_{d}.log').open('a'))
                active[d]=subprocess.Popen([sys.executable,'-u','-m','experiments.planar_tasks.authoring_v2.cap_inverse_fit',str(d)],stdout=logfile,stderr=subprocess.STDOUT)
            time.sleep(4)


def evaluate():
    if (OUT/'evaluation.json').exists():return
    protocol=read(OUT/'protocol.json')
    for f,h in protocol['files'].items():assert sha(OUT/f)==h
    seals={str(d):sha(checkpoint(d)) for d in (0,*DOSES)}
    assert seals['0']==protocol['baseline_sha256']
    if (OUT/'frozen.json').exists():assert read(OUT/'frozen.json')==seals
    else:save(OUT/'frozen.json',seals)
    cases=read(OUT/'evaluation_panel.json')
    jobs=[]
    for name in [*CAPS,*[f'q{d}' for d in (0,*DOSES)]]:
        opts=dict(cap=name) if name in CAPS else dict(checkpoint=str(checkpoint(int(name[1:]))),epsilon=0)
        for case in cases:
            jobs.append(dict(case=case,condition=name,directory=str(OUT/'evaluation'/name/case['case_id']),**opts))
    rows=[]
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=study.family.initialize) as pool:
        for f in as_completed([pool.submit(study.family.worker,j) for j in jobs]):
            r=f.result()
            assert r['complete'] and r['error'] is None
            rows.append(dict(case_id=r['case_id'],condition=r['condition'],success=int(r['success']),
                             ticks=r['ticks'],decisions=r['decisions'],stop=r['stop']))
            if len(rows)%20==0:study.state('eval',completed=len(rows),total=len(jobs))
    assert len(rows)==len({(r['case_id'],r['condition']) for r in rows})==7200
    save(OUT/'evaluation.json',sorted(rows,key=lambda r:(r['case_id'],r['condition'])))


if __name__=='__main__':
    try:
        fit_pending()
        evaluate()
        from tools.cap_inverse_scale_report import report
        report()
    except BaseException:
        save(OUT/'pipeline_error.json',dict(error=traceback.format_exc()))
        raise
