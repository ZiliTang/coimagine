"""Resumable host sequence after autonomous model rounds; no holdout tuning."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from tools.cap_symbolic_general import ROOT, OUT, read, save


def run(module,args,label):
    path=OUT/(label+'.log')
    with path.open('a')as log:
        process=subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m',module,*args],
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError(f'{label} exited {process.returncode}; see {path}')


def dev_worker(job):
    from experiments.planar_tasks.authoring_v2.symbolic_general import simulation
    from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import EffectModel
    from tools.cap_symbolic_general_runtime import Controller
    room,fixture,source,model_path,path=job
    result=simulation.episode(fixture,source,controller=Controller(EffectModel(read(model_path))),
                              keep_observations=False,replay=False,wall_seconds=240)
    save(path,result)
    return dict(room=room,**result['summary'])


def development_check():
    frozen=read(OUT/'policy_frozen.json');source=Path(frozen['source']).read_text()
    fixtures=read(OUT/'panels.json')['groups']['calibration'];jobs=[]
    folder=OUT/'deployment_development_v2'
    for room,fixture in fixtures.items():
        path=folder/(room+'.json.gz')
        if not path.exists():jobs.append((room,fixture,source,OUT/'models/v2.json',path))
    with ProcessPoolExecutor(max_workers=6)as pool:
        for i,result in enumerate(pool.map(dev_worker,jobs)):
            save(OUT/'progress.json',dict(stage='final_model_deployment_development_check',completed=i+1,total=len(jobs),last=result,updated=time.time()))
    rows=[dict(room=p.name[:-8],**read(p)['summary'])for p in sorted(folder.glob('*.json.gz'))]
    result=dict(rooms=len(rows),success=sum(r['success']for r in rows),rows=rows)
    save(folder/'summary.json',result)
    if result['rooms']!=16:raise ValueError('incomplete development deployment check')
    if not result['success']:
        save(OUT/'NEEDS_DEVELOPMENT.json',dict(reason='final gated model has no complete task on development check',at=time.time(),confirmation_opened=False))
        return False
    return True


def main():
    while not(OUT/'models/manifest.json').exists():
        info=read(OUT/'model_loop_host.json');pid=info['pid'];status=Path(f'/proc/{pid}/stat')
        if not status.exists()or status.read_text().split()[2]=='Z':
            raise RuntimeError('model loop ended without all model freezes; inspect model_loop.log')
        time.sleep(15)
    if not development_check():return
    source=read(OUT/'policy_frozen.json')['source']
    run('tools.cap_symbolic_general_evaluate',['freeze','--policy',source],'evaluation_freeze')
    run('tools.cap_symbolic_general_evaluate',['prepare','--workers','4'],'confirmation_prepare')
    jobs=[]
    for stage,workers in [('paired',3),('tasks',6)]:
        log=(OUT/('confirmation_'+stage+'.log')).open('a')
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','tools.cap_symbolic_general_evaluate',stage,'--workers',str(workers)],
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((stage,process,log));save(OUT/('confirmation_'+stage+'_host.json'),dict(pid=process.pid,workers=workers))
    failures=[]
    for stage,process,log in jobs:
        code=process.wait();log.close()
        if code:failures.append((stage,code))
    if failures:raise RuntimeError('confirmation workers failed: '+repr(failures))
    run('tools.cap_symbolic_general_evaluate',['summarize'],'confirmation_summary')
    for operation in ('rechecks','costs','raw-replay'):
        run('tools.cap_symbolic_general_audit',[operation],'final_'+operation)
    run('tools.cap_symbolic_general_report',[],'final_report')
    save(OUT/'finish_host_complete.json',dict(at=time.time(),status='complete'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(OUT/'finish_host_error.json',dict(at=time.time(),error=type(exc).__name__+': '+str(exc)))
        raise
