"""Bounded supervisor: source seal, independent stage logs, final exit evidence."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from . import da_joint_study as D


def main():
    host=D.HOST;host.mkdir(parents=True,exist_ok=True)
    root=D.V.ROOT
    files=list((root/'experiments/planar_tasks/semantic_rl').glob('*.py'))
    files += list((root/'experiments/planar_tasks/authoring_v2').glob('da_joint*.py'))
    hashes={str(p.relative_to(root)):D.V.sha(p) for p in files}
    dest=D.BASE/'model_sources'
    if (dest/'manifest.json').exists() and D.V.read(dest/'manifest.json')!=hashes:
        raise ValueError('model-stage sources changed; explicit version/audit required')
    for p in files:
        target=dest/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    D.V.save(dest/'manifest.json',hashes)
    deadline=D.V.read(host/'started.json')['at']+28800
    stages=('extract_base','preflight','extract_repair','fit_base','fit_repair','evaluate','summarize')
    def remaining():return deadline-time.time()
    for stage in stages:
        if stage=='extract_repair':
            while not (D.BASE/'branches/result.json').exists():
                if remaining()<=0:raise TimeoutError('waiting for branches exhausted study budget')
                time.sleep(5)
            rr=D.V.read(D.BASE/'branches/result.json')['rows']
            if any(r['stop']=='wall_budget' or r.get('error') for r in rr):
                raise ValueError('branch infrastructure error must be reviewed')
        if remaining()<=0:raise TimeoutError('8h total study wall')
        for rel,sha in hashes.items():
            if D.V.sha(root/rel)!=sha:raise ValueError('running source changed: '+rel)
        cmd=[sys.executable,'-u','-m','experiments.planar_tasks.authoring_v2.da_joint_models',stage]
        D.V.save(host/'supervisor.json',dict(stage=stage,status='running',at=time.time(),remaining_seconds=remaining(),command=cmd))
        started=time.time()
        with (host/(stage+'.log')).open('a') as log:
            process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=remaining())
            except BaseException:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                raise
        D.V.save(host/(stage+'.exit.json'),dict(code=code,started=started,finished=time.time()))
        if code:raise RuntimeError(f'{stage} exited {code}; see its log')
    D.V.save(host/'study.exit.json',dict(code=0,status='complete',at=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException:
        D.V.save(D.HOST/'study.exit.json',dict(code=1,status='incomplete',at=time.time(),traceback=traceback.format_exc()))
        raise
