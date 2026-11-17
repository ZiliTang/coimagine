"""Refresh this report after the already-running experiment exits; no experiments."""
import json, os, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
HOST=ROOT/'outputs/semantic_rl_authors/random_room_coverage_20261117_02'
OUT=ROOT/'outputs/research/supervisor_report_20261117/latest'
OUT.mkdir(exist_ok=True)

def run():
    while True:
        exits={}
        for name in ('study_amended','acceptance'):
            path=HOST/(name+'.exit.json')
            if path.exists():
                try: exits[name]=json.loads(path.read_text())
                except json.JSONDecodeError: pass
        if any(x['code'] for x in exits.values()):
            raise RuntimeError('Experiment exited unsuccessfully: '+str(exits))
        if len(exits)==2:break
        (OUT/'refresh_waiting.json').write_text(json.dumps(dict(status='waiting_for_existing_evaluation_and_acceptance',checked_at=time.time(),exits=exits)))
        time.sleep(30)
    subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'build_report.py')],cwd=ROOT,check=True,env=dict(os.environ,PYTHONPATH='.'))
    subprocess.run(['/tmp/coimagine_report_browser/bin/python',str(HERE/'check_report.py')],cwd=ROOT,check=True)
    (OUT/'refresh_complete.json').write_text(json.dumps(dict(status='complete',at=time.time(),exits=exits,report=str(HERE/'index.html'))))

if __name__=='__main__':
    try:run()
    except BaseException as e:
        (OUT/'refresh_failure.json').write_text(json.dumps(dict(at=time.time(),error=repr(e))))
        raise
