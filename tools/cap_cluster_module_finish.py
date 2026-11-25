"""Freeze an author's one final submission, then run the three paired continuations."""
from datetime import datetime
import time
import traceback

from tools import cap_cluster_module as experiment
from tools import cap_cluster_module_report as report

OUT, PUBLIC=experiment.OUT,experiment.PUBLIC
read,save=experiment.read,experiment.save


def lock():
    budget=read(OUT/'author_budget.json')
    deadline=datetime.fromisoformat(budget['deadline_utc'].replace('Z','+00:00')).timestamp()
    start=datetime.fromisoformat(budget['started_utc'].replace('Z','+00:00')).timestamp()
    final=PUBLIC/'FINAL.json'
    while not final.exists() and time.time()<deadline:
        time.sleep(1)
    if not final.exists() or final.stat().st_mtime>deadline:
        raise TimeoutError('no timely author-selected final')
    response=read(final)
    name=response['name']
    if not name.startswith('trial') or not name[5:].isdigit():
        raise ValueError('invalid selected trial')
    candidate=PUBLIC/(name+'.py')
    if candidate.stat().st_mtime>deadline:raise TimeoutError('late candidate')
    cases=read(PUBLIC/'cases.json')
    dev=read(experiment.study.OUT/'rooms.json')['rooms']['dev']
    src=experiment.sealed(name,{c['room']:dev[c['room']] for c in cases})
    return dict(name=name,description=response['description'],sha256=experiment.author.sha(src),
                snippet_sha256=experiment.author.sha(candidate.read_text()),
                submitted=final.stat().st_mtime,elapsed_seconds=final.stat().st_mtime-start)


def finish():
    if not (OUT/'locked.json').exists():save(OUT/'locked.json',lock())
    final=read(OUT/'locked.json')
    for mode in ('full','parent','forced','authored'):
        experiment.trial_phase(final['name'],'confirmation',mode,20)
    report.build()


if __name__=='__main__':
    try:
        finish()
    except BaseException:
        save(OUT/'ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()))
        raise
