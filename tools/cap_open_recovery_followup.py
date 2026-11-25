"""Queue same-history local continuations and exact replay evidence after author freeze."""
import sys
import time
import traceback
from tools.cap_open_recovery_local import summarize
from tools.cap_open_recovery_audit import replay
from tools.cap_open_recovery_finish import journal,usage
from tools.cap_open_recovery_round import OUT, DOC, read, save, seal
from pathlib import Path


def wait_for(file):
    while not (OUT/file).exists():
        if (OUT/'ERROR.json').exists():raise RuntimeError(read(OUT/'ERROR.json'))
        time.sleep(3)


def main(session):
    wait_for('locked.json')
    lock=read(OUT/'locked.json')
    name=lock['name']
    journal(progress=[
        dict(label='开发母程序复现',glob='runs/dev/P/full/*.json',total=22),
        dict(label='新面板母程序',glob='runs/confirm/P/full/*.json',total=200),
        dict(label='冻结作者定稿',file='locked.json',total=1),
        dict(label='新面板定稿配对',glob=f'runs/confirm/{name}/full/*.json',total=200),
        dict(label='同前缀局部效果与报告',file='finished.json',total=1)])
    wait_for('whole_room.json')
    summarize()
    keys=list(read(OUT/'fixtures.json'))
    replay(name,keys)
    missing=[k for k in keys if not (OUT/'local_observations/P'/(k+'.json.gz')).is_file()]
    if missing:replay('P',missing)
    (DOC/'policy.py').write_text(seal(name))
    usage(session)
    save(OUT/'replay_ready.json',dict(rooms=len(keys),author=name))


if __name__=='__main__':
    try:main(Path(sys.argv[1]))
    except Exception:
        save(OUT/'followup_error.json',dict(traceback=traceback.format_exc()));raise
