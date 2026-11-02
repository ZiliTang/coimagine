"""Atomic local telemetry shared by the trainer, dashboard and supervisor."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json,os,time


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data,indent=1,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def status(out,**fields):
    atomic_json(Path(out)/'live_status.json',dict(**fields,updated_unix=time.time(),
        updated_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')))


def stop_reason(out):
    request=read_json(Path(out)/'STOP.json')
    return request.get('reason','requested_stop') if request else None
