"""Export only aggregate per-author Codex telemetry; no message contents."""
import json
import os
import sys
import time
from pathlib import Path

from tools.cap_regional_dashboard import OUT,ARMS

def tail_usage(path):
    with path.open('rb') as f:
        f.seek(0,2);end=f.tell();size=65536
        while end:
            start=max(0,end-size)
            f.seek(start);lines=f.read(end-start).splitlines()
            for line in reversed(lines[1:] if start else lines):
                if b'"token_count"' not in line:continue
                try:event=json.loads(line)
                except ValueError:continue
                packet=event.get('payload',{})
                if event.get('type')=='event_msg' and packet.get('type')=='token_count' and packet.get('info'):
                    total=packet['info']['total_token_usage']
                    return dict(updated=event['timestamp'],**{k:total.get(k) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens','total_tokens')})
            if not start:break
            size*=2
    return None


def main():
    sessions=json.loads((OUT/'author_session_ids.json').read_text())
    paths={}
    for arm in ARMS:
        paths[arm]=next((Path.home()/'.codex/sessions').rglob('*'+sessions[arm]+'.jsonl'))
        with paths[arm].open() as f:meta=json.loads(f.readline())['payload']
        assert meta['id']==sessions[arm] and meta['agent_path']=='/root/'+arm
    while True:
        meter={arm:tail_usage(path) for arm,path in paths.items()}
        temp=OUT/'token_meter.pending'
        temp.write_text(json.dumps(meter,indent=2)+'\n');os.replace(temp,OUT/'token_meter.json')
        if '--once' in sys.argv:return
        time.sleep(5)


if __name__=='__main__':main()
