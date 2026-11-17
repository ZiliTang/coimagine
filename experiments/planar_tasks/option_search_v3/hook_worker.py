"""Local-node heuristic IPC; no inherited execution-world state or predictor RPC."""
import contextlib
import json
import resource
import sys
with contextlib.redirect_stdout(sys.stderr):
    from heuristic import rank_actions,evaluate_state
memory={};nodes={}
for line in sys.stdin:
    try:
        r=json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            if r['method']=='begin':
                nodes={};memory['context']=r['context'];memory.setdefault('history',[]).append(r['context']);result=True
            elif r['method']=='register':
                nodes[r['id']]=r['node'];result=evaluate_state(nodes[r['id']],memory)
            elif r['method']=='rank':result=rank_actions(nodes[r['id']],r['candidates'],memory)
            else:raise ValueError('unknown hook method')
        usage=resource.getrusage(resource.RUSAGE_SELF)
        reply=dict(ok=True,result=result,resources=dict(cpu_seconds=usage.ru_utime+usage.ru_stime,max_rss_kib=usage.ru_maxrss))
        data=json.dumps(reply,allow_nan=False)
    except Exception as exc:data=json.dumps(dict(ok=False,error=dict(type=type(exc).__name__,message=str(exc))))
    print(data,flush=True)
