"""Heuristic worker: generated code has no direct access to parent RPC/queues."""
import contextlib
import json
import sys
# Import prints must not corrupt the private worker protocol either.
with contextlib.redirect_stdout(sys.stderr):
    from heuristic import evaluate_nodes, score_actions
memory={};context=None;predictions={}
for line in sys.stdin:
    try:
        r=json.loads(line);method=r['method']
        with contextlib.redirect_stdout(sys.stderr):
            if method=='begin':
                context=r['context'];predictions={}
                memory.setdefault('history',[]).append(context)
                result=True
            elif method=='register':predictions[r['id']]=r['prediction'];result=True
            elif method=='score':
                nodes=r['nodes']
                for n in nodes:
                    n['prediction']=predictions.get(n['id'])
                    n['parent_prediction']=predictions.get(n['parent_id'])
                result={'values':evaluate_nodes(context,nodes,memory),
                        'actions':score_actions(context,nodes,memory)}
            else:raise ValueError('unknown heuristic worker method')
        reply=dict(ok=True,result=result)
        data=json.dumps(reply,allow_nan=False)
    except Exception as exc:data=json.dumps(dict(ok=False,error=dict(type=type(exc).__name__,message=str(exc))))
    print(data,flush=True)
