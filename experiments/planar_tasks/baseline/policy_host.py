"""Protocol-only deployment adapter. No action choice, repair or fallback."""
import json
import sys
from policy import Policy

if '--validate' in sys.argv:
    import inspect
    request=json.loads(sys.stdin.readline())
    inspect.signature(Policy).bind(request['contract'])
    instance=Policy(request['contract'])
    if not callable(getattr(instance,'act',None)):raise TypeError('Policy.act must be callable')
    inspect.signature(instance.act).bind({})
    print(json.dumps({'validation':'load_and_interface_only'}),flush=True)
    sys.exit(0)

policy=None
for line in sys.stdin:
    request=json.loads(line)
    if request.get('reset'):
        policy=Policy(request['contract'])
    if policy is None:
        raise RuntimeError('first request requires reset and contract')
    action=policy.act(request['observation'])
    print(json.dumps(action,allow_nan=False),flush=True)
