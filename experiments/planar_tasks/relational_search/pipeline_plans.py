"""Isolated initial-observation sequence generator, not an online policy."""
from pathlib import Path
import json
import subprocess
import tempfile
from experiments.planar_tasks.sandbox import policy_command
from .contract import canonical, public_observation
from .pipeline_execution import validate_commands

WRAPPER='''
import json,sys
request=json.loads(sys.stdin.read())
result=make_sequence(request['observation'],request['seed_commands'])
print(json.dumps(result,allow_nan=False))
'''


def generate(source,observation,seed_commands):
    if not isinstance(source,str) or not 0<len(source.encode())<=32768:
        raise ValueError('plan source requires 1..32768 bytes')
    packet=dict(observation=public_observation(observation),seed_commands=validate_commands(seed_commands))
    with tempfile.TemporaryDirectory(prefix='coimagine_plan_') as tmp:
        p=Path(tmp);(p/'policy.py').write_text(source+'\n'+WRAPPER)
        with (p/'stdout').open('w+b') as out,(p/'stderr').open('w+b') as err:
            proc=subprocess.run(policy_command(p/'policy.py'),input=canonical(packet).encode(),stdout=out,stderr=err,timeout=30)
            out.seek(0);err.seek(0);data=out.read(1048576);errors=err.read(2000)
        if proc.returncode:raise RuntimeError('isolated sequence generator failed: '+errors.decode(errors='replace'))
    commands=validate_commands(json.loads(data))
    if sum(a['ticks'] for a in commands)>1800:raise ValueError('generated sequence exceeds episode budget')
    return commands
