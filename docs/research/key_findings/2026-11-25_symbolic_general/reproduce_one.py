"""Run from the repository root; writes a new file, preserves frozen evidence."""
import argparse
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tools.cap_symbolic_general import OUT, read
from tools.cap_symbolic_general_evaluate import verify_freeze
from tools.cap_symbolic_general_compile import compile_policy
from tools.cap_symbolic_general_runtime import Controller
from experiments.planar_tasks.authoring_v2.symbolic_general import simulation, scaffold
from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import EffectModel

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--version',type=int,choices=[0,1,2],required=True)
p.add_argument('--room',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
frozen=verify_freeze()
source=Path(frozen['source']).read_text()
fixture=read(OUT/'panels.json')['groups']['confirmation'][a.room]
compile_policy(scaffold.verify_source(source))
model=EffectModel(read(frozen['model_files'][a.version]))
result=simulation.episode(fixture,source,controller=Controller(model),
    keep_observations=False,replay=True,wall_seconds=240)
result['reproduction']=dict(room=a.room,version=a.version,policy_sha256=frozen['source_sha256'],
    model_sha256=frozen['model_sha256'][a.version],
    caveat='Wall-clock budget can make branch counts differ with execution load.')
payload=json.dumps(result,ensure_ascii=False,separators=(',',':')).encode()
with a.output.open('xb')as f:f.write(gzip.compress(payload)if a.output.suffix=='.gz'else payload)
print(json.dumps(result['summary'],ensure_ascii=False))
