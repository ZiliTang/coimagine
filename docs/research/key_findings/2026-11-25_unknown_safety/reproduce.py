"""Reproduce one frozen case without overwriting its original evidence."""
import argparse
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tools.cap_unknown_safety_confirm import verify
from tools.cap_unknown_safety import run,fixture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pair',required=True)
p.add_argument('--arm',choices=['empty_observe','blocked_observe','empty_no_observe'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
f=verify();parameters=f['panels'][a.pair]
r=run(fixture(a.arm=='blocked_observe',**parameters),observe=a.arm!='empty_no_observe')
r['reproduction']=dict(pair=a.pair,arm=a.arm,source_manifest=f['sources'])
payload=json.dumps(r,ensure_ascii=False,separators=(',',':')).encode()
with a.output.open('xb')as out:out.write(gzip.compress(payload)if a.output.suffix=='.gz'else payload)
print(json.dumps(r['summary'],ensure_ascii=False))
