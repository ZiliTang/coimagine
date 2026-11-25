"""Offline diagnostic of public scan aliasing in the largest equal-input group."""
from collections import defaultdict
import json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tools.cap_guarded_wall_validation import ROOT,room,setup,loop,forks,scan_descriptor
def one(c):
 fixture,_,_=room(c);record,anchor,*_=setup()
 with forks.archive.S.phase.world(fixture) as (w,_):
  scans=[]
  walls=[]
  def add():
   obs=forks.archive.S.observe(w)
   walls.append(scan_descriptor(obs,anchor['original_action']))
   scans.append([[ [round(v,6) for v in r['scan']['distances']], r['scan']['valid'],r['scan']['category'],r['scan']['entity_ids']] for r in obs['robots']])
  add()
  for row in record['steps'][:anchor['step_index']]:
   for seg in row['segments']:
    w.step(seg['requested'],remaining_ticks=seg['end_tick']-seg['start_tick'])
    add()
  return dict(id=c['id'],now=sha(scans[-1]),prefix=sha(scans),earlier_descriptor=any(v is not None for v in walls[:-1]))
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
if __name__=='__main__':
 ids=loop.read(Path('docs/research/key_findings/2026-11-25_guarded_wall_validation/metrics.json'))['public_alias']['ids']
 plan=loop.read(ROOT/'plan.json')
 cs=[c for c in plan['partitions']['validation'] if c['id'] in ids]
 with ProcessPoolExecutor(6) as pool: hashes=list(pool.map(one,cs))
 for key in ('now','prefix'):
  gs=defaultdict(list)
  for h in hashes:gs[h[key]].append(h['id'])
  print(key,len(gs),sorted([len(g) for g in gs.values()],reverse=True)[:6])
 print('earlier descriptor',sum(r['earlier_descriptor'] for r in hashes))
 loop.seal(ROOT/'prefix_alias_audit.json',hashes)
