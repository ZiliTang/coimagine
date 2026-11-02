"""Offline contract reconstruction for mixed PUSH target projection records."""
from __future__ import annotations
import argparse,copy,hashlib,json,math
from dataclasses import asdict
from pathlib import Path
from experiments.option_testbench.normal_travel_audit import audit as normal_audit
from experiments.option_testbench.follow_peer_eval import ACTIONS


def audit(record):
    adapted=copy.deepcopy(record)
    for i in (0,1):
        pose=record['initial']['true']['robots'][i]
        adapted['case']['instance']['robots'][i].update(x=pose[0],y=pose[1])
    adapted['manifest_actions']=record.get('manifest_actions') or [asdict(a) for a in ACTIONS]
    adapted['metric'].update(normal_reference_budget=[0.,0.],actual_displacement_xy=record['metric']['object_displacement'],actual_effect=None)
    travel=normal_audit(adapted)
    max_projection_error=0.;min_slack=math.inf;allocated=0
    for row in record['rows']:
        for i in (0,1):
            e=(row[f'r{i}'].get('effect') or {});r=e.get('same_face')
            if not r or r['status']!='allocated':continue
            t=(-e['face_normal'][1],e['face_normal'][0]);dot=lambda a:sum(a[j]*t[j] for j in (0,1))
            expected=min(max(dot(r['preferred']),r['interval'][0]),r['interval'][1])
            max_projection_error=max(max_projection_error,abs(dot(r['target'])-expected))
            min_slack=min(min_slack,r['side']*(dot(r['target'])-dot(r['peer']))-r['target_separation']);allocated+=1
    assert max_projection_error<1e-8 and min_slack>=-1e-8
    return dict(case=record['case']['name'],variant=record['variant'],travel=travel,
        allocation_samples=allocated,projection_error=max_projection_error,min_reserve_slack=min_slack if allocated else None)


def main(out):
    results=[]
    for p in sorted((out/'logs').glob('*.json')):
        r=json.loads(p.read_text())
        if r.get('rows'):results.append(audit(r))
    result=dict(results=results,offline_truth_only=True,audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'contract_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(branches=len(results),max_travel_error=max(r['travel']['max_reconstruction_error_m'] for r in results),max_projection_error=max(r['projection_error'] for r in results))))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
