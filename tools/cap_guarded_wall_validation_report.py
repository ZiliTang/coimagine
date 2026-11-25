"""Paired counts for the frozen wall validation. Every unmeasured layout stays in the inventory."""
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from tools.cap_guarded_wall_validation import ROOT, SIZES, ARMS, loop, risk_decision

DEST=Path('docs/research/key_findings/2026-11-25_guarded_wall_validation')


def report():
    plan=loop.read(ROOT/'plan.json')
    panels={s:[loop.read(ROOT/s/(case['id']+'.json')) for case in cases]
            for s,cases in plan['partitions'].items()}
    seen=set()
    for packets in panels.values():
        for p in packets:
            assert p['fixture_hash'] not in seen
            seen.add(p['fixture_hash'])
            if p['status']=='measured':
                assert len(p['branches'])==2
                assert all(b['outcome']['public_history']==p['public_history'] for b in p['branches'])
    labels=['neither','long_only','short_only','both']
    def stratum(p):
        a,b=p['branches']
        return labels[2*int(a['safe'])+int(b['safe'])]
    rows=[p for p in panels['validation'] if p['status']=='measured']
    # Post-hoc diagnostic: supply actual pre-PUSH clearance and wall angle to the
    # same frozen neighbour rule, never suffix outcomes from validation layouts.
    normals={c['id']:math.radians(c['normal_offset']) for cases in plan['partitions'].values() for c in cases}
    def privileged(p):
        return [p['branches'][0]['outcome']['start_wall_gap_m'],normals[p['id']]]
    diagnostic_train=[dict(features=p['features'],wall=privileged(p),dose=b['dose'],safe=b['safe'])
                      for p in panels['development'] if p['status']=='measured' for b in p['branches']]
    groups=defaultdict(list)
    for p in rows:
        groups[json.dumps([p['features'], p['wall']],sort_keys=True)].append(p)
        for b in p['branches']:
            b['decisions']['true_geometry_diagnostic']=risk_decision(
                diagnostic_train,p['features'],privileged(p),b['dose'],'wall_scan')
    alias=max(groups.values(),key=len)
    alias_summary=dict(distinct=len(groups),largest=len(alias),strata=Counter(stratum(p) for p in alias),
                       ids=[p['id'] for p in alias],
                       unavoidable_binary_errors=sum(min(sum(p['branches'][i]['safe'] for p in g),
                           sum(not p['branches'][i]['safe'] for p in g)) for g in groups.values() for i in range(2)))
    keys=['old_frozen',*(f'{arm}-{n}' for n in SIZES for arm in ARMS),'always_short','always_long','true_geometry_diagnostic']
    records=[];methods={};per_arm={}
    for key in keys:
        stats=Counter();sub={k:Counter() for k in labels};perf=[]
        for p in rows:
            typ=stratum(p)
            s,l=p['branches']
            if key in ('always_short','always_long'):
                actions=['admit','reject'] if key=='always_short' else ['reject','admit']
            else:
                actions=[b['decisions'][key] for b in (s,l)]
            for b,d in zip((s,l),actions):
                stats[d]+=1
                if d=='admit':
                    stats['valid_admits' if b['safe'] else 'invalid_admits']+=1
                if d=='reject' and b['safe']:
                    stats['valid_rejected']+=1
                if d=='admit' and not b['safe']:
                    o=b['outcome']
                    stats['unsafe_with_static']+=int(not o['collision_free_through_contact_check'])
                    stats['unsafe_with_incomplete_followup']+=int(not o['followup_completed'])
            pick=1 if actions[1]=='admit' else (0 if actions[0]=='admit' else None)
            chosen='wait' if pick is None else ('safe' if (s,l)[pick]['safe'] else 'unsafe')
            stats['choice_'+chosen]+=1
            sub[typ][chosen]+=1
            if typ in ('long_only','short_only'):
                j=int(typ=='long_only')
                sub[typ]['strict_correct']+=int(actions[j]=='admit' and actions[1-j]=='reject')
                sub[typ]['safe_admit_other_defer']+=int(actions[j]=='admit' and actions[1-j]=='defer')
            sub[typ]['count']+=1
            perf.append([chosen=='safe',chosen=='unsafe',int(s['safe'] and actions[0]=='admit')+int(l['safe'] and actions[1]=='admit')])
            records.append(dict(id=p['id'],method=key,stratum=typ,short_decision=actions[0],long_decision=actions[1],chosen=chosen))
        methods[key]=dict(totals=stats,strata=sub)
        per_arm[key]=np.asarray(perf,int)
    rng=np.random.default_rng(71345)
    idx=rng.integers(len(rows),size=(5000,len(rows)))
    ci={}
    for a,b in [('wall_scan-72','nine_inputs-72'),('wall_scan-72','wall_scan-24'),('wall_scan-72','always_short')]:
        d=per_arm[a]-per_arm[b]
        rep=d[idx].mean(axis=1)
        ci[a+' minus '+b]={name:[round(float(v),4) for v in (d[:,i].mean(),*np.quantile(rep[:,i],[.025,.975]))]
                            for i,name in enumerate(('safe_choice_per_ready_layout','unsafe_choice_per_ready_layout','safe_admits_per_ready_layout'))}
    inventory={s:dict(status=Counter(p['status'] for p in packets),strata=Counter(stratum(p) for p in packets if p['status']=='measured'),
                      visible=Counter(str(p['wall'] is not None) for p in packets if p['status']=='measured'))
               for s,packets in panels.items()}
    summary=dict(inventory=inventory, public_alias=alias_summary,methods=methods,paired_bootstrap=ci)
    DEST.mkdir(exist_ok=True,parents=True)
    (DEST/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (DEST/'decisions.csv').open('w') as f:
        w=csv.DictWriter(f,records[0],lineterminator='\n');w.writeheader();w.writerows(records)
    print(inventory)
    for k,m in methods.items():print(k,dict(m['totals']),m['strata']['short_only'])
    print(ci)


if __name__=='__main__':
    report()
