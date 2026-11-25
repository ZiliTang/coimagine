"""Test the frozen author's public family and local contact-and-progress claim."""
import csv

from tools import cap_free_repair_round as r


def anchor(boundaries, rule):
    for i, b in enumerate(boundaries):
        tick=b['tick']
        low,high=rule['gap_m_exclusive']
        if (tick>rule['tick_max_inclusive'] or not low<b['gap']<high
                or b['touch']!=rule['touch_equals'] or b['static_contact']!=rule['static_contact_equals']
                or b['action']['target_id'] is not None
                or max(y for _,y in b['xy'])<rule['max_robot_y_m_min_inclusive']):
            continue
        lag_min,lag_max=rule['prior_boundary_lag_ticks_inclusive']
        prior=next((p for p in reversed(boundaries[:i]) if lag_min<=tick-p['tick']<=lag_max), None)
        if prior and round(prior['gap']-b['gap'],3)<=rule['prior_gap_minus_anchor_gap_m_max_inclusive']:
            return b
    return None


def local_tick(boundaries, b, spec):
    for p in boundaries:
        if (b['tick']<p['tick']<=b['tick']+spec['horizon_ticks_inclusive']
                and p['gap']>=0 and p['touch']>=spec['touch_min_inclusive']
                and round(b['gap']-p['gap'],3)>=spec['anchor_gap_minus_boundary_gap_m_min_inclusive']):
            return p['tick']
    return None


def paired(pairs):
    return dict(n=len(pairs),parent=sum(a for a,b in pairs),selected=sum(b for a,b in pairs),
                selected_only=sum(b and not a for a,b in pairs),parent_only=sum(a and not b for a,b in pairs),
                both=sum(a and b for a,b in pairs),neither=sum(not a and not b for a,b in pairs))


def main():
    r.configure()
    lock=r.read(r.OUT/'locked.json')
    rule=lock['family']['anchor']
    spec=lock['declared_local_test']
    fixtures=r.read(r.OUT/'fixtures.json')
    anchors={}
    full_pair={True:[],False:[]}
    for room in fixtures:
        base=r.read(r.OUT/'evaluation/parent'/(room+'.json'))
        new=r.read(r.OUT/'evaluation/selected'/(room+'.json'))
        b=anchor(base['boundaries'],rule)
        if b is not None:anchors[room]=b
        full_pair[b is not None].append((bool(base['result']['success']),bool(new['result']['success'])))
    groups={group:paired(full_pair[flag]) for group,flag in [('in_family',True),('outside_family',False)]}
    print('baseline-defined paired full-room results:',groups,flush=True)
    prior=r.host.seal('P')
    arms={'parent':prior,'selected':r.host.seal(lock['name'])}
    dest=r.OUT/'declared_replays'
    for arm,source in arms.items():
        jobs=[(k,fixtures[k],True,b['tick'],'boundary',False,r.host.enabled(source),prior) for k,b in anchors.items()]
        if jobs:r.host.old.collect(source,jobs,r.host.episode,min(len(jobs),12),dest/arm)
    rows=[]
    for k,b in anchors.items():
        base=r.read(dest/'parent'/(k+'.json'))
        new=r.read(dest/'selected'/(k+'.json'))
        assert base['result']==r.read(r.OUT/'evaluation/parent'/(k+'.json'))['result']
        a,c=base['reached'],new['reached']
        assert a==c and a['history']==b['public_history'] and a['tick']==b['tick'] and a['observation_sha256']
        row=dict(room=k,tick=b['tick'],history_sha256=a['history'],observation_sha256=a['observation_sha256'])
        for arm,outcome in [('parent',base),('selected',new)]:
            row[arm+'_pass_tick']=local_tick(outcome['boundaries'],b,spec)
            row[arm+'_suffix_success']=int(outcome['result']['success'])
        rows.append(row)
    if rows:
        with (r.DOC/'declared_local.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    report=dict(confirmed_panel=200,source=lock['source_sha256'],rooms_by_baseline_family=groups,
                matched_history_and_observation=len(rows),unreached=0,
                local=paired([(x['parent_pass_tick'] is not None,x['selected_pass_tick'] is not None) for x in rows]),
                suffix_success=paired([(x['parent_suffix_success'],x['selected_suffix_success']) for x in rows]))
    r.save(r.DOC/'declared_local.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
