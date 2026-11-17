"""Read-only artifacts for the existing research workbench; no policy inputs."""
from pathlib import Path
from experiments.planar_tasks.relational_search.pipeline import read,unpack,identifier


def campaigns(root):
    root=Path(root).resolve();out=[]
    for p in sorted(root.glob('*/campaign.json')):
        if not p.resolve().is_relative_to(root):continue
        try:
            s=read(p)
            if not isinstance(s.get('runs'),list):continue
            out.append(dict(id=p.parent.name,status=s['status'],created=s['created'],profile=s.get('profile','legacy'),
                            phase=s.get('phase'),completed_rounds=s.get('completed_rounds')))
        except (OSError,ValueError,KeyError):continue
    return sorted(out,key=lambda r:r['created'])


def location(root,name):
    root=Path(root).resolve();p=(root/identifier(name)).resolve()
    if not p.is_relative_to(root):raise ValueError('path')
    return p


def detail(root,name):
    p=location(root,name);s=read(p/'campaign.json');active=[]
    for r in s.get('runs',[]):
        if r.get('status')=='running':
            q=p/'runs'/identifier(r['id'])/'progress.json'
            active.append(dict(r,progress=read(q) if q.exists() else None))
    if (p/'metrics.json').exists():metrics=read(p/'metrics.json')
    else:
        # Old logs expose only the metrics actually recorded; never fill missing with0.
        updates=[];rounds=[]
        for u in read(p/'training_progress.json') if (p/'training_progress.json').exists() else []:
            for row in u['update'].get('curve',[]):updates.append(dict(row,optimizer_step=len(updates)+1,round=u['episode']))
            rounds.append(dict(round=u['episode'],logp_recompute_error=u['update'].get('logp_recompute_error')))
        metrics=dict(rounds=rounds,updates=updates)
    provenance=read(p/'inputs/provenance.json') if (p/'inputs/provenance.json').exists() else {}
    return dict(state=s,metrics=metrics,workers=active,schema=provenance.get('schema'),
                note='训练采样表现，不是独立测试成功率；回放包含仅供人工查看的完整几何。')


def replay(root,name,rid):
    p=location(root,name);d=(p/'runs'/identifier(rid)).resolve()
    if not d.is_relative_to(p):raise ValueError('path')
    if (d/'summary.json').exists() and read(d/'summary.json').get('record_details') is False:
        return dict(available=False,reason='此 trial 未启用详细记录；仅保存训练摘要。',frames=[])
    return unpack(d/'result.json.gz')['replay']
