"""Developer retention counts, without turning scene labels into policy inputs."""
from pathlib import Path
from experiments.option_rl_v15.live import atomic_json,read_json


def update(out,iteration,evaluation):
    out=Path(out);path=out/'retention.json';data=read_json(path,{}) or {}
    roots=evaluation.get('development_training',{}).get('roots',[])
    if not roots:return {}
    lost=[];gained=[]
    for r in roots:
        history=data.setdefault(r['name'],[])
        if any(h['iteration']==iteration for h in history):continue
        now=bool(r['success']);previous=history[-1]['success'] if history else False
        if previous and not now:lost.append(r['name'])
        if not previous and now:gained.append(r['name'])
        history.append(dict(iteration=iteration,success=now,progress=r['progress']))
    atomic_json(path,data)
    def rate(prefix):
        selected=[r for r in roots if r['name'].startswith(prefix)]
        return sum(bool(r['success']) for r in selected)/len(selected) if selected else None
    single=[r for r in roots if r['name'].startswith(('contact','approach'))]
    return dict(retention_lost=len(lost),retention_gained=len(gained),
        dev_single_success=sum(bool(r['success']) for r in single)/len(single) if single else None,
        dev_dual_success=rate('dual'),dev_reposition_success=rate('reposition'))
