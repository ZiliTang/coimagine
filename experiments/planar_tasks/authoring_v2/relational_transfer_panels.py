"""Outcome-independent train/IID/cross-layout panels, generated before fitting."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import argparse,json,math,random
from experiments.planar_tasks.relational_search import transfer_tasks as T,pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import digest

SEEDS={'train':2026111711,'iid':2026111712,'transfer':2026111713}
BOUNDS=dict(translation=.15,target_position=.06,robot_position=.08,goal_position=.12,obstacle_position=.08,target_angle=.08,robot_angle=.2,obstacle_angle=.12,obstacle_scale=[.8,1.2])


def relation(f):
    obj=f['object'];b=f['movable_objects'][0];dx=f['goal']['x']-obj['x'];dy=f['goal']['y']-obj['y'];theta=math.atan2(dy,dx);length=math.hypot(dx,dy)
    vx,vy=-math.sin(theta),math.cos(theta);offset=(b['x']-obj['x'])*vx+(b['y']-obj['y'])*vy
    a=.5*(.25*abs(math.sin(obj['angle']-theta))+abs(math.cos(obj['angle']-theta)))
    bb=.5*(b['width']*abs(math.sin(b['angle']-theta))+b['height']*abs(math.cos(b['angle']-theta)))
    along=(b['x']-obj['x'])*math.cos(theta)+(b['y']-obj['y'])*math.sin(theta)
    return dict(overlap=abs(offset)<a+bb and 0<along<length,margin=abs(offset)-a-bb)


def draw(base,rng,identifier,split,seen):
    failures=[]
    for attempt in range(1000):
        f=deepcopy(base['fixture']);translation=[rng.uniform(-.15,.15) for _ in range(2)]
        objects=[f['object'],*f['robots'],*f['movable_objects'],f['goal']]
        for b in objects:b['x']+=translation[0];b['y']+=translation[1]
        for body,p,a in [(f['object'],.06,.08),*[(r,.08,.2) for r in f['robots']],*[(r,.08,.12) for r in f['movable_objects']],(f['goal'],.12,0)]:
            body['x']+=rng.uniform(-p,p);body['y']+=rng.uniform(-p,p)
            if 'angle' in body:body['angle']+=rng.uniform(-a,a)
        f['movable_objects'][0]['width']*=rng.uniform(.8,1.2);f['movable_objects'][0]['height']*=rng.uniform(.8,1.2)
        # Opaque instance labels are not identifiers of relation/task categories.
        f['movable_objects'][0]['label']='body-'+str(rng.randrange(1000000,9999999))
        h=digest(f)
        try:
            if relation(f)['overlap']!=base['host_annotation']['straight_corridor_overlap']:raise ValueError('relation stratum changed')
            scene,boxes=X.fixture(f)
            X.T.validate_geometry(replace(scene,object_pose=replace(scene.object_pose,x=scene.goal_centre[0],y=scene.goal_centre[1])),boxes)
            if h in seen:raise ValueError('duplicate')
        except ValueError as exc:failures.append(dict(fixture_sha256=h,reason=str(exc)));continue
        seen.add(h);return dict(id=identifier,split=split,base_id=base['id'],layout=base['layout'],relation='blocked' if base['host_annotation']['straight_corridor_overlap'] else 'clear',fixture=f,fixture_sha256=h,rejected=failures,feasibility='initial and same-orientation goal poses geometry-valid; dynamics unverified')
    raise ValueError('geometry rejection budget')


def build():
    roots=T.cases();seen={r['fixture_sha256'] for r in roots};out={}
    for split,seed in SEEDS.items():
        rng=random.Random(seed);base=roots[:4] if split in ('train','iid') else roots[4:];count=6 if split=='iid' else 12;rows=[]
        for r in base:
            for _ in range(count):rows.append(draw(r,rng,split+'_'+str(len(rows)).zfill(4),split,seen))
        out[split]=dict(seed=seed,bounds=BOUNDS,cases=rows)
    out['roots']=dict(cases=[dict(id='root_'+r['id'],base_id=r['id'],layout=r['layout'],relation='blocked' if r['host_annotation']['straight_corridor_overlap'] else 'clear',split='root_training' if i<4 else 'root_seen_diagnostic',fixture=r['fixture'],fixture_sha256=r['fixture_sha256']) for i,r in enumerate(roots)])
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name,panel in build().items():(out/(name+'.json')).write_text(json.dumps(panel,ensure_ascii=False,indent=2)+'\n')
    manifest={p.name:digest(json.loads(p.read_text())) for p in sorted(out.glob('*.json'))};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
