"""Seeded random rooms. Sampling filters geometry only, never policy outcomes."""
import math,random,json,hashlib
from pathlib import Path
from collections import Counter
from experiments.planar_tasks.relational_search import pipeline_execution as X

VERSION='random-room-200-v1'
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/research/random_room_20261117_01'

def polygon(x,y,w,h,a):
    c,s=math.cos(a),math.sin(a)
    return [(x+c*u-s*v,y+s*u+c*v) for u,v in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]]

def separated(a,b,gap=.06):
    for p in (a,b):
        for i in range(len(p)):
            dx=p[(i+1)%len(p)][0]-p[i][0];dy=p[(i+1)%len(p)][1]-p[i][1];d=math.hypot(dx,dy)
            n=(-dy/d,dx/d);u=[x*n[0]+y*n[1] for x,y in a];v=[x*n[0]+y*n[1] for x,y in b]
            if max(u)+gap<=min(v) or max(v)+gap<=min(u):return True
    return False

def inside(p):return all(.11<=x<=5.89 and .11<=y<=4.69 for x,y in p)

def generate(seed=2026111703):
    rng=random.Random(seed);rows=[]
    for count in range(2,6):
        for j in range(50):
            occupied=[]
            def place(w,h,near=None):
                for _ in range(10000):
                    a=rng.uniform(-math.pi,math.pi)
                    if near is None:x,y=rng.uniform(.3,5.7),rng.uniform(.3,4.5)
                    else:
                        angle=rng.uniform(-math.pi,math.pi);radius=rng.uniform(.65,1.4)
                        x,y=near[0]+radius*math.cos(angle),near[1]+radius*math.sin(angle)
                    p=polygon(x,y,w,h,a)
                    if inside(p) and all(separated(p,q) for q in occupied):occupied.append(p);return x,y,a
                raise RuntimeError('geometric rejection limit')
            x,y,a=place(.25,1.);target=dict(x=x,y=y,angle=a)
            boxes=[]
            for k in range(1,count):
                w,h=rng.uniform(.25,.85),rng.uniform(.25,.85);bx,by,ba=place(w,h)
                boxes.append(dict(label='entity-'+chr(97+k),x=bx,y=by,angle=ba,width=w,height=h,mass=3.))
            obstacles=[]
            for _ in range(rng.randrange(4)):
                w,h=rng.uniform(.25,1.1),rng.uniform(.2,.7);bx,by,ba=place(w,h)
                obstacles.append(dict(cx=bx,cy=by,half_w=w/2,half_h=h/2,angle=ba))
            for _ in range(10000):
                gx,gy=rng.uniform(.6,5.4),rng.uniform(.6,4.2);p=polygon(gx,gy,.25,1.,a)
                if math.hypot(gx-x,gy-y)>=1.8 and inside(p) and all(separated(p,q,.1) for q in occupied):break
            else:raise RuntimeError('goal rejection limit')
            robots=[dict(zip(('x','y','angle'),place(.36,.36,near=(x,y)))) for _ in range(2)]
            f=dict(scene_id='random_room_v1',room=dict(arena=[6.,4.8],obstacles=obstacles),object=target,robots=robots,goal=dict(x=gx,y=gy,radius=.2),movable_objects=boxes,overrides={})
            split='dev' if j<40 else 'validation' if j<45 else 'test'
            rows.append(dict(id=f'{split}_{count}_{j:02d}',split=split,movable_count=count,fixture=f))
    return dict(version=VERSION,seed=seed,protocol=dict(arena=[6.,4.8],movables_including_target=[2,5],static_count=[0,3],min_goal_distance=1.8,initial_clearance=.06,wall_clearance=.11,goal_empty_same_orientation=True,robot_sampling='uniform angle and radius .65..1.4 about target, rejection for geometry',goal_radius=.2,max_ticks=1800,max_decisions=300,control_dt=.05,mode='duration',solvability='not established; no outcome filtering',splits={'dev':160,'validation':20,'test':20}),cases=rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True);data=generate();errors=[]
    for r in data['cases']:
        try:X.fixture(r['fixture'])
        except Exception as e:errors.append(dict(id=r['id'],error=repr(e)))
    if errors:raise RuntimeError(errors)
    content=json.dumps(data,ensure_ascii=False,indent=2);path=OUT/'scenes.json'
    if path.exists() and path.read_text()!=content:raise RuntimeError('immutable dataset differs')
    path.write_text(content)
    (OUT/'generation.json').write_text(json.dumps(dict(sha256=hashlib.sha256(content.encode()).hexdigest(),count=len(data['cases']),splits=dict(Counter(r['split'] for r in data['cases'])),physics_geometry_checks=200,physics_ticks=0),indent=2))
    print((OUT/'generation.json').read_text())
if __name__=='__main__':main()
