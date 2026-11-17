"""Development-only geometry drafts, not a frozen evaluation or solved benchmark.

Each layout pairs an obstacle intersecting the straight transport corridor with
that same obstacle moved sideways. Geometry annotations are host-only; deployment
still receives MultiObjectOptionEnv.public(). No teacher, fit or rollout is run.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import math
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search import testbed as T
from experiments.planar_tasks.relational_search.contract import digest


def make_layouts():
    # Different heading, location, distance and obstacle aspect/relative angle.
    specs=[('east',(1.8,2.0),0.,3.1,1.2,.6,.7,0.,1.5),
           ('north',(2.3,1.4),math.pi/2,2.7,1.2,.9,.4,math.pi/2,1.5),
           ('west',(4.4,2.5),math.pi,3.2,1.3,.5,.95,math.pi,1.2),
           ('diagonal',(1.8,1.5),math.pi/4,math.sqrt(7.22),math.sqrt(2),.9,.45,-.2,1.5)]
    cases=[]
    for name,(x,y),theta,length,along,width,height,angle,side in specs:
        u=(math.cos(theta),math.sin(theta));v=(-u[1],u[0])
        def point(forward,lateral):return (x+forward*u[0]+lateral*v[0],y+forward*u[1]+lateral*v[1])
        gx,gy=point(length,0);r0=point(-.45,-.25);r1=point(1.,-.8)
        for blocked in (True,False):
            lateral=0. if blocked else side;bx,by=point(along,lateral)
            f=dict(object=dict(x=x,y=y,angle=theta),robots=[dict(x=r0[0],y=r0[1],angle=theta),dict(x=r1[0],y=r1[1],angle=theta+math.pi/2)],goal=dict(x=gx,y=gy,radius=.2),movable_objects=[dict(label='entity-b',x=bx,y=by,width=width,height=height,angle=angle,mass=3.)],overrides={})
            # Primary rectangle's half-width across goal direction is .5m.
            obstacle_half_width=.5*(width*abs(math.sin(angle-theta))+height*abs(math.cos(angle-theta)))
            margin=abs(lateral)-.5-obstacle_half_width
            assert (margin<0)==blocked
            cases.append(dict(id=f'{name}_{"blocked" if blocked else "clear"}',split='development',family='relation_transfer_clear_transport',layout=name,fixture=f,fixture_sha256=digest(f),host_annotation=dict(straight_corridor_overlap=blocked,lateral_separation_margin_m=margin,scope='offline design annotation only; not a policy input or proof that clearing is necessary',solvability='unverified',recommended_action=None)))
    return cases


def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cases=make_layouts();checks=[]
    for c in cases:
        scene,boxes=X.fixture(c['fixture']);env=T.validate_geometry(scene,boxes)
        public=env.public()
        goal_scene=replace(scene,object_pose=replace(scene.object_pose,x=scene.goal_centre[0],y=scene.goal_centre[1]))
        T.validate_geometry(goal_scene,boxes)
        checks.append(dict(id=c['id'],geometry_valid=True,goal_pose_geometry_valid=True,visible_entity_ids=[e['id'] for e in public['entities']],initial_tick=public['tick']))
    payload=dict(version='relational-transfer-layout-drafts-v1',cases=cases,checks=checks,execution_ticks=0,training_updates=0,model_calls=0,selection='explicit development designs; no success-based filtering',limitations=['Geometry validity is not dynamic solvability.','Direct corridor overlap does not establish that clearing is optimal or necessary.','No new semantics, teachers, or learned policies have been run.','Not held out: all layouts are visible development drafts.','Arena and static obstacles remain those of the current public fixture adapter.'])
    (out/'layouts.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    render(cases,out/'layouts.png')
    print(json.dumps(dict(output=str(out),cases=len(cases),geometry_valid=len(checks),initial_visibility=checks,execution_ticks=0),ensure_ascii=False))


def render(cases,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Circle
    fig,axs=plt.subplots(2,4,figsize=(16,7.5))
    def rectangle(ax,x,y,w,h,angle,color):
        ca,sa=math.cos(angle),math.sin(angle)
        pts=[(x+ca*a-sa*b,y+sa*a+ca*b) for a,b in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]]
        ax.add_patch(Polygon(pts,facecolor=color,edgecolor='#333333',alpha=.85))
    for i,c in enumerate(cases):
        ax=axs[i%2,i//2];f=c['fixture'];scene,boxes=X.fixture(f)
        for b in scene.obstacles:rectangle(ax,b.cx,b.cy,2*b.half_w,2*b.half_h,b.angle,'#777777')
        o=f['object'];rectangle(ax,o['x'],o['y'],.25,1.,o['angle'],'#2f9bce')
        for b in f['movable_objects']:rectangle(ax,b['x'],b['y'],b['width'],b['height'],b['angle'],'#e4aa45')
        for j,r in enumerate(f['robots']):
            ax.add_patch(Circle((r['x'],r['y']),.18,color=['#54985e','#9758b0'][j]));ax.text(r['x'],r['y'],str(j),ha='center',va='center',color='white',fontsize=8)
        g=f['goal'];ax.add_patch(Circle((g['x'],g['y']),g['radius'],fill=False,color='#c03662',lw=2))
        ax.plot([o['x'],g['x']],[o['y'],g['y']],linestyle='--',color='#c03662',alpha=.65)
        ax.set(xlim=(0,6),ylim=(0,4.8),aspect='equal',title=c['id']);ax.grid(alpha=.15)
    fig.suptitle('Development geometry drafts: same relation across layouts, paired relation change\nBlue: transport target | Gold: movable obstacle | Circle: goal | Solvability not yet verified')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();build(a.output)
