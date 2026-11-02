"""Replay the exact effect62-fast greedy policy on a checkpoint's frozen validation panel."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from experiments.option_rl_v11.train import run_episode,load_checkpoint
from experiments.option_rl_v11.scenes import ARENA
from experiments.option_rl_v45.render_episodes import TEMPLATE
from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import layout,spec


def instance_from_dict(d):
    if d['obstacles']:raise ValueError('this renderer covers the frozen empty-arena training distribution')
    return layout.InstanceSpec(name=d['name'],map=d['map'],split=d['split'],index=d['index'],seed=d['seed'],
        arena=tuple(d['arena']),obstacles=(),object_pose=Pose(**d['object']),
        robot_poses=tuple(Pose(**p) for p in d['robots']),goal_centre=tuple(d['goal_centre']),
        goal_radius=d['goal_radius'],env_version=d['env_version'])


def render_episodes(policy,normalizer,instances,out,*,count=4):
    from coimagine.random_start import geometry as G
    import math
    instances=sorted(instances,key=lambda s:math.dist((s.object_pose.x,s.object_pose.y),s.goal_centre))
    if not instances:raise ValueError('fixed validation panel is empty')
    chosen=[instances[i] for i in np.linspace(0,len(instances)-1,min(count,len(instances))).round().astype(int)]
    episodes=[]
    for instance in chosen:
        _,summary=run_episode(policy,instance,normalizer,deterministic=True,record=True)
        frames=[dict(t=0.,o=[instance.object_pose.x,instance.object_pose.y,instance.object_pose.angle],
                     r=[[r.x,r.y,r.angle] for r in instance.robot_poses],ct=[False,False],st=['initial','initial'],fo=[0.,0.])]
        for r in summary.pop('rows'):
            frames.append(dict(t=r['time'],o=r['true']['object'][:3],r=[p[:3] for p in r['true']['robots']],
                ct=[r[f'r{i}']['contacting_object'] for i in (0,1)],st=[r[f'r{i}']['stage'] for i in (0,1)],
                fo=[r[f'r{i}']['force_norm'] for i in (0,1)]))
        # Show the option at its START, not only once its execution has ended.
        decisions=[];start=0.
        for d in summary['actions']:
            decisions.append(dict(t=start,a0=d['labels'][0],a1=d['labels'][1]));start=d['time']
        episodes.append(dict(name=instance.name,goal=instance.goal_centre,radius=instance.goal_radius,
            object_size=[spec.OBJECT_WIDTH,spec.OBJECT_HEIGHT],start_goal_distance=round(summary['initial_distance'],3),
            frames=frames,decisions=decisions,success=summary['success'],termination=summary['termination'],summary=summary))
    payload=dict(checkpoint='effect62-fast | frozen validation | conditional greedy joint probability',arena=ARENA,episodes=episodes)
    html=TEMPLATE.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
    html=html.replace('ARENA[1]','DATA.arena[1]').replace('ARENA[0]','DATA.arena[0]')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'episodes.html').write_text(html)
    (out/'rendered_episodes.json').write_text(json.dumps(payload,indent=1)+'\n')
    return out/'episodes.html'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=4);args=p.parse_args()
    policy,stats,data=load_checkpoint(args.checkpoint)
    instances=[instance_from_dict(d) for d in data['manifest']['validation_instances']]
    print(render_episodes(policy,stats,instances,args.checkpoint.parent,count=args.episodes))
if __name__=='__main__':main()
