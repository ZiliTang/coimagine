"""Bounded single-case PUSH speed and braking prototype; no production changes."""
import sys,json,time,hashlib,math
from pathlib import Path
from functools import partial
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from coimagine.random_start import efficient_profile as EP,env as E
from coimagine.random_start.effect_push import make_effect_skill
from experiments.option_testbench.effect_push_eval import make_case,ACTIONS as OLD
from experiments.option_testbench import mixed_contact_eval as M,follow_peer_eval as F
B=EP.CONFIG
CONFIGS={'baseline':B,'push_2x':replace(B,speed=.24),'matched_2x':replace(B,speed=.24,tangential_speed=.16,tangential_gain=3.6,point_rate=.12,contact_tangent_damping=.7),'matched_2x_fast_prepare':replace(B,speed=.24,tangential_speed=.16,tangential_gain=3.6,point_rate=.12,contact_tangent_damping=.7,approach_tangential_speed=.4,approach_tangential_gain=6.6,approach_velocity_damping=.8,approach_brake_accel=.7,approach_normal_speed=.22,gap_gain=6.)}
CONFIGS.update({'matched_decelerate':CONFIGS['matched_2x'],'matched_decelerate_fast_prepare':replace(CONFIGS['matched_2x'],approach_tangential_speed=.4,approach_tangential_gain=4.5,approach_velocity_damping=1.2,approach_brake_accel=.7,approach_normal_speed=.22)})
F.VARIANTS.update(CONFIGS)
def prototype_skill(action,limit,config,decelerate=False):
 skill=make_effect_skill(action,limit,config=config)
 if decelerate and action.kind=='push' and getattr(action,'effect_kind',None)=='translate':
  original=skill.reference.update
  def update(raw,dt):
   if skill.stage=='push' and skill.face is not None and not skill.joint_pair:
    n=skill.face.normal;radial=sum(raw[i]*n[i] for i in (0,1))
    remaining=max(0.,action.distance-skill.normal_travel)
    cap=max(.025,math.sqrt(2*.25*remaining))
    if radial>cap:raw=tuple(raw[i]-(radial-cap)*n[i] for i in (0,1))
   return original(raw,dt)
  skill.reference.update=update
 return skill
def factory(instance,variant):
 return E.RandomStartEnv(instance,E.EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend='numpy_batch'),skill_factory=partial(prototype_skill,config=CONFIGS[variant],decelerate=variant.startswith('matched_decelerate')),action_table=EP.ACTIONS,action_version=EP.VERSION+'-single-push-probe')
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='One single-robot north PUSH, six speed/braking comparisons; production defaults unchanged.')
 parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
 case=make_case('single_N_0.3_offset_0.18',distance=.3,offset=.18)
 case['actions']=[next(a.index for a in EP.ACTIONS if a.label==OLD[i].label) for i in case['actions']]
 (out/'manifest.json').write_text(json.dumps(dict(case=case,configs={k:asdict(c) for k,c in CONFIGS.items()},scope='one single-robot north push, 0.3m, 0.18m initial tangential offset; peer waits remotely; no RL or default changes',deceleration={'variants':['matched_decelerate','matched_decelerate_fast_prepare'],'law':'cap normal reference by max(0.025,sqrt(2*0.25*remaining_own_normal_travel)); solo translation only'},experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/coimagine/random_start/effect_push.py','src/coimagine/random_start/skills.py','src/coimagine/random_start/efficient_profile.py']}),indent=2))
 start=time.monotonic();results=[]
 for name in CONFIGS:
  assert time.monotonic()-start<55
  r=M.run(case,name,world_factory=factory,actions=EP.ACTIONS);r['manifest_actions']=[asdict(a) for a in EP.ACTIONS]
  r['variant']=name;m=r['metric'];stage={}
  for row in r['rows']:stage[row['r0']['stage']]=stage.get(row['r0']['stage'],0)+.05
  m['stage_seconds']=stage;m['robot_exit_time']=next((row['time']-r['initial']['time'] for row in r['rows'] if row['r0'].get('exit_reason')),None)
  (out/'logs'/f'{name}.json').write_text(json.dumps(r));results.append(m)
 M.render(out,template_path=ROOT/'experiments/option_testbench/option_efficiency_v2_template.html')
 (out/'speed_results.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(dict(seconds=time.monotonic()-start,results=results)))
