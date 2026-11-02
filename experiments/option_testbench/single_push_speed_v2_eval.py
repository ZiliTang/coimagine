"""Single-scene staged speed ceiling/approach diagnosis; production defaults untouched."""
import sys,json,time,math,hashlib,argparse,types
from pathlib import Path
from dataclasses import replace,asdict
from functools import partial
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from coimagine.random_start import efficient_profile as EP,env as E
from coimagine.random_start.effect_push import make_effect_skill,clamp
from experiments.option_testbench.effect_push_eval import make_case,ACTIONS as OLD
from experiments.option_testbench import mixed_contact_eval as M,follow_peer_eval as F
B=EP.CONFIG
CONFIGS={'baseline':B}
for speed in (.24,.32,.40,.50):
 ratio=speed/.24
 CONFIGS[f'push_{speed:.2f}']=replace(B,speed=speed,tangential_gain=3.6*ratio,tangential_speed=min(.5,.16*ratio),point_rate=.12*ratio,contact_tangent_damping=.7)
CONFIGS['gain_only']=replace(CONFIGS['push_0.40'],approach_tangential_speed=.5,approach_tangential_gain=6.6)
for speed in (.3,.4,.5):
 CONFIGS[f'approach_{speed:.2f}']=replace(CONFIGS['push_0.40'],approach_normal_speed=speed,approach_tangential_speed=speed,approach_tangential_gain=6.,approach_velocity_damping=1.,approach_brake_accel=.35,gap_gain=5.)
CONFIGS['light_object']=CONFIGS['push_0.40']
F.VARIANTS.update(CONFIGS)

def skill(action,limit,config,variant):
 s=make_effect_skill(action,limit,config=config)
 if action.kind!='push' or getattr(action,'effect_kind',None)!='translate':return s
 diag={'approach_calls':0,'near_face_low_cap_calls':0,'p_outside_cap_calls':0,'reference_rate_limited_calls':0,'peak_reference':0.,'peak_actual_speed':0.}
 s.probe_diag=diag
 oldlat=s._approach_lateral
 def lateral(self,error,velocity,gap,contacting):
  diag['approach_calls']+=1
  if gap<.04 or contacting:diag['near_face_low_cap_calls']+=1
  if variant.startswith('approach_'):
   cap=min(config.approach_tangential_speed,math.sqrt(2*config.approach_brake_accel*abs(error)))
   raw=config.approach_tangential_gain*error-config.approach_velocity_damping*velocity
   self.approach_speed_limit=cap
   return clamp(raw,-cap,cap)
  raw=oldlat(error,velocity,gap,contacting)
  if abs(raw)>=self.approach_speed_limit-1e-9:diag['p_outside_cap_calls']+=1
  return raw
 s._approach_lateral=types.MethodType(lateral,s)
 oldupdate=s.reference.update
 def update(raw,dt):
  if variant!='baseline' and s.stage=='push' and s.face is not None and not s.joint_pair:
   n=s.face.normal;radial=sum(raw[i]*n[i] for i in (0,1));remaining=max(0.,action.distance-s.normal_travel)
   cap=max(.025,math.sqrt(2*.25*remaining))
   if radial>cap:raw=tuple(raw[i]-(radial-cap)*n[i] for i in (0,1))
  length=math.hypot(*raw)
  if length>.5:raw=tuple(v*.5/length for v in raw)
  reference=oldupdate(raw,dt)
  if math.dist(reference,raw)>1e-8:diag['reference_rate_limited_calls']+=1
  diag['peak_reference']=max(diag['peak_reference'],math.hypot(*reference))
  return reference
 s.reference.update=update
 return s

WORLDS={}
def factory(instance,variant):
 ec=E.EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend='numpy_batch')
 if variant=='light_object':ec.base=replace(ec.base,physics=replace(ec.base.physics,object_mass=1.5))
 w=E.RandomStartEnv(instance,ec,skill_factory=partial(skill,config=CONFIGS[variant],variant=variant),action_table=EP.ACTIONS,action_version=EP.VERSION+'-solo-speed-v2')
 created=[];orig=w.skill_factory
 def sf(*args,**kw):
  s=orig(*args,**kw);created.append(s);return s
 w.skill_factory=sf;WORLDS[variant]=(w,created)
 return w

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
 c=make_case('single_N_0.3_offset_0.18',distance=.3,offset=.18);c['actions']=[next(a.index for a in EP.ACTIONS if a.label==OLD[i].label) for i in c['actions']]
 gate=dict(max_side_m=.02,max_turn_deg=2.,max_robot_travel_m=.31,min_robot_travel_m=.3,max_penetration_m=.002)
 paths=[Path(__file__),ROOT/'src/coimagine/random_start/effect_push.py',ROOT/'src/coimagine/random_start/skills.py',ROOT/'src/coimagine/continuous2d/backend.py',Path(__file__).with_name('single_push_speed_v2_refine.py')]
 (out/'manifest.json').write_text(json.dumps(dict(case=c,configs={k:asdict(v) for k,v in CONFIGS.items()},gate=gate,only_physics_ablation={'light_object':{'object_mass':1.5,'linear_damping':8.,'friction':.6}},normal_cap='max(.025,sqrt(2*.25*remaining_own_normal_travel))',total_reference_speed_cap=.5,production_changed=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
 results=[];start=time.monotonic()
 for name in CONFIGS:
  assert time.monotonic()-start<55
  r=M.run(c,name,world_factory=factory,actions=EP.ACTIONS);r['manifest_actions']=[asdict(a) for a in EP.ACTIONS];m=r['metric'];m['stage_seconds']={}
  for row in r['rows']:
   stage=row['r0']['stage'];m['stage_seconds'][stage]=m['stage_seconds'].get(stage,0)+.05
  m['diagnostic']=next(s.probe_diag for s in WORLDS[name][1] if hasattr(s,'probe_diag'))
  m['peak_actual_speed']=max(math.hypot(*row['true']['robots'][0][3:5]) for row in r['rows'])
  brakes=[row for row in r['rows'] if row['r0']['stage']=='brake'];m['diagnostic']['peak_actual_speed']=m['peak_actual_speed']
  m['object_motion_after_brake_m']=math.dist(brakes[0]['true']['object'][:2],r['rows'][-1]['true']['object'][:2]) if brakes else None
  travel=m['push_outcomes'][0]['normal_travel'];m['local_gate']=m['exits'][0]=='normal_travel_complete' and .3<=travel<=.31 and abs(m['object_delta'][0])<=.02 and abs(math.degrees(m['object_rotation_rad']))<=2 and m['max_penetration']<=.002 and not m['safety_triggers']
  (out/'logs'/f'{name}.json').write_text(json.dumps(r));results.append(m)
 M.render(out,template_path=ROOT/'experiments/option_testbench/option_efficiency_v2_template.html');(out/'speed_results.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(dict(seconds=time.monotonic()-start,results=[{k:m[k] for k in ('variant','seconds','stage_seconds','object_delta','object_rotation_rad','max_penetration','local_gate','diagnostic','object_motion_after_brake_m')} for m in results])))
if __name__=='__main__':main()
