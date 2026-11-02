import json,sys,hashlib,datetime
from pathlib import Path
sys.path.insert(0,'experiments/option_rl_v19')
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin
from experiments.option_rl_v14.render_episodes import instance_from_dict
ROOT=Path('outputs/agent_exchange/D010_20261030_continuation'); ROOT.mkdir(exist_ok=True)
d=json.load(open('docs/agent_exchange/tasks/D009_v2_acceptance.json'))
out=[]
for cid in ('T_reposition_0_p0','T_reposition_2_p0','T_dual_0_p0','T_dual_2_p0'):
 c=next(x for x in d['cases'] if x['id']==cid); env=OptionSystemEnv(instance_from_dict(c['instance'])); UnknownShapeEnvMixin.install_prior_free(env)
 rows=[]; old=env.env.step_control
 def wrapped(forces, headings, **kwargs):
  outv=old(forces, headings, **kwargs)
  i=len(rows); s0=env.slots[0].skill; s1=env.slots[1].skill
  rows.append(dict(tick=i,time=round(i*.05,3),phase=getattr(env.coordinator,'phase',None),
    action=[env.slots[0].action.index,env.slots[0].action.label,list(env.slots[0].action.effect_direction) if hasattr(env.slots[0].action,'effect_direction') else None],
    stage=[getattr(s0,'stage',None),getattr(s1,'stage',None)],
    contact_phase=[getattr(s0,'contact_phase',None),getattr(s1,'contact_phase',None)],
    contact=[bool(getattr(s0,'contact_ready',False)),bool(getattr(s1,'contact_ready',False))],
    gap=[getattr(s0,'gap',None),getattr(s1,'gap',None)],
    retry=[getattr(s0,'retry_count',None),getattr(s1,'retry_count',None)],
    force=[[float(forces[0][0]),float(forces[0][1])],[float(forces[1][0]),float(forces[1][1])]],
    gate_pre=getattr(env,'gate_pre',None),gate_post=getattr(env,'gate_post',None),
    obj=[round(env.object_state().x,5),round(env.object_state().y,5),round(env.object_state().angle,5)]))
  return outv
 env.env.step_control=wrapped
 for a0,a1 in c['commands']:
  if env.termination is not None: break
  env.joint_step(a0,a1)
 out.append(dict(id=cid,success=env.success,term=env.termination,periods=env.total_periods,
   dist=round(env.object_distance(),4),rows=rows))
 print(cid,'success',env.success,'term',env.termination,'periods',env.total_periods)
(ROOT/'contact_probe.json').write_text(json.dumps(dict(created=datetime.datetime.now().isoformat(),
 source_hash=hashlib.sha256(Path('experiments/option_rl_v19/unknown_shape.py').read_bytes()).hexdigest(),
 cases=out),indent=1))
