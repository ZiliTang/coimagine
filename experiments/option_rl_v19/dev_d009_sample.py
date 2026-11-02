import json,sys
from pathlib import Path
sys.path.insert(0,'experiments/option_rl_v19')
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin
from experiments.option_rl_v14.render_episodes import instance_from_dict

d=json.load(open('docs/agent_exchange/tasks/D009_v2_acceptance.json'))
cases=d['cases']
def run(c):
 env=OptionSystemEnv(instance_from_dict(c['instance'])); UnknownShapeEnvMixin.install_prior_free(env)
 for a0,a1 in c['commands']:
  if env.termination is not None: break
  env.joint_step(a0,a1)
 o=env.object_state()
 return dict(id=c['id'],group=c['group'],family=c['family'],success=bool(env.success),term=env.termination,
             periods=env.total_periods,dist=round(env.object_distance(),4),obj=(round(o.x,3),round(o.y,3),round(o.angle,3)))
# non-final dev sample: 2 task families x p0/p4, 2 contract families x r0/r1
sel=[]
for c in cases:
 if c['id'] in ('T_contact_2_p0','T_contact_2_p4','T_dual_2_p0','T_dual_2_p4','T_reposition_2_p0','T_reposition_2_p4','T_approach_2_p0','T_approach_2_p4'):
  sel.append(c)
for c in cases:
 if c['group']=='contract' and c['family'] in ('adjacent','same_face') and c['perturbation'] in (0,4):
  sel.append(c)
for c in sel:
 print(json.dumps(run(c),ensure_ascii=False),flush=True)
