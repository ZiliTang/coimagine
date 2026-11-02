import json,sys,collections
sys.path.insert(0,'experiments/option_rl_v19')
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin
from experiments.option_rl_v14.render_episodes import instance_from_dict

d=json.load(open('docs/agent_exchange/tasks/D009_v2_acceptance.json'))
tasks=[c for c in d['cases'] if c['group']=='task']
rows=[]; fam=collections.defaultdict(lambda:[0,0])
for c in tasks:
 env=OptionSystemEnv(instance_from_dict(c['instance'])); UnknownShapeEnvMixin.install_prior_free(env)
 for a0,a1 in c['commands']:
  if env.termination is not None: break
  env.joint_step(a0,a1)
 ok=bool(env.success); rows.append((c['id'],c['family'],ok,env.total_periods,round(env.object_distance(),3)))
 fam[c['family']][1]+=1; fam[c['family']][0]+=int(ok)
 print(c['id'],c['family'],ok,'periods',env.total_periods,'dist',round(env.object_distance(),3),flush=True)
print('TOTAL',sum(r[2] for r in rows),'/',len(rows))
for k,v in fam.items(): print(k,v[0],'/',v[1])
json.dump(dict(rows=rows,family={k:v for k,v in fam.items()}),open('outputs/agent_exchange/D009_v2_20261030_001/dev_task_eval.json','w'),indent=1)
