"""Small persistent-mode command schedules, explicit transitions and physical checks."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,math,time,hashlib,copy
from experiments.option_rl_v12 import controller as C
from experiments.option_rl_v12.mode_env import ContinuousModeEnv
from experiments.option_testbench.effect_push_eval import make_case
from experiments.option_testbench.follow_peer_eval import instance_from_dict
from experiments.option_testbench.peer_clearance_eval import frame
from experiments.option_testbench import mixed_contact_eval as M

def request(kind,*,v=None,typ=None,direction=None):return next(a.index for a in C.ACTIONS if a.kind==kind and (v is None or a.speed==v) and (typ is None or a.type_index==typ) and (direction is None or a.direction==direction))
def cases():
 single=make_case('single',offset=.18);coop=make_case('dual',cooperative=True)
 def c(name,base,seq):return dict(name=name,instance=copy.deepcopy(base['instance']),schedule=seq)
 close=copy.deepcopy(coop);close['instance']['robots'][0]['x']=2.31;close['instance']['robots'][1]['x']=2.69
 stop=[C.STOP,C.STOP];p=lambda v:request('push',v=v,typ=2);rp=request('reposition',v=.12,direction='CCW')
 rotation=make_case('rotation',rotation=1,angle=.1)
 return [c('close_peer_recovery',close,[(4.,[p(.16),p(.16)]),(6.,stop)]),c('rotation',rotation,[(6.,[request('push',typ=8,v=.04),C.STOP]),(8.,stop)]),c('single_slow',single,[(6.,[p(.08),C.STOP]),(8.,stop)]),c('single_fast',single,[(4.2,[p(.32),C.STOP]),(6.4,stop)]),c('speed_change',single,[(3.,[p(.08),C.STOP]),(5.4,[p(.32),C.STOP]),(7.4,stop)]),c('push_to_reposition',single,[(3.6,[p(.16),C.STOP]),(8.,[rp,C.STOP]),(10.,stop)]),c('reposition',single,[(8.,[rp,C.STOP]),(10.,stop)]),c('dual_push',coop,[(5.,[p(.16),p(.08)]),(7.,stop)]),c('push_follow',coop,[(5.,[p(.16),C.FOLLOW]),(7.,stop)]),c('push_reposition',coop,[(6.,[p(.16),rp]),(8.,stop)])]

def run(case,hz):
 w=ContinuousModeEnv(instance_from_dict(case['instance']),decision_hz=hz,collect_dense=True);initial=frame(w);boundaries=[];start=time.monotonic()
 for until,actions in case['schedule']:
  while w.total_periods*.05<until-1e-8 and w.termination is None:boundaries.append(w.joint_step(*actions,record=True))
 rows=w.rows;first=initial['true']['object'];last=rows[-1]['true']['object'];gaps=[math.dist(row['true']['robots'][0][:2],row['true']['robots'][1][:2])-.36 for row in rows]
 refs=[[row[f'r{i}']['reference_velocity'] for row in rows] for i in (0,1)]
 metric=dict(push_outcomes=[],case=case['name'],variant=f'modes_{hz}hz',seconds=w.total_periods*.05,decisions=len(boundaries),wall_seconds=time.monotonic()-start,exits=[w.slots[i].skill.exit_reason or 'holding' if w.slots[i].action.kind=='wait' else w.slots[i].skill.exit_reason or 'continuing' for i in (0,1)],max_penetration=max(row['penetration'] for row in rows),min_peer_gap=min(gaps),safety_triggers=sorted({str(row[f'r{i}']['safety_trigger']) for row in rows for i in (0,1) if row[f'r{i}']['safety_trigger']}),object_delta=[last[k]-first[k] for k in (0,1)],object_rotation_rad=last[2]-first[2],termination=w.termination,dense_samples=len(w.dense_samples),max_reference_step=max(math.dist(a,b) for ref in refs for a,b in zip([(0,0)]+ref,ref)),dense_reward_total=w.reward_total)
 return dict(case=case,variant=metric['variant'],instance=case['instance'],initial=initial,rows=rows,metric=metric,manifest_actions=[asdict(a) for a in C.ACTIONS],boundaries=boundaries)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);(a.out/'logs').mkdir();start=time.monotonic();results=[];panel=cases()
 (a.out/'manifest.json').write_text(json.dumps({'version':C.VERSION,'cases':panel,'cadences':[5,10],'physics':'same run11 120Hz physics / 20Hz control; no mass/friction/reward changes','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('controller.py'),Path(__file__).with_name('mode_env.py'),Path(__file__).with_name('push_controller.py')]}},indent=2))
 for case in panel:
  for hz in [5,10]:
   assert time.monotonic()-start<55
   r=run(case,hz);(a.out/'logs'/f"{case['name']}__{hz}.json").write_text(json.dumps(r));results.append(r)
 M.render(a.out,template_path=Path(__file__).with_name('testbench_template.html'))
 eq=[]
 for name in [c['name'] for c in panel]:
  x,y=[r for r in results if r['case']['name']==name]
  equal=len(x['rows'])==len(y['rows']) and all(a['true']==b['true'] and a['forces']==b['forces'] for a,b in zip(x['rows'],y['rows']));eq.append({'case':name,'same_physical_work_and_forces':equal})
 (a.out/'cadence_equivalence.json').write_text(json.dumps(eq,indent=2));print(json.dumps({'seconds':time.monotonic()-start,'results':[r['metric'] for r in results],'cadence':eq}))
if __name__=='__main__':main()
