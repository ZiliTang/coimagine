"""D009 diagnostic: 20Hz trace of the two reposition roots around face handoff."""
import json, sys, math, hashlib
from pathlib import Path
sys.path.insert(0, 'experiments/option_rl_v19')
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin
from experiments.option_rl_v14.render_episodes import instance_from_dict

ROOT=Path('outputs/agent_exchange/D009_v2_20261030_002_diag')
ROOT.mkdir(parents=True,exist_ok=True)

def run(cid):
    d=json.load(open('docs/agent_exchange/tasks/D009_v2_acceptance.json'))
    c=next(x for x in d['cases'] if x['id']==cid)
    env=OptionSystemEnv(instance_from_dict(c['instance'])); UnknownShapeEnvMixin.install_prior_free(env)
    rows=[]; old=env.env.step_control
    def wrapped(forces, headings, **kwargs):
        out=old(forces, headings, **kwargs)
        s0,s1=env.slots[0].skill,env.slots[1].skill
        o=env.object_state(); rs=env.robot_states()
        def eff(side):
            e=(env.slots[side].last_info or {}).get('effect') or {}
            pts=e.get('face_points') or []
            cen=(sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts)) if pts else None
            sp=(max(p[0] for p in pts)-min(p[0] for p in pts),
                max(p[1] for p in pts)-min(p[1] for p in pts)) if pts else None
            return dict(n=e.get('face_normal'), contact_target=e.get('contact_target'),
                        face_centroid=cen, face_span=sp, desired=e.get('desired_velocity'),
                        v_est=e.get('v_est'), stage=(env.slots[side].skill.stage if env.slots[side].skill else None))
        rows.append(dict(tick=int(env.total_periods),time=round(env.total_periods*0.05,4),
            phase=env.coordinator.phase,
            clear=[bool(env.clearance.started),bool(env.clearance.finished),bool(env.clearance.failed)],
            contact=[bool(getattr(s0,'contact_ready',False)),bool(getattr(s1,'contact_ready',False))],
            force_pre=[[float(forces[0][0]),float(forces[0][1])],[float(forces[1][0]),float(forces[1][1])]],
            gate_pre=getattr(env,'gate_pre',None),gate_post=getattr(env,'gate_post',None),
            obj=[round(o.x,5),round(o.y,5),round(o.angle,5)],
            robots=[[round(r.x,5),round(r.y,5),round(r.angle,5)] for r in rs],
            side=[eff(0),eff(1)]))
        return out
    env.env.step_control=wrapped
    for a0,a1 in c['commands']:
        if env.termination is not None: break
        env.joint_step(a0,a1)
    # event summary: first contact, first angle > .02 after PUSH decision 72, first gate zero
    first_contact=next((r for r in rows if any(r['contact'])),None)
    first_angle=next((r for r in rows if r['time']>=14.4 and abs(r['obj'][2])>=.02),None)
    first_gate=next((r for r in rows if r['gate_post']==[[0.,0.],[0.,0.]]),None)
    summary=dict(id=cid,success=env.success,term=env.termination,dist=round(env.object_distance(),5),
        final_obj=[round(env.object_state().x,5),round(env.object_state().y,5),round(env.object_state().angle,5)],
        first_contact_tick=first_contact['tick'] if first_contact else None,
        first_contact=first_contact, first_angle2=first_angle, first_gate_zero=first_gate,
        rows=rows)
    (ROOT/f'{cid}.json').write_text(json.dumps(summary,indent=1))
    print(cid,'success',env.success,'contact',summary['first_contact_tick'],
          'angle2',first_angle['tick'] if first_angle else None,
          'gate0',first_gate['tick'] if first_gate else None,'final',summary['final_obj'])

for cid in ('T_reposition_0_p0','T_reposition_2_p0'):
    run(cid)
# freeze current source hashes alongside the trace
files=['experiments/option_rl_v19/unknown_shape.py','experiments/option_rl_v19/system.py',
       'experiments/option_rl_v13/mode_env.py','experiments/option_rl_v13/push_controller.py']
(ROOT/'source_hashes.json').write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=1))
