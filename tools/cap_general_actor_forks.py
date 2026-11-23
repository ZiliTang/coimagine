"""Matched actor forks, fastest-teacher repairs and one-tick sensitivity on the frozen held panel."""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch

from tools import cap_general_actor_fit_closed as base
from tools.cap_general_actor_fit_closed import campaign, dq, fit, scale, sets, read, save
from experiments.planar_tasks.relational_search.motion import ACTIONS

OUT=base.OUT/'forks'
MODELS=[]
SIDS=[]


def choose(model,panel):
    x,c,s,valid,pool,_=panel
    scores=model.values(*(torch.from_numpy(a[None]) for a in (x,c,s)))[0].numpy()
    assert np.isfinite(scores[valid]).all()
    return pool[int(np.argmax(np.where(valid,scores,-np.inf)))]


def position(world):
    env=world.env
    return list(env.env.backend.bodies[env.primary_id].position),[list(q[:2]) for q in env._legal_robot_states()]


@torch.inference_mode()
def rollout(fixture,actor,*,teacher=None,cut=None,override=None,ref=None,script=False):
    env=dq.fast_runtime.ScopedEnvironment(fixture,dq.study.CFG,fast_feedback=True,record_details=False,
                                         history_factory=dq.fast_runtime.AuditedCachedHistory)
    lib=dq.Proposals()
    steps=[]
    try:
        for i in range(300):
            panel=lib.panel(env.history)
            choices=[choose(m,panel) for m in MODELS] if teacher is not None else []
            h=env.history.sha256
            if teacher is not None:
                action=teacher[i]['a']
                assert action in panel[4]
            elif i==cut:
                assert h==ref[i]['h']
                action=override
            elif script and i>cut:
                if i>=len(ref):
                    return dict(status='end_of_recorded_suffix',success=None,steps=steps)
                action=ref[i]['a']
                try:
                    dq.validate_action(action,dq.current_event(env.history)['observation'])
                except ValueError:
                    return dict(status='unavailable_command',success=None,steps=steps)
            else:
                action=choose(MODELS[actor],panel)
                if cut is not None and i<cut:
                    assert (h,action)==(ref[i]['h'],ref[i]['a'])
            ev=env.step(action,remaining_ticks=1800-env.env.total_periods)
            xy,robots=position(env)
            steps.append(dict(a=action,h=h,t=ev['start_tick'],dt=ev['periods'],xy=xy,robots=robots,
                              reason=ev['reason'],choices=choices,multi=len(panel[4])>1))
            if teacher is not None:
                assert teacher[i]['tick']==ev['end_tick']
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        assert env.env.termination in ('success','task_budget_exhausted') or (env.env.termination is None and (i==299 or env.env.total_periods==1800))
        if cut is not None:
            assert i>=cut
        return dict(status='complete',success=int(env.env.success),hash=env.history.sha256,steps=steps)
    finally:
        env.close_reads()
        dq.fast_runtime.ACTIVE=None


def first_diff(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x['a']!=y['a']),None)


def case(job):
    room,fixture,prior=job
    normal=[rollout(fixture,i) for i in range(2)]
    for i,(key,hkey) in enumerate([('actor800','actor800_history'),('layernorm','ln_history')]):
        assert (normal[i]['success'],normal[i]['hash'])==(int(prior[key]),prior[hkey])
    paths=[a['steps'] for a in normal]
    cut=first_diff(*paths)
    row=dict(room=room,actual=[a['success'] for a in normal],cut=cut)
    runs={s:read(campaign.OUT/'runs'/s/(room+'.json')) for s in SIDS}
    winners=sorted((r['ticks'],s) for s,r in runs.items() if r['success'])
    traces={s:sets.unzip(campaign.OUT/'runs'/s/(room+'.json.gz')) for _,s in winners}
    if winners:
        sid=winners[0][1]
        teacher=traces[sid]
        tour=rollout(fixture,None,teacher=teacher)
        assert tour['success'] and tour['hash']==runs[sid]['history']
        t=tour['steps']
        matches=[]
        for arm in range(2):
            wrong=next((i for i,v in enumerate(t) if v['choices'][arm]!=v['a']),None)
            hits=sum(v['multi'] and v['choices'][arm]==v['a'] for v in t)
            item=dict(hits=hits,first_error=wrong)
            if wrong is not None:
                assert paths[arm][wrong]['h']==t[wrong]['h'] and paths[arm][wrong]['a']!=t[wrong]['a']
                repair=rollout(fixture,arm,cut=wrong,override=t[wrong]['a'],ref=paths[arm])
                item['repair']=repair['success']
            else:
                assert normal[arm]['hash']==tour['hash']
            matches.append(item)
        row['teacher']=dict(sid=sid,multi=sum(v['multi'] for v in t),actors=matches)
    if cut is None:
        assert normal[0]['hash']==normal[1]['hash']
        return row
    fork=[p[cut] for p in paths]
    assert fork[0]['h']==fork[1]['h'] and fork[0]['t']==fork[1]['t']
    a,b=[r['a'] for r in fork]
    row.update(prefix=fork[0]['h'],tick=fork[0]['t'],actions=[a,b],
               kinds=[[ACTIONS[v['a0']].kind,ACTIONS[v['a1']].kind] for v in (a,b)],
               ticks=[r['dt'] for r in fork])
    # A recorded success supplies a label here only if its entire preceding
    # action sequence and elapsed time agree with the common prefix.
    active={s:tr[cut]['a'] for s,tr in traces.items() if len(tr)>cut and
            all(u['a']==v['a'] and u['tick']==v['t']+v['dt'] for u,v in zip(tr[:cut],paths[0][:cut]))}
    row['supported']=[[s for s,x in active.items() if x==v] for v in (a,b)]
    row['teacher_at_cut']=next((j for j,v in enumerate((a,b)) if winners and active.get(winners[0][1])==v),
                               'third' if winners and winners[0][1] in active else 'off_prefix' if winners else 'no_teacher')
    row['cross']=[]
    row['micro']=[]
    for arm in range(2):
        other=1-arm
        # Switch the single command, then resume the named suffix actor.
        mixed=rollout(fixture,arm,cut=cut,override=fork[other]['a'],ref=paths[arm])
        assert mixed['steps'][cut+0]['xy']==fork[other]['xy'] and mixed['steps'][cut]['robots']==fork[other]['robots']
        if cut+1<len(mixed['steps']):
            assert mixed['steps'][cut+1]['h']==paths[other][cut+1]['h']
        row['cross'].append(mixed['success'])
        original=fork[arm]['a']
        small=dict(original,ticks=original['ticks']-1 if original['ticks']>1 else 2)
        pert=rollout(fixture,arm,cut=cut,override=small,ref=paths[arm])
        before,after=fork[arm],pert['steps'][cut]
        delta=int(after['dt']-before['dt'])
        item=dict(success=pert['success'],requested_delta=small['ticks']-original['ticks'],dt_delta=delta,
                  xy_delta=float(np.linalg.norm(np.subtract(before['xy'],after['xy']))),
                  robot_delta=float(np.linalg.norm(np.subtract(before['robots'],after['robots']),axis=1).max()),
                  before_reason=before['reason'],after_reason=after['reason'])
        item['next_diff']=first_diff(paths[arm][cut+1:],pert['steps'][cut+1:])
        if pert['success']!=normal[arm]['success']:
            fixed=rollout(fixture,arm,cut=cut,override=small,ref=paths[arm],script=True)
            item.update(fixed_status=fixed['status'],fixed_success=fixed['success'])
        row['micro'].append(item)
    return row


def run():
    global MODELS,SIDS
    torch.set_num_threads(1)
    _,rooms,SIDS=scale.inputs()
    settings=read(base.OUT/'protocol.json')
    assert settings['panel']==campaign.base.digest(rooms) and SIDS==settings['sids']
    assert campaign.sha(scale.OUT/'actor.pt')==read(scale.OUT/'FINISHED.json')['checkpoint']
    MODELS=[sets.checkpoint(scale.OUT),fit.load_best()]
    dq.MODULES=[dq.load_native((campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in SIDS]
    with (base.OUT/'per_scene.csv').open() as f:
        prior={r['room']:r for r in csv.DictReader(f)}
    assert prior.keys()==rooms.keys() and len(rooms)==200
    OUT.mkdir(exist_ok=True)
    directory=OUT/'rooms'
    directory.mkdir(exist_ok=True)
    protocol=dict(panel=settings['panel'],old=campaign.sha(scale.OUT/'actor.pt'),layernorm=settings['model'],
                  source=campaign.sha(__file__),baseline=campaign.sha(base.OUT/'per_scene.csv'),sids=SIDS)
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==protocol
    save(OUT/'protocol.json',protocol)
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        jobs=[pool.submit(case,(k,f,prior[k])) for k,f in rooms.items() if not (directory/(k+'.json')).exists()]
        for n,future in enumerate(as_completed(jobs),1):
            row=future.result()
            save(directory/(row['room']+'.json'),row)
            save(OUT/'progress.json',dict(completed=len(rooms)-len(jobs)+n,total=len(rooms)))
    assert {p.stem for p in directory.glob('*.json')}==rooms.keys()


if __name__=='__main__':
    run()
