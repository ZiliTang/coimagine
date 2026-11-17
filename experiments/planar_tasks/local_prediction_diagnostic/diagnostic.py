"""Bounded local counterfactual diagnosis; no policy edits or model synthesis."""
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import shutil
import sys
import time

from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.baseline.panels import load_panel
from experiments.planar_tasks.baseline.prediction import PredictionService, PredictionLedger, LIMITS
from experiments.planar_tasks.baseline.session import package_command, verify_frozen
from experiments.planar_tasks.interface import TaskEnvironment, validate_action
from experiments.planar_tasks.policy import JsonlPolicy
from experiments.planar_tasks.runner import replay_record, write_json

ROOT=seal.ROOT
SOURCE=ROOT/'outputs/agentic/prediction_sol_runtime_20261110_01'
PRIOR=ROOT/'outputs/agentic/local_prediction_diagnostic_20261110_01'
OUT=ROOT/'outputs/agentic/local_prediction_diagnostic_20261110_02'
HERE=Path(__file__).resolve().parent
HORIZON=120
BUDGET=dict(execution_ticks=50000, prediction_ticks=6000, wall_seconds=600, max_boundaries=6, candidates_per_boundary=4)


def save(p,value):
    write_json(p,value)


def load(p):return json.loads(p.read_text())
def xy(state):return [state['x'],state['y']]
def distance(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])


def hypothesis(observation,state):
    # Identical construction to the submitted Policy.predict, using legal history only.
    r=observation['robots'][0];points=state['map'];obstacles=[]
    if points:
        xs=[p[0] for p in points];ys=[p[1] for p in points]
        lo,hi,bot,top=min(xs),max(xs),min(ys),max(ys)
        obstacles=[[(lo+hi)/2,(bot+top)/2,max(.008,(hi-lo)/2),max(.008,(top-bot)/2),0]]
    return dict(arena=r['arena'],goal=r['goal_xy'],goal_radius=r['goal_radius'],
        object=state['pose']+[0,0,0],robots=[[r['own'][k] for k in ('x','y','angle','vx','vy','omega')]
        for r in observation['robots']],obstacles=obstacles)


def candidates(observation,original):
    legal=observation['legal_pairs'];base=[original['a0'],original['a1']]
    validate_action(original,observation)
    result=[dict(name='original_pair',pair=base)]
    active=next((i for i,a in enumerate(base) if a<32),0)
    direction=base[active]//4 if base[active]<32 else 0
    speed=base[active]%4 if base[active]<32 else 3
    for sign in (1,-1):
        a=((direction+2*sign)%8)*4+speed
        options=[]
        for i in (active,1-active):
            pair=[46,46];pair[i]=a;options.append(pair)
        options.append([a,a])
        chosen=next((p for p in options if p in legal and p not in [c['pair'] for c in result]),None)
        if chosen is not None:result.append(dict(name='transverse_'+str(sign),pair=chosen))
    # One deterministic contact-changing candidate. No truth or rollout ranking.
    choices=[([34,34],'rotate_ccw'),([37,37],'rotate_cw')]
    for i in (active,1-active):
        for a in (40,43):
            p=[46,46];p[i]=a;choices.append((p,'reposition_'+str(i)+'_'+str(a)))
    chosen=next(((p,n) for p,n in choices if p in legal and p not in [c['pair'] for c in result]),None)
    if chosen:result.append(dict(name=chosen[1],pair=chosen[0]))
    return result[:4]


def select_boundaries(records):
    passing=next((i for i,r in enumerate(records) if r['state']['mode']=='pass' and (i==0 or records[i-1]['state']['mode']!='pass')),None)
    selected=[]
    if passing is not None:selected.append((passing,'first_lateral_to_pass'))
    for i,r in enumerate(records):
        if passing is None or i<=passing:continue
        if r['state']['mode'] not in ('pass','lateral'):continue
        pair=[r['action']['a0'],r['action']['a1']]
        if not any(a<32 for a in pair):continue
        t=r['time'];old=next((q for q in records if q['time']>=t-5),None)
        if old and t-old['time']>=4 and distance(r['state']['pose'],old['state']['pose'])<.04:
            selected.append((i,'first_post_pass_estimated_stall'));break
    return selected


def prepare():
    seal.verify_release();verify_frozen(SOURCE/'session')
    OUT.mkdir(parents=True,exist_ok=False)
    package=OUT/'inspection_package';shutil.copytree(SOURCE/'session/frozen/package',package)
    # Instrument the protocol host only; author executable files remain identical.
    (package/'__policy_host__.py').write_text('''import sys,json
from policy import Policy
p=None
for line in sys.stdin:
 r=json.loads(line)
 if r.get('reset'):p=Policy(r['contract'])
 a=p.act(r['observation'])
 state={k:getattr(p,k) for k in ('pose','mode','way','map','last','history')}
 print(json.dumps(dict(action=a,state=state)),flush=True)
''')
    evaluation=load(SOURCE/'evaluation/summary.json')
    failed=[r for r in evaluation['results'] if r['family']=='detour' and not r['success']][:3]
    selected=[];inspection=[]
    for row in failed:
        episode=SOURCE/'evaluation'/row['episode_path']
        transcript=[json.loads(s) for s in (episode/'sensors.jsonl').read_text().splitlines()]
        pred=episode/'prediction.jsonl'
        replies=[json.loads(s) for s in pred.read_text().splitlines()] if pred.exists() else []
        cursor=[0]
        def handler(req,deadline):
            x=replies[cursor[0]];cursor[0]+=1
            assert req==x['request'] and x['error'] is None
            return x['result']
        process=JsonlPolicy(package_command(package),OUT/f'inspect_{row["index"]}.stderr',rpc_handler=handler)
        records=[]
        try:
            for i in range(0,len(transcript),2):
                req=transcript[i]['request'];actual=process.request(req,5)
                assert actual['action']==transcript[i+1]['response']
                actual['observation']=req['observation'];actual['time']=req['observation']['robots'][0]['elapsed_seconds']
                records.append(actual)
        finally:process.close()
        assert cursor[0]==len(replies)
        save(OUT/f'inspection_{row["index"]}.json',records)
        decisions=load(episode/'decisions.json')
        for index,reason in select_boundaries(records):
            r=records[index];obs=r['observation'];state=r['state']
            selected.append(dict(id=f'case{row["index"]}_d{index}',index=row['index'],decision=index,
                reason=reason,start_tick=decisions[index]['start_tick'],episode_path=row['episode_path'],
                observation=obs,estimated_state=state,hypothesis=hypothesis(obs,state),
                target=state['way'] or obs['robots'][0]['goal_xy'],original=r['action'],
                candidates=candidates(obs,r['action'])))
        inspection.append(dict(index=row['index'],responses_verified=len(records),prediction_rpcs_replayed=len(replies)))
    assert len(selected)<=6 and all(b['start_tick']+HORIZON<=1800 for b in selected)
    save(OUT/'boundaries.json',selected)
    paths=[seal.LOCK,SOURCE/'session/frozen/manifest.json',SOURCE/'evaluation/panel.json',OUT/'boundaries.json',Path(__file__)]
    for row in failed:
        episode=SOURCE/'evaluation'/row['episode_path']
        paths += [episode/n for n in ('sensors.jsonl','decisions.json','truth.json','prediction.jsonl') if (episode/n).exists()]
    save(OUT/'preflight.json',dict(created_unix=time.time(),budget=BUDGET,horizon_ticks=HORIZON,
        source_files={str(p.relative_to(ROOT)):seal.digest(p) for p in paths},inspection=inspection,
        boundary_count=len(selected),candidate_count=sum(len(b['candidates']) for b in selected),
        primary_critic='lexicographic: completed nominal horizon without error/early failure, then reduction in fixed estimated waypoint distance; tie original/order',
        horizon_semantics='hold chosen joint pair for 120 ticks, validate each 40; NOT original policy continuation',
        offline_control='true initial geometry/velocities, cold controller; never policy input',
        evaluation_role='previous test cases now diagnostic/development; no new independent success estimate'))
    # The first infrastructure attempt stopped before any candidate was run.
    # Preserve its costs; selection and scoring are byte-identical across attempts.
    assert seal.digest(OUT/'boundaries.json')==seal.digest(PRIOR/'boundaries.json')
    pre=load(OUT/'preflight.json');pre['prior_attempt']=str(PRIOR.relative_to(ROOT))
    pre['source_files'][str((PRIOR/'summary.json').relative_to(ROOT))]=seal.digest(PRIOR/'summary.json')
    pre['repair']='Compare the same logged kinematic heading representation, not raw rigid-body angle/omega'
    save(OUT/'preflight.json',pre)
    print(json.dumps(dict(boundaries=[{k:b[k] for k in ('id','reason','start_tick','target','candidates')} for b in selected],inspection=inspection),indent=2))


def state(world):
    env=world._env
    return dict(object=asdict(env.object_state()),robots=[asdict(s) for s in env.robot_states()])


def quality(start,end,target,observation,ticks,termination,error):
    eligible=error is None and (ticks==HORIZON or termination=='success')
    return dict(eligible=eligible,waypoint_progress=distance(xy(start['object']),target)-distance(xy(end['object']),target),
        object_displacement=distance(xy(start['object']),xy(end['object'])),
        heading_change=(end['object']['angle']-start['object']['angle']+math.pi)%(2*math.pi)-math.pi,
        legal_pair_count=len(observation['legal_pairs']),
        push_direction_count=len({a//4 for pair in observation['legal_pairs'] for a in pair if a<32}),
        completed_ticks=ticks,termination=termination,error=error)


def rank(rows,key):
    return sorted(range(len(rows)),key=lambda i:(-int(rows[i][key]['quality']['eligible']),-rows[i][key]['quality']['waypoint_progress'],i))


class Accountant:
    def __init__(self):
        self.started=time.monotonic();previous=load(PRIOR/'summary.json')
        self.usage=Counter(previous['usage']);self.prior_wall=previous['wall_seconds']
    def check(self):
        if time.monotonic()-self.started+self.prior_wall>BUDGET['wall_seconds']:raise TimeoutError('diagnostic wall budget')
    def tick(self,kind):
        self.check()
        if self.usage[kind]>=BUDGET[kind]:raise RuntimeError('diagnostic '+kind+' budget')
        self.usage[kind]+=1


def replay_prefix(task,decisions,stop,expected,account):
    world=TaskEnvironment(task.instance)
    for d in decisions[:stop]:
        world.observe()  # match the original request boundary and sensor caches
        for _ in range(d['end_tick']-d['start_tick']):
            account.tick('execution_ticks');world._env.joint_step(*d['actions'],record=True)
        world.decisions+=1;world._env.skills_used=world.decisions
    assert world.observe()==expected,'reconstructed prefix observation mismatch'
    return world


def true_hypothesis(world):
    instance=world._env.instance;s=state(world)
    return dict(arena=list(instance.arena),goal=list(instance.goal_centre),goal_radius=instance.goal_radius,
        object=list(s['object'].values()),robots=world._env.rows[-1]['true']['robots'],
        obstacles=[[b.cx,b.cy,b.half_w,b.half_h,b.angle] for b in instance.obstacles])


def actual_branch(world,pair,b,task,path,account):
    start=state(world);start_tick=world._env.total_periods;actions=[];error=None
    try:
        for _ in range(3):
            a=dict(a0=pair[0],a1=pair[1],ticks=40)
            validate_action(a,world.observe());s=world._env.total_periods
            for _ in range(40):
                if world._env.termination:break
                account.tick('execution_ticks');world._env.joint_step(*pair,record=True)
            world.decisions+=1;world._env.skills_used=world.decisions
            actions.append(dict(start_tick=s,end_tick=world._env.total_periods,requested_ticks=40,actions=pair))
            if world._env.termination:break
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
    env=world._env;end=state(world);obs=world.observe()
    result=dict(start=start,end=end,observation=obs,quality=quality(start,end,b['target'],obs,
        env.total_periods-start_tick,env.termination,error))
    save(path/'actual.json',result);save(path/'truth.json',env.rows)
    # Original prefix plus the alternative suffix, for inspection in the existing console.
    old=load(SOURCE/'evaluation'/b['episode_path']/'decisions.json')[:b['decision']]
    summary=dict(run_id=path.name,success=bool(env.success),termination=env.termination)
    save(path/'replay.json',replay_record(task,env,old+actions,summary))
    return result


def predicted_branch(h,pair,b,path,account):
    ledger=PredictionLedger(dict(LIMITS,cpu_seconds=10.,decision_cpu_seconds=2.))
    predictor=PredictionService(ledger=ledger,log_path=path)
    def call(op,**kw):
        account.check();predictor.begin_decision()
        return predictor.rpc(dict(method='predict',operation=op,**kw),account.started+BUDGET['wall_seconds'])
    initial=None;last=None;error=None
    try:
        if account.usage['prediction_ticks']+HORIZON>BUDGET['prediction_ticks']:
            raise RuntimeError('insufficient prediction tick reservation')
        initial=call('create',hypothesis=h);last=initial
        for _ in range(3):
            if last['termination']:break
            # Service charges actual ticks; reservation reconciled below even on errors.
            last=call('step',handle=last['handle'],action=dict(a0=pair[0],a1=pair[1],ticks=40))
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
    ticks=ledger.usage.get('ticks',0)
    account.usage['prediction_ticks']+=ticks
    if account.usage['prediction_ticks']>BUDGET['prediction_ticks']:raise RuntimeError('prediction diagnostic budget')
    if initial is None:
        return dict(quality=dict(eligible=False,waypoint_progress=0.,error=error),usage=ledger.usage)
    q=quality(initial['state'],last['state'],b['target'],last['observation'],ticks,last['termination'],error)
    return dict(start=initial['state'],end=last['state'],observation=last['observation'],quality=q,usage=ledger.usage,
                partial_state_valid=error is None)


def run():
    pre=load(OUT/'preflight.json')
    if (OUT/'run_claim.json').exists():raise FileExistsError('already attempted; no automatic rerun')
    for path,digest in pre['source_files'].items():assert seal.digest(ROOT/path)==digest,path
    seal.verify_release();save(OUT/'run_claim.json',dict(started_unix=time.time(),budget=BUDGET))
    _,tasks=load_panel(SOURCE/'evaluation/panel.json');account=Accountant();results=[]
    try:
        for b in load(OUT/'boundaries.json'):
            task=tasks[b['index']];decisions=load(SOURCE/'evaluation'/b['episode_path']/'decisions.json');rows=[]
            source_truth=load(SOURCE/'evaluation'/b['episode_path']/'truth.json')
            for c in b['candidates']:
                path=OUT/'branches'/b['id']/c['name'];path.mkdir(parents=True,exist_ok=False)
                world=replay_prefix(task,decisions,b['decision'],b['observation'],account)
                # Offline truth verifies warm replay only, never candidate selection.
                actual=state(world);old=source_truth[b['start_tick']-1]['true']
                assert world._env.rows[-1]['true']==old,'reconstructed prefix logged state mismatch'
                control=true_hypothesis(world)
                estimated=predicted_branch(b['hypothesis'],c['pair'],b,path/'estimated_prediction.jsonl',account)
                cold=predicted_branch(control,c['pair'],b,path/'true_cold_prediction.jsonl',account)
                actual_result=actual_branch(world,c['pair'],b,task,path,account)
                rows.append(dict(candidate=c,estimated=estimated,true_cold=cold,actual=actual_result))
                save(OUT/'progress.json',dict(boundary=b['id'],candidate=c['name'],usage=account.usage))
            rankings={k:rank(rows,k) for k in ('estimated','true_cold','actual')}
            results.append(dict(id=b['id'],index=b['index'],decision=b['decision'],reason=b['reason'],
                target=b['target'],start_tick=b['start_tick'],rows=rows,rankings=rankings))
            save(OUT/'partial_results.json',results)
        summary=dict(status='complete',budget=BUDGET,usage=account.usage,wall_seconds=time.monotonic()-account.started+account.prior_wall,results=results)
    except Exception as exc:
        summary=dict(status='incomplete',error=dict(type=type(exc).__name__,message=str(exc)),budget=BUDGET,
            usage=account.usage,wall_seconds=time.monotonic()-account.started,results=results)
    save(OUT/'summary.json',summary)
    save(OUT/'artifacts_sha256.json',{str(p.relative_to(OUT)):seal.digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifacts_sha256.json'})
    print(json.dumps(dict(status=summary['status'],error=summary.get('error'),usage=account.usage,wall_seconds=summary['wall_seconds']),indent=2))


if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
