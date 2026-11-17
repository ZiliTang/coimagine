"""Host-privileged engineering witnesses, NOT an observation-only CAP benchmark.

Uses true geometry for navigation/phase checks, but every command is validated by
and executed through the unchanged public option menu. Successful command traces
can be replayed at the testbed's exact 4-tick boundaries. No learned policy input,
training label cache, semantic authoring session or held-out split is changed.
"""
import argparse,hashlib,heapq,json,math,time
from pathlib import Path
from experiments.planar_tasks.relational_search import transfer_tasks as T
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv
from experiments.planar_tasks.relational_search.motion import move_id,DIRECTIONS


def dist_box(p,b):
    x,y,w,h,a=b;c,s=math.cos(a),math.sin(a);dx,dy=p[0]-x,p[1]-y
    return math.hypot(max(0,abs(c*dx+s*dy)-w/2),max(0,abs(-s*dx+c*dy)-h/2))


def pathfind(start,goal,audit,scene,boxes):
    rects=[(b.cx,b.cy,2*b.half_w,2*b.half_h,b.angle) for b in scene.obstacles]
    state=audit['entities']['entity-a'];rects.append((*state[:2],.25,1.,state[2]))
    for b in boxes:
        state=audit['entities'][b.label];rects.append((*state[:2],b.width,b.height,state[2]))
    peer=audit['robots'][0];radius=.25
    def free(p):return radius<=p[0]<=scene.arena[0]-radius and radius<=p[1]<=scene.arena[1]-radius and math.hypot(p[0]-peer[0],p[1]-peer[1])>=radius+.18 and all(dist_box(p,b)>=radius for b in rects)
    def segment(a,b):
        n=max(1,math.ceil(math.dist(a,b)/.04))
        return all(free((a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n)) for i in range(1,n+1))
    step=.12;key=lambda p:(round(p[0]/step),round(p[1]/step));point=lambda k:(k[0]*step,k[1]*step)
    first=key(start);queue=[(math.dist(start,goal),0.,first)];cost={first:0.};parent={};visited=set();end=None
    while queue:
        _,g,k=heapq.heappop(queue)
        if k in visited:continue
        visited.add(k);p=start if k==first else point(k)
        if math.dist(p,goal)<.25 and segment(p,goal):end=k;break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            j=(k[0]+dx,k[1]+dy);q=point(j)
            if j in visited or not segment(p,q):continue
            new=g+math.dist(p,q)
            if new<cost.get(j,float('inf')):
                cost[j]=new;parent[j]=k;heapq.heappush(queue,(new+math.dist(q,goal),new,j))
    if end is None:return []
    path=[goal];k=end
    while k!=first:path.append(point(k));k=parent[k]
    return list(reversed(path))


def move_towards(p,q):
    dx,dy=q[0]-p[0],q[1]-p[1];distance=math.hypot(dx,dy)
    if distance<.065:return 44
    direction=int(round(math.atan2(dy,dx)/(math.pi/4)))%8
    return move_id(DIRECTIONS[direction],.3 if distance>.3 else .1)


def run(case,out,*,max_ticks=1800,wall_seconds=60.,clear_margin=.15,clear_direction=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    scene,boxes=X.fixture(case['fixture']);env=MultiObjectOptionEnv(scene,boxes)
    f=case['fixture'];origin=(f['object']['x'],f['object']['y']);theta=f['object']['angle'];u=(math.cos(theta),math.sin(theta));v=(-u[1],u[0]);direction=round(theta/(math.pi/4))%8
    blocked=case['host_annotation']['straight_corridor_overlap'];phase=('preclear' if clear_direction is not None else 'clear') if blocked else 'relocate'
    clear_dir=(direction+2)%8 if clear_direction is None else DIRECTIONS.index(clear_direction)
    width=boxes[0].width;height=boxes[0].height;angle=boxes[0].angle
    threshold=.5+.5*(width*abs(math.sin(angle-theta))+height*abs(math.cos(angle-theta)))+clear_margin
    commands=[];trace=[];path=[];error=None;started=time.monotonic();invalid=0
    while env.termination is None and env.total_periods<max_ticks:
        if time.monotonic()-started>wall_seconds:error='engineering wall budget';break
        o=env.public();a=env.offline_audit();obj=a['entities']['entity-a'];other=a['entities']['entity-b'];r=a['robots'][1];t=o['tick']
        lateral=(other[0]-origin[0])*v[0]+(other[1]-origin[1])*v[1]
        if phase=='clear' and lateral>=threshold:phase='retreat'
        if phase=='retreat' and dist_box(r[:2],(*other[:2],width,height,other[2]))>.34:phase='relocate'
        goal=(obj[0]-.55*u[0]+.32*v[0],obj[1]-.55*u[1]+.32*v[1])
        if phase=='preclear':
            ca=clear_dir*math.pi/4;goal=(other[0]-.9*math.cos(ca),other[1]-.9*math.sin(ca))
            if math.dist(r[:2],goal)<.11:phase='clear';path=[]
        if phase=='relocate' and math.dist(r[:2],goal)<.11:phase='transport'
        target=None;a0=a1=44
        if phase=='clear':target='entity-b';a1=4*clear_dir+2
        elif phase=='retreat':a1=move_id(DIRECTIONS[(clear_dir+4)%8],.3)
        elif phase in ('relocate','preclear'):
            if not path or len(commands)%8==0:path=pathfind(r[:2],goal,a,scene,boxes)
            while path and math.dist(r[:2],path[0])<.08:path.pop(0)
            if not path:
                if math.dist(r[:2],goal)>.15:error='no host navigation path';break
            else:a1=move_towards(r[:2],path[0])
        elif phase=='transport':
            dx,dy=scene.goal_centre[0]-obj[0],scene.goal_centre[1]-obj[1]
            if math.hypot(dx,dy)<.14:phase='brake'
            else:
                target='entity-a';a0=a1=4*direction+2
                desired=math.atan2(dy,dx);err=math.atan2(math.sin(desired-obj[2]),math.cos(desired-obj[2]))
                if err>.035:a0=4*direction+3;a1=4*direction+1
                elif err<-.035:a0=4*direction+1;a1=4*direction+3
        menu=o['global_legal_pairs'] if target is None else o['entity_legal_pairs'].get(target,[])
        if [a0,a1] not in menu:
            invalid+=1
            if target is not None and [a0,44] in menu and a0!=44:a1=44
            elif target is not None and [44,a1] in menu and a1!=44:a0=44
            else:target=None;a0=a1=44
        command=dict(target_id=target,a0=a0,a1=a1,ticks=4)
        trace.append(dict(tick=t,phase=phase,audit=a));commands.append(command)
        try:env.step_command(command)
        except Exception as exc:error=repr(exc);break
    final=env.offline_audit();segments=[]
    for command in commands:
        action={k:command[k] for k in ['target_id','a0','a1']}
        if segments and all(segments[-1][k]==x for k,x in action.items()) and segments[-1]['decisions']<200:segments[-1]['decisions']+=1
        else:segments.append(dict(**action,decisions=1))
    result=dict(case_id=case['id'],fixture_sha256=case['fixture_sha256'],success=bool(env.success),termination=env.termination,ticks=env.total_periods,wall_seconds=time.monotonic()-started,error=error,phase=phase,menu_fallbacks=invalid,final=final,commands=commands,sequence=segments,trace=trace,scope='host-privileged engineering witness, not learned CAP',config=dict(max_ticks=max_ticks,clear_margin=clear_margin,clear_direction=clear_direction),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['case_id','success','termination','ticks','wall_seconds','error','phase','menu_fallbacks']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case',choices=[c['id'] for c in T.cases()]);p.add_argument('--clear-direction',choices=DIRECTIONS);args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    chosen=[c for c in T.cases() if args.case is None or c['id']==args.case]
    (root/'protocol.json').write_text(json.dumps(dict(cases=[c['id'] for c in chosen],max_attempts=len(chosen),max_ticks_per_attempt=1800,wall_seconds_per_attempt=60,clear_direction=args.clear_direction,model_calls=0,training_updates=0,scope='privileged host acceptance; no research performance comparison'))+'\n')
    for c in chosen:run(c,root/c['id'],clear_direction=args.clear_direction)
