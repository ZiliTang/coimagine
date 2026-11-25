import cython
from math import sqrt, sin, cos, pi, atan2
SEMANTIC_API = 'typed-stream-v1'
USE_PUSH_REVIEW = True
NAMES = ['preferred_target', 'preferred_robot_action_fraction', 'duration_match']
STATE_NAMES = ['goal_distance', 'target_seen', 'active_contacts', 'stalled_seconds', 'progress_m']
HISTORY_LENGTH = 8
DIR = tuple((cos(i*pi/4), sin(i*pi/4)) for i in range(8))
def dot(a,b):
  return a[0]*b[0]+a[1]*b[1]
def sub(a,b):
  return a[0]-b[0],a[1]-b[1]
def shift(a,d,s):
  return a[0]+s*d[0],a[1]+s*d[1]
def mag(a):
  return sqrt(dot(a,a))
def norm(a):
  d=max(1e-9,mag(a))
  return a[0]/d,a[1]/d
def cross(a,b):
  return a[0]*b[1]-a[1]*b[0]
def clamp(v,lo,hi):
  return min(hi,max(lo,v))
def position(r):
  return r['own']['x'],r['own']['y']
def visible_faces(r):
  result=[]
  for p in r['patches']:
    pts=p['points_world']
    if not p['valid'] or len(pts)<3:
      continue
    n=p['normal_world']
    t=(-n[1],n[0])
    v=[dot(q,t) for q in pts]
    result.append((tuple(n),t,dot(pts[0],n),min(v),max(v)))
  return result
def compact_center(points, faces, previous):
  if not points:
    return previous
  origin = (sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points))
  shifted = [(p[0]-origin[0], p[1]-origin[1]) for p in points]
  planes = [(n,t,d-dot(n,origin),lo-dot(t,origin),hi-dot(t,origin)) for n,t,d,lo,hi in faces]
  last = None if previous is None else [previous[0]-origin[0],previous[1]-origin[1],*previous[2:]]
  center = fit_relative(shifted,planes,last)
  return [center[0]+origin[0],center[1]+origin[1],*center[2:]]
def fit_relative(points,faces,previous):
  if not points:
    return previous
  if faces:
    f=max(faces,key=lambda f:f[4]-f[3])
    a=f[0]
  elif previous:
    a=tuple(previous[2:4])
  else:
    extreme=max(((mag(sub(p,q)),p,q) for p in points for q in points),default=None)
    a=norm(sub(extreme[1],extreme[2])) if extreme[0]>.06 else (1.,0.)
  b=(-a[1],a[0])
  pa=[dot(p,a) for p in points]
  pb=[dot(p,b) for p in points]
  sa=max(pa)-min(pa)
  sb=max(pb)-min(pb)
  if max(sa,sb)>.43 and abs(sa-sb)>.15:
    thick=sa<sb
  elif previous and abs(dot(a,previous[2:4]))>.85:
    thick=previous[4]<previous[5]
  elif previous and abs(cross(a,previous[2:4]))>.85:
    thick=previous[4]>previous[5]
  else:
    thick=bool(faces)
  widths=(max(.24,sa),max(1.,sb)) if thick else (max(1.,sa),max(.24,sb))
  values=[]
  for axis,projections,w in zip((a,b),(pa,pb),widths):
    lo,hi=min(projections),max(projections)
    pos=[f for f in faces if dot(f[0],axis)>.985]
    neg=[f for f in faces if dot(f[0],axis)<-.985]
    if pos and neg:
      mid=(lo+hi)/2
    elif pos and not neg:
      mid=sum(f[2] for f in pos)/len(pos)+w/2
    elif neg and not pos:
      mid=-sum(f[2] for f in neg)/len(neg)-w/2
    else:
      mid=(lo+hi)/2
      if previous and hi-lo+.055<w:
        fit=clamp(dot(previous,axis),hi-w/2,lo+w/2)
        mid=.25*mid+.75*fit
    values.append(mid)
  c=shift(shift((0.,0.),a,values[0]),b,values[1])
  return [c[0],c[1],a[0],a[1],widths[0],widths[1]]
def get_points(o,tid):
  for e in o['entities']:
    if e['id']==tid:
      return e['points']
  return []
def parent_initial_memory() -> dict:
  return {'c':None,'last':None,'age':0,'stall':0,'avoid':{},'dist':None,'progress':0.,
      'tmark':None,'tslow':0,'pmark':None,'pslow':[0,0],'recent':[[],[]],'focus':0,'last_target':None,'compact':None,'route':None,'route_tick':0,'static_cells':[],'stale_since':None,'stale_ticks':0,'route_scale':0.}
def geometry(observation,previous):
  tid=observation['goal']['target_id']
  points=get_points(observation,tid)
  fs=[visible_faces(r) if observation['active_target']==tid else [] for r in observation['robots']]
  c=compact_center(points,fs[0]+fs[1],previous)
  return c,fs,points
def signature(action):
  return [a//4 if a<32 else a for a in (action['a0'],action['a1'])]
def parent_update_memory(memory: dict, event: dict) -> dict:
  o=event['observation']
  dt=event['executed_ticks'] or 0
  c,fs,pts=geometry(o,memory['c'])
  memory['c']=c
  memory['compact']=c
  if c:
    marker=memory['stale_since']
    if marker is None or mag(sub(c,marker))>.20:
      memory['stale_since']=c[:2]
      memory['stale_ticks']=0
    else:
      memory['stale_ticks']+=dt
  track_route(o,memory,c)
  action=event.get('requested_action') or event.get('action')
  if action is None or not dt:
    return memory
  sig=signature(action)
  same=memory['last']==sig
  memory['age']=(memory['age'] if same else 0)+dt
  memory['last']=sig
  if c:
    d=mag(sub(c,o['goal']['xy']))
    old=memory['dist']
    memory['progress']=.8*memory['progress']+.2*(0. if old is None else old-d)
    memory['dist']=d
    mark=memory['tmark']
    if mark is None or mag(sub(c,mark))>.16:
      memory['tmark']=c[:2]
      memory['tslow']=0
    else:
      memory['tslow']+=dt
  poses=[position(r) for r in o['robots']]
  if memory['pmark'] is None:
    memory['pmark']=poses
  for i,r in enumerate(o['robots']):
    if mag(sub(poses[i],memory['pmark'][i]))>.21:
      memory['pmark'][i]=poses[i]
      memory['pslow'][i]=0
    else:
      memory['pslow'][i]+=dt
    if min(sig)>=44 and (not memory['recent'][i] or mag(sub(poses[i],memory['recent'][i][-1]))>.16):
      memory['recent'][i]=(memory['recent'][i]+[poses[i]])[-13:]
  failed=any(a<32 and r['execution'].get('done') for a,r in zip((action['a0'],action['a1']),o['robots']))
  tick=o['tick']
  if min(sig)<8 and (failed or memory['age']>170 and memory['tslow']>130 or memory['tslow']>220):
    for i,d in enumerate(sig):
      if d<8:
        for delta in (-1,0,1):
          key='%d,%d'%(i,(d+delta)%8)
          memory['avoid'][key]=tick+(145 if delta==0 else 85)
    memory['age']=0
    memory['tslow']=0
    memory['stall']+=1
    if failed:
      memory['focus']=1-memory['focus']
  elif min(sig)<8 and memory['progress']>.06:
    memory['stall']=max(0,memory['stall']-1)
  memory['avoid']={k:v for k,v in memory['avoid'].items() if v>tick}
  return memory
def hits(r):
  scan=r['scan']
  return [(d*cos(a),d*sin(a),k) for a,d,k,v in zip(scan['ray_angles'],scan['distances'],scan['category'],scan['valid']) if v and d is not None]
def clearance(pos,d,arena,returns):
  gap=3.
  for i in (0,1):
    if abs(d[i])>.00001:
      edge=arena[i]-.23 if d[i]>0 else .23
      gap=min(gap,(edge-pos[i])/d[i])
  for x,y,k in returns:
    forward=x*d[0]+y*d[1]
    side=abs(x*d[1]-y*d[0])
    radius=.25 if k!=3 else .27
    if forward>0 and side<radius:
      gap=min(gap,forward-sqrt(max(0.,radius*radius-side*side)))
  return gap
def move(r,desired,previous=None,recent=(),urgent=False):
  p=position(r)
  v=sub(desired,p)
  dist=mag(v)
  if dist<.13:
    return 44
  heading=norm(v)
  returns=hits(r)
  selected=44
  best=-100.
  for i,d in enumerate(DIR):
    gap=clearance(p,d,r['arena'],returns)
    if gap<.09:
      continue
    ali=dot(d,heading)
    score=ali*min(.55,gap)+.085*min(1.1,gap)
    if previous is not None and previous>=47 and (previous-47)//4==i:
      score+=.045
    if urgent:
      nextp=shift(p,d,min(.41,gap))
      visits=sum(max(0.,.27-mag(sub(nextp,q))) for q in recent[:-1])
      score-=.85*min(.65,visits)
      score+=.19*min(.6,gap)
    if score>best:
      speed=3 if dist>.40 and gap>.44 else 2 if dist>.27 and gap>.30 else 1
      selected=47+4*i+speed
      best=score
  return selected
def route_goal(center,goal,o):
  v=sub(goal,center)
  d=norm(v)
  dist=mag(v)
  if dist<.45:
    return d
  av=(-d[1],d[0])
  risk=[]
  for e in o['entities']:
    if e['id']==o['goal']['target_id'] or not e['points']:
      continue
    p=e['points']
    for q in p:
      qv=sub(q,center)
      u=dot(qv,d)
      s=dot(qv,av)
      if .32<u<min(dist,1.45) and abs(s)<.56:
        risk.append((u,s))
  if len(risk)<2:
    return d
  side=sum(s for _,s in risk)/len(risk)
  return norm(shift(d,av,-.70 if side>0 else .70))
def obstacle_points(o,tid,static_only=False):
  arena=o['robots'][0]['arena']
  points=[]
  for r in o['robots']:
    scan=r['scan']
    origin=position(r)
    pre=None
    for angle,dist,kind,entity,valid in zip(scan['ray_angles'],scan['distances'],scan['category'],scan['entity_ids'],scan['valid']):
      if not valid or not dist or kind not in (1,2) or kind==1 and (static_only or entity==tid):
        pre=None
        continue
      p=shift(origin,(cos(angle),sin(angle)),dist)
      if kind==2 and min(p[0],p[1],arena[0]-p[0],arena[1]-p[1])<.06:
        pre=None
        continue
      points.append(p)
      if pre and kind==pre[0] and entity==pre[1] and mag(sub(p,pre[2]))<.60:
        for frac in (.25,.50,.75):
          points.append(shift(p,sub(pre[2],p),frac))
      pre=kind,entity,p
  return points
def room_for(c,d,arena):
  a=c[2:4]
  b=(-a[1],a[0])
  gap=3.
  for i in (0,1):
    if abs(d[i])<.005:
      continue
    half=.5*(abs(a[i])*c[4]+abs(b[i])*c[5])
    wall=arena[i]-half-.045 if d[i]>0 else half+.045
    gap=min(gap,(wall-c[i])/d[i])
  return gap
def safe_travel(c,d,pts,arena):
  a=c[2:4]
  b=(-a[1],a[0])
  dx,dy=dot(d,a),dot(d,b)
  half=(c[4]*.5+.10,c[5]*.5+.10)
  gap=room_for(c,d,arena)
  for p in pts:
    q=sub(p,c)
    x,y=dot(q,a),dot(q,b)
    begin=-5.
    end=5.
    for v,w,h in ((x,dx,half[0]),(y,dy,half[1])):
      if abs(w)<.001:
        if abs(v)>h:
          end=-10.
      else:
        first=(v-h)/w
        last=(v+h)/w
        begin=max(begin,min(first,last))
        end=min(end,max(first,last))
    if end>max(begin,.06) and begin<gap:
      if begin>.02:
        gap=begin
      elif dot(q,d)>.08:
        gap=min(gap,.13)
  return gap
def heappush(items,value):
  items.append(value)
  items.sort(reverse=True)
def heappop(items):
  return items.pop()
STEP=.20
def grid_key(p,nx,ny):
  return int(clamp(round(p[0]/STEP),1,nx-2)),int(clamp(round(p[1]/STEP),1,ny-2))
def target_path(start,goal,obstacles,arena):
  nx,ny=int(arena[0]/STEP)+1,int(arena[1]/STEP)+1
  sk=grid_key(start,nx,ny)
  gk=grid_key(goal,nx,ny)
  costs={sk:0.}
  prev={}
  visited=set()
  occupied=set()
  near=set()
  ax=start[2:4]
  by=(-ax[1],ax[0])
  margins=(start[4]*.5+.085,start[5]*.5+.085)
  span=tuple(abs(ax[i])*margins[0]+abs(by[i])*margins[1] for i in (0,1))
  bounds=tuple(int(span[i]/STEP)+2 for i in (0,1))
  for a,b in obstacles:
    if mag(sub((a,b),start))<.20:
      continue
    x,y=grid_key((a,b),nx,ny)
    for i in range(max(1,x-bounds[0]),min(nx-1,x+bounds[0]+1)):
      for j in range(max(1,y-bounds[1]),min(ny-1,y+bounds[1]+1)):
        off=(i*STEP-a,j*STEP-b)
        d0,d1=abs(dot(off,ax)),abs(dot(off,by))
        if d0<margins[0] and d1<margins[1]:
          occupied.add((i,j))
        elif d0<margins[0]+.17 and d1<margins[1]+.17:
          near.add((i,j))
  occupied.discard(gk)
  pending=[(0.,0.,sk)]
  best=sk
  long_axis=ax if start[4]>start[5] else by
  while pending and len(visited)<625:
    _,g,p=heappop(pending)
    if p in visited:
      continue
    visited.add(p)
    if mag(sub(p,gk))<mag(sub(best,gk)):
      best=p
    if p==gk:
      break
    for dx,dy,w in ((1,0,1.),(1,1,1.414),(0,1,1.),(-1,1,1.414),(-1,0,1.),(-1,-1,1.414),(0,-1,1.),(1,-1,1.414)):
      q=p[0]+dx,p[1]+dy
      if q in visited or q in occupied or q[0]<1 or q[0]>=nx-1 or q[1]<1 or q[1]>=ny-1:
        continue
      edge=min(q[0]*STEP-span[0], arena[0]-q[0]*STEP-span[0],
           q[1]*STEP-span[1], arena[1]-q[1]*STEP-span[1])
      if edge<-.06 and edge<min(p[0]*STEP-span[0],arena[0]-p[0]*STEP-span[0],
                     p[1]*STEP-span[1],arena[1]-p[1]*STEP-span[1])-.015:
        continue
      if dx and dy and (p[0]+dx,p[1]) in occupied and (p[0],p[1]+dy) in occupied:
        continue
      cost=g+w+(.19 if q in near else 0)+.16*abs(dot((dx,dy),long_axis))+.27*max(0.,.22-edge)
      if cost<costs.get(q,1000.):
        costs[q]=cost
        prev[q]=p
        heappush(pending,(cost+1.08*mag(sub(q,gk)),cost,q))
  if best==sk:
    return None
  path=[]
  while best!=sk:
    path.append(best)
    best=prev[best]
  dst=path[-1]
  for p in path[-2::-1]:
    if mag(sub(sk,p))>5.4:
      break
    if any((round(sk[0]+(p[0]-sk[0])*f),round(sk[1]+(p[1]-sk[1])*f)) in occupied for f in (.19,.32,.47,.63,.80,.91)):
      break
    dst=p
  return dst[0]*STEP,dst[1]*STEP
def track_route(o,memory,c):
  if not c:
    return
  tick=o['tick']
  target=o['goal']['target_id']
  goal=o['goal']['xy']
  arena=o['robots'][0]['arena']
  if tick%5==0:
    fixed=obstacle_points(o,target,True)
    cells=memory['static_cells']
    for p in fixed:
      k=round(p[0]*10)*64+round(p[1]*10)
      if k not in cells:
        cells.append(k)
    memory['static_cells']=cells[-220:]
  if mag(sub(c,goal))<.56:
    memory['route']=None
    return
  dst=memory['route']
  if dst and tick-memory['route_tick']<110 and mag(sub(c,dst))>.28:
    return
  memory['route']=None
  if tick-memory['route_tick']<42 and dst is None:
    return
  points=obstacle_points(o,target)
  points+= [(k//64*.10,k%64*.10) for k in memory['static_cells']]
  direct=norm(sub(goal,c))
  gap=safe_travel(c,direct,points,arena)
  memory['route_tick']=tick
  if gap>min(1.26,mag(sub(c,goal))+.12):
    return
  result=target_path(c,goal,points,arena)
  if result and mag(sub(c,result))>.32:
    memory['route']=result
    memory['route_scale']=max(0.,min(1.,.95-gap))
def action_values(o,act):
  index=next((i for i,e in enumerate(o['entities']) if e['id']==act['target_id']),-1)
  return [index,act['a0'],act['a1'],act['ticks']]
def make(target,a,b,ticks):
  return {'target_id':target,'a0':a,'a1':b,'ticks':ticks}
def face_for(faces,d):
  choices=[f for f in faces if dot(f[0],d)>.818]
  return max(choices,key=lambda f:dot(f[0],d)) if choices else None
def face_q(f):
  return shift(shift((0.,0.),f[0],f[2]),f[1],(f[3]+f[4])/2)
def segment_near(point,start,end):
  dr=sub(end,start)
  amount=clamp(dot(sub(point,start),dr)/max(.0001,dot(dr,dr)),0.,1.)
  return mag(sub(point,shift(start,dr,amount)))
def joint_points(f0,f1):
  same=dot(f0[0],f1[0])>.97 and abs(f0[2]-f1[2])<.04
  q0,q1=face_q(f0),face_q(f1)
  if not same:
    return (q0,q1) if mag(sub(q0,q1))>.43 else None
  n,t=f0[0],f0[1]
  lo=max(f0[3],f1[3])+.04
  hi=min(f0[4],f1[4])-.04
  if hi-lo<.42:
    return None
  mid=(lo+hi)*.5
  sep=min(hi-lo,max(.42,(hi-lo)/3))
  foot=shift((0.,0.),n,(f0[2]+f1[2])*.5)
  lower=shift(foot,t,mid-sep/2)
  upper=shift(foot,t,mid+sep/2)
  return lower,upper
def approach_cost(position,q,n,peer,arena):
  final=shift(q,n,-.185)
  stage=shift(q,n,-.33)
  arena_edge=min(stage[0],stage[1],arena[0]-stage[0],arena[1]-stage[1])
  travel=mag(sub(position,final))
  if abs(cross(sub(position,q),n))>.06:
    travel=.74*travel+.26*mag(sub(position,stage))+.07
  if dot(sub(position,q),n)>.035:
    travel+=.55
  travel+=.60*max(0.,.23-arena_edge)
  travel+=.28*max(0.,.4-segment_near(peer,position,stage))/.4
  return travel
def push_plan(memory,o,c,faces,heading,pairs):
  robots=o['robots']
  poses=[position(r) for r in robots]
  fs=[[face_for(fl,d) for d in DIR] for fl in faces]
  old=memory['last']
  contacts=[r['contacting_object'] for r in robots]
  arena=robots[0]['arena']
  compact=memory['compact'] or c
  active_route=memory['route'] and memory['stale_ticks']>70
  obstacles=obstacle_points(o,o['goal']['target_id']) if active_route else []
  risk={}
  best=None
  best_score=-100.
  for a,b in pairs:
    if a not in (44,46) and (a>=32 or a%4!=3) or b not in (44,46) and (b>=32 or b%4!=3):
      continue
    f0=fs[0][a//4] if a<32 else None
    f1=fs[1][b//4] if b<32 else None
    if a<32 and f0 is None or b<32 and f1 is None or f0 is None and f1 is None:
      continue
    sig=signature({'a0':a,'a1':b})
    if any(d<8 and '%d,%d'%(i,d) in memory['avoid'] for i,d in enumerate(sig)):
      continue
    joint=f0 is not None and f1 is not None
    gain=0.
    if joint:
      q=joint_points(f0,f1)
      if not q or dot(f0[0],f1[0])<-.10:
        continue
      costs=[approach_cost(poses[i],q[i],f[0],poses[1-i],arena) for i,f in enumerate((f0,f1))]
      n=norm(shift(f0[0],f1[0],1.))
      torque=sum(cross(sub(q[i],c),f[0]) for i,f in enumerate((f0,f1)))
      if all(contacts) and dot(f0[0],f1[0])<.88:
        u,v=differential(a,b,faces,heading,robots,pairs)
        w0,w1=u%4+1,v%4+1
        blended=shift(shift((0.,0.),f0[0],w0),f1[0],w1)
        n=norm(shift(n,norm(blended),1.30))
        torque=2*(w0*cross(sub(q[0],c),f0[0])+w1*cross(sub(q[1],c),f1[0]))/(w0+w1)
      gain=.27-.27*abs(costs[0]-costs[1])-.13*min(1.,abs(torque))
      if min(costs)<.24 and max(costs)>.70:
        gain-=.16
      if contacts[0] and contacts[1]:
        gain+=.18
    else:
      i=0 if f0 is not None else 1
      f=f0 if i==0 else f1
      n=f[0]
      q=face_q(f)
      costs=[approach_cost(poses[i],q,n,poses[1-i],arena)]
      peer_cmd=b if i==0 else a
      if segment_near(poses[1-i],poses[i],shift(q,n,-.20))<.38 or segment_near(poses[1-i],q,shift(q,n,.35))<.42:
        gain+=.055 if peer_cmd==46 else -.16
      elif peer_cmd==46:
        gain-=.04
      if contacts[i]:
        gain+=.12
    ali=dot(n,heading)
    if ali<.16:
      continue
    score=1.65*ali+.25*max(0.,ali)**3+gain-.47*max(costs)
    if active_route:
      if n not in risk:
        risk[n]=safe_travel(compact,n,obstacles,arena)
      gap=risk[n]
      score-=clamp((.38-gap)/.38,0.,1.)*(.27+.38*memory['route_scale'])
      if gap>.43 and ali>.36 and memory['stale_ticks']>150:
        score+=.12
    if sig==old:
      score+=.29 if memory['age']<120 else .13
      score+=.10*sum(contacts)
    support=[(f[4]-f[3]) for f in (f0,f1) if f]
    if min(support)<.17:
      score-=.10
    if score>best_score:
      best_score=score
      best=(a,b,ali)
  return best,best_score
def waypoints(o,memory,c,heading):
  robots=o['robots']
  side=(-heading[1],heading[0])
  half=.36
  if c is not None:
    olda=c[2:4]
    half=.5*(abs(dot(olda,heading))*c[4]+abs(cross(olda,heading))*c[5])
  p=[position(r) for r in robots]
  order=1 if dot(p[0],side)<dot(p[1],side) else -1
  goal=shift(c,heading,-half-.40)
  arena=robots[0]['arena']
  points=[]
  for i,sign in enumerate((-order,order)):
    dst=shift(goal,side,.38*sign)
    if dot(sub(p[i],c),heading)>-half+.03:
      s=dot(sub(p[i],c),side)
      corner_sign=1 if s>.02 else -1 if s<-.02 else sign
      flank=shift(c,side,corner_sign*(.5*(abs(dot(c[2:4],side))*c[4]+abs(cross(c[2:4],side))*c[5])+.45))
      dst=shift(flank,heading,-half-.25)
      if abs(s)<mag(sub(flank,c))-.09:
        dst=shift(flank,heading,dot(sub(p[i],c),heading)*.58)
    points.append((clamp(dst[0],.27,arena[0]-.27),clamp(dst[1],.27,arena[1]-.27)))
  return points
def differential(a,b,faces,heading,robots,pairs):
  if a>=32 or b>=32 or not all(r['contacting_object'] for r in robots):
    return a,b
  f0=face_for(faces[0],DIR[a//4])
  f1=face_for(faces[1],DIR[b//4])
  if not f0 or not f1 or dot(f0[0],f1[0])>.88:
    return a,b
  n0,n1=f0[0],f1[0]
  default=dot(norm(shift(n0,n1,1.)),heading)
  options=[(default,a,b)]
  for speed0,speed1 in ((3,2),(2,3),(3,1),(1,3)):
    action0=a//4*4+speed0
    action1=b//4*4+speed1
    if [action0,action1] not in pairs:
      continue
    effective=shift(shift((0.,0.),n0,speed0+1),n1,speed1+1)
    residual=dot(norm(effective),heading)-.022*abs(speed0-speed1)
    options.append((residual,action0,action1))
  _,u,v=max(options)
  return u,v
def parent_prepare_context(memory: dict, observation: dict) -> dict:
  o=observation
  goal=o['goal']['xy']
  tid=o['goal']['target_id']
  c,faces,points=geometry(o,memory['c'])
  robots=o['robots']
  act=None
  d=0.
  active=o['active_target']==tid
  pairs=o['entity_legal_pairs'].get(tid,[])
  if c:
    d=mag(sub(goal,c))
    waypoint=memory['route']
    heading=norm(sub(waypoint,c)) if waypoint and d>.63 else route_goal(c,goal,o)
    best,score=push_plan(memory,o,c,faces,heading,pairs) if pairs and active else (None,-5.)
    threshold=.47 if waypoint and memory['stale_ticks']>160 else .59
    if best and best[2]>threshold and score>.44:
      a,b,_=best
      a,b=differential(a,b,faces,heading,robots,pairs)
      ticks=40 if d>.52 or not any(r['contacting_object'] for r in robots) else 17 if d>.28 else 8
      if d<.46 and any(r['contacting_object'] for r in robots):
        level=2 if d>.29 else 1
        slowed=([a//4*4+min(a%4,level) if a<32 else a,
              b//4*4+min(b%4,level) if b<32 else b])
        if slowed in pairs:
          a,b=slowed
      act=make(tid,a,b,ticks)
    elif pairs and not active and [44,44] in pairs:
      act=make(tid,44,44,40)
    else:
      stuck=memory['tslow']>175
      if stuck and (not waypoint or memory['stale_ticks']>240):
        sector=(memory['tslow']//150+memory['stall'])%4
        turn=(.8,-.8,1.9,-1.9)[sector]
        heading=norm(shift(heading,(-heading[1],heading[0]),turn))
      g0,g1=waypoints(o,memory,c,heading)
      prev=memory['last'] or [44,44]
      a=move(robots[0],g0,prev[0],memory['recent'][0],stuck or memory['pslow'][0]>90)
      b=move(robots[1],g1,prev[1],memory['recent'][1],stuck or memory['pslow'][1]>90)
      if pairs and best and (a==b==44 or stuck and best[2]>.40 and memory['tslow']%250<90):
        act=make(tid,best[0],best[1],36)
      elif pairs and stuck and memory['tslow']%260>185:
        side=[40 if cross(sub(position(r),c),shift(heading,heading,-2.))>0 else 43 for r in robots]
        ranked=sorted(range(2),key=lambda i:mag(sub(position(robots[i]),c)))
        for i in ranked:
          rr=(side[0],46) if i==0 else (46,side[1])
          if list(rr) in pairs and not robots[i]['execution'].get('done'):
            act=make(tid,rr[0],rr[1],35)
            break
      if act is None:
        act=make(None,a,b,26 if stuck else 30)
  else:
    prev=memory['last'] or [44,44]
    a=move(robots[0],goal,prev[0])
    b=move(robots[1],(o['robots'][0]['arena'][0]*.5,o['robots'][0]['arena'][1]*.5),prev[1])
    act=make(None,a,b,40)
  return dict(values=action_values(o,act),
        state=[d if c else None, float(bool(points)),sum(r['contacting_object'] for r in robots),memory['age']*.05,memory['progress']],
        proposals=[act,make(None,44,44,20)])
def evaluate_candidates(context: cython.const[cython.double][:],
            candidates: cython.const[cython.double][:, :],
            out: cython.double[:, :]) -> None:
  i: cython.Py_ssize_t
  for i in range(candidates.shape[0]):
    out[i,0] = float(candidates[i,0] == context[0])
    out[i,1] = .5*float(candidates[i,1] == context[1])+.5*float(candidates[i,2] == context[2])
    out[i,2] = -abs(candidates[i,3]-context[3])/40.
def module_update(local,event,memory):
  o=event['observation']
  tick=o['tick']
  dt=event.get('executed_ticks') or 0
  action=event.get('action')
  c=memory.get('compact')
  tid=o['goal']['target_id']
  plan=local.get('plan')
  last=local.pop('proposal',None)
  pending=local.pop('pending',None)
  if plan is None and pending and dt and action==last:
    plan=local['plan']=pending
  cells=local.get('cells',[])
  fixed=obstacle_points(o,tid,True)
  observed={round(p[0]*10)*64+round(p[1]*10) for p in fixed}
  if observed:
    cells=[k for k in cells if k not in observed]+sorted(observed)
    local['cells']=cells[-150:]
  valid=bool(c and get_points(o,tid))
  samples=[s for s in local.get('samples',[]) if tick-s[0]<290]
  if valid and dt:
    stalled=0
    if action and action['target_id']==tid:
      for i,r in enumerate(o['robots']):
        if action['a%d'%i]>=32:
          continue
        stage=r['execution'].get('stage')
        if stage in ('push','wait_peer_contact'):
          own=r['own']
          if r['contacting_static'] or (r['contacting_object'] and own['vx']**2+own['vy']**2<.007):
            stalled=dt
    samples.append([tick,c[0],c[1],stalled])
  local['samples']=samples[-40:]
  if plan:
    if not valid or tick-plan['start']>235:
      local.pop('plan')
    else:
      d=DIR[plan['direction']]
      advance=dot(sub(c,plan['center']),d)
      i=plan['robot']
      if advance>plan['best']+.070:
        plan['best']=advance
        plan['progress_tick']=tick
      if action and action==last and action['target_id']==tid:
        r=o['robots'][i]
        braced_progress=(plan.get('through') and tick-plan['progress_tick']<=21)
        if r['contacting_static'] and not braced_progress:
          plan['blocked']=plan.get('blocked',0)+dt
        else:
          plan['blocked']=0
      minimum_reached=advance>=plan['length'] and not plan.get('through')
      if (minimum_reached or advance>=1.32 or advance<-.21 or abs(cross(sub(c,plan['center']),d))>.39
        or tick-plan['progress_tick']> (91 if plan['best']>.12 else 135)
        or plan.get('blocked',0)>30):
        local.pop('plan')
    if 'plan' not in local:
      local['release']=[tick,*plan['center']]
      return local
  packet=module_act(local,memory,o,None)
  if packet:
    if plan:
      local['proposal']=packet
    else:
      local['proposal'],local['pending']=packet
  elif plan:
    local.pop('plan')
    local['release']=[tick,*plan['center']]
  return local
def module_act(local,memory,observation,base_action):
  o=observation
  c=memory.get('compact')
  tick=o['tick']
  tid=o['goal']['target_id']
  goal=o['goal']['xy']
  if not c or o['active_target']!=tid or not get_points(o,tid):
    return None
  plan=local.get('plan')
  if base_action is not None:
    proposal=local.get('proposal')
    if plan or not proposal:
      return proposal
    if any(r['contacting_object'] for r in o['robots']):
      return None
    hist=local.get('samples',[])
    old=[s for s in hist if tick-s[0]>=165]
    if (not old or mag(sub(c,old[-1][1:3]))>.17 or
        memory['stale_ticks']<165 or memory['progress']>.027):
      return None
    p=local['pending']
    d=DIR[p['direction']]
    direct=norm(sub(goal,c))
    detour=dot(d,direct)<.68
    if sum(s[3] for s in hist)<(65 if detour else 100):
      return None
    if local.get('release') and (tick-local['release'][0]<205 or
                  mag(sub(c,local['release'][1:]))<.32):
      return None
    return proposal
  robots=o['robots']
  poses=[position(r) for r in robots]
  arena=robots[0]['arena']
  pairs=o['entity_legal_pairs'].get(tid,[])
  axes=(c[2:4],(-c[3],c[2]))
  direct=norm(sub(goal,c))
  static=obstacle_points(o,tid,True)
  static+= [(k//64*.10,k%64*.10) for k in local.get('cells',[])]
  obstacles=obstacle_points(o,tid)+static
  def edge(p):
    return min(p[0],p[1],arena[0]-p[0],arena[1]-p[1])
  def box_gap(p):
    q=sub(p,c)
    return mag(tuple(max(0.,abs(dot(u,q))-.5*w) for u,w in zip(axes,c[4:])))
  def segment_free(p,q,extra,radius):
    if edge(q)<.22:
      return False
    for x in extra:
      if segment_near(x,p,q)<min(radius,mag(sub(x,p))-.018):
        return False
    return all(box_gap(shift(p,sub(q,p),f))>=min(radius,box_gap(p)-.017)
           for f in (.28,.56,.8,1.))
  def steer(i,dst,other):
    p=poses[i]
    returns=hits(robots[i])
    selected=None
    score=.04
    for j,u in enumerate(DIR):
      gap=clearance(p,u,arena,returns)
      advance=dot(sub(dst,p),u)
      if gap<.055 or advance<.055:
        continue
      q=shift(p,u,min(.32,gap,advance))
      if not segment_free(p,q,other,.20):
        continue
      rank=mag(sub(p,dst))-mag(sub(q,dst))
      if rank>score:
        speed=3 if advance>.41 and gap>.38 else 2 if advance>.24 and gap>.23 else 1
        selected=47+4*j+speed
        score=rank
    if selected is None:
      return None
    return make(None,selected if i==0 else 44,44 if i==0 else selected,29)
  def station(i,k):
    d=DIR[k]
    r=robots[i]
    p=poses[i]
    peer=poses[1-i]
    f=face_for(visible_faces(r),d)
    if f is None and not any(abs(dot(d,u))>.94 for u in axes):
      return None
    n=d if f is None else f[0]
    half=.5*sum(abs(dot(n,u))*w for u,w in zip(axes,c[4:]))
    q=shift(c,n,-half) if f is None else face_q(f)
    stage=shift(q,n,-.31)
    offset=sub(p,q)
    if mag(sub(p,stage))>1.14 or edge(stage)<.23:
      return None
    push=[44,44]
    push[i]=4*k+3
    peer_gap=segment_near(peer,stage,shift(q,n,.40))
    if (f and push in pairs and -.77<dot(offset,n)<-.12 and abs(cross(offset,n))<.39
      and not r['contacting_static']):
      if peer_gap>.39:
        return make(tid,push[0],push[1],33)
      push[1-i]=46
      if push in pairs and peer_gap>.23:
        return make(tid,push[0],push[1],33)
    if peer_gap<.39:
      back=norm(sub(peer,q))
      tangent=(-n[1],n[0])
      options=[]
      for side in (tangent,(-tangent[0],-tangent[1]),back):
        dst=shift(peer,side,.38)
        free=segment_near(dst,stage,shift(q,n,.40))
        if free>.42 and segment_free(peer,dst,[p]+obstacles,.22):
          options.append((free,dst))
      if options:
        cmd=steer(1-i,max(options)[1],[p]+obstacles)
        if cmd:
          return cmd
      return None
    if any(mag(sub(stage,x))<.20 for x in obstacles) or box_gap(stage)<.16:
      return None
    if r['contacting_static'] and mag(sub(p,stage))<.20:
      stage=shift(stage,sub(stage,q),.50)
    points=[stage]
    tangent=(-n[1],n[0])
    span=.5*sum(abs(dot(tangent,u))*w for u,w in zip(axes,c[4:]))
    for sign in (-1,1):
      corner=shift(stage,tangent,sign*(span+.24))
      if segment_free(corner,stage,[peer]+obstacles,.24):
        points.append(corner)
    for dst in points:
      if mag(sub(p,dst))<.09 or not segment_free(p,dst,[peer]+obstacles,.215):
        continue
      wanted=sub(dst,p)
      if (dst==stage and mag(wanted)>.46 and abs(dot(wanted,n))<.26*mag(wanted)):
        m=move(r,dst)
        if m>=47:
          u=DIR[(m-47)//4]
          qmove=shift(p,u,min(.34,clearance(p,u,arena,hits(r))))
          if dot(u,norm(wanted))>.60 and segment_free(p,qmove,[peer]+obstacles,.215):
            return make(None,m if i==0 else 44,44 if i==0 else m,29)
      cmd=steer(i,dst,[peer]+obstacles)
      if cmd:
        return cmd
    return None
  if plan:
    i,k=plan['robot'],plan['direction']
    d=DIR[k]
    free=safe_travel(c,d,obstacles,arena)
    if free<.11:
      return None
    if plan.get('through') and plan['best']>=plan['length']-.07:
      if dot(d,direct)<.56:
        alternatives=[u for u in DIR if dot(u,direct)>.53 and abs(dot(d,u))<.34
                and any(abs(dot(u,a))>.92 for a in axes)]
        if any(safe_travel(c,u,obstacles,arena)>.72 for u in alternatives):
          return None
      r=robots[i]
      peer=poses[1-i]
      p=poses[i]
      push=[44,44]
      push[i]=4*k+3
      if (push in pairs and free>.20 and tick-plan['progress_tick']<=39
          and r['execution'].get('stage')=='push'
          and segment_near(peer,p,shift(p,d,.65))>.39):
        return make(tid,push[0],push[1],26)
    return station(i,k)
  if robots[0]['remaining_seconds']<12.0 or mag(sub(goal,c))<.72:
    return None
  direct_gap=safe_travel(c,direct,static,arena)
  if direct_gap>.72:
    return None
  if not any(dot(f[0],direct)>.49 and -.75<dot(sub(p,face_q(f)),f[0])<-.10
         and abs(cross(sub(p,face_q(f)),f[0]))<.40
         for r,p in zip(robots,poses) for f in visible_faces(r)):
    return None
  best=None
  for k,d in enumerate(DIR):
    alignment=dot(d,direct)
    if alignment<-.10 or not any(abs(dot(d,u))>.93 for u in axes):
      continue
    safe=safe_travel(c,d,obstacles,arena)
    for length in (.36,.50,.66):
      if safe<max(length+.115,direct_gap+.20):
        continue
      nxt=[*shift(c,d,length),*c[2:]]
      onward=safe_travel(nxt,norm(sub(goal,nxt)),obstacles,arena)
      if onward<max(.61,direct_gap+.24):
        continue
      for i in range(2):
        cmd=station(i,k)
        if not cmd:
          continue
        score=.93*min(1.1,onward)+.53*alignment-.30*length
        half=.5*sum(abs(dot(d,u))*w for u,w in zip(axes,c[4:]))
        rear=shift(c,d,-half-.31)
        score-=.27*mag(sub(poses[i],rear))
        score-=.44*max(0.,dot(sub(poses[i],rear),d)-.27)
        if cmd['target_id']==tid:
          score+=.36
        elif cmd['a%d'%i]==44:
          score-=.16
        if best is None or score>best[0]:
          best=(score,cmd,{'center':list(c[:2]),'robot':i,'direction':k,'length':length,
                   'start':tick,'progress_tick':tick,'best':0.,
                   'through':alignment>.69})
  return [best[1],best[2]] if best else None
from copy import deepcopy as owned
def initial_memory() -> dict:
  memory=parent_initial_memory()
  memory['slot']={}
  return memory
def update_memory(memory: dict, event: dict) -> dict:
  slot=memory.pop('slot')
  memory=parent_update_memory(memory,event)
  slot=module_update(slot,owned(event),owned(memory))
  memory['slot']=slot
  return memory
def prepare_context(memory: dict, observation: dict) -> dict:
  slot=memory.pop('slot')
  ctx=parent_prepare_context(memory,observation)
  base=ctx['proposals'][0]
  action=module_act(slot,owned(memory),owned(observation),dict(base))
  if action is not None:
    ctx['proposals'].append(base)
    ctx['proposals'][0]=action
    ctx['values']=action_values(observation,action)
  return ctx
def recovery_points(o, base, slot):
  tid=o['goal']['target_id']
  cells=set(base['static_cells']) | set(slot.get('cells',[]))
  for p in obstacle_points(o,tid,True):
    cells.add(round(p[0]*10)*64+round(p[1]*10))
  fixed=[(k//64*.10,k%64*.10) for k in sorted(cells)]
  moving=[]
  for e in o['entities']:
    if e['id']!=tid:
      moving+=e['points']
  return fixed,moving
def recovery_sweep(c,d,points,limit,pad=.064,overlap=.045):
  a=c[2:4]
  b=(-a[1],a[0])
  u,v=dot(a,d),dot(b,d)
  adj=[(c[4]*.5+pad,u),(c[5]*.5+pad,v)]
  for point in points:
    q=sub(point,c)
    if dot(q,d)<.04 or dot(q,d)>limit+max(c[4:])*.64:
      continue
    x,y=dot(q,a),dot(q,b)
    enter,leave=-4.,4.
    for t,(h,w) in zip((x,y),adj):
      if abs(w)<.001:
        if abs(t)>h:
          enter=5.
          break
      else:
        lo,hi=(t-h)/w,(t+h)/w
        enter=max(enter,min(lo,hi))
        leave=min(leave,max(lo,hi))
    if leave>max(.075,enter) and enter<limit:
      if enter>-.19:
        limit=min(limit,max(overlap,enter))
  return limit
def recovery_axes(c):
  a=c[2:4]
  b=(-a[1],a[0])
  return (a,b,(-a[0],-a[1]),(-b[0],-b[1]))
def recovery_station(c,n):
  a,b,_,_=recovery_axes(c)
  w=.5*(abs(dot(n,a))*c[4]+abs(dot(n,b))*c[5])
  return shift(c,n,-w-.34)
def recovery_path(p,dst,peer,c,obstacles,arena):
  spacing=.22
  a,b,_,_=recovery_axes(c)
  mx,my=int(arena[0]/spacing),int(arena[1]/spacing)
  dst_cell=(round(dst[0]/spacing),round(dst[1]/spacing))
  start=(round(p[0]/spacing),round(p[1]/spacing))
  if start==dst_cell:
    return dst
  used=set()
  blocked=set()
  for q in obstacles+[peer]:
    k,l=round(q[0]/spacing),round(q[1]/spacing)
    rr=.33 if q==peer else .23
    for i in range(k-2,k+3):
      for j in range(l-2,l+3):
        if (i*spacing-q[0])**2+(j*spacing-q[1])**2<rr*rr:
          blocked.add((i,j))
  poss=[(mag(sub(dst_cell,start)),0.,start)]
  cost={start:0.}
  prev={}
  best=start
  while poss and len(used)<320:
    _,g,k=heappop(poss)
    if k in used:
      continue
    used.add(k)
    if mag(sub(k,dst_cell))<mag(sub(best,dst_cell)):
      best=k
    if k==dst_cell:
      break
    for i,(dx,dy) in enumerate(((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))):
      s=(k[0]+dx,k[1]+dy)
      if s in used or s in blocked or not (1<=s[0]<mx and 1<=s[1]<my):
        continue
      q=sub((s[0]*spacing,s[1]*spacing),c)
      gd=mag((max(0.,abs(dot(q,a))-.5*c[4]),max(0.,abs(dot(q,b))-.5*c[5])))
      if gd<.225:
        continue
      ng=g+(1.414 if i%2 else 1.)+.20*max(0.,.48-gd)
      if ng<cost.get(s,1000.):
        cost[s]=ng
        prev[s]=k
        heappush(poss,(ng+1.12*mag(sub(s,dst_cell)),ng,s))
  if best==start:
    return dst
  while prev[best]!=start:
    best=prev[best]
  return (spacing*best[0],spacing*best[1])
def recovery_move(r,dst,p,peer,arena,c,obstacles):
  wanted=sub(dst,p)
  dist=mag(wanted)
  if dist<.07:
    return 44,0
  hit=hits(r)
  a,b,_,_=recovery_axes(c)
  rel=sub(p,c)
  box=mag((max(0.,abs(dot(rel,a))-.5*c[4]),max(0.,abs(dot(rel,b))-.5*c[5])))
  peers=mag(sub(p,peer))
  best=(0.,44,0)
  for k,u in enumerate(DIR):
    if dot(u,wanted)<.12*dist:
      continue
    gap=clearance(p,u,arena,hit)
    length=min(.43,dist,gap-.026)
    if length<.06:
      continue
    q=shift(p,u,length)
    rel=sub(q,c)
    new_box=mag((max(0.,abs(dot(rel,a))-.5*c[4]),max(0.,abs(dot(rel,b))-.5*c[5])))
    if new_box<min(.214,box-.008) or mag(sub(q,peer))<min(.34,peers-.018):
      continue
    if any(segment_near(x,p,q)<min(.201,mag(sub(p,x))-.015) for x in obstacles):
      continue
    score=dist-mag(sub(q,dst))
    speed=3 if length>.25 else 2 if length>.13 else 1
    ticks=int(clamp(6+length/(.005*(speed+1)),10,34))
    if score>best[0]:
      best=(score,47+4*k+speed,ticks)
  return best[1:]
def recovery_step(plan,o,c,faces,obstacles,arena):
  i=plan['robot']
  robots=o['robots']
  tid=o['goal']['target_id']
  pairs=o['entity_legal_pairs'].get(tid,[])
  p,peer=position(robots[i]),position(robots[1-i])
  n=max(recovery_axes(c),key=lambda x:dot(x,plan['axis']))
  k=max(range(8),key=lambda j:dot(DIR[j],n))
  f=face_for(faces[i],DIR[k])
  stage=recovery_station(c,n)
  if 'offset' in plan:
    stage=shift(stage,(-n[1],n[0]),plan['offset'])
  command=[44,44]
  command[i]=4*k+3
  if f:
    midpoint=face_q(f)
    off=sub(p,midpoint)
    rear,lateral=-dot(off,f[0]),abs(dot(off,f[1]))
    peer_gap=segment_near(peer,shift(midpoint,f[0],-.49),shift(midpoint,f[0],.50))
    r=robots[i]
    if ((.125<rear<.72 and lateral<.29 or
       r['contacting_object'] and .06<rear<.42 and lateral<.34) and
       not r['execution'].get('done') and not r['contacting_static']):
      if peer_gap<.45:
        command[1-i]=46
      if command in pairs:
        return make(tid,command[0],command[1],40)
    if .18<rear<.95 and abs(dot(off,f[1]))<.49:
      observed_stage=shift(midpoint,f[0],-.32)
      if all(mag(sub(observed_stage,q))>.23 for q in obstacles):
        stage=observed_stage
  if not all(.235<stage[j]<arena[j]-.235 for j in (0,1)):
    return None
  if segment_near(peer,stage,shift(stage,n,.67))<.34:
    for v in ((-n[1],n[0]),(n[1],-n[0]),(-n[0],-n[1])):
      dst=shift(peer,v,.44)
      if segment_near(dst,stage,shift(stage,n,.67))<.41:
        continue
      m,t=recovery_move(robots[1-i],dst,peer,p,arena,c,obstacles)
      if m!=44:
        command=[44,44]
        command[1-i]=m
        return make(None,command[0],command[1],t)
  dst=recovery_path(p,stage,peer,c,obstacles,arena)
  m,t=recovery_move(robots[i],dst,p,peer,arena,c,obstacles)
  if m==44 and mag(sub(stage,p))>.11:
    m,t=recovery_move(robots[i],stage,p,peer,arena,c,obstacles)
  if m==44:
    return None
  command=[44,44]
  command[i]=m
  return make(None,command[0],command[1],t)
def rear_stations(c,n,obstacles,arena):
  back=recovery_station(c,n)
  tangent=(-n[1],n[0])
  a,b,_,_=recovery_axes(c)
  span=.5*(abs(dot(tangent,a))*c[4]+abs(dot(tangent,b))*c[5])
  stations=[]
  for offset in (0.,-.60*span,.60*span):
    q=shift(back,tangent,offset)
    end=shift(q,n,.14)
    if all(.22<z[j]<arena[j]-.22 for z in (q,end) for j in (0,1)) and all(
        segment_near(x,q,end)>.205 for x in obstacles):
      stations.append((offset,q))
  return stations
def graph_plan(o,memory,c,faces,obstacles,arena,running=None):
  tick=o['tick']
  goal=o['goal']['xy']
  gap=mag(sub(c,goal))
  if gap<.55 or o['robots'][0]['remaining_seconds']<7.:
    return None
  hist=memory['slot']['samples']
  before=[s for s in hist if tick-s[0]>=120]
  recent=[s for s in hist if tick-s[0]<=105]
  if not running and (not before or len(recent)<4 or mag(sub(before[-1][1:3],goal))-gap>.13):
    return None
  if not running and max(mag(sub(s[1:3],goal)) for s in recent)-min(mag(sub(s[1:3],goal)) for s in recent)>.22:
    return None
  axes=recovery_axes(c)
  g=norm(sub(goal,c))
  rear=max(axes,key=lambda n:dot(n,g))
  direct=recovery_sweep(c,rear,obstacles,min(.62,room_for(c,rear,arena)),.05)
  if direct<.29 or rear_stations(c,rear,obstacles,arena):
    return None
  poses=[position(r) for r in o['robots']]
  choices=[]
  for n in axes:
    alignment=dot(n,g)
    if n==rear or alignment<-.30:
      continue
    safe=recovery_sweep(c,n,obstacles,min(.95,room_for(c,n,arena)),.045)
    if safe<.36:
      continue
    stations=rear_stations(c,n,obstacles,arena)
    if not stations:
      continue
    for length in (.31,.47,.62):
      if safe<length+.045:
        continue
      end=[*shift(c,n,length),*c[2:]]
      onward=recovery_sweep(end,rear,obstacles,min(.87,room_for(end,rear,arena)),.045)
      exit_lane=bool(rear_stations(end,rear,obstacles,arena))
      gain=gap-mag(sub(goal,end))
      close=gap<1.65
      if onward<.27 or close and gain<-.075 or not exit_lane and (
          gain<.10 if close else alignment<.26):
        continue
      for i,p in enumerate(poses):
        for offset,station in stations:
          approach=mag(sub(p,station))
          if approach>1.68:
            continue
          if close:
            score=.38*alignment+.54*onward+.56*exit_lane+.54*gain-.39*approach-.24*length
          else:
            score=.55*alignment+.33*onward+.45*exit_lane-.42*approach-.37*length
          score-=.22*max(0.,dot(sub(p,station),n))
          score+=.20*(face_for(faces[i],n) is not None)
          plan={'axis':n,'start':tick,'center':c[:2],'robot':i,'length':length,
                'best':0.,'progress':tick,'followup':False,'offset':offset}
          choices.append((score,plan))
  for score,plan in sorted(choices,key=lambda x:x[0],reverse=True)[:6]:
    cmd=recovery_step(plan,o,c,faces,obstacles,arena)
    if cmd and (not running or any(cmd[k]!=running[k] for k in ('target_id','a0','a1'))):
      return score,plan,cmd
  return None
def recovery_select(o,memory,c,faces,fixed,obstacles,arena,followup=False):
  tick=o['tick']
  goal=o['goal']['xy']
  g=norm(sub(goal,c))
  gap=mag(sub(goal,c))
  slot=memory['slot']
  hist=slot['samples']
  older=[x for x in hist if tick-x[0]>=140]
  recent=[x for x in hist if tick-x[0]<=145]
  if not older or len(recent)<4 or gap<.85 or o['robots'][0]['remaining_seconds']<11.0:
    return None
  if mag(sub(older[-1][1:3],goal))-gap>.14:
    return None
  if max(mag(sub(x[1:3],goal)) for x in recent)-min(mag(sub(x[1:3],goal)) for x in recent)>.25:
    return None
  direct=recovery_sweep(c,g,fixed,.85)
  if direct>.70 and not followup:
    return None
  poses=[position(r) for r in o['robots']]
  if not followup and not any(dot(f[0],g)>.42 and .09<-dot(sub(p,face_q(f)),f[0])<.85
         and abs(dot(sub(p,face_q(f)),f[1]))<.43
         for p,fs in zip(poses,faces) for f in fs):
    return None
  contact=any(r['contacting_object'] for r in o['robots'])
  ctx=original_prepare_context(owned(memory),o)
  if len(ctx['proposals'])>2:
    return None
  base=ctx['proposals'][0]
  paired=max(base['a0'],base['a1'])<32
  if not followup and not contact and not paired:
    return None
  if not followup and contact and not paired:
    if base['target_id'] is None:
      return None
    if signature(base)==memory['last']:
      return None
  best=None
  for n in recovery_axes(c):
    alignment=dot(n,g)
    minimum=(.58 if direct>.70 else -.15) if followup else (.15 if contact and paired else -.30 if contact else .05)
    if alignment<minimum:
      continue
    padding=.047 if followup else .064
    free=recovery_sweep(c,n,obstacles,min(1.42,room_for(c,n,arena)),padding)
    for length in ((.32,.53,.79,.98) if followup else (.43,.62,.85,1.06)):
      if free<length+.10:
        continue
      end=[*shift(c,n,length),*c[2:]]
      onward=recovery_sweep(end,norm(sub(goal,end)),obstacles,1.25,padding)
      if onward<(max(.43,min(.56,direct+.10)) if followup else max(.43,direct+.16)):
        continue
      for i,p in enumerate(poses):
        station=recovery_station(c,n)
        approach=mag(sub(station,p))
        if approach>(1.96 if followup else 1.60) or any(mag(sub(station,q))<.24 for q in obstacles):
          continue
        side=abs(cross(sub(p,station),n))
        extra=max(0.,dot(sub(p,station),n)) if side<.77 else 0.
        score=1.22*onward+.50*alignment-.35*length-.46*approach-.31*extra
        plan={'axis':n,'start':tick,'center':c[:2],'robot':i,'length':length,
            'best':0.,'progress':tick,'followup':followup}
        if best is None or score>best[0]:
          cmd=recovery_step(plan,o,c,faces,obstacles,arena)
          if cmd and (followup or contact or cmd['target_id']==o['goal']['target_id']):
            if not followup and not contact and (cmd['a%d'%i]//4!=base['a%d'%i]//4):
              continue
            best=score,plan,cmd
  return best
original_initial_memory=initial_memory
original_update_memory=update_memory
original_prepare_context=prepare_context
def initial_memory() -> dict:
  m=original_initial_memory()
  m['recovery']={}
  return m
def update_memory(memory: dict,event: dict) -> dict:
  o=event['observation']
  review=event.get('push_checkpoint')
  if review:
    memory['review']=None
    state=memory['recovery']
    if (review.get('force') and not state.get('plan') and not memory['slot'].get('plan')
        and o['tick']-state.get('release',-200)>=170):
      c,faces,points=geometry(o,memory['c'])
      running=event.get('requested_action') or event.get('action')
      if points and running and running['target_id']==o['goal']['target_id']:
        fixed,moving=recovery_points(o,memory,memory['slot'])
        plan=graph_plan(o,memory,c,faces,fixed+moving,o['robots'][0]['arena'],running)
        if plan:
          memory['review']=plan[2]
    return memory
  state=memory.pop('recovery')
  memory=original_update_memory(memory,event)
  tick=o['tick']
  tid=o['goal']['target_id']
  c=memory.get('compact')
  previous=state.pop('command',None)
  pending=state.pop('pending',None)
  requested=event.get('requested_action') or event.get('action')
  if pending and previous==requested and event.get('executed_ticks'):
    state['plan']=pending
    if pending['followup']:
      state['followups']=state.get('followups',0)+1
  plan=state.get('plan')
  arena=o['robots'][0]['arena']
  if c and o['active_target']==tid and get_points(o,tid):
    faces=[visible_faces(r) for r in o['robots']]
    extend=False
    if plan:
      advance=dot(sub(c,plan['center']),plan['axis'])
      if advance>plan['best']+.056:
        plan['progress']=tick
        plan['best']=advance
      goal=o['goal']['xy']
      gap_gain=mag(sub(goal,plan['center']))-mag(sub(goal,c))
      i=plan['robot']
      r=o['robots'][i]
      extend=(.07<advance<plan['length']+.53 and gap_gain<.22 and
          tick-plan['progress']<=45 and requested and previous==requested and
          requested['target_id']==tid and requested['a%d'%i]<32 and
          r['execution'].get('stage')=='push' and r['contacting_object'])
      graph='offset' in plan and mag(sub(goal,plan['center']))<1.65
      if graph:
        extend=False
      if ((advance>=plan['length'] and not extend) or advance<-.24 or
        graph and (gap_gain<-.185 or abs(cross(sub(c,plan['center']),plan['axis']))>.30) or
        tick-plan['start']>265 or tick-plan['progress']>(155 if graph else 135)):
        if advance>.24:
          state['prior']=tick
        state['release']=tick
        state.pop('plan')
        plan=None
    if plan or (not memory['slot'].get('plan') and tick-state.get('release',-200)>=170):
      fixed,moving=recovery_points(o,memory,memory['slot'])
      obstacles=fixed+moving
      if plan:
        if extend:
          a,b=requested['a0'],requested['a1']
          f=face_for(faces[i],DIR[requested['a%d'%i]//4])
          legal=o['entity_legal_pairs'].get(tid,[])
          if (f and [a,b] in legal and not r['contacting_static'] and
            recovery_sweep(c,f[0],obstacles,.36,.035)>.19):
            state['command']=make(tid,a,b,40)
          else:
            extend=False
        if not extend:
          cmd=recovery_step(plan,o,c,faces,obstacles,arena)
          if cmd and advance<plan['length']:
            state['command']=cmd
          else:
            if advance>.24:
              state['prior']=tick
            state['release']=tick
            state.pop('plan')
      else:
        best=graph_plan(o,memory,c,faces,obstacles,arena)
        if not best:
          best=recovery_select(o,memory,c,faces,fixed,obstacles,arena)
        if (not best and state.get('prior') is not None and
            tick-state['prior']<630 and state.get('followups',0)<2):
          best=recovery_select(o,memory,c,faces,fixed,obstacles,arena,True)
        if best:
          _,state['pending'],state['command']=best
  elif plan:
    state['release']=tick
    state.pop('plan')
  memory['recovery']=state
  return memory
def prepare_context(memory: dict,observation: dict) -> dict:
  if 'review' in memory:
    action=memory['review']
    return dict(values=action_values(observation,action) if action else [-1,44,44,20],
        state=[None,0.,0.,0.,0.],proposals=[action] if action else [])
  state=memory.pop('recovery')
  ctx=original_prepare_context(memory,observation)
  act=state.get('command')
  if act:
    ctx['proposals'].insert(0,act)
    ctx['values']=action_values(observation,act)
  return ctx
