"""Sensor-only persistent option policy; no simulator imports or file access."""
import json, math, sys

def dot(a,b): return a[0]*b[0]+a[1]*b[1]
def unit(v):
 d=math.hypot(*v); return (v[0]/d,v[1]/d) if d else (1.,0.)
class Policy:
 def __init__(self,c): self.actions={a['index']:a for a in c['actions']}; self.last=None; self.age=0; self.prev=None; self.stall=0; self.rotate=None; self.rot_age=0
 def act(self,o):
  rs=o['robots']; legal=o['legal_pairs']; pts=[]
  for r in rs:
   s=r['scan']
   for ang,d,cat in zip(s['ray_angles'],s['distances'],s['category']):
    if cat==1 and d is not None: pts.append((s['origin'][0]+d*math.cos(ang),s['origin'][1]+d*math.sin(ang)))
  if pts:
   lo=[min(p[i] for p in pts) for i in (0,1)]; hi=[max(p[i] for p in pts) for i in (0,1)]; center=[(lo[i]+hi[i])/2 for i in (0,1)]
   # A single visible planar face gives a surface, not the body centre.
   ns=[r['patch']['normal_world'] for r in rs if r.get('patch') and r['patch'].get('valid')]
   if ns and all(dot(ns[0],n)>.9 for n in ns): center=[center[i]+.125*ns[0][i] for i in (0,1)]
  else: center=self.prev or rs[0]['goal_xy']
  goal=rs[0]['goal_xy']; delta=[goal[i]-center[i] for i in (0,1)]; dist=math.hypot(*delta); u=unit(delta)
  self.stall=self.stall+1 if self.prev and math.dist(center,self.prev)<.005 else 0; self.prev=center
  desired=[]
  for r in rs:
   p=r.get('patch'); n=p['normal_world'] if p and p.get('valid') else unit([center[0]-r['own']['x'],center[1]-r['own']['y']])
   al=dot(n,u)
   if al>.78:
    available=[a for a in self.actions.values() if a.get('effect_kind')=='translate']
    a=max(available,key=lambda a:dot(a['effect_direction'],u)-abs(a['speed']-(.3 if dist>.45 else .1))*.5)
    desired.append(a['index'])
   else:
    cross=n[0]*u[1]-n[1]*u[0]; desired.append(40 if cross>0 else 43)
  # Release inaccessible contact geometry with a real unilateral push,
  # while the partner vacates instead of actively resisting the motion.
  normals=[r.get('patch',{}).get('normal_world') for r in rs]
  if all(normals) and dot(normals[0],normals[1])<-.7 and max(dot(n,u) for n in normals)<.72:
   candidates=[]
   for i,r in enumerate(rs):
    n=normals[i]; rel=[r['own']['x']-center[0],r['own']['y']-center[1]]
    # Reward torque that turns the selected inward normal toward the goal.
    torque=rel[0]*n[1]-rel[1]*n[0]
    turn=n[0]*u[1]-n[1]*u[0]
    for pair in legal:
     ac=self.actions[pair[i]]
     if ac.get('effect_kind')=='translate' and pair[1-i]==46:
      candidates.append((torque*turn+dot(ac['effect_direction'],n)*.3+ac['speed']*.1,pair))
   if candidates:
    pair=max(candidates,key=lambda v:v[0])[1];self.last=pair
    return {'a0':pair[0],'a1':pair[1],'ticks':20}
  def score(pair):
   sc=0
   for i,a in enumerate(pair):
    target=desired[i]; ac=self.actions[a]
    if a==target: sc+=10
    elif a==46: sc-=1
    elif a==44: sc-=3
    elif ac.get('effect_kind')=='translate': sc+=dot(ac['effect_direction'],u)*3-abs(ac['speed']-self.actions[target].get('speed',.3))
    else: sc-=5
   if pair==self.last: sc+=.2
   return sc
  pair=max(legal,key=score)
  self.last=pair
  return {'a0':pair[0],'a1':pair[1],'ticks':20 if dist>.4 else 6}
p=None
for line in sys.stdin:
 try:
  m=json.loads(line)
  if p is None or m.get('reset'): p=Policy(m['contract'])
  o=m.get('observation',m); print(json.dumps(p.act(o)),flush=True)
 except Exception:
  print(json.dumps({'a0':44,'a1':44,'ticks':20}),flush=True)
