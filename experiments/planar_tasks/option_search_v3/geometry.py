import math
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def estimate(o,prev=None):
 ps=[p for r in o['robots'] for p in r['patches'] if p.get('valid') and p.get('rms_m',1)<.002 and p['span_m']>.035]
 pts=[p for r in o['robots'] for pa in r['patches'] for p in pa['points_world']]
 if not ps:return prev
 p=max(ps,key=lambda p:p['span_m']);n=p['normal_world'];theta=math.atan2(n[1],n[0])+ (math.pi/2 if p['span_m']>.28 else 0)
 if prev and p['span_m']<=.28:
  base=math.atan2(n[1],n[0]);opts=[base,base+math.pi/2];theta=min(opts,key=lambda a:abs((a-prev[2]+math.pi/2)%math.pi-math.pi/2))
 u=[math.cos(theta),math.sin(theta)];v=[-u[1],u[0]];cs=[]
 for ax,h in [(u,.5),(v,.125)]:
  vals=[dot(pt,ax) for pt in pts];lo=min(vals);hi=max(vals)
  faces=[]
  for pa in ps:
   nd=dot(pa['normal_world'],ax)
   if abs(nd)>.98:
    pv=sum(dot(pt,ax) for pt in pa['points_world'])/len(pa['points_world']);faces.append(pv+(h if nd>0 else -h))
  if faces:c=sum(faces)/len(faces)
  elif hi-lo>1.5*h:c=(lo+hi)/2
  elif prev:c=max(hi-h,min(lo+h,dot(prev,ax)))
  else:c=(lo+hi)/2
  cs.append(c)
 return [cs[0]*u[0]+cs[1]*v[0],cs[0]*u[1]+cs[1]*v[1],theta]
def static(o):
 pts=[];w,h=o['robots'][0]['arena']
 for r in o['robots']:
  s=r['scan'];ox,oy=s['origin']
  for a,d,c in zip(s['ray_angles'],s['distances'],s['category']):
   if c==2 and d is not None:
    p=[ox+d*math.cos(a),oy+d*math.sin(a)]
    if .08<p[0]<w-.08 and .08<p[1]<h-.08:pts.append(p)
 return pts
