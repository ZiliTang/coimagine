"""Astra pilot: visible-normal alignment, persistent push, stall recovery.
Only stdin sensors and in-process history are used. No simulator imports.
"""
import json
import math
import sys


def norm(x,y):
    length=math.hypot(x,y)
    return (x/length,y/length) if length>1e-8 else (1.,0.)


def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi


class Policy:
    def __init__(self,contract):
        self.actions={a['index']:a for a in contract['actions']}
        self.last=None
        self.mode='transport'
        self.rotation=36
        self.mode_start=0.
        self.history=[]
        self.recoveries=0

    def act(self,o):
        robots=o['robots'];t=robots[0]['time']
        legal={tuple(p) for p in o['legal_pairs']}
        points=[p for r in robots for f in r['patches'] if f['valid'] for p in f['points_world']]
        if points:
            center=((min(p[0] for p in points)+max(p[0] for p in points))/2,
                    (min(p[1] for p in points)+max(p[1] for p in points))/2)
        else:
            center=tuple(sum(r['own'][k] for r in robots)/2 for k in ('x','y'))
        goal=robots[0]['goal_xy'];direction=norm(goal[0]-center[0],goal[1]-center[1])
        self.history.append((t,center));self.history=[h for h in self.history if t-h[0]<=10.]
        stalled=len(self.history)>3 and t-self.history[0][0]>7. and math.dist(center,self.history[0][1])<.045
        normals=[]
        for r in robots:
            faces=[f for f in r['patches'] if f['valid']]
            normals.append(max(faces,key=lambda f:f['span_m'])['normal_world'] if faces else (0.,0.))
        align=[sum(n*d for n,d in zip(normal,direction)) for normal in normals]
        distance=math.dist(center,goal)
        if distance < robots[0]['goal_radius']*.8:
            return self.output((44,44),10)
        preferred_speed=.1 if distance<.65 else .3
        both_push=[]
        for pair in legal:
            if all(i<32 for i in pair):
                ds=[self.actions[i]['effect_direction'] for i in pair]
                scores=[sum(v*d for v,d in zip(vect,direction)) for vect in ds]
                if min(scores)>.65:
                    both_push.append(pair)
        if self.mode=='rotate':
            if max(align)>.998 and t-self.mode_start>2.:
                self.mode='transport';self.history=[]
            elif not any(r['execution'].get('done') for r in robots) and t-self.mode_start<28. and (self.rotation,self.rotation) in legal:
                return self.output((self.rotation,self.rotation),4 if max(align)>.9 else 20)
            else:
                self.mode='transport';self.history=[]
        if self.mode=='reposition':
            if both_push or t-self.mode_start>5.:
                self.mode='transport';self.history=[]
            elif self.last in legal:
                return self.output(self.last,20)
        if (not both_push and max(align)<.82 and t<2.) or (stalled and not both_push):
            i=max(range(2),key=lambda j:align[j])
            normal=norm(*normals[i]);delta=wrap(math.atan2(direction[1],direction[0])-math.atan2(normal[1],normal[0]))
            rotation=34 if delta>=0 else 37
            if (rotation,rotation) in legal and self.recoveries<3:
                self.rotation=rotation;self.mode='rotate';self.mode_start=t;self.recoveries+=1;self.history=[]
                return self.output((rotation,rotation),20)
        if stalled and self.recoveries<5:
            # Move the poorly placed robot around the observed object, with peer holding.
            i=min(range(2),key=lambda j:align[j]);r=robots[i]['own']
            theta=math.atan2(r['y']-center[1],r['x']-center[0])
            target=math.atan2(-direction[1],-direction[0])
            action=40 if wrap(target-theta)>0 else 43
            pair=[44,44];pair[i]=action;pair=tuple(pair)
            if pair in legal:
                self.mode='reposition';self.mode_start=t;self.recoveries+=1;self.history=[]
                return self.output(pair,4 if distance<.65 else 20)
        def score(pair):
            total=0.;push=0
            for index in pair:
                a=self.actions[index]
                if index<32:
                    alignment=sum(v*d for v,d in zip(a['effect_direction'],direction))
                    if alignment<.45:return -100.
                    total+=alignment*2.-abs(a['speed']-preferred_speed)*3.;push+=1
                elif index==46:total-=.10
                elif index==44:total-=.7
                else:return -100.
            if push==2:
                if min(align)<.93:return -100.
                ds=[self.actions[i]['effect_direction'] for i in pair]
                total+=.5*sum(a*b for a,b in zip(*ds))
            if pair==self.last:total+=.10
            return total
        pair=max(sorted(legal),key=score)
        return self.output(pair,4 if distance<.65 else 20)

    def output(self,pair,ticks):
        self.last=tuple(pair)
        return dict(a0=pair[0],a1=pair[1],ticks=ticks)


policy=None
for line in sys.stdin:
    request=json.loads(line)
    if request.get('reset') or policy is None:policy=Policy(request['contract'])
    print(json.dumps(policy.act(request['observation'])),flush=True)
