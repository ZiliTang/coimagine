"""Frozen legal-sensor estimator from the 20/30 program; explicit cold root."""
from geometry import estimate, static


class Estimator:
    def __init__(self):self.pose=None;self.points=[]
    def update(self,observation):
        self.pose=estimate(observation,self.pose);self.points.extend(static(observation))
        if self.pose is None:return None
        r=observation['robots'][0];obstacles=[]
        if self.points:
            xs=[p[0] for p in self.points];ys=[p[1] for p in self.points]
            lo,hi,bot,top=min(xs),max(xs),min(ys),max(ys)
            obstacles=[[(lo+hi)/2,(bot+top)/2,max(.008,(hi-lo)/2),max(.008,(top-bot)/2),0]]
        return dict(arena=r['arena'],goal=r['goal_xy'],goal_radius=r['goal_radius'],
            object=self.pose+[0,0,0],obstacles=obstacles,
            robots=[[r['own'][k] for k in ('x','y','angle','vx','vy','omega')] for r in observation['robots']])
