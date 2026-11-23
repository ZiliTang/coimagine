"""Estimate centres at matched, gate-eligible snapshots along the local/gated trajectory."""
from math import cos, sin, dist, pi
from collections import Counter
import sys

from tools import cap_lidar_static_gate as study

prior, e = study.prior, study.e


def faces(patches):
    return e.AM.visible_faces({'patches':[{
        'points_world':p.points_world, 'normal_world':p.normal_world,'valid':p.valid
    } for p in patches]})


class CenterAudit(study.GatedProbe):
    def __init__(self, world):
        super().__init__(world)
        self.views = {}
        self.samples = []
        self.first_large_error = None

    def observe(self, i, t, time, contacts=None):
        previous = self.audit_pose[i]
        chosen = super().observe(i,t,time,contacts)
        env = self.env
        if not self.in_control or t%11 or env.active_target != env.primary_id:
            return chosen
        frame,_ = self.staggered(i,t,time)
        fused = study.is_quiet(self.recent[i],env.active_target)
        raw = prior.patches(env,frame,env.active_target) if fused else chosen.patches
        self.views[i] = (t,fused,faces(raw),faces(chosen.patches))
        if i!=1 or 0 not in self.views or self.views[0][0]!=t or not any(v[1] for v in self.views.values()):
            return chosen
        assert previous is not None and previous[0]==t-1
        pose = self.audit_pose[i][1]
        still = dist(previous[1][:2],pose[:2])<=.001 and abs(pose[2]-previous[1][2])<=pi/360
        points = []
        for j in (0,1):
            packet = env.ray_frame(j,t,time)
            scan = packet.scan
            for angle,r,label in zip(scan.ray_angles,scan.distances,packet.entity_ids):
                if label==env.primary_id:
                    points.append((scan.origin[0]+r*cos(angle),scan.origin[1]+r*sin(angle)))
        if points:
            errors = []
            for index in (2,3):
                planes = self.views[0][index]+self.views[1][index]
                initial = study.MODULES['local'].compact_center(points,planes,None)
                fitted = study.MODULES['local'].compact_center(points,planes,initial)
                errors.append(100*dist(fitted[:2],pose[:2]))
            if self.first_large_error is None and still and errors[1] > 30:
                # Compare the inferred widths with the rectangle's span on the same axes.
                ux,uy=fitted[2:4]
                long_axis=(-sin(pose[2]),cos(pose[2]))
                short_axis=(cos(pose[2]),sin(pose[2]))
                true_spans=[abs(ax*long_axis[0]+ay*long_axis[1])+.25*abs(ax*short_axis[0]+ay*short_axis[1])
                            for ax,ay in ((ux,uy),(-uy,ux))]
                self.first_large_error=dict(tick=t, center_error_cm=errors[1], estimated_widths=fitted[4:],
                                            true_spans=true_spans, points=len(points), fitted_faces=len(planes))
            self.samples.append(dict(tick=t,quiet=still,current_cm=errors[0],fusion_cm=errors[1]))
        return chosen


def audit(job):
    panel,key,fixture=job
    cap=e.s.phase.FrozenCAP(study.MODULES['local'])
    with e.s.phase.world(fixture) as (w,_):
        probe=CenterAudit(w)
        cap.update(w.history)
        while not e.s.ended(w):
            cmd=cap.propose(e.s.observe(w))
            e.s.tick_action(w,cmd,e.g.prep.prior.EXECUTOR,cap.update,Counter())
        e.save(study.OUT/'centers'/(key+'.json'),dict(history=w.history.sha256,samples=probe.samples,first_large_error=probe.first_large_error))


if __name__=='__main__':
    jobs=study.initialize()
    e.pool(audit,[j for j in jobs if j[0]=='original' and (len(sys.argv)==1 or j[1] in sys.argv[1:])],6)
