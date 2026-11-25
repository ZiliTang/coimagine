"""Empirical clearance range for a rigid rectangular envelope from public target returns.

Earlier poses contribute size estimates only. Current points anchor the envelope
in the room frame. The margins are geometric hypotheses, not verified size bounds.
"""
import math

import numpy as np


def _frame(obs, label):
    points = []
    edges = []
    spacings = []
    for robot in obs['robots']:
        scan = robot['scan']
        origin = np.asarray(scan['origin'])
        hits = []
        for valid, entity, angle, d in zip(scan['valid'], scan['entity_ids'], scan['ray_angles'],
                                            scan['distances'], strict=True):
            if valid and entity == label:
                point = origin + d * np.array([math.cos(angle), math.sin(angle)])
                points.append(point)
                hits.append(point)
            else:
                hits.append(None)
        # Three consecutive hits support one face and its observer-facing side.
        for i, a in enumerate(hits):
            b, c = hits[(i+1) % len(hits)], hits[(i+2) % len(hits)]
            if a is None or b is None or c is None:
                continue
            delta = c-a
            length = np.linalg.norm(delta)
            if not .025 <= length <= .50:
                continue
            tx, ty = delta / length
            if abs((b[0]-a[0])*ty-(b[1]-a[1])*tx) > .004:
                continue
            edges.append((math.atan2(ty,tx), length, b, origin))
            spacings.append(max(np.linalg.norm(b-a), np.linalg.norm(c-b)))
    if not edges:
        return None
    # In a rectangle, both face families agree on orientation modulo 90 degrees.
    vx = sum(length*math.cos(4*angle) for angle,length,_,_ in edges)
    vy = sum(length*math.sin(4*angle) for angle,length,_,_ in edges)
    if math.hypot(vx,vy) < .75*sum(e[1] for e in edges):
        return None
    theta = .25*math.atan2(vy,vx)
    axes = np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
    points = np.asarray(points)
    origin = points.mean(axis=0)
    projected = (points-origin) @ axes.T
    return axes,origin,projected.min(axis=0),projected.max(axis=0),edges,float(np.median(spacings)),points


class FootprintHistory:
    def __init__(self):
        self.max_spans = np.zeros(2)
        self.frame = None
        self.spacing = 0.

    def observe(self, obs, label):
        self.frame = _frame(obs,label)
        if self.frame is not None:
            _,_,lo,hi,_,spacing,_ = self.frame
            self.max_spans = np.maximum(self.max_spans,np.sort(hi-lo))
            self.spacing = max(self.spacing,spacing)

    def gap_range(self, visible_gap, normal):
        if self.frame is None:
            return None
        axes,origin,lo,hi,edges,_,points = self.frame
        n = np.asarray(normal)
        coefficients = axes @ n
        sizes = self.max_spans + 2*self.spacing
        upper_support = []
        # Both assignments of long and short sides are considered if compatible.
        for widths in (sizes,sizes[::-1]):
            if np.any(widths < hi-lo-1e-8):
                continue
            # A compatible rectangle contains all points. For a visible face the
            # scanner origin also tells which side of the rectangle was observed.
            start_min = hi-widths
            start_max = lo.copy()
            for angle,_,point,sensor in edges:
                tangent=np.array([math.cos(angle),math.sin(angle)])
                k=int(np.argmin(np.abs(axes @ tangent)))
                if abs(axes[k] @ tangent) > math.sin(math.radians(8)):
                    continue
                q=(point-origin) @ axes[k]
                if (sensor-point) @ axes[k] > 0 and abs(q-hi[k])<=.012:
                    start_max[k]=min(start_max[k],q-widths[k]+.012)
                elif (sensor-point) @ axes[k] < 0 and abs(q-lo[k])<=.012:
                    start_min[k]=max(start_min[k],q-.012)
            if np.any(start_min>start_max+1e-8):
                continue
            offset=np.maximum(coefficients,0) @ widths
            upper_support.append(float(coefficients @ np.where(coefficients>=0,start_max,start_min)+offset))
        if not upper_support:
            return None
        # The inner limit is the observed footprint; the outer one adds unseen material.
        seen=float(np.max((points-origin) @ n))
        return [float(visible_gap - max(0.,max(upper_support)-seen)), float(visible_gap)]
