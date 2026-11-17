"""Explicit geometric hypotheses from public ray hits, with uncertainty metadata."""
from copy import deepcopy
import math
import numpy as np
from .contract import public_observation


def fit_box(points, origins, *, depth_prior=.25, padding=.02):
    points=np.unique(np.asarray(points,dtype=float),axis=0)
    if len(points)<3:
        return None
    centre=points.mean(axis=0)
    _,singular,axes=np.linalg.svd(points-centre,full_matrices=False)
    tangent=axes[0]
    # Canonical sign prevents equivalent PCA axes flipping between observations.
    if tangent[np.argmax(np.abs(tangent))]<0:tangent=-tangent
    normal=np.array([-tangent[1],tangent[0]])
    projected=np.stack((points@tangent,points@normal),axis=1)
    lo,hi=projected.min(axis=0),projected.max(axis=0)
    mid=(lo+hi)/2
    width=max(.04,float(hi[0]-lo[0])+2*padding)
    line_like=bool(singular[-1]<max(1e-6,.08*singular[0]))
    if line_like:
        # A single visible surface does not reveal depth. Extrude away from
        # observing robots under an explicit common nominal depth assumption.
        direction=1. if np.dot(centre-np.mean(origins,axis=0),normal)>=0 else -1.
        mid[1]+=direction*depth_prior/2
        height=depth_prior
    else:
        height=max(.04,float(hi[1]-lo[1])+2*padding)
    c=mid[0]*tangent+mid[1]*normal
    return dict(x=float(c[0]),y=float(c[1]),angle=math.atan2(tangent[1],tangent[0]),
                width=width,height=height,point_count=len(points),
                uncertainty='unseen depth filled by nominal prior' if line_like else 'visible support box; unseen extent unknown')


def estimate(history):
    if not history.events:raise ValueError('nonempty legal history required')
    latest={}; static_points=[]
    for event in history.events:
        o=public_observation(event['observation'])
        origins=[r['scan']['origin'] for r in o['robots'] if 'origin' in r.get('scan',{})]
        for e in o['entities']:
            # Never pool samples from different object poses as one contour.
            if e['last_seen_tick']!=o['tick']:continue
            fitted=fit_box(e['points'],origins) if origins else None
            if fitted:
                latest[e['id']]=dict(fitted,last_seen_tick=o['tick'])
        for r in o['robots']:
            s=r['scan']
            current=[]
            for a,d,v,c in zip(s['ray_angles'],s['distances'],s['valid'],s['category']):
                if v and c==2:
                    p=[s['origin'][0]+d*math.cos(a),s['origin'][1]+d*math.sin(a)]
                    arena=r['arena']
                    if .08<p[0]<arena[0]-.08 and .08<p[1]<arena[1]-.08:
                        if current and math.dist(p,current[-1])>.15:
                            static_points.append(current);current=[]
                        current.append(p)
                elif current:
                    static_points.append(current);current=[]
            if current:static_points.append(current)
    root=public_observation(history.events[-1]['observation'])
    if len(root['robots'])!=2:raise ValueError('two public robot states required')
    primary=root['goal']['target_id']
    if primary not in latest:raise ValueError('transport object has insufficient observed geometry')
    entities=[]
    for label,box in latest.items():
        entities.append(dict(id=label,**box,stale_ticks=root['tick']-box['last_seen_tick'],
                             velocity_assumption='zero; not observed rigid-body truth'))
    # Retain separate local occupied patches; never merge all static hits into
    # one huge obstacle. These thin boxes are hypotheses, not a complete map.
    obstacles=[]; seen=set()
    for cluster in reversed(static_points):
        if len(cluster)<2:continue
        xs,ys=zip(*cluster)
        box=[(min(xs)+max(xs))/2,(min(ys)+max(ys))/2,
             max(.015,(max(xs)-min(xs))/2),max(.015,(max(ys)-min(ys))/2),0.]
        key=tuple(round(v,2) for v in box)
        if key not in seen:obstacles.append(box);seen.add(key)
        if len(obstacles)>=32:break
    return dict(version='ray-box-hypothesis-v1',root_tick=root['tick'],
                active_target=root.get('active_target',primary),
                arena=deepcopy(root['robots'][0]['arena']),goal=deepcopy(root['goal']),
                robots=[deepcopy(r['own']) for r in root['robots']],entities=entities,obstacles=obstacles,
                omitted_current_ids=[e['id'] for e in root['entities'] if e['id'] not in latest],
                assumptions=['unknown space unconstrained in this nominal model',
                             'mass/friction/damping use shared defaults',
                             'object velocities and controller state cold-start',
                             'stale objects retained at last estimated pose'])
