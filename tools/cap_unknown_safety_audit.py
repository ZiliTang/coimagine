"""Replay old first-contact prefixes; hidden geometry is audit labeling only."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import math
import time

from shapely.geometry import Point, Polygon
from tools.cap_symbolic_general import OUT as OLD, ROOT, read, save
from experiments.planar_tasks.authoring_v2.symbolic_general import simulation, frontend
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.authoring_v2.random_room_tasks import polygon

OUT=ROOT/'outputs/semantic_rl/cap_unknown_safety_v1'


def worker(room):
    fixture=read(OLD/'panels.json')['groups']['confirmation'][room]
    prior=read(OLD/'confirmation/tasks/2'/(room+'.json.gz'))
    end=next(i for i,r in enumerate(prior['records'])if r['audit']['object_static'])
    # Select the nearest static fixture geometry at the recorded contact endpoint.
    endpoint=prior['records'][end]['audit']['after']
    body=Polygon(polygon(*endpoint['xy'],.25,1.,endpoint['angle']))
    obstacles=[Polygon(polygon(b['cx'],b['cy'],2*b['half_w'],2*b['half_h'],b['angle']))for b in fixture['room']['obstacles']]
    nearest=min(range(len(obstacles)),key=lambda i:body.distance(obstacles[i]))if obstacles else None
    obstacle=obstacles[nearest]if nearest is not None else None
    world,memory,_=simulation.replay_boundary(fixture,[])
    observations=[];ever=0;first_seen=None;matches=0
    for i in range(end+1):
        o=simulation.observation(world);memory=frontend.update_memory(memory,dict(observation=o))
        matches+=digest(o)==prior['records'][i]['public_sha256']
        hits=[]
        for robot in o['robots']:
            s=robot['scan'];ox,oy=s['origin']
            for ok,kind,a,d in zip(s['valid'],s['category'],s['ray_angles'],s['distances']):
                if ok and kind==2:
                    p=[ox+d*math.cos(a),oy+d*math.sin(a)]
                    if obstacle is not None and obstacle.boundary.distance(Point(p))<.004:hits.append(p)
        if hits:
            ever+=len(hits)
            if first_seen is None:first_seen=o['tick']
        kept=sum(obstacle.boundary.distance(Point(p))<.004 for p in memory['static'])if obstacle is not None else 0
        observations.append(dict(tick=o['tick'],colliding_geometry_hits=len(hits),retained_hits=kept,
                                 public=o,command=prior['records'][i]['command']))
        actual=simulation.measured_step(world,prior['records'][i]['command'])
    last=observations[-1]
    result=dict(room=room,first_contact_tick=prior['records'][end]['tick'],replay_ticks=world.env.total_periods,
        public_hash_matches=matches,boundaries=end+1,reproduced_contact=actual['object_static'],
        audited_obstacle_index=nearest,endpoint_distance_to_obstacle_m=body.distance(obstacle)if obstacle is not None else None,
        ever_observed_hit_count=ever,first_observed_tick=first_seen,current_hit_count=last['colliding_geometry_hits'],
        retained_hit_count=last['retained_hits'],
        classification='never_seen_in_boundary_scans'if not ever else 'seen_but_not_retained'if not last['retained_hits']else 'seen_and_retained_but_summary_or_gate_failed')
    save(OUT/'historical'/ (room+'.json.gz'),dict(summary=result,observations=observations))
    return result


def main():
    from tools.cap_symbolic_general_evaluate import verify_freeze
    verify_freeze()
    rooms=[r['room']for r in read(OLD/'metrics.json')['tasks']['2']['rows']if r['object_static_commands']]
    with ProcessPoolExecutor(max_workers=4)as pool:rows=list(pool.map(worker,rooms))
    from collections import Counter
    packet=dict(at=time.time(),rooms=len(rows),ticks=sum(r['replay_ticks']for r in rows),
        classifications=dict(Counter(r['classification']for r in rows)),rows=rows,
        scope='observations at original command boundaries; geometry identity assigned offline from contact endpoint; intermediate per-tick scans were not policy inputs')
    save(OUT/'historical_audit.json',packet)
    print({k:v for k,v in packet.items()if k!='rows'})


if __name__=='__main__':main()
