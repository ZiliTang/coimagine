"""Project the planar public lidar packets into a scene-local static surface map."""
import math

from coimagine.random_start.lidar import CATEGORY_STATIC


def update_static_map(obstacle_map, observation):
    for robot in observation['robots']:
        scan = robot['scan']
        ox, oy = scan['origin']
        hits = []
        for ok, kind, angle, distance in zip(scan['valid'], scan['category'], scan['ray_angles'],
                                              scan['distances'], strict=True):
            if ok and kind == CATEGORY_STATIC:
                hits.append((ox + distance * math.cos(angle), oy + distance * math.sin(angle)))
            else:
                hits.append(None)
        obstacle_map.observe_hits(hits, observation['tick'], circular=True)
