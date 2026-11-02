"""Bounded, replayable effect-PUSH testbench on the existing 2D environment.

Run: .venv/bin/python experiments/option_testbench/effect_push_eval.py
Outputs a frozen manifest, raw logs, metrics.json, SUMMARY.md, and offline index.html.
Each branch runs in its own process with a wall timeout. No RL/model training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coimagine.continuous2d.scenes import Box, Pose
from coimagine.random_start import env as E, layout, spec
from coimagine.random_start.effect_push import (
    ACTIONS, DISTANCES, DIRECTION_NAMES, DIRECTIONS, VERSION, WAIT_INDEX,
    ACTIONS_REPOSITION_V2, REPOSITION_V2_VERSION, REPOSITION_DISTANCES_V2,
    EffectAction, EffectPushConfig, EffectPushSkill, action_mask, make_effect_skill)


def make_case(name, direction=2, distance=.3, offset=0., tilt=0., rotation=0, angle=.2,
              negative=None, cooperative=False):
    d = DIRECTIONS[direction]
    t = (-d[1], d[0])
    phi = direction * math.pi / 4 + tilt
    n = (math.cos(phi), math.sin(phi))
    centre = (2.5, 2.)
    robot = Pose(centre[0] - .405 * n[0] + offset * t[0],
                 centre[1] - .405 * n[1] + offset * t[1], phi)
    peer = Pose(4.4, .5, 0.)
    object_pose = Pose(*centre, phi - math.pi / 2)
    if rotation:
        action = next(a for a in ACTIONS if isinstance(a, EffectAction)
                      and a.effect_kind == "rotate" and a.rotation_sign == rotation
                      and abs(a.rotation_angle - angle) < 1e-9)
    else:
        action = next(a for a in ACTIONS if isinstance(a, EffectAction)
                      and a.type_index == direction and abs(a.distance - distance) < 1e-9)
    peer_action = WAIT_INDEX
    if negative == "opposite":
        action = ACTIONS[((direction + 4) % 8) * 5 + 2]
    elif negative == "tangent":
        action = ACTIONS[((direction + 2) % 8) * 5 + 2]
    elif negative == "invisible":
        robot = Pose(.5, .5, phi)
    elif negative == "occluded":
        robot, peer = Pose(2.5, 1.0, math.pi / 2), Pose(2.5, 1.5, math.pi / 2)
    if cooperative:
        robot = replace(robot, x=robot.x - .25 * t[0], y=robot.y - .25 * t[1])
        peer = Pose(centre[0] - .405 * n[0] + .25 * t[0],
                    centre[1] - .405 * n[1] + .25 * t[1], phi)
        peer_action = action.index
    instance = layout.InstanceSpec(
        name=name, map="L0", split="dev", index=0, seed=0, arena=(5., 4.),
        obstacles=(), object_pose=object_pose, robot_poses=(robot, peer),
        goal_centre=(4., 3.3), goal_radius=spec.GOAL_RADIUS, env_version=spec.ENV_VERSION)
    return {"name": name, "instance": instance.as_dict(), "actions": [action.index, peer_action],
            "negative_control": negative, "cooperative": cooperative}


def cases():
    out = [make_case(f"direction_{name}", direction=i) for i, name in enumerate(DIRECTION_NAMES)]
    out += [make_case(f"scale_{d:.2f}", distance=d) for d in DISTANCES]
    out += [make_case(f"offset_{x:+.2f}", offset=x) for x in (-.03, .03, .18, .32, .34, .36)]
    out += [make_case(f"tilt_{a:+.0f}", tilt=math.radians(a)) for a in (-8, 8)]
    out += [make_case(f"rotate_{'CCW' if sign == 1 else 'CW'}_{angle:.1f}", rotation=sign,
                      angle=angle) for sign in (1, -1) for angle in (.1, .3)]
    out += [make_case(f"negative_{name}", negative=name)
            for name in ("opposite", "tangent", "invisible", "occluded")]
    out += [make_case("cooperative_N", cooperative=True)]
    short = make_case("short_face_E", direction=0)
    short["instance"]["object"]["angle"] = 0.0
    short["instance"]["robots"][0] = asdict(Pose(1.72, 2., 0.))
    out.append(short)
    short_rotate = json.loads(json.dumps(short))
    short_rotate["name"] = short_rotate["instance"]["name"] = "short_rotate_CCW"
    short_rotate["actions"][0] = 40
    out.append(short_rotate)
    couple = make_case("rotation_couple_CCW", rotation=1, angle=.1)
    couple["instance"]["robots"] = [asdict(Pose(2.72, 1.595, math.pi / 2)),
                                    asdict(Pose(2.28, 2.405, -math.pi / 2))]
    couple["actions"] = [40, 40]
    couple["cooperative"] = True
    out.append(couple)
    return out


def endpoint_cases():
    panel = cases()
    corner = make_case("two_faces_N")
    corner["instance"]["robots"][0] = asdict(Pose(3.3, 1.6, math.atan2(.4, -.8)))
    panel.append(corner)
    panel.append(make_case("cooperative_tilt_+8", cooperative=True, tilt=math.radians(8)))
    return panel


def approach_cases():
    """Frozen small perturbations, paired within each unchanged task request."""
    panel = endpoint_cases()
    by_name = {c["name"]: c for c in panel}
    for base in ("two_faces_N", "tilt_+8", "cooperative_N", "rotate_CCW_0.1"):
        for dx in (-.01, .01):
            case = json.loads(json.dumps(by_name[base]))
            case["name"] = case["instance"]["name"] = f"{base}_dx{dx:+.2f}"
            case["instance"]["robots"][0]["x"] += dx
            case["perturbation"] = {"base": base, "robot0_dx_m": dx}
            panel.append(case)
    for yaw in (-1, 1):
        case = json.loads(json.dumps(by_name["two_faces_N"]))
        case["name"] = case["instance"]["name"] = f"two_faces_N_yaw{yaw:+d}"
        case["instance"]["robots"][0]["angle"] += math.radians(yaw)
        case["perturbation"] = {"base": "two_faces_N", "robot0_dyaw_deg": yaw}
        panel.append(case)
    return panel


def joint_cases():
    panel = []
    for a, b in ((.1, .1), (.1, .2), (.1, .5), (.5, .1), (.3, .3), (.3, .5), (.5, .5)):
        c = make_case(f"same_N_{a:.1f}_{b:.1f}", cooperative=True)
        c["actions"] = [10 + DISTANCES.index(a), 10 + DISTANCES.index(b)]
        panel.append(c)
    for a, b in ((.5, .1), (.1, .5), (.3, .3)):
        c = make_case(f"N_E_{a:.1f}_{b:.1f}")
        c["instance"]["robots"][1] = asdict(Pose(1.72, 2., 0.))
        c["actions"] = [10 + DISTANCES.index(a), DISTANCES.index(b)]
        panel.append(c)
    for a, b in ((.5, .1), (.3, .3), (.5, .5)):
        c = make_case(f"N_S_{a:.1f}_{b:.1f}")
        c["instance"]["robots"][1] = asdict(Pose(2.5, 2.405, -math.pi / 2))
        c["actions"] = [10 + DISTANCES.index(a), 30 + DISTANCES.index(b)]
        panel.append(c)
    c = make_case("same_face_N_NW", cooperative=True, tilt=math.pi / 8)
    c["actions"] = [14, 15]
    panel.append(c)
    c = make_case("peer_direction_unavailable", cooperative=True)
    c["actions"] = [14, 30]
    c["negative_control"] = "peer_unavailable"
    panel.append(c)
    for base in ("same_N_0.1_0.5", "N_E_0.5_0.1"):
        original = next(c for c in panel if c["name"] == base)
        for dx in (-.01, .01):
            c = json.loads(json.dumps(original))
            c["name"] = c["instance"]["name"] = f"{base}_dx{dx:+.2f}"
            c["instance"]["robots"][0]["x"] += dx
            panel.append(c)
    # Unit distinction: rotation pairs retain independent angle-effect semantics.
    c = next(c for c in cases() if c["name"] == "rotation_couple_CCW")
    c["actions"] = [40, 44]
    c["name"] = c["instance"]["name"] = "rotation_mixed_angles"
    panel.append(c)
    return panel


def allocation_cases():
    panel = joint_cases()
    original = next(c for c in panel if c["name"] == "same_face_N_NW")
    equal = json.loads(json.dumps(original))
    equal["name"] = equal["instance"]["name"] = "same_face_N_NW_equal_gap"
    p = equal["instance"]["object"]
    n = (-math.sin(p["angle"]), math.cos(p["angle"]))
    t = (-n[1], n[0])
    for i, offset in enumerate((-.25, .25)):
        r = equal["instance"]["robots"][i]
        r["x"], r["y"] = p["x"] - .405 * n[0] + offset * t[0], p["y"] - .405 * n[1] + offset * t[1]
    panel.append(equal)
    for dx in (-.01, .01):
        c = json.loads(json.dumps(equal))
        c["name"] = c["instance"]["name"] = f"same_face_equal_dx{dx:+.2f}"
        c["instance"]["robots"][0]["x"] += dx
        panel.append(c)
    narrow = make_case("short_face_two_robots", direction=0, cooperative=True)
    narrow["instance"]["object"]["angle"] = 0.
    narrow["instance"]["robots"] = [asdict(Pose(1.72, 1.8, 0.)), asdict(Pose(1.72, 2.2, 0.))]
    panel.append(narrow)
    return panel


def normal_travel_cases():
    panel = [c for c in cases() if c["name"].startswith(("direction_", "scale_", "tilt_"))
             or c["name"] in ("offset_-0.03", "offset_+0.03", "offset_+0.32", "rotate_CCW_0.1", "rotate_CW_0.1")]
    names = {"same_N_0.1_0.1", "same_N_0.3_0.3", "same_N_0.5_0.5", "same_N_0.1_0.5", "same_N_0.5_0.1",
             "N_E_0.5_0.1", "N_E_0.1_0.5", "N_S_0.5_0.5", "N_S_0.5_0.1", "same_face_N_NW",
             "same_face_N_NW_equal_gap", "peer_direction_unavailable", "short_face_two_robots", "rotation_mixed_angles"}
    panel += [c for c in allocation_cases() if c["name"] in names]
    return panel


def comparison_config(comparison, variant, overrides):
    values = dict(overrides)
    if comparison == "reposition_peer_detour":
        values.update(endpoint_tracking=True, fast_approach=True, same_face_allocation=True,
            reposition_boundary="rounded_scan", reposition_peer_detour="outer_arc" if variant else "off",
            translation_completion="normal_travel", post_push="contact_follow",contact_recovery="joint_approach")
        return EffectPushConfig(feedback=True, **values)
    if comparison == "reposition_scales_v2":
        values.update(endpoint_tracking=True,fast_approach=True,same_face_allocation=True,
            reposition_boundary="rounded_scan",translation_completion="normal_travel",
            post_push="contact_follow",contact_recovery="joint_approach")
        return EffectPushConfig(feedback=True,**values)
    if comparison == "reposition_corners":
        values.update(endpoint_tracking=True, fast_approach=True, same_face_allocation=True,
                      translation_completion="normal_travel", post_push="contact_follow", contact_recovery="joint_approach",
                      reposition_boundary="rounded_scan" if variant else "legacy_face")
        return EffectPushConfig(feedback=True,**values)
    if comparison == "option_combinations":
        values.update(endpoint_tracking=True, fast_approach=True, same_face_allocation=True,
                      translation_completion="normal_travel", post_push="contact_follow",
                      contact_recovery="joint_approach")
        return EffectPushConfig(feedback=True, **values)
    if comparison == "joint_recovery":
        values.update(endpoint_tracking=True, fast_approach=True, same_face_allocation=True,
                      joint_execution="proportional_normal_v1", translation_completion="normal_travel",
                      post_push="contact_follow", contact_recovery="joint_approach" if variant else "abort")
        return EffectPushConfig(feedback=True, **values)
    if comparison == "post_follow":
        values.update(endpoint_tracking=True, fast_approach=True, same_face_allocation=True,
                      joint_execution="proportional_normal_v1", translation_completion="normal_travel",
                      post_push="contact_follow" if variant else "hold_brake")
        return EffectPushConfig(feedback=True, **values)
    if comparison == "normal_travel":
        values.update(endpoint_tracking=True, fast_approach=True, same_face_allocation=True,
                      joint_execution="proportional_normal_v1",
                      translation_completion="normal_travel" if variant else "normal_reference")
        return EffectPushConfig(feedback=True, **values)
    if comparison == "contact_allocation":
        values.update(endpoint_tracking=True, fast_approach=True, same_face_allocation=variant,
                      joint_execution="proportional_normal_v1", translation_completion="normal_reference")
        return EffectPushConfig(feedback=True, **values)
    if comparison == "joint_execution":
        values.update(endpoint_tracking=True, fast_approach=True,
                      joint_execution="proportional_normal_v1" if variant else "legacy_v3",
                      translation_completion="normal_reference" if variant else "object_effect")
        return EffectPushConfig(feedback=True, **values)
    values.setdefault("joint_execution", "legacy_v3")
    values.setdefault("translation_completion", "object_effect")
    if comparison == "fast_approach":
        values.update(endpoint_tracking=True, fast_approach=variant)
        return EffectPushConfig(feedback=True, **values)
    if comparison == "endpoint_tracking":
        values.update(endpoint_tracking=variant)
        return EffectPushConfig(feedback=True, **values)
    return EffectPushConfig(feedback=variant, **values)


def post_follow_cases():
    """One frozen lifecycle comparison, including failures and +/-1 cm inputs."""
    panel = allocation_cases()
    omit = {"N_S_0.3_0.3"}
    panel = [c for c in panel if c["name"] not in omit]
    panel.append(next(c for c in cases() if c["name"] == "scale_0.50"))
    for c in panel:
        if "_dx" in c["name"]:
            base, dx = c["name"].rsplit("_dx", 1)
            c["perturbation"] = {"base": "same_face_N_NW_equal_gap" if base == "same_face_equal" else base,
                                 "robot0_dx_m": float(dx)}
    return panel


def recovery_cases():
    panel = post_follow_cases()
    # Declared diagnostic disturbances, identical inputs across both variants.
    # These are motor-force pulses, never teleports or changes to physics/rewards.
    for base, side, start, force in (("same_N_0.1_0.5", 0, 3.95, [0., -4.]),
                                     ("N_E_0.5_0.1", 1, 5., [-4., 0.])):
        c = json.loads(json.dumps(next(c for c in panel if c["name"] == base)))
        c["name"] = c["instance"]["name"] = base + "_outward_pulse"
        c["disturbance"] = {"robot": side, "start": start, "end": start + .25, "force_xy": force}
        c["diagnostic_only"] = True
        panel.append(c)
    return panel


def reposition_cases():
    """Mixed-option coverage, one current configuration; no fake off/on ablation."""
    def action(kind, direction=None, distance=None, duration=None):
        return next(a.index for a in ACTIONS if a.kind == kind
                    and (direction is None or a.direction == direction)
                    and (distance is None or abs(a.distance-distance) < 1e-9)
                    and (duration is None or abs(a.duration-duration) < 1e-9))
    panel = []
    for direction in ("CW", "CCW"):
        for distance in spec.REPOSITION_DISTANCES:
            c = make_case(f"rp_{direction}_solo_{distance:.2f}")
            c["actions"] = [action("reposition", direction, distance), WAIT_INDEX]
            panel.append(c)
        c = make_case(f"rp_{direction}_corner_1.00")
        c["instance"]["robots"][0] = asdict(Pose(2.05 if direction=="CW" else 2.95, 1.575, -math.pi/2))
        c["actions"] = [action("reposition", direction, 1.), WAIT_INDEX]
        panel.append(c)
        for arrangement in ("opposite", "same"):
            c = make_case(f"rp_{direction}_push_{arrangement}_0.50_0.30")
            if arrangement == "opposite":
                c["instance"]["robots"][1] = asdict(Pose(2.5, 2.405, -math.pi/2))
                push = 32  # S, 0.3 m
            else:
                c["instance"]["robots"] = [asdict(Pose(2.25, 1.595, math.pi/2)),
                                             asdict(Pose(2.75, 1.595, math.pi/2))]
                push = 12  # N, 0.3 m
            c["actions"] = [action("reposition", direction, .5), push]
            panel.append(c)
        for duration in (2., 6.):
            c = make_case(f"rp_{direction}_follow_0.50_{duration:.0f}s")
            c["instance"]["robots"][1] = asdict(Pose(2.5, 2.405, -math.pi/2))
            c["actions"] = [action("reposition", direction, .5), action("follow", duration=duration)]
            panel.append(c)
    for first, second in (("CW", "CW"), ("CCW", "CCW"), ("CW", "CCW")):
        c = make_case(f"rp_pair_{first}_{second}_0.50")
        c["instance"]["robots"][1] = asdict(Pose(2.5, 2.405, -math.pi/2))
        c["actions"] = [action("reposition", first, .5), action("reposition", second, .5)]
        panel.append(c)
    for base in ("rp_CW_solo_0.50", "rp_CCW_push_opposite_0.50_0.30"):
        c = json.loads(json.dumps(next(c for c in panel if c["name"] == base)))
        c["name"] = c["instance"]["name"] = base + "_swapped"
        c["actions"].reverse();c["instance"]["robots"].reverse()
        panel.append(c)
    c = make_case("rp_CW_invisible")
    c["instance"]["robots"][0] = asdict(Pose(.5, .5, 0.))
    c["actions"] = [action("reposition", "CW", .5), WAIT_INDEX]
    c["negative_control"] = "invisible"
    panel.append(c)
    return panel


def reposition_corner_cases():
    panel = []
    for original in reposition_cases():
        c = json.loads(json.dumps(original));c["name"] = c["instance"]["name"] = "arc_"+c["name"]
        panel.append(c)
    for direction in ("CW","CCW"):
        base = next(c for c in panel if c["name"]==f"arc_rp_{direction}_corner_1.00")
        for dx in (-.01,.01):
            c = json.loads(json.dumps(base));c["name"] = c["instance"]["name"] = base["name"]+f"_dx{dx:+.2f}"
            c["instance"]["robots"][0]["x"] += dx
            c["perturbation"]={"base":base["name"],"robot0_dx_m":dx};panel.append(c)
        c = json.loads(json.dumps(base));c["name"] = c["instance"]["name"] = base["name"]+"_peer_shadow"
        robot=c["instance"]["robots"][0]
        c["instance"]["robots"][1] = asdict(Pose(robot["x"]+(-.41 if direction=="CW" else .41),robot["y"],0.))
        c["negative_control"]="peer_shadow";panel.append(c)
    return panel


def reposition_scale_cases():
    def lookup(kind,direction=None,distance=None,duration=None):
        return next(a.index for a in ACTIONS_REPOSITION_V2 if a.kind==kind
                    and (direction is None or a.direction==direction)
                    and (distance is None or abs(a.distance-distance)<1e-9)
                    and (duration is None or abs(a.duration-duration)<1e-9))
    wait=lookup('wait',duration=2.);panel=[]
    for direction in ('CW','CCW'):
        for distance in REPOSITION_DISTANCES_V2:
            c=make_case(f'scale2_rp_{direction}_solo_{distance:.2f}')
            c['actions']=[lookup('reposition',direction,distance),wait];panel.append(c)
        c=make_case(f'scale2_rp_{direction}_corner_1.20')
        c['instance']['robots'][0]=asdict(Pose(2.05 if direction=='CW' else 2.95,1.575,-math.pi/2))
        c['actions']=[lookup('reposition',direction,1.2),wait];panel.append(c)
        for arrangement,distance in (('opposite',1.8),('same',.8)):
            c=make_case(f'scale2_rp_{direction}_push_{arrangement}_{distance:.2f}')
            if arrangement=='opposite':
                c['instance']['robots'][1]=asdict(Pose(2.5,2.405,-math.pi/2));push=32
            else:
                c['instance']['robots']=[asdict(Pose(2.25,1.595,math.pi/2)),asdict(Pose(2.75,1.595,math.pi/2))];push=12
            c['actions']=[lookup('reposition',direction,distance),push];panel.append(c)
        c=make_case(f'scale2_rp_{direction}_follow_1.20_6s')
        c['instance']['robots'][1]=asdict(Pose(2.5,2.405,-math.pi/2))
        c['actions']=[lookup('reposition',direction,1.2),lookup('follow',duration=6.)];panel.append(c)
    for second in ('CW','CCW'):
        c=make_case(f'scale2_rp_pair_CW_{second}_0.80')
        c['instance']['robots'][1]=asdict(Pose(2.5,2.405,-math.pi/2))
        c['actions']=[lookup('reposition','CW',.8),lookup('reposition',second,.8)];panel.append(c)
    c=json.loads(json.dumps(next(c for c in panel if c['name']=='scale2_rp_CW_solo_1.80')))
    c['name']=c['instance']['name']='scale2_rp_CW_solo_1.80_swapped'
    c['actions'].reverse();c['instance']['robots'].reverse();panel.append(c)
    c=make_case('scale2_rp_CW_invisible');c['instance']['robots'][0]=asdict(Pose(.5,.5,0.))
    c['actions']=[lookup('reposition','CW',.8),wait];c['negative_control']='invisible';panel.append(c)
    for direction in ('CW','CCW'):
        base=next(c for c in panel if c['name']==f'scale2_rp_{direction}_solo_1.80')
        for dx in (-.01,.01):
            c=json.loads(json.dumps(base));c['name']=c['instance']['name']=base['name']+f'_dx{dx:+.2f}'
            c['instance']['robots'][0]['x']+=dx;panel.append(c)
    for c in panel:c['action_version']=REPOSITION_V2_VERSION
    return panel


def reposition_peer_cases():
    """Matched obstacles, larger/smaller requests, swaps, occlusion and no-obstacle controls."""
    source = reposition_scale_cases()
    names = [f"scale2_rp_{d}_push_{side}_{distance:.2f}"
             for d in ("CW", "CCW") for side,distance in (("same",.8),("opposite",1.8))]
    names += [f"scale2_rp_{d}_solo_1.80" for d in ("CW", "CCW")]
    names += ["scale2_rp_pair_CW_CW_0.80", "scale2_rp_pair_CW_CCW_0.80", "scale2_rp_CW_invisible"]
    panel = [json.loads(json.dumps(c)) for c in source if c['name'] in names]
    lookup = lambda d: next(a.index for a in ACTIONS_REPOSITION_V2
                            if a.kind=='reposition' and a.direction==d and a.distance==1.8)
    wait = next(a.index for a in ACTIONS_REPOSITION_V2 if a.kind=='wait' and a.duration==2.)
    for direction in ('CW','CCW'):
        c = json.loads(json.dumps(next(c for c in source if c['name']==f'scale2_rp_{direction}_push_same_0.80')))
        # Mirror the CW blocked approach: CW goes left, CCW goes right.
        if direction == 'CW':
            c['instance']['robots'].reverse()
        c['name']=f'scale2_rp_{direction}_push_same_1.80';c['actions'][0]=lookup(direction);panel.append(c)
        c=json.loads(json.dumps(c));c['name']=f'scale2_rp_{direction}_wait_same_1.80';c['actions'][1]=wait;panel.append(c)
    base=next(c for c in panel if c['name']=='scale2_rp_CCW_push_same_1.80')
    for dx in (-.01,.01):
        c=json.loads(json.dumps(base));c['name']+=f'_dx{dx:+.2f}'
        c['instance']['robots'][0]['x']+=dx;panel.append(c)
    for direction, sign in (('CW',-1),('CCW',1)):
        for gap in (.26,.78):
            c=json.loads(json.dumps(next(c for c in source if c['name']==f'scale2_rp_{direction}_solo_1.80')))
            c['name']=f'scale2_rp_{direction}_static_gap_{gap:.2f}'
            c['instance']['obstacles']=[asdict(Box(2.5+sign*(.5+gap+.10),2.,.10,.9))]
            c['negative_control']='static_narrow' if gap<.36 else None
            panel.append(c)
        c=json.loads(json.dumps(next(c for c in panel if c['name']==f'scale2_rp_{direction}_push_same_1.80')))
        c['name']=f'scale2_rp_{direction}_peer_static_blocked'
        c['instance']['obstacles']=[asdict(Box(2.5,1.10,1.1,.10))]
        c['negative_control']='peer_static_narrow';panel.append(c)
    for c in panel:
        c['name']=c['name'].replace('scale2_','detour_');c['instance']['name']=c['name']
    return panel


def record_actions(record):
    if 'requested_actions' in record:
        result=[]
        for params in record['requested_actions']:
            params=dict(params)
            if 'effect_kind' in params:
                params['effect_direction']=tuple(params['effect_direction']);result.append(EffectAction(**params))
            else:result.append(spec.RobotAction(**params))
        return result
    table=ACTIONS_REPOSITION_V2 if record.get('case',{}).get('action_version')==REPOSITION_V2_VERSION else ACTIONS
    return [table[i] for i in record['case']['actions']]


def hydrate_actions(record,manifest):
    if record['metric']['status']!='completed':return
    requested=[manifest['actions'][i] for i in record['case']['actions']]
    if 'requested_actions' in record and json.dumps(record['requested_actions'],sort_keys=True)!=json.dumps(requested,sort_keys=True):
        raise ValueError('recorded requested actions disagree with frozen source menu')
    expected=hashlib.sha256(json.dumps(manifest['actions'],sort_keys=True).encode()).hexdigest()
    if record['boundary']['action_codebook_hash']!=expected:
        raise ValueError('source action codebook does not match its frozen manifest')
    record['requested_actions']=requested


class ForcePulseSkill(EffectPushSkill):
    """Testbench-only declared additive motor disturbance; own state feedback stays real."""
    def __init__(self, action, force_limit, config, pulse):
        super().__init__(action, force_limit, config)
        self.pulse = pulse

    def step(self, observation, dt):
        force, info = super().step(observation, dt)
        p = self.pulse
        if observation.robot_id == p["robot"] and p["start"] <= observation.time < p["end"]:
            applied = tuple(force[j] + p["force_xy"][j] for j in (0, 1))
            length = math.hypot(*applied)
            if length > self.force_limit:
                applied = tuple(v * self.force_limit / length for v in applied)
            info["command_disturbance"] = {"nominal_force": force, "added_force": p["force_xy"]}
            force = applied
            info.update(force=force, force_norm=math.hypot(*force))
        return force, info


def instance_from_dict(d):
    return layout.InstanceSpec(
        name=d["name"], map=d["map"], split=d["split"], index=d["index"], seed=d["seed"],
        arena=tuple(d["arena"]), obstacles=tuple(Box(**p) for p in d.get("obstacles", ())), object_pose=Pose(**d["object"]),
        robot_poses=tuple(Pose(**p) for p in d["robots"]),
        goal_centre=tuple(d["goal_centre"]), goal_radius=d["goal_radius"], env_version=d["env_version"])


def run_one(case, feedback, overrides=None, comparison="contact_feedback"):
    started = time.monotonic()
    config = EffectPushConfig(feedback=feedback, **(overrides or {}))
    table=ACTIONS_REPOSITION_V2 if case.get('action_version')==REPOSITION_V2_VERSION else ACTIONS
    action_version=REPOSITION_V2_VERSION if table is ACTIONS_REPOSITION_V2 else VERSION
    instance = instance_from_dict(case["instance"])
    factory = (lambda a, f: ForcePulseSkill(a, f, config, case["disturbance"])) if case.get("disturbance") else partial(make_effect_skill, config=config)
    world = E.RandomStartEnv(instance, skill_factory=factory,
                             action_table=table, action_version=action_version)
    initial_observation = world.boundary_observations()[0]
    mask = action_mask(initial_observation, config,actions=table)
    initial = world.object_state()
    boundary = world.joint_step(*case["actions"], record=True)
    final = world.object_state()
    action = table[case["actions"][0]]
    is_effect = isinstance(action, EffectAction)
    active = [r for r in world.rows if r["r0"].get("kind") != "hold"]
    effect = active[-1]["r0"].get("effect") or {}
    dx, dy = final.x - initial.x, final.y - initial.y
    rotation = final.angle - initial.angle
    if is_effect and action.effect_kind == "translate":
        actual = dx * action.effect_direction[0] + dy * action.effect_direction[1]
        sideways = -dx * action.effect_direction[1] + dy * action.effect_direction[0]
        fulfilled = (action.distance - .02 <= actual <= action.distance + .04
                     and abs(sideways) <= .04 and abs(rotation) <= math.radians(10))
    elif is_effect:
        actual = action.rotation_sign * rotation
        sideways = None
        fulfilled = action.rotation_angle - .03 <= actual <= action.rotation_angle + .08
    else:
        actual = sideways = None  # non-push progress is not an object-effect target
        fulfilled = None
    target = action.effect_target if is_effect else action.distance or action.duration
    # This is offline effect accuracy, kept separate from the controller's completion claim.
    metric = {
        "case": case["name"], "feedback": feedback, "status": "completed",
        "available": bool(mask[action.index]), "negative_control": case["negative_control"],
        "action": action.label, "exit": boundary["skill_ends"][0]["exit_reason"],
        "target": target, "unit": effect.get("unit", "s" if action.duration else "m"),
        "estimated_effect": effect.get("progress"), "actual_effect": actual,
        "error": actual - target if actual is not None else None, "sideways": sideways,
        "object_displacement": math.hypot(dx, dy), "rotation_deg": math.degrees(rotation),
        "fulfilled": bool(fulfilled) if fulfilled is not None else None, "seconds": boundary["periods"] * spec.CONTROL_DT,
        "reward": boundary["reward"]["total"], "max_penetration": max(r["penetration"] for r in world.rows),
        "contact_fraction": sum(r["r0"]["contacting_object"] for r in active) / len(active),
        "wall_seconds": time.monotonic() - started,
        "control_uses_truth": False,
        "comparison": comparison,
        "variant": (config.reposition_peer_detour == "outer_arc" if comparison == "reposition_peer_detour" else
                    config.reposition_boundary == "rounded_scan" if comparison == "reposition_corners" else
                    config.contact_recovery == "joint_approach" if comparison == "joint_recovery" else
                    config.post_push == "contact_follow" if comparison == "post_follow" else
                    config.translation_completion == "normal_travel" if comparison == "normal_travel" else
                    config.same_face_allocation if comparison == "contact_allocation" else
                    config.joint_execution == "proportional_normal_v1" if comparison == "joint_execution" else
                    config.fast_approach if comparison == "fast_approach" else
                    config.endpoint_tracking if comparison == "endpoint_tracking" else feedback),
        "endpoint_tracking": config.endpoint_tracking,
        "fast_approach": config.fast_approach,
        "preparation_seconds": next((r["time"] for r in active if r["r0"]["stage"] == "push"),
                                    boundary["periods"] * spec.CONTROL_DT),
        "first_contact_seconds": next((r["time"] for r in active if r["r0"]["contacting_object"]), None),
        "actual_displacement_xy": [dx, dy],
        "execution_protocol": boundary["execution"]["protocol"],
        "scale_semantics": effect.get("scale_semantics", "signed_boundary_path_m" if action.kind=="reposition" else "duration"),
        "normal_travel": boundary["execution"]["normal_travel"],
        "follow_normal_travel": boundary["execution"]["follow_normal_travel"],
        "drive_complete_times": boundary["execution"]["drive_complete_times"],
    }
    if boundary["execution"]["protocol"] == "proportional_normal_v1" or effect.get("scale_semantics") in ("normal_reference_m", "measured_normal_travel_m"):
        metric["fulfilled"] = None  # a later joint endpoint cannot score an earlier local exit
    metric["exits"] = [boundary["skill_ends"][i]["exit_reason"] for i in (0, 1)]
    metric["normal_reference_budget"] = boundary["execution"]["normal_reference_budget"]
    outcomes = []
    for i, action_id in enumerate(case["actions"]):
        a = table[action_id]
        infos = [row[f"r{i}"] for row in world.rows if row[f"r{i}"]["kind"] != "hold"]
        last = infos[-1]
        progress = (boundary["execution"]["normal_travel"][i] if isinstance(a, EffectAction) and a.effect_kind=="translate"
                    and config.translation_completion=="normal_travel" else (last.get("effect") or {}).get("progress")
                    if isinstance(a, EffectAction) else last.get("signed_progress") if a.kind=="reposition"
                    else last.get("follow_time") if a.kind=="follow" else last.get("progress"))
        outcomes.append({"action": a.label, "kind": a.kind,
                         "target": a.effect_target if isinstance(a, EffectAction) else a.distance or a.duration,
                         "unit": "rad" if isinstance(a, EffectAction) and a.effect_kind=="rotate" else "s" if a.duration else "m",
                         "recorded_progress": progress, "exit": boundary["skill_ends"][i]["exit_reason"],
                         "face_switches": last.get("face_switches")})
    metric["option_outcomes"] = outcomes
    metric["termination"] = boundary["termination"]
    return {"case": case, "config": asdict(config), "metric": metric,
            "requested_actions":[asdict(table[i]) for i in case['actions']],
            "availability_mask": mask.tolist(), "boundary": boundary, "rows": world.rows,
            "initial_object": list(initial.as_tuple())}


def replay_robot_options(record):
    """Both requested contracts, without reusing robot 0's labels or mask for robot 1."""
    options = []
    for i, action in enumerate(record_actions(record)):
        effects = [row[f"r{i}"]["effect"] for row in record["rows"] if row[f"r{i}"].get("effect")]
        effect = effects[-1] if effects else {}
        kind = getattr(action, "effect_kind", action.kind)
        options.append({"label": action.label, "kind": kind,
                        "target": action.effect_target if isinstance(action, EffectAction) else action.distance or action.duration,
                        "unit": "rad" if kind == "rotate" else "s" if action.duration else "m",
                        "direction": list(getattr(action, "effect_direction", (0., 0.))),
                        "rotation_sign": getattr(action, "rotation_sign", 0),
                        "boundary_direction": action.direction,
                        "scale_semantics": effect.get("scale_semantics", "signed_boundary_path_m" if action.kind=="reposition"
                                                      else "follow_duration" if action.kind=="follow" else "robot_travel" if action.distance else "duration")})
    return options


def replay_frames(record):
    frames = []
    a = record_actions(record)[0]
    initial = record["initial_object"]
    options = replay_robot_options(record)
    last_estimate, last_estimate_time = [None, None], [None, None]
    last_generic_progress = [None, None]
    last_switches = [None, None]
    for row in record["rows"]:
        obj = row["true"]["object"]
        effect = row["r0"].get("effect") or {}
        actual = ((obj[0] - initial[0]) * a.effect_direction[0]
                  + (obj[1] - initial[1]) * a.effect_direction[1]
                  if isinstance(a, EffectAction) and a.effect_kind == "translate" else a.rotation_sign * (obj[2] - initial[2])
                  if isinstance(a, EffectAction) else None)
        frames.append({"t": row["time"], "o": obj[:3],
                       "r": [robot[:3] for robot in row["true"]["robots"]],
                       "stage": row["r0"]["stage"], "contact": row["r0"]["contacting_object"],
                       "other_stage": row["r1"]["stage"], "other_contact": row["r1"]["contacting_object"],
                       "effect": effect, "actual": actual,
                       "other_effect": row["r1"].get("effect"),
                       "execution": row.get("execution")})
        frames[-1]["normal_travel"] = row["r0"].get("completion_progress")
        sides = []
        for i, option in enumerate(options):
            info = row[f"r{i}"]
            effect = info.get("effect") or {}
            execution = row.get("execution") or {}
            def persistent(name, fallback=None):
                values = execution.get(name)
                return values[i] if values is not None and values[i] is not None else effect.get(name, fallback)
            if effect:
                last_estimate[i] = effect.get("progress")
                last_estimate_time[i] = effect.get("observation_time")
            if info.get("face_switches") is not None:
                last_switches[i] = info["face_switches"]
            actual = (((obj[0]-initial[0])*option["direction"][0] + (obj[1]-initial[1])*option["direction"][1])
                      if option["kind"] == "translate" else option["rotation_sign"]*(obj[2]-initial[2])
                      if option["kind"] == "rotate" else None)
            if option["scale_semantics"] == "measured_normal_travel_m":
                progress = info.get("completion_progress")
                if progress is None:
                    progress = persistent("normal_travel")
            elif option["scale_semantics"] == "normal_reference_m":
                progress = persistent("normal_reference_budget")
            elif option["kind"] in ("translate", "rotate"):
                progress = last_estimate[i]
            else:
                if info.get("kind") != "hold":
                    observed = (info.get("signed_progress") if option["unit"] == "m" else
                                info.get("follow_time") if option["kind"] == "follow" else info.get("progress"))
                    if observed is not None:
                        last_generic_progress[i] = observed
                progress = last_generic_progress[i]
            end = record["boundary"]["skill_ends"].get(i, record["boundary"]["skill_ends"].get(str(i)))
            exit_reason = info.get("exit_reason")
            if not exit_reason and (info.get("kind") == "hold" or row is record["rows"][-1]):
                exit_reason = end.get("exit_reason") if end else None
            sides.append({"stage": info["stage"], "contact": info["contacting_object"],
                          "gap": info.get("gap"), "progress": progress, "actual": actual,
                          "estimated_effect": last_estimate[i], "estimate_time": last_estimate_time[i],
                          "estimate_stale": bool(not effect and last_estimate[i] is not None),
                          "normal_travel": persistent("normal_travel", info.get("completion_progress")),
                          "follow_normal_travel": persistent("follow_normal_travel"),
                          "recovery_normal_travel": persistent("recovery_normal_travel"),
                          "reference_budget": persistent("normal_reference_budget"),
                          "exit_reason": exit_reason, "force": row["forces"][i],
                          "surface_points": ((info.get("boundary_follow") or {}).get("support_points",info.get("patch_points_world"))
                                             if info.get("kind") != "hold" else []),
                          "boundary_follow": info.get("boundary_follow"),
                          "peer_detour": info.get("peer_detour"),
                          "static_route_guard": info.get("static_route_guard"),
                          "normal": info.get("normal_world"), "tangent": info.get("tangent_world"),
                          "face_switches": last_switches[i],
                          "desired_velocity": info.get("desired_velocity"),
                          "reference_velocity": info.get("reference_velocity")})
        frames[-1]["robots"] = sides
    return frames


def plot_overview(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = [name for name in ("tilt_+8", "offset_+0.03", "short_face_E", "cooperative_N",
                                "rotate_CCW_0.1", "rotation_couple_CCW")
              if any(r["metric"]["case"] == name for r in records)]
    joint_plot = any(r["metric"].get("comparison") in ("joint_execution", "contact_allocation", "normal_travel", "post_follow", "joint_recovery") for r in records)
    if joint_plot:
        chosen = [name for name in ("same_N_0.1_0.5", "N_E_0.5_0.1", "N_S_0.5_0.1",
                                    "same_face_N_NW", "peer_direction_unavailable", "rotation_mixed_angles")
                  if any(r["metric"]["case"] == name for r in records)]
    if not chosen:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.flat, chosen):
        for feedback, color in ((False, "#d38b32"), (True, "#218d70")):
            record = next((r for r in records if r["metric"]["case"] == name
                           and r["metric"].get("variant", r["metric"]["feedback"]) == feedback), None)
            if record is None or record["metric"]["status"] != "completed":
                continue
            m = record["metric"]
            label = {"endpoint_tracking": "tracking", "fast_approach": "fast approach",
                     "joint_execution": "joint protocol", "contact_allocation": "contact regions",
                     "normal_travel": "measured normal travel", "post_follow": "post-push following",
                     "joint_recovery": "joint recovery"}.get(
                m.get("comparison"), "correction")
            label += " on" if feedback else " off"
            action = record_actions(record)[0]
            initial = record["initial_object"]
            frames = replay_frames(record)
            if action.effect_kind == "translate":
                lateral = [-(f["o"][0] - initial[0]) * action.effect_direction[1]
                           + (f["o"][1] - initial[1]) * action.effect_direction[0] for f in frames]
                ax.plot(lateral, [f["actual"] for f in frames], color=color,
                        label=label)
                ax.set_xlabel("cross-track / m")
                ax.set_ylabel("forward travel / m")
                if not joint_plot:
                    ax.set_xlim(-.06, .06)
            else:
                ax.plot([f["t"] for f in frames], [f["actual"] for f in frames], color=color,
                        label=label)
                ax.set_xlabel("time / s")
                ax.set_ylabel("signed rotation / rad")
            if not joint_plot:
                ax.axhline(action.effect_target, color="#65758b", ls="--", lw=.8)
        ax.set_title(name)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle("Effect PUSH: paired physical outcomes (same physics and thresholds)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)


def render(out, include_rounds=()):
    manifest = json.loads((out / "manifest.json").read_text())
    tracking_comparison = manifest.get("comparison") == "endpoint_tracking"
    fast_comparison = manifest.get("comparison") == "fast_approach"
    joint_comparison = manifest.get("comparison") == "joint_execution"
    allocation_comparison = manifest.get("comparison") == "contact_allocation"
    travel_comparison = manifest.get("comparison") == "normal_travel"
    follow_comparison = manifest.get("comparison") == "post_follow"
    recovery_comparison = manifest.get("comparison") == "joint_recovery"
    scale_panel = manifest.get("comparison") == "reposition_scales_v2"
    combination_panel = manifest.get("comparison") in ("option_combinations","reposition_scales_v2")
    arc_comparison = manifest.get("comparison") == "reposition_corners"
    comparison_label = "同伴外侧绕行" if manifest.get("comparison") == "reposition_peer_detour" else "新换位尺度" if scale_panel else "圆弧换面" if arc_comparison else "动作组合" if combination_panel else "按间隙联合恢复" if recovery_comparison else "完成后接触随动" if follow_comparison else "实测法向行程" if travel_comparison else "同面接触分区" if allocation_comparison else "联合执行协议" if joint_comparison else "接近加速" if fast_comparison else "端点跟踪" if tracking_comparison else "接触点纠偏"
    records = [json.loads(p.read_text()) for p in sorted((out / "logs").glob("*.json"))]
    for r in records:hydrate_actions(r,manifest)
    for r in records:
        if (r.get("boundary", {}).get("execution", {}).get("protocol") == "proportional_normal_v1"
                or r["metric"].get("scale_semantics") in ("normal_reference_m", "measured_normal_travel_m")):
            r["metric"]["fulfilled"] = None  # display/metric scope correction, raw logs untouched
    metrics = [r["metric"] for r in records]
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Effect PUSH 测台", "", f"版本：{manifest['version']}；基础环境：{manifest['env_version']}。",
             "", "所有行包括失败/不可用请求。fulfilled 为离线效果阈值，exit 为控制器声明，二者分别记录。",
             "旋转请求允许伴随平移，不声称纯旋转。正进度和完成率不是 RL 任务成功率。",
             ("对比局部排斥与外侧安全轮廓绕行；静态短程通道被阻挡时退出。请求距离和原时间预算不变。" if manifest.get("comparison")=="reposition_peer_detour" else
              "只切换 REPOSITION 的局部边界控制；请求距离、计程、间隙目标和时间窗保持原值。" if arc_comparison else
              "REPOSITION 与其他 option 的覆盖面板，只有一个执行配置，不构造虚假的开/关对照；非 PUSH 不评分物体位移目标。" if combination_panel else
              "两组均接触随动；对比二值失联退出与按距离滞回的联合重接近。outward_pulse 是预声明的施力扰动诊断，不是常规场景。" if recovery_comparison else
              "两组均按本体实测行程完成主动推进；只比较随后原地制动/受限接触随动。随动路程另记；物体位移不作完成目标。" if follow_comparison else
              "平移两组分别采用参考积分/实测本体法向行程。物体位移不是终止目标，fulfilled 不适用。" if travel_comparison else
              "两组均为 v4 法向参考协议；仅对比同面接触分区及局部速度相关避碰缓冲。fulfilled 不适用。" if allocation_comparison else
              "双平移 PUSH 的尺度为名义法向参考预算；fulfilled 不适用，不承诺最终物体位移。" if joint_comparison else
              "接近阶段切向速度控制开/关对照，两组均开启端点跟踪及接触点纠偏。" if fast_comparison else
              "端点跟踪开/关对照，两组均开启接触点纠偏。" if tracking_comparison else
              "纠偏开关仅改变平移时的方向/侧偏接触点外环；两组都保留速度、间隙和效果终止反馈。"), "",
             f"| 场景 | {comparison_label} | 可用 | 退出 | 目标 | 实际 | 侧偏 m | 转角 ° | fulfilled | 耗时 s |",
             "|---|---|---|---|---:|---:|---:|---:|---|---:|"]
    slim = []
    for r in records:
        m = r["metric"]
        variant = m.get("variant", m["feedback"])
        if m["status"] != "completed":
            lines.append(f"| {m['case']} | {variant} | - | {m['status']} | - | - | - | - | - | - |")
            continue
        side = "—" if m["sideways"] is None else f"{m['sideways']:.3f}"
        actual_text = "—" if m["actual_effect"] is None else f"{m['actual_effect']:.3f}"
        lines.append(f"| {m['case']} | {variant} | {m['available']} | {m['exit']} | "
                     f"{m['target']:.2f} {m['unit']} | {actual_text} | {side} | "
                     f"{m['rotation_deg']:.2f} | {m['fulfilled']} | {m['seconds']:.2f} |")
        slim.append({"metric": m, "case": r["case"], "config": r["config"],
                     "action_version": r["boundary"]["action_version"],
                     "robot_options": replay_robot_options(r),
                     "available_directions": [DIRECTION_NAMES[i] for i in range(8)
                                              if r["availability_mask"][i * 5]],
                     "frames": replay_frames(r)})
    lines += ["", "## 配置与边界", "", "- mask 只使用本侧拟合几何，是局部几何可用性，不是隐藏物理下的成功保证。",
              "- 已知矩形尺寸为控制器公开先验；不可观测切向运动不靠真值补齐。",
              "- 不可用 PUSH 明确拒绝，不自动切换 REPOSITION。", "- 物体真值仅用于离线指标与图形。",
              f"- 冻结 manifest 中登记 {len(manifest['cases'])} 个场景；未完成分支保留显式状态。"]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    plot_overview(records, out)
    display_incomplete = [m for m in metrics if m["status"] != "completed"]
    primary_source = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    for item in slim:
        item["source_round"] = primary_source
    sources = [{"round": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
                "action_version": manifest["version"], "action_count": len(manifest["actions"]),
                "manifest_sha256": hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest(),
                "displayed_branches": len(slim)}]
    for extra in include_rounds:
        extra = Path(extra).resolve()
        extra_source = str(extra.relative_to(ROOT)) if extra.is_relative_to(ROOT) else str(extra)
        extra_manifest = json.loads((extra/"manifest.json").read_text())
        added = 0
        for path in sorted((extra/"logs").glob("*.json")):
            r = json.loads(path.read_text())
            hydrate_actions(r,extra_manifest)
            if r["metric"]["status"] != "completed":
                display_incomplete.append({**r["metric"], "source_round": extra_source})
                continue
            if any(s["metric"]["case"] == r["metric"]["case"] and s["metric"]["variant"] == r["metric"]["variant"] for s in slim):
                raise ValueError("duplicate case/variant across display sources")
            slim.append({"metric": r["metric"], "case": r["case"], "config": r["config"],
                         "action_version": r["boundary"]["action_version"],
                         "source_round": extra_source,
                         "robot_options": replay_robot_options(r),
                         "available_directions": [DIRECTION_NAMES[i] for i in range(8) if r["availability_mask"][i*5]],
                         "frames": replay_frames(r)})
            added += 1
        sources.append({"round": str(extra.relative_to(ROOT)) if extra.is_relative_to(ROOT) else str(extra),
                        "action_version": extra_manifest["version"], "action_count": len(extra_manifest["actions"]),
                        "manifest_sha256": hashlib.sha256((extra/"manifest.json").read_bytes()).hexdigest(),
                        "comparison": extra_manifest["comparison"], "displayed_branches": added})
    (out/"viewer_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2)+"\n")
    template = Path(__file__).with_name("effect_push_template.html").read_text()
    def compact(value):
        if isinstance(value, float):
            return round(value, 5)
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [compact(v) for v in value]
        return value
    payload = json.dumps(compact({"version": manifest['version'], "records": slim,
                          "comparison_label": comparison_label,
                          "initial_filter": "rp_detour" if any(s["metric"]["comparison"]=="reposition_peer_detour" for s in slim) else "rp_scale_v2" if any(s["metric"]["comparison"]=="reposition_scales_v2" for s in slim) else "rp_arc" if any(s["metric"]["comparison"]=="reposition_corners" for s in slim) else "reposition" if combination_panel or include_rounds else "all",
                          "initial_case": "detour_rp_CCW_push_same_1.80" if any(s["metric"]["case"]=="detour_rp_CCW_push_same_1.80" for s in slim) else "scale2_rp_CCW_solo_1.80" if any(s["metric"]["comparison"]=="reposition_scales_v2" for s in slim) else "arc_rp_CCW_corner_1.00" if any(s["metric"]["comparison"]=="reposition_corners" for s in slim) else None,
                          "incomplete": display_incomplete}), ensure_ascii=False,
                         separators=(",", ":")).replace("<", "\\u003c")
    (out / "index.html").write_text(template.replace("__DATA__", payload))
    (out / "render_manifest.json").write_text(json.dumps({
        "captured_utc": datetime.now(timezone.utc).isoformat(), "display_decimals": 5,
        "raw_logs_precision": "unchanged",
        "viewer_schema": "two-robot-inspector-v1",
        "paired_effect_score": "N/A: final joint pose is not a local exit contract",
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(Path(__file__).with_name("effect_push_template.html").read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/option_testbench_v1/effect_push_v1/round_001")
    parser.add_argument("--cases", nargs="*", help="optional exact names; default is the complete frozen panel")
    parser.add_argument("--branch-seconds", type=float, default=10.)
    parser.add_argument("--total-seconds", type=float, default=120.)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--include-round", type=Path, action="append", default=[],
                        help="additional frozen logs for display only; each source retains its own manifest/metrics")
    compare = parser.add_mutually_exclusive_group()
    compare.add_argument("--compare-tracking", action="store_true",
                        help="paired endpoint tracking off/on, with contact correction on in both")
    compare.add_argument("--compare-approach", action="store_true",
                         help="paired preparation speed profile off/on, endpoint tracking on in both")
    compare.add_argument("--compare-joint-execution", action="store_true",
                         help="paired legacy/new joint protocol; independent nominal scales retained")
    compare.add_argument("--compare-contact-allocation", action="store_true",
                         help="paired same-face regions and local clearance off/on, v4 protocol in both")
    compare.add_argument("--compare-normal-travel", action="store_true",
                         help="reference-clock vs measured own normal travel, for single and dual translations")
    compare.add_argument("--compare-post-follow", action="store_true",
                         help="paired post-threshold hold/brake vs bounded contact follow")
    compare.add_argument("--compare-joint-recovery", action="store_true",
                         help="v7 binary-contact abort vs gap-based joint reapproach, with declared pulse diagnostics")
    compare.add_argument("--reposition-panel", action="store_true",
                         help="single-configuration REPOSITION plus WAIT/PUSH/FOLLOW/REPOSITION coverage")
    compare.add_argument("--compare-reposition-corners", action="store_true",
                         help="legacy face tangent vs finite-boundary rounded offset, matched distances and physics")
    compare.add_argument("--reposition-scales-v2", action="store_true",
                         help="versioned 68-action menu, rounded REPOSITION at 0.2/0.4/0.8/1.2/1.8 m")
    compare.add_argument("--compare-reposition-peer", action="store_true",
                         help="rounded boundary with local repulsion vs outer peer-disc detour, same menu and budget")
    parser.add_argument("--heading-gain", type=float)
    parser.add_argument("--lateral-gain", type=float)
    parser.add_argument("--angular-rate-gain", type=float)
    parser.add_argument("--point-rate", type=float)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        task = json.load(sys.stdin)
        result = run_one(task["case"], task["feedback"], task.get("overrides"),
                         task.get("comparison", "contact_feedback"))
        Path(task["path"]).write_text(json.dumps(result, ensure_ascii=False) + "\n")
        return
    out = args.out.resolve()
    if args.render_only:
        render(out, args.include_round)
        print(out / "index.html")
        return
    if out.exists() and any(out.iterdir()):
        parser.error("output is nonempty; select a new round directory or use --render-only")
    (out / "logs").mkdir(parents=True)
    comparison = "reposition_peer_detour" if args.compare_reposition_peer else "reposition_scales_v2" if args.reposition_scales_v2 else "reposition_corners" if args.compare_reposition_corners else "option_combinations" if args.reposition_panel else "joint_recovery" if args.compare_joint_recovery else "post_follow" if args.compare_post_follow else "normal_travel" if args.compare_normal_travel else "contact_allocation" if args.compare_contact_allocation else "joint_execution" if args.compare_joint_execution else "fast_approach" if args.compare_approach else "endpoint_tracking" if args.compare_tracking else "contact_feedback"
    panel = reposition_peer_cases() if args.compare_reposition_peer else reposition_scale_cases() if args.reposition_scales_v2 else reposition_corner_cases() if args.compare_reposition_corners else reposition_cases() if args.reposition_panel else recovery_cases() if args.compare_joint_recovery else post_follow_cases() if args.compare_post_follow else normal_travel_cases() if args.compare_normal_travel else allocation_cases() if args.compare_contact_allocation else joint_cases() if args.compare_joint_execution else approach_cases() if args.compare_approach else endpoint_cases() if args.compare_tracking else cases()
    variants = (True,) if args.reposition_panel or args.reposition_scales_v2 else (False, True)
    overrides = {key: getattr(args, key) for key in
                 ("heading_gain", "lateral_gain", "angular_rate_gain", "point_rate")
                 if getattr(args, key) is not None}
    EffectPushConfig(**overrides)  # validate before starting workers
    if args.cases:
        unknown = set(args.cases) - {c["name"] for c in panel}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        panel = [c for c in panel if c["name"] in args.cases]
    started = time.monotonic()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    controller = ROOT / "src/coimagine/random_start/effect_push.py"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    menu=ACTIONS_REPOSITION_V2 if args.reposition_scales_v2 or args.compare_reposition_peer else ACTIONS
    manifest = {"version": REPOSITION_V2_VERSION if args.reposition_scales_v2 or args.compare_reposition_peer else VERSION, "env_version": spec.ENV_VERSION, "code_base": revision,
                "comparison": comparison,
                "controller_sha256": sha(controller), "runner_sha256": sha(Path(__file__)),
                "cases": panel, "actions": [asdict(a) for a in menu],
                "feedback_configs": [asdict(comparison_config(comparison, f, overrides)) for f in variants],
                "branch_seconds": args.branch_seconds, "total_seconds": args.total_seconds,
                "effect_thresholds": {"translation_under_m": .02, "translation_over_m": .04,
                                      "sideways_m": .04, "rotation_drift_deg": 10,
                                      "rotation_under_rad": .03, "rotation_over_rad": .08},
                "command": sys.argv, "reward_config": E.EnvConfig().as_dict()}
    manifest["captured_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in
        [controller, Path(__file__).resolve(), Path(__file__).with_name("effect_push_template.html"),
         ROOT / "src/coimagine/random_start/env.py", ROOT / "src/coimagine/random_start/lidar.py",
         ROOT / "src/coimagine/random_start/skills.py", ROOT / "src/coimagine/random_start/spec.py"]}
    tracking_source = ROOT / "src/coimagine/random_start/endpoint_tracking.py"
    manifest["source_sha256"][str(tracking_source.relative_to(ROOT))] = sha(tracking_source)
    allocation_source = ROOT / "src/coimagine/random_start/same_face.py"
    manifest["source_sha256"][str(allocation_source.relative_to(ROOT))] = sha(allocation_source)
    boundary_source = ROOT / "src/coimagine/random_start/boundary_follow.py"
    manifest["source_sha256"][str(boundary_source.relative_to(ROOT))] = sha(boundary_source)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "ROUND_PLAN.md").write_text(
        "# Effect PUSH bounded development round\n\n"
        "Written before branch execution; exact inputs and source hashes are in manifest.json.\n\n"
        f"- Cases: {', '.join(c['name'] for c in panel)}\n"
        + ("- Single current configuration; coverage panel, not an off/on ablation.\n" if args.reposition_panel or args.reposition_scales_v2 else
         f"- Paired variable: {comparison} off/on; exact configurations in manifest.\n")
        + f"- Explicit controller overrides: {json.dumps(overrides, sort_keys=True)}\n"
        f"- Wall limits: {args.branch_seconds} s/branch, {args.total_seconds} s/batch.\n"
        "- Preserve all exits and unavailable requests; do not change physics, task costs or thresholds.\n"
        "- Deliver replay and metrics for human feedback; no automatic RL training.\n")
    for case in panel:
        for feedback in variants:
            suffix = ("detour_on" if feedback else "detour_off") if args.compare_reposition_peer else "current" if args.reposition_scales_v2 else ("arc_on" if feedback else "arc_off") if args.compare_reposition_corners else "current" if args.reposition_panel else ("recovery_on" if feedback else "recovery_off") if args.compare_joint_recovery else \
                ("follow_on" if feedback else "follow_off") if args.compare_post_follow else \
                ("travel_on" if feedback else "travel_off") if args.compare_normal_travel else \
                ("allocation_on" if feedback else "allocation_off") if args.compare_contact_allocation else \
                ("joint_on" if feedback else "joint_off") if args.compare_joint_execution else \
                ("approach_on" if feedback else "approach_off") if args.compare_approach else \
                ("tracking_on" if feedback else "tracking_off") if args.compare_tracking \
                else ("feedback" if feedback else "no_correction")
            path = out / "logs" / f"{case['name']}__{suffix}.json"
            remaining = args.total_seconds - (time.monotonic() - started)
            status = None
            error = None
            if remaining <= 0:
                status = "batch_budget_incomplete"
            else:
                config = asdict(comparison_config(comparison, feedback, overrides))
                branch_feedback = config.pop("feedback")
                task = {"case": case, "feedback": branch_feedback,
                        "path": str(path), "overrides": config, "comparison": comparison}
                try:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                            input=json.dumps(task), text=True, capture_output=True,
                                            timeout=min(args.branch_seconds, remaining), cwd=ROOT)
                    if result.returncode:
                        status = "branch_error"
                        error = result.stderr[-4000:]
                except subprocess.TimeoutExpired:
                    status = "branch_timeout_incomplete"
            if status:
                path.write_text(json.dumps({"metric": {"case": case["name"], "feedback": feedback,
                                                       "variant": feedback, "comparison": comparison,
                                                       "status": status, "error": error}},
                                           ensure_ascii=False) + "\n")
    metrics = render(out, args.include_round)
    print(json.dumps({"out": str(out), "branches": len(metrics),
                      "completed": sum(m["status"] == "completed" for m in metrics),
                      "seconds": time.monotonic() - started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
