"""Public-only evidence and action-parameterized semantic registration.

The geometric certificate is conditional on the explicitly recorded sensor,
initial separation, CAD and motion assumptions. No fixture or environment is
accepted by this interface. Safety audits are performed elsewhere.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import math
import time

from shapely.geometry import Point
from shapely.ops import unary_union

from ..symbolic_general import frontend as base
from .evidence import CachedEvidenceMap
from ..symbolic_unknown.geometry import MotionContract, rectangle_pose, envelopes, disc
from experiments.planar_tasks.relational_search.contract import public_observation
from experiments.planar_tasks.semantic_planning.runtime import public_copy


def options():
    out = {}
    for direction in range(8):
        code = 4 * direction + 1
        for role, pair in (("both", [code, code]), ("r0", [code, 44]), ("r1", [44, code])):
            for ticks in (4, 8):
                name = f"push_{role}_{direction}_{ticks}"
                out[name] = dict(family="push", pair=pair, heading=direction,
                                 ticks=ticks, robots=[i for i, a in enumerate(pair) if a != 44])
        for robot in (0, 1):
            pair = [44, 44]
            pair[robot] = 48 + 4 * direction
            out[f"move_{robot}_{direction}"] = dict(family="move", pair=pair,
                heading=direction, ticks=4, robots=[robot])
    for robot in (0, 1):
        for side, code in (("ccw", 40), ("cw", 43)):
            pair = [44, 44]
            pair[robot] = code
            out[f"orbit_{robot}_{side}"] = dict(family="orbit", pair=pair,
                heading=None, ticks=4, robots=[robot])
    out["brake"] = dict(family="brake", pair=[44, 44], heading=None, ticks=4, robots=[])
    return out


OPTIONS = options()


def schema():
    metrics = {
        "seen": dict(unit="bool"),
        "distance": dict(unit="m", domain=[0, 20]),
        "gx": dict(unit="m", domain=[-20, 20]),
        "gy": dict(unit="m", domain=[-20, 20]),
        "speed": dict(unit="m/s", domain=[0, 100]),
        "stagnation": dict(unit="s", domain=[0, 10000]),
        "contact0": dict(unit="bool"), "contact1": dict(unit="bool"),
        "target_stationary": dict(unit="bool"),
    }
    for i in range(2):
        for suffix in ("rx", "ry", "normal_position", "tangent_position"):
            metrics[f"r{i}_{suffix}"] = dict(unit="m", domain=[-20, 20])
    for d in range(8):
        for suffix in ("align", "normal_align", "tangent_align"):
            metrics[f"{suffix}_{d}"] = dict(unit="1", domain=[-1.000001, 1.000001])
    opts = {}
    for name, opt in OPTIONS.items():
        for flag in ("legal", "free", "occupied", "unknown", "dynamic", "bound_valid"):
            metrics[f"{flag}:{name}"] = dict(unit="bool")
        metrics[f"missing:{name}"] = dict(unit="1", domain=[0, 100])
        opts[name] = dict(requires=["var", "legal:" + name], productive=opt["family"] != "brake",
                          command=deepcopy(opt))
    for opt in opts.values():
        opt.update(writes=list(metrics), frame=[])
    return dict(metrics=metrics, options=opts)


class PublicFrontend:
    def __init__(self, contract=None, *, evidence=None):
        self.contract = contract or MotionContract()
        self.evidence = evidence if evidence is not None else CachedEvidenceMap(minimum_static_component_area=self.contract.minimum_static_component_area)
        self.memory = base.initial_memory()
        self.target_stationary = True
        self.pose_history = None
        self.initialized = False
        self.last_tick = None
        self.last_packet = None
        self.recent_observations = []
        self.ledger = dict(boundaries=0, seconds=0.)

    def record_command(self, name):
        if OPTIONS[name]["family"] == "push":
            self.target_stationary = False
            self.last_packet = None

    def motion_regions(self, observation, pose, opt):
        contract = replace(self.contract, move_ticks=opt["ticks"], push_ticks=opt["ticks"],
                           push_robot_path=self.contract.push_robot_path * opt["ticks"] / 4,
                           target_push_boundary_path=self.contract.target_push_boundary_path * opt["ticks"] / 4)
        kind = opt["family"]
        if kind == "move":
            return envelopes(observation, pose, "observe", opt["robots"][0], contract, opt["heading"],
                             target_stationary=self.target_stationary)
        if kind == "brake":
            return envelopes(observation, pose, "brake", 0, contract,
                             target_stationary=self.target_stationary)
        # Each selected pusher receives its own motion envelope; its peer is not
        # silently treated as stationary. Independent ORBIT has a separate bound.
        region = envelopes(observation, pose, "brake", 0, contract,
                           target_stationary=self.target_stationary)
        if kind == "push":
            for robot in opt["robots"]:
                pushed = envelopes(observation, pose, "push", robot, contract, opt["heading"],
                                   target_stationary=self.target_stationary)
                region["robots"][robot] = pushed["robots"][robot]
                region["target"] = pushed["target"]
        elif kind == "orbit":
            for robot in opt["robots"]:
                own = observation["robots"][robot]["own"]
                radius = contract.robot_radius + contract.robot_stop_path + .4 * opt["ticks"] * .05
                region["robots"][robot] = disc(own["x"], own["y"], radius)
        region["static"] = unary_union([*region["robots"], *([region["target"]] if region["target"] is not None else [])])
        return region

    def prepare(self, raw_observation):
        o = public_observation(raw_observation)
        if o["tick"] == self.last_tick and self.last_packet is not None:
            if o != self.last_packet["observation"]:
                raise ValueError("different public packet at the same boundary")
            return self.last_packet
        if self.last_tick is not None and o["tick"] < self.last_tick:
            raise ValueError("public history cannot run backwards")
        started = time.monotonic()
        self.memory = base.update_memory(self.memory, dict(observation=o))
        pose = rectangle_pose(o, error=self.contract.pose_error)
        if pose is not None and self.target_stationary:
            if self.pose_history is not None:
                pose.outer = pose.outer.intersection(self.pose_history.outer)
                pose.inner = unary_union([pose.inner, self.pose_history.inner])
                if pose.outer.is_empty:
                    pose = None
            if pose is not None:
                pose.centre = (pose.outer.centroid.x, pose.outer.centroid.y)
                self.pose_history = deepcopy(pose)
        if not self.initialized:
            if o["tick"] != 0:
                raise ValueError("frontend must receive the complete public prefix from tick zero")
            for r in o["robots"]:
                self.evidence.add_initial_certificate(Point(r["own"]["x"], r["own"]["y"]).buffer(
                    self.contract.robot_radius + self.contract.initial_static_separation),
                    source="declared_initial_separation")
            self.initialized = True
        if pose is not None and self.target_stationary:
            self.evidence.add_initial_certificate(pose.inner.buffer(self.contract.initial_static_separation),
                                                  source="declared_initial_separation")
        self.evidence.update(o)
        # public_observation already owns a detached sensor tree. Older trees
        # are never mutated; predictors receive their own copy at the gateway.
        self.recent_observations.append(o)
        self.recent_observations = self.recent_observations[-4:]
        centre = pose.centre if pose else self.memory.get("centre") or o["goal"]["xy"]
        normal = pose.normal if pose else (1., 0.)
        tangent = pose.tangent if pose else (0., 1.)
        gx, gy = (o["goal"]["xy"][i] - centre[i] for i in (0, 1))
        distance = math.hypot(gx, gy)
        # Orient the fitted axis toward the public goal so changing the visible
        # long face does not silently flip the author's coordinate convention.
        if gx * normal[0] + gy * normal[1] < 0:
            normal = tuple(-v for v in normal)
            tangent = tuple(-v for v in tangent)
        speed = max(math.hypot(r["own"]["vx"], r["own"]["vy"]) for r in o["robots"])
        z = dict(seen=pose is not None, distance=distance, gx=gx, gy=gy, speed=speed,
                 stagnation=self.memory["stagnation"], target_stationary=self.target_stationary,
                 contact0=bool(o["robots"][0]["contacting_object"]),
                 contact1=bool(o["robots"][1]["contacting_object"]))
        for i, r in enumerate(o["robots"]):
            rel = (r["own"]["x"] - centre[0], r["own"]["y"] - centre[1])
            z.update({f"r{i}_rx": rel[0], f"r{i}_ry": rel[1],
                      f"r{i}_normal_position": sum(a * b for a, b in zip(rel, normal)),
                      f"r{i}_tangent_position": sum(a * b for a, b in zip(rel, tangent))})
        for d in range(8):
            u = (math.cos(d * math.pi / 4), math.sin(d * math.pi / 4))
            z[f"align_{d}"] = (gx * u[0] + gy * u[1]) / max(distance, 1e-9)
            z[f"normal_align_{d}"] = sum(a * b for a, b in zip(u, normal))
            z[f"tangent_align_{d}"] = sum(a * b for a, b in zip(u, tangent))
        menu = {tuple(p) for p in o["entity_legal_pairs"].get(o["goal"]["target_id"], [])}
        global_menu = {tuple(p) for p in o["global_legal_pairs"]}
        reports, commands, regions = {}, {}, {}
        geometry_cache = {}
        for name, opt in OPTIONS.items():
            target_bound = opt["family"] in ("push", "orbit")
            legal = tuple(opt["pair"]) in (menu if target_bound else global_menu)
            legal = bool(legal and (pose is not None or not target_bound))
            command = dict(target_id=o["goal"]["target_id"] if target_bound else None,
                           a0=opt["pair"][0], a1=opt["pair"][1], ticks=opt["ticks"])
            # PUSH's declared disk bound is direction independent; ORBIT's is
            # side independent. Reuse only geometrically identical queries.
            key = (opt["family"], tuple(opt["robots"]), opt["ticks"],
                   opt["heading"] if opt["family"] == "move" else None)
            if key not in geometry_cache:
                region = self.motion_regions(o, pose, opt)
                if hasattr(self.evidence, "query_parts"):
                    evidence = self.evidence.query_parts([*region["robots"], region["target"]])
                else:
                    evidence = self.evidence.query(region["static"])
                dynamic = pose is not None and not region["robots"][0].intersects(region["robots"][1])
                if pose is not None and opt["family"] == "move":
                    dynamic = dynamic and not region["robots"][opt["robots"][0]].intersects(pose.outer.buffer(.005))
                if pose is not None and opt["family"] == "push":
                    dynamic = dynamic and all(not region["robots"][i].intersects(region["target"])
                                              for i in range(2) if i not in opt["robots"])
                geometry_cache[key] = region, evidence, dynamic
            region, evidence, dynamic = geometry_cache[key]
            bound_valid = pose is not None and speed <= .325
            flags = dict(legal=legal, free=evidence.certified, occupied=evidence.status == "occupied",
                         unknown=evidence.status == "unknown", dynamic=bool(dynamic), bound_valid=bool(bound_valid))
            z.update({f"{flag}:{name}": value for flag, value in flags.items()})
            z["missing:" + name] = min(100., evidence.unknown_area_m2)
            report_evidence = dict(status=evidence.status, unknown_area_m2=evidence.unknown_area_m2,
                occupied_hits=evidence.occupied_hits, evidence_ticks=evidence.evidence_ticks,
                assumptions=evidence.assumptions)
            reports[name] = dict(**flags, evidence=report_evidence,
                                 allowed=bool(legal and evidence.certified and dynamic and bound_valid),
                                 quality="conditional_on_declared_host_assumptions")
            commands[name] = command
            regions[name] = region
        self.ledger["boundaries"] += 1
        self.ledger["seconds"] += time.monotonic() - started
        self.last_tick = o["tick"]
        self.last_packet = dict(observation=o, symbols=z, commands=commands, reports=reports,
                                regions=regions, pose=pose, contract=asdict(self.contract))
        self.last_packet["public_context"] = dict(observation=o,
            history=list(self.recent_observations), memory=public_copy(self.memory),
            symbols=dict(z), target_stationary=self.target_stationary)
        return self.last_packet

    def register(self, prepared):
        """Stage 2 returns a detached value packet; predictions cannot mutate it."""
        return public_copy(prepared["symbols"])
