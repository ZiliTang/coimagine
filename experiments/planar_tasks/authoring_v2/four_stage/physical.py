"""Independent physics labels. This module is never passed to the policy."""
from copy import deepcopy
import math
import time

from shapely.geometry import Polygon

from ..symbolic_general.simulation import observation
from ..symbolic_unknown.geometry import disc
from ..random_room_tasks import polygon


def state(world):
    backend = world.env.env.backend
    target = backend.bodies[world.env.primary_id]
    xy = (float(target.position.x), float(target.position.y))
    angle = float(target.angle)
    contacts = backend.contacts()
    return dict(target_xy=xy, target_angle=angle,
        target_corners=polygon(*xy, .25, 1., angle),
        target_speed=math.hypot(target.velocity.x, target.velocity.y),
        target_boundary_speed=math.hypot(target.velocity.x, target.velocity.y)
                              + abs(target.angular_velocity) * math.hypot(.125, .5),
        robots=[(float(b.position.x), float(b.position.y)) for b in backend.robot_bodies],
        robot_speeds=[math.hypot(b.velocity.x, b.velocity.y) for b in backend.robot_bodies],
        target_static=any(c["pair"] == "object_wall" and c.get("entity_id") == world.env.primary_id for c in contacts),
        robot_static=any(c["pair"] == "robot_wall" for c in contacts),
        robot_other=any(c["pair"] == "robot_object" and c.get("entity_id",world.env.primary_id) != world.env.primary_id for c in contacts),
        peer=any(c["pair"] == "robot_robot" for c in contacts),
        target_other=any(c["pair"] == "object_object" and c.get("entity_id") == world.env.primary_id for c in contacts))


def process_bounds(start, states):
    states = [start, *states]
    bounds = {}
    for robot in (0, 1):
        dx = [s["robots"][robot][0] - start["robots"][robot][0] for s in states]
        dy = [s["robots"][robot][1] - start["robots"][robot][1] for s in states]
        bounds[f"r{robot}"] = [min(dx), max(dx), min(dy), max(dy)]
    bounds["target_boundary"] = max(math.dist(p, q)
        for s in states for p, q in zip(s["target_corners"], start["target_corners"]))
    return bounds


def measure(world, command, *, remaining=None, regions=None):
    """Intercept each actual integration tick, including switch/brake ticks."""
    start = state(world)
    samples = []
    original = world.env.joint_step
    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        samples.append(state(world))
        return result
    world.env.joint_step = step
    begun = time.monotonic()
    try:
        event = world.step(command, remaining_ticks=remaining)
    finally:
        world.env.joint_step = original
    end = state(world)
    violations = []
    if regions is not None:
        for index, sample in enumerate([start, *samples]):
            for robot in (0, 1):
                if not regions["robots"][robot].buffer(1e-7).covers(disc(*sample["robots"][robot], .10)):
                    violations.append(dict(sample=index, body=f"robot{robot}"))
            if regions["target"] is not None and not regions["target"].buffer(1e-7).covers(Polygon(sample["target_corners"])):
                violations.append(dict(sample=index, body="target"))
    goal = observation(world)["goal"]["xy"]
    return dict(event=event, start=start, end=end, samples=samples,
                process=process_bounds(start, samples),
                target_static=any(s["target_static"] for s in [start, *samples]),
                robot_static=any(s["robot_static"] for s in [start, *samples]),
                robot_other=any(s["robot_other"] for s in [start, *samples]),
                peer=any(s["peer"] for s in [start, *samples]),
                target_other=any(s["target_other"] for s in [start, *samples]),
                goal_gain=math.dist(start["target_xy"], goal) - math.dist(end["target_xy"], goal),
                envelope_violations=violations, seconds=time.monotonic() - begun)


def command_and_stop(world, command, *, brake_ticks=40, brake_chunk=4, regions=None):
    """Label one command and its physical stopping tail, with separate endpoint."""
    root = measure(world, command, regions=regions)
    endpoint = deepcopy(observation(world))
    tails = []
    left = brake_ticks
    while left > 0 and not world.env.termination:
        ticks = min(brake_chunk, left)
        row = measure(world, dict(target_id=None, a0=44, a1=44, ticks=ticks), regions=regions)
        tails.append(row)
        actual = row["event"]["periods"]
        left -= actual
        if actual == 0:
            break
    samples = [*root["samples"], *[s for row in tails for s in row["samples"]]]
    end = state(world)
    return dict(root=root, endpoint=endpoint, stopping=tails,
        process=process_bounds(root["start"], samples),
        stop_ticks=brake_ticks-left,
        stopped=max([end["target_boundary_speed"], *end["robot_speeds"]]) < .005,
        interrupted_by_termination=bool(world.env.termination),
        full_target_static=any(s["target_static"] for s in [root["start"], *samples]),
        full_robot_static=any(s["robot_static"] for s in [root["start"], *samples]),
        full_robot_other=any(s["robot_other"] for s in [root["start"], *samples]),
        full_peer=any(s["peer"] for s in [root["start"], *samples]),
        envelope_violations=[v for row in [root, *tails] for v in row["envelope_violations"]])
