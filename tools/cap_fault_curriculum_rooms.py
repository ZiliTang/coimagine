"""Freeze the paired diagonal-transport development and holdout fixtures.

Call freeze(out) in the host, giving the authors access only to the development
observations. The geometrical strata and room records are offline audit data.
"""
from coimagine.random_start.spec import ROBOT_DIAMETER
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import tempfile

from experiments.planar_tasks.authoring_v2 import random_room_tasks as geometry
from experiments.planar_tasks.relational_search import pipeline_execution
from experiments.planar_tasks.relational_search.contract import digest


VERSION = "cap-diagonal-two-obstacles-v1"
SEEDS = {"dev": 662401017, "held": 662401019}
STRATA = ("clear", "static", "movable", "both")
START = (1.45, 1.10)
GOAL = (4.25, 3.90)
HEADING = math.pi / 4
TARGET = (.25, 1.0)
STATIC = (.78, .52)
MOVABLE = (.60, .60)


def robot_poses():
    unit = math.sqrt(.5)
    return [dict(x=START[0] - unit * (.61 + side), y=START[1] + unit * (side - .61),
                 angle=HEADING) for side in (-.34, .34)]


def content_hash(fixture):
    return digest({k: v for k, v in fixture.items() if k != "scene_id"})


def panel(split, count):
    if type(count) is not int or count < 1:
        raise ValueError("room count must be a positive integer")
    rng = random.Random(SEEDS[split])
    poses = robot_poses()
    forbidden = [geometry.polygon(*START, *TARGET, HEADING),
                 geometry.polygon(*GOAL, *TARGET, HEADING)]
    forbidden.extend(geometry.polygon(r["x"], r["y"], ROBOT_DIAMETER, ROBOT_DIAMETER, HEADING) for r in poses)
    center = ((START[0] + GOAL[0]) / 2, (START[1] + GOAL[1]) / 2)
    swept = geometry.polygon(*center, TARGET[0] + math.dist(START, GOAL), TARGET[1], HEADING)
    rooms, audit = {}, {}
    strata = [STRATA[i % len(STRATA)] for i in range(count)]
    rng.shuffle(strata)
    for number, stratum in enumerate(strata, 1):
        occupied = list(forbidden)
        obstacles = []
        for name, size in (("static", STATIC), ("movable", MOVABLE)):
            should_block = stratum in ("both", name)
            for _ in range(10000):
                x, y = rng.uniform(.4, 5.6), rng.uniform(.4, 4.4)
                polygon = geometry.polygon(x, y, *size, 0.)
                if not geometry.inside(polygon):
                    continue
                if (not geometry.separated(polygon, swept, gap=0.)) != should_block:
                    continue
                if any(not geometry.separated(polygon, previous, .10) for previous in occupied):
                    continue
                obstacles.append((x, y))
                occupied.append(polygon)
                break
            else:
                raise RuntimeError(f"failed to place the {name} box in {stratum}")
        (sx, sy), (mx, my) = obstacles
        room = dict(
            scene_id="random_room_v1",
            room=dict(arena=[6., 4.8], obstacles=[dict(cx=sx, cy=sy,
                      half_w=STATIC[0]/2, half_h=STATIC[1]/2, angle=0.)]),
            object=dict(x=START[0], y=START[1], angle=HEADING), robots=[r.copy() for r in poses],
            goal=dict(x=GOAL[0], y=GOAL[1], radius=.20),
            movable_objects=[dict(label="entity-b", x=mx, y=my, angle=0., width=MOVABLE[0],
                                  height=MOVABLE[1], mass=3.)], overrides={},
        )
        key = f"diag_{split}_{number:03d}"
        rooms[key] = room
        audit[key] = dict(stratum=stratum, fixture_sha256=digest(room), content_sha256=content_hash(room))
    return rooms, audit


def freeze(out, *, dev_count=160, held_count=200):
    """Return {dev: {room_id: fixture}, held: {...}}; reject a differing rooms.json."""
    out = Path(out)
    groups, records = {}, {}
    for split, size in (("dev", dev_count), ("held", held_count)):
        groups[split], audit = panel(split, size)
        records.update(audit)
    if len({row["content_sha256"] for row in records.values()}) != dev_count + held_count:
        raise ValueError("fixture content overlap")
    payload = dict(version=VERSION, seeds=SEEDS, rooms=groups, records=records,
                   stratum_definition="Initial axis-aligned boxes intersect the straight fixed-orientation swept target rectangle. No goal path or robot feasibility is implied.",
                   counts={s: dict(Counter(records[k]["stratum"] for k in group)) for s, group in groups.items()},
                   solvability="Geometrically valid; no policy-outcome filtering or solvability guarantee.")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rooms.json"
    if not path.exists():
        for rooms in groups.values():
            for fixture in rooms.values():
                pipeline_execution.fixture(fixture)
        with tempfile.NamedTemporaryFile(mode="w", dir=out, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
    if json.loads(path.read_text()) != payload:
        raise ValueError(f"frozen room set differs at {path}")
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dev-count", type=int, default=160)
    parser.add_argument("--held-count", type=int, default=200)
    args = parser.parse_args()
    freeze(args.out, dev_count=args.dev_count, held_count=args.held_count)
    payload = json.loads((args.out / "rooms.json").read_text())
    print(json.dumps(dict(version=VERSION, counts=payload["counts"], manifest_digest=digest(payload))))
