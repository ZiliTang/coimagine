"""Exercise a checked two-command policy through the real duration-boundary simulator.

This host certifies only public command availability at issue and remaining time.
The independently measured object and peer displacements are diagnostics.
"""
import argparse
import json
from math import dist
from pathlib import Path

from experiments.planar_tasks.authoring_v2.guarded_options import compile_policy
from experiments.planar_tasks.relational_search.contract import validate_action
from tools import cap_safe_nominal_study as study


WINDOW = 8
# Both the contract and the command issued below use these exact bindings.
BINDINGS = {
    "short": (4, 46, 6),   # robot 0 PUSH_NE, robot 1 YIELD_CLEARANCE
    "long": (4, 46, 8),
    "brake": (44, 44, 2),
}


def make_policy(push):
    units = {"in_menu": "flag", "remaining": "tick", "elapsed": "tick"}
    options = {"brake": {"requires": {"remaining": [BINDINGS["brake"][2], None]}}}
    for name in ("short", "long"):
        ticks = BINDINGS[name][2]
        options[name] = {
            "requires": {"in_menu": [1, 1], "remaining": [ticks, None]},
            "quality": "model",
            "outcomes": [{
                "observe": "boundary", "change": {"elapsed": [0, ticks]},
                "post": {"remaining": ["sub", "remaining", "elapsed"]},
            }],
        }
    cells = {name: {"remaining": [WINDOW, WINDOW], "in_menu": [flag, flag]}
             for name, flag in (("available", 1), ("unavailable", 0))}
    rule = {"name": "push_then_brake", "option": push, "when": {"in_menu": [1, 1]},
            "expect": {"elapsed": [0, BINDINGS[push][2]]}, "then": {"boundary": "brake"}}
    return compile_policy({"rules": [rule]}, {"units": units, "cells": cells, "options": options})


def command(name, obs):
    a0, a1, ticks = BINDINGS[name]
    target = None if name == "brake" else obs["goal"]["target_id"]
    return validate_action(dict(a0=a0, a1=a1, ticks=ticks, target_id=target), obs)


def public_cell(obs):
    try:
        command("short", obs)
    except ValueError:
        return "unavailable"
    return "available"


def run_room(room, fixture, short, long):
    with study.S.phase.world(fixture) as (world, _):
        obs = study.S.observe(world)
        cell = public_cell(obs)
        start = obs["tick"]
        row = {"room": room, "fixture_sha256": study.content_hash(fixture),
               "cell": cell, "public_history": world.history.sha256,
               "start_tick": start, "long_issued": long.decide(cell) is not None}
        if row["long_issued"]:
            raise ValueError("long option was admitted without reserved follow-up time")
        name = short.decide(cell)
        if name is None:
            return {**row, "events": []}

        events = []
        while name is not None:
            request = command(name, obs)
            left = WINDOW - (obs["tick"] - start)
            if not 0 < request["ticks"] <= left:
                raise ValueError("observed time budget disagrees with certificate")
            # Hidden truth is sampled only after the action has been chosen.
            before = study.truth(world)["object"][:2]
            peer = obs["robots"][1]["own"]
            event = world.step(request, remaining_ticks=left)
            obs = study.S.observe(world)
            after = study.truth(world)["object"][:2]
            new_peer = obs["robots"][1]["own"]
            if event["requested"] != request or not 0 <= event["periods"] <= request["ticks"]:
                raise ValueError("recorded request or elapsed ticks disagree with certificate")
            events.append({
                "token": name, "requested": request, "reason": event["reason"],
                "periods": event["periods"], "controls": event["segments"],
                "object_displacement_m": dist(before, after),
                "peer_displacement_m": dist((peer["x"], peer["y"]), (new_peer["x"], new_peer["y"])),
            })
            if name == "brake" or world.env.termination is not None:
                break
            name = short.resume(cell, "boundary")
            if name != "brake":
                raise ValueError("missing checked continuation at the live boundary")
        return {**row, "events": events, "end_tick": obs["tick"], "terminal": world.env.termination}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    short, long = make_policy("short"), make_policy("long")
    rooms = [run_room(key, fixture, short, long) for key, fixture in sorted(study.dev_rooms().items())]
    result = {
        "scope": "All 16 prior development reset states. Logical time and public issue menu only.",
        "window_ticks": WINDOW, "bindings_a0_a1_ticks": BINDINGS,
        "short_check": short.check(), "long_check": long.check(), "rooms": rooms,
        "short_sent": sum(bool(row["events"]) for row in rooms),
        "long_sent": sum(row["long_issued"] for row in rooms),
        "brake_sent": sum(any(e["token"] == "brake" for e in row["events"]) for row in rooms),
        "with_physical_push_control": sum(any(
            s["pair"] == list(BINDINGS["short"][:2]) for e in row["events"] for s in e["controls"])
            for row in rooms),
    }
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rooms)} rooms, {result['short_sent']} short, {result['long_sent']} long, "
          f"{result['brake_sent']} subsequent brakes")


if __name__ == "__main__":
    main()
