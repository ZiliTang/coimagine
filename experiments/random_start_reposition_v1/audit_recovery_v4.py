"""Bounded physical recovery probes on four saved states, with repaired PUSH timing.

Uses the frozen v3 world and current repaired controllers; reports this deliberate
offline hybrid explicitly. No policy training and no reward-based candidate selection.
"""
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/random_start_rl_diagnosis_v4"
sys.path.insert(0, str(OUT / "code/src"))

from coimagine.random_start import env, skills, spec


def main():
    path = ROOT / "src/coimagine/random_start/skills.py"
    module_name = "coimagine.random_start.skills_clock_fixed"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    repaired = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = repaired
    module_spec.loader.exec_module(repaired)
    env.make_skill = repaired.make_skill
    with (OUT / "short_loop_states.pkl").open("rb") as handle:
        states = pickle.load(handle)
    actions = [(9, 9), (11, 11), (9, 11), (11, 9), (13, 13), (4, 4)]
    report = dict(scope="four fixed observed states; six fixed repeated joint actions; no optimality claim",
                  world_version=spec.ENV_VERSION, controllers=str(path),
                  controller_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  stop="both robots clear sensor-based safety gate, >=4s elapsed at a boundary, or task end",
                  actions=actions, states=[], branches=[])
    started = time.time()
    for state in states:
        first_period = []
        for obs in state["observations"]:
            commands = [repaired.make_skill(a).step(obs, spec.CONTROL_DT)
                        for a in spec.ROBOT_ACTIONS]
            first_period.append(dict(robot=obs.robot_id,
                unique_forces=len({tuple(round(x, 9) for x in force) for force, _ in commands}),
                all_exit=all(info["done"] for _, info in commands),
                static_edge_gap=None if obs.static_nearest is None else obs.static_nearest[0]-spec.ROBOT_RADIUS,
                teammate_edge_gap=None if obs.teammate_nearest is None else obs.teammate_nearest[0]-spec.ROBOT_RADIUS,
                reasons=sorted({info["exit_reason"] for _, info in commands})))
        report["states"].append(dict(model=state["model"], instance=state["instance"].name,
                                     boundary=state["boundary"], first_period=first_period))
        for pair in actions:
            world = env.RandomStartEnv(state["instance"])
            world.restore_branch_state(state["branch_state"])
            begin = world.total_periods
            distance_before = world.object_distance()
            boundaries = []
            clear = False
            while world.termination is None and (world.total_periods-begin)*spec.CONTROL_DT < 4.0:
                boundary = world.joint_step(*pair)
                boundaries.append(boundary)
                obs = world.boundary_observations()
                clear = all(repaired.make_skill(spec.ROBOT_ACTIONS[13])._safety(o) is None for o in obs)
                if clear:
                    break
            report["branches"].append(dict(model=state["model"], instance=state["instance"].name,
                boundary=state["boundary"], action=pair, clear_safety_gate=clear,
                elapsed_seconds=(world.total_periods-begin)*spec.CONTROL_DT,
                decisions=len(boundaries), termination=world.termination,
                goal_distance_before=distance_before, goal_distance_after=world.object_distance(),
                end_object=list(world.object_state().as_tuple()), events=boundaries))
        print(state["instance"].name, state["boundary"],
              [r["clear_safety_gate"] for r in report["branches"][-6:]], flush=True)
        report["wall_seconds"] = time.time()-started
        (OUT / "recovery_probes.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
