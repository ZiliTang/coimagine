"""Inspect effective commands in recorded short-skill loops; use frozen v3 controllers."""
import gzip
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/random_start_rl_diagnosis_v4"
sys.path.insert(0, str(OUT / "code/src"))

from coimagine.random_start import env, layout, skills, spec


def main():
    panel = {x.name: x for x in layout.build_split("L0", "train", count=64)[0]}
    report = dict(environment=spec.ENV_VERSION, controller_source=skills.__file__, states=[])
    states = []
    # One high-count failure from initial and late checkpoint; selection is diagnostic only.
    for model, name in (("initial", "L0_train_0010"), ("late_4600", "L0_train_0063")):
        with gzip.open(OUT / "logs" / model / f"{name}_0.json.gz", "rt") as handle:
            log = json.load(handle)
        boundaries = log["boundaries"]
        short = [i for i, b in enumerate(boundaries) if b["periods"] == 1]
        chosen = {short[0], short[-1]}
        world = env.RandomStartEnv(panel[name])
        for i, recorded in enumerate(boundaries):
            if i in chosen:
                states.append(dict(model=model, instance=panel[name], boundary=i,
                                   branch_state=world.export_branch_state(),
                                   observations=world.boundary_observations()))
                row = dict(model=model, instance=name, boundary=i,
                           time=world.total_periods*spec.CONTROL_DT, robots=[])
                for observation in world.boundary_observations():
                    commands = []
                    for action in spec.ROBOT_ACTIONS:
                        skill = skills.make_skill(action)
                        force, info = skill.step(observation, spec.CONTROL_DT)
                        commands.append(dict(action=action.index, label=action.label, force=force,
                                             stage=info["stage"], exit_reason=info["exit_reason"]))
                    groups = {}
                    for command in commands:
                        key = tuple(round(float(x), 9) for x in command["force"])
                        groups.setdefault(str(key), []).append(command["action"])
                    row["robots"].append(dict(robot=observation.robot_id,
                        unique_first_period_forces=len(groups), force_equivalence_groups=groups,
                        all_end_immediately=all(c["exit_reason"] is not None for c in commands),
                        commands=commands))
                report["states"].append(row)
            actual = world.joint_step(*recorded["action_index"])
            assert actual["periods"] == recorded["periods"]
            assert abs(actual["distance_after"] - recorded["distance_after"]) < 1e-9
    (OUT / "skill_alias.json").write_text(json.dumps(report, indent=2))
    with (OUT / "short_loop_states.pkl").open("wb") as handle:
        pickle.dump(states, handle)
    print([(r["instance"],r["boundary"],[(x["unique_first_period_forces"],x["all_end_immediately"])
           for x in r["robots"]]) for r in report["states"]])


if __name__ == "__main__":
    main()
