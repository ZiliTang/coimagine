from __future__ import annotations

import argparse
import json

from coimagine.evaluation.gate_a import evaluate_gate_a
from coimagine.evaluation.runner import run_gate_a
from coimagine.envs.grid.scenarios import two_robot_approach_then_push
from coimagine.visualization.ascii import render_ascii


def demo_main() -> None:
    state = two_robot_approach_then_push()
    result = evaluate_gate_a(state, horizon=2)
    print(render_ascii(state))
    print()
    print(
        json.dumps(
            {
                "evolving_value": result.evolving.value,
                "root_structure_reuse_value": result.fixed.value,
                "oracle_gain": result.oracle_gain,
                "first_action_reversal": result.ranking_reversal,
                "evolving_first_action": result.evolving.joint_option.template_key,
                "fixed_first_action": result.fixed.joint_option.template_key,
            },
            indent=2,
            default=str,
        )
    )


def gate_a_main() -> None:
    parser = argparse.ArgumentParser(description="Run the CoImagine Gate-A pipeline.")
    parser.add_argument("--profile", default="smoke", help="smoke, pilot, paper, or a YAML path")
    parser.add_argument("--output", default=None, help="Override the profile output directory")
    args = parser.parse_args()
    artifacts = run_gate_a(args.profile, output_dir=args.output)
    print(
        json.dumps(
            {
                "output_dir": str(artifacts.output_dir),
                "scene_records": len(artifacts.scene_records),
                "baseline_records": len(artifacts.baseline_records),
                "scaling_records": len(artifacts.scaling_records),
                "episode_records": len(artifacts.episode_records),
                "failures": len(artifacts.failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    gate_a_main()
