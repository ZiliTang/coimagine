"""Render the S0 and S1 showcase videos from real executions with the current renderer.

    python experiments/continuous2d/render_showcase.py --out outputs/migration_v1/<run>/continuous2d/media

Re-runs the two accepted plans (S0 depth-1, S1 depth-3), writes per-control-step PNG frames
and a GIF for each. The frames are the executed trajectory; no animation is generated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coimagine.continuous2d import Continuous2DEnv, default_config
from coimagine.continuous2d.mpc import run_mpc_episode
from coimagine.continuous2d.render import render_episode

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/migration_v1/phaseD-final/continuous2d/media")
    parser.add_argument("--every", type=int, default=4)
    args = parser.parse_args()
    config = default_config()
    env = Continuous2DEnv(config)
    out_dir = ROOT / args.out
    jobs = (("S0_dev", 1, config.mpc.beam), ("S1_dev", 3, config.mpc.beam))
    for scene, depth, beam in jobs:
        episode = run_mpc_episode(env, scene, config, depth=depth, beam=beam, seed=0)
        paths = render_episode(
            episode, scene, config, str(out_dir), stem=f"{scene}_mpc_success", every=args.every
        )
        print(
            f"{scene} depth={depth} success={episode.success} windows={episode.windows} "
            f"files={len(paths)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
