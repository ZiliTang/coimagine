"""Build the review package: config dump, source copy, manifest and the summary documents.

The package is meant to be openable without the repository for reading, and to be runnable from
the repository for reproduction. Everything numeric in it comes from a file that a script wrote;
nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import spec as S  # noqa: E402

SOURCE_FILES = [
    "src/coimagine/random_start/__init__.py",
    "src/coimagine/random_start/spec.py",
    "src/coimagine/random_start/geometry.py",
    "src/coimagine/random_start/layout.py",
    "src/coimagine/random_start/lidar.py",
    "src/coimagine/random_start/skills.py",
    "src/coimagine/random_start/env.py",
    "src/coimagine/random_start/render.py",
    "src/coimagine/random_start/verify.py",
    "experiments/random_start_reposition_v1/run_scenes.py",
    "experiments/random_start_reposition_v1/run_skills.py",
    "experiments/random_start_reposition_v1/run_episodes.py",
    "experiments/random_start_reposition_v1/run_verify.py",
    "experiments/random_start_reposition_v1/rl.py",
    "experiments/random_start_reposition_v1/run_collect.py",
    "experiments/random_start_reposition_v1/run_subset.py",
    "experiments/random_start_reposition_v1/run_resume.py",
    "experiments/random_start_reposition_v1/report.py",
    "tests/test_random_start.py",
    "experiments/random_start_reposition_v1/package.py",
]


def git_revision() -> Dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                             text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as error:  # pragma: no cover
        return {"commit": None, "error": repr(error)}


def frozen_config() -> Dict[str, object]:
    config = E.EnvConfig()
    keys = ("CONTROL_DT", "PHYSICS_DT", "SUBSTEPS", "ROBOT_RADIUS", "ROBOT_MASS",
            "OBJECT_WIDTH", "OBJECT_HEIGHT", "OBJECT_MASS", "MAX_FORCE", "FRICTION",
            "ELASTICITY", "LINEAR_DAMPING", "ANGULAR_DAMPING", "MAX_ACCEL",
            "BOUNDARY_RADIUS", "ARENA", "L1_CENTRE", "L1_FULL_WIDTH", "L1_FULL_HEIGHT",
            "MAX_SAMPLE_ATTEMPTS", "OBJECT_EDGE_MARGIN", "ROBOT_OBJECT_GAP_MIN",
            "ROBOT_OBJECT_GAP_MAX", "ROBOT_CLEARANCE", "GOAL_RADIUS", "GOAL_DISTANCE_MIN",
            "GOAL_DISTANCE_MAX", "GOAL_TERMINAL_CLEARANCE", "SPLIT_SIZES", "LIDAR_RANGE",
            "LIDAR_RAYS", "LIDAR_HZ", "PATCH_GAP_TOLERANCE", "PATCH_MIN_POINTS",
            "PATCH_RELATIVE_FACTOR", "MAX_FIT_RMS", "NORMAL_FILTER_TAU",
            "NORMAL_JUMP_LIMIT_DEG", "CONTACT_LOSS_TOLERANCE", "CONTACT_SEEK_SECONDS",
            "ALPHA_DEG", "PUSH_SPEEDS", "REPOSITION_DIRECTIONS", "REPOSITION_DISTANCES",
            "WAIT_DURATIONS", "SKILL_MAX_SECONDS", "JOINT_MAX_SECONDS", "PUSH_ACTIVE_SECONDS",
            "PUSH_WALL_SECONDS", "APPROACH_TIMEOUT", "APPROACH_MIN_SPEED",
            "APPROACH_MAX_SPEED", "APPROACH_GAIN", "NO_PROGRESS_SECONDS", "REPOSITION_GAP",
            "REPOSITION_GAP_DEADBAND", "REPOSITION_SPEED", "REPOSITION_SPEED_MAX",
            "REPOSITION_ALIGN_ENTRY", "REPOSITION_GAP_GAIN", "A_REF_MAX", "KV",
            "FIND_SECONDS", "FIND_SPEED", "SAFETY_GAP", "OBSTACLE_AVOID_GAIN",
            "OBSTACLE_AVOID_MARGIN", "SUCCESS_DISTANCE", "SUCCESS_LINEAR", "SUCCESS_ANGULAR",
            "SUCCESS_HOLD_SECONDS", "TIME_BUDGET_SECONDS", "GAMMA",
            "TIME_COST", "SUCCESS_BONUS", "FAILURE_PENALTY", "PROGRESS_WEIGHT", "DISTANCE_COST",
            "ENV_VERSION", "CONFIG_HASH")
    dump = {key: getattr(S, key) for key in keys if hasattr(S, key)}
    dump["action_table"] = [asdict(action) for action in S.ROBOT_ACTIONS]
    dump["env_config"] = config.as_dict()
    dump["observation_fields"] = E.RandomStartEnv.observation_fields(
        E.RandomStartEnv.__new__(E.RandomStartEnv))
    return dump


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_sources(out: Path) -> List[Dict[str, object]]:
    copied = []
    code_dir = out / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.exists():
            copied.append({"path": relative, "missing": True})
            continue
        target = code_dir / relative.replace("/", "__")
        target.write_bytes(source.read_bytes())
        copied.append({"path": relative, "packaged_as": target.name,
                       "sha256": sha256(source), "bytes": source.stat().st_size})
    return copied


def manifest(out: Path) -> Dict[str, object]:
    entries = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            entries[str(path.relative_to(out))] = {"bytes": path.stat().st_size,
                                                   "sha256": sha256(path)[:32]}
    return {"files": entries, "count": len(entries),
            "total_bytes": int(sum(entry["bytes"] for entry in entries.values()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs/random_start_reposition_v1"))
    args = parser.parse_args()
    out = Path(args.out)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    frozen = frozen_config()
    (out / "configs" / "frozen_config.json").write_text(json.dumps(frozen, indent=1, default=str))
    copied = copy_sources(out)
    requirements = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=ROOT,
                                  capture_output=True, text=True).stdout
    keep = ("torch", "numpy", "pymunk", "matplotlib", "imageio", "Pillow", "pytest")
    locked = "\n".join(line for line in requirements.splitlines()
                       if line.split("==")[0].lower().replace("-", "_") in
                       tuple(name.lower() for name in keep) or line.lower().startswith(keep))
    (out / "code" / "requirements-lock.txt").write_text(locked + "\n")
    (out / "code" / "README.md").write_text(
        "# Packaged source\n\n"
        "These are byte copies of the files that produced the results in this package:\n\n"
        "```\n" + "\n".join(entry["path"] for entry in copied if not entry.get("missing")) +
        "\n```\n\n"
        "`common__*.py` copies keep their original relative path in the file name.\n\n"
        "## What runs without the repository\n\n"
        "Nothing in this directory is a standalone program: every module import starts with\n"
        "`coimagine.random_start`, and the environment depends on the repository's frozen\n"
        "`coimagine.continuous2d` physics backend and `coimagine.local_contact_option.control`\n"
        "feedback law. To reproduce, use the repository at the commit in `MANIFEST.json`:\n\n"
        "```bash\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -e '.[dev]'\n"
        "python experiments/random_start_reposition_v1/run_scenes.py\n"
        "python experiments/random_start_reposition_v1/run_skills.py\n"
        "python experiments/random_start_reposition_v1/run_episodes.py\n"
        "python experiments/random_start_reposition_v1/run_verify.py\n"
        "python experiments/random_start_reposition_v1/rl.py\n"
        "python experiments/random_start_reposition_v1/package.py\n"
        "```\n\n"
        "## What can be checked by reading alone\n\n"
        "* every constant: `../configs/frozen_config.json` is a dump of `spec.py`;\n"
        "* the information boundary: `verify.py::check_information_boundary` plus its result in\n"
        "  `../results/invariants.json`;\n"
        "* the reward identity: `env.py::recompute_episode_reward` against the environment total;\n"
        "* the sensor: `lidar.py` against the brute-force re-derivation in `verify.py`.\n")
    report = {
        "git": git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "frozen_config_hash": frozen.get("CONFIG_HASH"),
        "env_version": frozen.get("ENV_VERSION"),
        "packaged_sources": copied,
        "requirements_lock": "code/requirements-lock.txt",
    }
    (out / "MANIFEST.json").write_text(json.dumps({**report, "manifest": manifest(out)},
                                                  indent=1, default=str))
    print(json.dumps({"files": len(copied), "out": str(out),
                      "env_version": frozen.get("ENV_VERSION")}, indent=1))
    print("PACKAGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
