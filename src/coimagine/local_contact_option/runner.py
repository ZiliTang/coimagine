"""The paired four-arm experiment runner.

One *execution* is one (root, variant, arm). The eight variants of a root share one frozen
snapshot, so the arms cannot contaminate each other: every arm starts from an exact
``restore_exact`` of the same snapshot, and the only difference between arms is the force
source and the stage protocol.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.config import Continuous2DConfig, default_config
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.scenes import Pose
from .contract import OptionConfig
from .control import LocalControlState  # noqa: F401 - part of the public surface
from .metrics import execution_metrics
from .option import ARM_ORDER, OptionExecutor
from .roots import (
    RootSpec, VARIANT_SPECS, VariantSpec, apply_initial_velocities, initial_velocities,
    variant_config, variant_scene,
)
from .skill import LocalSurfacePush


def root_from_metadata(meta: Dict[str, object]) -> RootSpec:
    def _pose(values):
        return Pose(float(values[0]), float(values[1]), float(values[2]))

    return RootSpec(
        root_id=str(meta["root_id"]), split=str(meta["split"]),
        interaction_class=str(meta["interaction_class"]), template=str(meta["template"]),
        seed=int(meta["seed"]),
        object_pose=_pose(meta["object_pose"]),
        robot_poses=(_pose(meta["robot_poses"][0]), _pose(meta["robot_poses"][1])),
        alpha=(meta["alpha_rad"][0], meta["alpha_rad"][1]),
        speed=(meta["speed"][0], meta["speed"][1]),
        gap=(meta["gap"][0], meta["gap"][1]),
        contact_local=(
            None if meta["contact_local"][0] is None else tuple(meta["contact_local"][0]),
            None if meta["contact_local"][1] is None else tuple(meta["contact_local"][1]),
        ),
        push_direction_world=(
            None if meta["push_direction_world"][0] is None else tuple(meta["push_direction_world"][0]),
            None if meta["push_direction_world"][1] is None else tuple(meta["push_direction_world"][1]),
        ),
        active_robots=tuple(int(v) for v in meta["active_robots"]),
        contact_face=(meta["contact_face"][0], meta["contact_face"][1]),
        contact_lateral=(meta["contact_lateral"][0], meta["contact_lateral"][1]),
        notes=str(meta.get("notes", "")),
    )


def root_metadata(root: RootSpec) -> Dict[str, object]:
    payload = root.as_metadata()
    payload["alpha_rad"] = list(root.alpha)
    return payload


def build_snapshot(root: RootSpec, variant: VariantSpec, base_config: Continuous2DConfig):
    """The shared, frozen starting world for one (root, variant).

    The perturbed poses live in the ``SceneSpec`` and the residual velocity is an explicit
    initial condition, so ``restore_exact`` reproduces the same world for every arm and the
    engine's cached contact state is rebuilt rather than approximated.
    """

    scene, report = variant_scene(root, variant)
    config = variant_config(base_config, variant)
    env = Continuous2DEnv(config)
    env.reset(scene)
    velocities = initial_velocities(root, variant)
    # the snapshot is taken before the residual velocity is written, so its prefix replay is
    # exact; the velocity is then an explicit initial condition applied identically per arm
    snapshot = env.export_state()
    initial_object = env.backend.object_state().as_tuple()
    initial_robots = [state.as_tuple() for state in env.backend.robot_states()]
    report["initial_velocities"] = {key: list(value) for key, value in velocities.items()}
    report["scene"] = scene.name
    return snapshot, config, initial_object, initial_robots, report, velocities


def run_execution(
    root: RootSpec,
    variant: VariantSpec,
    arm: str,
    option_config: OptionConfig,
    base_config: Continuous2DConfig,
    *,
    log_every: int = 1,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    snapshot, config, initial_object, initial_robots, variant_report, velocities = build_snapshot(
        root, variant, base_config
    )
    env = Continuous2DEnv(config)
    env.restore(snapshot, exact=True)
    apply_initial_velocities(env, velocities)
    specs: Dict[int, object] = {}
    for robot_id in (0, 1):
        if arm in ("O", "P"):
            specs[robot_id] = root.oracle_spec(robot_id)
        else:
            parameters = root.local_parameters(robot_id)
            specs[robot_id] = LocalSurfacePush(**parameters) if parameters else None
    # the variant's sensing conditions belong to the observation pipeline, not to the world:
    # the delay variant must really delay, and the noise variant must really add range noise
    option_config = replace(
        option_config,
        observation_delay=variant.observation_delay,
        range_noise_sigma=variant.noise_sigma,
    )
    executor = OptionExecutor(option_config, arm, log_every=log_every)
    noise_rng = None
    if variant.noise_sigma > 0.0:
        import random

        noise_rng = random.Random(root.seed ^ 0xBEEF)
    started = time.time()
    result = executor.run(
        env, root_id=root.root_id, variant=variant.name, joint_specs=specs,
        active_robots=root.active_robots, noise_rng=noise_rng,
    )
    metrics = execution_metrics(
        result, initial_object=initial_object, initial_robots=initial_robots,
        force_limit=option_config.force_limit, physics=config.physics, option=option_config,
    )
    record = {
        "root_id": root.root_id,
        "split": root.split,
        "interaction_class": root.interaction_class,
        "template": root.template,
        "variant": variant.name,
        "arm": arm,
        "active_robots": list(root.active_robots),
        "alpha_deg": [None if a is None else round(math.degrees(a), 3) for a in root.alpha],
        "speed": list(root.speed),
        "option_config": option_config.as_dict(),
        "variant_spec": variant.as_dict(),
        "variant_report": variant_report,
        "initial_object": list(initial_object),
        "initial_robots": [list(row) for row in initial_robots],
        "wall_clock_seconds": time.time() - started,
        "arm_outcome": result.as_dict(),
        "terminal_object": list(result.terminal_object) if result.terminal_object else None,
        "terminal_robots": [list(row) for row in result.terminal_robots] if result.terminal_robots else None,
        "metrics": metrics,
    }
    return record, result.rows, variant_report


def process_root(job: Dict[str, object]) -> Dict[str, object]:
    """Worker entry point: one root, all variants, all arms."""

    root = root_from_metadata(job["root"])
    base_config = default_config()
    option_config = OptionConfig(**job["option_config"])
    log_every = int(job.get("log_every", 1))
    arms = tuple(job.get("arms", ARM_ORDER))
    log_dir = Path(job["log_dir"])
    records: List[Dict[str, object]] = []
    for variant_name in job["variants"]:
        variant = next(spec for spec in VARIANTS_BY_NAME if spec.name == variant_name)
        log_path = log_dir / f"{root.root_id.replace('/', '_')}__{variant.name}.jsonl"
        with log_path.open("w") as handle:
            for arm in arms:
                record, rows, _ = run_execution(
                    root, variant, arm, option_config, base_config, log_every=log_every,
                )
                for row in rows:
                    row = dict(row)
                    row["arm"] = arm
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                record["log_path"] = str(log_path)
                records.append(record)
    return {"root_id": root.root_id, "records": records}


VARIANTS_BY_NAME = VARIANT_SPECS


def run_split(
    roots: Sequence[RootSpec],
    *,
    out_dir: Path,
    option_config: OptionConfig,
    variants: Sequence[str],
    arms: Sequence[str] = ARM_ORDER,
    workers: int = 8,
    log_every: int = 1,
    tag: str = "test",
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "root": root_metadata(root),
            "option_config": option_config.as_dict(),
            "variants": list(variants),
            "arms": list(arms),
            "log_dir": str(log_dir),
            "log_every": log_every,
        }
        for root in roots
    ]
    started = time.time()
    records: List[Dict[str, object]] = []
    if workers <= 1:
        for job in jobs:
            records.extend(process_root(job)["records"])
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            for index, payload in enumerate(pool.imap_unordered(process_root, jobs, chunksize=1)):
                records.extend(payload["records"])
                if (index + 1) % 10 == 0:
                    print(f"  {tag}: {index + 1}/{len(jobs)} roots done", flush=True)
    path = out_dir / f"executions_{tag}.jsonl"
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "tag": tag, "roots": len(roots), "variants": list(variants), "arms": list(arms),
        "executions": len(records), "wall_clock": time.time() - started,
        "option_config": option_config.as_dict(), "workers": workers,
        "executions_path": str(path), "log_dir": str(log_dir),
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary
