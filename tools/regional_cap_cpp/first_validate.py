"""Deterministic synthetic branch/memory checks; performs no physics/model calls."""

import copy
import importlib.util
import math
import random
import subprocess
import sysconfig
import tempfile
from pathlib import Path

MAX_FLOAT_ERROR = 0.0
FLOAT_DIFFERENCES = 0
import ast

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/semantic_rl/cap_regional_five_20261119_01/regional/semantics"
SIDS = ("s004", "s008", "s011", "s013", "s019")


def compare(a, b, path=""):
    global MAX_FLOAT_ERROR, FLOAT_DIFFERENCES
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            compare(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), (path, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + "/" + str(i))
    elif isinstance(a, float):
        if a != b:
            FLOAT_DIFFERENCES += 1
            MAX_FLOAT_ERROR = max(MAX_FLOAT_ERROR, abs(a - b))
        assert isinstance(b, (float, int)) and a == b, (path, a, b)
    else:
        assert a == b, (path, a, b)


def main(module):
    rng = random.Random(1729)
    count = 0
    for ident, sid in enumerate(SIDS):
        tree = ast.parse((BASE / f"{sid}.py").read_text())
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.Import) and any(alias.name == "cython" for alias in node.names))
            and not (isinstance(node, ast.FunctionDef) and node.name == "evaluate_candidates")
        ]
        source = {}
        exec(compile(tree, str(BASE / f"{sid}.py"), "exec"), source)
        for trial in range(300):
            m = source["initial_memory"]()
            compare(m, module.initial_memory(ident))
            angle = rng.choice([0.0, math.pi / 2, 0.61, -0.4, 0.2])
            center = [rng.uniform(0.8, 4.2), rng.uniform(0.8, 4.2)]
            goal = [rng.uniform(0.5, 4.5), rng.uniform(0.5, 4.5)]
            if trial % 5 == 0:
                goal = [center[0] + 0.1, center[1] + 0.05]
            u = [math.cos(angle), math.sin(angle)]
            v = [-u[1], u[0]]
            dims = [rng.choice([0.3, 0.8, 1.2]), rng.choice([0.2, 0.6, 1.0])]
            points = [
                [
                    center[0] + i * dims[0] * u[0] / 2 + j * dims[1] * v[0] / 2,
                    center[1] + i * dims[0] * u[1] / 2 + j * dims[1] * v[1] / 2,
                ]
                for i, j in [(-1, -1), (-1, 1), (1, 1), (1, -1)]
            ]
            robots = []
            for k in range(2):
                robots.append(
                    dict(
                        arena=[5.0, 5.0],
                        own=dict(x=rng.uniform(0.3, 4.7), y=rng.uniform(0.3, 4.7)),
                        contacting_object=rng.choice([False, True]),
                        patches=[dict(valid=True, span_m=0.7, tangent_world=u, normal_world=v)],
                        execution=dict(
                            stage=rng.choice(
                                [
                                    "find",
                                    "view_insufficient",
                                    "blocked",
                                    "wait",
                                    "peer_wait",
                                    "move_blocked",
                                    "push",
                                ]
                            ),
                            exit_reason=rng.choice([None, "blocked"]),
                            done=trial % 7 == 0,
                            direction_unavailable=rng.choice([True, False, None]),
                        ),
                        scan=dict(
                            origin=[1.0, 1.0],
                            ray_angles=[],
                            distances=[],
                            valid=[],
                            category=[],
                            entity_ids=[],
                        ),
                    )
                )
            legal = [
                [i, j]
                for i in range(47)
                for j in range(47)
                if i == j or i == 46 or j == 46 or (i < 32 and j < 32 and rng.random() < 0.1)
            ]
            if trial % 4 == 0:
                legal = rng.sample(legal, 5)
            o = dict(
                robots=robots,
                goal=dict(target_id="target", xy=goal),
                entities=[] if trial % 31 == 0 else [dict(id="target", points=points)],
                active_target="target",
                entity_legal_pairs={"target": legal},
            )
            m.update(
                tick=trial * 3,
                stall=trial % 10,
                last=center.copy(),
                context={"proposals": [{"target_id": "target"}]},
            )
            if trial % 2:
                m.update(nav_ticks=rng.choice([110, 350]), goal_brake=-100)
            if ident == 3 and trial % 2:
                m.update(progress_dist=10.0, progress_tick=-1000, rotate_until=0, rotate_cooldown=0)
            n = copy.deepcopy(m)
            for step in range(3):
                action = rng.choice(
                    [
                        None,
                        dict(target_id=None, a0=44, a1=44),
                        dict(target_id="target", a0=40, a1=43),
                        dict(target_id="target", a0=3, a1=3),
                    ]
                )
                e = dict(executed_ticks=rng.choice([None, 0, 15, 40, 120]), observation=o, action=action)
                m = source["update_memory"](m, copy.deepcopy(e))
                n = module.update_memory(ident, n, copy.deepcopy(e))
                compare(m, n, f"{sid}/trial{trial}/step{step}")
                count += 1
    print(
        f"PASS {count} synthetic full-memory/context update comparisons; float differences={FLOAT_DIFFERENCES}; max absolute error={MAX_FLOAT_ERROR}"
    )


if __name__ == "__main__":
    import torch

    with tempfile.TemporaryDirectory(prefix="first-cap-validate-") as directory:
        directory = Path(directory)
        cpp = directory / "first_validate_module.cpp"
        cpp.write_text(
            '#include "first.hpp"\nPYBIND11_MODULE(_first_validate,m){m.def("initial_memory",&rc::first_initial);m.def("update_memory",&rc::first_update); }\n'
        )
        binary = directory / "_first_validate.so"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-ffp-contract=off",
                "-fno-builtin-sin",
                "-fno-builtin-cos",
                "-I" + sysconfig.get_path("include"),
                "-I" + str(Path(torch.__file__).parent / "include"),
                "-I" + str(Path(__file__).parent),
                str(cpp),
                "-o",
                str(binary),
            ],
            check=True,
        )
        spec = importlib.util.spec_from_file_location("_first_validate", binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        main(module)
