"""Focused source/native differential coverage for regional CAP IDs 5..9."""

import ast
import copy
import importlib.util
import math
import random
import subprocess
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def equal(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            equal(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (path, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, path + "/" + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and a == b, (path, a, b)
    else:
        assert a == b, (path, a, b)


def observation(rng, t):
    angle = rng.choice([0.0, 0.25, 0.7, 1.2])
    u = (math.cos(angle), math.sin(angle))
    v = (-u[1], u[0])
    center = [rng.uniform(1.0, 5.0), rng.uniform(1.0, 5.0)]
    entities = [
        {
            "id": "target",
            "points": [
                [center[0] + s * u[0] + q * v[0], center[1] + s * u[1] + q * v[1]]
                for s, q in [(-0.7, -0.15), (-0.7, 0.15), (0.7, -0.15), (0.7, 0.15)]
            ],
        },
        {"id": "secondary", "points": [[center[0] + 0.4, center[1]], [center[0] + 0.8, center[1] + 0.2]]},
    ]
    if t % 31 == 0:
        entities[0]["points"] = []
    scan = {
        "origin": [1.0, 1.0],
        "ray_angles": [],
        "distances": [],
        "valid": [],
        "category": [],
        "entity_ids": [],
    }
    robots = []
    for i in range(2):
        stage = rng.choice(
            ["find", "view_insufficient", "blocked", "wait", "peer_wait", "move_blocked", "push", ""]
        )
        execution = {
            "stage": stage,
            "done": rng.random() < 0.08,
            "exit_reason": rng.choice([None, None, "blocked"]),
            "direction_unavailable": rng.random() < 0.1,
        }
        robots.append(
            {
                "arena": [6.0, 6.0],
                "own": {"x": rng.uniform(0.4, 5.6), "y": rng.uniform(0.4, 5.6)},
                "execution": execution,
                "contacting_object": rng.random() < 0.4,
                "scan": copy.deepcopy(scan),
                "patches": [{"valid": True, "span_m": 0.8, "tangent_world": u, "normal_world": v}],
            }
        )
    legal = []
    for d in range(8):
        for speed in range(4):
            a = d * 4 + speed
            for q in ([a, a], [a, 46], [46, a], [(a + 4) % 32, a]):
                if rng.random() < 0.8:
                    legal.append(q)
    legal += [[40, 43], [43, 40], [40, 40], [43, 43], [34, 34], [37, 37]]
    return {
        "goal": {
            "target_id": "target",
            "xy": rng.choice([[5.0, 5.0], center[:], [center[0] + 0.15, center[1] + 0.1]]),
        },
        "active_target": rng.choice(["target", "secondary", None]),
        "robots": robots,
        "entities": entities,
        "entity_legal_pairs": {
            "target": legal,
            "secondary": [[46, a] for a in range(32)] + [[a, 46] for a in range(32)],
        },
    }


def main(native):
    rng = random.Random(17529)
    count = 0
    for id, name in enumerate(["s000", "s001", "s002", "s005", "s009"], 5):
        path = (
            ROOT
            / "outputs/semantic_rl/cap_regional_extension_20261119_01/author/semantics"
            / f"{name}.py"
        )
        tree = ast.parse(path.read_text())
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.Import) and any(alias.name == "cython" for alias in node.names))
            and not (isinstance(node, ast.FunctionDef) and node.name == "evaluate_candidates")
        ]
        source_namespace = {}
        exec(compile(tree, str(path), "exec"), source_namespace)
        from types import SimpleNamespace

        source = SimpleNamespace(**source_namespace)
        a = source.initial_memory()
        b = native.initial_memory(id)
        equal(a, b)
        for t in range(400):
            if t % 40 == 0:
                a = source.initial_memory()
                b = native.initial_memory(id)
            if id == 8 and t % 40 == 5:
                for m in (a, b):
                    m.update(clear_progress=9.0, clear_progress_tick=-999, clear_until=0, clear_after=0)
            event = {
                "executed_ticks": rng.choice([0, 15, 25, 40, None]),
                "action": rng.choice(
                    [
                        None,
                        {"target_id": None, "a0": 50, "a1": 66},
                        {"target_id": "target", "a0": 3, "a1": 46},
                        {"target_id": "secondary", "a0": 46, "a1": 15},
                        {"target_id": "target", "a0": 40, "a1": 43},
                    ]
                ),
                "observation": observation(rng, t),
            }
            source.update_memory(a, copy.deepcopy(event))
            native.update_memory(id, b, copy.deepcopy(event))
            equal(a, b, f"{name}/{t}")
            count += 1
    print(f"PASS: {count} source/native complete-memory comparisons")


if __name__ == "__main__":
    import torch

    with tempfile.TemporaryDirectory(prefix="next-cap-validate-") as directory:
        directory = Path(directory)
        cpp = directory / "next_validate_module.cpp"
        cpp.write_text(
            '#include "next.hpp"\nPYBIND11_MODULE(_next_validate,m){m.def("initial_memory",&rc::next_initial);m.def("update_memory",&rc::next_update); }\n'
        )
        binary = directory / ("_next_validate" + sysconfig.get_config_var("EXT_SUFFIX"))
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
        spec = importlib.util.spec_from_file_location("_next_validate", binary)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        main(native)
